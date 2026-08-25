"""Async, Typer-free orchestration for offline bundle-contract validation."""

from __future__ import annotations

import json
from collections.abc import Hashable, Mapping
from typing import TYPE_CHECKING, Any, Protocol, cast

import httpx
import yaml
from yaml.constructor import ConstructorError

from .bundle_contract import ContractReport, Finding, Severity, check_bundle_contract
from .bundle_registry import BundleReference, BundleRegistry, BundleRegistryManager
from .bundle_resolution import (
    BundleResolutionError,
    parse_bundle_reference,
    resolve_bundle,
)
from .registry_service import get_or_default_registry

if TYPE_CHECKING:
    from pathlib import Path

    from yaml.nodes import MappingNode


class BundleContractServiceError(Exception):
    """An expected registry-resolution failure for the contract command."""


class EmptyBundleRegistryError(BundleContractServiceError):
    """A --all validation had no bundle entries to validate."""


class _DuplicateKeyError(yaml.YAMLError):
    """A mapping key that the standard YAML loader would silently collapse."""

    def __init__(self, duplicates: tuple[tuple[object, int], ...]) -> None:
        self.duplicates = duplicates
        key, line = duplicates[0]
        super().__init__(f"duplicate key {key!r} at line {line}")


class _StrictLoader(yaml.SafeLoader):
    """SafeLoader that refuses to silently drop a repeated mapping key."""

    def __init__(self, stream: str | bytes) -> None:
        super().__init__(stream)
        self._duplicate_keys: list[tuple[object, int]] = []

    def construct_mapping(self, node: MappingNode, deep: bool = False) -> dict[Hashable, Any]:
        """Construct a mapping while retaining every duplicate's source line."""
        self.flatten_mapping(node)
        mapping: dict[Hashable, Any] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                if key in mapping:
                    self._duplicate_keys.append((key, key_node.start_mark.line + 1))
                mapping[key] = self.construct_object(value_node, deep=deep)
            except TypeError as exc:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    "found unhashable key",
                    key_node.start_mark,
                ) from exc
        return mapping

    @property
    def duplicate_keys(self) -> tuple[tuple[object, int], ...]:
        """Every duplicate encountered while constructing the document."""
        return tuple(self._duplicate_keys)


class _DisposableLoader(Protocol):
    """The stable subset of PyYAML's loader cleanup interface."""

    def dispose(self) -> None: ...


_OBJECT_INFO_TIMEOUT = httpx.Timeout(connect=2.0, read=5.0, write=5.0, pool=5.0)


async def _fetch_object_info(comfyui_url: str) -> Mapping[str, object]:
    """Stream a live ComfyUI object-info document with a bounded local timeout."""
    endpoint = f"{comfyui_url.rstrip('/')}/object_info"
    try:
        async with (
            httpx.AsyncClient(timeout=_OBJECT_INFO_TIMEOUT) as client,
            client.stream("GET", endpoint) as response,
        ):
            response.raise_for_status()
            chunks = [chunk async for chunk in response.aiter_bytes()]
    except (httpx.HTTPError, httpx.InvalidURL) as exc:
        raise BundleContractServiceError(
            f"Unable to fetch ComfyUI /object_info from {endpoint}: {exc}"
        ) from exc

    try:
        object_info = json.loads(b"".join(chunks))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise BundleContractServiceError(
            f"ComfyUI /object_info at {endpoint} returned invalid JSON: {exc}"
        ) from exc
    if not isinstance(object_info, Mapping):
        raise BundleContractServiceError(
            f"ComfyUI /object_info at {endpoint} must return a JSON object."
        )
    return object_info


def _schema_error(bundle_name: str, error: Exception) -> ContractReport:
    return ContractReport(
        bundle_name=bundle_name,
        findings=(Finding(Severity.ERROR, "schema.invalid", str(error), "bundle.yaml"),),
    )


def _strict_yaml_load(source: str) -> object:
    """Load operator-authored YAML without losing duplicate-key evidence."""
    loader = _StrictLoader(source)
    try:
        document = loader.get_single_data()
        duplicates = loader.duplicate_keys
    finally:
        cast("_DisposableLoader", loader).dispose()
    if duplicates:
        raise _DuplicateKeyError(duplicates)
    return document


def _duplicate_key_findings(
    error: _DuplicateKeyError, *, document_name: str = "bundle.yaml"
) -> tuple[Finding, ...]:
    """Render duplicate-key parse failures without suppressing contract checks."""
    return tuple(
        Finding(
            Severity.ERROR,
            "bundle.duplicate_key",
            f"Duplicate key {key!r}; the later value wins when parsed.",
            f"{document_name}:{line}",
        )
        for key, line in error.duplicates
    )


def _contract_index_entries(
    registry: BundleRegistry,
) -> tuple[tuple[Mapping[str, object], ...], tuple[Finding, ...]]:
    """Read raw index entries because Apex has fields beyond Aisha's general index schema."""
    registry_path = registry.path
    index_path = registry_path / "bundle-index.yaml"
    if not index_path.exists():
        return (), ()
    try:
        source = index_path.read_text()
    except OSError:
        return (), ()
    try:
        data = _strict_yaml_load(source)
    except _DuplicateKeyError as exc:
        duplicate_findings = _duplicate_key_findings(exc, document_name="bundle-index.yaml")
        try:
            data = yaml.load(source, Loader=yaml.SafeLoader)
        except yaml.YAMLError:
            return (), ()
    except yaml.YAMLError:
        return (), ()
    else:
        duplicate_findings = ()
    if not isinstance(data, Mapping) or not isinstance(data.get("bundles"), list):
        return (), duplicate_findings
    return (
        tuple(entry for entry in data["bundles"] if isinstance(entry, Mapping)),
        duplicate_findings,
    )


def _attach_index_findings(
    reports: tuple[ContractReport, ...], index_findings: tuple[Finding, ...]
) -> tuple[ContractReport, ...]:
    """Attach registry-level diagnostics once, rather than once per bundle."""
    if not reports or not index_findings:
        return reports
    first, *rest = reports
    return (ContractReport(first.bundle_name, (*index_findings, *first.findings)), *rest)


def _load_report(
    bundle_name: str,
    bundle_path: Path,
    *,
    index_entries: tuple[Mapping[str, object], ...],
    all_bundles: bool,
    object_info: Mapping[str, object] | None = None,
    workflow_provider_check: bool = False,
) -> ContractReport:
    """Load a YAML contract or turn expected read/parse errors into a report."""
    try:
        source = (bundle_path / "bundle.yaml").read_text()
    except OSError as exc:
        return _schema_error(bundle_name, exc)
    try:
        raw = _strict_yaml_load(source)
    except yaml.YAMLError as exc:
        if not isinstance(exc, _DuplicateKeyError):
            return _schema_error(bundle_name, exc)
        duplicate_findings = _duplicate_key_findings(exc)
        try:
            raw = yaml.load(source, Loader=yaml.SafeLoader)
        except yaml.YAMLError as parse_exc:
            return _schema_error(bundle_name, parse_exc)
    else:
        duplicate_findings = ()
    report = check_bundle_contract(
        bundle_name,
        bundle_path,
        raw,
        index_entries=index_entries,
        all_bundles=all_bundles,
        bundle_root=bundle_path.parent,
        object_info=object_info,
        workflow_provider_check=workflow_provider_check,
    )
    if duplicate_findings:
        return ContractReport(bundle_name, (*duplicate_findings, *report.findings))
    return report


async def validate_bundle_contracts(
    manager: BundleRegistryManager,
    *,
    bundle: str | None,
    all_bundles: bool,
    sync: bool,
    allow_empty: bool = False,
    comfyui_url: str | None = None,
) -> tuple[ContractReport, ...]:
    """Resolve and validate one bundle or all resolved registry entries."""
    object_info = await _fetch_object_info(comfyui_url) if comfyui_url else None
    workflow_provider_check = True
    try:
        if sync:
            await manager.sync_all()
        if bundle is not None:
            ref = parse_bundle_reference(bundle)
            registry = get_or_default_registry(manager, ref)
            resolved = await resolve_bundle(manager, ref, sync=False)
            index_entries, index_findings = _contract_index_entries(registry)
            return _attach_index_findings(
                (
                    _load_report(
                        ref.name,
                        resolved.path,
                        index_entries=index_entries,
                        all_bundles=False,
                        object_info=object_info,
                        workflow_provider_check=workflow_provider_check,
                    ),
                ),
                index_findings,
            )

        if not all_bundles:
            raise BundleContractServiceError("Specify BUNDLE or --all")
        registry = get_or_default_registry(manager, BundleReference(name=""))
        index = await registry.get_index()
    except BundleResolutionError as exc:
        if bundle is not None and exc.bundle_path is not None:
            ref = parse_bundle_reference(bundle)
            registry = get_or_default_registry(manager, ref)
            index_entries, index_findings = _contract_index_entries(registry)
            return _attach_index_findings(
                (
                    _load_report(
                        ref.name,
                        exc.bundle_path,
                        index_entries=index_entries,
                        all_bundles=False,
                        object_info=object_info,
                        workflow_provider_check=workflow_provider_check,
                    ),
                ),
                index_findings,
            )
        raise BundleContractServiceError(str(exc)) from exc
    except ValueError as exc:
        raise BundleContractServiceError(str(exc)) from exc

    if not index.bundles:
        if allow_empty:
            return ()
        raise EmptyBundleRegistryError(
            "no bundles found in the resolved registry; nothing was validated"
        )

    index_entries, index_findings = _contract_index_entries(registry)
    reports: list[ContractReport] = []
    for entry in index.bundles:
        try:
            resolved = await resolve_bundle(manager, BundleReference(name=entry.name), sync=False)
        except BundleResolutionError as exc:
            if exc.bundle_path is not None:
                reports.append(
                    _load_report(
                        entry.name,
                        exc.bundle_path,
                        index_entries=index_entries,
                        all_bundles=True,
                        object_info=object_info,
                        workflow_provider_check=workflow_provider_check,
                    )
                )
                continue
            reports.append(_schema_error(entry.name, exc))
            continue
        reports.append(
            _load_report(
                entry.name,
                resolved.path,
                index_entries=index_entries,
                all_bundles=True,
                object_info=object_info,
                workflow_provider_check=workflow_provider_check,
            )
        )
    return _attach_index_findings(tuple(reports), index_findings)
