"""Async, Typer-free orchestration for offline bundle-contract validation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import yaml

from .bundle_contract import ContractReport, Finding, Severity, check_bundle_contract
from .bundle_registry import BundleReference, BundleRegistry, BundleRegistryManager
from .registry_service import get_or_default_registry


class BundleContractServiceError(Exception):
    """An expected registry-resolution failure for the contract command."""


class EmptyBundleRegistryError(BundleContractServiceError):
    """A --all validation had no bundle entries to validate."""


def _schema_error(bundle_name: str, error: Exception) -> ContractReport:
    return ContractReport(
        bundle_name=bundle_name,
        findings=(Finding(Severity.ERROR, "schema.invalid", str(error), "bundle.yaml"),),
    )


def _contract_index_entries(registry: BundleRegistry) -> tuple[Mapping[str, object], ...]:
    """Read raw index entries because Apex has fields beyond Aisha's general index schema."""
    registry_path = getattr(registry, "path", None)
    if not isinstance(registry_path, Path):
        return ()
    index_path = next(
        (
            path
            for path in (
                registry_path / "bundle-index.yaml",
                registry_path.parent / "bundle-index.yaml",
            )
            if path.exists()
        ),
        None,
    )
    if index_path is None:
        return ()
    try:
        data = yaml.safe_load(index_path.read_text())
    except (OSError, yaml.YAMLError):
        return ()
    if not isinstance(data, Mapping) or not isinstance(data.get("bundles"), list):
        return ()
    return tuple(entry for entry in data["bundles"] if isinstance(entry, Mapping))


def _load_report(
    bundle_name: str,
    bundle_path: Path,
    *,
    index_entries: tuple[Mapping[str, object], ...],
    all_bundles: bool,
) -> ContractReport:
    """Load a YAML contract or turn expected read/parse errors into a report."""
    try:
        raw = yaml.safe_load((bundle_path / "bundle.yaml").read_text())
    except (OSError, yaml.YAMLError) as exc:
        return _schema_error(bundle_name, exc)
    return check_bundle_contract(
        bundle_name,
        bundle_path,
        raw,
        index_entries=index_entries,
        all_bundles=all_bundles,
        bundle_root=bundle_path.parent,
    )


async def validate_bundle_contracts(
    manager: BundleRegistryManager,
    *,
    bundle: str | None,
    all_bundles: bool,
    sync: bool,
    allow_empty: bool = False,
) -> tuple[ContractReport, ...]:
    """Resolve and validate one bundle or all resolved registry entries."""
    try:
        if sync:
            await manager.sync_all()
        if bundle is not None:
            ref = BundleReference.parse(bundle)
            registry = get_or_default_registry(manager, ref)
            bundle_path = await manager.resolve(ref)
            return (
                _load_report(
                    ref.name,
                    bundle_path,
                    index_entries=_contract_index_entries(registry),
                    all_bundles=False,
                ),
            )

        if not all_bundles:
            raise BundleContractServiceError("Specify BUNDLE or --all")
        registry = get_or_default_registry(manager, BundleReference(name=""))
        index = await registry.get_index()
    except ValueError as exc:
        raise BundleContractServiceError(str(exc)) from exc

    if not index.bundles:
        if allow_empty:
            return ()
        raise EmptyBundleRegistryError(
            "no bundles found in the resolved registry; nothing was validated"
        )

    index_entries = _contract_index_entries(registry)
    reports: list[ContractReport] = []
    for entry in index.bundles:
        try:
            bundle_path = await registry.resolve_bundle_path(entry.name)
        except ValueError as exc:
            reports.append(_schema_error(entry.name, exc))
            continue
        reports.append(
            _load_report(
                entry.name,
                bundle_path,
                index_entries=index_entries,
                all_bundles=True,
            )
        )
    return tuple(reports)
