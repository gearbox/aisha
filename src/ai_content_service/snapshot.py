"""Snapshot management for AI Content Service."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import uuid
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Final
from urllib.parse import quote, urlparse

import httpx
import structlog
import yaml
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from rich import get_console
from rich.progress import BarColumn, Progress, TaskID, TextColumn

from . import bundle_contract
from .bundle import set_current_symlink
from .bundle_contract import (
    Severity,
    check_api_graph_links,
    check_workflow_sync,
    is_api_workflow,
)
from .bundle_registry import resolve_bundles_dir
from .config import (
    BundleConfig,
    BundleMetadata,
    BundleVersion,
    ComfyUIConfig,
    CustomNodeConfig,
    ModelConfig,
    ModelFileConfig,
    ModelType,
    WorkflowMapConfig,
)
from .downloader import ModelDownloader
from .requirement_refs import is_missing_local_reference
from .workflow_map import _normalize_workflow_comment, infer_workflow_map

if TYPE_CHECKING:
    from collections.abc import Callable

console = get_console()
log = structlog.get_logger()

_MODEL_EXTENSIONS = {".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf"}
_SKIPPED_MODEL_DIRECTORIES = {".cache", ".git"}
_MAX_CONCURRENT_HASHES = 4
_PROGRESS_UPDATE_BYTES = ModelDownloader.CHUNK_SIZE * 8
_NO_BASE_MANIFEST_MESSAGE = (
    "No usable base manifest was found; snapshot will carry no requirements file."
)
_INVALID_BASE_MANIFEST_MESSAGE = (
    "Base manifest has no usable packages mapping; snapshot will carry no requirements file."
)
_WORKFLOW_CONVERTER_TIMEOUT: Final = httpx.Timeout(connect=5.0, read=10.0, write=10.0, pool=10.0)


try:
    import tomllib  # type: ignore[import-not-found]
except ImportError:  # Python 3.10
    import tomli as tomllib  # pyright: ignore[reportMissingImports]


def _load_toml(source: str) -> dict[str, object] | None:
    """Load TOML on Python 3.10 through 3.12."""
    try:
        parsed = tomllib.loads(source)
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


class SnapshotError(Exception):
    """Raised when snapshot operations fail."""


@dataclass(frozen=True, slots=True)
class CustomNodeSkip:
    """One custom-node directory that could not be captured."""

    name: str
    reason: str
    stderr: str | None = None


@dataclass(frozen=True, slots=True)
class RequiredCustomNode:
    """A workflow class whose known provider was skipped during capture."""

    class_name: str
    directory: str
    skip_reason: str
    repository: str | None = None


@dataclass(frozen=True, slots=True)
class UnverifiedCustomNodeSkip:
    """A skipped node that could not be correlated with workflow providers."""

    name: str
    reason: str


@dataclass(frozen=True, slots=True)
class CustomNodeScanReport:
    """Captured and skipped custom-node directories from one snapshot scan."""

    captured: tuple[str, ...] = ()
    skipped: tuple[CustomNodeSkip, ...] = ()
    carried: tuple[str, ...] = ()
    attributed: tuple[RequiredCustomNode, ...] = ()
    required: tuple[RequiredCustomNode, ...] = ()
    unverified: tuple[UnverifiedCustomNodeSkip, ...] = ()


@dataclass(frozen=True, slots=True)
class CarryForwardReport:
    """What a seed bundle contributed, and what it failed to cover."""

    urls_carried: tuple[str, ...]
    files_without_url: tuple[str, ...]
    seed_files_unmatched: tuple[str, ...]
    blocks_carried: tuple[str, ...]
    custom_nodes: CustomNodeScanReport = CustomNodeScanReport()
    overlay_dropped_lines: tuple[str, ...] = ()

    @property
    def has_unverified_custom_nodes(self) -> bool:
        """Whether snapshot provider coverage could not be verified."""
        return bool(self.custom_nodes.unverified)


_PhysicalIdentity = tuple[int, int] | str


@dataclass(frozen=True, slots=True)
class _ModelRoot:
    """A logical ComfyUI model root with the precedence used for scanning."""

    model_type: ModelType
    path: Path
    priority: int
    config_order: int
    section: str
    is_default: bool


@dataclass(frozen=True, slots=True)
class _ModelCandidate:
    """A selected model candidate before its contents are hashed."""

    model_type: ModelType
    subdirectory: str | None
    path: Path
    preliminary_size: int
    identity: _PhysicalIdentity
    root: _ModelRoot

    @property
    def destination(self) -> tuple[str, str, str]:
        """Return the normalized destination used by ComfyUI model lookup."""
        return (self.model_type.value, self.subdirectory or "", self.path.name)


@dataclass(frozen=True, slots=True)
class _HashResult:
    """A digest and byte accounting from one stable open file descriptor."""

    sha256: str
    bytes_read: int
    initial_size: int
    final_size: int


def _mtime_ns(file_stat: os.stat_result) -> int:
    """Get nanosecond mtime, including for minimal stat mocks in tests."""
    return getattr(file_stat, "st_mtime_ns", int(file_stat.st_mtime * 1_000_000_000))


def _same_file_snapshot(before: os.stat_result, after: os.stat_result) -> bool:
    """Return whether relevant file identity and mutation metadata stayed stable."""
    return (
        before.st_dev == after.st_dev
        and before.st_ino == after.st_ino
        and before.st_size == after.st_size
        and _mtime_ns(before) == _mtime_ns(after)
    )


def _hash_model_file(path: Path, on_chunk: Callable[[int], None]) -> _HashResult:
    """Hash one stable open file descriptor, reporting bytes as they are read.

    This deliberately remains synchronous: callers dispatch it with
    ``asyncio.to_thread`` so multi-gigabyte scans never block the CLI event
    loop while retaining normal buffered file I/O.
    """
    hasher = hashlib.sha256()
    with path.open("rb") as file:
        before = os.fstat(file.fileno())
        bytes_read = 0
        while chunk := file.read(ModelDownloader.CHUNK_SIZE):
            hasher.update(chunk)
            chunk_size = len(chunk)
            bytes_read += chunk_size
            on_chunk(chunk_size)
        after = os.fstat(file.fileno())

    if not _same_file_snapshot(before, after) or bytes_read != before.st_size:
        raise SnapshotError(
            f"Model file changed while hashing {path}; stop model writes or downloads and retry"
        )
    return _HashResult(
        sha256=hasher.hexdigest(),
        bytes_read=bytes_read,
        initial_size=before.st_size,
        final_size=after.st_size,
    )


_WORKFLOW_FALLBACK_TODO_PREFIX: Final = "TODO: export via Graph -> Export (API)"


def _render_yaml_comment(comment: str, *, indent: str = "") -> str:
    """Prefix every line of a generated comment so it cannot escape YAML."""
    return "".join(f"{indent}# {line}\n" for line in comment.splitlines() or [""])


def _render_bundle_yaml(config: BundleConfig, *, workflow_comments: tuple[str, ...] = ()) -> str:
    """Serialize a snapshot bundle and annotate generated TODOs without changing data.

    Every generated comment is deliberately ASCII (see ``_snapshot_workflow_api``'s
    fallback text) so it survives a write under any locale, including the bare
    ``LC_ALL=C`` container a Vast.ai node actually is. Non-fallback comments are
    appended as bare trailing ``#`` lines rather than annotated inline; that is
    the deliberate, simpler default, not an oversight.
    """
    data = config.model_dump(mode="json", by_alias=True, exclude_none=True)
    original_data = yaml.safe_load(yaml.safe_dump(data, sort_keys=True))
    sentinel_prefix = f"__AISHA_SNAPSHOT_URL_TODO_{uuid.uuid4().hex}_"
    sentinels: list[str] = []

    models = data.get("models")
    if isinstance(models, list):
        for model in models:
            if not isinstance(model, dict):
                continue
            files = model.get("files")
            if not isinstance(files, list):
                continue
            for file_config in files:
                if not isinstance(file_config, dict) or file_config.get("url") != "":
                    continue
                sentinel = f"{sentinel_prefix}{len(sentinels)}__"
                file_config["url"] = sentinel
                sentinels.append(sentinel)

    bundle_yaml = yaml.safe_dump(data, default_flow_style=False, sort_keys=True)
    if sentinels:
        sentinel_pattern = "|".join(re.escape(sentinel) for sentinel in sentinels)
        url_pattern = re.compile(
            rf"^(?P<indent>[ ]*)url: [\"']?(?:{sentinel_pattern})[\"']?[ ]*$",
            flags=re.MULTILINE,
        )
        bundle_yaml, replacements = url_pattern.subn(
            lambda match: f"{match.group('indent')}url: ''  # TODO: source URL",
            bundle_yaml,
        )
        if replacements != len(sentinels):
            raise SnapshotError(
                "Unable to annotate all snapshot model source URLs "
                f"({replacements} of {len(sentinels)} placeholders)"
            )

    for comment in workflow_comments:
        if comment.startswith(_WORKFLOW_FALLBACK_TODO_PREFIX):
            fallback_comment = comment
            workflow_pattern = re.compile(
                r"^(?P<indent>[ ]*)workflow_api_file: (?P<value>[^\n]+)$", flags=re.MULTILINE
            )

            def annotate_fallback(match: re.Match[str], comment: str = fallback_comment) -> str:
                lines = comment.splitlines() or [""]
                return (
                    f"{match.group('indent')}workflow_api_file: {match.group('value')}  # "
                    f"{lines[0]}\n"
                    + _render_yaml_comment("\n".join(lines[1:]), indent=match.group("indent"))
                    if len(lines) > 1
                    else f"{match.group('indent')}workflow_api_file: {match.group('value')}  # {lines[0]}"
                )

            bundle_yaml, replacements = workflow_pattern.subn(
                annotate_fallback,
                bundle_yaml,
                count=1,
            )
            if replacements == 0:
                # workflow_api_file is omitted from the dump entirely when no
                # API graph was produced (see create_snapshot) -- there is no
                # line left to annotate inline, so fall back to a trailing
                # comment like every other TODO.
                bundle_yaml += _render_yaml_comment(comment)
            elif replacements != 1:
                raise SnapshotError("Unable to annotate workflow_api_file fallback TODO")
            continue
        bundle_yaml += _render_yaml_comment(comment)

    round_tripped = yaml.safe_load(bundle_yaml)
    if round_tripped != original_data:
        raise SnapshotError("Snapshot bundle URL annotations did not round-trip through YAML")
    return bundle_yaml


def _write_bundle_files(
    config_path: Path,
    config: BundleConfig,
    requirements_path: Path | None = None,
    requirements_overlay: str | None = None,
    workflow_comments: tuple[str, ...] = (),
) -> None:
    """Write bundle.yaml and, when present, its additive requirements overlay."""
    with config_path.open("w", encoding="utf-8") as f:
        f.write(_render_bundle_yaml(config, workflow_comments=workflow_comments))

    if requirements_path is not None and requirements_overlay is not None:
        with requirements_path.open("w", encoding="utf-8") as f:
            f.write(requirements_overlay)


@dataclass(frozen=True, slots=True)
class _BaseManifest:
    """The parts of a pristine base-image manifest the overlay computation needs."""

    packages: dict[str, str] | None
    base_image: str | None
    captured_before_install: bool | None
    baked_custom_nodes: frozenset[str] | None


def _base_packages_from_manifest(
    base_manifest: Path,
) -> tuple[_BaseManifest | None, str | None, str | None]:
    """Load a pristine package inventory without logging from a worker thread."""
    try:
        payload = json.loads(base_manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, _NO_BASE_MANIFEST_MESSAGE, str(exc)
    if not isinstance(payload, dict):
        return None, _INVALID_BASE_MANIFEST_MESSAGE, None
    packages = payload.get("packages")
    baked_nodes_raw = payload.get("baked_custom_nodes")
    baked_custom_nodes = (
        frozenset(entry.casefold() for entry in baked_nodes_raw if isinstance(entry, str))
        if isinstance(baked_nodes_raw, list)
        else None
    )
    if not isinstance(packages, dict) or not all(
        isinstance(name, str) and isinstance(version, str) for name, version in packages.items()
    ):
        return (
            _BaseManifest(
                packages=None,
                base_image=payload.get("base_image")
                if isinstance(payload.get("base_image"), str)
                else None,
                captured_before_install=(
                    payload.get("captured_before_install")
                    if isinstance(payload.get("captured_before_install"), bool)
                    else None
                ),
                baked_custom_nodes=baked_custom_nodes,
            ),
            _INVALID_BASE_MANIFEST_MESSAGE,
            None,
        )
    base_image = payload.get("base_image")
    captured_before_install = payload.get("captured_before_install")
    manifest = _BaseManifest(
        packages={canonicalize_name(name): version for name, version in packages.items()},
        base_image=base_image if isinstance(base_image, str) else None,
        captured_before_install=(
            captured_before_install if isinstance(captured_before_install, bool) else None
        ),
        baked_custom_nodes=baked_custom_nodes,
    )
    return manifest, None, None


def _requirements_overlay(
    pip_freeze: str, base_packages: dict[str, str]
) -> tuple[str, tuple[str, ...]]:
    """Render the package delta from a freeze as a sorted overlay, and report dropped lines.

    An exact pin (``name==version``) that differs from the base image becomes
    a ``name==version`` overlay entry. A portable direct reference
    (``name @ url``) always overrides the base image regardless of whether
    the name appears there, and is emitted verbatim since it has no version
    to reconstruct. Only a local-only ``file://`` reference — the base
    image's own package manager pointing at a path that cannot exist on a
    deployment node — is excluded, matching the consumer's
    ``is_missing_local_reference`` rule. Every other unusable line (markers,
    ranges, extras, unparseable syntax) is reported as dropped rather than
    silently discarded.
    """
    overlay: dict[str, tuple[str, str]] = {}
    dropped: list[str] = []
    for raw_line in pip_freeze.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            requirement = Requirement(line)
        except InvalidRequirement:
            dropped.append(line)
            continue
        normalized_name = canonicalize_name(requirement.name)
        if requirement.url:
            if is_missing_local_reference(requirement.url):
                continue
            overlay[normalized_name] = (requirement.name, line)
            continue
        specifiers = tuple(requirement.specifier)
        if requirement.marker is not None or len(specifiers) != 1 or specifiers[0].operator != "==":
            dropped.append(line)
            continue
        version = specifiers[0].version
        if base_packages.get(normalized_name) != version:
            overlay[normalized_name] = (requirement.name, f"{requirement.name}=={version}")
    overlay_text = "".join(f"{line}\n" for _normalized, (_name, line) in sorted(overlay.items()))
    return overlay_text, tuple(dropped)


class SnapshotManager:
    """Creates bundle snapshots from working ComfyUI setups."""

    def __init__(
        self,
        comfyui_path: Path,
        bundles_path: Path,
        *,
        python_executable: Path,
        comfyui_url: str | None = None,
        github_token: str | None = None,
    ) -> None:
        self._comfyui_path = comfyui_path
        self._bundles_path = resolve_bundles_dir(bundles_path)
        self._python_executable = python_executable
        self._comfyui_url = comfyui_url.rstrip("/") if comfyui_url else None
        self._github_token = github_token
        self._last_custom_node_scan = CustomNodeScanReport()

    async def create_snapshot(
        self,
        name: str,
        workflow_path: Path,
        description: str | None = None,
        extra_model_paths: Path | None = None,
        scan_models: bool = True,
        *,
        carry_from: BundleConfig | None = None,
        base_manifest: Path | None = None,
        include_workflow_map: bool = True,
        force: bool = False,
    ) -> tuple[str, CarryForwardReport]:
        """Create a snapshot bundle from current ComfyUI state.

        Args:
            name: Bundle name.
            workflow_path: Path to workflow JSON.
            description: Explicit bundle description, which overrides a seed description.
            extra_model_paths: Optional path to extra_model_paths.yaml.
            scan_models: Discover installed model files and record their hashes.
            carry_from: Seed bundle whose authoring intent is carried forward.
            base_manifest: Pristine base-image package inventory used to compute an overlay.
            force: Write an incomplete artifact when a workflow provider is known to
                be missing. The artifact is annotated and is never current.

        Returns:
            The new version string and a carry-forward report.
        """
        if not self._comfyui_path.exists():
            raise SnapshotError(f"ComfyUI not found: {self._comfyui_path}")

        if not workflow_path.exists():
            raise SnapshotError(f"Workflow not found: {workflow_path}")

        if extra_model_paths is not None and not extra_model_paths.exists():
            raise SnapshotError(f"Extra model paths file not found: {extra_model_paths}")

        # Generate version
        version = self._generate_version(name)

        # Create the directory before collecting files so all later writes stay
        # within one version. A failed model capture removes this incomplete
        # version rather than leaving a bundle that looks deployable.
        bundle_dir = self._bundles_path / name / version
        bundle_dir.mkdir(parents=True)
        try:
            # Get ComfyUI commit
            comfyui_commit_result = await self._git(self._comfyui_path, "rev-parse", "HEAD")
            comfyui_commit = comfyui_commit_result[1] if comfyui_commit_result[0] == 0 else None

            base_manifest_data: _BaseManifest | None = None
            if base_manifest is not None:
                (
                    base_manifest_data,
                    overlay_skip_message,
                    overlay_skip_error,
                ) = await asyncio.to_thread(_base_packages_from_manifest, base_manifest)
                if overlay_skip_message is not None:
                    details: dict[str, str] = {
                        "message": overlay_skip_message,
                        "base_manifest": str(base_manifest),
                    }
                    if overlay_skip_error is not None:
                        details["error"] = overlay_skip_error
                    log.warning("snapshot.overlay_skipped", **details)

            # The base manifest is also the source of truth for custom nodes
            # already baked into this image.  Read it before scanning so those
            # directories never become deploy-time overlay dependencies.
            custom_nodes = await self._scan_custom_nodes(
                carry_from,
                baked_custom_nodes=(
                    base_manifest_data.baked_custom_nodes
                    if base_manifest_data is not None
                    else None
                ),
                base_manifest=base_manifest,
            )
            custom_node_report = self._last_custom_node_scan

            requirements_overlay: str | None = None
            overlay_dropped_lines: tuple[str, ...] = ()
            if base_manifest_data is not None and base_manifest_data.packages is not None:
                if base_manifest_data.captured_before_install is False:
                    log.warning(
                        "snapshot.base_manifest_not_pristine",
                        base_manifest=str(base_manifest),
                    )
                requirements_overlay, overlay_dropped_lines = _requirements_overlay(
                    await self._pip_freeze(), base_manifest_data.packages
                )
                if overlay_dropped_lines:
                    log.warning(
                        "snapshot.overlay_lines_dropped",
                        count=len(overlay_dropped_lines),
                        samples=list(overlay_dropped_lines[:5]),
                    )

            if (
                base_manifest_data is not None
                and base_manifest_data.base_image is not None
                and carry_from is not None
                and carry_from.hardware is not None
                and carry_from.hardware.base_image is not None
                and carry_from.hardware.base_image != base_manifest_data.base_image
            ):
                log.warning(
                    "snapshot.overlay_base_mismatch",
                    manifest_base_image=base_manifest_data.base_image,
                    hardware_base_image=carry_from.hardware.base_image,
                )

            # Capture the weights that made this ComfyUI installation work. Source
            # URLs cannot be inferred from a local file, but sizes and digests can.
            models = await self._scan_models(extra_model_paths) if scan_models else []
            models, carry_report = self._carry_forward_models(models, carry_from)
            carry_report = replace(
                carry_report,
                custom_nodes=custom_node_report,
                overlay_dropped_lines=overlay_dropped_lines,
            )

            if carry_from is not None and carry_from.hardware is not None:
                total_scanned_bytes = sum(
                    file.size_bytes or 0 for model in models for file in model.files
                )
                log.warning(
                    "snapshot.hardware.carried",
                    message=(
                        "Re-check hardware.min_disk_gb against the model set that was actually "
                        "captured."
                    ),
                    min_disk_gb=carry_from.hardware.min_disk_gb,
                    scanned_bytes=total_scanned_bytes,
                )

            # Rejected converter payloads are diagnostics, not bundle artifacts.
            # Keep each version's payload outside the bundles tree and discard an
            # unused temporary directory when conversion succeeds.
            rejected_dir = Path(tempfile.mkdtemp(prefix="aisha-snapshot-"))
            rejected_path = rejected_dir / f"{name}-{version}-workflow.api.json.rejected"
            workflow_graph, workflow_comments = await self._snapshot_workflow_api(
                workflow_path, rejected_path
            )
            if not rejected_path.exists():
                await asyncio.to_thread(shutil.rmtree, rejected_dir, ignore_errors=True)
            workflow_map: WorkflowMapConfig | None = None
            if include_workflow_map and workflow_graph is not None:
                workflow_map, inference_comments = infer_workflow_map(workflow_graph, models)
                workflow_comments = (*workflow_comments, *inference_comments)

            custom_node_report = await self._verify_workflow_providers(
                workflow_graph,
                custom_nodes,
                custom_node_report,
            )
            if custom_node_report.required:
                if not force:
                    raise SnapshotError(
                        self._required_custom_node_message(custom_node_report.required)
                    )
                workflow_comments = (
                    *workflow_comments,
                    *self._forced_bundle_comments(custom_node_report.required),
                )

            # Build bundle config
            seed_metadata = carry_from.metadata if carry_from is not None else None
            config = BundleConfig(
                metadata=BundleMetadata(
                    name=name,
                    version=version,
                    description=(
                        description
                        if description is not None
                        else (seed_metadata.description if seed_metadata is not None else "")
                    ),
                    created_at=datetime.now(timezone.utc),
                    tested=False,
                    author=seed_metadata.author if seed_metadata is not None else None,
                    notes=seed_metadata.notes if seed_metadata is not None else None,
                    tags=seed_metadata.tags if seed_metadata is not None else None,
                ),
                comfyui=ComfyUIConfig(commit=comfyui_commit) if comfyui_commit else None,
                custom_nodes=custom_nodes,
                models=models,
                requirements_overlay_file=(
                    "requirements.overlay.txt" if requirements_overlay is not None else None
                ),
                workflow_file="workflow.json",
                # Omitted, not "workflow.api.json", when conversion fell back:
                # a bundle.yaml naming a file that was never written would
                # fail its own `acs bundle validate` with workflow.api.missing.
                workflow_api_file="workflow.api.json" if workflow_graph is not None else None,
                workflow=workflow_map,
                extra_model_paths_file="extra_model_paths.yaml" if extra_model_paths else None,
                hardware=carry_from.hardware if carry_from is not None else None,
                generation=carry_from.generation if carry_from is not None else None,
                readiness_marker=carry_from.readiness_marker if carry_from is not None else None,
            )

            # Write files
            config_path = bundle_dir / "bundle.yaml"
            requirements_path = (
                bundle_dir / "requirements.overlay.txt"
                if requirements_overlay is not None
                else None
            )
            await asyncio.to_thread(
                _write_bundle_files,
                config_path,
                config,
                requirements_path,
                requirements_overlay,
                workflow_comments,
            )

            await asyncio.to_thread(shutil.copy2, workflow_path, bundle_dir / "workflow.json")
            if workflow_graph is not None:
                await asyncio.to_thread(
                    (bundle_dir / "workflow.api.json").write_text,
                    json.dumps(workflow_graph, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )

            if extra_model_paths:
                await asyncio.to_thread(
                    shutil.copy2, extra_model_paths, bundle_dir / "extra_model_paths.yaml"
                )

            # Unverified and forced artifacts are retained for inspection, but must
            # never become the default deployable version.
            name_dir = self._bundles_path / name
            if (
                not custom_node_report.unverified
                and not custom_node_report.required
                and not (name_dir / "current").exists()
            ):
                set_current_symlink(name_dir, version)
        except Exception:
            await asyncio.to_thread(shutil.rmtree, bundle_dir, ignore_errors=True)
            # Preserve an existing bundle family, but do not leave an empty
            # name directory behind when this was the first attempted version.
            with contextlib.suppress(OSError):
                bundle_dir.parent.rmdir()
            raise

        return version, replace(carry_report, custom_nodes=custom_node_report)

    async def _snapshot_workflow_api(
        self, workflow_path: Path, rejected_path: Path
    ) -> tuple[Mapping[str, object] | None, tuple[str, ...]]:
        """Convert a GUI Save graph before committing its API counterpart.

        A converter response is accepted only after the exact offline sync
        check used by ``acs bundle validate`` agrees with the source graph.
        The local ComfyUI v0.32 converter was checked against the qwen.rapid.aio
        workflow: its GUI ``mode=4`` node 8 is omitted from the API response.
        The disabled-node check remains a guard against a converter-version change.
        Conversion failures are deliberately non-fatal: authors can still use
        Graph → Export (API), and the generated YAML says exactly that.
        """
        fallback = (f"{_WORKFLOW_FALLBACK_TODO_PREFIX} and commit alongside workflow.json",)
        try:
            raw_graph = await asyncio.to_thread(workflow_path.read_text, encoding="utf-8")
            gui_graph = json.loads(raw_graph)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            log.warning(
                "snapshot.workflow_api_conversion_failed",
                status="local_graph_invalid",
                error=str(exc),
            )
            return None, fallback
        if (
            not isinstance(gui_graph, Mapping)
            or not isinstance(gui_graph.get("nodes"), list)
            or not isinstance(gui_graph.get("links"), list)
        ):
            log.warning(
                "snapshot.workflow_api_conversion_failed",
                status="local_graph_not_gui_save_format",
            )
            return None, fallback
        if self._comfyui_url is None:
            log.warning("snapshot.workflow_api_conversion_failed", status="comfyui_url_unset")
            return None, fallback

        convert_endpoint = f"{self._comfyui_url}/workflow/convert"
        try:
            async with httpx.AsyncClient(timeout=_WORKFLOW_CONVERTER_TIMEOUT) as client:
                with contextlib.suppress(httpx.HTTPError, ValueError):
                    version_response = await client.get(convert_endpoint)
                    if version_response.status_code == 200:
                        version_payload = version_response.json()
                        if isinstance(version_payload, Mapping) and isinstance(
                            version_payload.get("version"), str
                        ):
                            log.info(
                                "snapshot.workflow_converter",
                                version=version_payload["version"],
                            )
                response = await client.post(convert_endpoint, json=gui_graph)
        except (httpx.HTTPError, httpx.InvalidURL) as exc:
            log.warning(
                "snapshot.workflow_api_conversion_failed",
                status="connection_error",
                error=str(exc),
            )
            return None, fallback

        if response.status_code == 413:
            log.warning("snapshot.workflow_api_too_large", limit_bytes=1_048_576)
            return None, fallback
        if response.status_code != 200:
            log.warning(
                "snapshot.workflow_api_conversion_failed",
                status=response.status_code,
            )
            return None, fallback
        try:
            api_graph = response.json()
        except ValueError as exc:
            log.warning(
                "snapshot.workflow_api_conversion_failed",
                status=200,
                error=f"invalid_json: {exc}",
            )
            return None, fallback
        if not is_api_workflow(api_graph):
            await self._write_rejected_api(rejected_path, api_graph)
            log.warning(
                "snapshot.workflow_api_conversion_failed",
                status=200,
                error="converter response is not a flat API graph",
            )
            return None, fallback

        api_link_findings = check_api_graph_links(api_graph, "workflow.api.json")
        if any(finding.severity is Severity.ERROR for finding in api_link_findings):
            await self._write_rejected_api(rejected_path, api_graph)
            log.warning(
                "snapshot.workflow_api_conversion_failed",
                status=200,
                error="converter response contains dangling or self-referential API links",
            )
            return None, fallback

        sync_findings = check_workflow_sync(gui_graph, api_graph)
        # Only findings that prove the API graph disagrees with the GUI graph
        # may be ERROR. GUI format/completeness findings are deliberately
        # WARNINGs, so they cannot reject an otherwise valid conversion.
        if any(finding.severity is Severity.ERROR for finding in sync_findings):
            await self._write_rejected_api(rejected_path, api_graph)
            log.warning(
                "snapshot.workflow_api_conversion_failed",
                status=200,
                error="converter response failed GUI/API sync validation",
            )
            return None, fallback
        return api_graph, ()

    async def _write_rejected_api(self, rejected_path: Path, api_graph: object) -> None:
        """Keep a rejected converter response in a diagnostic temp directory.

        Best-effort only: a write failure here must not turn a soft converter
        fallback into a hard snapshot failure, so it degrades to a warning.
        """
        try:
            payload = json.dumps(api_graph, indent=2, ensure_ascii=False) + "\n"
        except (TypeError, ValueError):
            payload = repr(api_graph) + "\n"
        try:
            await asyncio.to_thread(rejected_path.write_text, payload, encoding="utf-8")
        except OSError as exc:
            log.warning(
                "snapshot.workflow_api_rejected_write_failed",
                path=str(rejected_path),
                error=str(exc),
            )
            return
        log.warning("snapshot.workflow_api_rejected_written", path=str(rejected_path))

    async def _verify_workflow_providers(
        self,
        api_graph: Mapping[str, object] | None,
        custom_nodes: list[CustomNodeConfig],
        scan_report: CustomNodeScanReport,
    ) -> CustomNodeScanReport:
        """Refuse a snapshot that omits a known workflow provider.

        The API graph gives the exact classes the snapshot will submit and
        ComfyUI's ``/object_info`` supplies the authoritative class-to-directory
        relationship.  A missing provider is an authoring-time failure.  If
        that relationship cannot be checked, retain the artifact but mark it
        unverified so the CLI cannot present it as a successful snapshot.
        """
        if api_graph is None:
            return self._mark_skipped_nodes_unverified(
                scan_report,
                "the workflow API graph was unavailable for provider correlation",
            )

        object_info, fetch_error = await self._fetch_object_info()
        if object_info is None:
            return self._mark_skipped_nodes_unverified(
                scan_report,
                fetch_error or "ComfyUI /object_info could not be fetched",
            )

        class_names = tuple(
            sorted(
                {
                    class_name
                    for raw_node in api_graph.values()
                    if (node := raw_node if isinstance(raw_node, Mapping) else None) is not None
                    and isinstance((class_name := node.get("class_type")), str)
                }
            )
        )
        missing_provider_metadata: list[str] = []
        attributed: list[RequiredCustomNode] = []
        required: list[RequiredCustomNode] = []
        skipped_by_directory = {skip.name.casefold(): skip for skip in scan_report.skipped}
        declared_directories = {node.name.casefold() for node in custom_nodes}

        for class_name in class_names:
            class_info = object_info.get(class_name)
            if not isinstance(class_info, Mapping):
                missing_provider_metadata.append(
                    f"ComfyUI /object_info has no provider metadata for workflow class {class_name!r}"
                )
                continue
            python_module = class_info.get("python_module")
            if not isinstance(python_module, str) or not python_module:
                missing_provider_metadata.append(
                    f"ComfyUI /object_info reported no python_module for workflow class {class_name!r}"
                )
                continue
            directory = bundle_contract.custom_node_directory(python_module)
            if directory is None:
                continue
            skip = skipped_by_directory.get(directory.casefold())
            if skip is not None:
                _, _, repository = self._pyproject_metadata(
                    self._comfyui_path / "custom_nodes" / skip.name
                )
                attribution = RequiredCustomNode(
                    class_name=class_name,
                    directory=skip.name,
                    skip_reason=skip.reason,
                    repository=repository,
                )
            else:
                attribution = RequiredCustomNode(
                    class_name=class_name,
                    directory=directory,
                    skip_reason="not_declared",
                )
            attributed.append(attribution)
            if directory.casefold() not in declared_directories:
                required.append(attribution)

        if required:
            for required_node in required:
                log.error(
                    "snapshot.custom_node_required_unprovided",
                    class_name=required_node.class_name,
                    directory=required_node.directory,
                    reason=required_node.skip_reason,
                    repository=required_node.repository,
                )
            return replace(
                scan_report,
                attributed=tuple(attributed),
                required=tuple(required),
            )
        if missing_provider_metadata:
            return self._mark_skipped_nodes_unverified(
                replace(scan_report, attributed=tuple(attributed)),
                "; ".join(missing_provider_metadata),
            )
        return replace(scan_report, attributed=tuple(attributed))

    async def _fetch_object_info(self) -> tuple[Mapping[str, object] | None, str | None]:
        """Fetch ComfyUI provider metadata with the snapshot converter's timeout."""
        if self._comfyui_url is None:
            return None, "ComfyUI URL is unset; /object_info could not be checked"
        endpoint = f"{self._comfyui_url}/object_info"
        try:
            async with httpx.AsyncClient(timeout=_WORKFLOW_CONVERTER_TIMEOUT) as client:
                response = await client.get(endpoint)
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, httpx.InvalidURL, ValueError) as exc:
            return None, f"unable to fetch /object_info from {endpoint}: {exc}"
        if not isinstance(payload, Mapping):
            return None, f"/object_info from {endpoint} returned a non-object JSON document"
        return payload, None

    def _mark_skipped_nodes_unverified(
        self, scan_report: CustomNodeScanReport, reason: str
    ) -> CustomNodeScanReport:
        """Record unverified provider coverage, even when the scan skipped nothing."""
        unverified = tuple(
            UnverifiedCustomNodeSkip(name=skip.name, reason=reason) for skip in scan_report.skipped
        )
        if not unverified:
            unverified = (UnverifiedCustomNodeSkip(name="<workflow>", reason=reason),)
        for skip in unverified:
            log.warning(
                "snapshot.custom_node_skipped_unverified",
                name=skip.name,
                reason=skip.reason,
            )
        return replace(scan_report, unverified=unverified)

    @staticmethod
    def _required_custom_node_message(required: tuple[RequiredCustomNode, ...]) -> str:
        """Render every known missing provider with the remediation it needs."""
        messages: list[str] = []
        for node in required:
            if node.skip_reason == "not_declared":
                messages.append(
                    f"snapshot aborted: workflow class {node.class_name!r} is provided by "
                    f"{node.directory!r}, which is not under ComfyUI/custom_nodes and is not "
                    "declared by this bundle. ComfyUI loaded it from elsewhere (an additional "
                    "node path, or a directory removed after start-up). Install it under "
                    "custom_nodes/ as a git clone, or declare it explicitly, before snapshotting."
                )
                continue
            message = (
                f"snapshot aborted: workflow class {node.class_name!r} is provided by custom "
                f"node {node.directory!r}, which was skipped ({node.skip_reason}). The bundle "
                "would deploy and fail at generation. Pin it first:"
            )
            if node.repository:
                message += (
                    f"\n  rm -rf custom_nodes/{node.directory}"
                    f"\n  git clone {node.repository} custom_nodes/{node.directory}"
                )
            else:
                message += (
                    f"\n  replace custom_nodes/{node.directory} with a git clone from its upstream "
                    "repository"
                )
            messages.append(message)
        return "\n\n".join(messages)

    @staticmethod
    def _forced_bundle_comments(required: tuple[RequiredCustomNode, ...]) -> tuple[str, ...]:
        """Return durable, YAML-safe TODOs for every forced missing provider."""
        comments: list[str] = []
        for node in required:
            missing_description = (
                "not declared"
                if node.skip_reason == "not_declared"
                else f"skipped: {node.skip_reason}"
            )
            comments.append(
                _normalize_workflow_comment(
                    "TODO: INCOMPLETE BUNDLE -- created with --force. "
                    f"Class {node.class_name!r} needs custom node {node.directory!r} "
                    f"({missing_description}). Deployment succeeds; generation fails. "
                    "Add it before use."
                )
            )
        return tuple(comments)

    @staticmethod
    def _model_file_target(target_subpath: str, filename: str) -> str:
        """Render one model destination for a stable operator-facing report."""
        return f"{target_subpath}/{filename}"

    def _carry_forward_models(
        self,
        models: list[ModelConfig],
        carry_from: BundleConfig | None,
    ) -> tuple[list[ModelConfig], CarryForwardReport]:
        """Apply seed authoring intent while preserving scanned local byte metadata."""
        if carry_from is None:
            return models, CarryForwardReport((), (), (), ())

        seed_files: dict[tuple[str, str], ModelFileConfig] = {
            (model.target_subpath, file.filename): file
            for model in carry_from.models
            for file in model.files
        }
        seed_models: dict[str, ModelConfig] = {
            model.target_subpath: model for model in carry_from.models
        }
        scanned_keys: set[tuple[str, str]] = set()
        urls_carried: list[str] = []
        files_without_url: list[str] = []
        carried_models: list[ModelConfig] = []

        for model in models:
            target_subpath = model.target_subpath
            seed_model = seed_models.get(target_subpath)
            carried_files: list[ModelFileConfig] = []
            for file in model.files:
                key = (target_subpath, file.filename)
                scanned_keys.add(key)
                seed_file = seed_files.get(key)
                target = self._model_file_target(*key)
                if seed_file is None:
                    files_without_url.append(target)
                    carried_files.append(file)
                    continue

                if seed_file.url:
                    urls_carried.append(target)
                carried_files.append(
                    file.model_copy(
                        update={
                            "name": seed_file.name,
                            "url": seed_file.url or "",
                        }
                    )
                )

            if seed_model is None:
                carried_models.append(model)
                continue
            carried_models.append(
                model.model_copy(
                    update={
                        "description": seed_model.description,
                        "custom_node_required": seed_model.custom_node_required,
                        "files": carried_files,
                    }
                )
            )

        unmatched = [self._model_file_target(*key) for key in seed_files if key not in scanned_keys]
        blocks_carried = tuple(
            block_name
            for block_name, block in (
                ("hardware", carry_from.hardware),
                ("generation", carry_from.generation),
                ("readiness_marker", carry_from.readiness_marker),
            )
            if block is not None
        )
        return carried_models, CarryForwardReport(
            urls_carried=tuple(sorted(urls_carried)),
            files_without_url=tuple(sorted(files_without_url)),
            seed_files_unmatched=tuple(sorted(unmatched)),
            blocks_carried=blocks_carried,
        )

    async def _scan_models(self, extra_model_paths: Path | None) -> list[ModelConfig]:
        """Discover installed model files and record their real hash and size.

        Local bytes are authoritative for checksums and sizes. URLs are the
        only unavailable provenance field, so their blank placeholders are
        explicitly marked in the generated YAML for the bundle author.
        """
        roots = self._model_roots(extra_model_paths)
        candidates_by_destination: dict[tuple[str, str, str], _ModelCandidate] = {}
        shadowed: dict[tuple[str, str, str], list[_ModelCandidate]] = defaultdict(list)
        for root in roots:
            for candidate in self._iter_model_files(root):
                selected = candidates_by_destination.get(candidate.destination)
                if selected is None:
                    candidates_by_destination[candidate.destination] = candidate
                elif selected.identity != candidate.identity:
                    shadowed[candidate.destination].append(candidate)

        self._warn_about_unknown_model_directories()

        for destination in sorted(shadowed):
            self._warn_about_shadowed_models(
                destination, candidates_by_destination[destination], shadowed[destination]
            )

        candidates = sorted(
            candidates_by_destination.values(),
            key=lambda candidate: (
                candidate.model_type.value,
                candidate.subdirectory or "",
                candidate.path.name,
            ),
        )

        if not candidates:
            return []

        hash_results = await self._hash_model_candidates(candidates)

        grouped: dict[tuple[ModelType, str | None], list[ModelFileConfig]] = defaultdict(list)
        for candidate, result in zip(candidates, hash_results, strict=True):
            if result.bytes_read <= 0:
                self._warn_about_zero_byte_model(candidate.path)
                continue
            grouped[(candidate.model_type, candidate.subdirectory)].append(
                ModelFileConfig(
                    name=candidate.path.name,
                    url="",
                    filename=candidate.path.name,
                    sha256=result.sha256,
                    size_bytes=result.bytes_read,
                )
            )

        models: list[ModelConfig] = []
        for model_type in ModelType:
            model_groups = sorted(
                (
                    (subdirectory, files)
                    for (candidate_type, subdirectory), files in grouped.items()
                    if candidate_type == model_type
                ),
                key=lambda group: group[0] or "",
            )
            for subdirectory, files in model_groups:
                group_name = (
                    f"{model_type.value}/{subdirectory}" if subdirectory else model_type.value
                )
                models.append(
                    ModelConfig(
                        name=group_name,
                        model_type=model_type.value,
                        subdirectory=subdirectory,
                        files=sorted(files, key=lambda file: file.filename),
                    )
                )
        return models

    async def _hash_model_candidates(self, candidates: list[_ModelCandidate]) -> list[_HashResult]:
        """Hash candidates concurrently while the event loop exclusively owns Rich UI."""
        if not candidates:
            return []

        total_estimate = sum(candidate.preliminary_size for candidate in candidates)
        loop = asyncio.get_running_loop()
        updates: asyncio.Queue[int | None] = asyncio.Queue()
        semaphore = asyncio.Semaphore(_MAX_CONCURRENT_HASHES)

        async def consume_progress(progress: Progress, task_id: TaskID) -> None:
            while (delta := await updates.get()) is not None:
                progress.advance(task_id, delta)

        async def hash_candidate(candidate: _ModelCandidate) -> _HashResult:
            """Run one synchronous hasher without allowing a worker to touch Rich."""
            pending_bytes = 0

            def report_progress(count: int) -> None:
                nonlocal pending_bytes
                pending_bytes += count
                if pending_bytes >= _PROGRESS_UPDATE_BYTES:
                    loop.call_soon_threadsafe(updates.put_nowait, pending_bytes)
                    pending_bytes = 0

            try:
                async with semaphore:
                    thread_task = asyncio.create_task(
                        asyncio.to_thread(_hash_model_file, candidate.path, report_progress)
                    )
                    try:
                        result = await asyncio.shield(thread_task)
                    except asyncio.CancelledError:
                        # Cancelling to_thread does not stop its file read. Keep the
                        # progress consumer alive until that worker has stopped
                        # enqueueing updates, then propagate cancellation.
                        with contextlib.suppress(Exception):
                            await asyncio.shield(thread_task)
                        raise
            finally:
                # This is also needed for a failed worker: report bytes already
                # read before surfacing its error, then let the consumer drain.
                if pending_bytes:
                    loop.call_soon_threadsafe(updates.put_nowait, pending_bytes)
            return result

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console,
        ) as progress:
            task_id = progress.add_task("Hashing model files", total=total_estimate)
            consumer = asyncio.create_task(consume_progress(progress, task_id))
            workers = [asyncio.create_task(hash_candidate(candidate)) for candidate in candidates]
            try:
                outcomes = await asyncio.gather(*workers, return_exceptions=True)
            except BaseException:
                for worker in workers:
                    if not worker.done():
                        worker.cancel()
                await asyncio.gather(*workers, return_exceptions=True)
                await asyncio.sleep(0)
                await updates.put(None)
                await consumer
                raise

            # All workers have stopped. Yield once so every progress callback
            # submitted by their final chunk has reached the queue before its
            # sentinel, then drain the consumer before closing Rich.
            await asyncio.sleep(0)
            await updates.put(None)
            await consumer

            results: list[_HashResult] = []
            for candidate, outcome in zip(candidates, outcomes, strict=True):
                if isinstance(outcome, BaseException):
                    if isinstance(outcome, SnapshotError):
                        raise outcome
                    if isinstance(outcome, OSError):
                        raise SnapshotError(
                            f"Unable to hash model file {candidate.path}: {outcome}"
                        ) from outcome
                    raise SnapshotError(
                        f"Unable to hash model file {candidate.path}: {outcome}"
                    ) from outcome
                results.append(outcome)

            actual_total = sum(result.bytes_read for result in results)
            progress.update(task_id, total=actual_total, completed=actual_total)
            return results

    @staticmethod
    def _root_description(root: _ModelRoot) -> str:
        """Return concise source metadata for an operator-facing warning."""
        return f"section {root.section!r}, root {root.path}"

    def _warn_about_shadowed_models(
        self,
        destination: tuple[str, str, str],
        selected: _ModelCandidate,
        shadowed: list[_ModelCandidate],
    ) -> None:
        """Report a real duplicate rather than silently selecting one path."""
        destination_text = "/".join(part for part in destination if part)
        selected_source = self._root_description(selected.root)
        shadowed_paths = [str(candidate.path) for candidate in shadowed]
        shadowed_sources = [self._root_description(candidate.root) for candidate in shadowed]
        console.print(
            "[yellow]Warning:[/yellow] "
            f"model destination {destination_text!r} selects {selected.path} ({selected_source}) "
            f"and shadows {', '.join(shadowed_paths)}"
        )
        log.warning(
            "snapshot.model_shadowed",
            destination=destination_text,
            selected_path=str(selected.path),
            selected_source=selected_source,
            shadowed_paths=shadowed_paths,
            shadowed_sources=shadowed_sources,
        )

    @staticmethod
    def _warn_about_zero_byte_model(path: Path) -> None:
        """Report a model candidate that cannot form a valid bundle entry."""
        console.print(f"[yellow]Warning:[/yellow] skipping zero-byte model file {path}")
        log.warning("snapshot.model_zero_bytes", path=str(path))

    def _warn_about_unknown_model_directories(self) -> None:
        """Surface model weights in untyped ComfyUI directories without guessing their type."""
        models_dir = self._comfyui_path / "models"
        try:
            with os.scandir(models_dir) as entries:
                directories = sorted(entries, key=lambda entry: entry.name)
        except FileNotFoundError:
            return
        except OSError as e:
            raise SnapshotError(f"Unable to enumerate model directory {models_dir}: {e}") from e

        known_directories = {model_type.value for model_type in ModelType}
        for entry in directories:
            if entry.name in _SKIPPED_MODEL_DIRECTORIES or entry.name in known_directories:
                continue
            path = models_dir / entry.name
            try:
                entry_stat = entry.stat(follow_symlinks=True)
            except OSError as e:
                raise SnapshotError(f"Unable to stat model path {path}: {e}") from e
            if not stat.S_ISDIR(entry_stat.st_mode):
                continue
            if file_count := self._count_model_files(path, entry_stat):
                log.warning(
                    "snapshot.unknown_model_dir",
                    directory=str(path),
                    file_count=file_count,
                )

    def _count_model_files(self, root: Path, root_stat: os.stat_result) -> int:
        """Count model files under an untyped root without following directory cycles."""
        visited = {self._physical_identity(root, root_stat)}
        directories = [root]
        file_count = 0
        while directories:
            directory = directories.pop()
            try:
                with os.scandir(directory) as entries:
                    children = sorted(entries, key=lambda entry: entry.name)
            except OSError as e:
                raise SnapshotError(f"Unable to enumerate model directory {directory}: {e}") from e

            child_directories: list[Path] = []
            for entry in children:
                if entry.name in _SKIPPED_MODEL_DIRECTORIES:
                    continue
                path = directory / entry.name
                try:
                    entry_stat = entry.stat(follow_symlinks=True)
                except OSError as e:
                    raise SnapshotError(f"Unable to stat model path {path}: {e}") from e
                if stat.S_ISDIR(entry_stat.st_mode):
                    identity = self._physical_identity(path, entry_stat)
                    if identity not in visited:
                        visited.add(identity)
                        child_directories.append(path)
                    continue
                if (
                    stat.S_ISREG(entry_stat.st_mode)
                    and path.suffix.lower() in _MODEL_EXTENSIONS
                    and not entry.name.endswith((".part", ".r2tmp"))
                ):
                    file_count += 1
            directories.extend(reversed(child_directories))
        return file_count

    def _model_roots(self, extra_model_paths: Path | None) -> list[_ModelRoot]:
        """Return roots in the same precedence order ComfyUI searches them."""
        extra_roots = self._extra_model_roots(extra_model_paths) if extra_model_paths else []
        ordered: list[_ModelRoot] = []
        for model_type in ModelType:
            configured = [root for root in extra_roots if root.model_type == model_type]
            # ComfyUI inserts each ``is_default`` root at position zero. Its
            # last configured default root therefore wins, followed by the
            # built-in root and ordinary extra roots in their YAML order.
            roots_for_type = [
                *reversed([root for root in configured if root.is_default]),
                _ModelRoot(
                    model_type=model_type,
                    path=self._comfyui_path / "models" / model_type.value,
                    priority=0,
                    config_order=-1,
                    section="ComfyUI built-in models",
                    is_default=True,
                ),
                *(root for root in configured if not root.is_default),
            ]
            ordered.extend(
                _ModelRoot(
                    model_type=root.model_type,
                    path=root.path,
                    priority=priority,
                    config_order=root.config_order,
                    section=root.section,
                    is_default=root.is_default,
                )
                for priority, root in enumerate(roots_for_type)
            )
        return ordered

    def _iter_model_files(self, root: _ModelRoot) -> list[_ModelCandidate]:
        """Return files under one root, following directory symlinks safely."""
        try:
            root_stat = root.path.stat()
        except FileNotFoundError:
            return []
        except OSError as e:
            raise SnapshotError(f"Unable to stat model directory {root.path}: {e}") from e
        if not stat.S_ISDIR(root_stat.st_mode):
            return []

        visited = {self._physical_identity(root.path, root_stat)}
        discovered: list[_ModelCandidate] = []
        directories = [root.path]
        while directories:
            directory = directories.pop()
            try:
                with os.scandir(directory) as entries:
                    children = sorted(entries, key=lambda entry: entry.name)
            except OSError as e:
                raise SnapshotError(f"Unable to enumerate model directory {directory}: {e}") from e

            relative_parent = directory.relative_to(root.path)
            relative_subdirectory = relative_parent.as_posix()
            subdirectory = None if relative_subdirectory == "." else relative_subdirectory
            child_directories: list[Path] = []
            for entry in children:
                path = directory / entry.name
                if entry.name in _SKIPPED_MODEL_DIRECTORIES:
                    continue
                try:
                    entry_stat = entry.stat(follow_symlinks=True)
                except OSError as e:
                    raise SnapshotError(f"Unable to stat model path {path}: {e}") from e

                if stat.S_ISDIR(entry_stat.st_mode):
                    identity = self._physical_identity(path, entry_stat)
                    if identity not in visited:
                        visited.add(identity)
                        child_directories.append(path)
                    continue

                if (
                    path.suffix.lower() not in _MODEL_EXTENSIONS
                    or entry.name.endswith((".part", ".r2tmp"))
                    or not stat.S_ISREG(entry_stat.st_mode)
                ):
                    continue
                if entry_stat.st_size <= 0:
                    self._warn_about_zero_byte_model(path)
                    continue
                discovered.append(
                    _ModelCandidate(
                        model_type=root.model_type,
                        subdirectory=subdirectory,
                        path=path,
                        preliminary_size=entry_stat.st_size,
                        identity=self._physical_identity(path, entry_stat),
                        root=root,
                    )
                )
            directories.extend(reversed(child_directories))
        return discovered

    @staticmethod
    def _physical_identity(path: Path, path_stat: os.stat_result) -> _PhysicalIdentity:
        """Return a stable identity for deduplication and symlink-cycle detection."""
        if path_stat.st_ino:
            return (path_stat.st_dev, path_stat.st_ino)
        try:
            return str(path.resolve(strict=True))
        except OSError as e:
            raise SnapshotError(f"Unable to resolve model path {path}: {e}") from e

    def _extra_model_roots(self, config_path: Path) -> list[_ModelRoot]:
        """Resolve native ComfyUI ``extra_model_paths.yaml`` sections."""
        try:
            raw = yaml.safe_load(config_path.read_text())
        except (OSError, yaml.YAMLError) as e:
            raise SnapshotError(f"Unable to read extra model paths file {config_path}: {e}") from e

        if raw is None:
            return []
        if not isinstance(raw, dict):
            raise SnapshotError(f"Invalid extra model paths file {config_path}: expected a mapping")

        roots: list[_ModelRoot] = []
        config_order = 0
        for section_name, section in raw.items():
            if not isinstance(section_name, str) or not isinstance(section, dict):
                raise SnapshotError(
                    f"Invalid extra model paths section {section_name!r} in {config_path}: "
                    "expected a mapping"
                )
            base_path = self._section_base_path(config_path, section_name, section)
            default_value = section.get("is_default", False)
            if not isinstance(default_value, bool):
                raise SnapshotError(
                    f"Invalid is_default in section {section_name!r} of {config_path}: "
                    "expected a boolean"
                )
            for model_type in ModelType:
                model_path = section.get(model_type.value)
                if model_path is None:
                    continue
                if not isinstance(model_path, str):
                    raise SnapshotError(
                        f"Invalid {model_type.value} path in section {section_name!r} of "
                        f"{config_path}: expected a string"
                    )
                for raw_path in model_path.splitlines():
                    if not raw_path.strip():
                        continue
                    path = self._resolve_extra_path(base_path or config_path.parent, raw_path)
                    roots.append(
                        _ModelRoot(
                            model_type=model_type,
                            path=path,
                            priority=0,
                            config_order=config_order,
                            section=section_name,
                            is_default=default_value,
                        )
                    )
                    config_order += 1
        return roots

    @staticmethod
    def _resolve_extra_path(base_path: Path, value: str) -> Path:
        """Expand a ComfyUI path relative to its configured base directory."""
        path = Path(os.path.expandvars(value.strip())).expanduser()
        return path if path.is_absolute() else base_path / path

    def _section_base_path(
        self, config_path: Path, section_name: str, section: dict[object, object]
    ) -> Path | None:
        """Resolve one optional ``base_path`` with strict configuration validation."""
        if "base_path" not in section:
            return None
        base_value = section["base_path"]
        if not isinstance(base_value, str):
            raise SnapshotError(
                f"Invalid base_path in section {section_name!r} of {config_path}: expected a string"
            )
        if not base_value.strip():
            return None
        return self._resolve_extra_path(config_path.parent, base_value)

    def _generate_version(self, bundle_name: str) -> str:
        """Generate version string in YYMMDD-nn format."""
        bundle_dir = self._bundles_path / bundle_name

        existing: list[str] = []
        if bundle_dir.exists():
            existing = [d.name for d in bundle_dir.iterdir() if d.is_dir()]

        return str(BundleVersion.create_new(existing))

    async def _scan_custom_nodes(
        self,
        carry_from: BundleConfig | None = None,
        *,
        baked_custom_nodes: frozenset[str] | None = None,
        base_manifest: Path | None = None,
    ) -> list[CustomNodeConfig]:
        """Scan custom_nodes directory for immutable, local git node pins.

        A custom-node installation may be a ComfyUI-Manager registry archive
        rather than a git clone. When its upstream GitHub tag can be resolved,
        snapshot records that tag's commit SHA. This is more precise than
        cloning the repository at whatever HEAD happens to be, but a registry
        archive at a version is not guaranteed to be byte-identical to that
        upstream tag. If no immutable pin can be resolved, the directory is
        skipped and retained in the CLI report.

        ``pip_requirements`` is never populated from a node's own
        requirements.txt: that file is installed from disk at deploy time, and
        copying its lines into an explicit argument list double-installs the
        node and breaks on any directive (``-r``, ``-c``, ``-e``) that only
        resolves relative to the file. The field instead only ever carries a
        seed bundle's authored intent forward, keyed by node name.
        """
        custom_nodes_dir = self._comfyui_path / "custom_nodes"
        if not custom_nodes_dir.exists():
            self._last_custom_node_scan = CustomNodeScanReport()
            return []

        if baked_custom_nodes is None:
            log.warning(
                "snapshot.baked_nodes_unavailable",
                base_manifest=str(base_manifest) if base_manifest is not None else None,
            )

        carried_nodes = (
            {node.name.casefold(): node for node in carry_from.custom_nodes}
            if carry_from is not None
            else {}
        )

        nodes: list[CustomNodeConfig] = []
        captured: list[str] = []
        skipped: list[CustomNodeSkip] = []
        carried: list[str] = []

        for node_dir in sorted(custom_nodes_dir.iterdir(), key=lambda path: path.name.casefold()):
            if self._is_expected_non_node(node_dir):
                continue

            if not node_dir.is_dir() or node_dir.name.startswith("."):
                skip = self._skip_custom_node(skipped, node_dir.name, "not_a_directory")
                self._carry_skipped_custom_node(nodes, carried, carried_nodes, skip)
                continue

            if baked_custom_nodes is not None and node_dir.name.casefold() in baked_custom_nodes:
                log.info("snapshot.custom_node_baked", name=node_dir.name)
                continue

            # `.git` may be a directory (normal clone) or file (worktree / submodule).
            # Do not use `.git*`: registry archives legitimately contain .github and
            # .gitignore while lacking the metadata required for a pin.
            if not (node_dir / ".git").exists():
                project_name, version, repository = self._pyproject_metadata(node_dir)
                commit_sha = await self._resolve_registry_pin(
                    repository, version, directory=node_dir.name
                )
                if commit_sha is not None and repository is not None:
                    seed_node = carried_nodes.get(node_dir.name.casefold())
                    nodes.append(
                        CustomNodeConfig(
                            name=node_dir.name,
                            git_url=repository,
                            commit_sha=commit_sha,
                            pip_requirements=seed_node.pip_requirements if seed_node else [],
                        )
                    )
                    captured.append(node_dir.name)
                    log.info(
                        "snapshot.custom_node_pinned_from_registry",
                        name=node_dir.name,
                        project_name=project_name,
                        version=version,
                        repository=repository,
                        commit_sha=commit_sha,
                        pin_source="tag-derived; registry archive not archive-verified",
                    )
                    continue
                self._warn_unsupported_custom_node_source(
                    node_dir, project_name, version, repository
                )
                skip = self._skip_custom_node(skipped, node_dir.name, "no_git_metadata")
                self._carry_skipped_custom_node(nodes, carried, carried_nodes, skip)
                continue

            root_code, root, root_stderr = await self._git(node_dir, "rev-parse", "--show-toplevel")
            if root_code != 0 or not self._is_repo_root(root, node_dir):
                skip = self._skip_custom_node(skipped, node_dir.name, "not_repo_root", root_stderr)
                self._carry_skipped_custom_node(nodes, carried, carried_nodes, skip)
                continue

            remote_code, remote_url, remote_stderr = await self._git(
                node_dir, "remote", "get-url", "origin"
            )
            if remote_code != 0 or not remote_url:
                skip = self._skip_custom_node(skipped, node_dir.name, "no_remote", remote_stderr)
                self._carry_skipped_custom_node(nodes, carried, carried_nodes, skip)
                continue

            commit_code, commit_sha, commit_stderr = await self._git(node_dir, "rev-parse", "HEAD")
            if commit_code != 0 or not commit_sha:
                skip = self._skip_custom_node(skipped, node_dir.name, "no_commit", commit_stderr)
                self._carry_skipped_custom_node(nodes, carried, carried_nodes, skip)
                continue

            requirement_lines = self._node_requirements(node_dir)
            if requirement_lines:
                log.info(
                    "snapshot.custom_node_requirements",
                    name=node_dir.name,
                    count=len(requirement_lines),
                )
            self._log_uncovered_pyproject_dependencies(node_dir, requirement_lines)
            nodes.append(
                CustomNodeConfig(
                    name=node_dir.name,
                    git_url=remote_url,
                    commit_sha=commit_sha,
                    pip_requirements=(
                        carried_nodes[node_dir.name.casefold()].pip_requirements
                        if node_dir.name.casefold() in carried_nodes
                        else []
                    ),
                )
            )
            captured.append(node_dir.name)

        self._last_custom_node_scan = CustomNodeScanReport(
            captured=tuple(captured),
            skipped=tuple(skipped),
            carried=tuple(carried),
        )
        log.info(
            "snapshot.custom_nodes_summary",
            captured=len(captured),
            skipped=len(skipped),
        )
        return nodes

    @staticmethod
    def _is_expected_non_node(path: Path) -> bool:
        """Return whether a customary helper artefact should not be reported."""
        return path.name == "__pycache__" or (
            not path.is_dir() and path.suffix.lower() in {".py", ".example"}
        )

    @staticmethod
    def _is_repo_root(root: str, repo_path: Path) -> bool:
        """Confirm that git did not ascend into ComfyUI's parent repository."""
        if not root:
            return False
        try:
            return Path(root).resolve() == repo_path.resolve()
        except OSError:
            return False

    @staticmethod
    def _skip_custom_node(
        skipped: list[CustomNodeSkip], name: str, reason: str, stderr: str | None = None
    ) -> CustomNodeSkip:
        """Record and emit a machine-readable warning for a skipped node."""
        details: dict[str, str] = {"name": name, "reason": reason}
        if stderr:
            details["stderr"] = stderr
        log.warning("snapshot.custom_node_skipped", **details)
        skip = CustomNodeSkip(name=name, reason=reason, stderr=stderr or None)
        skipped.append(skip)
        return skip

    @staticmethod
    def _carry_skipped_custom_node(
        nodes: list[CustomNodeConfig],
        carried: list[str],
        seed_nodes: Mapping[str, CustomNodeConfig],
        skip: CustomNodeSkip,
    ) -> None:
        """Keep an explicit seed pin when live inspection could not replace it."""
        seed_node = seed_nodes.get(skip.name.casefold())
        if seed_node is None:
            return
        nodes.append(seed_node)
        carried.append(seed_node.name)
        log.info(
            "snapshot.custom_node_carried",
            name=seed_node.name,
            commit_sha=seed_node.commit_sha,
            reason=skip.reason,
        )

    @staticmethod
    def _node_requirements(node_dir: Path) -> list[str]:
        """Read the meaningful requirement lines that the node itself declares."""
        requirements_path = node_dir / "requirements.txt"
        if not requirements_path.is_file():
            return []
        try:
            return [
                line
                for raw_line in requirements_path.read_text().splitlines()
                if (line := raw_line.strip()) and not line.startswith("#")
            ]
        except OSError as exc:
            log.warning(
                "snapshot.custom_node_requirements_unreadable",
                name=node_dir.name,
                path=str(requirements_path),
                error=str(exc),
            )
            return []

    @staticmethod
    def _pyproject_metadata(node_dir: Path) -> tuple[str | None, str | None, str | None]:
        """Read optional registry-install provenance without trusting it as a pin."""
        pyproject_path = node_dir / "pyproject.toml"
        if pyproject_path.is_file():
            try:
                data = _load_toml(pyproject_path.read_text())
            except (OSError, UnicodeError):
                return None, None, None
            if data is None:
                return None, None, None
            project = data.get("project")
            if not isinstance(project, dict):
                return None, None, None
            urls = project.get("urls")
            repository = urls.get("Repository") if isinstance(urls, dict) else None
            return (
                project.get("name") if isinstance(project.get("name"), str) else None,
                project.get("version") if isinstance(project.get("version"), str) else None,
                repository if isinstance(repository, str) else None,
            )

        return SnapshotManager._tracking_metadata(node_dir / ".tracking")

    @staticmethod
    def _tracking_metadata(tracking_path: Path) -> tuple[str | None, str | None, str | None]:
        """Best-effort fallback for ComfyUI-Manager's undocumented tracking file."""
        if not tracking_path.is_file():
            return None, None, None
        try:
            data = json.loads(tracking_path.read_text())
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None, None, None
        if not isinstance(data, dict):
            return None, None, None

        def string_value(*keys: str) -> str | None:
            for key in keys:
                value = data.get(key)
                if isinstance(value, str):
                    return value
            return None

        return (
            string_value("name", "title", "id"),
            string_value("version"),
            string_value("repository", "repository_url", "git_url", "url"),
        )

    def _warn_unsupported_custom_node_source(
        self,
        node_dir: Path,
        project_name: str | None,
        version: str | None,
        repository: str | None,
    ) -> None:
        """Explain why a registry archive cannot be represented in a pinned bundle."""
        log.warning(
            "snapshot.custom_node_unsupported_source",
            directory=node_dir.name,
            project_name=project_name,
            version=version,
            repository=repository,
        )
        console.print(
            "[yellow]"
            f"{node_dir.name} is a registry install (no git metadata) and was NOT captured."
            "[/yellow]"
        )
        console.print(f"  version:  {version or 'unknown'}")
        console.print(f"  upstream: {repository or 'unknown'}")
        if repository:
            console.print("  reinstall to pin it:")
            console.print(f"    rm -rf custom_nodes/{node_dir.name}")
            console.print(f"    git clone {repository} custom_nodes/{node_dir.name}")

    async def _resolve_registry_pin(
        self,
        repository: str | None,
        version: str | None,
        *,
        directory: str | None = None,
    ) -> str | None:
        """Resolve a registry version to an immutable upstream GitHub tag commit.

        Registry archives can diverge from a tag with the same version, so the
        resulting pin is deliberately logged as tag-derived rather than
        archive-verified. Network and API failures are best-effort misses: the
        caller falls back to the existing skipped-node path, and every miss is
        logged with a distinguishable reason so an operator can tell "no such
        tag" apart from "rate limited" apart from "not on GitHub".

        On 2026-08-20, ``GET https://api.comfy.org/nodes/comfyui-kjnodes/versions/1.5.0``
        returned package metadata and ``downloadUrl`` ending in ``node.zip``, but
        no commit SHA, source ref, or commit-bearing URL. The registry endpoint
        therefore cannot authoritatively pin its installed archive; tags remain
        a best-effort fallback.
        """
        if repository is None:
            return None
        if version is None:
            self._log_registry_pin_miss("version_missing", repository=repository, version=version)
            return None
        repository_parts = self._github_repository_parts(repository)
        if repository_parts is None:
            reason = (
                "repository_unparseable"
                if "github.com" in repository.casefold()
                else "repository_not_github"
            )
            self._log_registry_pin_miss(reason, repository=repository, version=version)
            return None
        owner, repo = repository_parts
        headers: dict[str, str] = {"Accept": "application/vnd.github+json"}
        if self._github_token:
            headers["Authorization"] = f"Bearer {self._github_token}"

        tags_tried = (f"v{version}", version)
        rate_limited = False
        all_tags_not_found = True
        last_status: int | None = None
        repository_has_no_tags = False
        try:
            async with httpx.AsyncClient(timeout=_WORKFLOW_CONVERTER_TIMEOUT) as client:
                for tag in tags_tried:
                    response = await client.get(
                        (
                            "https://api.github.com/repos/"
                            f"{owner}/{repo}/git/ref/tags/{quote(tag, safe='')}"
                        ),
                        headers=headers,
                    )
                    last_status = response.status_code
                    if response.status_code != 404:
                        all_tags_not_found = False
                    if response.status_code == 200:
                        payload = response.json()
                        commit_sha = await self._tag_commit_sha(
                            client, owner, repo, payload, headers
                        )
                        if commit_sha is not None:
                            return commit_sha
                        continue
                    if self._is_rate_limited_response(response):
                        rate_limited = True
                if all_tags_not_found:
                    repository_has_no_tags = await self._repository_has_no_tags(
                        client, owner, repo, headers
                    )
        except (httpx.HTTPError, httpx.InvalidURL, ValueError) as exc:
            self._log_registry_pin_miss(
                "network_error",
                repository=repository,
                version=version,
                tags=tags_tried,
                exception=type(exc).__name__,
            )
            return None

        if rate_limited:
            self._log_registry_pin_miss(
                "rate_limited",
                repository=repository,
                version=version,
                tags=tags_tried,
                status_code=403,
            )
        elif all_tags_not_found:
            if repository_has_no_tags:
                self._log_registry_pin_miss(
                    "repository_has_no_tags",
                    repository=repository,
                    version=version,
                    tags=tags_tried,
                    status_code=404,
                )
                if directory is not None:
                    self._warn_registry_repository_has_no_tags(directory, owner, repo, version)
                return None
            self._log_registry_pin_miss(
                "tag_not_found",
                repository=repository,
                version=version,
                tags=tags_tried,
                status_code=404,
            )
        else:
            self._log_registry_pin_miss(
                "http_error",
                repository=repository,
                version=version,
                tags=tags_tried,
                status_code=last_status,
            )
        return None

    @staticmethod
    async def _repository_has_no_tags(
        client: httpx.AsyncClient,
        owner: str,
        repository: str,
        headers: Mapping[str, str],
    ) -> bool:
        """Return whether GitHub confirms a repository publishes no tags.

        This runs only after both plausible version tags returned 404. Any API,
        transport, or payload failure is intentionally treated as inconclusive so
        the caller preserves the more conservative ``tag_not_found`` diagnosis.
        """
        try:
            response = await client.get(
                f"https://api.github.com/repos/{owner}/{repository}/tags?per_page=1",
                headers=headers,
            )
            return response.status_code == 200 and response.json() == []
        except (httpx.HTTPError, httpx.InvalidURL, ValueError):
            return False

    @staticmethod
    def _warn_registry_repository_has_no_tags(
        directory: str, owner: str, repository: str, version: str
    ) -> None:
        """Tell the operator why this registry archive cannot be pinned."""
        console.print(
            "[yellow]"
            f"{directory} cannot be pinned from the registry: {owner}/{repository} publishes no "
            f"git tags, so version {version} has no corresponding commit. Install it as a git "
            "clone instead:"
            "[/yellow]"
        )

    @staticmethod
    def _is_rate_limited_response(response: httpx.Response) -> bool:
        """Distinguish GitHub's rate-limit 403 from an authorization 403."""
        if response.status_code != 403:
            return False
        if response.headers.get("x-ratelimit-remaining") == "0":
            return True
        try:
            body = response.json()
        except ValueError:
            return False
        message = body.get("message") if isinstance(body, dict) else None
        return isinstance(message, str) and "rate limit" in message.casefold()

    @staticmethod
    def _log_registry_pin_miss(
        reason: str,
        *,
        repository: str | None,
        version: str | None,
        tags: tuple[str, ...] = (),
        status_code: int | None = None,
        exception: str | None = None,
    ) -> None:
        """Make every registry-pin miss diagnosable: a miss is expected, not an error."""
        log.info(
            "snapshot.registry_pin_miss",
            reason=reason,
            repository=repository,
            version=version,
            tags=tags,
            status_code=status_code,
            exception=exception,
        )

    @staticmethod
    def _github_repository_parts(repository: str) -> tuple[str, str] | None:
        """Return an owner/repository pair for GitHub HTTP(S), SSH, or PEP 508 URLs.

        Normalises a leading ``git+`` (PEP 508 style), an SSH
        ``git@github.com:owner/repo`` remote, and a ``www.github.com`` host,
        then takes the first two path segments so a URL with a trailing
        ``/tree/<ref>`` still resolves. Non-GitHub hosts are rejected.
        """
        normalized = repository.strip()
        if normalized.startswith("git+"):
            normalized = normalized[len("git+") :]
        if normalized.startswith("git@github.com:"):
            normalized = "https://github.com/" + normalized[len("git@github.com:") :]
        parsed = urlparse(normalized)
        host = parsed.netloc.casefold()
        if host.startswith("www."):
            host = host[len("www.") :]
        if parsed.scheme not in {"http", "https"} or host != "github.com":
            return None
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) < 2:
            return None
        owner, repo = parts[0], parts[1]
        if repo.endswith(".git"):
            repo = repo[:-4]
        return (owner, repo) if owner and repo else None

    @staticmethod
    async def _tag_commit_sha(
        client: httpx.AsyncClient,
        owner: str,
        repository: str,
        payload: object,
        headers: Mapping[str, str],
    ) -> str | None:
        """Dereference a lightweight or annotated Git tag to its commit SHA."""
        current = payload
        for _ in range(4):
            if not isinstance(current, Mapping):
                return None
            target = current.get("object")
            if not isinstance(target, Mapping):
                return None
            target_type = target.get("type")
            sha = target.get("sha")
            if not isinstance(sha, str) or not sha:
                return None
            if target_type == "commit":
                return sha
            if target_type != "tag":
                return None
            response = await client.get(
                f"https://api.github.com/repos/{owner}/{repository}/git/tags/{sha}",
                headers=headers,
            )
            if response.status_code != 200:
                return None
            current = response.json()
        return None

    @staticmethod
    def _pyproject_dependencies(node_dir: Path) -> list[str]:
        """Read PEP 621 dependencies without changing the node's deploy behavior."""
        pyproject_path = node_dir / "pyproject.toml"
        if not pyproject_path.is_file():
            return []
        try:
            data = _load_toml(pyproject_path.read_text())
        except (OSError, UnicodeError):
            return []
        if data is None:
            return []
        project = data.get("project")
        dependencies = project.get("dependencies") if isinstance(project, dict) else None
        return (
            [dependency for dependency in dependencies if isinstance(dependency, str)]
            if isinstance(dependencies, list)
            else []
        )

    def _log_uncovered_pyproject_dependencies(
        self, node_dir: Path, pip_requirements: list[str]
    ) -> None:
        """Make dependencies missing from requirements.txt visible to operators."""
        requirement_names: set[str] = set()
        for raw_requirement in pip_requirements:
            try:
                requirement_names.add(Requirement(raw_requirement).name.lower())
            except InvalidRequirement:
                continue

        uncovered: list[str] = []
        for dependency in self._pyproject_dependencies(node_dir):
            try:
                dependency_name = Requirement(dependency).name.lower()
            except InvalidRequirement:
                uncovered.append(dependency)
                continue
            if dependency_name not in requirement_names:
                uncovered.append(dependency)
        if uncovered:
            log.info(
                "snapshot.custom_node_pyproject_deps",
                name=node_dir.name,
                uncovered_dependencies=uncovered,
            )

    async def _git(self, repo_path: Path, *args: str) -> tuple[int | None, str, str]:
        """Run git at one path, retaining stderr for actionable skip warnings."""
        try:
            result = await asyncio.create_subprocess_exec(
                "git",
                "-C",
                str(repo_path),
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await result.communicate()
        except OSError as exc:
            return None, "", str(exc)
        return (
            result.returncode,
            stdout.decode(errors="replace").strip(),
            stderr.decode(errors="replace").strip(),
        )

    async def _pip_freeze(self) -> str:
        """Get pip freeze output from the ComfyUI interpreter's environment."""
        result = await asyncio.create_subprocess_exec(
            str(self._python_executable),
            "-m",
            "pip",
            "freeze",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await result.communicate()
        if result.returncode != 0:
            raise SnapshotError(
                f"pip freeze failed (exit {result.returncode}): {stderr.decode(errors='replace')}"
            )
        return stdout.decode()
