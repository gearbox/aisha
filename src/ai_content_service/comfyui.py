"""ComfyUI management for AI Content Service."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import shutil
import stat
import tempfile
import tomllib
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Final, Literal
from urllib.parse import quote

import aiofiles
import httpx
import structlog
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

from .requirement_refs import is_missing_local_reference

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .config import CustomNodeConfig

log = structlog.get_logger()

MIN_CHECKPOINT_BYTES = 100 * 1024 * 1024  # 100 MB — floor to detect truncated downloads

_REGISTRY_API_BASE: Final = "https://api.comfy.org"
_REGISTRY_TIMEOUT: Final = httpx.Timeout(connect=10.0, read=60.0, write=60.0, pool=60.0)
# Compressed download ceiling. This limits the archive received over the network;
# extraction has separate limits because a small ZIP can expand enormously.
_REGISTRY_ARCHIVE_MAX_BYTES: Final = 512 * 1024 * 1024
# Uncompressed extraction ceiling and file-count limit for archive contents.
_REGISTRY_ARCHIVE_MAX_UNCOMPRESSED_BYTES: Final = 512 * 1024 * 1024
_REGISTRY_ARCHIVE_MAX_MEMBERS: Final = 5_000
_REGISTRY_DOWNLOAD_CHUNK_SIZE: Final = 1024 * 1024
_REGISTRY_EXTRACT_CHUNK_SIZE: Final = 1024 * 1024
_REGISTRY_VERSION_MARKER: Final = ".aisha-registry-version"


class ComfyUIError(Exception):
    """Raised when ComfyUI operations fail."""


async def fetch_registry_version(
    node_id: str,
    version: str,
    *,
    client_factory: Callable[..., httpx.AsyncClient] | None = None,
) -> Mapping[str, object] | None:
    """Fetch one immutable Comfy Registry version record, or ``None`` for 404.

    This is shared by deployment and snapshot authoring so both use the same
    authority when deciding whether a node/version pair is a registry pin.
    """
    url = f"{_REGISTRY_API_BASE}/nodes/{quote(node_id, safe='')}/versions/{quote(version, safe='')}"
    constructor = client_factory or httpx.AsyncClient
    try:
        async with constructor(timeout=_REGISTRY_TIMEOUT) as client:
            response = await client.get(url)
    except httpx.HTTPError as exc:
        raise ComfyUIError(
            f"Unable to reach the Comfy Registry for {node_id}@{version}: {exc}"
        ) from exc
    if response.status_code == 404:
        return None
    if response.status_code != 200:
        raise ComfyUIError(
            f"Comfy Registry returned {response.status_code} for {node_id}@{version}"
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise ComfyUIError(f"Comfy Registry returned invalid JSON for {node_id}@{version}") from exc
    if not isinstance(payload, Mapping):
        raise ComfyUIError(f"Comfy Registry response for {node_id}@{version} is not an object")
    return payload


@dataclass(frozen=True, slots=True)
class RequirementConflict:
    """One package whose lock version differs from the live environment."""

    name: str
    locked_version: str
    installed_version: str


@dataclass(frozen=True, slots=True)
class RequirementPin:
    """One requirement the delta can pass to pip, retaining its original syntax."""

    name: str
    version: str | None
    source: str


RequirementsInstallOutcome = Literal["installed", "skipped", "conflict_install_failed"]
RequirementsLockMetrics = dict[str, int | str | list[dict[str, str]]]


@dataclass(frozen=True, slots=True)
class RequirementsLockDelta:
    """The install-relevant difference between a lock and ``pip list``."""

    total: int
    missing: tuple[str, ...] = ()
    conflicting: tuple[RequirementConflict, ...] = ()
    unparseable: int = 0
    requirements: tuple[RequirementPin, ...] = ()
    installation_outcome: RequirementsInstallOutcome = "skipped"

    @property
    def satisfied(self) -> int:
        """Number of parsed lock entries already present at the requested version."""
        return self.total - len(self.missing) - len(self.conflicting)

    @property
    def should_install(self) -> bool:
        """Whether pip must run to honour the lock safely."""
        return bool(self.missing or self.conflicting or self.unparseable)

    def requirements_to_install(self) -> tuple[RequirementPin, ...]:
        """Return the original pins that are missing or conflict with the image."""
        install_names = {
            *(canonicalize_name(name) for name in self.missing),
            *(canonicalize_name(conflict.name) for conflict in self.conflicting),
        }
        return tuple(
            requirement
            for requirement in self.requirements
            if canonicalize_name(requirement.name) in install_names
        )

    def metrics(self) -> RequirementsLockMetrics:
        """JSON-safe, stable telemetry for ``DeploymentResult`` and timings."""
        return {
            "total": self.total,
            "satisfied": self.satisfied,
            "missing": len(self.missing),
            "conflicting": len(self.conflicting),
            "conflicting_sample": [
                {
                    "name": conflict.name,
                    "locked": conflict.locked_version,
                    "installed": conflict.installed_version,
                }
                for conflict in self.conflicting[:5]
            ],
            "unparseable": self.unparseable,
            "outcome": self.installation_outcome,
        }


@dataclass(frozen=True, slots=True)
class ExpectedArtifact:
    """A model file `verify` expects to find on disk, relative to `models/`."""

    relative_path: Path
    min_bytes: int
    declared_bytes: int | None = None


@dataclass
class ComfyUIStatus:
    """Status information about ComfyUI installation."""

    commit: str | None
    custom_node_count: int
    is_running: bool


class ComfyUIManager:
    """Manages ComfyUI installation, updates, and verification."""

    CUSTOM_NODES_DIR = "custom_nodes"
    OBJECT_INFO_ENDPOINT = "/object_info"
    DEFAULT_PORT = 8188
    # cloudflared reaches ComfyUI over the container bridge
    DEFAULT_HOST = "0.0.0.0"  # noqa: S104

    def __init__(
        self,
        comfyui_path: Path,
        python_executable: Path,
        port: int = DEFAULT_PORT,
        host: str = DEFAULT_HOST,
        registry_archive_dir: Path | None = None,
    ) -> None:
        self._comfyui_path = comfyui_path
        self._python_executable = python_executable
        self._port = port
        self._host = host
        self._registry_archive_dir = registry_archive_dir

    async def checkout(self, commit: str) -> None:
        """Checkout ComfyUI to specific commit."""
        if not self._comfyui_path.exists():
            raise ComfyUIError(f"ComfyUI not found at {self._comfyui_path}")

        # Fetch latest
        await self._run_git(["fetch", "--all"])

        # Checkout specific commit
        await self._run_git(["checkout", commit])

    async def install_base_requirements(self) -> None:
        """Install only base requirements absent from the current environment."""
        requirements_path = self._comfyui_path / "requirements.txt"
        if not requirements_path.exists():
            raise ComfyUIError("ComfyUI requirements.txt not found")

        delta = await self._resolve_requirements_delta(requirements_path)
        if delta.should_install:
            delta = await self._install_requirements_delta(requirements_path, delta)
        log.info("requirements.base.delta", **delta.metrics())

    async def install_locked_requirements(
        self,
        requirements_path: Path,
        *,
        source: Literal["lock", "overlay", "custom_node"] = "lock",
        on_conflict: Literal["install", "fail"] = "install",
    ) -> RequirementsLockDelta:
        """Install the part of a requirement lock absent from the live environment.

        Template images own the base ComfyUI/CUDA/Python environment. A bundle
        lock, overlay, or custom node's own requirements.txt is therefore
        optional: a matching file is measured and logged as a zero-cost skip,
        while a real delta is still passed to pip -- never a blind uninstall
        of an image-provided package.
        """
        if not requirements_path.exists():
            raise ComfyUIError(f"Requirements file not found: {requirements_path}")

        delta = await self._resolve_requirements_delta(requirements_path)
        log_prefix = f"requirements.{source}"
        if delta.conflicting:
            log.warning(
                f"{log_prefix}.conflict",
                packages={
                    conflict.name: {
                        "locked": conflict.locked_version,
                        "installed": conflict.installed_version,
                    }
                    for conflict in delta.conflicting[:5]
                },
            )
        if on_conflict == "fail" and delta.conflicting:
            conflicts = "; ".join(
                f"{conflict.name}: locked={conflict.locked_version} "
                f"installed={conflict.installed_version}"
                for conflict in delta.conflicting[:5]
            )
            raise ComfyUIError(
                "requirements conflict in shared ComfyUI environment; refusing to run pip: "
                f"{conflicts}"
            )
        if delta.should_install:
            delta = await self._install_requirements_delta(
                requirements_path,
                delta,
                tolerate_conflict_failure=True,
            )
        log.info(f"{log_prefix}.delta", **delta.metrics())
        return delta

    async def install_custom_node(
        self,
        node: CustomNodeConfig,
        *,
        on_conflict: Literal["install", "fail"] = "install",
    ) -> RequirementsLockDelta | None:
        """Install or update a custom node, branching on its declared source.

        Branches on the ``source`` enum only -- never on a URL substring or
        provider-name check -- so a new source never needs business-logic
        branching added elsewhere.
        """
        if node.source == "registry":
            return await self._install_registry_custom_node(node, on_conflict=on_conflict)
        return await self._install_git_custom_node(node, on_conflict=on_conflict)

    async def _install_git_custom_node(
        self,
        node: CustomNodeConfig,
        *,
        on_conflict: Literal["install", "fail"],
    ) -> RequirementsLockDelta | None:
        """Install or update a custom node to a specific git commit."""
        custom_nodes_dir = self._comfyui_path / self.CUSTOM_NODES_DIR
        custom_nodes_dir.mkdir(exist_ok=True)

        node_dir = custom_nodes_dir / node.name

        if not node.git_url:
            raise ComfyUIError(f"No git_url specified for custom node '{node.name}'")
        if node_dir.exists():
            # Update existing node
            await self._run_git(["fetch", "--all"], cwd=node_dir)
        else:
            # Clone new node
            await self._run_git(
                ["clone", node.git_url, node.name],
                cwd=custom_nodes_dir,
            )
        if not node.commit_sha:
            raise ComfyUIError(f"No commit SHA specified for custom node '{node.name}'")
        await self._run_git(["checkout", node.commit_sha], cwd=node_dir)

        return await self._install_node_requirements(node, node_dir, on_conflict=on_conflict)

    async def _install_registry_custom_node(
        self,
        node: CustomNodeConfig,
        *,
        on_conflict: Literal["install", "fail"],
    ) -> RequirementsLockDelta | None:
        """Install or update a custom node from an immutable Comfy Registry version.

        Idempotent only when Aisha's version marker proves the requested release
        was installed. A node-owned ``pyproject.toml`` is useful diagnostics,
        never provenance; the marker currently records the version only, not an
        archive digest, so local modifications remain undetected.
        """
        if not node.node_id or not node.version:
            raise ComfyUIError(f"Registry custom node '{node.name}' has no node_id or version")
        node_id, version = node.node_id, node.version

        custom_nodes_dir = self._comfyui_path / self.CUSTOM_NODES_DIR
        custom_nodes_dir.mkdir(exist_ok=True)
        node_dir = custom_nodes_dir / node.name
        self._ensure_registry_node_destination(node_dir)

        installed_version = self._installed_registry_version(node_dir)
        if installed_version == version:
            log.info(
                "custom_node.registry.up_to_date",
                name=node.name,
                node_id=node_id,
                version=version,
            )
        else:
            if node_dir.exists() and installed_version is None:
                log.info(
                    "custom_node.registry.no_provenance",
                    name=node.name,
                    installed_version=self._installed_pyproject_version(node_dir),
                    reason="marker_absent",
                )
            await self._install_registry_archive(node, node_dir, node_id, version)
            log.info(
                "custom_node.registry.installed",
                name=node.name,
                node_id=node_id,
                version=version,
            )

        return await self._install_node_requirements(node, node_dir, on_conflict=on_conflict)

    async def _install_registry_archive(
        self, node: CustomNodeConfig, node_dir: Path, node_id: str, version: str
    ) -> None:
        """Download, verify, and atomically extract one registry version.

        Never leaves a partial extraction: the download lands in a temp file
        outside ``node_dir``, verification (when a digest exists) happens
        before extraction touches disk, and extraction itself builds a
        staging directory that only replaces ``node_dir`` after it is fully
        populated. Any failure along the way is loud (``ComfyUIError``) and
        leaves neither the temp file nor the staging directory behind.
        """
        version_payload = await self._fetch_registry_version(node_id, version)
        download_url = version_payload.get("downloadUrl")
        if not isinstance(download_url, str) or not download_url:
            raise ComfyUIError(f"Comfy Registry version {node_id}@{version} has no downloadUrl")

        archive_path = await self._download_registry_archive(node.name, download_url)
        try:
            if node.archive_sha256 is not None:
                digest = await asyncio.to_thread(self._sha256_file, archive_path)
                if digest != node.archive_sha256:
                    raise ComfyUIError(
                        f"Registry archive digest mismatch for '{node.name}' "
                        f"{node_id}@{version}: expected {node.archive_sha256}, got {digest}"
                    )
            await asyncio.to_thread(
                self._extract_registry_archive,
                archive_path,
                node_dir,
                version,
                node_id=node_id,
            )
        finally:
            with contextlib.suppress(OSError):
                archive_path.unlink()

    async def _fetch_registry_version(self, node_id: str, version: str) -> Mapping[str, object]:
        """Fetch one immutable version record from the Comfy Registry HTTP API."""
        payload = await fetch_registry_version(node_id, version)
        if payload is None:
            raise ComfyUIError(f"Comfy Registry has no version {version!r} for node {node_id!r}")
        return payload

    async def _download_registry_archive(self, name: str, url: str) -> Path:
        """Stream a bounded archive to volume-backed staging outside custom_nodes."""
        archive_path = self._create_registry_archive_temp_file(name)
        try:
            async with (
                httpx.AsyncClient(timeout=_REGISTRY_TIMEOUT, follow_redirects=True) as client,
                client.stream("GET", url) as response,
            ):
                if response.status_code != 200:
                    raise ComfyUIError(
                        f"Registry archive download failed ({response.status_code}) for {url}"
                    )
                content_length = self._content_length(response.headers)
                if content_length is not None and content_length > _REGISTRY_ARCHIVE_MAX_BYTES:
                    raise ComfyUIError(
                        f"Registry archive for '{name}' exceeds the maximum allowed size "
                        f"of {_REGISTRY_ARCHIVE_MAX_BYTES} bytes"
                    )

                bytes_written = 0
                async with aiofiles.open(archive_path, "wb") as archive_file:
                    async for chunk in response.aiter_bytes(_REGISTRY_DOWNLOAD_CHUNK_SIZE):
                        bytes_written += len(chunk)
                        if bytes_written > _REGISTRY_ARCHIVE_MAX_BYTES:
                            raise ComfyUIError(
                                f"Registry archive for '{name}' exceeds the maximum allowed "
                                f"size of {_REGISTRY_ARCHIVE_MAX_BYTES} bytes"
                            )
                        await archive_file.write(chunk)
        except (httpx.HTTPError, OSError, ComfyUIError) as exc:
            with contextlib.suppress(OSError):
                archive_path.unlink()
            if isinstance(exc, OSError):
                raise ComfyUIError(f"Unable to write registry archive for '{name}': {exc}") from exc
            raise
        return archive_path

    def _create_registry_archive_temp_file(self, name: str) -> Path:
        """Create archive staging on the cache volume, then beside ComfyUI, then /tmp."""
        candidate_dirs = (self._registry_archive_dir, self._comfyui_path.parent)
        tried: set[Path] = set()
        for directory in candidate_dirs:
            if directory is None or directory in tried:
                continue
            tried.add(directory)
            try:
                directory.mkdir(parents=True, exist_ok=True)
                fd, tmp_name = tempfile.mkstemp(prefix=f".{name}-", suffix=".zip", dir=directory)
            except OSError as exc:
                log.warning(
                    "custom_node.registry.archive_staging_unavailable",
                    directory=str(directory),
                    error=str(exc),
                )
                continue
            os.close(fd)
            archive_path = Path(tmp_name)
            log.info(
                "custom_node.registry.archive_staging",
                directory=str(archive_path.parent),
                fallback=False,
            )
            return archive_path

        fd, tmp_name = tempfile.mkstemp(prefix=f".{name}-", suffix=".zip")
        os.close(fd)
        archive_path = Path(tmp_name)
        log.warning(
            "custom_node.registry.archive_staging",
            directory=str(archive_path.parent),
            fallback=True,
        )
        return archive_path

    @staticmethod
    def _content_length(headers: Mapping[str, str]) -> int | None:
        """Return a valid archive content length, if the response declares one."""
        value = headers.get("content-length") or headers.get("Content-Length")
        if value is None:
            return None
        try:
            content_length = int(value)
        except ValueError:
            return None
        return content_length if content_length >= 0 else None

    @staticmethod
    def _sha256_file(path: Path) -> str:
        hasher = hashlib.sha256()
        with path.open("rb") as file:
            while chunk := file.read(1024 * 1024):
                hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    def _archive_top_level(names: Sequence[str]) -> str | None:
        """Return one shared top-level directory, irrespective of its meaning."""
        top_levels: set[str] = set()
        for name in names:
            normalized = name.replace("\\", "/").lstrip("/")
            parts = normalized.split("/", 1)
            if len(parts) < 2 or not parts[0]:
                return None
            top_levels.add(parts[0])
        return top_levels.pop() if len(top_levels) == 1 else None

    @classmethod
    def _shared_archive_prefix(
        cls,
        names: Sequence[str],
        *,
        node_name: str,
        node_id: str | None = None,
        version: str | None = None,
    ) -> str | None:
        """Return a wrapper-shaped top-level directory shared by every member.

        A registry archive may serve its files flat (confirmed for
        comfyui-kjnodes 1.5.0: the zip root holds the node's own files
        directly, no wrapping directory) or wrapped in a single
        ``name-version/`` prefix. Both must extract to ``node.name`` --
        wrong here silently produces a directory the provider-attribution
        check does not recognise.
        """
        top_level = cls._archive_top_level(names)
        if top_level is None:
            return None
        candidates = [node_name]
        if node_id is not None:
            candidates.append(node_id)
        for candidate in candidates:
            if top_level.casefold() == candidate.casefold():
                return top_level
            if version is not None and top_level.casefold() in {
                f"{candidate}-{version}".casefold(),
                f"{candidate}_{version}".casefold(),
            }:
                return top_level
        return None

    def _extract_registry_archive(
        self,
        archive_path: Path,
        node_dir: Path,
        version: str,
        *,
        node_id: str | None = None,
    ) -> None:
        """Extract a registry archive into ``node_dir``, atomically and safely."""
        custom_nodes_dir = self._ensure_registry_node_destination(node_dir)
        with zipfile.ZipFile(archive_path) as archive:
            all_members = archive.infolist()
            if len(all_members) > _REGISTRY_ARCHIVE_MAX_MEMBERS:
                raise ComfyUIError(
                    f"Registry archive for '{node_dir.name}' has {len(all_members)} members, "
                    f"exceeding the cap of {_REGISTRY_ARCHIVE_MAX_MEMBERS}"
                )
            for info in all_members:
                if stat.S_ISLNK(info.external_attr >> 16):
                    raise ComfyUIError(
                        f"Registry archive for '{node_dir.name}' contains symlink member: "
                        f"{info.filename!r}"
                    )
            members = [info for info in all_members if not info.is_dir()]
            if not members:
                raise ComfyUIError(f"Registry archive for '{node_dir.name}' contains no files")
            declared_uncompressed_bytes = sum(info.file_size for info in members)
            if declared_uncompressed_bytes > _REGISTRY_ARCHIVE_MAX_UNCOMPRESSED_BYTES:
                raise ComfyUIError(
                    f"Registry archive for '{node_dir.name}' declares "
                    f"{declared_uncompressed_bytes} uncompressed bytes, exceeding the cap of "
                    f"{_REGISTRY_ARCHIVE_MAX_UNCOMPRESSED_BYTES} bytes"
                )
            member_names = [info.filename for info in members]
            top_level = self._archive_top_level(member_names)
            prefix = self._shared_archive_prefix(
                member_names,
                node_name=node_dir.name,
                node_id=node_id,
                version=version,
            )
            log.info(
                "custom_node.registry.archive_prefix",
                top_level=top_level,
                stripped=prefix is not None,
            )

            staging_dir = Path(
                tempfile.mkdtemp(prefix=f".{node_dir.name}-extract-", dir=custom_nodes_dir)
            )
            try:
                bytes_written = 0
                for info in members:
                    relative = self._safe_archive_member_path(info.filename, prefix)
                    if relative is None:
                        raise ComfyUIError(
                            f"Registry archive for '{node_dir.name}' has an unsafe path: "
                            f"{info.filename!r}"
                        )
                    if relative == PurePosixPath("."):
                        continue
                    target = staging_dir / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(info) as source, target.open("wb") as dest:
                        while chunk := source.read(_REGISTRY_EXTRACT_CHUNK_SIZE):
                            bytes_written += len(chunk)
                            if bytes_written > _REGISTRY_ARCHIVE_MAX_UNCOMPRESSED_BYTES:
                                raise ComfyUIError(
                                    f"Registry archive for '{node_dir.name}' wrote "
                                    f"more than {_REGISTRY_ARCHIVE_MAX_UNCOMPRESSED_BYTES} "
                                    "uncompressed bytes"
                                )
                            dest.write(chunk)
                    self._apply_registry_archive_permissions(info, target)

                # Archives without a PEP 621 project cannot otherwise prove
                # their installed version on the next deployment. Writing the
                # marker into staging keeps it atomic with the extraction.
                (staging_dir / _REGISTRY_VERSION_MARKER).write_text(
                    f"{version}\n", encoding="utf-8"
                )

                # Re-check immediately before destructive operations. The model
                # validator protects normal callers; this protects future paths
                # that call extraction directly.
                self._ensure_registry_node_destination(node_dir)
                if node_dir.exists():
                    shutil.rmtree(node_dir)
                self._ensure_registry_node_destination(node_dir)
                staging_dir.replace(node_dir)
            except BaseException:
                shutil.rmtree(staging_dir, ignore_errors=True)
                raise

    def _ensure_registry_node_destination(self, node_dir: Path) -> Path:
        """Refuse an extraction target whose parent is not ComfyUI/custom_nodes."""
        custom_nodes_dir = self._comfyui_path / self.CUSTOM_NODES_DIR
        try:
            resolved_custom_nodes_dir = custom_nodes_dir.resolve()
            if (
                node_dir.parent.resolve() != resolved_custom_nodes_dir
                or node_dir.resolve().parent != resolved_custom_nodes_dir
            ):
                raise ComfyUIError(
                    f"Refusing registry install outside ComfyUI custom_nodes: {node_dir}"
                )
        except OSError as exc:
            raise ComfyUIError(
                f"Unable to validate registry install destination {node_dir}: {exc}"
            ) from exc
        return custom_nodes_dir

    @staticmethod
    def _apply_registry_archive_permissions(info: zipfile.ZipInfo, target: Path) -> None:
        """Restore safe Unix permissions carried by a registry ZIP member."""
        if info.create_system != 3:  # Unix; non-Unix external attrs are DOS flags.
            return
        if permissions := (info.external_attr >> 16) & 0o777:
            target.chmod(permissions)

    @staticmethod
    def _safe_archive_member_path(filename: str, prefix: str | None) -> PurePosixPath | None:
        """Return a member's path relative to the (optionally stripped) archive root.

        Rejects any absolute path or ``..`` traversal segment -- a registry
        archive is fetched over the network and must never be trusted to
        stay within the directory it is extracted into (zip-slip).
        """
        normalized = filename.replace("\\", "/")
        if normalized.startswith("/"):
            return None
        parts = [part for part in normalized.split("/") if part and part != "."]
        if ".." in parts:
            return None
        if prefix is not None:
            if not parts or parts[0] != prefix:
                return None
            parts = parts[1:]
        return PurePosixPath(*parts) if parts else PurePosixPath(".")

    @staticmethod
    def _installed_pyproject_version(node_dir: Path) -> str | None:
        """Read the installed version from a node's own pyproject.toml, if any."""
        pyproject_path = node_dir / "pyproject.toml"
        if not pyproject_path.is_file():
            return None
        try:
            data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, tomllib.TOMLDecodeError):
            return None
        project = data.get("project")
        if not isinstance(project, dict):
            return None
        version = project.get("version")
        return version if isinstance(version, str) else None

    @staticmethod
    def _installed_registry_version(node_dir: Path) -> str | None:
        """Read only Aisha's registry-provenance marker, if present and readable."""
        marker_path = node_dir / _REGISTRY_VERSION_MARKER
        try:
            marker_version = marker_path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError):
            return None
        return marker_version or None

    async def _install_node_requirements(
        self,
        node: CustomNodeConfig,
        node_dir: Path,
        *,
        on_conflict: Literal["install", "fail"],
    ) -> RequirementsLockDelta | None:
        """Install a node's own requirements.txt, then the bundle author's additions.

        Install the node's own requirements.txt, if present, through the same
        delta machinery a bundle lock/overlay uses: a pin the image already
        satisfies costs nothing, and a real delta never blindly uninstalls an
        image-provided package.
        """
        requirements_path = node_dir / "requirements.txt"
        delta: RequirementsLockDelta | None = None
        if requirements_path.exists():
            delta = await self.install_locked_requirements(
                requirements_path,
                source="custom_node",
                on_conflict=on_conflict,
            )

        # Install the bundle author's additions beyond what the node declares.
        # This is never redundant with the file install above: that installs
        # the node's own requirements.txt, from its own directory, with its
        # own directives resolving correctly; this installs only what the
        # bundle author wrote in bundle.yaml by hand. Merging the two back
        # into one call reintroduces the double-install (and the directive
        # breakage) this split exists to prevent.
        if node.pip_requirements:
            await self._run_pip(["install", *node.pip_requirements])

        return delta

    async def verify(self, *, expected: Sequence[ExpectedArtifact]) -> list[str]:
        """Verify deployment artifacts exist on disk, at the path they were written to.

        Return a list of human-readable problems; empty means verified. Works
        without ComfyUI running — safe to call in provisioning context. Stats
        only, never hashes: the download path already verified checksums, and
        re-reading a multi-GB model set here would be pure waste. The
        /object_info HTTP probe is handled later by Apex's readiness gate.
        """
        models_dir = self._comfyui_path / "models"
        problems: list[str] = []
        for artifact in expected:
            full_path = models_dir / artifact.relative_path
            try:
                size = full_path.stat().st_size
            except OSError:
                log.warning("verify.artifact_missing", path=str(full_path))
                problems.append(f"{artifact.relative_path}: missing")
                continue
            if size < artifact.min_bytes:
                log.warning("verify.artifact_truncated", path=str(full_path), size_bytes=size)
                problems.append(
                    f"{artifact.relative_path}: too small ({size} bytes, "
                    f"expected at least {artifact.min_bytes})"
                )
                continue
            if artifact.declared_bytes is not None and size != artifact.declared_bytes:
                log.warning(
                    "verify.size.declared_mismatch",
                    path=str(full_path),
                    declared=artifact.declared_bytes,
                    actual=size,
                )
        return problems

    async def get_status(self) -> ComfyUIStatus:
        """Get current status of ComfyUI installation."""
        commit = await self._get_current_commit()
        custom_node_count = self._count_custom_nodes()
        is_running = await self._check_running()

        return ComfyUIStatus(
            commit=commit,
            custom_node_count=custom_node_count,
            is_running=is_running,
        )

    async def restart_and_wait(
        self,
        *,
        node_class: str | None,
        restart_command: Sequence[str],
        timeout_s: float,
        poll_interval_s: float,
    ) -> None:
        """Restart ComfyUI and wait for an optional custom-node readiness class."""
        if not restart_command:
            raise ComfyUIError("ComfyUI restart command is empty")
        process = await asyncio.create_subprocess_exec(
            *restart_command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _stdout, stderr = await process.communicate()
        if process.returncode != 0:
            raise ComfyUIError(
                "ComfyUI restart command failed "
                f"({' '.join(restart_command)}): {stderr.decode(errors='replace').strip()}"
            )

        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        down_deadline = min(deadline, loop.time() + min(30.0, timeout_s / 4))
        saw_down_transition = False
        while True:
            if not await self._check_running():
                saw_down_transition = True
                break
            if loop.time() >= down_deadline:
                break
            await asyncio.sleep(min(poll_interval_s, max(down_deadline - loop.time(), 0.0)))

        if not saw_down_transition:
            log.warning(
                "comfyui.restart.no_down_transition",
                timeout_s=min(30.0, timeout_s / 4),
            )

        process_came_up = False
        logged_fallback = False
        while True:
            if await self._check_running():
                process_came_up = True
                if node_class is None:
                    return
                ready, used_fallback = await self._node_class_available(node_class)
                if used_fallback and not logged_fallback:
                    log.debug("comfyui.restart.object_info_fallback", node_class=node_class)
                    logged_fallback = True
                if ready:
                    return
            if loop.time() >= deadline:
                break
            await asyncio.sleep(min(poll_interval_s, max(deadline - loop.time(), 0.0)))

        if not process_came_up:
            raise ComfyUIError(f"ComfyUI did not come up within {timeout_s} seconds after restart")
        raise ComfyUIError(
            f"ComfyUI came up but readiness class {node_class!r} did not appear within "
            f"{timeout_s} seconds; its custom node may have failed to import"
        )

    async def _node_class_available(self, node_class: str) -> tuple[bool, bool]:
        """Check a class endpoint, using the full document only after a 404."""
        class_endpoint = f"{self.OBJECT_INFO_ENDPOINT}/{quote(node_class, safe='')}"
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(self._server_url(class_endpoint), timeout=5.0)
            except httpx.RequestError:
                return False, False
            if response.status_code == 200:
                return self._object_info_contains(response, node_class), False
            if response.status_code != 404:
                return False, False
            try:
                full_response = await client.get(
                    self._server_url(self.OBJECT_INFO_ENDPOINT), timeout=5.0
                )
            except httpx.RequestError:
                return False, True
        return self._object_info_contains(full_response, node_class), True

    @staticmethod
    def _object_info_contains(response: httpx.Response, node_class: str) -> bool:
        """Return whether a successful object-info response names a class."""
        if response.status_code != 200:
            return False
        try:
            payload = response.json()
        except ValueError:
            return False
        return isinstance(payload, Mapping) and bool(payload) and node_class in payload

    async def _get_current_commit(self) -> str | None:
        """Get current git commit SHA."""
        if not self._comfyui_path.exists():
            return None

        with contextlib.suppress(Exception):
            result = await asyncio.create_subprocess_exec(
                "git",
                "rev-parse",
                "HEAD",
                cwd=self._comfyui_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await result.communicate()
            if result.returncode == 0:
                return stdout.decode().strip()
        return None

    def _count_custom_nodes(self) -> int:
        """Count installed custom nodes."""
        custom_nodes_dir = self._comfyui_path / self.CUSTOM_NODES_DIR
        if not custom_nodes_dir.exists():
            return 0

        return sum(
            bool(p.is_dir() and not p.name.startswith(".")) for p in custom_nodes_dir.iterdir()
        )

    async def _check_running(self) -> bool:
        """Check if ComfyUI is running."""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    self._server_url(self.OBJECT_INFO_ENDPOINT), timeout=5.0
                )
                return response.status_code == 200
            except httpx.RequestError:
                return False

    def _server_url(self, endpoint: str) -> str:
        """Return an HTTP endpoint using loopback for a wildcard listener."""
        # comparison, not a bind; substitutes a connectable loopback address for the wildcard host
        probe_host = (
            "127.0.0.1" if self._host in ("0.0.0.0", "::") else self._host  # noqa: S104
        )
        return f"http://{probe_host}:{self._port}{endpoint}"

    async def _run_git(
        self,
        args: list[str],
        cwd: Path | None = None,
    ) -> None:
        """Run a git command."""
        work_dir = cwd or self._comfyui_path

        result = await asyncio.create_subprocess_exec(
            "git",
            *args,
            cwd=work_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _stdout, stderr = await result.communicate()

        if result.returncode != 0:
            raise ComfyUIError(
                f"Git command failed: git {' '.join(args)}\n, stderr: {stderr.decode()}"
            )

    async def _run_pip(self, args: list[str]) -> None:
        """Run a pip command targeting the ComfyUI interpreter explicitly."""
        result = await asyncio.create_subprocess_exec(
            str(self._python_executable),
            "-m",
            "pip",
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _stdout, stderr = await result.communicate()

        if result.returncode != 0:
            raise ComfyUIError(
                f"Pip command failed: {self._python_executable} -m pip "
                f"{' '.join(args)}\n, stderr: {stderr.decode()}"
            )

    async def _install_requirements_delta(
        self,
        requirements_path: Path,
        delta: RequirementsLockDelta,
        *,
        tolerate_conflict_failure: bool = False,
    ) -> RequirementsLockDelta:
        """Install a requirements delta, preserving pip's original requirement syntax."""
        delta_path: Path | None = None
        pip_requirements_path = requirements_path
        try:
            if not delta.unparseable:
                delta_path = self._write_delta_requirements_file(requirements_path, delta)
                pip_requirements_path = delta_path
            await self._run_pip(["install", "-r", str(pip_requirements_path)])
        except ComfyUIError as exc:
            # An optional bundle overlay may not be able to replace a conda-owned
            # package. Never hide missing requirements or a non-authoritative lock.
            if not tolerate_conflict_failure or delta.missing or delta.unparseable:
                raise
            log.warning(
                "requirements.lock.conflict_install_failed",
                error=str(exc),
                packages=[conflict.name for conflict in delta.conflicting[:5]],
            )
            return replace(delta, installation_outcome="conflict_install_failed")
        finally:
            if delta_path is not None:
                with contextlib.suppress(OSError):
                    delta_path.unlink()
        return replace(delta, installation_outcome="installed")

    @staticmethod
    def _write_delta_requirements_file(
        requirements_path: Path, delta: RequirementsLockDelta
    ) -> Path:
        """Write only missing/conflicting pins to an adjacent temporary file."""
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=".requirements-delta-",
            suffix=".txt",
            dir=requirements_path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            try:
                for requirement in delta.requirements_to_install():
                    temporary_file.write(f"{requirement.source}\n")
            except OSError:
                with contextlib.suppress(OSError):
                    temporary_path.unlink()
                raise
        return temporary_path

    async def _resolve_requirements_delta(self, requirements_path: Path) -> RequirementsLockDelta:
        """Compare parseable ``name==version`` lock entries to ``pip list`` JSON.

        Lines pip cannot express as a single exact pin are deliberately counted
        as unparseable. They make the comparison non-authoritative, so we keep
        the safe path of invoking pip instead of incorrectly skipping an overlay.
        """
        lock, unparseable, satisfied_direct_references = self._parse_requirements_lock(
            requirements_path
        )
        installed = await self._installed_packages()
        missing: list[str] = []
        conflicting: list[RequirementConflict] = []
        for normalized_name, requirement in lock.items():
            installed_version = installed.get(normalized_name)
            if requirement.version is None or installed_version is None:
                missing.append(requirement.name)
            elif not self._versions_match(requirement.version, installed_version):
                conflicting.append(
                    RequirementConflict(
                        name=requirement.name,
                        locked_version=requirement.version,
                        installed_version=installed_version,
                    )
                )
        return RequirementsLockDelta(
            total=len(lock) + satisfied_direct_references,
            missing=tuple(sorted(missing)),
            conflicting=tuple(sorted(conflicting, key=lambda conflict: conflict.name)),
            unparseable=unparseable,
            requirements=tuple(lock.values()),
        )

    @staticmethod
    def _parse_requirements_lock(
        requirements_path: Path,
    ) -> tuple[dict[str, RequirementPin], int, int]:
        """Return installable requirements, unsafe lines, and satisfied local references."""
        packages: dict[str, RequirementPin] = {}
        unparseable = 0
        satisfied_direct_references = 0
        for raw_line in requirements_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                requirement = Requirement(line)
            except InvalidRequirement:
                unparseable += 1
                continue
            normalized_name = canonicalize_name(requirement.name)
            if requirement.url is not None:
                if is_missing_local_reference(requirement.url):
                    log.warning(
                        "requirements.lock.unresolvable_reference",
                        package=requirement.name,
                    )
                    satisfied_direct_references += 1
                    continue
                if normalized_name in packages:
                    unparseable += 1
                    continue
                packages[normalized_name] = RequirementPin(
                    name=requirement.name,
                    version=None,
                    source=line,
                )
                continue
            specifiers = tuple(requirement.specifier)
            if (
                requirement.marker is not None
                or len(specifiers) != 1
                or specifiers[0].operator != "=="
            ):
                unparseable += 1
                continue
            if normalized_name in packages:
                # A duplicate lock entry cannot be represented by one version
                # without changing its semantics. Let pip adjudicate it.
                unparseable += 1
                continue
            packages[normalized_name] = RequirementPin(
                name=requirement.name,
                version=specifiers[0].version,
                source=line,
            )
        return packages, unparseable, satisfied_direct_references

    async def _installed_packages(self) -> dict[str, str]:
        """Read the exact interpreter's installed packages from pip's JSON output."""
        args = ["list", "--format=json", "--disable-pip-version-check"]
        result = await asyncio.create_subprocess_exec(
            str(self._python_executable),
            "-m",
            "pip",
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await result.communicate()
        if result.returncode != 0:
            raise ComfyUIError(
                f"Pip command failed: {self._python_executable} -m pip "
                f"{' '.join(args)}\n, stderr: {stderr.decode()}"
            )
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise ComfyUIError("Pip list did not return valid JSON") from exc
        if not isinstance(payload, list):
            raise ComfyUIError("Pip list JSON must be an array")
        installed: dict[str, str] = {}
        for package in payload:
            if not isinstance(package, dict):
                continue
            name = package.get("name")
            version = package.get("version")
            if isinstance(name, str) and isinstance(version, str):
                installed[canonicalize_name(name)] = version
        return installed

    @staticmethod
    def _versions_match(locked: str, installed: str) -> bool:
        """Compare PEP 440 versions while preserving non-standard build distinctions."""
        try:
            return Version(locked) == Version(installed)
        except InvalidVersion:
            return locked == installed
