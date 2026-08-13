"""ComfyUI management for AI Content Service."""

from __future__ import annotations

import asyncio
import contextlib
import json
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from urllib.parse import unquote, urlparse

import httpx
import structlog
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .config import CustomNodeConfig

log = structlog.get_logger()

MIN_CHECKPOINT_BYTES = 100 * 1024 * 1024  # 100 MB — floor to detect truncated downloads


class ComfyUIError(Exception):
    """Raised when ComfyUI operations fail."""


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
    ) -> None:
        self._comfyui_path = comfyui_path
        self._python_executable = python_executable
        self._port = port
        self._host = host

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
        self, requirements_path: Path, *, source: Literal["lock", "overlay"] = "lock"
    ) -> RequirementsLockDelta:
        """Install the part of a requirement lock absent from the live environment.

        Template images own the base ComfyUI/CUDA/Python environment. A bundle
        lock is therefore an optional overlay: a matching lock is measured and
        logged as a zero-cost skip, while a real delta is still passed to pip.
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
        if delta.should_install:
            delta = await self._install_requirements_delta(
                requirements_path,
                delta,
                tolerate_conflict_failure=True,
            )
        log.info(f"{log_prefix}.delta", **delta.metrics())
        return delta

    async def install_custom_node(self, node: CustomNodeConfig) -> None:
        """Install or update a custom node to specific commit."""
        custom_nodes_dir = self._comfyui_path / self.CUSTOM_NODES_DIR
        custom_nodes_dir.mkdir(exist_ok=True)

        node_dir = custom_nodes_dir / node.name

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
        # Install node requirements if present
        requirements_path = node_dir / "requirements.txt"
        if requirements_path.exists():
            await self._run_pip(["install", "-r", str(requirements_path)])

        # Install explicit pip requirements
        if node.pip_requirements:
            await self._run_pip(["install", *node.pip_requirements])

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
        # comparison, not a bind; substitutes a connectable loopback address for the wildcard host
        probe_host = (
            "127.0.0.1" if self._host in ("0.0.0.0", "::") else self._host  # noqa: S104
        )
        url = f"http://{probe_host}:{self._port}{self.OBJECT_INFO_ENDPOINT}"
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, timeout=5.0)
                return response.status_code == 200
            except httpx.RequestError:
                return False

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
                if ComfyUIManager._is_missing_local_reference(requirement.url):
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

    @staticmethod
    def _is_missing_local_reference(url: str) -> bool:
        """Whether a direct reference is an unavailable local file URL.

        Conda's ``pip freeze`` emits builder-local ``file://`` URLs. They name
        packages already present in the environment, but the source path cannot
        exist on a deployment node and must never be handed back to pip.
        """
        parsed = urlparse(url)
        if parsed.scheme != "file":
            return False
        if parsed.netloc and parsed.netloc != "localhost":
            path = Path(f"//{parsed.netloc}{unquote(parsed.path)}")
        else:
            path = Path(unquote(parsed.path))
        return not path.exists()

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
