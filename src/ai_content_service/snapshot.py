"""Snapshot management for AI Content Service."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import os
import re
import shutil
import stat
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import structlog
import yaml
from rich.console import Console
from rich.progress import BarColumn, Progress, TaskID, TextColumn

from .bundle import set_current_symlink
from .config import (
    BundleConfig,
    BundleMetadata,
    BundleVersion,
    ComfyUIConfig,
    CustomNodeConfig,
    ModelConfig,
    ModelFileConfig,
    ModelType,
)
from .downloader import ModelDownloader

if TYPE_CHECKING:
    from collections.abc import Callable

console = Console()
log = structlog.get_logger()

_MODEL_EXTENSIONS = {".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf"}
_SKIPPED_MODEL_DIRECTORIES = {".cache", ".git"}
_MAX_CONCURRENT_HASHES = 4
_PROGRESS_UPDATE_BYTES = ModelDownloader.CHUNK_SIZE * 8


class SnapshotError(Exception):
    """Raised when snapshot operations fail."""


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


def _render_bundle_yaml(config: BundleConfig) -> str:
    """Serialize a snapshot bundle and annotate only generated blank model URLs."""
    data = config.model_dump(mode="json", exclude_none=True)
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
    if not sentinels:
        return bundle_yaml

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

    round_tripped = yaml.safe_load(bundle_yaml)
    if round_tripped != original_data:
        raise SnapshotError("Snapshot bundle URL annotations did not round-trip through YAML")
    return bundle_yaml


def _write_bundle_files(
    config_path: Path,
    config: BundleConfig,
    requirements_path: Path,
    requirements_lock: str,
) -> None:
    """Write bundle.yaml and requirements.lock; run via a single to_thread dispatch."""
    with config_path.open("w") as f:
        f.write(_render_bundle_yaml(config))

    with requirements_path.open("w") as f:
        f.write(requirements_lock)


class SnapshotManager:
    """Creates bundle snapshots from working ComfyUI setups."""

    def __init__(
        self,
        comfyui_path: Path,
        bundles_path: Path,
        *,
        python_executable: Path,
    ) -> None:
        self._comfyui_path = comfyui_path
        self._bundles_path = bundles_path
        self._python_executable = python_executable

    async def create_snapshot(
        self,
        name: str,
        workflow_path: Path,
        description: str = "",
        extra_model_paths: Path | None = None,
        scan_models: bool = True,
    ) -> str:
        """Create a snapshot bundle from current ComfyUI state.

        Args:
            name: Bundle name.
            workflow_path: Path to workflow JSON.
            description: Bundle description.
            extra_model_paths: Optional path to extra_model_paths.yaml.
            scan_models: Discover installed model files and record their hashes.

        Returns:
            Version string (YYMMDD-nn format).
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
            comfyui_commit = await self._get_git_commit(self._comfyui_path)

            # Get custom nodes
            custom_nodes = await self._scan_custom_nodes()

            # Generate pip freeze
            requirements_lock = await self._pip_freeze()

            # Capture the weights that made this ComfyUI installation work. Source
            # URLs cannot be inferred from a local file, but sizes and digests can.
            models = await self._scan_models(extra_model_paths) if scan_models else []

            # Build bundle config
            config = BundleConfig(
                metadata=BundleMetadata(
                    name=name,
                    version=version,
                    description=description,
                    created_at=datetime.now(timezone.utc),
                    tested=False,
                ),
                comfyui=ComfyUIConfig(commit=comfyui_commit) if comfyui_commit else None,
                custom_nodes=custom_nodes,
                models=models,
                requirements_lock_file="requirements.lock",
                workflow_file="workflow.json",
                extra_model_paths_file="extra_model_paths.yaml" if extra_model_paths else None,
            )

            # Write files
            config_path = bundle_dir / "bundle.yaml"
            requirements_path = bundle_dir / "requirements.lock"
            await asyncio.to_thread(
                _write_bundle_files, config_path, config, requirements_path, requirements_lock
            )

            await asyncio.to_thread(shutil.copy2, workflow_path, bundle_dir / "workflow.json")

            if extra_model_paths:
                await asyncio.to_thread(
                    shutil.copy2, extra_model_paths, bundle_dir / "extra_model_paths.yaml"
                )

            # Set as current only if no current version exists yet
            name_dir = self._bundles_path / name
            if not (name_dir / "current").exists():
                set_current_symlink(name_dir, version)
        except Exception:
            await asyncio.to_thread(shutil.rmtree, bundle_dir, ignore_errors=True)
            raise

        return version

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

    async def _get_git_commit(self, repo_path: Path) -> str | None:
        """Get current git commit SHA."""
        with contextlib.suppress(Exception):
            result = await asyncio.create_subprocess_exec(
                "git",
                "rev-parse",
                "HEAD",
                cwd=repo_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await result.communicate()
            if result.returncode == 0:
                return stdout.decode().strip()
        return None

    async def _scan_custom_nodes(self) -> list[CustomNodeConfig]:
        """Scan custom_nodes directory for installed nodes."""
        custom_nodes_dir = self._comfyui_path / "custom_nodes"
        if not custom_nodes_dir.exists():
            return []

        nodes: list[CustomNodeConfig] = []

        for node_dir in custom_nodes_dir.iterdir():
            if not node_dir.is_dir() or node_dir.name.startswith("."):
                continue

            # Check if it's a git repo
            if not (node_dir / ".git").exists():
                continue

            # Get remote URL
            remote_url = await self._get_git_remote(node_dir)
            if not remote_url:
                continue

            # Get commit SHA
            commit_sha = await self._get_git_commit(node_dir)
            if not commit_sha:
                continue

            nodes.append(
                CustomNodeConfig(
                    name=node_dir.name,
                    git_url=remote_url,
                    commit_sha=commit_sha,
                )
            )

        return nodes

    async def _get_git_remote(self, repo_path: Path) -> str | None:
        """Get git remote origin URL."""
        with contextlib.suppress(Exception):
            result = await asyncio.create_subprocess_exec(
                "git",
                "remote",
                "get-url",
                "origin",
                cwd=repo_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await result.communicate()
            if result.returncode == 0:
                return stdout.decode().strip()
        return None

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
