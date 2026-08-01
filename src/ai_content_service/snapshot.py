"""Snapshot management for AI Content Service."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import os
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn

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

_MODEL_EXTENSIONS = {".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf"}
_SKIPPED_MODEL_DIRECTORIES = {".cache", ".git"}


def _hash_model_file(path: Path, on_chunk: Callable[[int], None]) -> str:
    """Return a SHA256 digest for *path*, reporting bytes as they are read.

    This deliberately remains synchronous: callers dispatch it with
    ``asyncio.to_thread`` so multi-gigabyte scans never block the CLI event
    loop while retaining normal buffered file I/O.
    """
    hasher = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(ModelDownloader.CHUNK_SIZE):
            hasher.update(chunk)
            on_chunk(len(chunk))
    return hasher.hexdigest()


class SnapshotError(Exception):
    """Raised when snapshot operations fail."""


def _write_bundle_files(
    config_path: Path,
    config: BundleConfig,
    requirements_path: Path,
    requirements_lock: str,
) -> None:
    """Write bundle.yaml and requirements.lock; run via a single to_thread dispatch."""
    with config_path.open("w") as f:
        bundle_yaml = yaml.dump(
            config.model_dump(mode="json", exclude_none=True), default_flow_style=False
        )
        # PyYAML does not retain comments in its data model. The one annotation
        # we need is an authoring instruction for snapshot-generated empty URLs.
        bundle_yaml = bundle_yaml.replace("url: ''\n", "url: ''  # TODO: source URL\n")
        f.write(bundle_yaml)

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

        # Create bundle directory
        bundle_dir = self._bundles_path / name / version
        bundle_dir.mkdir(parents=True, exist_ok=True)

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

        return version

    async def _scan_models(self, extra_model_paths: Path | None) -> list[ModelConfig]:
        """Discover installed model files and record their real hash and size.

        Local bytes are authoritative for checksums and sizes. URLs are the
        only unavailable provenance field, so their blank placeholders are
        explicitly marked in the generated YAML for the bundle author.
        """
        roots: list[tuple[ModelType, Path]] = [
            (model_type, self._comfyui_path / "models" / model_type.value)
            for model_type in ModelType
        ]
        if extra_model_paths is not None:
            roots.extend(self._extra_model_roots(extra_model_paths))

        candidates: list[tuple[ModelType, str | None, Path, int]] = []
        # A standard root and an extra path can refer to the same destination.
        # Keep the first one, which is the normal ComfyUI location.
        seen_destinations: set[tuple[str, str | None, str]] = set()
        for model_type, root in roots:
            for subdirectory, path in self._iter_model_files(root):
                destination = (model_type.value, subdirectory, path.name)
                if destination in seen_destinations:
                    continue
                seen_destinations.add(destination)
                try:
                    size_bytes = path.stat().st_size
                except OSError as e:
                    raise SnapshotError(f"Unable to stat model file {path}: {e}") from e
                candidates.append((model_type, subdirectory, path, size_bytes))

        if not candidates:
            return []

        total_bytes = sum(size_bytes for _, _, _, size_bytes in candidates)
        hashes: dict[Path, str] = {}
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console,
        ) as progress:
            task_id = progress.add_task("Hashing model files", total=total_bytes)
            for _model_type, _subdirectory, path, _size_bytes in candidates:
                try:
                    hashes[path] = await asyncio.to_thread(
                        _hash_model_file, path, lambda count: progress.advance(task_id, count)
                    )
                except OSError as e:
                    raise SnapshotError(f"Unable to hash model file {path}: {e}") from e

        grouped: dict[tuple[ModelType, str | None], list[ModelFileConfig]] = defaultdict(list)
        for model_type, subdirectory, path, size_bytes in candidates:
            grouped[(model_type, subdirectory)].append(
                ModelFileConfig(
                    name=path.name,
                    url="",
                    filename=path.name,
                    sha256=hashes[path],
                    size_bytes=size_bytes,
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

    def _iter_model_files(self, root: Path) -> list[tuple[str | None, Path]]:
        """Return eligible files below one configured model root, deterministically."""
        if not root.is_dir():
            return []

        discovered: list[tuple[str | None, Path]] = []
        for directory, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(
                name for name in dirnames if name not in _SKIPPED_MODEL_DIRECTORIES
            )
            current = Path(directory)
            relative_parent = current.relative_to(root)
            relative_subdirectory = relative_parent.as_posix()
            subdirectory = None if relative_subdirectory == "." else relative_subdirectory
            for filename in sorted(filenames):
                path = current / filename
                if (
                    path.suffix.lower() not in _MODEL_EXTENSIONS
                    or filename.endswith((".part", ".r2tmp"))
                    or not path.is_file()
                ):
                    continue
                discovered.append((subdirectory, path))
        return discovered

    def _extra_model_roots(self, config_path: Path) -> list[tuple[ModelType, Path]]:
        """Resolve recognized ComfyUI ``extra_model_paths.yaml`` entries.

        ComfyUI accepts named sections (for example ``a111``) and also the
        newer top-level ``extra_model_paths`` wrapper. Each section may provide
        a ``base_path`` plus paths keyed by a model directory name.
        """
        try:
            raw = yaml.safe_load(config_path.read_text())
        except (OSError, yaml.YAMLError) as e:
            raise SnapshotError(f"Unable to read extra model paths file {config_path}: {e}") from e

        if raw is None:
            return []
        if not isinstance(raw, dict):
            raise SnapshotError(f"Invalid extra model paths file {config_path}: expected a mapping")

        sections_data = raw.get("extra_model_paths", raw)
        if not isinstance(sections_data, dict):
            raise SnapshotError(f"Invalid extra model paths file {config_path}: expected a mapping")

        recognized_names = {model_type.value for model_type in ModelType}
        if recognized_names.intersection(sections_data):
            sections = [sections_data]
        else:
            sections = [value for value in sections_data.values() if isinstance(value, dict)]

        roots: list[tuple[ModelType, Path]] = []
        for section in sections:
            base_value = section.get("base_path")
            base_path = self._resolve_extra_path(config_path.parent, base_value)
            for model_type in ModelType:
                model_path = section.get(model_type.value)
                if not isinstance(model_path, str) or not model_path.strip():
                    continue
                path = Path(os.path.expandvars(model_path)).expanduser()
                if not path.is_absolute():
                    path = (base_path or config_path.parent) / path
                roots.append((model_type, path))
        return roots

    @staticmethod
    def _resolve_extra_path(config_dir: Path, value: object) -> Path | None:
        if not isinstance(value, str) or not value.strip():
            return None
        path = Path(os.path.expandvars(value)).expanduser()
        return path if path.is_absolute() else config_dir / path

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
