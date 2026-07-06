"""Snapshot management for AI Content Service."""

from __future__ import annotations

import asyncio
import contextlib
import shutil
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import yaml

from .bundle import set_current_symlink
from .config import BundleConfig, BundleMetadata, BundleVersion, ComfyUIConfig, CustomNodeConfig

if TYPE_CHECKING:
    from pathlib import Path


class SnapshotError(Exception):
    """Raised when snapshot operations fail."""

    pass


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
    ) -> str:
        """Create a snapshot bundle from current ComfyUI state.

        Args:
            name: Bundle name.
            workflow_path: Path to workflow JSON.
            description: Bundle description.
            extra_model_paths: Optional path to extra_model_paths.yaml.

        Returns:
            Version string (YYMMDD-nn format).
        """
        if not self._comfyui_path.exists():
            raise SnapshotError(f"ComfyUI not found: {self._comfyui_path}")

        if not workflow_path.exists():
            raise SnapshotError(f"Workflow not found: {workflow_path}")

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
            models=[],  # User must add manually
            requirements_lock_file="requirements.lock",
            workflow_file="workflow.json",
            extra_model_paths_file="extra_model_paths.yaml" if extra_model_paths else None,
        )

        # Write files
        config_path = bundle_dir / "bundle.yaml"
        with config_path.open("w") as f:
            yaml.dump(
                config.model_dump(mode="json", exclude_none=True), f, default_flow_style=False
            )

        requirements_path = bundle_dir / "requirements.lock"
        with requirements_path.open("w") as f:
            f.write(requirements_lock)

        shutil.copy2(workflow_path, bundle_dir / "workflow.json")

        if extra_model_paths:
            shutil.copy2(extra_model_paths, bundle_dir / "extra_model_paths.yaml")

        # Set as current only if no current version exists yet
        name_dir = self._bundles_path / name
        if not (name_dir / "current").exists():
            set_current_symlink(name_dir, version)

        return version

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
