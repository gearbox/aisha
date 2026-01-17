"""Snapshot management for AI Content Service."""



from __future__ import annotations

import asyncio
import contextlib
import shutil
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .config import BundleConfig, BundleMetadata, ComfyUIConfig, CustomNodeConfig


class SnapshotError(Exception):
    """Raised when snapshot operations fail."""

    pass


class SnapshotManager:
    """Creates bundle snapshots from working ComfyUI setups."""

    def __init__(
        self,
        comfyui_path: Path,
        bundles_path: Path,
    ) -> None:
        self._comfyui_path = comfyui_path
        self._bundles_path = bundles_path

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
        with Path.open(config_path, "w") as f:
            yaml.dump(
                config.model_dump(mode="json", exclude_none=True), f, default_flow_style=False
            )

        requirements_path = bundle_dir / "requirements.lock"
        with Path.open(requirements_path, "w") as f:
            f.write(requirements_lock)

        shutil.copy2(workflow_path, bundle_dir / "workflow.json")

        if extra_model_paths:
            shutil.copy2(extra_model_paths, bundle_dir / "extra_model_paths.yaml")

        # Set as current if first version
        if len(list((self._bundles_path / name).iterdir())) == 1:
            current_link = self._bundles_path / name / "current"
            current_link.symlink_to(version)

        return version

    def _generate_version(self, bundle_name: str) -> str:
        """Generate version string in YYMMDD-nn format."""
        today = datetime.now().strftime("%y%m%d")
        bundle_dir = self._bundles_path / bundle_name

        if not bundle_dir.exists():
            return f"{today}-01"

        # Find existing versions for today
        existing = [d.name for d in bundle_dir.iterdir() if d.is_dir() and d.name.startswith(today)]

        if not existing:
            return f"{today}-01"

        # Find next sequence number
        max_seq = 0
        for v in existing:
            with contextlib.suppress(IndexError, ValueError):
                seq = int(v.split("-")[1])
                max_seq = max(max_seq, seq)
        return f"{today}-{max_seq + 1:02d}"

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
        """Get pip freeze output."""
        result = await asyncio.create_subprocess_exec(
            "pip",
            "freeze",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await result.communicate()
        return stdout.decode()
