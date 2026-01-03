"""Bundle management for AI Content Service.

Handles bundle discovery, loading, creation, and versioning.
Bundle structure:
    config/bundles/<bundle_name>/<bundle_version>/
        ├── bundle.yaml          # Main configuration
        ├── requirements.lock    # Pip freeze output
        ├── workflow.json        # ComfyUI workflow
        └── extra_model_paths.yaml  # Optional
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import yaml
from rich.console import Console

from ai_content_service.config import (
    BundleConfig,
    BundleFiles,
    BundleMetadata,
    BundleVersion,
    ComfyUIConfig,
    CustomNode,
)

if TYPE_CHECKING:
    from pathlib import Path

    from ai_content_service.config import Settings


class BundleError(Exception):
    """Base exception for bundle operations."""


class BundleNotFoundError(BundleError):
    """Raised when a bundle or version is not found."""


class BundleValidationError(BundleError):
    """Raised when bundle validation fails."""


@dataclass
class BundleInfo:
    """Information about a discovered bundle."""

    name: str
    path: Path
    versions: list[str] = field(default_factory=list)
    current_version: str | None = None

    @property
    def current_path(self) -> Path | None:
        """Get path to current version directory."""
        return self.path / self.current_version if self.current_version else None


@dataclass
class SnapshotResult:
    """Result of a snapshot capture operation."""

    success: bool
    bundle_name: str
    version: str
    path: Path
    message: str


class BundleManager:
    """Manages bundle discovery, loading, and creation."""

    BUNDLE_YAML = "bundle.yaml"
    REQUIREMENTS_LOCK = "requirements.lock"
    WORKFLOW_JSON = "workflow.json"
    EXTRA_MODEL_PATHS = "extra_model_paths.yaml"
    CURRENT_SYMLINK = "current"

    def __init__(
        self,
        settings: Settings,
        console: Console | None = None,
    ) -> None:
        self._settings = settings
        self._console = console or Console()
        self._bundles_path = settings.bundles_path

    def ensure_bundles_directory(self) -> None:
        """Ensure the bundles directory exists."""
        self._bundles_path.mkdir(parents=True, exist_ok=True)

    def list_bundles(self) -> list[BundleInfo]:
        """List all available bundles."""
        bundles: list[BundleInfo] = []

        if not self._bundles_path.exists():
            return bundles

        for bundle_dir in sorted(self._bundles_path.iterdir()):
            if not bundle_dir.is_dir() or bundle_dir.name.startswith("."):
                continue

            info = self._get_bundle_info(bundle_dir)
            if info.versions:  # Only include bundles with at least one version
                bundles.append(info)

        return bundles

    def get_bundle(self, bundle_name: str) -> BundleInfo:
        """Get information about a specific bundle."""
        bundle_path = self._bundles_path / bundle_name

        if not bundle_path.exists():
            raise BundleNotFoundError(f"Bundle not found: {bundle_name}")

        return self._get_bundle_info(bundle_path)

    def _get_bundle_info(self, bundle_path: Path) -> BundleInfo:
        """Extract bundle information from directory."""
        versions: list[str] = []
        current_version: str | None = None

        # Find all version directories
        versions.extend(
            item.name
            for item in sorted(bundle_path.iterdir())
            if (
                item.is_dir()
                and not item.name.startswith(".")
                and self._is_valid_version(item.name)
            )
        )
        # Check for current symlink
        current_link = bundle_path / self.CURRENT_SYMLINK
        if current_link.is_symlink():
            target = current_link.resolve()
            if target.exists() and target.name in versions:
                current_version = target.name

        return BundleInfo(
            name=bundle_path.name,
            path=bundle_path,
            versions=sorted(versions, reverse=True),  # Newest first
            current_version=current_version,
        )

    def _is_valid_version(self, version: str) -> bool:
        """Check if version string matches YYMMDD-nn format."""
        import re

        return bool(re.match(r"^\d{6}-\d{2}$", version))

    def load_bundle(
        self,
        bundle_name: str,
        version: str | None = None,
    ) -> BundleFiles:
        """Load a complete bundle configuration.

        Args:
            bundle_name: Name of the bundle (e.g., "wan_2.2_i2v")
            version: Specific version or None to use current symlink

        Returns:
            BundleFiles containing all bundle data
        """
        bundle_info = self.get_bundle(bundle_name)

        # Resolve version
        if version is None:
            if bundle_info.current_version is None:
                raise BundleNotFoundError(
                    f"Bundle '{bundle_name}' has no current version set. "
                    f"Available versions: {bundle_info.versions}"
                )
            version = bundle_info.current_version

        if version not in bundle_info.versions:
            raise BundleNotFoundError(
                f"Version '{version}' not found in bundle '{bundle_name}'. "
                f"Available: {bundle_info.versions}"
            )

        version_path = bundle_info.path / version
        return self._load_bundle_files(version_path)

    def _load_bundle_files(self, version_path: Path) -> BundleFiles:
        """Load all files from a bundle version directory."""
        # Load bundle.yaml
        bundle_yaml_path = version_path / self.BUNDLE_YAML
        if not bundle_yaml_path.exists():
            raise BundleValidationError(f"Missing {self.BUNDLE_YAML} in {version_path}")

        with bundle_yaml_path.open() as f:
            bundle_data = yaml.safe_load(f)

        bundle_config = BundleConfig.model_validate(bundle_data)

        # Load requirements.lock
        requirements_path = version_path / bundle_config.requirements_lock_file
        if not requirements_path.exists():
            raise BundleValidationError(
                f"Missing {bundle_config.requirements_lock_file} in {version_path}"
            )
        requirements_lock = requirements_path.read_text()

        # Load workflow.json
        workflow_path = version_path / bundle_config.workflow_file
        if not workflow_path.exists():
            raise BundleValidationError(f"Missing {bundle_config.workflow_file} in {version_path}")
        with workflow_path.open() as f:
            workflow_json = json.load(f)

        # Load extra_model_paths.yaml (optional)
        extra_model_paths: str | None = None
        if bundle_config.extra_model_paths_file:
            extra_paths_path = version_path / bundle_config.extra_model_paths_file
            if extra_paths_path.exists():
                extra_model_paths = extra_paths_path.read_text()

        return BundleFiles(
            bundle_config=bundle_config,
            requirements_lock=requirements_lock,
            workflow_json=workflow_json,
            extra_model_paths=extra_model_paths,
        )

    def set_current_version(self, bundle_name: str, version: str) -> None:
        """Set the current version symlink for a bundle."""
        bundle_info = self.get_bundle(bundle_name)

        if version not in bundle_info.versions:
            raise BundleNotFoundError(f"Version '{version}' not found in bundle '{bundle_name}'")

        current_link = bundle_info.path / self.CURRENT_SYMLINK
        _target_path = bundle_info.path / version

        # Remove existing symlink if present
        if current_link.is_symlink() or current_link.exists():
            current_link.unlink()

        # Create relative symlink
        current_link.symlink_to(version)
        self._console.print(
            f"[green]✓ Set current version for '{bundle_name}' to '{version}'[/green]"
        )

    def create_snapshot(
        self,
        bundle_name: str,
        workflow_path: Path,
        description: str = "",
        extra_model_paths_path: Path | None = None,
        set_as_current: bool = True,
    ) -> SnapshotResult:
        """Capture a snapshot from the current ComfyUI installation.

        Args:
            bundle_name: Name for the bundle
            workflow_path: Path to workflow.json file
            description: Description for this bundle version
            extra_model_paths_path: Optional path to extra_model_paths.yaml
            set_as_current: Whether to set this as the current version

        Returns:
            SnapshotResult with details about the created snapshot
        """
        self.ensure_bundles_directory()

        # Validate inputs
        if not workflow_path.exists():
            raise BundleError(f"Workflow file not found: {workflow_path}")

        comfyui_path = self._settings.comfyui_path
        if not comfyui_path.exists():
            raise BundleError(f"ComfyUI not found at: {comfyui_path}")

        # Get ComfyUI commit
        comfyui_commit = self._get_git_commit(comfyui_path)
        if not comfyui_commit:
            raise BundleError("Failed to get ComfyUI git commit")

        # Discover custom nodes
        custom_nodes = self._discover_custom_nodes()

        # Get pip freeze output
        requirements_lock = self._capture_pip_freeze()

        # Determine version
        bundle_path = self._bundles_path / bundle_name
        existing_versions = []
        if bundle_path.exists():
            bundle_info = self._get_bundle_info(bundle_path)
            existing_versions = bundle_info.versions

        new_version = BundleVersion.create_new(existing_versions)
        version_path = bundle_path / str(new_version)
        version_path.mkdir(parents=True, exist_ok=True)

        # Create bundle config
        metadata = BundleMetadata(
            name=bundle_name,
            version=str(new_version),
            description=description,
            tested=False,
        )

        bundle_config = BundleConfig(
            metadata=metadata,
            comfyui=ComfyUIConfig(commit=comfyui_commit),
            custom_nodes=custom_nodes,
            models=[],  # Models to be added manually or via separate command
            extra_model_paths_file=(self.EXTRA_MODEL_PATHS if extra_model_paths_path else None),
        )

        # Write bundle.yaml
        bundle_yaml_path = version_path / self.BUNDLE_YAML
        with bundle_yaml_path.open("w") as f:
            yaml.dump(
                bundle_config.model_dump(mode="json", exclude_none=True),
                f,
                default_flow_style=False,
                sort_keys=False,
            )

        # Write requirements.lock
        requirements_path = version_path / self.REQUIREMENTS_LOCK
        requirements_path.write_text(requirements_lock)

        # Copy workflow.json
        workflow_dest = version_path / self.WORKFLOW_JSON
        shutil.copy2(workflow_path, workflow_dest)

        # Copy extra_model_paths.yaml if provided
        if extra_model_paths_path and extra_model_paths_path.exists():
            extra_dest = version_path / self.EXTRA_MODEL_PATHS
            shutil.copy2(extra_model_paths_path, extra_dest)

        # Set as current if requested
        if set_as_current:
            self.set_current_version(bundle_name, str(new_version))

        self._console.print(f"[green]✓ Created snapshot: {bundle_name}/{new_version}[/green]")

        return SnapshotResult(
            success=True,
            bundle_name=bundle_name,
            version=str(new_version),
            path=version_path,
            message=f"Snapshot created successfully at {version_path}",
        )

    def _get_git_commit(self, repo_path: Path) -> str | None:
        """Get the current git commit SHA for a repository."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return None

    def _get_git_remote_url(self, repo_path: Path) -> str | None:
        """Get the git remote URL for a repository."""
        try:
            result = subprocess.run(
                ["git", "remote", "get-url", "origin"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return None

    def _discover_custom_nodes(self) -> list[CustomNode]:
        """Discover installed custom nodes and their versions."""
        custom_nodes: list[CustomNode] = []
        nodes_path = self._settings.custom_nodes_path

        if not nodes_path.exists():
            return custom_nodes

        for node_dir in sorted(nodes_path.iterdir()):
            if not node_dir.is_dir() or node_dir.name.startswith("."):
                continue

            # Skip if not a git repository
            git_dir = node_dir / ".git"
            if not git_dir.exists():
                self._console.print(
                    f"[yellow]Warning: {node_dir.name} is not a git repository, skipping[/yellow]"
                )
                continue

            commit = self._get_git_commit(node_dir)
            remote_url = self._get_git_remote_url(node_dir)

            if not commit or not remote_url:
                self._console.print(
                    f"[yellow]Warning: Could not get git info for {node_dir.name}[/yellow]"
                )
                continue

            custom_nodes.append(
                CustomNode(
                    name=node_dir.name,
                    git_url=remote_url,
                    commit_sha=commit,
                )
            )

        return custom_nodes

    def _capture_pip_freeze(self) -> str:
        """Capture pip freeze output."""
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout

    def delete_version(self, bundle_name: str, version: str) -> None:
        """Delete a specific bundle version."""
        bundle_info = self.get_bundle(bundle_name)

        if version not in bundle_info.versions:
            raise BundleNotFoundError(f"Version '{version}' not found in bundle '{bundle_name}'")

        # Don't allow deleting current version
        if version == bundle_info.current_version:
            raise BundleError(
                "Cannot delete current version. Set a different version as current first."
            )

        version_path = bundle_info.path / version
        shutil.rmtree(version_path)
        self._console.print(
            f"[green]✓ Deleted version '{version}' from bundle '{bundle_name}'[/green]"
        )

    def resolve_bundle(
        self,
        bundle_name: str | None = None,
        version: str | None = None,
    ) -> tuple[str, str]:
        """Resolve bundle name and version from arguments or environment.

        Args:
            bundle_name: Explicit bundle name or None to use settings/env
            version: Explicit version or None to use current symlink

        Returns:
            Tuple of (bundle_name, version)
        """
        # Resolve bundle name
        resolved_name = bundle_name or self._settings.bundle
        if not resolved_name:
            raise BundleError(
                "No bundle specified. Use --bundle flag or set ACS_BUNDLE environment variable."
            )

        # Resolve version
        if version:
            resolved_version = version
        elif self._settings.bundle_version:
            resolved_version = self._settings.bundle_version
        else:
            # Use current symlink
            bundle_info = self.get_bundle(resolved_name)
            if not bundle_info.current_version:
                raise BundleError(
                    f"Bundle '{resolved_name}' has no current version set. "
                    f"Use --version flag or run: acs bundle set-current {resolved_name} <version>"
                )
            resolved_version = bundle_info.current_version

        return resolved_name, resolved_version
