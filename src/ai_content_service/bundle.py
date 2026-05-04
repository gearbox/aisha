"""Bundle management for AI Content Service."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import yaml

from .config import BundleConfig

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from .config import Settings


class BundleError(Exception):
    """Raised when bundle operations fail."""

    pass


class BundleNotFoundError(BundleError):
    """Raised when a bundle or version is not found."""

    pass


class BundleValidationError(BundleError):
    """Raised when bundle configuration fails validation."""

    pass


@dataclass
class BundleFiles:
    """Loaded bundle content."""

    bundle_config: BundleConfig
    requirements_lock: str
    workflow_json: dict[str, Any]

    @property
    def expected_node_types(self) -> set[str]:
        return {n["type"] for n in self.workflow_json.get("nodes", []) if "type" in n}


@dataclass
class BundleInfo:
    """Summary information about a bundle."""

    name: str
    current_version: str | None
    versions: list[str] = field(default_factory=list)


@dataclass
class VersionInfo:
    """Summary information about a bundle version."""

    version: str
    tested: bool
    description: str


class BundleManager:
    """Manages bundle storage and retrieval.

    Bundles are stored in a directory structure:

        bundles_path/
        ├── bundle_name/
        │   ├── current -> version/  # Symlink to active version
        │   ├── 260101-01/
        │   │   ├── bundle.yaml
        │   │   ├── requirements.lock
        │   │   └── workflow.json
        │   └── 260101-02/
        │       └── ...
        └── another_bundle/
            └── ...
    """

    BUNDLE_CONFIG_FILE = "bundle.yaml"
    CURRENT_LINK = "current"

    def __init__(self, settings: Settings) -> None:
        self._bundles_path = settings.bundles_path

    def list_bundles(self) -> list[BundleInfo]:
        """List all available bundles."""
        if not self._bundles_path.exists():
            return []

        bundles: list[BundleInfo] = []
        for path in sorted(self._bundles_path.iterdir()):
            if not path.is_dir() or path.name.startswith("."):
                continue

            version_names = sorted(p.name for p in self._iter_versions(path))
            current = self.get_current_version(path.name)

            bundles.append(
                BundleInfo(
                    name=path.name,
                    current_version=current,
                    versions=version_names,
                )
            )

        return bundles

    def get_bundle(self, bundle_name: str) -> BundleInfo:
        """Get info for a specific bundle."""
        bundle_dir = self._bundles_path / bundle_name
        if not bundle_dir.exists():
            raise BundleNotFoundError(f"Bundle not found: {bundle_name}")

        version_names = sorted(p.name for p in self._iter_versions(bundle_dir))
        current = self.get_current_version(bundle_name)
        return BundleInfo(name=bundle_name, current_version=current, versions=version_names)

    def list_versions(self, bundle_name: str) -> list[VersionInfo]:
        """List all versions of a bundle."""
        bundle_dir = self._bundles_path / bundle_name
        if not bundle_dir.exists():
            raise BundleError(f"Bundle not found: {bundle_name}")

        versions: list[VersionInfo] = []
        for version_dir in self._iter_versions(bundle_dir):
            config_path = version_dir / self.BUNDLE_CONFIG_FILE
            if config_path.exists():
                try:
                    bundle = self._load_config(config_path)
                    versions.append(
                        VersionInfo(
                            version=version_dir.name,
                            tested=bundle.metadata.tested,
                            description=bundle.metadata.description,
                        )
                    )
                except Exception:
                    versions.append(
                        VersionInfo(
                            version=version_dir.name,
                            tested=False,
                            description="(invalid config)",
                        )
                    )

        return sorted(versions, key=lambda v: v.version, reverse=True)

    def get_current_version(self, bundle_name: str) -> str | None:
        """Get the current version of a bundle."""
        bundle_dir = self._bundles_path / bundle_name
        current_link = bundle_dir / self.CURRENT_LINK

        if not current_link.exists():
            return None

        if current_link.is_symlink():
            target = current_link.resolve()
            return target.name

        return None

    def set_current_version(self, bundle_name: str, version: str) -> None:
        """Set the current version of a bundle."""
        bundle_dir = self._bundles_path / bundle_name
        version_dir = bundle_dir / version

        if not version_dir.exists():
            raise BundleNotFoundError(f"Version not found: {bundle_name}/{version}")

        current_link = bundle_dir / self.CURRENT_LINK

        # Remove existing symlink
        if current_link.exists() or current_link.is_symlink():
            current_link.unlink()

        # Create new symlink (relative)
        current_link.symlink_to(version)

    def resolve_bundle_path(
        self,
        bundle_name: str,
        version: str | None = None,
    ) -> Path:
        """Resolve full path to a bundle version directory."""
        bundle_dir = self._bundles_path / bundle_name
        if not bundle_dir.exists():
            raise BundleNotFoundError(f"Bundle not found: {bundle_name}")

        if version:
            version_dir = bundle_dir / version
        else:
            current_link = bundle_dir / self.CURRENT_LINK
            if current_link.exists():
                version_dir = current_link.resolve()
            elif versions := list(self._iter_versions(bundle_dir)):
                version_dir = max(versions, key=lambda p: p.name)
            else:
                raise BundleNotFoundError(f"No versions found for bundle: {bundle_name}")

        if not version_dir.exists():
            raise BundleNotFoundError(f"Version not found: {bundle_name}/{version}")

        return version_dir

    def load_bundle(self, bundle_name: str, version: str | None = None) -> BundleFiles:
        """Load a bundle by name and optional version."""
        bundle_dir = self._bundles_path / bundle_name
        if not bundle_dir.exists():
            raise BundleNotFoundError(f"Bundle not found: {bundle_name}")

        if version:
            version_dir = bundle_dir / version
            if not version_dir.exists():
                raise BundleNotFoundError(f"Version not found: {bundle_name}/{version}")
        else:
            current_link = bundle_dir / self.CURRENT_LINK
            if not current_link.exists():
                raise BundleNotFoundError(f"No current version set for bundle: {bundle_name}")
            version_dir = current_link.resolve()

        return self._load_bundle_files(version_dir)

    def _load_bundle_files(self, version_dir: Path) -> BundleFiles:
        """Load all bundle files from a version directory."""
        config_path = version_dir / self.BUNDLE_CONFIG_FILE
        if not config_path.exists():
            raise BundleValidationError(f"Missing bundle.yaml in {version_dir}")

        requirements_path = version_dir / "requirements.lock"
        if not requirements_path.exists():
            raise BundleValidationError(f"Missing requirements.lock in {version_dir}")

        workflow_path = version_dir / "workflow.json"
        if not workflow_path.exists():
            raise BundleValidationError(f"Missing workflow.json in {version_dir}")

        bundle_config = self._load_config(config_path)
        requirements_lock = requirements_path.read_text()
        workflow_json = json.loads(workflow_path.read_text())

        return BundleFiles(
            bundle_config=bundle_config,
            requirements_lock=requirements_lock,
            workflow_json=workflow_json,
        )

    def load_bundle_config_from_path(self, bundle_path: Path) -> BundleConfig:
        """Load bundle configuration from a version directory path."""
        return self._load_bundle_config_from_path(bundle_path)

    def _load_bundle_config_from_path(self, bundle_path: Path) -> BundleConfig:
        config_path = bundle_path / self.BUNDLE_CONFIG_FILE
        if not config_path.exists():
            raise BundleValidationError(f"Bundle config not found: {config_path}")
        return self._load_config(config_path)

    def resolve_bundle(self, name: str | None, version: str | None) -> tuple[str, str]:
        """Resolve bundle name and version, returning (name, version)."""
        if not name:
            raise BundleError("No bundle specified")

        bundle_dir = self._bundles_path / name
        if not bundle_dir.exists():
            raise BundleNotFoundError(f"Bundle not found: {name}")

        if version:
            version_dir = bundle_dir / version
            if not version_dir.exists():
                raise BundleNotFoundError(f"Version not found: {name}/{version}")
            return name, version

        current_link = bundle_dir / self.CURRENT_LINK
        if current_link.exists():
            return name, current_link.resolve().name

        raise BundleNotFoundError(f"No current version set for bundle: {name}")

    def delete_version(self, bundle_name: str, version: str) -> None:
        """Delete a bundle version.

        Raises:
            BundleError: If trying to delete the current version.
        """
        bundle_dir = self._bundles_path / bundle_name
        version_dir = bundle_dir / version

        if not version_dir.exists():
            raise BundleError(f"Version not found: {bundle_name}/{version}")

        # Check if this is the current version
        current = self.get_current_version(bundle_name)
        if current == version:
            raise BundleError(
                "Cannot delete current version. Set a different version as current first."
            )

        shutil.rmtree(version_dir)

    def _iter_versions(self, bundle_dir: Path) -> Iterator[Path]:
        """Iterate over version directories in a bundle."""
        for path in bundle_dir.iterdir():
            if (
                path.is_dir()
                and not path.is_symlink()
                and not path.name.startswith(".")
                and path.name != self.CURRENT_LINK
            ):
                yield path

    def _load_config(self, config_path: Path) -> BundleConfig:
        """Load and parse bundle configuration."""
        with config_path.open() as f:
            data = yaml.safe_load(f)

        return BundleConfig.model_validate(data)
