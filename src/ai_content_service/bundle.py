"""Bundle management for AI Content Service."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from .config import BundleConfig

if TYPE_CHECKING:
    from collections.abc import Iterator


class BundleError(Exception):
    """Raised when bundle operations fail."""

    pass


@dataclass
class BundleInfo:
    """Summary information about a bundle."""

    name: str
    current_version: str | None
    version_count: int


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

    def __init__(self, bundles_path: Path) -> None:
        self._bundles_path = bundles_path

    def list_bundles(self) -> list[BundleInfo]:
        """List all available bundles."""
        if not self._bundles_path.exists():
            return []

        bundles: list[BundleInfo] = []
        for path in sorted(self._bundles_path.iterdir()):
            if not path.is_dir() or path.name.startswith("."):
                continue

            versions = list(self._iter_versions(path))
            current = self.get_current_version(path.name)

            bundles.append(
                BundleInfo(
                    name=path.name,
                    current_version=current,
                    version_count=len(versions),
                )
            )

        return bundles

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
            raise BundleError(f"Version not found: {bundle_name}/{version}")

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
        """Resolve full path to a bundle version.

        Args:
            bundle_name: Name of the bundle.
            version: Specific version or None for current.

        Returns:
            Path to the bundle version directory.

        Raises:
            BundleError: If bundle or version not found.
        """
        bundle_dir = self._bundles_path / bundle_name
        if not bundle_dir.exists():
            raise BundleError(f"Bundle not found: {bundle_name}")

        if version:
            version_dir = bundle_dir / version
        else:
            # Try current symlink
            current_link = bundle_dir / self.CURRENT_LINK
            if current_link.exists():
                version_dir = current_link.resolve()
            elif versions := list(self._iter_versions(bundle_dir)):
                version_dir = max(versions, key=lambda p: p.name)

            else:
                raise BundleError(f"No versions found for bundle: {bundle_name}")
        if not version_dir.exists():
            raise BundleError(f"Version not found: {bundle_name}/{version}")

        return version_dir

    def load_bundle(self, bundle_path: Path) -> BundleConfig:
        """Load bundle configuration from a version directory."""
        config_path = bundle_path / self.BUNDLE_CONFIG_FILE
        if not config_path.exists():
            raise BundleError(f"Bundle config not found: {config_path}")

        return self._load_config(config_path)

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
        with Path.open(config_path) as f:
            data = yaml.safe_load(f)

        return BundleConfig.model_validate(data)
