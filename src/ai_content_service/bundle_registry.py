"""Bundle registry for external bundle repositories.

This module provides support for loading bundles from external Git repositories,
enabling separation of concerns between the deployment tool (aisha) and
bundle configurations (ai-bundles).
"""

from __future__ import annotations

import asyncio
from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import yaml

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class BundleReference:
    """Reference to a bundle in a registry."""

    name: str
    version: str | None = None
    registry: str | None = None  # Registry name, None = default

    @classmethod
    def parse(cls, spec: str) -> BundleReference:
        """Parse bundle specification string.

        Formats:
            - "bundle_name" -> BundleReference(name="bundle_name")
            - "bundle_name:version" -> BundleReference(name="bundle_name", version="version")
            - "registry/bundle_name" -> BundleReference(name="bundle_name", registry="registry")
            - "registry/bundle_name:version" -> all three fields
        """
        registry = None
        version = None

        # Check for registry prefix
        if "/" in spec:
            registry, spec = spec.split("/", 1)

        # Check for version suffix
        if ":" in spec:
            spec, version = spec.rsplit(":", 1)

        return cls(name=spec, version=version, registry=registry)


@dataclass
class BundleIndexEntry:
    """Entry in a bundle index."""

    name: str
    path: str
    description: str = ""
    tags: list[str] | None = None
    default_version: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> BundleIndexEntry:
        return cls(
            name=data["name"],
            path=data["path"],
            description=data.get("description", ""),
            tags=data.get("tags"),
            default_version=data.get("default_version"),
        )


@dataclass
class BundleIndex:
    """Index of available bundles in a registry."""

    bundles: list[BundleIndexEntry]
    version: str = "1"

    @classmethod
    def from_yaml(cls, content: str) -> BundleIndex:
        """Parse bundle index from YAML content."""
        data = yaml.safe_load(content)
        return cls(
            version=data.get("version", "1"),
            bundles=[BundleIndexEntry.from_dict(b) for b in data.get("bundles", [])],
        )

    def find(self, name: str) -> BundleIndexEntry | None:
        """Find a bundle by name."""
        return next((entry for entry in self.bundles if entry.name == name), None)


@runtime_checkable
class BundleRegistry(Protocol):
    """Protocol for bundle registries."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Registry name for identification."""
        ...

    @abstractmethod
    async def sync(self) -> None:
        """Sync/update the registry (e.g., git pull)."""
        ...

    @abstractmethod
    async def get_index(self) -> BundleIndex:
        """Get the bundle index."""
        ...

    @abstractmethod
    async def resolve_bundle_path(self, bundle_name: str, version: str | None = None) -> Path:
        """Resolve full path to a bundle version."""
        ...

    @abstractmethod
    async def list_versions(self, bundle_name: str) -> list[str]:
        """List available versions for a bundle."""
        ...


class LocalBundleRegistry:
    """Bundle registry backed by a local directory."""

    def __init__(self, path: Path, name: str = "local") -> None:
        self._path = path
        self._name = name
        self._index: BundleIndex | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def path(self) -> Path:
        return self._path

    async def sync(self) -> None:
        """No-op for local registry."""
        self._index = None  # Invalidate cached index

    async def get_index(self) -> BundleIndex:
        """Get or build the bundle index."""
        if self._index is not None:
            return self._index

        index_path = self._path / "bundle-index.yaml"
        if index_path.exists():
            self._index = BundleIndex.from_yaml(index_path.read_text())
        else:
            # Auto-discover bundles from directory structure
            self._index = await self._discover_bundles()

        return self._index

    async def _discover_bundles(self) -> BundleIndex:
        """Auto-discover bundles from directory structure."""
        bundles = []
        bundles_dir = self._path / "bundles" if (self._path / "bundles").exists() else self._path

        for bundle_dir in bundles_dir.iterdir():
            if bundle_dir.is_dir() and not bundle_dir.name.startswith("."):
                # Check for bundle.yaml in any version subdirectory
                has_bundle = any(
                    (bundle_dir / v / "bundle.yaml").exists()
                    for v in bundle_dir.iterdir()
                    if v.is_dir()
                )
                if has_bundle or (bundle_dir / "bundle.yaml").exists():
                    bundles.append(
                        BundleIndexEntry(
                            name=bundle_dir.name,
                            path=str(bundle_dir.relative_to(self._path)),
                        )
                    )

        return BundleIndex(bundles=bundles)

    async def resolve_bundle_path(self, bundle_name: str, version: str | None = None) -> Path:
        """Resolve path to a specific bundle version."""
        index = await self.get_index()
        entry = index.find(bundle_name)

        if entry is None:
            raise ValueError(f"Bundle '{bundle_name}' not found in registry '{self._name}'")

        bundle_dir = self._path / entry.path

        # Handle "current" symlink
        current_link = bundle_dir / "current"
        if version is None and current_link.exists():
            return current_link.resolve()

        if target_version := version or entry.default_version:
            version_path = bundle_dir / target_version
            if version_path.exists():
                return version_path

        # Fall back to bundle_dir if no version structure
        if (bundle_dir / "bundle.yaml").exists():
            return bundle_dir

        raise ValueError(f"Version '{version}' not found for bundle '{bundle_name}'")

    async def list_versions(self, bundle_name: str) -> list[str]:
        """List available versions for a bundle."""
        index = await self.get_index()
        entry = index.find(bundle_name)

        if entry is None:
            raise ValueError(f"Bundle '{bundle_name}' not found")

        bundle_dir = self._path / entry.path
        versions = []

        versions.extend(
            item.name
            for item in bundle_dir.iterdir()
            if item.is_dir()
            and not item.name.startswith(".")
            and item.name != "current"
            and (item / "bundle.yaml").exists()
        )
        return sorted(versions, reverse=True)


class GitBundleRegistry:
    """Bundle registry backed by a Git repository."""

    def __init__(
        self,
        repo_url: str,
        local_path: Path,
        name: str = "git",
        branch: str = "main",
        auth_token: str | None = None,
        ssh_key_path: Path | None = None,
    ) -> None:
        self._repo_url = repo_url
        self._local_path = local_path
        self._name = name
        self._branch = branch
        self._auth_token = auth_token
        self._ssh_key_path = ssh_key_path
        self._local_registry: LocalBundleRegistry | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def path(self) -> Path:
        return self._local_path

    def _get_authenticated_url(self) -> str:
        """Get URL with authentication if using HTTPS + token."""
        if self._auth_token and self._repo_url.startswith("https://"):
            # Insert token into HTTPS URL
            # https://github.com/... -> https://TOKEN@github.com/...
            return self._repo_url.replace("https://", f"https://{self._auth_token}@")
        return self._repo_url

    def _get_git_ssh_command(self) -> str | None:
        """Get GIT_SSH_COMMAND for SSH key authentication."""
        if self._ssh_key_path:
            return f"ssh -i {self._ssh_key_path} -o StrictHostKeyChecking=accept-new"
        return None

    async def sync(self) -> None:
        """Clone or pull the repository."""
        env = {}
        ssh_command = self._get_git_ssh_command()
        if ssh_command:
            env["GIT_SSH_COMMAND"] = ssh_command

        if self._local_path.exists():
            # Pull latest changes
            cmd = ["git", "-C", str(self._local_path), "pull", "--ff-only"]
            process = await asyncio.create_subprocess_exec(
                *cmd,
                env={**dict(__import__("os").environ), **env},
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await process.communicate()
            if process.returncode != 0:
                # Try fetch + reset for diverged branches
                await asyncio.create_subprocess_exec(
                    "git",
                    "-C",
                    str(self._local_path),
                    "fetch",
                    "origin",
                    env={**dict(__import__("os").environ), **env},
                )
                await asyncio.create_subprocess_exec(
                    "git",
                    "-C",
                    str(self._local_path),
                    "reset",
                    "--hard",
                    f"origin/{self._branch}",
                )
        else:
            # Clone repository
            self._local_path.parent.mkdir(parents=True, exist_ok=True)
            url = self._get_authenticated_url()
            cmd = [
                "git",
                "clone",
                "--branch",
                self._branch,
                "--depth",
                "1",
                url,
                str(self._local_path),
            ]
            process = await asyncio.create_subprocess_exec(
                *cmd,
                env={**dict(__import__("os").environ), **env},
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await process.communicate()
            if process.returncode != 0:
                raise RuntimeError(f"Failed to clone repository: {stderr.decode()}")

        # Invalidate local registry cache
        self._local_registry = None

    def _get_local_registry(self) -> LocalBundleRegistry:
        """Get local registry wrapper for the cloned repo."""
        if self._local_registry is None:
            bundles_path = self._local_path / "bundles"
            if bundles_path.exists():
                self._local_registry = LocalBundleRegistry(bundles_path, self._name)
            else:
                self._local_registry = LocalBundleRegistry(self._local_path, self._name)
        return self._local_registry

    async def get_index(self) -> BundleIndex:
        """Get the bundle index."""
        return await self._get_local_registry().get_index()

    async def resolve_bundle_path(self, bundle_name: str, version: str | None = None) -> Path:
        """Resolve path to a specific bundle version."""
        return await self._get_local_registry().resolve_bundle_path(bundle_name, version)

    async def list_versions(self, bundle_name: str) -> list[str]:
        """List available versions for a bundle."""
        return await self._get_local_registry().list_versions(bundle_name)


class BundleRegistryManager:
    """Manages multiple bundle registries."""

    def __init__(self, default_registry: BundleRegistry | None = None) -> None:
        self._registries: dict[str, BundleRegistry] = {}
        self._default_registry: BundleRegistry | None = default_registry
        if default_registry:
            self._registries[default_registry.name] = default_registry

    def register(self, registry: BundleRegistry, *, default: bool = False) -> None:
        """Register a bundle registry."""
        self._registries[registry.name] = registry
        if default or self._default_registry is None:
            self._default_registry = registry

    def get(self, name: str) -> BundleRegistry | None:
        """Get a registry by name."""
        return self._registries.get(name)

    @property
    def default(self) -> BundleRegistry | None:
        """Get the default registry."""
        return self._default_registry

    async def sync_all(self) -> None:
        """Sync all registries."""
        await asyncio.gather(*[r.sync() for r in self._registries.values()])

    async def resolve(self, ref: BundleReference) -> Path:
        """Resolve a bundle reference to a path."""
        registry = self._registries.get(ref.registry) if ref.registry else self._default_registry
        if registry is None:
            raise ValueError(f"Registry '{ref.registry}' not found")

        return await registry.resolve_bundle_path(ref.name, ref.version)

    def list_registries(self) -> list[str]:
        """List registered registry names."""
        return list(self._registries.keys())
