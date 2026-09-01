"""Shared, Typer-free bundle reference resolution and schema loading."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import yaml
from pydantic import ValidationError

from .bundle_registry import BundleReference, BundleRegistryManager
from .config import BundleConfig
from .registry_service import create_registry_manager

if TYPE_CHECKING:
    from pathlib import Path

    from .config import Settings


@dataclass(frozen=True, slots=True)
class ResolvedBundle:
    """A resolved bundle version directory and its validated configuration."""

    name: str
    path: Path
    config: BundleConfig
    model_type: str | None = None


class BundleResolutionError(Exception):
    """Raised when a bundle reference cannot be resolved or parsed."""

    def __init__(self, message: str, *, bundle_path: Path | None = None) -> None:
        super().__init__(message)
        self.bundle_path = bundle_path


def parse_bundle_reference(reference: str) -> BundleReference:
    """Parse one public bundle reference in the shared resolution boundary."""
    return BundleReference.parse(reference)


def _load_bundle_config(bundle_path: Path) -> BundleConfig:
    """Load and validate the bundle configuration at an already-resolved path."""
    bundle_yaml = bundle_path / "bundle.yaml"
    try:
        raw = yaml.safe_load(bundle_yaml.read_text())
        return BundleConfig.model_validate(raw)
    except FileNotFoundError as exc:
        raise BundleResolutionError(
            f"Bundle config not found at {bundle_path}", bundle_path=bundle_path
        ) from exc
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        raise BundleResolutionError(
            f"Invalid bundle config:\n{exc}", bundle_path=bundle_path
        ) from exc


async def resolve_bundle_with_manager(
    manager: BundleRegistryManager,
    reference: BundleReference,
    *,
    sync: bool,
) -> ResolvedBundle:
    """Resolve an already-parsed reference through a caller-owned manager.

    Multi-bundle workflows build one manager and reuse it, so this entrypoint
    does not construct one. Single-bundle command entrypoints should use
    ``resolve_bundle``.
    """
    try:
        if sync:
            await manager.sync_all()
        bundle_path = await manager.resolve(reference)
    except ValueError as exc:
        raise BundleResolutionError(str(exc)) from exc

    index = await manager.get_index(reference.registry)
    entry = index.find(reference.name)
    return ResolvedBundle(
        name=reference.name,
        path=bundle_path,
        config=_load_bundle_config(bundle_path),
        model_type=entry.model_type if entry is not None else None,
    )


async def resolve_bundle(
    settings: Settings,
    reference: str,
    *,
    sync: bool,
) -> ResolvedBundle:
    """Resolve ``[registry/]name[:version]`` for a command composition root.

    Builds a registry manager from *settings*. Callers that already own a
    manager across several bundles should use ``resolve_bundle_with_manager``.
    """
    return await resolve_bundle_with_manager(
        create_registry_manager(settings),
        parse_bundle_reference(reference),
        sync=sync,
    )
