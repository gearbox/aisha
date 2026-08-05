"""Shared, Typer-free bundle reference resolution and schema loading."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

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


async def _resolve_bundle_with_manager(
    manager: BundleRegistryManager,
    reference: BundleReference,
    *,
    sync: bool,
) -> ResolvedBundle:
    """Resolve and parse a bundle through an already-constructed registry manager.

    This private adapter keeps manager-owning multi-bundle workflows on the
    same resolution and validation path as the public composition-root API.
    """
    try:
        if sync:
            await manager.sync_all()
        bundle_path = await manager.resolve(reference)
    except ValueError as exc:
        raise BundleResolutionError(str(exc)) from exc

    return ResolvedBundle(
        name=reference.name,
        path=bundle_path,
        config=_load_bundle_config(bundle_path),
    )


async def resolve_bundle(
    settings: Settings | BundleRegistryManager,
    reference: str | BundleReference,
    *,
    sync: bool,
) -> ResolvedBundle:
    """Resolve ``[registry/]name[:version]`` to its version directory and config.

    A manager plus parsed reference is accepted for existing multi-bundle
    workflows; command composition roots pass ``Settings`` and a reference
    string as the public API specifies.
    """
    if isinstance(reference, BundleReference):
        return await _resolve_bundle_with_manager(
            cast("BundleRegistryManager", settings), reference, sync=sync
        )
    return await _resolve_bundle_with_manager(
        create_registry_manager(cast("Settings", settings)),
        parse_bundle_reference(reference),
        sync=sync,
    )
