"""Typer-free orchestration shared by cache push and cache verify."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import yaml
from pydantic import ValidationError

from . import cache_service
from .config import BundleConfig

if TYPE_CHECKING:
    from pathlib import Path

    from .bundle_registry import BundleReference, BundleRegistryManager
    from .config import Settings


class CacheWorkflowError(Exception):
    """A clean, expected cache-command resolution or bundle-loading failure."""


@dataclass(frozen=True, slots=True)
class ResolvedCacheTargets:
    """A validated bundle and the model targets selected from it."""

    bundle_path: Path
    config: BundleConfig
    targets: tuple[cache_service.PushTarget, ...]


async def resolve_cache_targets(
    settings: Settings,
    manager: BundleRegistryManager,
    ref: BundleReference,
    *,
    only_filename: str | None,
    sync: bool,
) -> ResolvedCacheTargets:
    """Resolve, parse, validate, and select cache targets without CLI concerns."""
    try:
        if sync:
            await manager.sync_all()
        bundle_path = await manager.resolve(ref)
    except ValueError as exc:
        raise CacheWorkflowError(str(exc)) from exc

    bundle_yaml = bundle_path / "bundle.yaml"
    try:
        raw = yaml.safe_load(bundle_yaml.read_text())
        config = BundleConfig.model_validate(raw)
    except FileNotFoundError as exc:
        raise CacheWorkflowError(f"Bundle config not found at {bundle_path}") from exc
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        raise CacheWorkflowError(f"Invalid bundle config:\n{exc}") from exc

    if targets := tuple(cache_service.collect_targets(config, settings.models_path, only_filename)):
        return ResolvedCacheTargets(bundle_path=bundle_path, config=config, targets=targets)
    raise CacheWorkflowError("No matching model files found in bundle")


async def verify_cache_targets(
    settings: Settings,
    manager: BundleRegistryManager,
    ref: BundleReference,
    *,
    only_filename: str | None,
    sync: bool,
    deep: bool,
) -> cache_service.VerifyReport:
    """Resolve selected models then verify through the read-only cache path."""
    resolved = await resolve_cache_targets(
        settings,
        manager,
        ref,
        only_filename=only_filename,
        sync=sync,
    )
    return cache_service.verify_models(settings, list(resolved.targets), deep=deep)
