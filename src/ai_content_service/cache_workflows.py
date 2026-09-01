"""Typer-free orchestration shared by cache push and cache verify."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

from . import cache_service
from .bundle_resolution import BundleResolutionError, resolve_bundle_with_manager

if TYPE_CHECKING:
    from pathlib import Path

    from .bundle_registry import BundleReference, BundleRegistryManager
    from .config import BundleConfig, Settings


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
        resolved = await resolve_bundle_with_manager(manager, ref, sync=sync)
    except BundleResolutionError as exc:
        raise CacheWorkflowError(str(exc)) from exc

    if targets := tuple(
        cache_service.collect_targets(resolved.config, settings.models_path, only_filename)
    ):
        return ResolvedCacheTargets(
            bundle_path=resolved.path, config=resolved.config, targets=targets
        )
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
    return await asyncio.to_thread(
        cache_service.verify_models,
        settings,
        list(resolved.targets),
        deep=deep,
    )
