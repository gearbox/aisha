"""Tests for shared bundle reference resolution."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_content_service.bundle_registry import BundleReference, BundleRegistryManager
from ai_content_service.bundle_resolution import BundleResolutionError, resolve_bundle
from ai_content_service.config import Settings
from ai_content_service.registry_service import create_registry_manager

if TYPE_CHECKING:
    from pathlib import Path


def _settings(tmp_path: Path) -> Settings:
    bundles = tmp_path / "bundles"
    version = bundles / "demo" / "260101-01"
    version.mkdir(parents=True)
    (bundles / "demo" / "current").symlink_to("260101-01")
    (version / "bundle.yaml").write_text(
        "metadata:\n  name: demo\n  version: '260101-01'\nmodels: []\n"
    )
    return Settings(bundles_path=bundles)


@pytest.mark.parametrize(
    ("reference", "expected"),
    [
        ("demo", BundleReference(name="demo")),
        ("demo:260101-01", BundleReference(name="demo", version="260101-01")),
        ("local/demo", BundleReference(name="demo", registry="local")),
        (
            "local/demo:260101-01",
            BundleReference(name="demo", version="260101-01", registry="local"),
        ),
    ],
)
async def test_resolves_reference_forms(
    tmp_path: Path, reference: str, expected: BundleReference
) -> None:
    settings = _settings(tmp_path)

    resolved = await resolve_bundle(settings, reference, sync=False)

    assert BundleReference.parse(reference) == expected
    assert resolved.name == "demo"
    assert resolved.path.name == "260101-01"
    assert resolved.config.metadata.version == "260101-01"


async def test_sync_is_forwarded_to_registry_manager(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    manager = MagicMock(spec=BundleRegistryManager)
    manager.sync_all = AsyncMock()
    manager.resolve = AsyncMock(return_value=tmp_path / "resolved")
    (tmp_path / "resolved").mkdir()
    (tmp_path / "resolved" / "bundle.yaml").write_text(
        "metadata:\n  name: demo\n  version: '260101-01'\n"
    )

    with patch(
        "ai_content_service.bundle_resolution.create_registry_manager", return_value=manager
    ):
        await resolve_bundle(settings, "demo", sync=True)

    manager.sync_all.assert_awaited_once()
    manager.resolve.assert_awaited_once_with(BundleReference(name="demo"))


async def test_resolution_translates_unknown_bundle_and_invalid_yaml(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    with pytest.raises(BundleResolutionError, match="not found"):
        await resolve_bundle(settings, "missing", sync=False)

    bundle_yaml = tmp_path / "bundles" / "demo" / "260101-01" / "bundle.yaml"
    bundle_yaml.write_text("metadata: [broken")
    with pytest.raises(BundleResolutionError, match="Invalid bundle config"):
        await resolve_bundle(settings, "demo", sync=False)


def test_registry_manager_construction_keeps_local_resolution_available(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    assert create_registry_manager(settings).list_registries() == ["local"]
