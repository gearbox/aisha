"""Tests for Typer-free bundle contract orchestration."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

import pytest

from ai_content_service.bundle_contract_service import (
    EmptyBundleRegistryError,
    validate_bundle_contracts,
)
from ai_content_service.config import Settings
from ai_content_service.registry_service import create_registry_manager

if TYPE_CHECKING:
    from pathlib import Path


def _settings(tmp_path: Path, *, bundle_yaml: str | None = None) -> Settings:
    bundles = tmp_path / "bundles"
    version = bundles / "demo" / "260101-01"
    version.mkdir(parents=True)
    (bundles / "demo" / "current").symlink_to("260101-01")
    (version / "bundle.yaml").write_text(
        bundle_yaml or "metadata:\n  name: demo\n  version: '260101-01'\n"
    )
    (version / "workflow.json").write_text(
        json.dumps(
            {
                "nodes": [
                    {"id": 9, "type": "EmptyLatentImage"},
                    {"id": 3, "type": "TextEncodeQwenImageEditPlus"},
                    {"id": 2, "type": "KSampler"},
                ]
            }
        )
    )
    return Settings(bundles_path=bundles)


def test_validate_single_bundle_returns_report(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    reports = asyncio.get_event_loop().run_until_complete(
        validate_bundle_contracts(
            create_registry_manager(settings), bundle="demo", all_bundles=False, sync=False
        )
    )

    assert len(reports) == 1
    assert reports[0].bundle_name == "demo"


def test_yaml_read_error_becomes_schema_report(tmp_path: Path) -> None:
    settings = _settings(tmp_path, bundle_yaml="metadata: [broken")
    reports = asyncio.get_event_loop().run_until_complete(
        validate_bundle_contracts(
            create_registry_manager(settings), bundle="demo", all_bundles=False, sync=False
        )
    )

    assert [finding.check for finding in reports[0].findings] == ["schema.invalid"]


def test_empty_all_validation_requires_explicit_allow_empty(tmp_path: Path) -> None:
    bundles = tmp_path / "bundles"
    bundles.mkdir()
    settings = Settings(bundles_path=bundles)
    manager = create_registry_manager(settings)

    with pytest.raises(EmptyBundleRegistryError, match="no bundles"):
        asyncio.get_event_loop().run_until_complete(
            validate_bundle_contracts(manager, bundle=None, all_bundles=True, sync=False)
        )
    assert (
        asyncio.get_event_loop().run_until_complete(
            validate_bundle_contracts(
                manager, bundle=None, all_bundles=True, sync=False, allow_empty=True
            )
        )
        == ()
    )
