"""Tests for Typer-free bundle contract orchestration."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import pytest

from ai_content_service.bundle_contract import Severity
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
    reports = asyncio.run(
        validate_bundle_contracts(
            create_registry_manager(settings), bundle="demo", all_bundles=False, sync=False
        )
    )

    assert len(reports) == 1
    assert reports[0].bundle_name == "demo"


def test_yaml_read_error_becomes_schema_report(tmp_path: Path) -> None:
    settings = _settings(tmp_path, bundle_yaml="metadata: [broken")
    reports = asyncio.run(
        validate_bundle_contracts(
            create_registry_manager(settings), bundle="demo", all_bundles=False, sync=False
        )
    )

    assert [finding.check for finding in reports[0].findings] == ["schema.invalid"]


def test_all_validation_reports_clean_and_forced_bundles_independently(tmp_path: Path) -> None:
    bundles = tmp_path / "bundles"
    bundle_index: list[dict[str, str]] = []
    for name in ("forced", "clean"):
        version = bundles / name / "260101-01"
        version.mkdir(parents=True)
        (version.parent / "current").symlink_to("260101-01")
        payload: dict[str, object] = {
            "metadata": {"name": name, "version": "260101-01"},
            "hardware": {
                "gpu_whitelist": ["RTX 4090"],
                "min_disk_gb": 100,
                "min_network_upload_mbps": 100,
                "min_network_download_mbps": 100,
                "cuda_min_version": "12.1",
                "num_gpus": 1,
                "comfyui_port": 18188,
            },
            "readiness_marker": {"node_class": "KSampler"},
            "workflow_file": "workflow.json",
        }
        if name == "forced":
            payload["errors"] = ["FORCED: provider coverage unverified — start ComfyUI."]
        (version / "bundle.yaml").write_text(json.dumps(payload))
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
        bundle_index.append({"name": name, "path": f"bundles/{name}", "model_type": "aisha-image"})
    (tmp_path / "bundle-index.yaml").write_text(json.dumps({"bundles": bundle_index}))

    reports = asyncio.run(
        validate_bundle_contracts(
            create_registry_manager(Settings(bundles_path=tmp_path)),
            bundle=None,
            all_bundles=True,
            sync=False,
        )
    )

    assert [report.bundle_name for report in reports] == ["forced", "clean"]
    assert "bundle.forced_incomplete" in {finding.check for finding in reports[0].findings}
    assert not [finding for finding in reports[1].findings if finding.severity is Severity.ERROR]


def test_empty_all_validation_requires_explicit_allow_empty(tmp_path: Path) -> None:
    bundles = tmp_path / "bundles"
    bundles.mkdir()
    settings = Settings(bundles_path=bundles)
    manager = create_registry_manager(settings)

    with pytest.raises(EmptyBundleRegistryError, match="no bundles"):
        asyncio.run(validate_bundle_contracts(manager, bundle=None, all_bundles=True, sync=False))
    assert (
        asyncio.run(
            validate_bundle_contracts(
                manager, bundle=None, all_bundles=True, sync=False, allow_empty=True
            )
        )
        == ()
    )


def test_contract_validation_requires_registry_root_for_index_fields(tmp_path: Path) -> None:
    """Pointing the registry at bundles/ must not recover its parent's index."""
    repo = tmp_path / "ai-bundles"
    bundles = repo / "bundles"
    version = bundles / "demo" / "260101-01"
    version.mkdir(parents=True)
    (bundles / "demo" / "current").symlink_to("260101-01")
    (version / "bundle.yaml").write_text("metadata:\n  name: demo\n  version: '260101-01'\n")
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
    (repo / "bundle-index.yaml").write_text(
        "bundles:\n  - name: demo\n    path: bundles/demo\n    model_type: aisha-image\n"
    )

    root_reports = asyncio.run(
        validate_bundle_contracts(
            create_registry_manager(Settings(bundles_path=repo)),
            bundle="demo",
            all_bundles=False,
            sync=False,
        )
    )
    root_checks = {finding.check for finding in root_reports[0].findings}
    assert "index.entry.missing" not in root_checks
    assert "index.model_type.missing" not in root_checks

    bundles_reports = asyncio.run(
        validate_bundle_contracts(
            create_registry_manager(Settings(bundles_path=bundles)),
            bundle="demo",
            all_bundles=False,
            sync=False,
        )
    )
    bundles_checks = {finding.check for finding in bundles_reports[0].findings}
    assert "index.entry.missing" in bundles_checks


def test_validate_fetches_live_object_info_once_when_url_is_supplied(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    object_info = {
        "EmptyLatentImage": {"python_module": "nodes"},
        "TextEncodeQwenImageEditPlus": {"python_module": "nodes"},
        "KSampler": {"python_module": "nodes"},
    }
    fetch = AsyncMock(return_value=object_info)

    with patch("ai_content_service.bundle_contract_service._fetch_object_info", new=fetch):
        reports = asyncio.run(
            validate_bundle_contracts(
                create_registry_manager(settings),
                bundle="demo",
                all_bundles=False,
                sync=False,
                comfyui_url="http://localhost:18188",
            )
        )

    fetch.assert_awaited_once_with("http://localhost:18188")
    assert not {
        finding.check
        for finding in reports[0].findings
        if finding.check.startswith("workflow.class_")
    }
