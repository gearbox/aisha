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
    _contract_index_entries,
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


def test_duplicate_bundle_key_is_an_error_and_does_not_suppress_other_findings(
    tmp_path: Path,
) -> None:
    settings = _settings(
        tmp_path,
        bundle_yaml=(
            "metadata:\n"
            "  name: demo\n"
            "  version: '260101-01'\n"
            "hardware:\n"
            "  cuda_min_version: '12.1'\n"
            "  cuda_min_version: '13.0'\n"
        ),
    )
    reports = asyncio.run(
        validate_bundle_contracts(
            create_registry_manager(settings), bundle="demo", all_bundles=False, sync=False
        )
    )

    duplicate = next(
        finding for finding in reports[0].findings if finding.check == "bundle.duplicate_key"
    )
    assert duplicate.severity is Severity.ERROR
    assert "cuda_min_version" in duplicate.message
    assert duplicate.location == "bundle.yaml:6"
    assert "hardware.base_image.absent" in {finding.check for finding in reports[0].findings}


def test_duplicate_keys_at_each_mapping_depth_are_all_reported(tmp_path: Path) -> None:
    settings = _settings(
        tmp_path,
        bundle_yaml=(
            "metadata:\n"
            "  name: first\n"
            "  name: demo\n"
            "  version: '260101-01'\n"
            "hardware:\n"
            "  cuda_min_version: '12.1'\n"
            "  cuda_min_version: '13.0'\n"
            "models:\n"
            "  - filename: first.safetensors\n"
            "    filename: second.safetensors\n"
        ),
    )
    reports = asyncio.run(
        validate_bundle_contracts(
            create_registry_manager(settings), bundle="demo", all_bundles=False, sync=False
        )
    )

    duplicates = [
        finding for finding in reports[0].findings if finding.check == "bundle.duplicate_key"
    ]
    assert [(finding.location, finding.message) for finding in duplicates] == [
        ("bundle.yaml:3", "Duplicate key 'name'; the later value wins when parsed."),
        ("bundle.yaml:7", "Duplicate key 'cuda_min_version'; the later value wins when parsed."),
        ("bundle.yaml:10", "Duplicate key 'filename'; the later value wins when parsed."),
    ]


def test_clean_bundle_yaml_has_no_duplicate_key_finding(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    reports = asyncio.run(
        validate_bundle_contracts(
            create_registry_manager(settings), bundle="demo", all_bundles=False, sync=False
        )
    )

    assert all(finding.check != "bundle.duplicate_key" for finding in reports[0].findings)


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
    assert "workflow.map.absent" in {finding.check for finding in reports[1].findings}


def test_all_validation_keeps_clean_bundle_report_when_another_has_duplicate_key(
    tmp_path: Path,
) -> None:
    bundles = tmp_path / "bundles"
    for name, bundle_yaml in (
        (
            "duplicate",
            (
                "metadata:\n  name: duplicate\n  version: '260101-01'\nmetadata:\n"
                "  name: duplicate\n  version: '260101-01'\n"
            ),
        ),
        ("clean", "metadata:\n  name: clean\n  version: '260101-01'\n"),
    ):
        version = bundles / name / "260101-01"
        version.mkdir(parents=True)
        (version.parent / "current").symlink_to("260101-01")
        (version / "bundle.yaml").write_text(bundle_yaml)
        (version / "workflow.json").write_text(json.dumps({"nodes": []}))
    (tmp_path / "bundle-index.yaml").write_text(
        json.dumps(
            {
                "bundles": [
                    {"name": "duplicate", "path": "bundles/duplicate", "model_type": "aisha-image"},
                    {"name": "clean", "path": "bundles/clean", "model_type": "aisha-image"},
                ]
            }
        )
    )

    reports = asyncio.run(
        validate_bundle_contracts(
            create_registry_manager(Settings(bundles_path=tmp_path)),
            bundle=None,
            all_bundles=True,
            sync=False,
        )
    )

    reports_by_name = {report.bundle_name: report for report in reports}
    assert any(
        finding.check == "bundle.duplicate_key" for finding in reports_by_name["duplicate"].findings
    )
    assert all(
        finding.check != "bundle.duplicate_key" for finding in reports_by_name["clean"].findings
    )


def test_duplicate_bundle_index_key_is_reported_once_without_hiding_entries(
    tmp_path: Path,
) -> None:
    bundles = tmp_path / "bundles"
    for name in ("first", "second", "third"):
        version = bundles / name / "260101-01"
        version.mkdir(parents=True)
        (version.parent / "current").symlink_to("260101-01")
        (version / "bundle.yaml").write_text(f"metadata:\n  name: {name}\n  version: '260101-01'\n")
        (version / "workflow.json").write_text(json.dumps({"nodes": []}))
    (tmp_path / "bundle-index.yaml").write_text(
        "bundles:\n"
        "  - name: first\n"
        "    path: bundles/first\n"
        "    path: bundles/first\n"
        "    model_type: aisha-image\n"
        "  - name: second\n"
        "    path: bundles/second\n"
        "    model_type: aisha-image\n"
        "  - name: third\n"
        "    path: bundles/third\n"
        "    model_type: aisha-image\n"
    )

    reports = asyncio.run(
        validate_bundle_contracts(
            create_registry_manager(Settings(bundles_path=tmp_path)),
            bundle=None,
            all_bundles=True,
            sync=False,
        )
    )

    duplicates = [
        finding
        for report in reports
        for finding in report.findings
        if finding.check == "bundle.duplicate_key"
    ]

    assert len(duplicates) == 1
    assert duplicates[0].location == "bundle-index.yaml:4"
    assert "path" in duplicates[0].message
    assert all(
        finding.check != "index.entry.missing" for report in reports for finding in report.findings
    )


def test_malformed_bundle_index_keeps_the_previous_empty_entries_behavior(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    registry = create_registry_manager(settings).default
    assert registry is not None
    (settings.bundles_path / "bundle-index.yaml").write_text("bundles: [\n")

    assert _contract_index_entries(registry) == ((), ())


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
