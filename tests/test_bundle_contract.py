"""Tests for the offline Apex bundle-contract validator."""

from __future__ import annotations

import copy
import json
from typing import TYPE_CHECKING

from ai_content_service.bundle_contract import ContractReport, Severity, check_bundle_contract

if TYPE_CHECKING:
    from pathlib import Path


def _raw_bundle() -> dict[str, object]:
    return {
        "metadata": {"name": "demo", "version": "260101-01", "tested": True},
        "hardware": {
            "gpu_whitelist": ["RTX_4090"],
            "min_disk_gb": 100,
            "min_network_upload_mbps": 100,
            "min_network_download_mbps": 100,
            "cuda_min_version": "12.1",
            "num_gpus": 1,
            "comfyui_port": 18188,
        },
        "readiness_marker": {"node_class": "KSampler"},
        "models": [
            {
                "name": "lora",
                "model_type": "loras",
                "files": [
                    {
                        "name": "model.safetensors",
                        "url": "https://example.com/model.safetensors",
                        "filename": "model.safetensors",
                        "sha256": "a" * 64,
                        "size_bytes": 1,
                    }
                ],
            }
        ],
    }


def _workflow(path: Path) -> None:
    (path / "workflow.json").write_text(
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


def _report(tmp_path: Path, raw: object) -> ContractReport:
    root = tmp_path / "demo"
    version = root / "260101-01"
    version.mkdir(parents=True)
    root.joinpath("current").symlink_to("260101-01")
    _workflow(version)
    return check_bundle_contract(
        "demo",
        version,
        raw,
        bundle_root=root,
        index_entries=({"name": "demo", "path": "bundles/demo", "model_type": "aisha-image"},),
    )


def test_valid_bundle_has_no_findings(tmp_path: Path) -> None:
    report = _report(tmp_path, _raw_bundle())
    assert report.ok is True
    assert report.findings == ()


def test_hardware_bool_is_not_an_integer(tmp_path: Path) -> None:
    raw = _raw_bundle()
    hardware = raw["hardware"]
    assert isinstance(hardware, dict)
    hardware["comfyui_port"] = True

    report = _report(tmp_path, raw)

    assert any(finding.check == "hardware.comfyui_port.not_int" for finding in report.findings)


def test_schema_error_becomes_one_error_finding(tmp_path: Path) -> None:
    raw = _raw_bundle()
    raw["unexpected"] = "field"

    report = _report(tmp_path, raw)

    assert report.ok is False
    assert len(report.findings) == 1
    assert report.findings[0].severity is Severity.ERROR
    assert report.findings[0].check == "schema.invalid"


def test_warnings_do_not_fail_validation(tmp_path: Path) -> None:
    raw = copy.deepcopy(_raw_bundle())
    raw.pop("readiness_marker")
    metadata = raw["metadata"]
    assert isinstance(metadata, dict)
    metadata["tested"] = False

    report = _report(tmp_path, raw)

    assert report.ok is True
    assert {finding.check for finding in report.findings} == {
        "metadata.tested_false",
        "readiness_marker.absent",
    }
