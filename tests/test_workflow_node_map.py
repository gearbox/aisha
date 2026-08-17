"""Focused coverage for workflow maps and committed GUI/API graph pairs."""

from __future__ import annotations

import copy
import json
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from ai_content_service.bundle_contract import Severity, check_bundle_contract, check_workflow_sync
from ai_content_service.config import BundleConfig, ModelConfig, ModelFileConfig
from ai_content_service.snapshot import infer_workflow_map

if TYPE_CHECKING:
    from pathlib import Path


def _raw_bundle() -> dict[str, object]:
    return {
        "metadata": {"name": "demo", "version": "260101-01", "tested": True},
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
        "workflow_api_file": "workflow.api.json",
        "workflow": {
            "nodes": {
                "latent": {"id": 9, "class": "EmptyLatentImage", "inputs": {"width": "width"}},
                "positive_prompt": {
                    "id": 3,
                    "class": "TextEncodeQwenImageEditPlus",
                    "inputs": {"text": "prompt"},
                },
                "sampler": {"id": 2, "class": "KSampler", "inputs": {"steps": "steps"}},
            }
        },
    }


def _gui_graph() -> dict[str, object]:
    return {
        "nodes": [
            {"id": 9, "type": "EmptyLatentImage", "inputs": [], "widgets_values": []},
            {
                "id": 3,
                "type": "TextEncodeQwenImageEditPlus",
                "inputs": [{"name": "prompt", "widget": {}}],
                "widgets_values": ["hello"],
            },
            {
                "id": 2,
                "type": "KSampler",
                "inputs": [{"name": "steps", "widget": {}}],
                "widgets_values": [8],
            },
        ],
        "links": [],
    }


def _api_graph() -> dict[str, object]:
    return {
        "9": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024}},
        "3": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"prompt": "hello"}},
        "2": {"class_type": "KSampler", "inputs": {"steps": 8}},
    }


def _report(tmp_path: Path, raw: dict[str, object]) -> tuple[str, ...]:
    bundle = tmp_path / "demo" / "260101-01"
    bundle.mkdir(parents=True)
    (bundle / "workflow.json").write_text(json.dumps(_gui_graph()))
    (bundle / "workflow.api.json").write_text(json.dumps(_api_graph()))
    report = check_bundle_contract(
        "demo",
        bundle,
        raw,
        bundle_root=bundle.parent,
        index_entries=({"name": "demo", "model_type": "aisha-image"},),
    )
    return tuple(finding.check for finding in report.findings)


def test_workflow_map_normalizes_ids_and_serializes_class_alias() -> None:
    config = BundleConfig.model_validate(_raw_bundle())

    assert config.workflow is not None
    assert (
        config.workflow.nodes[
            next(role for role in config.workflow.nodes if role.value == "latent")
        ].id
        == "9"
    )
    node = config.model_dump(mode="json", by_alias=True)["workflow"]["nodes"]["latent"]
    assert node["class"] == "EmptyLatentImage"
    assert "class_" not in node


@pytest.mark.parametrize(
    "workflow",
    [
        {"nodes": {"latent": {"id": 9, "class": "EmptyLatentImage"}}},
        {
            "nodes": {
                "latent": {"id": 9, "class": "EmptyLatentImage", "inputs": {"steps": "steps"}},
                "positive_prompt": {"id": 3, "class": "TextEncodeQwenImageEditPlus"},
                "sampler": {"id": 2, "class": "KSampler"},
            }
        },
    ],
)
def test_workflow_map_rejects_missing_required_role_and_wrong_vocabulary(
    workflow: dict[str, object],
) -> None:
    raw = _raw_bundle()
    raw["workflow"] = workflow

    with pytest.raises(ValidationError):
        BundleConfig.model_validate(raw)


def test_workflow_map_allows_no_negative_prompt_and_empty_inputs() -> None:
    raw = _raw_bundle()
    workflow = raw["workflow"]
    assert isinstance(workflow, dict)
    nodes = workflow["nodes"]
    assert isinstance(nodes, dict)
    latent = nodes["latent"]
    assert isinstance(latent, dict)
    latent["inputs"] = {}

    config = BundleConfig.model_validate(raw)

    assert config.workflow is not None
    assert all(role.value != "negative_prompt" for role in config.workflow.nodes)


def test_contract_catches_prompt_alias_and_link_valued_input_offline(tmp_path: Path) -> None:
    raw = _raw_bundle()
    workflow = raw["workflow"]
    assert isinstance(workflow, dict)
    nodes = workflow["nodes"]
    assert isinstance(nodes, dict)
    positive = nodes["positive_prompt"]
    sampler = nodes["sampler"]
    assert isinstance(positive, dict)
    assert isinstance(sampler, dict)
    positive["inputs"] = {"text": "text"}
    sampler["inputs"] = {"steps": "steps", "seed": "seed"}
    api = _api_graph()
    api["3"] = {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"prompt": "hello"}}
    api["2"] = {"class_type": "KSampler", "inputs": {"steps": 8, "seed": ["50", 0]}}

    bundle = tmp_path / "demo" / "260101-01"
    bundle.mkdir(parents=True)
    (bundle / "workflow.json").write_text(json.dumps(_gui_graph()))
    (bundle / "workflow.api.json").write_text(json.dumps(api))
    report = check_bundle_contract(
        "demo",
        bundle,
        raw,
        bundle_root=bundle.parent,
        index_entries=({"name": "demo", "model_type": "aisha-image"},),
    )

    checks = {finding.check for finding in report.findings}
    assert {"workflow.map.input_unknown", "workflow.map.input_is_link"} <= checks


def test_contract_reports_invalid_workflow_configuration_once(tmp_path: Path) -> None:
    raw = _raw_bundle()
    raw.pop("workflow_api_file")

    bundle = tmp_path / "demo" / "260101-01"
    bundle.mkdir(parents=True)
    report = check_bundle_contract("demo", bundle, raw)

    assert [finding.check for finding in report.findings] == ["bundle.config_invalid"]


def test_gui_api_sync_checks_aligned_values_but_skips_unaligned_nodes() -> None:
    gui = _gui_graph()
    api = _api_graph()
    assert not [
        finding for finding in check_workflow_sync(gui, api) if finding.severity is Severity.ERROR
    ]

    changed = copy.deepcopy(api)
    changed["2"] = {"class_type": "KSampler", "inputs": {"steps": 4}}
    assert any(
        finding.check == "workflow.sync.value_mismatch"
        for finding in check_workflow_sync(gui, changed)
    )

    gui_nodes = gui["nodes"]
    assert isinstance(gui_nodes, list)
    sampler = next(node for node in gui_nodes if isinstance(node, dict) and node.get("id") == 2)
    assert isinstance(sampler, dict)
    sampler["widgets_values"] = [8, "control_after_generate"]
    findings = check_workflow_sync(gui, changed)
    assert all(finding.check != "workflow.sync.value_mismatch" for finding in findings)
    assert any(finding.check == "workflow.sync.unaligned_nodes" for finding in findings)


def test_inference_omits_non_prompt_negative_and_derives_parameter_aliases() -> None:
    api = {
        "65": {"class_type": "EmptySD3LatentImage", "inputs": {"width": 1024, "height": 1024}},
        "69": {"class_type": "CLIPTextEncode", "inputs": {"text": "cat"}},
        "70": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["69", 0]}},
        "71": {
            "class_type": "KSampler",
            "inputs": {
                "positive": ["69", 0],
                "negative": ["70", 0],
                "latent_image": ["65", 0],
                "seed": 1,
                "steps": 8,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
            },
        },
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Aisha"}},
    }
    model = ModelConfig(
        name="unet",
        model_type="diffusion_models",
        files=[ModelFileConfig(name="unet", url="", filename="unet.safetensors")],
    )

    workflow_map, comments = infer_workflow_map(api, [model])

    assert workflow_map is not None
    assert all(role.value != "negative_prompt" for role in workflow_map.nodes)
    sampler = workflow_map.nodes[
        next(role for role in workflow_map.nodes if role.value == "sampler")
    ]
    assert sampler.inputs == {
        "seed": "seed",
        "steps": "steps",
        "cfg": "cfg",
        "sampler": "sampler_name",
        "scheduler": "scheduler",
        "denoise": "denoise",
    }
    assert any("ConditioningZeroOut" in comment for comment in comments)
