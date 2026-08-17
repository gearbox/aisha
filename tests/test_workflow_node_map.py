"""Focused coverage for workflow maps and committed GUI/API graph pairs."""

from __future__ import annotations

import copy
import json
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from ai_content_service.bundle_contract import (
    _NON_EXECUTABLE_GUI_CLASSES,
    Severity,
    check_bundle_contract,
    check_workflow_sync,
)
from ai_content_service.config import (
    BundleConfig,
    ModelConfig,
    ModelFileConfig,
    WorkflowModelInputConfig,
    WorkflowNodeConfig,
)
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


def _raw_bundle_with_bad_custom_node() -> dict[str, object]:
    raw = _raw_bundle()
    raw["custom_nodes"] = [{"name": "foo", "git_url": "https://example.com/foo.git"}]
    return raw


def test_schema_error_does_not_truncate_report_with_or_without_workflow_map(
    tmp_path: Path,
) -> None:
    """P1-1: a schema-invalid bundle must still surface its unrelated findings.

    An identical unrelated error (missing commit_sha) must produce the same
    non-schema findings whether or not the bundle declares a workflow: map --
    only the check name distinguishing bundle.config_invalid/schema.invalid
    may differ.
    """
    with_map = _raw_bundle_with_bad_custom_node()
    without_map = _raw_bundle_with_bad_custom_node()
    del without_map["workflow"]
    del without_map["workflow_api_file"]

    bundle = tmp_path / "demo" / "260101-01"
    bundle.mkdir(parents=True)

    report_with_map = check_bundle_contract("demo", bundle, with_map)
    report_without_map = check_bundle_contract("demo", bundle, without_map)

    checks_with_map = [finding.check for finding in report_with_map.findings]
    checks_without_map = [finding.check for finding in report_without_map.findings]

    assert checks_with_map[0] == "bundle.config_invalid"
    assert checks_without_map[0] == "schema.invalid"
    assert set(checks_with_map[1:]) == set(checks_without_map[1:])
    assert set(checks_with_map[1:]) == {
        "hardware.base_image.absent",
        "index.current_symlink.missing",
        "index.entry.missing",
    }


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


# ---------------------------------------------------------------------------
# P0-1: legacy NodeIDs checks are gated on the presence of a workflow map
# ---------------------------------------------------------------------------


def _zit_like_raw_bundle(*, with_map: bool) -> dict[str, object]:
    """A bundle whose real node ids (71/69/65/9) don't match Apex's legacy 2/3/9."""
    raw: dict[str, object] = {
        "metadata": {"name": "zit", "version": "260101-01", "tested": True},
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
    if with_map:
        raw["workflow_api_file"] = "workflow.api.json"
        raw["workflow"] = {
            "nodes": {
                "latent": {"id": 65, "class": "EmptyLatentImage", "inputs": {"width": "width"}},
                "positive_prompt": {
                    "id": 69,
                    "class": "CLIPTextEncode",
                    "inputs": {"text": "text"},
                },
                "sampler": {"id": 71, "class": "KSampler", "inputs": {"steps": "steps"}},
            }
        }
    return raw


def _zit_like_gui_graph() -> dict[str, object]:
    return {
        "nodes": [
            {
                "id": 65,
                "type": "EmptyLatentImage",
                "inputs": [{"name": "width", "widget": {}}],
                "widgets_values": [1024],
            },
            {
                "id": 69,
                "type": "CLIPTextEncode",
                "inputs": [{"name": "text", "widget": {}}],
                "widgets_values": ["cat"],
            },
            {
                "id": 71,
                "type": "KSampler",
                "inputs": [{"name": "steps", "widget": {}}],
                "widgets_values": [8],
            },
            {
                "id": 9,
                "type": "SaveImage",
                "inputs": [{"name": "filename_prefix", "widget": {}}],
                "widgets_values": ["Aisha"],
            },
        ],
        "links": [],
    }


def _zit_like_api_graph() -> dict[str, object]:
    return {
        "65": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024}},
        "69": {"class_type": "CLIPTextEncode", "inputs": {"text": "cat"}},
        "71": {"class_type": "KSampler", "inputs": {"steps": 8}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Aisha"}},
    }


def _zit_like_report(tmp_path: Path, *, with_map: bool) -> tuple[str, ...]:
    bundle = tmp_path / "zit" / "260101-01"
    bundle.mkdir(parents=True)
    (bundle / "workflow.json").write_text(json.dumps(_zit_like_gui_graph()))
    if with_map:
        (bundle / "workflow.api.json").write_text(json.dumps(_zit_like_api_graph()))
    report = check_bundle_contract(
        "zit",
        bundle,
        _zit_like_raw_bundle(with_map=with_map),
        bundle_root=bundle.parent,
        index_entries=({"name": "zit", "model_type": "aisha-image"},),
    )
    return tuple(finding.check for finding in report.findings)


_LEGACY_NODE_ID_CHECKS = frozenset(
    {"workflow.missing_node_id", "workflow.node_class_mismatch", "workflow.prompt_key"}
)


def test_workflow_map_suppresses_legacy_node_id_findings(tmp_path: Path) -> None:
    checks = _zit_like_report(tmp_path, with_map=True)
    assert not _LEGACY_NODE_ID_CHECKS & set(checks)


def test_missing_workflow_map_still_runs_legacy_node_id_checks(tmp_path: Path) -> None:
    checks = _zit_like_report(tmp_path, with_map=False)
    assert "workflow.missing_node_id" in checks
    assert "workflow.node_class_mismatch" in checks


# ---------------------------------------------------------------------------
# P0-2 / P2-3: notes and other non-executable nodes are not the Export format
# ---------------------------------------------------------------------------


def test_markdown_note_with_empty_inputs_never_triggers_gui_not_save_format() -> None:
    gui = {
        "nodes": [
            {"id": 1, "type": "MarkdownNote", "inputs": [], "widgets_values": ["## docs"]},
        ],
        "links": [],
    }
    findings = check_workflow_sync(gui, {})
    assert all(finding.check != "workflow.sync.gui_not_save_format" for finding in findings)


def test_export_format_node_with_declared_inputs_still_triggers_gui_not_save_format() -> None:
    gui = {
        "nodes": [
            {
                "id": 1,
                "type": "KSampler",
                "inputs": [{"name": "steps"}],  # no widget metadata: this is Export format
                "widgets_values": [8],
            },
        ],
        "links": [],
    }
    findings = check_workflow_sync(gui, {})
    assert any(finding.check == "workflow.sync.gui_not_save_format" for finding in findings)


@pytest.mark.parametrize("node_class", sorted(_NON_EXECUTABLE_GUI_CLASSES))
def test_non_executable_classes_never_report_node_missing_in_api(node_class: str) -> None:
    gui = {
        "nodes": [{"id": "1", "type": node_class, "inputs": [], "widgets_values": []}],
        "links": [],
    }
    findings = check_workflow_sync(gui, {})
    assert all(finding.check != "workflow.sync.node_missing_in_api" for finding in findings)


# ---------------------------------------------------------------------------
# P1-2: an API link is [str, int], not any 2-element list
# ---------------------------------------------------------------------------


def test_list_valued_widget_is_a_value_not_a_link() -> None:
    gui = {
        "nodes": [
            {
                "id": 1,
                "type": "EmptyLatentImage",
                "inputs": [{"name": "size", "widget": {}}],
                "widgets_values": [[512, 512]],
            }
        ],
        "links": [],
    }
    api = {"1": {"class_type": "EmptyLatentImage", "inputs": {"size": [64, 64]}}}

    findings = check_workflow_sync(gui, api)
    checks = {finding.check for finding in findings}
    assert "workflow.sync.value_mismatch" in checks
    assert "workflow.sync.link_mismatch" not in checks


# ---------------------------------------------------------------------------
# P1-3: malformed API nodes degrade to TODO comments, never raise
# ---------------------------------------------------------------------------


def test_infer_workflow_map_degrades_on_blank_class_type_instead_of_raising() -> None:
    api = {
        "71": {
            "class_type": "KSampler",
            "inputs": {"positive": ["69", 0], "latent_image": ["65", 0], "steps": 8},
        },
        "69": {"class_type": "", "inputs": {}},
        "65": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024}},
    }

    workflow_map, comments = infer_workflow_map(api, [])

    assert workflow_map is None
    assert any("node 69" in comment for comment in comments)


def test_infer_workflow_map_degrades_on_empty_node_id_key_instead_of_raising() -> None:
    # The linked *positive* id, not latent, is used here: latent_id uses
    # `_api_link_origin(...) or _api_link_origin(...)`, where an empty-string
    # origin id is falsy and short-circuits to the other candidate instead of
    # reaching _workflow_node -- positive_id's `is None` check has no such gap.
    api = {
        "71": {
            "class_type": "KSampler",
            "inputs": {"positive": ["", 0], "latent_image": ["65", 0], "steps": 8},
        },
        "": {"class_type": "CLIPTextEncode", "inputs": {"text": "cat"}},
        "65": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024}},
    }

    workflow_map, comments = infer_workflow_map(api, [])

    assert workflow_map is None
    assert any("failed workflow map validation" in comment for comment in comments)


def test_generated_workflow_comments_are_ascii() -> None:
    _, comments = infer_workflow_map({}, [])
    assert comments
    assert all(comment.isascii() for comment in comments)


# ---------------------------------------------------------------------------
# P2-1: an optional widget absent from the API graph is a warning, not drift
# ---------------------------------------------------------------------------


def test_missing_optional_widget_in_api_is_a_warning_not_value_mismatch() -> None:
    gui = {
        "nodes": [
            {
                "id": 1,
                "type": "CLIPLoader",
                "inputs": [{"name": "device", "widget": {}}],
                "widgets_values": ["default"],
            }
        ],
        "links": [],
    }
    api = {"1": {"class_type": "CLIPLoader", "inputs": {}}}

    findings = check_workflow_sync(gui, api)
    checks = {finding.check: finding.severity for finding in findings}
    assert checks.get("workflow.sync.input_missing_in_api") is Severity.WARNING
    assert "workflow.sync.value_mismatch" not in checks


# ---------------------------------------------------------------------------
# P2-2: `unaligned` names only widget-count mismatches
# ---------------------------------------------------------------------------


def _unaligned_node_ids(findings: list) -> set[str]:  # type: ignore[type-arg]
    for finding in findings:
        if finding.check == "workflow.sync.unaligned_nodes":
            body = finding.message.removeprefix("Skipped widget-value comparison for ").rstrip(".")
            return {entry.split(" ", 1)[0] for entry in body.split("; ")}
    return set()


def test_unaligned_names_only_widget_count_mismatches_and_ignores_notes() -> None:
    gui_nodes = [
        {
            "id": "65",
            "type": "EmptyLatentImage",
            "inputs": [{"name": "width", "widget": {}}],
            "widgets_values": [1024],
        },
        {
            "id": "69",
            "type": "CLIPTextEncode",
            "inputs": [{"name": "text", "widget": {}}],
            "widgets_values": ["cat"],
        },
        {
            "id": "71",
            "type": "KSampler",
            "inputs": [{"name": "steps", "widget": {}}],
            "widgets_values": [8, "control_after_generate"],
        },
    ]
    api = {
        "65": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024}},
        "69": {"class_type": "CLIPTextEncode", "inputs": {"text": "cat"}},
        "71": {"class_type": "KSampler", "inputs": {"steps": 8}},
    }

    findings = check_workflow_sync({"nodes": gui_nodes, "links": []}, api)
    assert _unaligned_node_ids(findings) == {"71"}

    gui_with_note = {
        "nodes": [
            *gui_nodes,
            {"id": "99", "type": "MarkdownNote", "inputs": [], "widgets_values": ["## docs"]},
        ],
        "links": [],
    }
    findings_with_note = check_workflow_sync(gui_with_note, api)
    assert _unaligned_node_ids(findings_with_note) == {"71"}


# ---------------------------------------------------------------------------
# P2-6: degenerate config values are rejected
# ---------------------------------------------------------------------------


def test_model_input_rejects_blank_filename() -> None:
    with pytest.raises(ValidationError):
        WorkflowModelInputConfig.model_validate(
            {"id": "1", "class": "CheckpointLoaderSimple", "input": "ckpt_name", "filename": ""}
        )


def test_model_input_rejects_whitespace_only_filename() -> None:
    with pytest.raises(ValidationError):
        WorkflowModelInputConfig.model_validate(
            {
                "id": "1",
                "class": "CheckpointLoaderSimple",
                "input": "ckpt_name",
                "filename": "   ",
            }
        )


def test_workflow_node_rejects_blank_input_value() -> None:
    with pytest.raises(ValidationError):
        WorkflowNodeConfig.model_validate({"id": "1", "class": "KSampler", "inputs": {"steps": ""}})


def test_bundle_config_rejects_whitespace_only_workflow_file() -> None:
    raw = _raw_bundle()
    raw["workflow_file"] = "   "

    with pytest.raises(ValidationError):
        BundleConfig.model_validate(raw)


# ---------------------------------------------------------------------------
# P2-7: two groups sharing a model_type are disambiguated by a unique filename
# ---------------------------------------------------------------------------


def _raw_bundle_with_two_checkpoint_groups() -> dict[str, object]:
    raw = _raw_bundle()
    raw["models"] = [
        {
            "name": "checkpoints-a",
            "model_type": "checkpoints",
            "files": [{"name": "a", "url": "", "filename": "a.safetensors"}],
        },
        {
            "name": "checkpoints-b",
            "model_type": "checkpoints",
            "files": [{"name": "b", "url": "", "filename": "b.safetensors"}],
        },
    ]
    return raw


def test_ambiguous_model_type_disambiguated_by_unique_filename() -> None:
    raw = _raw_bundle_with_two_checkpoint_groups()
    workflow = raw["workflow"]
    assert isinstance(workflow, dict)
    workflow["model_inputs"] = [
        {
            "id": 99,
            "class": "CheckpointLoaderSimple",
            "input": "ckpt_name",
            "model_type": "checkpoints",
            "filename": "a.safetensors",
        }
    ]

    config = BundleConfig.model_validate(raw)

    assert config.workflow is not None
    assert config.workflow.model_inputs[0].filename == "a.safetensors"


def test_ambiguous_model_type_without_disambiguating_filename_is_rejected() -> None:
    raw = _raw_bundle_with_two_checkpoint_groups()
    workflow = raw["workflow"]
    assert isinstance(workflow, dict)
    workflow["model_inputs"] = [
        {
            "id": 99,
            "class": "CheckpointLoaderSimple",
            "input": "ckpt_name",
            "model_type": "checkpoints",
        }
    ]

    with pytest.raises(ValidationError):
        BundleConfig.model_validate(raw)
