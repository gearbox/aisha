"""Coverage for bundle_contract's workflow map and GUI/API sync checks."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from ai_content_service.bundle_contract import (
    _NON_EXECUTABLE_GUI_CLASSES,
    Finding,
    Severity,
    _check_workflow_map,
    check_bundle_contract,
    check_workflow_sync,
)
from ai_content_service.config import BundleConfig
from tests.workflow_map_helpers import _api_graph, _api_inputs, _gui_graph, _raw_bundle


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


@pytest.mark.parametrize(
    ("mode", "state"),
    ((4, "muted"), (2, "bypassed")),
)
def test_api_node_disabled_in_gui_is_an_error(mode: int, state: str) -> None:
    gui = _gui_graph()
    gui_nodes = gui["nodes"]
    assert isinstance(gui_nodes, list)
    sampler = next(node for node in gui_nodes if isinstance(node, dict) and node.get("id") == 2)
    assert isinstance(sampler, dict)
    sampler["mode"] = mode

    findings = check_workflow_sync(gui, _api_graph())

    finding = next(
        finding for finding in findings if finding.check == "workflow.sync.disabled_node_in_api"
    )
    assert finding.severity is Severity.ERROR
    assert state in finding.message


def test_disabled_gui_node_absent_from_api_is_not_reported() -> None:
    gui = _gui_graph()
    gui_nodes = gui["nodes"]
    assert isinstance(gui_nodes, list)
    sampler = next(node for node in gui_nodes if isinstance(node, dict) and node.get("id") == 2)
    assert isinstance(sampler, dict)
    sampler["mode"] = 4
    api = _api_graph()
    del api["2"]

    findings = check_workflow_sync(gui, api)

    assert all(finding.check != "workflow.sync.disabled_node_in_api" for finding in findings)


@pytest.mark.parametrize(
    ("api_mutation", "expected_count"),
    (
        (lambda api: _api_inputs(api, "2").update(seed=["7", 0]), 1),
        (lambda api: _api_inputs(api, "2").update(seed=["2", 0]), 1),
    ),
)
def test_api_dangling_and_self_links_are_errors(api_mutation: object, expected_count: int) -> None:
    api = _api_graph()
    assert callable(api_mutation)
    api_mutation(api)  # type: ignore[operator]

    config = BundleConfig.model_validate(_raw_bundle())
    findings = _check_workflow_map(config, api)

    dangling = [finding for finding in findings if finding.check == "workflow.api.dangling_link"]
    assert len(dangling) == expected_count
    assert all(finding.severity is Severity.ERROR for finding in dangling)


def test_valid_api_graph_has_no_dangling_link_finding() -> None:
    config = BundleConfig.model_validate(_raw_bundle())

    findings = _check_workflow_map(config, _api_graph())

    assert all(finding.check != "workflow.api.dangling_link" for finding in findings)


def test_mapless_bundle_still_checks_api_link_origins(tmp_path: Path) -> None:
    raw = _raw_bundle()
    raw.pop("workflow")
    raw["workflow_file"] = None
    api = _api_graph()
    _api_inputs(api, "2")["seed"] = ["missing", 0]
    bundle = tmp_path / "demo" / "260101-01"
    bundle.mkdir(parents=True)
    (bundle / "workflow.api.json").write_text(json.dumps(api))

    report = check_bundle_contract("demo", bundle, raw)

    assert any(
        finding.check == "workflow.api.dangling_link" and finding.severity is Severity.ERROR
        for finding in report.findings
    )


def _workflow_map_image_findings(image_value: object, *, declared_id: int = 4) -> list[Finding]:
    raw = _raw_bundle()
    workflow = raw["workflow"]
    assert isinstance(workflow, dict)
    workflow["image_inputs"] = [{"id": declared_id, "class": "LoadImage", "target_input": "image1"}]
    config = BundleConfig.model_validate(raw)
    api = _api_graph()
    api["3"] = {
        "class_type": "TextEncodeQwenImageEditPlus",
        "inputs": {"prompt": "hello", "image1": image_value},
    }
    api[str(declared_id)] = {"class_type": "LoadImage", "inputs": {"image": "x.png"}}
    return _check_workflow_map(config, api)


def test_image_target_must_be_a_link() -> None:
    findings = _workflow_map_image_findings("uploaded.png")

    assert any(
        finding.check == "workflow.map.image_target_not_linked"
        and finding.severity is Severity.ERROR
        for finding in findings
    )


def test_image_target_origin_must_match_declared_loader() -> None:
    findings = _workflow_map_image_findings(["5", 0])

    finding = next(
        finding for finding in findings if finding.check == "workflow.map.image_target_wrong_origin"
    )
    assert finding.severity is Severity.ERROR
    assert "'4'" in finding.message
    assert "'5'" in finding.message


def test_correctly_wired_image_target_passes_and_mask_slot_is_informational() -> None:
    findings = _workflow_map_image_findings(["4", 1])

    assert not [finding for finding in findings if finding.severity is Severity.ERROR]
    assert any(
        finding.check == "workflow.map.image_target_slot" and finding.severity is Severity.INFO
        for finding in findings
    )


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


def test_absent_widget_metadata_is_one_graph_level_warning() -> None:
    gui = {
        "nodes": [
            {"id": 1, "type": "MarkdownNote", "inputs": [], "widgets_values": ["## docs"]},
        ],
        "links": [],
    }
    findings = check_workflow_sync(gui, {})
    metadata_findings = [
        finding for finding in findings if finding.check == "workflow.sync.widget_metadata_absent"
    ]
    assert len(metadata_findings) == 1
    assert metadata_findings[0].severity is Severity.WARNING
    assert "1 GUI node" in metadata_findings[0].message


def test_inconsistent_widget_metadata_names_each_offending_node() -> None:
    gui = {
        "nodes": [
            {
                "id": 1,
                "type": "KSampler",
                "inputs": [{"name": "steps", "widget": {}}],
                "widgets_values": [8],
            },
            {
                "id": 2,
                "type": "CLIPTextEncode",
                "inputs": [{"name": "clip"}],
                "widgets_values": ["cat"],
            },
        ],
        "links": [],
    }
    findings = check_workflow_sync(gui, {})
    finding = next(
        finding
        for finding in findings
        if finding.check == "workflow.sync.widget_metadata_inconsistent"
    )
    assert finding.severity is Severity.WARNING
    assert "2" in finding.message


def test_markdown_note_with_widget_values_is_not_metadata_inconsistent() -> None:
    gui = {
        "nodes": [
            {
                "id": 1,
                "type": "KSampler",
                "inputs": [{"name": "steps", "widget": {}}],
                "widgets_values": [8],
            },
            {"id": 2, "type": "MarkdownNote", "inputs": [], "widgets_values": ["docs"]},
        ],
        "links": [],
    }

    findings = check_workflow_sync(gui, {})

    assert all(
        finding.check != "workflow.sync.widget_metadata_inconsistent" for finding in findings
    )


def test_widget_metadata_absent_count_excludes_nodes_without_widget_values() -> None:
    gui = {
        "nodes": [
            {"id": 1, "type": "VAEDecode", "inputs": []},
            {"id": 2, "type": "KSampler", "inputs": [], "widgets_values": [8]},
        ],
        "links": [],
    }

    findings = check_workflow_sync(gui, {})
    finding = next(
        finding for finding in findings if finding.check == "workflow.sync.widget_metadata_absent"
    )

    assert "all 1 GUI node(s) carrying widget values" in finding.message


def _api_derived_from_gui(gui: dict[str, object]) -> dict[str, object]:
    nodes = gui["nodes"]
    assert isinstance(nodes, list)
    by_id = {str(node["id"]): node for node in nodes if isinstance(node, dict)}
    api: dict[str, object] = {
        node_id: {"class_type": node["type"], "inputs": {}} for node_id, node in by_id.items()
    }
    links = gui["links"]
    assert isinstance(links, list)
    for link in links:
        assert isinstance(link, list)
        target = by_id[str(link[3])]
        inputs = target.get("inputs")
        assert isinstance(inputs, list)
        input_config = inputs[link[4]]
        assert isinstance(input_config, dict)
        api_node = api[str(link[3])]
        assert isinstance(api_node, dict)
        api_inputs = api_node["inputs"]
        assert isinstance(api_inputs, dict)
        api_inputs[input_config["name"]] = [str(link[1]), link[2]]
    return api


def test_older_save_format_graph_is_accepted() -> None:
    gui = json.loads((Path(__file__).parents[1] / "examples/wan22_basic_workflow.json").read_text())
    assert isinstance(gui, dict)
    api = _api_derived_from_gui(gui)

    findings = check_workflow_sync(gui, api)

    assert not [finding for finding in findings if finding.severity is Severity.ERROR]
    assert [finding.check for finding in findings].count(
        "workflow.sync.widget_metadata_absent"
    ) == 1


@pytest.mark.parametrize("node_class", sorted(_NON_EXECUTABLE_GUI_CLASSES))
def test_non_executable_classes_never_report_node_missing_in_api(node_class: str) -> None:
    gui = {
        "nodes": [{"id": "1", "type": node_class, "inputs": [], "widgets_values": []}],
        "links": [],
    }
    findings = check_workflow_sync(gui, {})
    assert all(finding.check != "workflow.sync.node_missing_in_api" for finding in findings)


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
