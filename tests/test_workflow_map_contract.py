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
    _gui_links_by_target_input,
    _gui_nodes_by_id,
    check_api_graph_links,
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


def _raw_bundle_with_unknown_field() -> dict[str, object]:
    raw = _raw_bundle()
    raw["unexpected"] = "field"
    return raw


def test_schema_error_does_not_truncate_report_with_or_without_workflow_map(
    tmp_path: Path,
) -> None:
    """P1-1: a schema-invalid bundle must still surface its unrelated findings.

    An identical unrelated error (an unknown top-level field) must produce
    the same non-schema findings whether or not the bundle declares a
    workflow: map -- only the check name distinguishing
    bundle.config_invalid/schema.invalid may differ. (A custom_nodes schema
    error is deliberately not used here: it now gets its own dedicated
    custom_node.source_fields_invalid check regardless of the workflow: map,
    which is exercised separately in test_bundle_contract.py.)
    """
    with_map = _raw_bundle_with_unknown_field()
    without_map = _raw_bundle_with_unknown_field()
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
    ((2, "muted"), (4, "bypassed")),
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


def test_link_resolution_does_not_require_widget_metadata() -> None:
    """P1-3a: inputs[target_slot]['name'] resolves a link the same way with or
    without widget metadata -- the invariant that makes it safe to accept a
    GUI graph missing widget metadata instead of rejecting it outright.
    ComfyUI orders link sockets before widget sockets, so whether a *later*
    input carries a "widget" key never shifts an earlier link's target index.
    """
    gui = {
        "nodes": [
            {"id": 9, "type": "EmptyLatentImage", "inputs": [], "widgets_values": []},
            {
                "id": 2,
                "type": "KSampler",
                "inputs": [
                    {"name": "positive"},
                    {"name": "latent_image"},
                    {"name": "steps", "widget": {}},
                ],
                "widgets_values": [8],
            },
        ],
        "links": [[1, 9, 0, 2, 1, "LATENT"]],
    }
    with_widget = _gui_links_by_target_input(gui, _gui_nodes_by_id(gui))
    assert with_widget == {"2": {"latent_image": ("9", 0)}}

    stripped = copy.deepcopy(gui)
    for node in stripped["nodes"]:
        for input_config in node["inputs"]:
            input_config.pop("widget", None)
    without_widget = _gui_links_by_target_input(stripped, _gui_nodes_by_id(stripped))
    assert without_widget == with_widget

    wan22 = json.loads(
        (Path(__file__).parents[1] / "examples/wan22_basic_workflow.json").read_text()
    )
    wan22_links = _gui_links_by_target_input(wan22, _gui_nodes_by_id(wan22))
    declared_links = wan22["links"]
    assert wan22_links and not any(
        "widget" in inp for node in wan22["nodes"] for inp in (node.get("inputs") or [])
    )
    for link in declared_links:
        target_id, target_slot = str(link[3]), link[4]
        target_node = next(node for node in wan22["nodes"] if str(node["id"]) == target_id)
        expected_name = target_node["inputs"][target_slot]["name"]
        assert wan22_links[target_id][expected_name] == (str(link[1]), link[2])


def test_modern_export_and_older_save_produce_the_same_finding_codes() -> None:
    """P1-3b: the executable record of why the old GUI-format ERROR was unsound.

    An older Save file and a modern Export of the identical graph must
    produce the same structural finding codes; the only allowed difference
    is the non-blocking widget-metadata capability signal (a WARNING plus
    the INFO it implies for nodes whose widget values can't be
    cross-checked). If someone reads that WARNING as too lenient and
    reintroduces a hard rejection for missing widget metadata, this test
    fails and points here.
    """
    api = {
        "9": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "hello"}},
        "2": {
            "class_type": "KSampler",
            "inputs": {"positive": ["3", 0], "latent_image": ["9", 0], "steps": 8},
        },
    }
    older_save = {
        "nodes": [
            {"id": 9, "type": "EmptyLatentImage", "inputs": None, "widgets_values": [1024]},
            {"id": 3, "type": "CLIPTextEncode", "inputs": None, "widgets_values": ["hello"]},
            {
                "id": 2,
                "type": "KSampler",
                "inputs": [
                    {"name": "positive", "type": "CONDITIONING", "link": 1},
                    {"name": "latent_image", "type": "LATENT", "link": 2},
                ],
                "widgets_values": [8],
            },
        ],
        "links": [[1, 3, 0, 2, 0, "CONDITIONING"], [2, 9, 0, 2, 1, "LATENT"]],
    }
    modern_export = copy.deepcopy(older_save)
    modern_nodes = {node["id"]: node for node in modern_export["nodes"]}
    modern_nodes[9]["inputs"] = [{"name": "width", "widget": {}}]
    modern_nodes[3]["inputs"] = [{"name": "text", "widget": {}}]
    modern_nodes[2]["inputs"].append({"name": "steps", "widget": {}})

    older_checks = {finding.check for finding in check_workflow_sync(older_save, api)}
    modern_checks = {finding.check for finding in check_workflow_sync(modern_export, api)}
    non_blocking_capability_checks = {
        "workflow.sync.widget_metadata_absent",
        "workflow.sync.unaligned_nodes",
    }

    assert older_checks - non_blocking_capability_checks == modern_checks
    assert non_blocking_capability_checks <= older_checks
    assert not non_blocking_capability_checks & modern_checks


def _output_chain_gui_graph(
    *, sampler_mode: int = 0, decode_mode: int = 0, save_mode: int = 0
) -> dict[str, object]:
    """KSampler(3) -> VAEDecode(8) -> SaveImage(9), the P0-1 fixture from the r5 prompt."""
    return {
        "nodes": [
            {"id": 3, "type": "KSampler", "mode": sampler_mode, "inputs": [], "widgets_values": []},
            {
                "id": 8,
                "type": "VAEDecode",
                "mode": decode_mode,
                "inputs": [{"name": "samples", "link": 1}],
                "widgets_values": [],
            },
            {
                "id": 9,
                "type": "SaveImage",
                "mode": save_mode,
                "inputs": [{"name": "images", "link": 2}],
                "widgets_values": [],
            },
        ],
        "links": [
            [1, 3, 0, 8, 0, "LATENT"],
            [2, 8, 0, 9, 0, "IMAGE"],
        ],
    }


def _output_chain_api_graph(
    *, include: frozenset[str] = frozenset({"3", "8", "9"})
) -> dict[str, object]:
    api: dict[str, object] = {}
    if "3" in include:
        api["3"] = {"class_type": "KSampler", "inputs": {}}
    if "8" in include:
        api["8"] = {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0]}}
    if "9" in include:
        api["9"] = {"class_type": "SaveImage", "inputs": {"images": ["8", 0]}}
    return api


def test_missing_terminal_output_node_is_an_error() -> None:
    gui = _output_chain_gui_graph()
    api = _output_chain_api_graph(include=frozenset({"3", "8"}))

    findings = check_workflow_sync(gui, api)

    finding = next(
        finding for finding in findings if finding.check == "workflow.sync.node_missing_in_api"
    )
    assert finding.severity is Severity.ERROR
    assert "9" in finding.message
    assert "produces no output" in finding.message


def test_missing_mid_chain_node_stays_a_warning() -> None:
    gui = _output_chain_gui_graph()
    api = _output_chain_api_graph(include=frozenset({"3", "9"}))

    findings = [
        *check_workflow_sync(gui, api),
        *check_api_graph_links(api, "workflow.api.json"),
    ]

    missing = [f for f in findings if f.check == "workflow.sync.node_missing_in_api"]
    assert len(missing) == 1
    assert missing[0].severity is Severity.WARNING
    assert "8" in missing[0].message
    dangling = [f for f in findings if f.check == "workflow.api.dangling_link"]
    assert len(dangling) == 1
    assert dangling[0].severity is Severity.ERROR


def test_missing_whole_output_branch_is_an_error() -> None:
    gui = _output_chain_gui_graph()
    api = _output_chain_api_graph(include=frozenset({"3"}))

    findings = check_workflow_sync(gui, api)

    errors = [f for f in findings if f.severity is Severity.ERROR]
    assert len(errors) == 1
    assert errors[0].check == "workflow.sync.node_missing_in_api"
    assert "9" in errors[0].message


def test_isolated_node_omission_stays_a_warning() -> None:
    gui = _output_chain_gui_graph()
    gui_nodes = gui["nodes"]
    assert isinstance(gui_nodes, list)
    gui_nodes.append({"id": 99, "type": "Note2", "mode": 0, "inputs": [], "widgets_values": []})
    api = _output_chain_api_graph()

    findings = check_workflow_sync(gui, api)

    isolated = [
        f for f in findings if f.check == "workflow.sync.node_missing_in_api" and "99" in f.message
    ]
    assert len(isolated) == 1
    assert isolated[0].severity is Severity.WARNING


def test_muted_terminal_node_omission_is_not_an_error() -> None:
    gui = _output_chain_gui_graph(save_mode=2)
    api = _output_chain_api_graph(include=frozenset({"3", "8"}))

    findings = check_workflow_sync(gui, api)

    assert all(
        "9" not in finding.message
        for finding in findings
        if finding.check == "workflow.sync.node_missing_in_api"
    )


def test_non_executable_terminal_class_is_excluded() -> None:
    gui = _output_chain_gui_graph()
    gui_nodes = gui["nodes"]
    assert isinstance(gui_nodes, list)
    gui_nodes.append({"id": 5, "type": "Note", "mode": 0, "inputs": [], "widgets_values": []})
    api = _output_chain_api_graph()

    findings = check_workflow_sync(gui, api)

    assert all(
        "5" not in finding.message
        for finding in findings
        if finding.check == "workflow.sync.node_missing_in_api"
    )


def test_malformed_links_array_does_not_raise() -> None:
    gui = _output_chain_gui_graph()
    gui["links"] = [[1], "not-a-list", None]
    api = _output_chain_api_graph(include=frozenset({"3", "8"}))

    findings = check_workflow_sync(gui, api)

    finding = next(
        finding for finding in findings if finding.check == "workflow.sync.node_missing_in_api"
    )
    assert finding.severity is Severity.ERROR
    assert "9" in finding.message


def test_mode_labels_match_observed_comfyui_semantics() -> None:
    """P1-1: verified 2026-08-18 against the local ComfyUI v0.32 stack's converter
    (comfyui-workflow-to-api-converter-endpoint/workflow_converter.py): mode 2 is
    skipped outright ("Mode 2 is muted"), mode 4 is skipped but traced through to
    rewire its consumers ("Mode 4 is bypassed/disabled", trace_through_bypassed).
    """
    gui = _gui_graph()
    gui_nodes = gui["nodes"]
    assert isinstance(gui_nodes, list)
    sampler = next(node for node in gui_nodes if isinstance(node, dict) and node.get("id") == 2)
    assert isinstance(sampler, dict)

    sampler["mode"] = 2
    muted = next(
        f
        for f in check_workflow_sync(gui, _api_graph())
        if f.check == "workflow.sync.disabled_node_in_api"
    )
    assert "muted (mode=2)" in muted.message
    assert "absent from the API graph" in muted.message

    sampler["mode"] = 4
    bypassed = next(
        f
        for f in check_workflow_sync(gui, _api_graph())
        if f.check == "workflow.sync.disabled_node_in_api"
    )
    assert "bypassed (mode=4)" in bypassed.message
    assert "rewired through it" in bypassed.message
