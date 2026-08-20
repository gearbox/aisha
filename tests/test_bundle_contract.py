"""Tests for the offline Apex bundle-contract validator."""

from __future__ import annotations

import copy
import json
from typing import TYPE_CHECKING, NoReturn

import pytest

from ai_content_service import bundle_contract
from ai_content_service.bundle_contract import ContractReport, Severity, check_bundle_contract

if TYPE_CHECKING:
    from collections.abc import Mapping
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
            "base_image": "vastai/comfy:v0.30.0-cuda-13.2-py312",
        },
        "readiness_marker": {"node_class": "KSampler"},
        "workflow_file": "workflow.json",
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


def _workflow(path: Path, document: object | None = None) -> None:
    (path / "workflow.json").write_text(
        json.dumps(
            document
            if document is not None
            else {
                "nodes": [
                    {"id": 9, "type": "EmptyLatentImage"},
                    {"id": 3, "type": "TextEncodeQwenImageEditPlus"},
                    {"id": 2, "type": "KSampler"},
                ]
            }
        )
    )


def _report(
    tmp_path: Path,
    raw: object,
    *,
    workflow_document: object | None = None,
    object_info: Mapping[str, object] | None = None,
    workflow_provider_check: bool = False,
) -> ContractReport:
    root = tmp_path / "demo"
    version = root / "260101-01"
    version.mkdir(parents=True)
    root.joinpath("current").symlink_to("260101-01")
    _workflow(version, workflow_document)
    return check_bundle_contract(
        "demo",
        version,
        raw,
        bundle_root=root,
        index_entries=({"name": "demo", "path": "bundles/demo", "model_type": "aisha-image"},),
        object_info=object_info,
        workflow_provider_check=workflow_provider_check,
    )


def test_valid_bundle_has_no_findings(tmp_path: Path) -> None:
    report = _report(tmp_path, _raw_bundle())
    assert report.ok is True
    assert [finding.check for finding in report.findings] == ["workflow.map.absent"]


def test_live_provider_check_rejects_undeclared_custom_node(tmp_path: Path) -> None:
    workflow = {
        "nodes": [
            {"id": 9, "type": "EmptyLatentImage"},
            {"id": 3, "type": "TextEncodeQwenImageEditPlus"},
            {"id": 2, "type": "KSampler"},
            {"id": 42, "type": "PatchFlashAttentionKJ"},
        ]
    }
    object_info = {
        "EmptyLatentImage": {"python_module": "nodes"},
        "TextEncodeQwenImageEditPlus": {"python_module": "comfy_extras.nodes_qwen"},
        "KSampler": {"python_module": "nodes"},
        "PatchFlashAttentionKJ": {"python_module": "custom_nodes.comfyui-kjnodes"},
    }

    findings = {
        finding.check: finding
        for finding in _report(
            tmp_path,
            _raw_bundle(),
            workflow_document=workflow,
            object_info=object_info,
        ).findings
    }

    finding = findings["workflow.class_unprovided"]
    assert finding.severity is Severity.ERROR
    assert "PatchFlashAttentionKJ" in finding.message
    assert "comfyui-kjnodes" in finding.message


def test_live_provider_check_uses_default_workflow_when_declaration_is_null(
    tmp_path: Path,
) -> None:
    raw = _raw_bundle()
    raw["workflow_file"] = None
    workflow = {
        "nodes": [
            {"id": 42, "type": "PatchFlashAttentionKJ"},
        ]
    }
    object_info = {
        "PatchFlashAttentionKJ": {"python_module": "custom_nodes.comfyui-kjnodes"},
    }

    findings = {
        finding.check: finding
        for finding in _report(
            tmp_path,
            raw,
            workflow_document=workflow,
            object_info=object_info,
        ).findings
    }

    assert findings["workflow.class_unprovided"].location == "workflow.json"


def test_live_provider_check_allows_core_classes(tmp_path: Path) -> None:
    object_info = {
        "EmptyLatentImage": {"python_module": "nodes"},
        "TextEncodeQwenImageEditPlus": {"python_module": "comfy_extras.nodes_qwen"},
        "KSampler": {"python_module": "nodes"},
    }

    findings = _report(tmp_path, _raw_bundle(), object_info=object_info).findings

    assert not {
        finding.check for finding in findings if finding.check.startswith("workflow.class_")
    }


def test_live_provider_check_matches_custom_node_directory_case_insensitively(
    tmp_path: Path,
) -> None:
    raw = _raw_bundle()
    raw["custom_nodes"] = [
        {
            "name": "ComfyUI-KJNodes",
            "git_url": "https://github.com/kijai/ComfyUI-KJNodes",
            "commit_sha": "3f20054214fec9f9234fd3841ae6f1e4287948f6",
        }
    ]
    workflow = {"nodes": [{"id": 42, "type": "PatchFlashAttentionKJ"}]}
    object_info = {"PatchFlashAttentionKJ": {"python_module": "custom_nodes.comfyui-kjnodes"}}

    findings = _report(
        tmp_path,
        raw,
        workflow_document=workflow,
        object_info=object_info,
    ).findings

    assert all(finding.check != "workflow.class_unprovided" for finding in findings)


def test_live_provider_check_reports_missing_workflow_class(tmp_path: Path) -> None:
    workflow = {
        "nodes": [
            {"id": 9, "type": "EmptyLatentImage"},
            {"id": 3, "type": "TextEncodeQwenImageEditPlus"},
            {"id": 2, "type": "KSampler"},
            {"id": 42, "type": "MissingNode"},
        ]
    }
    object_info = {
        "EmptyLatentImage": {"python_module": "nodes"},
        "TextEncodeQwenImageEditPlus": {"python_module": "nodes"},
        "KSampler": {"python_module": "nodes"},
    }

    checks = {
        finding.check
        for finding in _report(
            tmp_path,
            _raw_bundle(),
            workflow_document=workflow,
            object_info=object_info,
        ).findings
    }

    assert "workflow.class_unknown" in checks


def test_live_provider_check_skips_missing_python_module(tmp_path: Path) -> None:
    object_info = {
        "EmptyLatentImage": {"python_module": "nodes"},
        "TextEncodeQwenImageEditPlus": {"python_module": "nodes"},
        "KSampler": {},
    }

    findings = _report(tmp_path, _raw_bundle(), object_info=object_info).findings

    assert any(
        finding.check == "workflow.class_provider_unknown" and finding.severity is Severity.INFO
        for finding in findings
    )
    assert all(finding.check != "workflow.class_unprovided" for finding in findings)


def test_live_provider_check_notes_when_no_comfyui_url_is_supplied(tmp_path: Path) -> None:
    findings = _report(tmp_path, _raw_bundle(), workflow_provider_check=True).findings

    assert any(
        finding.check == "workflow.class_provider_check_skipped"
        and finding.severity is Severity.INFO
        for finding in findings
    )


def test_hardware_bool_is_not_an_integer(tmp_path: Path) -> None:
    raw = _raw_bundle()
    hardware = raw["hardware"]
    assert isinstance(hardware, dict)
    hardware["comfyui_port"] = True

    report = _report(tmp_path, raw)

    assert any(finding.check == "hardware.comfyui_port.not_int" for finding in report.findings)


def test_schema_error_is_reported(tmp_path: Path) -> None:
    raw = _raw_bundle()
    raw["unexpected"] = "field"

    report = _report(tmp_path, raw)

    assert report.ok is False
    assert len(report.findings) == 1
    assert report.findings[0].severity is Severity.ERROR
    assert report.findings[0].check == "schema.invalid"


def test_schema_and_semantic_findings_are_reported_together(tmp_path: Path) -> None:
    raw = _raw_bundle()
    raw["unexpected"] = "field"
    hardware = raw["hardware"]
    assert isinstance(hardware, dict)
    hardware["gpu_whitelist"] = []

    checks = {finding.check for finding in _report(tmp_path, raw).findings}

    assert {"schema.invalid", "hardware.gpu_whitelist.empty"} <= checks


def test_schema_error_does_not_hide_index_or_symlink_findings(tmp_path: Path) -> None:
    raw = _raw_bundle()
    raw["unexpected"] = "field"
    root = tmp_path / "demo"
    version = root / "260101-01"
    version.mkdir(parents=True)
    _workflow(version)

    report = check_bundle_contract("demo", version, raw, bundle_root=root)

    checks = {finding.check for finding in report.findings}
    assert {"schema.invalid", "index.entry.missing", "index.current_symlink.missing"} <= checks


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
        "workflow.map.absent",
    }


@pytest.mark.parametrize(
    "field",
    [
        "min_disk_gb",
        "min_network_upload_mbps",
        "min_network_download_mbps",
        "num_gpus",
    ],
)
@pytest.mark.parametrize("value", [None, True, 1.0])
def test_hardware_integer_fields_report_missing_bool_and_float(
    tmp_path: Path, field: str, value: object
) -> None:
    raw = _raw_bundle()
    hardware = raw["hardware"]
    assert isinstance(hardware, dict)
    if value is None:
        hardware.pop(field)
    else:
        hardware[field] = value

    report = _report(tmp_path, raw)

    assert f"hardware.{field}.not_int" in {finding.check for finding in report.findings}


def test_hardware_integer_strings_are_accepted_by_apex_compatible_check(tmp_path: Path) -> None:
    raw = _raw_bundle()
    hardware = raw["hardware"]
    assert isinstance(hardware, dict)
    for field in (
        "min_disk_gb",
        "min_network_upload_mbps",
        "min_network_download_mbps",
        "num_gpus",
        "comfyui_port",
    ):
        hardware[field] = "18188"

    report = _report(tmp_path, raw)

    assert not {finding.check for finding in report.findings if finding.check.endswith(".not_int")}


@pytest.mark.parametrize("value", [None, 18188, "18188"])
def test_comfyui_port_default_or_absent_produces_no_finding(tmp_path: Path, value: object) -> None:
    raw = _raw_bundle()
    hardware = raw["hardware"]
    assert isinstance(hardware, dict)
    if value is None:
        hardware.pop("comfyui_port")
    else:
        hardware["comfyui_port"] = value

    checks = {finding.check for finding in _report(tmp_path, raw).findings}

    assert not {check for check in checks if check.startswith("hardware.comfyui_port.")}


def test_comfyui_port_non_default_is_a_warning(tmp_path: Path) -> None:
    raw = _raw_bundle()
    raw["hardware"]["comfyui_port"] = 18189  # type: ignore[index]

    findings = {finding.check: finding for finding in _report(tmp_path, raw).findings}

    assert findings["hardware.comfyui_port.non_default"].severity is Severity.WARNING


@pytest.mark.parametrize("value", [True, 8188.0])
def test_comfyui_port_rejects_bool_and_float_when_present(tmp_path: Path, value: object) -> None:
    raw = _raw_bundle()
    raw["hardware"]["comfyui_port"] = value  # type: ignore[index]

    checks = {finding.check for finding in _report(tmp_path, raw).findings}

    assert "hardware.comfyui_port.not_int" in checks


def test_environment_dual_pinning_is_warned(tmp_path: Path) -> None:
    raw = _raw_bundle()
    raw["hardware"]["template_hash_id"] = "template-123"  # type: ignore[index]
    raw["comfyui"] = {"commit": "a" * 40}
    raw["requirements_lock_file"] = "requirements.lock"

    checks = {finding.check for finding in _report(tmp_path, raw).findings}

    assert {"environment.dual_pinning", "requirements_lock.deprecated"} <= checks


def test_both_requirements_artifacts_are_rejected(tmp_path: Path) -> None:
    raw = _raw_bundle()
    raw["requirements_lock_file"] = "requirements.lock"
    raw["requirements_overlay_file"] = "requirements.overlay.txt"

    findings = {finding.check: finding for finding in _report(tmp_path, raw).findings}

    assert findings["requirements.both_declared"].severity is Severity.ERROR
    assert findings["requirements_lock.deprecated"].severity is Severity.WARNING


def test_overlay_pinned_alone_does_not_trigger_dual_pinning(tmp_path: Path) -> None:
    """R4: template + overlay is the recommended additive shape, not a duplicate.

    An overlay is additive by construction — it is not a second source of
    truth for the base environment the template already pins — so this
    configuration must not warn.
    """
    raw = _raw_bundle()
    raw["hardware"]["template_hash_id"] = "template-123"  # type: ignore[index]
    raw["requirements_overlay_file"] = "requirements.overlay.txt"

    checks = {finding.check for finding in _report(tmp_path, raw).findings}

    assert "environment.dual_pinning" not in checks


def test_comfyui_pin_without_template_is_warned(tmp_path: Path) -> None:
    raw = _raw_bundle()
    raw["comfyui"] = {"commit": "a" * 40}

    checks = {finding.check for finding in _report(tmp_path, raw).findings}

    assert "comfyui.pinned_without_template" in checks
    assert "environment.dual_pinning" not in checks


def test_template_only_bundle_has_no_environment_pinning_warning(tmp_path: Path) -> None:
    raw = _raw_bundle()
    raw["hardware"]["template_hash_id"] = "template-123"  # type: ignore[index]

    checks = {finding.check for finding in _report(tmp_path, raw).findings}

    assert (
        not {
            "environment.dual_pinning",
            "comfyui.pinned_without_template",
            "requirements_lock.deprecated",
        }
        & checks
    )


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        (lambda raw: raw.pop("hardware"), "hardware.missing"),
        (
            lambda raw: raw["hardware"].__setitem__("cuda_min_version", "not-a-version"),  # type: ignore[index, union-attr]
            "hardware.cuda_min_version.not_numeric",
        ),
        (
            lambda raw: raw["hardware"].__setitem__("gpu_whitelist", []),  # type: ignore[index, union-attr]
            "hardware.gpu_whitelist.empty",
        ),
        (
            lambda raw: raw["hardware"].__setitem__("gpu_whitelist", ["RTX_4090"]),  # type: ignore[index, union-attr]
            "hardware.gpu_whitelist.underscore_name",
        ),
        (
            lambda raw: raw["hardware"].__setitem__("gpu_whitelist", [4090]),  # type: ignore[index, union-attr]
            "hardware.gpu_whitelist.not_string",
        ),
        (
            lambda raw: raw["hardware"].__setitem__("gpu_whitelist", ["  "]),  # type: ignore[index, union-attr]
            "hardware.gpu_whitelist.blank",
        ),
        (
            lambda raw: raw["hardware"].__setitem__("gpu_whitelist", ["RTX  4090 "]),  # type: ignore[index, union-attr]
            "hardware.gpu_whitelist.not_normalized",
        ),
        (
            lambda raw: raw["hardware"].__setitem__("template_hash_id", " "),  # type: ignore[index, union-attr]
            "hardware.template_hash_id.blank",
        ),
    ],
)
def test_hardware_contract_findings(tmp_path: Path, change: object, expected: str) -> None:
    raw = _raw_bundle()
    assert callable(change)
    change(raw)

    assert expected in {finding.check for finding in _report(tmp_path, raw).findings}


def test_space_separated_gpu_names_are_accepted(tmp_path: Path) -> None:
    """Vast.ai's REST gpu_name values contain spaces; only the CLI DSL uses underscores."""
    raw = _raw_bundle()
    raw["hardware"]["gpu_whitelist"] = ["RTX 4090", "RTX 5090", "A100 SXM4"]  # type: ignore[index]

    checks = {finding.check for finding in _report(tmp_path, raw).findings}

    assert not {check for check in checks if check.startswith("hardware.gpu_whitelist.")}


def test_gpu_whitelist_reports_every_offending_entry(tmp_path: Path) -> None:
    raw = _raw_bundle()
    raw["hardware"]["gpu_whitelist"] = ["RTX_4090", "RTX 5090", "H100_SXM"]  # type: ignore[index]

    locations = {
        finding.location
        for finding in _report(tmp_path, raw).findings
        if finding.check == "hardware.gpu_whitelist.underscore_name"
    }

    assert locations == {
        "bundle.yaml:hardware.gpu_whitelist[0]",
        "bundle.yaml:hardware.gpu_whitelist[2]",
    }


def test_underscore_gpu_name_is_an_error(tmp_path: Path) -> None:
    """The Vast.ai REST API rejects CLI-style underscore GPU names."""
    raw = _raw_bundle()
    raw["hardware"]["gpu_whitelist"] = ["RTX_4090"]  # type: ignore[index]

    report = _report(tmp_path, raw)

    assert report.ok is False
    assert any(
        finding.check == "hardware.gpu_whitelist.underscore_name"
        and finding.severity is Severity.ERROR
        for finding in report.findings
    )


def test_hardware_base_image_absent_is_a_warning_not_error(tmp_path: Path) -> None:
    """Part C: a bundle with no recorded base_image is still ok=True -- it is
    advisory, not a provisioning gate."""
    raw = _raw_bundle()
    hardware = raw["hardware"]
    assert isinstance(hardware, dict)
    hardware.pop("base_image")

    report = _report(tmp_path, raw)

    findings = {finding.check: finding for finding in report.findings}
    assert findings["hardware.base_image.absent"].severity is Severity.WARNING
    assert report.ok is True


def test_hardware_base_image_blank_is_also_absent(tmp_path: Path) -> None:
    raw = _raw_bundle()
    hardware = raw["hardware"]
    assert isinstance(hardware, dict)
    hardware["base_image"] = "   "

    checks = {finding.check for finding in _report(tmp_path, raw).findings}

    assert "hardware.base_image.absent" in checks


@pytest.mark.parametrize("value", [42, True, [], {"image": "x"}])
def test_hardware_base_image_non_string_is_an_error(tmp_path: Path, value: object) -> None:
    raw = _raw_bundle()
    hardware = raw["hardware"]
    assert isinstance(hardware, dict)
    hardware["base_image"] = value

    findings = {finding.check: finding for finding in _report(tmp_path, raw).findings}

    assert findings["hardware.base_image.not_string"].severity is Severity.ERROR


@pytest.mark.parametrize(
    ("generation", "expected"),
    [
        ({"defaults": {"sampler": "unknown"}}, "generation.defaults.sampler.unknown_enum"),
        ({"defaults": {"scheduler": "unknown"}}, "generation.defaults.scheduler.unknown_enum"),
        ({"defaults": {"resolution": "unknown"}}, "generation.defaults.resolution.unknown_enum"),
        (
            {"constraints": {"allowed_samplers": ["unknown"]}},
            "generation.constraints.allowed_samplers.unknown_enum",
        ),
        (
            {"constraints": {"allowed_schedulers": ["unknown"]}},
            "generation.constraints.allowed_schedulers.unknown_enum",
        ),
    ],
)
def test_generation_unknown_enums_are_reported(
    tmp_path: Path, generation: dict[str, object], expected: str
) -> None:
    raw = _raw_bundle()
    raw["generation"] = generation

    assert expected in {finding.check for finding in _report(tmp_path, raw).findings}


@pytest.mark.parametrize(
    "constraints",
    [
        {"latent_multiple": 0},
        {"max_megapixels": 0},
        {"latent_multiple": 8, "max_edge": 4},
        {"min_steps": 10, "max_steps": 5},
        {"min_cfg": 2.0, "max_cfg": 1.0},
    ],
)
def test_generation_constraint_invariants_are_independent(
    tmp_path: Path, constraints: dict[str, object]
) -> None:
    raw = _raw_bundle()
    raw["generation"] = {"constraints": constraints}

    assert "generation.constraints.invariant" in {
        finding.check for finding in _report(tmp_path, raw).findings
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("latent_multiple", "abc"),
        ("max_megapixels", "abc"),
        ("max_edge", 1536.5),
        ("min_steps", []),
        ("max_steps", "abc"),
        ("min_cfg", []),
        ("max_cfg", "1,5"),
    ],
)
def test_generation_numeric_constraints_must_be_parseable_by_apex(
    tmp_path: Path, field: str, value: object
) -> None:
    raw = _raw_bundle()
    raw["generation"] = {"constraints": {field: value}}

    findings = _report(tmp_path, raw).findings

    assert "schema.invalid" in {finding.check for finding in findings}
    assert any(
        finding.check == f"generation.constraints.{field}.not_numeric"
        and finding.location == f"bundle.yaml:generation.constraints.{field}"
        for finding in findings
    )


def test_generation_max_edge_invariant_names_both_constraint_keys(tmp_path: Path) -> None:
    raw = _raw_bundle()
    raw["generation"] = {"constraints": {"latent_multiple": 8, "max_edge": 4}}

    findings = _report(tmp_path, raw).findings

    assert any(
        finding.check == "generation.constraints.invariant"
        and "max_edge" in finding.message
        and "latent_multiple" in finding.message
        for finding in findings
    )


def test_model_contract_findings_are_retained_together(tmp_path: Path) -> None:
    raw = _raw_bundle()
    raw["models"] = [
        {
            "name": "first",
            "model_type": "checkpoints",
            "subdirectory": "nested",
            "files": [
                {"name": "one", "url": "", "filename": "one", "sha256": None},
                {
                    "name": "two",
                    "url": "https://example.com/two",
                    "filename": "two",
                    "sha256": "b" * 64,
                },
            ],
        }
    ]

    checks = {finding.check for finding in _report(tmp_path, raw).findings}

    assert {
        "checkpoints.nested",
        "checkpoints.multiple",
        "models.file.sha256_missing",
        "models.file.size_bytes_missing",
    } <= checks


def test_model_file_http_url_is_an_error(tmp_path: Path) -> None:
    raw = _raw_bundle()
    raw["models"][0]["files"][0]["url"] = "http://example.com/model.safetensors"  # type: ignore[index]

    report = _report(tmp_path, raw)

    finding = next(f for f in report.findings if f.check == "models.file.url_not_https")
    assert finding.severity is Severity.ERROR
    assert report.ok is False


def test_model_file_empty_url_placeholder_does_not_trigger_https_check(tmp_path: Path) -> None:
    raw = _raw_bundle()
    raw["models"][0]["files"][0]["url"] = ""  # type: ignore[index]

    checks = {finding.check for finding in _report(tmp_path, raw).findings}

    assert "models.file.url_not_https" not in checks


def _bundle_with_custom_node_pip_requirement(entry: object) -> dict[str, object]:
    raw = _raw_bundle()
    raw["custom_nodes"] = [
        {
            "name": "TestNode",
            "git_url": "https://github.com/test/node",
            "commit_sha": "a" * 40,
            "pip_requirements": [entry],
        }
    ]
    return raw


@pytest.mark.parametrize(
    ("entry", "expected_check", "expected_severity"),
    [
        ("-r extras.txt", "custom_nodes.pip_requirements.directive", Severity.ERROR),
        ("-e .", "custom_nodes.pip_requirements.directive", Severity.ERROR),
        (
            "--extra-index-url https://example.com",
            "custom_nodes.pip_requirements.directive",
            Severity.ERROR,
        ),
        ("not a valid requirement!!", "custom_nodes.pip_requirements.unparseable", Severity.ERROR),
        ("color-matcher", "custom_nodes.pip_requirements.unpinned", Severity.WARNING),
        ("color-matcher>=1.0", "custom_nodes.pip_requirements.unpinned", Severity.WARNING),
    ],
)
def test_custom_node_pip_requirement_findings(
    tmp_path: Path, entry: str, expected_check: str, expected_severity: Severity
) -> None:
    raw = _bundle_with_custom_node_pip_requirement(entry)

    report = _report(tmp_path, raw)

    finding = next(f for f in report.findings if f.check == expected_check)
    assert finding.severity is expected_severity
    assert finding.location == "bundle.yaml:custom_nodes[0].pip_requirements[0]"
    if expected_severity is Severity.ERROR:
        assert report.ok is False


def test_custom_node_pip_requirement_pinned_entry_has_no_finding(tmp_path: Path) -> None:
    raw = _bundle_with_custom_node_pip_requirement("color-matcher==1.2.3")

    checks = {finding.check for finding in _report(tmp_path, raw).findings}

    assert not any(check.startswith("custom_nodes.pip_requirements.") for check in checks)


def test_custom_node_pip_requirement_direct_url_reference_has_no_finding(tmp_path: Path) -> None:
    raw = _bundle_with_custom_node_pip_requirement(
        "color-matcher @ https://example.com/color-matcher-1.0-py3-none-any.whl"
    )

    checks = {finding.check for finding in _report(tmp_path, raw).findings}

    assert not any(check.startswith("custom_nodes.pip_requirements.") for check in checks)


def test_custom_node_without_pip_requirements_has_no_finding(tmp_path: Path) -> None:
    raw = _raw_bundle()
    raw["custom_nodes"] = [
        {"name": "TestNode", "git_url": "https://github.com/test/node", "commit_sha": "a" * 40}
    ]

    checks = {finding.check for finding in _report(tmp_path, raw).findings}

    assert not any(check.startswith("custom_nodes.pip_requirements.") for check in checks)


def test_custom_node_registry_missing_version_is_source_fields_invalid(tmp_path: Path) -> None:
    raw = _raw_bundle()
    raw["custom_nodes"] = [
        {"name": "comfyui-kjnodes", "source": "registry", "node_id": "comfyui-kjnodes"}
    ]

    report = _report(tmp_path, raw)

    assert report.ok is False
    finding = next(f for f in report.findings if f.check == "custom_node.source_fields_invalid")
    assert finding.severity is Severity.ERROR
    assert "version" in finding.message


def test_custom_node_git_with_registry_fields_is_source_fields_invalid(tmp_path: Path) -> None:
    raw = _raw_bundle()
    raw["custom_nodes"] = [
        {
            "name": "TestNode",
            "git_url": "https://github.com/test/node",
            "commit_sha": "a" * 40,
            "node_id": "TestNode",
        }
    ]

    report = _report(tmp_path, raw)

    finding = next(f for f in report.findings if f.check == "custom_node.source_fields_invalid")
    assert finding.severity is Severity.ERROR
    assert "node_id" in finding.message


def test_custom_node_valid_registry_entry_has_no_source_fields_finding(tmp_path: Path) -> None:
    raw = _raw_bundle()
    raw["custom_nodes"] = [
        {
            "name": "comfyui-kjnodes",
            "source": "registry",
            "node_id": "comfyui-kjnodes",
            "version": "1.5.0",
        }
    ]

    checks = {finding.check for finding in _report(tmp_path, raw).findings}

    assert "custom_node.source_fields_invalid" not in checks
    assert "schema.invalid" not in checks
    assert "bundle.config_invalid" not in checks


@pytest.mark.parametrize("version", ["latest", "*", "^1.5.0", ">=1.0", "1.5, 2.0"])
def test_custom_node_registry_inexact_version_is_unpinned(tmp_path: Path, version: str) -> None:
    raw = _raw_bundle()
    raw["custom_nodes"] = [
        {
            "name": "comfyui-kjnodes",
            "source": "registry",
            "node_id": "comfyui-kjnodes",
            "version": version,
        }
    ]

    report = _report(tmp_path, raw)

    finding = next(f for f in report.findings if f.check == "custom_node.registry_unpinned")
    assert finding.severity is Severity.ERROR
    assert finding.location == "bundle.yaml:custom_nodes[0].version"


def test_custom_node_registry_exact_version_has_no_unpinned_finding(tmp_path: Path) -> None:
    raw = _raw_bundle()
    raw["custom_nodes"] = [
        {
            "name": "comfyui-kjnodes",
            "source": "registry",
            "node_id": "comfyui-kjnodes",
            "version": "1.5.0",
        }
    ]

    checks = {finding.check for finding in _report(tmp_path, raw).findings}

    assert "custom_node.registry_unpinned" not in checks


def test_custom_node_pinned_to_head_never_fires_from_parsed_data(tmp_path: Path) -> None:
    """acs snapshot --pin-to-head records its compromise only as a YAML `#`
    comment (see SnapshotManager._pin_to_head_bundle_comments), which
    yaml.safe_load discards before `raw` reaches this validator, and the
    schema carries no machine-readable field for it. A plain git entry --
    even one an operator might have pinned to HEAD -- must never trigger
    this finding from parsed data alone.
    """
    raw = _raw_bundle()
    raw["custom_nodes"] = [
        {
            "name": "comfyui-kjnodes",
            "git_url": "https://github.com/kijai/ComfyUI-KJNodes",
            "commit_sha": "3f20054214fec9f9234fd3841ae6f1e4287948f6",
        }
    ]

    checks = {finding.check for finding in _report(tmp_path, raw).findings}

    assert "custom_node.pinned_to_head" not in checks


@pytest.mark.parametrize(
    ("document", "expected"),
    [
        ({}, "workflow.not_gui_format"),
        ({"nodes": [{"id": 9, "type": "EmptyLatentImage"}]}, "workflow.missing_node_id"),
        (
            {
                "nodes": [
                    {"id": 9, "type": "Wrong"},
                    {"id": 3, "type": "WrongPrompt"},
                    {"id": 2, "type": "Wrong"},
                ]
            },
            "workflow.node_class_mismatch",
        ),
    ],
)
def test_workflow_contract_findings(tmp_path: Path, document: object, expected: str) -> None:
    raw = _raw_bundle()
    root = tmp_path / "demo"
    version = root / "260101-01"
    version.mkdir(parents=True)
    root.joinpath("current").symlink_to("260101-01")
    _workflow(version, document)

    report = check_bundle_contract(
        "demo",
        version,
        raw,
        bundle_root=root,
        index_entries=({"name": "demo", "model_type": "aisha-image"},),
    )
    assert expected in {finding.check for finding in report.findings}


def test_missing_workflow_declaration_is_reported(tmp_path: Path) -> None:
    raw = _raw_bundle()
    raw.pop("workflow_file")

    report = _report(tmp_path, raw)

    assert any(finding.check == "workflow.missing" for finding in report.findings)


def test_non_default_workflow_file_is_checked_and_rejected_for_apex(tmp_path: Path) -> None:
    raw = _raw_bundle()
    raw["workflow_file"] = "custom.json"
    root = tmp_path / "demo"
    version = root / "260101-01"
    version.mkdir(parents=True)
    root.joinpath("current").symlink_to("260101-01")
    _workflow(version)
    (version / "custom.json").write_text("{}")

    report = check_bundle_contract(
        "demo",
        version,
        raw,
        bundle_root=root,
        index_entries=({"name": "demo", "model_type": "aisha-image"},),
    )

    findings = {finding.check: finding for finding in report.findings}
    assert "workflow.non_default_filename" in findings
    assert findings["workflow.not_gui_format"].location == "custom.json"


def test_metadata_index_and_warning_findings(tmp_path: Path) -> None:
    raw = _raw_bundle()
    metadata = raw["metadata"]
    hardware = raw["hardware"]
    assert isinstance(metadata, dict)
    assert isinstance(hardware, dict)
    metadata.update({"version": "260101-02", "tested": False})
    raw.pop("readiness_marker")
    hardware["comfyui_port"] = 8188
    root = tmp_path / "demo"
    version = root / "260101-01"
    version.mkdir(parents=True)
    _workflow(version)

    report = check_bundle_contract(
        "demo",
        version,
        raw,
        bundle_root=root,
        index_entries=(
            {"name": "demo"},
            {"name": "other", "model_type": "not-apex"},
            {"name": "one", "model_type": "aisha-image", "default_bundle": True},
            {"name": "two", "model_type": "aisha-image", "default_bundle": True},
        ),
        all_bundles=True,
    )
    checks = {finding.check for finding in report.findings}
    assert {
        "metadata.version_mismatch",
        "metadata.tested_false",
        "readiness_marker.absent",
        "hardware.comfyui_port.non_default",
        "index.model_type.missing",
        "index.current_symlink.missing",
    } <= checks


def test_index_unknown_and_duplicate_defaults_are_reported_for_matching_bundle(
    tmp_path: Path,
) -> None:
    raw = _raw_bundle()
    root = tmp_path / "demo"
    version = root / "260101-01"
    version.mkdir(parents=True)
    root.joinpath("current").symlink_to("260101-01")
    _workflow(version)
    report = check_bundle_contract(
        "demo",
        version,
        raw,
        bundle_root=root,
        index_entries=(
            {"name": "demo", "model_type": "not-apex", "default_bundle": True},
            {"name": "other", "model_type": "not-apex", "default_bundle": True},
        ),
        all_bundles=True,
    )
    checks = {finding.check for finding in report.findings}
    assert "index.model_type.unknown" in checks
    assert "index.default_bundle.duplicate" in checks


def test_index_missing_entry_has_a_distinct_check_id(tmp_path: Path) -> None:
    raw = _raw_bundle()
    root = tmp_path / "demo"
    version = root / "260101-01"
    version.mkdir(parents=True)
    root.joinpath("current").symlink_to("260101-01")
    _workflow(version)

    report = check_bundle_contract("demo", version, raw, bundle_root=root)

    assert "index.entry.missing" in {finding.check for finding in report.findings}


def test_duplicate_index_entries_are_reported_without_skipping_entry_checks(tmp_path: Path) -> None:
    raw = _raw_bundle()
    root = tmp_path / "demo"
    version = root / "260101-01"
    version.mkdir(parents=True)
    _workflow(version)

    report = check_bundle_contract(
        "demo",
        version,
        raw,
        bundle_root=root,
        index_entries=(
            {"name": "demo"},
            {"name": "demo", "model_type": "aisha-image"},
        ),
    )

    checks = [finding.check for finding in report.findings]
    assert "index.entry.duplicate" in checks
    assert "index.model_type.missing" in checks
    assert checks.count("index.current_symlink.missing") == 1


def test_checker_failure_becomes_a_contract_finding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def raise_hardware_error(_raw: object) -> NoReturn:
        raise OSError("bad")

    monkeypatch.setattr(
        bundle_contract,
        "_check_hardware",
        raise_hardware_error,
    )

    report = _report(tmp_path, _raw_bundle())

    assert [finding.check for finding in report.findings] == ["contract.check_failed"]
