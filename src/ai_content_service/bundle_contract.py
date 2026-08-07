"""Offline validation of the bundle contract consumed by Apex.

The schema models protect Aisha's own boundary. This module validates the
stricter conventions Apex applies while indexing and executing a bundle, and
returns findings rather than throwing so it can be used from CI as well as the
CLI.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Final

from pydantic import ValidationError

from .config import BundleConfig

if TYPE_CHECKING:
    from pathlib import Path

# --- VENDORED FROM APEX -- keep in sync with apex/src/core/enums.py and
# --- apex/src/api/services/workflow_service.py::NodeIDs.
# --- Consolidation is tracked as follow-up B5; until then this is the ONLY
# --- place these values may appear in this repo.
_APEX_SAMPLERS: Final[frozenset[str]] = frozenset(
    {
        "euler",
        "euler_ancestral",
        "euler_cfg_pp",
        "heun",
        "dpm_2",
        "dpm_2_ancestral",
        "lms",
        "dpmpp_2s_ancestral",
        "dpmpp_sde",
        "dpmpp_2m",
        "dpmpp_2m_sde",
        "dpmpp_3m_sde",
        "ddim",
        "uni_pc",
        "uni_pc_bh2",
        "lcm",
        "res_multistep",
    }
)
_APEX_SCHEDULERS: Final[frozenset[str]] = frozenset(
    {
        "normal",
        "karras",
        "exponential",
        "sgm_uniform",
        "simple",
        "ddim_uniform",
        "beta",
        "linear_quadratic",
        "kl_optimal",
    }
)
_APEX_RESOLUTION_TIERS: Final[frozenset[str]] = frozenset({"draft", "standard", "high", "ultra"})
_APEX_MODEL_TYPES: Final[frozenset[str]] = frozenset({"aisha-image", "aisha-video"})
_APEX_REQUIRED_WORKFLOW_NODE_IDS: Final[frozenset[str]] = frozenset({"9", "3", "2"})
_APEX_WORKFLOW_NODE_CLASSES: Final[dict[str, str]] = {
    "9": "EmptyLatentImage",
    "2": "KSampler",
}
_APEX_DEFAULT_COMFYUI_PORT: Final = 18188
_APEX_PROMPT_WIDGET_NODE_CLASSES: Final[frozenset[str]] = frozenset({"TextEncodeQwenImageEditPlus"})
_APEX_WORKFLOW_FILENAME: Final = "workflow.json"

_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}")
_HARDWARE_INT_FIELDS: Final[tuple[str, ...]] = (
    "min_disk_gb",
    "min_network_upload_mbps",
    "min_network_download_mbps",
    "num_gpus",
    "comfyui_port",
)


class Severity(str, Enum):
    """Severity of a static contract finding."""

    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True, slots=True)
class Finding:
    """One stable, machine-readable contract finding."""

    severity: Severity
    check: str
    message: str
    location: str


@dataclass(frozen=True, slots=True)
class ContractReport:
    """The findings for one resolved bundle."""

    bundle_name: str
    findings: tuple[Finding, ...]

    @property
    def ok(self) -> bool:
        return all(finding.severity is not Severity.ERROR for finding in self.findings)


def _finding(severity: Severity, check: str, message: str, location: str) -> Finding:
    return Finding(severity=severity, check=check, message=message, location=location)


def _bundle_location(suffix: str = "") -> str:
    return f"bundle.yaml{suffix}"


def _as_mapping(value: object) -> Mapping[str, object] | None:
    return value if isinstance(value, Mapping) else None


def _as_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _as_number(value: object) -> float | None:
    """Parse Apex's numeric constraint values while rejecting bool coercion."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _check_hardware(raw: Mapping[str, object]) -> list[Finding]:
    hardware = _as_mapping(raw.get("hardware"))
    if hardware is None:
        return [
            _finding(
                Severity.ERROR,
                "hardware.missing",
                "Apex requires hardware to be a mapping.",
                _bundle_location(":hardware"),
            )
        ]

    findings: list[Finding] = []
    for field in _HARDWARE_INT_FIELDS:
        value = hardware.get(field)
        if _as_int(value) is None:
            findings.append(
                _finding(
                    Severity.ERROR,
                    f"hardware.{field}.not_int",
                    "Must be an integer or an integer-valued string; booleans and floats are invalid.",
                    _bundle_location(f":hardware.{field}"),
                )
            )

    cuda_version = hardware.get("cuda_min_version")
    try:
        if cuda_version is None or isinstance(cuda_version, bool):
            raise ValueError
        float(str(cuda_version))
    except (TypeError, ValueError):
        findings.append(
            _finding(
                Severity.ERROR,
                "hardware.cuda_min_version.not_numeric",
                "Must be present and float()-parseable.",
                _bundle_location(":hardware.cuda_min_version"),
            )
        )

    whitelist = hardware.get("gpu_whitelist")
    if not isinstance(whitelist, list) or not whitelist:
        findings.append(
            _finding(
                Severity.ERROR,
                "hardware.gpu_whitelist.empty",
                "Must be a non-empty list of Vast.ai GPU names.",
                _bundle_location(":hardware.gpu_whitelist"),
            )
        )
    else:
        for entry in whitelist:
            if not isinstance(entry, str) or " " in entry:
                findings.append(
                    _finding(
                        Severity.ERROR,
                        "hardware.gpu_whitelist.space_separated",
                        "GPU whitelist entries must use Vast.ai underscore names, not space-separated names.",
                        _bundle_location(":hardware.gpu_whitelist"),
                    )
                )
                break

    template_hash = hardware.get("template_hash_id")
    if template_hash is not None and (
        not isinstance(template_hash, str) or not template_hash.strip()
    ):
        findings.append(
            _finding(
                Severity.ERROR,
                "hardware.template_hash_id.blank",
                "When present, template_hash_id must be a non-empty string.",
                _bundle_location(":hardware.template_hash_id"),
            )
        )

    port = _as_int(hardware.get("comfyui_port"))
    if port is not None and port != _APEX_DEFAULT_COMFYUI_PORT:
        findings.append(
            _finding(
                Severity.WARNING,
                "hardware.comfyui_port.non_default",
                f"Apex defaults to ComfyUI port {_APEX_DEFAULT_COMFYUI_PORT}; bundle declares {port}.",
                _bundle_location(":hardware.comfyui_port"),
            )
        )

    base_image = hardware.get("base_image")
    if base_image is None or (isinstance(base_image, str) and not base_image.strip()):
        findings.append(
            _finding(
                Severity.WARNING,
                "hardware.base_image.absent",
                "No base_image recorded; this bundle cannot be reasoned about when the template moves.",
                _bundle_location(":hardware.base_image"),
            )
        )
    return findings


def _check_enum(
    value: object,
    permitted: frozenset[str],
    check: str,
    location: str,
) -> list[Finding]:
    if value is None or (isinstance(value, str) and value in permitted):
        return []
    return [
        _finding(
            Severity.ERROR,
            check,
            f"Unknown value {value!r}; expected one of {', '.join(sorted(permitted))}.",
            location,
        )
    ]


def _check_generation(raw: Mapping[str, object]) -> list[Finding]:
    generation = _as_mapping(raw.get("generation"))
    if generation is None:
        return []
    defaults = _as_mapping(generation.get("defaults")) or {}
    constraints = _as_mapping(generation.get("constraints")) or {}
    findings: list[Finding] = []
    findings.extend(
        _check_enum(
            defaults.get("sampler"),
            _APEX_SAMPLERS,
            "generation.defaults.sampler.unknown_enum",
            _bundle_location(":generation.defaults.sampler"),
        )
    )
    findings.extend(
        _check_enum(
            defaults.get("scheduler"),
            _APEX_SCHEDULERS,
            "generation.defaults.scheduler.unknown_enum",
            _bundle_location(":generation.defaults.scheduler"),
        )
    )
    findings.extend(
        _check_enum(
            defaults.get("resolution"),
            _APEX_RESOLUTION_TIERS,
            "generation.defaults.resolution.unknown_enum",
            _bundle_location(":generation.defaults.resolution"),
        )
    )
    for field, permitted in (
        ("allowed_samplers", _APEX_SAMPLERS),
        ("allowed_schedulers", _APEX_SCHEDULERS),
    ):
        values = constraints.get(field)
        if values is None:
            continue
        if not isinstance(values, list):
            findings.extend(
                _check_enum(
                    values,
                    permitted,
                    f"generation.constraints.{field}.unknown_enum",
                    _bundle_location(f":generation.constraints.{field}"),
                )
            )
            continue
        for value in values:
            findings.extend(
                _check_enum(
                    value,
                    permitted,
                    f"generation.constraints.{field}.unknown_enum",
                    _bundle_location(f":generation.constraints.{field}"),
                )
            )

    numeric_constraints = {
        "latent_multiple": _as_int(constraints.get("latent_multiple")),
        "max_megapixels": _as_number(constraints.get("max_megapixels")),
        "max_edge": _as_int(constraints.get("max_edge")),
        "min_steps": _as_int(constraints.get("min_steps")),
        "max_steps": _as_int(constraints.get("max_steps")),
        "min_cfg": _as_number(constraints.get("min_cfg")),
        "max_cfg": _as_number(constraints.get("max_cfg")),
    }
    numeric_checks = {
        "latent_multiple": "generation.constraints.latent_multiple.not_numeric",
        "max_megapixels": "generation.constraints.max_megapixels.not_numeric",
        "max_edge": "generation.constraints.max_edge.not_numeric",
        "min_steps": "generation.constraints.min_steps.not_numeric",
        "max_steps": "generation.constraints.max_steps.not_numeric",
        "min_cfg": "generation.constraints.min_cfg.not_numeric",
        "max_cfg": "generation.constraints.max_cfg.not_numeric",
    }
    findings.extend(
        _finding(
            Severity.ERROR,
            numeric_checks[field],
            "Must be parseable as the numeric type Apex accepts for this constraint.",
            _bundle_location(f":generation.constraints.{field}"),
        )
        for field, value in numeric_constraints.items()
        if field in constraints and value is None
    )
    latent_multiple = numeric_constraints["latent_multiple"]
    max_megapixels = numeric_constraints["max_megapixels"]
    max_edge = numeric_constraints["max_edge"]
    min_steps = numeric_constraints["min_steps"]
    max_steps = numeric_constraints["max_steps"]
    min_cfg = numeric_constraints["min_cfg"]
    max_cfg = numeric_constraints["max_cfg"]

    if latent_multiple is not None and latent_multiple <= 0:
        findings.append(
            _finding(
                Severity.ERROR,
                "generation.constraints.invariant",
                "latent_multiple must be positive.",
                _bundle_location(":generation.constraints.latent_multiple"),
            )
        )
    if max_megapixels is not None and max_megapixels <= 0:
        findings.append(
            _finding(
                Severity.ERROR,
                "generation.constraints.invariant",
                "max_megapixels must be positive.",
                _bundle_location(":generation.constraints.max_megapixels"),
            )
        )
    if latent_multiple is not None and max_edge is not None and max_edge < latent_multiple:
        findings.append(
            _finding(
                Severity.ERROR,
                "generation.constraints.invariant",
                "max_edge must be greater than or equal to latent_multiple.",
                _bundle_location(":generation.constraints.max_edge"),
            )
        )
    if min_steps is not None and max_steps is not None and min_steps > max_steps:
        findings.append(
            _finding(
                Severity.ERROR,
                "generation.constraints.invariant",
                "min_steps must be less than or equal to max_steps.",
                _bundle_location(":generation.constraints.max_steps"),
            )
        )
    if min_cfg is not None and max_cfg is not None and min_cfg > max_cfg:
        findings.append(
            _finding(
                Severity.ERROR,
                "generation.constraints.invariant",
                "min_cfg must be less than or equal to max_cfg.",
                _bundle_location(":generation.constraints.max_cfg"),
            )
        )
    return findings


def _check_models(raw: Mapping[str, object]) -> list[Finding]:
    findings: list[Finding] = []
    checkpoint_files = 0
    models = raw.get("models")
    raw_models = models if isinstance(models, list) else []
    for model_index, raw_model in enumerate(raw_models):
        model = _as_mapping(raw_model)
        if model is None:
            continue
        model_type = model.get("model_type")
        if model_type == "checkpoints":
            if model.get("subdirectory") is not None or model.get("subfolder") is not None:
                findings.append(
                    _finding(
                        Severity.ERROR,
                        "checkpoints.nested",
                        "Checkpoint groups cannot declare subdirectory/subfolder because Apex injects a bare checkpoint filename.",
                        _bundle_location(f":models[{model_index}].subdirectory"),
                    )
                )
            files = model.get("files")
            if isinstance(files, list):
                checkpoint_files += len(files)
        files = model.get("files")
        if not isinstance(files, list):
            continue
        for file_index, raw_file in enumerate(files):
            file = _as_mapping(raw_file)
            if file is None:
                continue
            location = _bundle_location(f":models[{model_index}].files[{file_index}]")
            sha256 = file.get("sha256")
            if not isinstance(sha256, str) or not _SHA256_RE.fullmatch(sha256):
                findings.append(
                    _finding(
                        Severity.ERROR,
                        "models.file.sha256_missing",
                        "Every cached model file needs a 64-character lowercase SHA-256 digest.",
                        f"{location}.sha256",
                    )
                )
            if file.get("size_bytes") is None:
                findings.append(
                    _finding(
                        Severity.WARNING,
                        "models.file.size_bytes_missing",
                        "Without size_bytes, deploy cannot preflight disk capacity or budget the transfer timeout.",
                        f"{location}.size_bytes",
                    )
                )
    if checkpoint_files > 1:
        findings.append(
            _finding(
                Severity.ERROR,
                "checkpoints.multiple",
                "Apex supports at most one checkpoint file across all checkpoint groups.",
                _bundle_location(":models"),
            )
        )
    return findings


def _check_workflow(bundle_path: Path, workflow_file: str | None) -> list[Finding]:
    if workflow_file is None:
        return [
            _finding(
                Severity.ERROR,
                "workflow.missing",
                "Apex requires a workflow_file declaration.",
                _bundle_location(":workflow_file"),
            )
        ]

    findings: list[Finding] = []
    if workflow_file != _APEX_WORKFLOW_FILENAME:
        findings.append(
            _finding(
                Severity.ERROR,
                "workflow.non_default_filename",
                f"Apex unconditionally loads {_APEX_WORKFLOW_FILENAME!r}, not {workflow_file!r}.",
                _bundle_location(":workflow_file"),
            )
        )
    workflow_path = bundle_path / workflow_file
    try:
        workflow = json.loads(workflow_path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return [
            *findings,
            _finding(
                Severity.ERROR,
                "workflow.not_gui_format",
                f"{workflow_file} must parse as a GUI workflow: {exc}",
                workflow_file,
            ),
        ]
    if not isinstance(workflow, Mapping) or not isinstance(workflow.get("nodes"), list):
        return [
            *findings,
            _finding(
                Severity.ERROR,
                "workflow.not_gui_format",
                f"{workflow_file} must be a GUI workflow mapping with a nodes list.",
                workflow_file,
            ),
        ]
    node_by_id: dict[str, Mapping[str, object]] = {}
    for node in workflow["nodes"]:
        node_map = _as_mapping(node)
        if node_map is None or "id" not in node_map:
            continue
        node_by_id[str(node_map["id"])] = node_map

    findings.extend(
        _finding(
            Severity.ERROR,
            "workflow.missing_node_id",
            f"Apex requires workflow node id {node_id}.",
            workflow_file,
        )
        for node_id in sorted(_APEX_REQUIRED_WORKFLOW_NODE_IDS)
        if node_id not in node_by_id
    )
    for node_id, class_name in _APEX_WORKFLOW_NODE_CLASSES.items():
        node = node_by_id.get(node_id)
        if node is not None and node.get("type") != class_name:
            findings.append(
                _finding(
                    Severity.ERROR,
                    "workflow.node_class_mismatch",
                    f"Node {node_id} must be {class_name}, got {node.get('type')!r}.",
                    workflow_file,
                )
            )
    for node_id in ("3", "4"):
        node = node_by_id.get(node_id)
        if node is not None and node.get("type") not in _APEX_PROMPT_WIDGET_NODE_CLASSES:
            findings.append(
                _finding(
                    Severity.WARNING,
                    "workflow.prompt_key",
                    f"Node {node_id} ({node.get('type')!r}) is not mapped by Apex to a prompt widget.",
                    workflow_file,
                )
            )
    return findings


def _check_metadata(config: BundleConfig, bundle_path: Path) -> list[Finding]:
    findings: list[Finding] = []
    if config.metadata.version != bundle_path.name:
        findings.append(
            _finding(
                Severity.ERROR,
                "metadata.version_mismatch",
                f"metadata.version is {config.metadata.version!r}, but the resolved version directory is {bundle_path.name!r}.",
                _bundle_location(":metadata.version"),
            )
        )
    if not config.metadata.tested:
        findings.append(
            _finding(
                Severity.WARNING,
                "metadata.tested_false",
                "metadata.tested is false; this bundle is not marked as tested.",
                _bundle_location(":metadata.tested"),
            )
        )
    if config.readiness_marker is None:
        findings.append(
            _finding(
                Severity.WARNING,
                "readiness_marker.absent",
                "Apex will fall back to a 200-OK readiness probe.",
                _bundle_location(":readiness_marker"),
            )
        )
    return findings


def _matching_index_entries(
    bundle_name: str, index_entries: Sequence[Mapping[str, object]]
) -> list[Mapping[str, object]]:
    return [entry for entry in index_entries if entry.get("name") == bundle_name]


def _check_index(
    bundle_name: str,
    bundle_root: Path,
    index_entries: Sequence[Mapping[str, object]],
    *,
    all_bundles: bool,
) -> list[Finding]:
    findings: list[Finding] = []
    current = bundle_root / "current"
    try:
        current_ok = current.is_symlink() and current.resolve().exists()
    except (OSError, RuntimeError):
        current_ok = False
    if not current_ok:
        findings.append(
            _finding(
                Severity.ERROR,
                "index.current_symlink.missing",
                "Bundle root must contain a resolvable current symlink.",
                f"{bundle_root}:current",
            )
        )

    entries = _matching_index_entries(bundle_name, index_entries)
    if not entries:
        findings.append(
            _finding(
                Severity.ERROR,
                "index.entry.missing",
                f"bundle-index.yaml has no entry for {bundle_name!r}.",
                f"bundle-index.yaml:{bundle_name}",
            )
        )
    elif len(entries) > 1:
        findings.append(
            _finding(
                Severity.ERROR,
                "index.entry.duplicate",
                f"bundle-index.yaml has {len(entries)} entries for {bundle_name!r}; Apex resolves duplicates last-wins.",
                f"bundle-index.yaml:{bundle_name}",
            )
        )
    for entry in entries:
        model_type = entry.get("model_type")
        location = f"bundle-index.yaml:{bundle_name}"
        if model_type is None:
            findings.append(
                _finding(
                    Severity.ERROR,
                    "index.model_type.missing",
                    "Index entry must declare model_type.",
                    location,
                )
            )
        elif not isinstance(model_type, str) or model_type not in _APEX_MODEL_TYPES:
            findings.append(
                _finding(
                    Severity.ERROR,
                    "index.model_type.unknown",
                    f"Unknown Apex model_type {model_type!r}.",
                    location,
                )
            )
    if all_bundles:
        by_model_type: dict[str, int] = {}
        for entry in index_entries:
            model_type = entry.get("model_type")
            if entry.get("default_bundle") is True and isinstance(model_type, str):
                by_model_type[model_type] = by_model_type.get(model_type, 0) + 1
        findings.extend(
            _finding(
                Severity.ERROR,
                "index.default_bundle.duplicate",
                f"model_type {model_type!r} has {count} default_bundle entries.",
                "bundle-index.yaml",
            )
            for model_type, count in by_model_type.items()
            if count > 1 and any(entry.get("model_type") == model_type for entry in entries)
        )
    return findings


def check_bundle_contract(
    bundle_name: str,
    bundle_path: Path,
    raw_bundle: object,
    *,
    index_entries: Sequence[Mapping[str, object]] = (),
    all_bundles: bool = False,
    bundle_root: Path | None = None,
) -> ContractReport:
    """Return every static Apex-contract finding for one resolved bundle.

    `BundleConfig.model_validate` contributes a `schema.invalid` finding for
    malformed YAML. Semantic checks consume the original YAML mapping because
    Apex's parsing is intentionally stricter than Pydantic's coercion in a few
    critical places, so they continue even when schema validation fails.
    """
    raw = _as_mapping(raw_bundle)
    if raw is None:
        return ContractReport(
            bundle_name=bundle_name,
            findings=(
                _finding(
                    Severity.ERROR,
                    "schema.invalid",
                    "bundle.yaml must contain a YAML mapping.",
                    "bundle.yaml",
                ),
            ),
        )
    config: BundleConfig | None = None
    findings: list[Finding] = []
    try:
        config = BundleConfig.model_validate(raw)
    except ValidationError as exc:
        findings.append(_finding(Severity.ERROR, "schema.invalid", str(exc), "bundle.yaml"))

    root = bundle_root if bundle_root is not None else bundle_path.parent
    try:
        findings.extend(
            [
                *_check_hardware(raw),
                *_check_generation(raw),
                *_check_models(raw),
                *_check_index(bundle_name, root, index_entries, all_bundles=all_bundles),
            ]
        )
        if config is not None:
            findings.extend(
                [
                    *_check_workflow(bundle_path, config.workflow_file),
                    *_check_metadata(config, bundle_path),
                ]
            )
    except Exception as exc:
        findings = [
            _finding(
                Severity.ERROR,
                "contract.check_failed",
                f"Unable to complete static contract validation: {exc}",
                "bundle.yaml",
            )
        ]
    return ContractReport(bundle_name=bundle_name, findings=tuple(findings))
