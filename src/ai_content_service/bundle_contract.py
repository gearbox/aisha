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
from typing import TYPE_CHECKING, Final, cast
from urllib.parse import urlparse

from packaging.requirements import InvalidRequirement, Requirement
from pydantic import ValidationError

from .config import (
    BundleConfig,
    WorkflowMapConfig,
    WorkflowModelInputConfig,
    WorkflowRole,
    _validate_custom_node_name,
)

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
_WORKFLOW_SUBGRAPH_ID_RE: Final = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)

_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}")
_HARDWARE_INT_FIELDS: Final[tuple[str, ...]] = (
    "min_disk_gb",
    "min_network_upload_mbps",
    "min_network_download_mbps",
    "num_gpus",
)
# Never exported by ComfyUI's Graph -> Export (API): documentation/routing nodes
# that legitimately carry widgets_values with no inputs[] entries. Exclude these
# classes from node_missing_in_api, which would otherwise warn on every run forever.
#
# This is a best-effort list, not a closed set: ComfyUI decides what to omit from
# an API export via isVirtualNode, which any custom node can set, so the set of
# non-executable classes can never be fully enumerated here. It deliberately
# gates WARNING-level findings only -- never the terminal-node ERROR in
# _check_node_correspondence -- so an incomplete list can cause a missed
# warning but never a false build failure.
_NON_EXECUTABLE_GUI_CLASSES: Final[frozenset[str]] = frozenset(
    {"MarkdownNote", "Note", "Reroute", "PrimitiveNode"}
)


class Severity(str, Enum):
    """Severity of a static contract finding."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


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


def _check_gpu_whitelist_entries(whitelist: list[object]) -> list[Finding]:
    """Validate entries against Vast.ai's REST ``gpu_name`` values.

    Apex forwards these verbatim as ``{"gpu_name": {"in": [...]}}`` to
    ``POST /bundles/``, which matches the API's own space-separated values
    ("RTX 4090") -- the form every provisioning bundle uses. The underscore
    form belongs to the ``vastai search offers`` CLI query DSL, whose tokenizer
    splits on whitespace; it is warned about rather than rejected because no
    bundle has ever used it over REST, so its failure mode is inferred, not
    observed.
    """
    findings: list[Finding] = []
    for index, entry in enumerate(whitelist):
        location = _bundle_location(f":hardware.gpu_whitelist[{index}]")
        if not isinstance(entry, str):
            findings.append(
                _finding(
                    Severity.ERROR,
                    "hardware.gpu_whitelist.not_string",
                    f"Must be a string; got {type(entry).__name__}.",
                    location,
                )
            )
            continue
        if not entry.strip():
            findings.append(
                _finding(
                    Severity.ERROR,
                    "hardware.gpu_whitelist.blank",
                    "Must be a non-empty Vast.ai gpu_name value.",
                    location,
                )
            )
            continue
        if "_" in entry:
            findings.append(
                _finding(
                    Severity.ERROR,
                    "hardware.gpu_whitelist.underscore_name",
                    (
                        "Vast.ai's REST gpu_name values contain spaces "
                        f"({entry.replace('_', ' ')!r}); the underscore form is "
                        "vastai CLI query syntax."
                    ),
                    location,
                )
            )
            continue
        if entry != " ".join(entry.split()):
            findings.append(
                _finding(
                    Severity.ERROR,
                    "hardware.gpu_whitelist.not_normalized",
                    (
                        "Leading, trailing, or repeated whitespace is unlikely to "
                        f"match a Vast.ai gpu_name; use {' '.join(entry.split())!r}."
                    ),
                    location,
                )
            )
    return findings


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
        findings.extend(_check_gpu_whitelist_entries(whitelist))

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

    if "comfyui_port" in hardware:
        port = _as_int(hardware["comfyui_port"])
        if port is None:
            findings.append(
                _finding(
                    Severity.ERROR,
                    "hardware.comfyui_port.not_int",
                    "Must be an integer or an integer-valued string; booleans and floats are invalid.",
                    _bundle_location(":hardware.comfyui_port"),
                )
            )
        elif port != _APEX_DEFAULT_COMFYUI_PORT:
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
    elif not isinstance(base_image, str):
        findings.append(
            _finding(
                Severity.ERROR,
                "hardware.base_image.not_string",
                "When present, base_image must be a non-empty string.",
                _bundle_location(":hardware.base_image"),
            )
        )
    return findings


def _check_environment_pinning(raw: Mapping[str, object]) -> list[Finding]:
    """Warn when a bundle duplicates the template-owned base environment."""
    findings: list[Finding] = []
    hardware = _as_mapping(raw.get("hardware"))
    template_pinned = hardware is not None and hardware.get("template_hash_id") is not None
    comfyui_pinned = raw.get("comfyui") is not None
    lock_pinned = raw.get("requirements_lock_file") is not None
    overlay_pinned = raw.get("requirements_overlay_file") is not None

    if lock_pinned and overlay_pinned:
        findings.append(
            _finding(
                Severity.ERROR,
                "requirements.both_declared",
                (
                    "Declare only requirements_overlay_file or the deprecated "
                    "requirements_lock_file; deploying both is ambiguous."
                ),
                _bundle_location(),
            )
        )
    if template_pinned and (comfyui_pinned or lock_pinned):
        findings.append(
            _finding(
                Severity.WARNING,
                "environment.dual_pinning",
                (
                    "hardware.template_hash_id already pins the tested ComfyUI/CUDA/Python/base "
                    "package environment; bundle-level ComfyUI or the deprecated requirements lock "
                    "adds a second source of truth. Keep them only as a template escape hatch."
                ),
                _bundle_location(":hardware.template_hash_id"),
            )
        )
    if comfyui_pinned and not template_pinned:
        findings.append(
            _finding(
                Severity.WARNING,
                "comfyui.pinned_without_template",
                (
                    "Bundle pins ComfyUI but does not record hardware.template_hash_id, so the "
                    "tested CUDA, Python, and base-package environment is unknown."
                ),
                _bundle_location(":comfyui"),
            )
        )
    if lock_pinned:
        findings.append(
            _finding(
                Severity.WARNING,
                "requirements_lock.deprecated",
                (
                    "requirements_lock_file is deprecated; use requirements_overlay_file for the "
                    "additive dependencies captured against the base image. Inspect "
                    "requirements.lock.delta in the deploy log while migrating retained locks."
                ),
                _bundle_location(":requirements_lock_file"),
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
            url = file.get("url")
            if isinstance(url, str) and url and urlparse(url).scheme != "https":
                findings.append(
                    _finding(
                        Severity.ERROR,
                        "models.file.url_not_https",
                        "Model file URLs must use HTTPS; plain HTTP has no integrity or "
                        "confidentiality, and the digest check only catches a wrong fetch "
                        "after the fact.",
                        f"{location}.url",
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


_INEXACT_REGISTRY_VERSION_MARKERS: Final[tuple[str, ...]] = ("*", "^", "~", "<", ">", "=", ",", " ")


def _is_exact_registry_version(value: str) -> bool:
    """Return whether a registry version string names one immutable release.

    The Comfy Registry serves an installed archive at a bare version string
    (confirmed for comfyui-kjnodes: ``"1.5.0"``, no specifier syntax) -- there
    is no range or "latest" concept to resolve at deploy time the way a pip
    requirement has one. Anything that looks like a specifier or a moving
    target cannot be re-resolved to the archive that was actually tested.
    """
    stripped = value.strip()
    if not stripped or stripped.casefold() == "latest":
        return False
    return not any(marker in stripped for marker in _INEXACT_REGISTRY_VERSION_MARKERS)


def _check_custom_nodes(raw: Mapping[str, object]) -> list[Finding]:
    """Reject custom-node declarations Apex cannot deploy reproducibly.

    A node's own requirements.txt is installed from the file at deploy time;
    ``pip_requirements`` exists only for what the bundle author adds by hand,
    so it is validated with the same posture as model URLs and workflow node
    classes: reject what cannot work, at validate time, before a node is
    rented. Registry version pinning is validated with the same posture.

    Consumes ``raw``, not the parsed ``BundleConfig``, so these findings
    still surface when an unrelated field fails schema validation elsewhere
    in the bundle (see ``check_bundle_contract``'s docstring).
    """
    findings: list[Finding] = []
    custom_nodes = raw.get("custom_nodes")
    if not isinstance(custom_nodes, list):
        return findings
    for node_index, raw_node in enumerate(custom_nodes):
        node = _as_mapping(raw_node)
        if node is None:
            continue
        name = node.get("name")
        name_location = _bundle_location(f":custom_nodes[{node_index}].name")
        if not isinstance(name, str):
            findings.append(
                _finding(
                    Severity.ERROR,
                    "custom_node.name_invalid",
                    f"Custom node name must be a string; got {type(name).__name__}.",
                    name_location,
                )
            )
        else:
            try:
                _validate_custom_node_name(name)
            except ValueError as exc:
                findings.append(
                    _finding(
                        Severity.ERROR,
                        "custom_node.name_invalid",
                        str(exc),
                        name_location,
                    )
                )
        if node.get("source", "git") == "registry":
            version = node.get("version")
            if not isinstance(version, str) or not _is_exact_registry_version(version):
                findings.append(
                    _finding(
                        Severity.ERROR,
                        "custom_node.registry_unpinned",
                        (
                            f"custom node {name!r} declares source: registry with "
                            f"version {version!r}, which is not one exact, immutable "
                            "release. A Comfy Registry install must pin a specific "
                            "version string, not a range or 'latest'."
                        ),
                        _bundle_location(f":custom_nodes[{node_index}].version"),
                    )
                )
        pip_requirements = node.get("pip_requirements")
        if not isinstance(pip_requirements, list):
            continue
        for entry_index, entry in enumerate(pip_requirements):
            location = _bundle_location(
                f":custom_nodes[{node_index}].pip_requirements[{entry_index}]"
            )
            if not isinstance(entry, str):
                findings.append(
                    _finding(
                        Severity.ERROR,
                        "custom_nodes.pip_requirements.not_string",
                        f"Must be a string; got {type(entry).__name__}.",
                        location,
                    )
                )
                continue
            stripped = entry.strip()
            if stripped.startswith("-"):
                findings.append(
                    _finding(
                        Severity.ERROR,
                        "custom_nodes.pip_requirements.directive",
                        (
                            f"{entry!r} is a pip flag or -r/-c/-e directive, not a package "
                            f"requirement; node {name!r}'s own requirements.txt is already "
                            "installed from the file. Only additive packages belong here."
                        ),
                        location,
                    )
                )
                continue
            try:
                requirement = Requirement(stripped)
            except InvalidRequirement as exc:
                findings.append(
                    _finding(
                        Severity.ERROR,
                        "custom_nodes.pip_requirements.unparseable",
                        f"{entry!r} is not a valid PEP 508 requirement: {exc}",
                        location,
                    )
                )
                continue
            is_pinned = any(spec.operator == "==" for spec in requirement.specifier)
            if requirement.url is None and not is_pinned:
                findings.append(
                    _finding(
                        Severity.WARNING,
                        "custom_nodes.pip_requirements.unpinned",
                        f"{entry!r} has no == pin; the installed version can drift between deploys.",
                        location,
                    )
                )
    return findings


def _check_custom_node_pinned_to_head(raw: Mapping[str, object]) -> list[Finding]:  # noqa: ARG001
    """Flag a git custom node pinned by ``acs snapshot --pin-to-head`` (WARNING).

    ``--pin-to-head`` records its compromise -- a resolved SHA that is not
    necessarily the code that was tested (G4) -- only as a human-readable
    ``# TODO`` comment in bundle.yaml (see
    ``SnapshotManager._pin_to_head_bundle_comments``). YAML comments are
    discarded by ``yaml.safe_load`` before ``raw`` ever reaches this
    function, and the schema deliberately carries no machine-readable field
    for it (G6: aisha ships before any bundle declares a new key). This
    check therefore cannot detect the compromise from parsed bundle data and
    never fires today -- reparsing the comment text back out of bundle.yaml
    would be inferring a fact this module has no authoritative way to
    confirm, which is worse than not checking at all. It exists so that if a
    future schema revision records the pin source machine-readably, a
    validator is already in place to flag it.
    """
    return []


def _check_workflow(
    bundle_path: Path, workflow_file: str | None, *, has_workflow_map: bool = False
) -> list[Finding]:
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

    if not has_workflow_map:
        # A bundle that declares its own workflow: map must not be measured
        # against Apex's legacy hardcoded qwen.rapid.aio node ids/classes --
        # that graph shape is simply not what a mapped bundle carries.
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


def _is_valid_api_node(node_id: object, node: object) -> bool:
    return (
        isinstance(node_id, str)
        and isinstance(node, Mapping)
        and isinstance(node.get("class_type"), str)
        and isinstance(node.get("inputs"), Mapping)
    )


def is_api_workflow(value: object) -> bool:
    """Return whether ``value`` is the flat API graph shape we can inspect offline.

    Public: shared with ``snapshot`` to accept a converter response before it
    is written to disk.
    """
    if not isinstance(value, Mapping) or not value:
        return False
    return all(_is_valid_api_node(node_id, node) for node_id, node in value.items())


def _first_invalid_api_node(value: Mapping[str, object]) -> str | None:
    """Return the first node id that fails the flat API graph shape, for diagnostics."""
    return next(
        (str(node_id) for node_id, node in value.items() if not _is_valid_api_node(node_id, node)),
        None,
    )


def _load_api_workflow(
    bundle_path: Path, workflow_api_file: str
) -> tuple[Mapping[str, object] | None, list[Finding]]:
    """Load one API graph, retaining a precise missing-vs-malformed finding."""
    api_path = bundle_path / workflow_api_file
    if not api_path.is_file():
        return None, [
            _finding(
                Severity.ERROR,
                "workflow.api.missing",
                f"workflow_api_file {workflow_api_file!r} does not exist.",
                workflow_api_file,
            )
        ]
    try:
        api_graph = json.loads(api_path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, [
            _finding(
                Severity.ERROR,
                "workflow.api.malformed",
                f"{workflow_api_file} must be a JSON API graph: {exc}",
                workflow_api_file,
            )
        ]
    if not is_api_workflow(api_graph):
        bad_node = _first_invalid_api_node(api_graph) if isinstance(api_graph, Mapping) else None
        detail = f" First offending node: {bad_node!r}." if bad_node is not None else ""
        return None, [
            _finding(
                Severity.ERROR,
                "workflow.api.malformed",
                (
                    f"{workflow_api_file} must be a flat object of "
                    f"{{id: {{class_type, inputs}}}} nodes.{detail}"
                ),
                workflow_api_file,
            )
        ]
    return api_graph, []


def _api_link(value: object) -> list[object] | None:
    """Return an API ``[origin_id, output_slot]`` link, when this is one.

    ComfyUI's link shape is exact: a string origin node id and an integer
    output slot. Any other 2-element list (e.g. a ``[width, height]`` widget
    value) is data, not a link. ``bool`` is a subclass of ``int``, so it is
    excluded explicitly -- a boolean slot index is not a realistic input.
    """
    if not isinstance(value, list) or len(value) != 2:
        return None
    origin_id, output_slot = value
    if not isinstance(origin_id, str):
        return None
    if not isinstance(output_slot, int) or isinstance(output_slot, bool):
        return None
    return value


def _is_api_link(value: object) -> bool:
    """Return whether an API value is fed by an upstream link."""
    return _api_link(value) is not None


def check_api_graph_links(api_graph: Mapping[str, object], workflow_api_file: str) -> list[Finding]:
    """Require every API link origin to exist in the same graph and not be its own node.

    Public: shared with ``snapshot`` to reject a converter response with a
    dangling or self-referential link before it is written to disk.
    """
    findings: list[Finding] = []
    for node_id, raw_node in api_graph.items():
        node = _as_mapping(raw_node)
        inputs = _as_mapping(node.get("inputs")) if node is not None else None
        if inputs is None:
            continue
        for input_name, value in inputs.items():
            link = _api_link(value)
            if link is None:
                continue
            origin_id = link[0]
            if origin_id == node_id:
                message = (
                    f"Node {node_id} input {input_name!r} is self-referentially linked "
                    f"to ({origin_id!r}, {link[1]!r}) in the API graph."
                )
            elif origin_id not in api_graph:
                message = (
                    f"Node {node_id} input {input_name!r} is linked to ({origin_id!r}, "
                    f"{link[1]!r}), but origin node {origin_id!r} is absent from the API graph."
                )
            else:
                continue
            findings.append(
                _finding(
                    Severity.ERROR,
                    "workflow.api.dangling_link",
                    message,
                    workflow_api_file,
                )
            )
    return findings


def _resolved_model_input_filename(
    model_input: WorkflowModelInputConfig, config: BundleConfig
) -> str | None:
    """Resolve a validated model-input entry to the filename Apex will inject."""
    if model_input.filename is not None:
        return model_input.filename
    if model_input.model_type is None:
        return None
    groups = [model for model in config.models if model.model_type == model_input.model_type]
    if len(groups) == 1 and len(groups[0].files) == 1:
        return groups[0].files[0].filename
    return None


def _check_mapped_nodes(
    workflow: WorkflowMapConfig, api_graph: Mapping[str, object], api_file: str
) -> list[Finding]:
    """Check declared map nodes exist with the right class, and their inputs are writable."""
    findings: list[Finding] = []
    declared_nodes: list[tuple[str, str, str]] = [
        (f"nodes.{role.value}", node.id, node.class_) for role, node in workflow.nodes.items()
    ]
    declared_nodes.extend(
        (f"image_inputs[{index}]", node.id, node.class_)
        for index, node in enumerate(workflow.image_inputs)
    )
    declared_nodes.extend(
        (f"model_inputs[{index}]", node.id, node.class_)
        for index, node in enumerate(workflow.model_inputs)
    )

    for label, node_id, class_name in declared_nodes:
        api_node = _as_mapping(api_graph.get(node_id))
        if api_node is None:
            findings.append(
                _finding(
                    Severity.ERROR,
                    "workflow.map.node_missing",
                    f"Map {label} declares node id {node_id!r}, absent from the API graph.",
                    api_file,
                )
            )
            continue
        if api_node.get("class_type") != class_name:
            findings.append(
                _finding(
                    Severity.ERROR,
                    "workflow.map.class_mismatch",
                    (
                        f"Map {label} declares node {node_id} as {class_name!r}, but the API "
                        f"graph has {api_node.get('class_type')!r}."
                    ),
                    api_file,
                )
            )

    for role, node in workflow.nodes.items():
        api_node = _as_mapping(api_graph.get(node.id))
        if api_node is None:
            continue
        api_inputs = _as_mapping(api_node.get("inputs"))
        if api_inputs is None:
            continue
        for parameter, input_name in node.inputs.items():
            if input_name not in api_inputs:
                findings.append(
                    _finding(
                        Severity.ERROR,
                        "workflow.map.input_unknown",
                        (
                            f"Map nodes.{role.value}.{parameter} targets API input "
                            f"{input_name!r} on node {node.id}, but it does not exist."
                        ),
                        api_file,
                    )
                )
                continue
            if _is_api_link(api_inputs[input_name]):
                findings.append(
                    _finding(
                        Severity.ERROR,
                        "workflow.map.input_is_link",
                        (
                            f"Map nodes.{role.value}.{parameter} targets {input_name!r} on "
                            f"node {node.id}, but that API input is fed by an upstream link."
                        ),
                        api_file,
                    )
                )
    return findings


def _check_image_inputs(
    workflow: WorkflowMapConfig, api_graph: Mapping[str, object], api_file: str
) -> list[Finding]:
    """Check declared image inputs are linked from the positive-prompt node correctly."""
    findings: list[Finding] = []
    positive_node = workflow.nodes.get(WorkflowRole.POSITIVE_PROMPT)
    # ``positive_prompt`` is a required role. The explicit guard keeps this
    # function robust when called with a future/partially-constructed model.
    if positive_node is None:
        return findings
    api_positive = _as_mapping(api_graph.get(positive_node.id))
    api_positive_inputs = (
        _as_mapping(api_positive.get("inputs")) if api_positive is not None else None
    )
    if api_positive_inputs is None:
        return findings
    for index, image_input in enumerate(workflow.image_inputs):
        if image_input.target_input not in api_positive_inputs:
            findings.append(
                _finding(
                    Severity.ERROR,
                    "workflow.map.image_target_unknown",
                    (
                        f"Map image_inputs[{index}].target_input "
                        f"{image_input.target_input!r} is not an input on positive_prompt "
                        f"node {positive_node.id}."
                    ),
                    api_file,
                )
            )
            continue

        target_value = api_positive_inputs[image_input.target_input]
        target_link = _api_link(target_value)
        if target_link is None:
            findings.append(
                _finding(
                    Severity.ERROR,
                    "workflow.map.image_target_not_linked",
                    (
                        f"Map image_inputs[{index}].target_input "
                        f"{image_input.target_input!r} on positive_prompt node "
                        f"{positive_node.id} has scalar value {target_value!r}; it must be "
                        "fed by the declared LoadImage node."
                    ),
                    api_file,
                )
            )
            continue

        origin_id, output_slot = target_link
        # _api_link() guarantees these types; keeping the check local
        # makes the relationship explicit for this map-specific rule.
        if isinstance(origin_id, str) and origin_id != image_input.id:
            findings.append(
                _finding(
                    Severity.ERROR,
                    "workflow.map.image_target_wrong_origin",
                    (
                        f"Map image_inputs[{index}] declares LoadImage node "
                        f"{image_input.id!r}, but positive_prompt input "
                        f"{image_input.target_input!r} is linked from {origin_id!r}."
                    ),
                    api_file,
                )
            )
        if isinstance(output_slot, int) and output_slot != 0:
            findings.append(
                _finding(
                    Severity.INFO,
                    "workflow.map.image_target_slot",
                    (
                        f"Map image_inputs[{index}].target_input "
                        f"{image_input.target_input!r} uses output slot {output_slot} from "
                        f"LoadImage node {origin_id!r}; slot 1 is normally the mask output."
                    ),
                    api_file,
                )
            )
    return findings


def _check_model_inputs(
    workflow: WorkflowMapConfig,
    api_graph: Mapping[str, object],
    config: BundleConfig,
    api_file: str,
) -> list[Finding]:
    """Check declared model-loader inputs exist, are writable, and match scanned filenames."""
    findings: list[Finding] = []
    for index, model_input in enumerate(workflow.model_inputs):
        api_node = _as_mapping(api_graph.get(model_input.id))
        api_inputs = _as_mapping(api_node.get("inputs")) if api_node is not None else None
        if api_inputs is not None and model_input.input not in api_inputs:
            findings.append(
                _finding(
                    Severity.ERROR,
                    "workflow.map.input_unknown",
                    (
                        f"Map model_inputs[{index}].input targets API input "
                        f"{model_input.input!r} on node {model_input.id}, but it does not exist."
                    ),
                    api_file,
                )
            )
            continue
        if api_inputs is not None and _is_api_link(api_inputs[model_input.input]):
            findings.append(
                _finding(
                    Severity.ERROR,
                    "workflow.map.input_is_link",
                    (
                        f"Map model_inputs[{index}].input targets {model_input.input!r} on "
                        f"node {model_input.id}, but that API input is fed by an upstream link."
                    ),
                    api_file,
                )
            )
        expected_filename = _resolved_model_input_filename(model_input, config)
        if (
            api_inputs is not None
            and expected_filename is not None
            and model_input.input in api_inputs
            and api_inputs[model_input.input] != expected_filename
        ):
            findings.append(
                _finding(
                    Severity.WARNING,
                    "workflow.map.model_filename_stale",
                    (
                        f"Map model_inputs[{index}] expects {expected_filename!r}, but API node "
                        f"{model_input.id} has {api_inputs[model_input.input]!r}; Apex will overwrite it."
                    ),
                    api_file,
                )
            )
    return findings


def _check_workflow_map(
    config: BundleConfig,
    api_graph: Mapping[str, object] | None,
) -> list[Finding]:
    """Check a validated workflow map against its committed API graph."""
    api_findings = (
        check_api_graph_links(api_graph, config.workflow_api_file or "workflow.api.json")
        if api_graph is not None
        else []
    )
    if config.workflow is None:
        return [
            *api_findings,
            _finding(
                Severity.WARNING,
                "workflow.map.absent",
                "No workflow map is declared; Apex will use Qwen-shaped built-in defaults.",
                _bundle_location(":workflow"),
            ),
        ]
    if api_graph is None:
        return api_findings

    workflow: WorkflowMapConfig = config.workflow
    # Guaranteed non-None by BundleConfig.validate_workflow_references
    # whenever workflow is set; not a runtime fallback.
    api_file = cast("str", config.workflow_api_file)
    findings: list[Finding] = list(api_findings)
    findings.extend(_check_mapped_nodes(workflow, api_graph, api_file))
    findings.extend(_check_image_inputs(workflow, api_graph, api_file))
    findings.extend(_check_model_inputs(workflow, api_graph, config, api_file))
    return findings


def _gui_nodes_by_id(gui_graph: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    nodes = gui_graph.get("nodes")
    if not isinstance(nodes, list):
        return {}
    result: dict[str, Mapping[str, object]] = {}
    for raw_node in nodes:
        node = _as_mapping(raw_node)
        if node is not None and "id" in node:
            result[str(node["id"])] = node
    return result


def _gui_links_by_target_input(
    gui_graph: Mapping[str, object], gui_nodes: Mapping[str, Mapping[str, object]]
) -> dict[str, dict[str, tuple[str, object]]]:
    """Resolve GUI link records to ``target id -> input name -> origin``."""
    links = gui_graph.get("links")
    if not isinstance(links, list):
        return {}
    resolved: dict[str, dict[str, tuple[str, object]]] = {}
    for raw_link in links:
        if not isinstance(raw_link, list) or len(raw_link) < 5:
            continue
        target_id = str(raw_link[3])
        target_node = gui_nodes.get(target_id)
        if target_node is None:
            continue
        inputs = target_node.get("inputs")
        target_slot = raw_link[4]
        if not isinstance(inputs, list) or not isinstance(target_slot, int):
            continue
        if target_slot < 0 or target_slot >= len(inputs):
            continue
        input_config = _as_mapping(inputs[target_slot])
        input_name = input_config.get("name") if input_config is not None else None
        if not isinstance(input_name, str):
            continue
        resolved.setdefault(target_id, {})[input_name] = (str(raw_link[1]), raw_link[2])
    return resolved


def _widget_inputs(inputs: object) -> list[Mapping[str, object]]:
    """Return the Save-format input records that carry widget metadata."""
    if not isinstance(inputs, list):
        return []
    result: list[Mapping[str, object]] = []
    for raw_input in inputs:
        input_config = _as_mapping(raw_input)
        if input_config is not None and "widget" in input_config:
            result.append(input_config)
    return result


def _check_gui_structure(
    gui_graph: Mapping[str, object],
    gui_nodes: Mapping[str, Mapping[str, object]],
    workflow_file: str,
) -> list[Finding]:
    """Check GUI-only structure: unsupported subgraphs and widget-metadata capability."""
    findings: list[Finding] = []
    definitions = _as_mapping(gui_graph.get("definitions"))
    subgraphs = definitions.get("subgraphs") if definitions is not None else None
    if subgraphs:
        findings.append(
            _finding(
                Severity.ERROR,
                "workflow.sync.subgraph_unsupported",
                "GUI workflows with non-empty definitions.subgraphs are not supported.",
                workflow_file,
            )
        )
    for node_id, node in gui_nodes.items():
        node_type = node.get("type")
        if isinstance(node_type, str) and _WORKFLOW_SUBGRAPH_ID_RE.fullmatch(node_type):
            findings.append(
                _finding(
                    Severity.ERROR,
                    "workflow.sync.subgraph_unsupported",
                    f"GUI node {node_id} references subgraph type {node_type!r}.",
                    workflow_file,
                )
            )

    widget_metadata_node_ids = [
        node_id for node_id, node in gui_nodes.items() if _widget_inputs(node.get("inputs"))
    ]
    # Widget metadata only permits the optional GUI/API widget-value comparison.
    # Older Save files and modern Export files can be structurally identical here,
    # so format quality is a non-blocking capability signal, never an ERROR.
    if gui_nodes and not widget_metadata_node_ids:
        widget_value_node_ids = [
            node_id
            for node_id, node in gui_nodes.items()
            if isinstance(node.get("widgets_values"), list) and bool(node["widgets_values"])
        ]
        findings.append(
            _finding(
                Severity.WARNING,
                "workflow.sync.widget_metadata_absent",
                (
                    f"Widget metadata is absent from all {len(widget_value_node_ids)} GUI node(s) "
                    "carrying widget values; widget "
                    "values cannot be cross-checked against the API graph. Committing a Save "
                    "export from a current ComfyUI frontend enables drift detection."
                ),
                workflow_file,
            )
        )
    elif widget_metadata_node_ids:
        if inconsistent_node_ids := [
            node_id
            for node_id, node in gui_nodes.items()
            if isinstance(node.get("widgets_values"), list)
            and bool(node["widgets_values"])
            and isinstance(node.get("inputs"), list)
            and bool(node["inputs"])
            and not _widget_inputs(node.get("inputs"))
            and node.get("type") not in _NON_EXECUTABLE_GUI_CLASSES
        ]:
            findings.append(
                _finding(
                    Severity.WARNING,
                    "workflow.sync.widget_metadata_inconsistent",
                    (
                        "Widget metadata is present elsewhere in the GUI graph but absent for "
                        "node(s) with widget values: "
                        + ", ".join(inconsistent_node_ids)
                        + ". Widget values for those nodes cannot be cross-checked."
                    ),
                    workflow_file,
                )
            )
    return findings


def _gui_link_origins(links: object) -> set[str]:
    """Return the GUI node ids that originate at least one link.

    Reads the raw ``links`` array rather than per-node ``outputs[].links``,
    which is redundant data that a hand-edited graph can desynchronise from
    ``links``. A malformed entry (short tuple, non-list, ``None``) is skipped
    so a corrupted array degrades instead of raising.
    """
    if not isinstance(links, list):
        return set()
    origins: set[str] = set()
    for raw_link in links:
        if not isinstance(raw_link, list) or len(raw_link) < 2:
            continue
        origins.add(str(raw_link[1]))
    return origins


def _is_terminal_gui_node(node: Mapping[str, object], link_origins: set[str]) -> bool:
    """A node that consumes input and feeds nothing: an output of the graph."""
    inputs = node.get("inputs")
    has_incoming = isinstance(inputs, list) and any(
        isinstance(entry, Mapping) and entry.get("link") is not None for entry in inputs
    )
    return has_incoming and str(node.get("id")) not in link_origins


def _check_node_correspondence(
    api_graph: Mapping[str, object],
    gui_nodes: Mapping[str, Mapping[str, object]],
    mapped_ids: set[str],
    links: object,
    workflow_file: str,
    workflow_api_file: str,
) -> list[Finding]:
    """Check every node exists on both sides, is enabled, and agrees on class."""
    findings: list[Finding] = []
    link_origins = _gui_link_origins(links)
    for node_id, raw_api_node in api_graph.items():
        api_node = _as_mapping(raw_api_node)
        if api_node is None:
            continue
        gui_node = gui_nodes.get(node_id)
        if gui_node is None:
            findings.append(
                _finding(
                    Severity.ERROR,
                    "workflow.sync.node_missing_in_gui",
                    f"API node {node_id} is absent from the GUI workflow.",
                    workflow_api_file,
                )
            )
            continue
        gui_mode = gui_node.get("mode", 0)
        if gui_mode != 0:
            # Verified 2026-08-18 against the local ComfyUI v0.32 stack's converter
            # (comfyui-workflow-to-api-converter-endpoint/workflow_converter.py):
            # mode 2 is skipped outright as "muted", mode 4 is skipped but traced
            # through (trace_through_bypassed) to rewire its consumers' links --
            # i.e. mode 2 is mute and mode 4 is bypass, matching LiteGraph's
            # LGraphNode.mode enum (NEVER=2) with ComfyUI's bypass extension (4).
            if gui_mode == 2:
                state = "muted (mode=2); it should be absent from the API graph"
            elif gui_mode == 4:
                state = "bypassed (mode=4); its links should be rewired through it"
            else:
                state = f"disabled (mode={gui_mode!r}); it should not be executed"
            findings.append(
                _finding(
                    Severity.ERROR,
                    "workflow.sync.disabled_node_in_api",
                    f"API node {node_id} is present although its GUI counterpart is {state}.",
                    workflow_api_file,
                )
            )
        if gui_node.get("type") != api_node.get("class_type"):
            findings.append(
                _finding(
                    Severity.ERROR,
                    "workflow.sync.class_mismatch",
                    (
                        f"Node {node_id} is {gui_node.get('type')!r} in the GUI graph but "
                        f"{api_node.get('class_type')!r} in the API graph."
                    ),
                    workflow_api_file,
                )
            )

    for node_id, gui_node in gui_nodes.items():
        if (
            node_id in api_graph
            or gui_node.get("mode", 0) != 0
            or gui_node.get("type") in _NON_EXECUTABLE_GUI_CLASSES
        ):
            continue
        # A terminal node -- one with an incoming link and no outgoing link -- is an
        # output of the graph; dropping it silently produces a workflow that runs
        # but yields nothing (R23). Terminality is structural, not class-based
        # (R24), because isVirtualNode is extensible by custom nodes and a
        # hardcoded output-class list has the same unbounded-set problem.
        is_terminal = _is_terminal_gui_node(gui_node, link_origins)
        severity = Severity.ERROR if node_id in mapped_ids or is_terminal else Severity.WARNING
        if is_terminal:
            message = (
                f"Executable GUI node {node_id} ({gui_node.get('type')}) is a terminal "
                "output node but is absent from the API graph; the converted workflow "
                "produces no output."
            )
        else:
            message = f"Executable GUI node {node_id} is absent from the API graph."
        findings.append(
            _finding(
                severity,
                "workflow.sync.node_missing_in_api",
                message,
                workflow_file,
            )
        )
    return findings


def _check_link_and_value_sync(
    api_graph: Mapping[str, object],
    gui_nodes: Mapping[str, Mapping[str, object]],
    gui_links: Mapping[str, dict[str, tuple[str, object]]],
    workflow_file: str,
    workflow_api_file: str,
) -> list[Finding]:
    """Check every API link matches its GUI counterpart, and widget values agree."""
    findings: list[Finding] = []
    for node_id, raw_api_node in api_graph.items():
        api_node = _as_mapping(raw_api_node)
        if api_node is None:
            continue
        gui_node = gui_nodes.get(node_id)
        if gui_node is None:
            # Already reported by _check_node_correspondence's node_missing_in_gui.
            continue
        api_inputs = _as_mapping(api_node.get("inputs"))
        if api_inputs is None:
            continue
        expected_links = gui_links.get(node_id, {})
        for input_name, api_value in api_inputs.items():
            api_link = _api_link(api_value)
            if api_link is None:
                continue
            expected = expected_links.get(input_name)
            actual = (str(api_link[0]), api_link[1])
            if expected != actual:
                findings.append(
                    _finding(
                        Severity.ERROR,
                        "workflow.sync.link_mismatch",
                        (
                            f"Node {node_id} input {input_name!r} is linked to {actual!r} in "
                            f"the API graph but {expected!r} in the GUI graph."
                        ),
                        workflow_api_file,
                    )
                )

    unaligned: list[str] = []
    for node_id, gui_node in gui_nodes.items():
        api_node = _as_mapping(api_graph.get(node_id))
        # A node absent from the API graph is already reported by
        # node_missing_in_api above; unaligned is reserved for the coverage
        # guarantee that the value check silently becoming a no-op is caught.
        if api_node is None:
            continue
        inputs = gui_node.get("inputs")
        widget_inputs = _widget_inputs(inputs)
        widget_values = gui_node.get("widgets_values")
        values = widget_values if isinstance(widget_values, list) else []
        if len(widget_inputs) != len(values):
            unaligned.append(
                f"{node_id} ({gui_node.get('type', 'unknown')}: "
                f"{len(widget_inputs)} widget inputs, {len(values)} widget values)"
            )
            continue
        api_inputs = _as_mapping(api_node.get("inputs"))
        if api_inputs is None:
            continue
        for raw_input, gui_value in zip(widget_inputs, values, strict=True):
            widget_input_name = raw_input.get("name")
            if not isinstance(widget_input_name, str):
                continue
            if widget_input_name not in api_inputs:
                findings.append(
                    _finding(
                        Severity.WARNING,
                        "workflow.sync.input_missing_in_api",
                        (
                            f"Node {node_id} input {widget_input_name!r} is {gui_value!r} in the "
                            "GUI graph but absent from the API graph; this is expected for an "
                            "optional widget the export omitted."
                        ),
                        workflow_api_file,
                    )
                )
                continue
            api_value = api_inputs[widget_input_name]
            if _is_api_link(api_value):
                continue
            if api_value != gui_value:
                findings.append(
                    _finding(
                        Severity.ERROR,
                        "workflow.sync.value_mismatch",
                        (
                            f"Node {node_id} input {widget_input_name!r} is {gui_value!r} in the GUI "
                            f"graph but {api_value!r} in the API graph."
                        ),
                        workflow_api_file,
                    )
                )
    if unaligned:
        findings.append(
            _finding(
                Severity.INFO,
                "workflow.sync.unaligned_nodes",
                "Skipped widget-value comparison for " + "; ".join(unaligned) + ".",
                workflow_file,
            )
        )
    return findings


def check_workflow_sync(
    gui_graph: Mapping[str, object],
    api_graph: Mapping[str, object],
    *,
    workflow_file: str = "workflow.json",
    workflow_api_file: str = "workflow.api.json",
    workflow_map: WorkflowMapConfig | None = None,
) -> list[Finding]:
    """Compare committed GUI Save and API graph files without ComfyUI access.

    This is public enough for snapshot conversion to validate a response before
    it writes ``workflow.api.json``. Contract validation calls the same helper,
    so the snapshot and CI never drift in what they consider a synchronized pair.
    """
    gui_nodes = _gui_nodes_by_id(gui_graph)
    gui_links = _gui_links_by_target_input(gui_graph, gui_nodes)
    links = gui_graph.get("links")
    mapped_ids = (
        {node.id for node in workflow_map.nodes.values()}
        | {node.id for node in workflow_map.image_inputs}
        | {node.id for node in workflow_map.model_inputs}
        if workflow_map is not None
        else set()
    )

    findings: list[Finding] = []
    findings.extend(_check_gui_structure(gui_graph, gui_nodes, workflow_file))
    findings.extend(
        _check_node_correspondence(
            api_graph, gui_nodes, mapped_ids, links, workflow_file, workflow_api_file
        )
    )
    findings.extend(
        _check_link_and_value_sync(
            api_graph, gui_nodes, gui_links, workflow_file, workflow_api_file
        )
    )
    return findings


def _check_workflow_sync(
    bundle_path: Path,
    config: BundleConfig,
    api_graph: Mapping[str, object] | None,
) -> list[Finding]:
    """Load the GUI graph then dispatch Family 2 when both graph files exist.

    An unreadable/malformed GUI graph returns ``[]`` here rather than a
    finding of its own -- ``_check_workflow``'s unconditional
    ``workflow.not_gui_format`` already reports it. If that check is ever
    removed or narrowed, this silent return becomes a real hole.
    """
    if config.workflow_api_file is None or api_graph is None or config.workflow_file is None:
        return []
    try:
        gui_graph = json.loads((bundle_path / config.workflow_file).read_text())
    except (OSError, UnicodeError, json.JSONDecodeError):
        return []
    if not isinstance(gui_graph, Mapping) or not isinstance(gui_graph.get("nodes"), list):
        return []
    return check_workflow_sync(
        gui_graph,
        api_graph,
        workflow_file=config.workflow_file,
        workflow_api_file=config.workflow_api_file,
        workflow_map=config.workflow,
    )


def _workflow_node_classes(bundle_path: Path, workflow_file: str | None) -> tuple[str, ...]:
    """Return the GUI workflow's unique class names when it can be read safely."""
    if workflow_file is None:
        return ()
    try:
        workflow = json.loads((bundle_path / workflow_file).read_text())
    except (OSError, UnicodeError, json.JSONDecodeError):
        return ()
    if not isinstance(workflow, Mapping) or not isinstance(workflow.get("nodes"), list):
        return ()
    return tuple(
        sorted(
            {
                node_type
                for raw_node in workflow["nodes"]
                if (node := _as_mapping(raw_node)) is not None
                and isinstance((node_type := node.get("type")), str)
            }
        )
    )


def custom_node_directory(python_module: object) -> str | None:
    """Return a custom-node directory from ComfyUI ``python_module`` metadata.

    ``/object_info`` is ComfyUI's authoritative class-to-provider map.  Keep
    this small extraction public so snapshot authoring and bundle validation
    cannot drift on how that map attributes a class to a custom-node directory.
    """
    if not isinstance(python_module, str) or not python_module.startswith("custom_nodes."):
        return None
    directory = python_module.removeprefix("custom_nodes.").split(".", 1)[0]
    return directory or None


def _check_workflow_class_providers(
    bundle_path: Path,
    config: BundleConfig,
    object_info: Mapping[str, object],
) -> list[Finding]:
    """Compare workflow classes with ComfyUI's live provider metadata.

    ComfyUI, rather than a vendored core-node list, is authoritative for
    ``python_module``. This deliberately only rejects custom-node classes that
    the bundle failed to declare; non-custom providers are outside this
    bundle's ownership.
    """
    workflow_file = config.workflow_file or _APEX_WORKFLOW_FILENAME
    declared_nodes = {node.name.casefold() for node in config.custom_nodes}
    findings: list[Finding] = []
    for class_name in _workflow_node_classes(bundle_path, workflow_file):
        class_info = _as_mapping(object_info.get(class_name))
        if class_info is None:
            findings.append(
                _finding(
                    Severity.ERROR,
                    "workflow.class_unknown",
                    f"ComfyUI /object_info has no class named {class_name!r}.",
                    workflow_file,
                )
            )
            continue
        python_module = class_info.get("python_module")
        if not isinstance(python_module, str) or not python_module:
            findings.append(
                _finding(
                    Severity.INFO,
                    "workflow.class_provider_unknown",
                    (
                        f"ComfyUI did not report python_module for {class_name!r}; "
                        "the bundle provider check was skipped for this class."
                    ),
                    workflow_file,
                )
            )
            continue
        directory = custom_node_directory(python_module)
        if directory is not None and directory.casefold() not in declared_nodes:
            findings.append(
                _finding(
                    Severity.ERROR,
                    "workflow.class_unprovided",
                    (
                        f"Workflow class {class_name!r} is provided by custom node "
                        f"{directory!r}, but bundle.yaml does not declare it."
                    ),
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
    object_info: Mapping[str, object] | None = None,
    workflow_provider_check: bool = False,
) -> ContractReport:
    """Return every static Apex-contract finding for one resolved bundle.

    `BundleConfig.model_validate` contributes a `schema.invalid` finding for
    malformed YAML. Semantic checks consume the original YAML mapping because
    Apex's parsing is intentionally stricter than Pydantic's coercion in a few
    critical places, so they continue even when schema validation fails. When
    supplied, ``object_info`` adds a best-effort live provider check using the
    running ComfyUI instance as the source of truth.
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
        # Workflow-map structure is deliberately a deployment-stopping schema
        # gate: config stays None below, so no graph check runs against an
        # object whose map may be incoherent. But every semantic check below
        # consumes ``raw``, not ``config``, and is already guarded by
        # ``if config is not None`` where it needs the parsed model -- so a
        # schema error here must never suppress those unrelated findings.
        if any(error["loc"][:1] == ("custom_nodes",) for error in exc.errors()):
            check = "custom_node.source_fields_invalid"
        elif "workflow" in raw or "workflow_api_file" in raw:
            check = "bundle.config_invalid"
        else:
            check = "schema.invalid"
        findings.append(_finding(Severity.ERROR, check, str(exc), "bundle.yaml"))

    root = bundle_root if bundle_root is not None else bundle_path.parent
    try:
        findings.extend(
            [
                *_check_hardware(raw),
                *_check_environment_pinning(raw),
                *_check_generation(raw),
                *_check_models(raw),
                *_check_custom_nodes(raw),
                *_check_custom_node_pinned_to_head(raw),
                *_check_index(bundle_name, root, index_entries, all_bundles=all_bundles),
            ]
        )
        if config is not None:
            api_graph: Mapping[str, object] | None = None
            api_findings: list[Finding] = []
            if config.workflow_api_file is not None:
                api_graph, api_findings = _load_api_workflow(bundle_path, config.workflow_api_file)
            findings.extend(
                [
                    *_check_workflow(
                        bundle_path,
                        config.workflow_file,
                        has_workflow_map=config.workflow is not None,
                    ),
                    *api_findings,
                    *_check_workflow_map(config, api_graph),
                    *_check_workflow_sync(bundle_path, config, api_graph),
                    *_check_metadata(config, bundle_path),
                ]
            )
            if object_info is not None:
                findings.extend(_check_workflow_class_providers(bundle_path, config, object_info))
            elif workflow_provider_check:
                findings.append(
                    _finding(
                        Severity.INFO,
                        "workflow.class_provider_check_skipped",
                        "No ComfyUI URL supplied; skipped live workflow class provider validation.",
                        config.workflow_file or _APEX_WORKFLOW_FILENAME,
                    )
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
