"""Inference of a bundle's workflow map from a converted API graph."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal

from pydantic import ValidationError

from .config import (
    WORKFLOW_CONTRACT_VERSION,
    ModelConfig,
    WorkflowMapConfig,
    WorkflowMedia,
    WorkflowMediaInputConfig,
    WorkflowMediaSlot,
    WorkflowModelInputConfig,
    WorkflowNodeConfig,
    WorkflowRole,
)
from .workflow_semantics import MEDIA_LOADER_SPECS, WorkflowMediaKind, allowed_media_slots

_WORKFLOW_COMMENT_MAX_LENGTH: Final = 200
_SAMPLER_CLASSES: Final[frozenset[str]] = frozenset(
    {"KSampler", "KSamplerAdvanced", "SamplerCustom", "SamplerCustomAdvanced"}
)
_NON_PROMPT_CONDITIONING: Final[frozenset[str]] = frozenset({"ConditioningZeroOut"})
# Input names and slot indices verified against a running ComfyUI's
# /object_info for each class, e.g.:
#   curl -s http://127.0.0.1:8188/object_info/ControlNetApplyAdvanced | python -m json.tool
# Confirm new entries the same way before adding them.
_CONDITIONING_PASSTHROUGH: Final[dict[tuple[str, int], tuple[str, ...]]] = {
    ("ConditioningCombine", 0): ("conditioning_1", "conditioning_2"),
    ("ConditioningConcat", 0): ("conditioning_to", "conditioning_from"),
    ("ConditioningSetArea", 0): ("conditioning",),
    ("ConditioningSetAreaPercentage", 0): ("conditioning",),
    ("ConditioningSetMask", 0): ("conditioning",),
    ("ConditioningSetTimestepRange", 0): ("conditioning",),
    ("ControlNetApply", 0): ("conditioning",),
    ("ControlNetApplyAdvanced", 0): ("positive",),
    ("ControlNetApplyAdvanced", 1): ("negative",),
    ("ControlNetApplySD3", 0): ("positive",),
    ("ControlNetApplySD3", 1): ("negative",),
}
_MULTI_SOURCE_CONDITIONING: Final[frozenset[str]] = frozenset(
    {"ConditioningCombine", "ConditioningConcat"}
)
# Comfortably covers realistic conditioning-adapter chains (combine -> concat
# -> set-area -> ControlNet, etc.) while still bounding traversal of a
# malformed or cyclic graph. Exceeding it is reported as an explicit
# depth_exhausted TODO in the trace, never a silent truncation.
_MAX_CONDITIONING_PASSTHROUGH_DEPTH: Final = 8
_LOADER_BY_MODEL_TYPE: Final[dict[str, tuple[str, str]]] = {
    "checkpoints": ("CheckpointLoaderSimple", "ckpt_name"),
    "diffusion_models": ("UNETLoader", "unet_name"),
    "text_encoders": ("CLIPLoader", "clip_name"),
    "vae": ("VAELoader", "vae_name"),
    "loras": ("LoraLoaderModelOnly", "lora_name"),
    "upscale_models": ("UpscaleModelLoader", "model_name"),
    "clip_vision": ("CLIPVisionLoader", "clip_name"),
}


def _build_model_type_by_loader() -> dict[str, tuple[str, str]]:
    """Invert the bindable-loader table without silently losing a model type."""
    loader_classes = [loader_class for loader_class, _input in _LOADER_BY_MODEL_TYPE.values()]
    if len(loader_classes) != len(set(loader_classes)):
        raise AssertionError(
            "_LOADER_BY_MODEL_TYPE must use a distinct loader class for every model type"
        )
    return {
        loader_class: (model_type, loader_input)
        for model_type, (loader_class, loader_input) in _LOADER_BY_MODEL_TYPE.items()
    }


_MODEL_TYPE_BY_LOADER: Final[dict[str, tuple[str, str]]] = _build_model_type_by_loader()
_PARAM_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    "width": ("width",),
    "height": ("height",),
    "batch_size": ("batch_size",),
    "text": ("text", "prompt"),
    "seed": ("seed", "noise_seed"),
    "steps": ("steps",),
    "cfg": ("cfg", "guidance"),
    "sampler": ("sampler_name", "sampler"),
    "scheduler": ("scheduler",),
    "denoise": ("denoise",),
    "filename_prefix": ("filename_prefix",),
    "length": ("length", "num_frames", "frames", "video_frames"),
    "fps": ("fps", "frame_rate"),
    "format": ("format", "container"),
    "shift": ("shift",),
}
_ROLE_PARAMETERS: Final[dict[WorkflowRole, tuple[str, ...]]] = {
    WorkflowRole.LATENT: ("width", "height", "batch_size", "length"),
    WorkflowRole.POSITIVE_PROMPT: ("text",),
    WorkflowRole.NEGATIVE_PROMPT: ("text",),
    WorkflowRole.SAMPLER: ("seed", "steps", "cfg", "sampler", "scheduler", "denoise"),
    WorkflowRole.MODEL_SAMPLING: ("shift",),
    WorkflowRole.SAVE: ("filename_prefix", "fps", "format"),
    WorkflowRole.PREVIEW: (),
}


def _workflow_comment_source(value: str | BaseException) -> str:
    """Return normalized, unbounded comment text before the final length limit."""
    if isinstance(value, ValidationError):
        details: list[str] = []
        for error in value.errors():
            raw_location = error.get("loc")
            location = (
                ".".join(str(part) for part in raw_location)
                if isinstance(raw_location, tuple)
                else ""
            )
            message = error.get("msg")
            if isinstance(message, str):
                details.append(f"{location}: {message}" if location else message)
        source = "; ".join(details) or str(value)
    else:
        source = str(value)

    normalized = " ".join(source.strip().split())
    return normalized.encode("ascii", "backslashreplace").decode("ascii")


def normalize_workflow_comment(value: str | BaseException) -> str:
    """Render generated diagnostics as compact, ASCII, one-line YAML comments."""
    ascii_normalized = _workflow_comment_source(value)
    if len(ascii_normalized) <= _WORKFLOW_COMMENT_MAX_LENGTH:
        return ascii_normalized
    return f"{ascii_normalized[: _WORKFLOW_COMMENT_MAX_LENGTH - 3].rstrip()}..."


def _api_mapping(value: object) -> Mapping[str, object] | None:
    return value if isinstance(value, Mapping) else None


def _api_link_origin_and_slot(value: object) -> tuple[str, int] | None:
    """Return the origin node id for a resolved API link, if this is one.

    Mirrors ``bundle_contract._api_link``'s exact shape check: a string
    origin id and an integer (non-bool) output slot, not any 2-element list.
    """
    if not isinstance(value, list) or len(value) != 2:
        return None
    origin_id, output_slot = value
    if not isinstance(origin_id, str):
        return None
    if not isinstance(output_slot, int) or isinstance(output_slot, bool):
        return None
    return origin_id, output_slot


def _api_link_origin(value: object) -> str | None:
    """Return the origin node id for a resolved API link, if this is one."""
    link = _api_link_origin_and_slot(value)
    return link[0] if link is not None else None


def _node_inputs(api_graph: Mapping[str, object], node_id: str) -> Mapping[str, object] | None:
    node = _api_mapping(api_graph.get(node_id))
    return _api_mapping(node.get("inputs")) if node is not None else None


def _infer_role_inputs(role: WorkflowRole, api_inputs: Mapping[str, object]) -> dict[str, str]:
    """Map only scalar API inputs that match the role's closed vocabulary."""
    inferred: dict[str, str] = {}
    for parameter in _ROLE_PARAMETERS[role]:
        for alias in _PARAM_ALIASES[parameter]:
            if alias in api_inputs and _api_link_origin(api_inputs[alias]) is None:
                inferred[parameter] = alias
                break
    return inferred


@dataclass(frozen=True, slots=True)
class _ConditioningTrace:
    """Result of following one conditioning output through known adapters."""

    node_id: str
    traversed_classes: tuple[str, ...]
    depth_exhausted: bool = False
    ambiguous_class: str | None = None
    ambiguous_origins: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True, slots=True)
class UnresolvedMediaInput:
    """A recognized upload edge that cannot yet become a deployable contract.

    The graph tells us which loader reaches which mapped role, but it cannot
    safely tell us the request capability that Apex must expose.  Keep that
    fact as data rather than relying on the accompanying authoring comment.
    """

    loader_id: str
    loader_class: str
    kind: WorkflowMediaKind
    target_role: WorkflowRole
    target_input: str
    reason: Literal["slot_required", "unsupported_output_slot", "media_incompatible"]
    output_slot: int


@dataclass(frozen=True, slots=True)
class _MediaInference:
    """The certifiable and unresolved parts of recognized media inference."""

    media_inputs: tuple[WorkflowMediaInputConfig, ...]
    unresolved: tuple[UnresolvedMediaInput, ...]


@dataclass(frozen=True, slots=True)
class WorkflowMapInferenceResult:
    """The inferred map plus authoring guidance and certification blockers."""

    workflow_map: WorkflowMapConfig | None
    comments: tuple[str, ...]
    unresolved_media_inputs: tuple[UnresolvedMediaInput, ...]


def _trace_conditioning_passthrough(
    api_graph: Mapping[str, object], node_id: str, output_slot: int = 0
) -> _ConditioningTrace:
    """Follow a conditioning output back to its source encoder.

    A malformed graph can contain a conditioning cycle. Keep traversal bounded
    and return its last reachable node in that case. Multi-source adapters are
    reported to the caller instead of choosing one input silently.
    """
    current_id = node_id
    current_slot = output_slot
    visited_ids: set[str] = set()
    traversed_classes: list[str] = []
    for _ in range(_MAX_CONDITIONING_PASSTHROUGH_DEPTH):
        if current_id in visited_ids:
            return _ConditioningTrace(current_id, tuple(traversed_classes))
        visited_ids.add(current_id)
        node = _api_mapping(api_graph.get(current_id))
        class_name = node.get("class_type") if node is not None else None
        if not isinstance(class_name, str):
            return _ConditioningTrace(current_id, tuple(traversed_classes))
        input_names = _CONDITIONING_PASSTHROUGH.get((class_name, current_slot))
        if input_names is None:
            return _ConditioningTrace(current_id, tuple(traversed_classes))
        inputs = _api_mapping(node.get("inputs")) if node is not None else None
        if inputs is None:
            return _ConditioningTrace(current_id, tuple(traversed_classes))
        candidates = [
            link
            for input_name in input_names
            if (link := _api_link_origin_and_slot(inputs.get(input_name))) is not None
        ]
        if len(candidates) > 1 and class_name in _MULTI_SOURCE_CONDITIONING:
            return _ConditioningTrace(
                current_id,
                tuple(traversed_classes),
                ambiguous_class=class_name,
                ambiguous_origins=tuple(candidates),
            )
        if not candidates:
            return _ConditioningTrace(current_id, tuple(traversed_classes))
        traversed_classes.append(class_name)
        current_id, current_slot = candidates[0]
    return _ConditioningTrace(
        current_id,
        tuple(traversed_classes),
        depth_exhausted=True,
    )


def _node_class_name(api_graph: Mapping[str, object], node_id: str) -> str:
    """Return a node class for author-facing diagnostics without raising."""
    node = _api_mapping(api_graph.get(node_id))
    class_name = node.get("class_type") if node is not None else None
    return class_name if isinstance(class_name, str) else "unknown"


def _workflow_node(
    api_graph: Mapping[str, object], node_id: str, role: WorkflowRole, comments: list[str]
) -> WorkflowNodeConfig | None:
    """Return a validated node config, or None with a TODO naming why.

    Every construction here must degrade to a comment, never raise: a
    malformed-but-parseable graph (blank class_type, an empty-string node id
    key) must still let ``infer_workflow_map`` finish and hand the operator an
    actionable map instead of a traceback.
    """
    node = _api_mapping(api_graph.get(node_id))
    if node is None:
        return None
    class_name = node.get("class_type")
    inputs = _api_mapping(node.get("inputs"))
    if not isinstance(class_name, str) or inputs is None:
        return None
    try:
        return WorkflowNodeConfig.model_validate(
            {
                "id": node_id,
                "class": class_name,
                "inputs": _infer_role_inputs(role, inputs),
            }
        )
    except ValueError as exc:
        error_text = _workflow_comment_source(exc)
        comments.append(
            normalize_workflow_comment(
                f"TODO: node {node_id} failed workflow map validation: {error_text}"
            )
        )
        return None


def _resolve_prompt_role(
    api_graph: Mapping[str, object],
    link: tuple[str, int],
    role: WorkflowRole,
    comments: list[str],
) -> str | None:
    """Trace a conditioning chain to a writable text encoder for one prompt role.

    The ambiguity, depth-exhaustion and traversal-note messages are shaped
    identically between roles and derived from ``role.value``. The
    unwritable-text failure is worded, and reacted to, differently per role:
    fatal for the required positive_prompt, informational for the optional
    negative_prompt, since an absent negative role is a valid, shippable map
    (zit ships that way). Either way the caller -- not this helper -- decides
    what a ``None`` return means.
    """
    node_id, output_slot = link
    trace = _trace_conditioning_passthrough(api_graph, node_id, output_slot)
    if trace.ambiguous_class is not None:
        candidates = ", ".join(
            f"{origin_id} (slot {origin_slot})"
            for origin_id, origin_slot in trace.ambiguous_origins
        )
        comments.append(
            normalize_workflow_comment(
                f"TODO: {role.value} cannot be inferred through multi-source "
                f"{trace.ambiguous_class}; candidate origins: {candidates}"
            )
        )
        return None

    resolved_id = trace.node_id
    if trace.depth_exhausted:
        comments.append(
            normalize_workflow_comment(
                f"TODO: {role.value} conditioning trace reached maximum depth "
                f"{_MAX_CONDITIONING_PASSTHROUGH_DEPTH} at node {resolved_id}"
            )
        )

    resolved_inputs = _node_inputs(api_graph, resolved_id)
    if resolved_inputs is None or "text" not in _infer_role_inputs(role, resolved_inputs):
        if role is WorkflowRole.POSITIVE_PROMPT:
            comments.append(
                normalize_workflow_comment(
                    f"TODO: {role.value} resolved to node {resolved_id} "
                    f"({_node_class_name(api_graph, resolved_id)}) but no writable text "
                    "input was found"
                )
            )
        else:
            comments.append(
                normalize_workflow_comment(
                    f"{role.value} omitted: node {resolved_id} "
                    f"({_node_class_name(api_graph, resolved_id)}) has no writable text input"
                )
            )
        return None

    if trace.traversed_classes:
        comments.append(
            normalize_workflow_comment(
                f"{role.value} traced through "
                f"{', '.join(trace.traversed_classes)} to node {resolved_id} "
                f"({_node_class_name(api_graph, resolved_id)})"
            )
        )
    return resolved_id


def _infer_sampler(
    api_graph: Mapping[str, object], comments: list[str]
) -> tuple[str, Mapping[str, object], tuple[str, int], str] | None:
    """Find the unique sampler node and its required positive/latent links.

    Returns ``(sampler_id, sampler_inputs, positive_link, latent_id)``, or
    leaves a TODO and returns None if the sampler or either required link is
    missing. Tracing the positive link to a writable encoder happens
    separately, after this existence check.
    """
    # Every sorted(...) of node ids in this module is lexicographic, not
    # numeric -- harmless because each is only used for a length check or to
    # pick the sole match when exactly one exists.
    sampler_ids = sorted(
        node_id
        for node_id, raw_node in api_graph.items()
        if (node := _api_mapping(raw_node)) is not None
        and node.get("class_type") in _SAMPLER_CLASSES
    )
    if len(sampler_ids) != 1:
        comments.append(
            normalize_workflow_comment(
                "TODO: identify exactly one sampler node before emitting workflow map "
                f"(found {len(sampler_ids)})"
            )
        )
        return None

    sampler_id = sampler_ids[0]
    sampler_inputs = _node_inputs(api_graph, sampler_id)
    if sampler_inputs is None:
        comments.append(
            normalize_workflow_comment(f"TODO: sampler node {sampler_id} has no API inputs")
        )
        return None

    positive_link = _api_link_origin_and_slot(sampler_inputs.get("positive"))
    if positive_link is None:
        comments.append(
            normalize_workflow_comment(
                f"TODO: sampler node {sampler_id} has no linked positive conditioning input"
            )
        )
        return None

    latent_id = _api_link_origin(sampler_inputs.get("latent_image")) or _api_link_origin(
        sampler_inputs.get("latent")
    )
    if latent_id is None:
        comments.append(
            normalize_workflow_comment(
                f"TODO: sampler node {sampler_id} has no linked latent input"
            )
        )
        return None
    return sampler_id, sampler_inputs, positive_link, latent_id


def _infer_negative_prompt(
    api_graph: Mapping[str, object], sampler_inputs: Mapping[str, object], comments: list[str]
) -> str | None:
    """Resolve the sampler's optional negative conditioning role, if any."""
    negative_link = _api_link_origin_and_slot(sampler_inputs.get("negative"))
    if negative_link is None:
        return None
    negative_node_id = negative_link[0]
    negative_raw = _api_mapping(api_graph.get(negative_node_id))
    negative_class = negative_raw.get("class_type") if negative_raw is not None else None
    if isinstance(negative_class, str) and negative_class in _NON_PROMPT_CONDITIONING:
        comments.append(
            normalize_workflow_comment(
                f"negative_prompt omitted: sampler negative is supplied by {negative_class}"
            )
        )
        return None
    return _resolve_prompt_role(api_graph, negative_link, WorkflowRole.NEGATIVE_PROMPT, comments)


def _infer_output_roles(
    api_graph: Mapping[str, object],
    nodes: dict[WorkflowRole, WorkflowNodeConfig],
    comments: list[str],
) -> None:
    """Add the optional save/preview roles to ``nodes`` in place, when unambiguous."""
    save_ids = sorted(
        node_id
        for node_id, raw_node in api_graph.items()
        if (node := _api_mapping(raw_node)) is not None and node.get("class_type") == "SaveImage"
    )
    if len(save_ids) == 1:
        save = _workflow_node(api_graph, save_ids[0], WorkflowRole.SAVE, comments)
        if save is not None:
            nodes[WorkflowRole.SAVE] = save
    elif len(save_ids) > 1:
        comments.append(
            normalize_workflow_comment(
                f"TODO: identify one SaveImage node before emitting save role (found {len(save_ids)})"
            )
        )

    preview_ids = sorted(
        node_id
        for node_id, raw_node in api_graph.items()
        if (node := _api_mapping(raw_node)) is not None and node.get("class_type") == "PreviewImage"
    )
    if len(preview_ids) == 1:
        preview = _workflow_node(api_graph, preview_ids[0], WorkflowRole.PREVIEW, comments)
        if preview is not None:
            nodes[WorkflowRole.PREVIEW] = preview


def _is_scalar(value: object) -> bool:
    """Return whether an API value is a literal rather than structured graph data."""
    return value is None or isinstance(value, (str, int, float, bool))


def _model_chain_origins(
    api_graph: Mapping[str, object], sampler_inputs: Mapping[str, object]
) -> tuple[str, ...]:
    """Walk upstream through the sampler's model-bearing links.

    ComfyUI model adapters conventionally expose their upstream model under
    ``model`` (and a few model-sampling nodes use ``unet``).  Restricting the
    traversal to those named inputs avoids wandering into unrelated control or
    conditioning links while retaining architecture-independent shift support.
    """
    sampler_model = _api_link_origin(sampler_inputs.get("model"))
    if sampler_model is None:
        return ()

    chain: list[str] = []
    current_id = sampler_model
    visited: set[str] = set()
    while current_id not in visited:
        visited.add(current_id)
        chain.append(current_id)
        inputs = _node_inputs(api_graph, current_id)
        if inputs is None:
            break
        next_id = _api_link_origin(inputs.get("model")) or _api_link_origin(inputs.get("unet"))
        if next_id is None:
            break
        current_id = next_id
    return tuple(chain)


def _infer_model_sampling_role(
    api_graph: Mapping[str, object],
    sampler_inputs: Mapping[str, object],
    nodes: dict[WorkflowRole, WorkflowNodeConfig],
    comments: list[str],
) -> None:
    """Bind a unique scalar ``shift`` found upstream of the sampler model."""
    candidates: list[str] = []
    for node_id in _model_chain_origins(api_graph, sampler_inputs):
        inputs = _node_inputs(api_graph, node_id)
        if inputs is not None and "shift" in inputs and _is_scalar(inputs["shift"]):
            candidates.append(node_id)

    if len(candidates) > 1:
        rendered = ", ".join(
            f"{node_id} ({_node_class_name(api_graph, node_id)})" for node_id in candidates
        )
        comments.append(
            normalize_workflow_comment(
                "TODO: model_sampling shift is exposed by multiple model-chain nodes; "
                f"choose one: {rendered}"
            )
        )
        return
    if len(candidates) != 1:
        return

    model_sampling = _workflow_node(api_graph, candidates[0], WorkflowRole.MODEL_SAMPLING, comments)
    if model_sampling is not None and "shift" in model_sampling.inputs:
        nodes[WorkflowRole.MODEL_SAMPLING] = model_sampling


def _infer_media_inputs(
    api_graph: Mapping[str, object],
    nodes: Mapping[WorkflowRole, WorkflowNodeConfig],
    media: WorkflowMedia,
    comments: list[str],
) -> _MediaInference:
    """Infer recognized loaders linked to any addressable workflow role.

    A video's first/last/source semantics cannot be recovered from a graph
    edge.  Leave unresolved loaders out of the machine-readable map and
    preserve the exact certification blocker alongside the authoring TODO.
    """
    media_inputs: list[WorkflowMediaInputConfig] = []
    unresolved: list[UnresolvedMediaInput] = []
    seen_loaders: set[str] = set()
    for target_role in sorted(nodes, key=lambda role: role.value):
        target_node = nodes[target_role]
        target_inputs = _node_inputs(api_graph, target_node.id) or {}
        for target_input in sorted(target_inputs):
            value = target_inputs[target_input]
            origin_link = _api_link_origin_and_slot(value)
            if origin_link is None:
                continue
            origin_id, output_slot = origin_link
            origin = _api_mapping(api_graph.get(origin_id))
            class_name = origin.get("class_type") if origin is not None else None
            if not isinstance(class_name, str):
                continue
            loader = MEDIA_LOADER_SPECS.get(class_name)
            if loader is None:
                continue
            if output_slot not in loader.output_slots:
                unresolved.append(
                    UnresolvedMediaInput(
                        loader_id=origin_id,
                        loader_class=class_name,
                        kind=loader.kind,
                        target_role=target_role,
                        target_input=target_input,
                        reason="unsupported_output_slot",
                        output_slot=output_slot,
                    )
                )
                comments.append(
                    normalize_workflow_comment(
                        f"TODO: media loader node {origin_id} ({class_name}) feeds "
                        f"target_role: {target_role.value}, target_input: {target_input} from "
                        f"unsupported output slot {output_slot}"
                    )
                )
                continue
            allowed_slots = allowed_media_slots(graph_media=media, kind=loader.kind)
            if not allowed_slots:
                unresolved.append(
                    UnresolvedMediaInput(
                        loader_id=origin_id,
                        loader_class=class_name,
                        kind=loader.kind,
                        target_role=target_role,
                        target_input=target_input,
                        reason="media_incompatible",
                        output_slot=output_slot,
                    )
                )
                comments.append(
                    normalize_workflow_comment(
                        f"TODO: media loader node {origin_id} ({class_name}) feeds "
                        f"target_role: {target_role.value}, target_input: {target_input}; "
                        f"{media.value} workflows have no legal {loader.kind.value} upload slot"
                    )
                )
                continue
            if media is WorkflowMedia.VIDEO:
                choices = ", ".join(slot.value for slot in allowed_slots)
                unresolved.append(
                    UnresolvedMediaInput(
                        loader_id=origin_id,
                        loader_class=class_name,
                        kind=loader.kind,
                        target_role=target_role,
                        target_input=target_input,
                        reason="slot_required",
                        output_slot=output_slot,
                    )
                )
                comments.append(
                    normalize_workflow_comment(
                        f"TODO: media loader node {origin_id} ({class_name}) feeds "
                        f"target_role: {target_role.value}, target_input: {target_input}; "
                        f"choose its required slot ({choices})"
                    )
                )
                continue
            if origin_id in seen_loaders:
                comments.append(
                    normalize_workflow_comment(
                        f"TODO: media loader node {origin_id} feeds multiple role inputs; "
                        "the first target is the deterministic representative for this uploaded asset"
                    )
                )
                continue
            seen_loaders.add(origin_id)
            try:
                media_inputs.append(
                    WorkflowMediaInputConfig.model_validate(
                        {
                            "id": origin_id,
                            "class": class_name,
                            "input": loader.input_name,
                            "kind": loader.kind,
                            "slot": WorkflowMediaSlot.REFERENCE,
                            "target_role": target_role,
                            "target_input": target_input,
                        }
                    )
                )
            except ValueError as exc:
                error_text = _workflow_comment_source(exc)
                comments.append(
                    normalize_workflow_comment(
                        f"TODO: media input node {origin_id} failed workflow map validation: "
                        f"{error_text}"
                    )
                )
    return _MediaInference(tuple(media_inputs), tuple(unresolved))


def _infer_model_inputs(
    api_graph: Mapping[str, object], models: list[ModelConfig], comments: list[str]
) -> list[WorkflowModelInputConfig]:
    """Infer writable model-loader bindings from the API graph itself.

    The graph is authoritative for which loader values Apex can replace.
    Scanned ``models`` merely refines an otherwise complete binding with a
    filename when a declared group has more than one file.
    """
    model_inputs: list[WorkflowModelInputConfig] = []
    loader_classes_in_graph: set[str] = set()
    for loader_id in sorted(api_graph):
        raw_node = _api_mapping(api_graph[loader_id])
        if raw_node is None:
            continue
        raw_class_name = raw_node.get("class_type")
        if not isinstance(raw_class_name, str):
            continue
        class_name = raw_class_name
        loader = _MODEL_TYPE_BY_LOADER.get(class_name)
        if loader is None:
            continue
        loader_classes_in_graph.add(class_name)
        model_type, loader_input = loader
        loader_values = _api_mapping(raw_node.get("inputs"))
        if loader_values is None:
            comments.append(
                normalize_workflow_comment(f"TODO: loader node {loader_id} has no API inputs")
            )
            continue
        if loader_input not in loader_values:
            comments.append(
                normalize_workflow_comment(
                    f"TODO: loader node {loader_id} has no {loader_input!r} API input"
                )
            )
            continue
        baked_value = loader_values.get(loader_input)
        if _api_link_origin(baked_value) is not None:
            continue

        model_input_kwargs: dict[str, object] = {
            "id": loader_id,
            "class": class_name,
            "input": loader_input,
            "model_type": model_type,
        }
        matching_groups = [group for group in models if group.model_type == model_type]
        if not matching_groups:
            comments.append(
                normalize_workflow_comment(
                    f"TODO: node {loader_id!r} loads {baked_value!r}; confirm the models group "
                    "has exactly one file, or add `filename` explicitly."
                )
            )
        else:
            matching_filename_groups = [
                group
                for group in matching_groups
                if isinstance(baked_value, str)
                and baked_value in {file.filename for file in group.files}
            ]
            selected_group = (
                matching_filename_groups[0]
                if len(matching_filename_groups) == 1
                else matching_groups[0]
                if len(matching_groups) == 1
                else None
            )
            # Mirrors BundleConfig.validate_workflow_references: `filename`
            # disambiguates both across groups of one model_type and across
            # files within one group. Keep the two rules in lockstep.
            needs_filename = (
                True
                if selected_group is None
                else len(matching_groups) != 1 or len(selected_group.files) != 1
            )
            if needs_filename:
                filenames = (
                    {file.filename for file in selected_group.files}
                    if selected_group is not None
                    else set()
                )
                if not isinstance(baked_value, str) or baked_value not in filenames:
                    comments.append(
                        normalize_workflow_comment(
                            f"TODO: select a filename for {model_type!r} loader node {loader_id}"
                        )
                    )
                    continue
                model_input_kwargs["filename"] = baked_value

        try:
            model_inputs.append(WorkflowModelInputConfig.model_validate(model_input_kwargs))
        except ValueError as exc:
            error_text = _workflow_comment_source(exc)
            comments.append(
                normalize_workflow_comment(
                    f"TODO: model input node {loader_id} failed workflow map validation: "
                    f"{error_text}"
                )
            )

    for model_type in sorted({model.model_type for model in models}):
        loader = _LOADER_BY_MODEL_TYPE.get(model_type)
        if loader is None:
            continue
        loader_class, _loader_input = loader
        if loader_class not in loader_classes_in_graph:
            comments.append(
                normalize_workflow_comment(
                    f"TODO: model_type {model_type!r} is declared but its loader class "
                    f"{loader_class!r} does not appear in the API graph"
                )
            )
    return model_inputs


def infer_workflow_map(
    api_graph: Mapping[str, object],
    models: list[ModelConfig],
    *,
    media: WorkflowMedia = WorkflowMedia.IMAGE,
) -> WorkflowMapInferenceResult:
    """Infer a workflow map, its authoring guidance, and any media blockers.

    API exports resolve links and carry named inputs, so inference never needs
    a frontend widget-order table.  Comments guide a human author; unresolved
    media inputs are the separate machine-readable incomplete state.
    """
    comments: list[str] = []

    sampler_context = _infer_sampler(api_graph, comments)
    if sampler_context is None:
        return WorkflowMapInferenceResult(None, tuple(comments), ())
    sampler_id, sampler_inputs, positive_link, latent_id = sampler_context

    positive_id = _resolve_prompt_role(
        api_graph, positive_link, WorkflowRole.POSITIVE_PROMPT, comments
    )
    if positive_id is None:
        return WorkflowMapInferenceResult(None, tuple(comments), ())

    sampler = _workflow_node(api_graph, sampler_id, WorkflowRole.SAMPLER, comments)
    positive = _workflow_node(api_graph, positive_id, WorkflowRole.POSITIVE_PROMPT, comments)
    latent = _workflow_node(api_graph, latent_id, WorkflowRole.LATENT, comments)
    if sampler is None or positive is None or latent is None:
        comments.append(
            normalize_workflow_comment(
                "TODO: required workflow nodes are missing class_type or inputs metadata"
            )
        )
        return WorkflowMapInferenceResult(None, tuple(comments), ())

    nodes: dict[WorkflowRole, WorkflowNodeConfig] = {
        WorkflowRole.LATENT: latent,
        WorkflowRole.POSITIVE_PROMPT: positive,
        WorkflowRole.SAMPLER: sampler,
    }

    _infer_model_sampling_role(api_graph, sampler_inputs, nodes, comments)

    negative_id = _infer_negative_prompt(api_graph, sampler_inputs, comments)
    if negative_id is not None:
        negative = _workflow_node(api_graph, negative_id, WorkflowRole.NEGATIVE_PROMPT, comments)
        if negative is None:
            comments.append(
                normalize_workflow_comment(
                    f"TODO: negative conditioning node {negative_id} is missing API metadata"
                )
            )
        else:
            nodes[WorkflowRole.NEGATIVE_PROMPT] = negative

    _infer_output_roles(api_graph, nodes, comments)

    media_inference = _infer_media_inputs(api_graph, nodes, media, comments)
    model_inputs = _infer_model_inputs(api_graph, models, comments)

    try:
        return WorkflowMapInferenceResult(
            WorkflowMapConfig(
                contract_version=WORKFLOW_CONTRACT_VERSION,
                media=media,
                nodes=nodes,
                media_inputs=list(media_inference.media_inputs),
                model_inputs=model_inputs,
            ),
            tuple(comments),
            media_inference.unresolved,
        )
    except ValueError as exc:
        error_text = _workflow_comment_source(exc)
        comments.append(
            normalize_workflow_comment(
                f"TODO: inferred workflow map is structurally invalid: {error_text}"
            )
        )
        return WorkflowMapInferenceResult(None, tuple(comments), media_inference.unresolved)
