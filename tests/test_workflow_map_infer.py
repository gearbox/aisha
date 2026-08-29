"""Coverage for inferring a workflow map from a converted API graph."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import pytest
import yaml

from ai_content_service import workflow_map as workflow_map_module
from ai_content_service.config import (
    BundleConfig,
    ModelConfig,
    ModelFileConfig,
    WorkflowMapConfig,
    WorkflowMedia,
    WorkflowRole,
)
from ai_content_service.snapshot import _render_bundle_yaml
from ai_content_service.workflow_map import infer_workflow_map
from ai_content_service.workflow_semantics import (
    MEDIA_LOADER_SPECS,
    WorkflowMediaKind,
    allowed_media_slots,
)
from tests.workflow_map_helpers import _api_inputs, _raw_bundle

if TYPE_CHECKING:
    from collections.abc import Mapping


def _infer_workflow_map(
    api_graph: Mapping[str, object],
    models: list[ModelConfig],
    *,
    media: WorkflowMedia = WorkflowMedia.IMAGE,
) -> tuple[WorkflowMapConfig | None, tuple[str, ...]]:
    """Keep legacy assertions concise while the production result carries blockers."""
    result = infer_workflow_map(api_graph, models, media=media)
    return result.workflow_map, result.comments


@pytest.mark.parametrize(
    ("graph_media", "kind", "expected"),
    (
        (WorkflowMedia.IMAGE, WorkflowMediaKind.IMAGE, ("reference",)),
        (WorkflowMedia.IMAGE, WorkflowMediaKind.VIDEO, ()),
        (WorkflowMedia.VIDEO, WorkflowMediaKind.IMAGE, ("reference", "first_frame", "last_frame")),
        (WorkflowMedia.VIDEO, WorkflowMediaKind.VIDEO, ("source",)),
    ),
)
def test_allowed_media_slots_match_graph_and_uploaded_asset_semantics(
    graph_media: WorkflowMedia,
    kind: WorkflowMediaKind,
    expected: tuple[str, ...],
) -> None:
    assert (
        tuple(slot.value for slot in allowed_media_slots(graph_media=graph_media, kind=kind))
        == expected
    )


@pytest.mark.parametrize(
    ("graph_media", "loader_class", "kind"),
    (
        (WorkflowMedia.IMAGE, "LoadImage", WorkflowMediaKind.IMAGE),
        (WorkflowMedia.VIDEO, "LoadImage", WorkflowMediaKind.IMAGE),
        (WorkflowMedia.VIDEO, "LoadVideo", WorkflowMediaKind.VIDEO),
    ),
)
def test_allowed_media_slot_guidance_is_accepted_by_the_typed_contract(
    graph_media: WorkflowMedia,
    loader_class: str,
    kind: WorkflowMediaKind,
) -> None:
    raw = _raw_bundle()
    workflow = raw["workflow"]
    assert isinstance(workflow, dict)
    workflow["media"] = graph_media.value
    workflow["media_inputs"] = [
        {
            "id": index + 4,
            "class": loader_class,
            "input": MEDIA_LOADER_SPECS[loader_class].input_name,
            "kind": kind.value,
            "slot": slot.value,
            "target_input": f"media_{slot.value}",
        }
        for index, slot in enumerate(allowed_media_slots(graph_media=graph_media, kind=kind))
    ]

    assert BundleConfig.model_validate(raw).workflow is not None


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

    workflow_map, comments = _infer_workflow_map(api, [model])

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


def _model_loader_graph(
    *,
    loader_class: str = "CheckpointLoaderSimple",
    loader_input: str = "ckpt_name",
    baked_value: object = "model.safetensors",
) -> dict[str, object]:
    """Return a small complete graph with one otherwise independent model loader."""
    return {
        "9": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "cat"}},
        "2": {
            "class_type": "KSampler",
            "inputs": {"positive": ["3", 0], "latent_image": ["9", 0], "steps": 8},
        },
        "1": {"class_type": loader_class, "inputs": {loader_input: baked_value}},
    }


def _model_group(
    model_type: str,
    filenames: list[str],
    *,
    subdirectory: str | None = None,
) -> ModelConfig:
    return ModelConfig(
        name=model_type,
        model_type=model_type,
        subdirectory=subdirectory,
        files=[ModelFileConfig(name=filename, url="", filename=filename) for filename in filenames],
    )


def test_model_loader_inference_is_graph_driven_when_the_scan_is_empty() -> None:
    workflow_map, comments = _infer_workflow_map(_model_loader_graph(), [])

    assert workflow_map is not None
    assert [
        entry.model_dump(mode="json", by_alias=True) for entry in workflow_map.model_inputs
    ] == [
        {
            "id": "1",
            "class": "CheckpointLoaderSimple",
            "input": "ckpt_name",
            "model_type": "checkpoints",
            "filename": None,
            "label": None,
        }
    ]
    assert any("confirm the models group has exactly one file" in comment for comment in comments)
    assert (
        BundleConfig.model_validate(
            {**_raw_bundle(), "workflow": workflow_map.model_dump(mode="json", by_alias=True)}
        ).workflow
        is not None
    )


def test_model_loader_inference_uses_models_only_to_refine_filename() -> None:
    single = _model_group("checkpoints", ["model.safetensors"])
    multiple = _model_group("checkpoints", ["other.safetensors", "model.safetensors"])

    single_map, single_comments = _infer_workflow_map(_model_loader_graph(), [single])
    multiple_map, multiple_comments = _infer_workflow_map(_model_loader_graph(), [multiple])

    assert single_map is not None and multiple_map is not None
    assert single_map.model_inputs[0].filename is None
    assert not any("confirm the models group" in comment for comment in single_comments)
    assert multiple_map.model_inputs[0].filename == "model.safetensors"
    assert not multiple_comments


def test_model_loader_inference_requires_a_selection_for_an_unmatched_multi_file_group() -> None:
    workflow_map, comments = _infer_workflow_map(
        _model_loader_graph(),
        [_model_group("checkpoints", ["other.safetensors", "third.safetensors"])],
    )

    assert workflow_map is not None
    assert workflow_map.model_inputs == []
    assert any("select a filename" in comment for comment in comments)


def test_model_loader_inference_skips_linked_loader_values() -> None:
    workflow_map, comments = _infer_workflow_map(
        _model_loader_graph(baked_value=["upstream", 0]), []
    )

    assert workflow_map is not None
    assert workflow_map.model_inputs == []
    assert not comments


def test_model_loader_inference_handles_multiple_groups_by_baked_filename() -> None:
    api = _model_loader_graph(
        loader_class="UNETLoader",
        loader_input="unet_name",
        baked_value="cyberrealisticZImage_v70.safetensors",
    )
    first = _model_group("diffusion_models", ["other.safetensors"], subdirectory="first")
    matching = _model_group(
        "diffusion_models",
        ["cyberrealisticZImage_v70.safetensors", "another.safetensors"],
        subdirectory="second",
    )

    workflow_map, comments = _infer_workflow_map(api, [first, matching])

    assert workflow_map is not None
    assert workflow_map.model_inputs[0].model_type == "diffusion_models"
    assert workflow_map.model_inputs[0].filename == "cyberrealisticZImage_v70.safetensors"
    assert not comments


def test_model_loader_inference_disambiguates_a_matching_group_with_one_file() -> None:
    """R1: two groups, baked filename matches a single-file group -- filename must be emitted.

    Regression guard for S1: when the matching group happens to hold exactly
    one file, ``len(selected_group.files) != 1`` used to be false on its own
    and the disambiguating `filename` was silently dropped, producing a map
    that fails BundleConfig.validate_workflow_references.
    """
    api = _model_loader_graph(
        loader_class="UNETLoader",
        loader_input="unet_name",
        baked_value="cyberrealisticZImage_v70.safetensors",
    )
    first = _model_group("diffusion_models", ["other.safetensors"], subdirectory="first")
    matching = _model_group(
        "diffusion_models", ["cyberrealisticZImage_v70.safetensors"], subdirectory="second"
    )

    workflow_map, comments = _infer_workflow_map(api, [first, matching])

    assert workflow_map is not None
    assert workflow_map.model_inputs[0].model_type == "diffusion_models"
    assert workflow_map.model_inputs[0].filename == "cyberrealisticZImage_v70.safetensors"
    assert not comments
    full_config = BundleConfig.model_validate(
        {
            **_raw_bundle(),
            "models": [group.model_dump(mode="json", by_alias=True) for group in (first, matching)],
            "workflow": workflow_map.model_dump(mode="json", by_alias=True),
        }
    )
    assert full_config.workflow is not None


def test_model_loader_inference_disambiguates_by_filename_within_a_multi_file_group() -> None:
    """R2: two groups, baked filename matches a multi-file group -- filename emitted."""
    api = _model_loader_graph(
        loader_class="UNETLoader",
        loader_input="unet_name",
        baked_value="cyberrealisticZImage_v70.safetensors",
    )
    first = _model_group("diffusion_models", ["other.safetensors"], subdirectory="first")
    matching = _model_group(
        "diffusion_models",
        ["cyberrealisticZImage_v70.safetensors", "another.safetensors"],
        subdirectory="second",
    )

    workflow_map, comments = _infer_workflow_map(api, [first, matching])

    assert workflow_map is not None
    assert workflow_map.model_inputs[0].filename == "cyberrealisticZImage_v70.safetensors"
    assert not comments
    full_config = BundleConfig.model_validate(
        {
            **_raw_bundle(),
            "models": [group.model_dump(mode="json", by_alias=True) for group in (first, matching)],
            "workflow": workflow_map.model_dump(mode="json", by_alias=True),
        }
    )
    assert full_config.workflow is not None


def test_model_loader_inference_declines_when_baked_filename_matches_no_group() -> None:
    """R3: two groups, baked filename matches neither -- no entry, TODO emitted."""
    api = _model_loader_graph(
        loader_class="UNETLoader", loader_input="unet_name", baked_value="unmatched.safetensors"
    )
    first = _model_group("diffusion_models", ["other.safetensors"], subdirectory="first")
    second = _model_group("diffusion_models", ["another.safetensors"], subdirectory="second")

    workflow_map, comments = _infer_workflow_map(api, [first, second])

    assert workflow_map is not None
    assert workflow_map.model_inputs == []
    assert any("select a filename" in comment for comment in comments)


def test_model_loader_inference_omits_filename_for_a_single_one_file_group() -> None:
    """R4: one group with one file -- filename omitted, as today (regression guard for R1)."""
    workflow_map, comments = _infer_workflow_map(
        _model_loader_graph(), [_model_group("checkpoints", ["model.safetensors"])]
    )

    assert workflow_map is not None
    assert workflow_map.model_inputs[0].filename is None
    assert not any("confirm the models group" in comment for comment in comments)


def test_model_loader_inference_always_emits_a_schema_valid_map_across_group_counts() -> None:
    """R5: exhaustive check that inference never emits a schema-invalid map.

    Exercises every group count from 1 to 3 extra groups (2-4 total), for
    both a checkpoint-style and a diffusion-model-style loader (standing in
    for the qwen and zit fixtures), with the target group placed first and
    last to catch any group-order sensitivity in selection.
    """
    cases = (
        ("CheckpointLoaderSimple", "ckpt_name", "checkpoints"),
        ("UNETLoader", "unet_name", "diffusion_models"),
    )
    for loader_class, loader_input, model_type in cases:
        baked_value = "target.safetensors"
        api = _model_loader_graph(
            loader_class=loader_class, loader_input=loader_input, baked_value=baked_value
        )
        target_single = _model_group(model_type, [baked_value], subdirectory="target_single")
        target_multi = _model_group(
            model_type, [baked_value, "sibling.safetensors"], subdirectory="target_multi"
        )
        for target_group in (target_single, target_multi):
            for extra_count in (1, 2, 3):
                extras = [
                    _model_group(model_type, [f"extra{i}.safetensors"], subdirectory=f"extra{i}")
                    for i in range(extra_count)
                ]
                for models in ([target_group, *extras], [*extras, target_group]):
                    workflow_map, _comments = _infer_workflow_map(api, models)
                    assert workflow_map is not None, (loader_class, extra_count, models)
                    full_config = BundleConfig.model_validate(
                        {
                            **_raw_bundle(),
                            "models": [
                                group.model_dump(mode="json", by_alias=True) for group in models
                            ],
                            "workflow": workflow_map.model_dump(mode="json", by_alias=True),
                        }
                    )
                    assert full_config.workflow is not None


def test_declared_model_group_without_a_loader_is_an_advisory_comment() -> None:
    workflow_map, comments = _infer_workflow_map(
        _model_loader_graph(), [_model_group("vae", ["model.vae.safetensors"])]
    )

    assert workflow_map is not None
    assert any("VAELoader" in comment and "does not appear" in comment for comment in comments)


def test_reverse_model_loader_table_rejects_duplicate_loader_classes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        workflow_map_module,
        "_LOADER_BY_MODEL_TYPE",
        {"one": ("SharedLoader", "one"), "two": ("SharedLoader", "two")},
    )

    with pytest.raises(AssertionError, match="distinct loader class"):
        workflow_map_module._build_model_type_by_loader()


def test_model_sampling_role_is_inferred_from_a_scalar_shift_in_the_model_chain() -> None:
    api = _model_loader_graph(loader_class="UNETLoader", loader_input="unet_name")
    sampler = _api_inputs(api, "2")
    sampler["model"] = ["73", 0]
    api["73"] = {
        "class_type": "ModelSamplingSD3",
        "inputs": {"model": ["1", 0], "shift": 3},
    }

    workflow_map, comments = _infer_workflow_map(api, [])

    assert workflow_map is not None
    assert workflow_map.nodes[WorkflowRole.MODEL_SAMPLING].id == "73"
    assert workflow_map.nodes[WorkflowRole.MODEL_SAMPLING].inputs == {"shift": "shift"}
    assert any("confirm the models group" in comment for comment in comments)


def test_model_sampling_role_is_omitted_without_shift_or_when_ambiguous() -> None:
    api = _model_loader_graph(loader_class="UNETLoader", loader_input="unet_name")
    sampler = _api_inputs(api, "2")
    sampler["model"] = ["73", 0]
    api["73"] = {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["72", 0], "shift": 3}}
    api["72"] = {"class_type": "ModelSamplingFlux", "inputs": {"model": ["1", 0], "shift": 2}}

    ambiguous_map, ambiguous_comments = _infer_workflow_map(api, [])
    no_shift = copy.deepcopy(api)
    _api_inputs(no_shift, "73").pop("shift")
    _api_inputs(no_shift, "72").pop("shift")
    no_shift_map, no_shift_comments = _infer_workflow_map(no_shift, [])

    assert ambiguous_map is not None
    assert WorkflowRole.MODEL_SAMPLING not in ambiguous_map.nodes
    assert any("73" in comment and "72" in comment for comment in ambiguous_comments)
    assert no_shift_map is not None
    assert WorkflowRole.MODEL_SAMPLING not in no_shift_map.nodes
    assert not any("model_sampling" in comment for comment in no_shift_comments)


def test_image_inference_emits_reference_media_input_on_its_actual_target_role() -> None:
    api = {
        "9": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024}},
        "3": {
            "class_type": "TextEncodeQwenImageEditPlus",
            "inputs": {"prompt": "cat", "image1": ["7", 0]},
        },
        "7": {"class_type": "LoadImage", "inputs": {"image": "reference.png"}},
        "2": {
            "class_type": "KSampler",
            "inputs": {"positive": ["3", 0], "latent_image": ["9", 0], "steps": 8},
        },
    }

    workflow_map, _ = _infer_workflow_map(api, [], media=WorkflowMedia.IMAGE)

    assert workflow_map is not None
    media_input = workflow_map.media_inputs[0]
    assert media_input.slot.value == "reference"
    assert media_input.target_role is WorkflowRole.POSITIVE_PROMPT
    assert media_input.target_input == "image1"


@pytest.mark.parametrize(
    ("loader_class", "loader_input"),
    tuple(
        (class_name, spec.input_name)
        for class_name, spec in MEDIA_LOADER_SPECS.items()
        if spec.kind.value == "image"
    ),
)
def test_image_inference_uses_the_shared_loader_spec(loader_class: str, loader_input: str) -> None:
    api = {
        "9": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024}},
        "3": {
            "class_type": "TextEncodeQwenImageEditPlus",
            "inputs": {"prompt": "cat", "image1": ["7", 0]},
        },
        "7": {"class_type": loader_class, "inputs": {loader_input: "reference.png"}},
        "2": {
            "class_type": "KSampler",
            "inputs": {"positive": ["3", 0], "latent_image": ["9", 0], "steps": 8},
        },
    }

    workflow_map, _ = _infer_workflow_map(api, [], media=WorkflowMedia.IMAGE)

    assert workflow_map is not None
    assert (
        BundleConfig.model_validate(
            {
                **_raw_bundle(),
                "workflow": workflow_map.model_dump(mode="json", by_alias=True),
            }
        ).workflow
        is not None
    )


def test_inference_does_not_certify_a_loader_edge_from_an_unsupported_output() -> None:
    api = {
        "9": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024}},
        "3": {
            "class_type": "TextEncodeQwenImageEditPlus",
            "inputs": {"prompt": "cat", "image1": ["7", 1]},
        },
        "7": {"class_type": "LoadImage", "inputs": {"image": "reference.png"}},
        "2": {
            "class_type": "KSampler",
            "inputs": {"positive": ["3", 0], "latent_image": ["9", 0], "steps": 8},
        },
    }

    result = infer_workflow_map(api, [], media=WorkflowMedia.IMAGE)

    assert result.workflow_map is not None
    assert result.workflow_map.media_inputs == []
    assert len(result.unresolved_media_inputs) == 1
    unresolved = result.unresolved_media_inputs[0]
    assert unresolved.reason == "unsupported_output_slot"
    assert unresolved.loader_id == "7"
    assert unresolved.target_role is WorkflowRole.POSITIVE_PROMPT
    assert unresolved.target_input == "image1"
    assert unresolved.output_slot == 1
    assert any("unsupported output slot 1" in comment for comment in result.comments)


def test_inference_uses_one_deterministic_representative_for_loader_fan_out() -> None:
    api = {
        "9": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024}},
        "3": {
            "class_type": "TextEncodeQwenImageEditPlus",
            "inputs": {
                "prompt": "cat",
                "second_reference": ["7", 0],
                "first_reference": ["7", 0],
            },
        },
        "7": {"class_type": "LoadImage", "inputs": {"image": "reference.png"}},
        "2": {
            "class_type": "KSampler",
            "inputs": {"positive": ["3", 0], "latent_image": ["9", 0], "steps": 8},
        },
    }

    workflow_map, comments = _infer_workflow_map(api, [], media=WorkflowMedia.IMAGE)

    assert workflow_map is not None
    assert len(workflow_map.media_inputs) == 1
    assert workflow_map.media_inputs[0].target_input == "first_reference"
    assert (
        BundleConfig.model_validate(
            {
                **_raw_bundle(),
                "workflow": workflow_map.model_dump(mode="json", by_alias=True),
            }
        ).workflow
        is not None
    )
    assert all("declare each intended target manually" not in comment for comment in comments)


def test_video_inference_leaves_loader_slot_for_the_author_without_guessing() -> None:
    api = {
        "9": {
            "class_type": "WanImageToVideo",
            "inputs": {"width": 1024, "length": 81, "image": ["7", 0]},
        },
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "cat"}},
        "7": {"class_type": "LoadImage", "inputs": {"image": "frame.png"}},
        "2": {
            "class_type": "KSampler",
            "inputs": {"positive": ["3", 0], "latent_image": ["9", 0], "steps": 8},
        },
    }

    result = infer_workflow_map(api, [], media=WorkflowMedia.VIDEO)

    assert result.workflow_map is not None
    assert result.workflow_map.media_inputs == []
    assert len(result.unresolved_media_inputs) == 1
    unresolved = result.unresolved_media_inputs[0]
    assert unresolved.reason == "slot_required"
    assert unresolved.kind is WorkflowMediaKind.IMAGE
    assert unresolved.target_role is WorkflowRole.LATENT
    assert unresolved.target_input == "image"
    assert any(
        comment.startswith("TODO:")
        and "node 7" in comment
        and "target_role: latent" in comment
        and "reference, first_frame, last_frame" in comment
        and "source" not in comment
        for comment in result.comments
    )


def test_video_loader_inference_requires_explicit_source_slot() -> None:
    api = {
        "9": {
            "class_type": "WanVideoToVideo",
            "inputs": {"width": 1024, "length": 81, "video": ["7", 0]},
        },
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "cat"}},
        "7": {"class_type": "LoadVideo", "inputs": {"file": "source.mp4"}},
        "2": {
            "class_type": "KSampler",
            "inputs": {"positive": ["3", 0], "latent_image": ["9", 0], "steps": 8},
        },
    }

    result = infer_workflow_map(api, [], media=WorkflowMedia.VIDEO)

    assert result.workflow_map is not None
    assert result.workflow_map.media_inputs == []
    assert result.unresolved_media_inputs[0].reason == "slot_required"
    assert result.unresolved_media_inputs[0].kind is WorkflowMediaKind.VIDEO
    assert any("choose its required slot (source)" in comment for comment in result.comments)


def test_incompatible_loader_kind_does_not_advertise_impossible_slots() -> None:
    api = {
        "9": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024}},
        "3": {
            "class_type": "TextEncodeQwenImageEditPlus",
            "inputs": {"prompt": "cat", "video": ["7", 0]},
        },
        "7": {"class_type": "LoadVideo", "inputs": {"file": "source.mp4"}},
        "2": {
            "class_type": "KSampler",
            "inputs": {"positive": ["3", 0], "latent_image": ["9", 0], "steps": 8},
        },
    }

    result = infer_workflow_map(api, [], media=WorkflowMedia.IMAGE)

    assert result.workflow_map is not None
    assert result.workflow_map.media_inputs == []
    assert result.unresolved_media_inputs[0].reason == "media_incompatible"
    assert all("choose its required slot" not in comment for comment in result.comments)
    assert any("no legal video upload slot" in comment for comment in result.comments)


def _inference_api(positive_id: str) -> dict[str, object]:
    return {
        "65": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024}},
        "71": {
            "class_type": "KSampler",
            "inputs": {
                "positive": [positive_id, 0],
                "latent_image": ["65", 0],
                "steps": 8,
            },
        },
    }


@pytest.mark.parametrize(
    ("passthrough_class", "passthrough_inputs"),
    (
        ("ConditioningCombine", {"conditioning_1": ["3", 0]}),
        ("ControlNetApplyAdvanced", {"positive": ["3", 0]}),
    ),
)
def test_inference_traces_known_conditioning_passthroughs(
    passthrough_class: str, passthrough_inputs: dict[str, list[object]]
) -> None:
    api = _inference_api("5")
    api.update(
        {
            "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "cat"}},
            "5": {"class_type": passthrough_class, "inputs": passthrough_inputs},
        }
    )

    workflow_map, comments = _infer_workflow_map(api, [])

    assert workflow_map is not None
    positive = workflow_map.nodes[WorkflowRole.POSITIVE_PROMPT]
    assert positive.id == "3"
    assert positive.inputs == {"text": "text"}
    assert any(
        passthrough_class in comment and not comment.startswith("TODO:") for comment in comments
    )


def test_inference_declines_unwritable_resolved_positive_prompt() -> None:
    api = _inference_api("5")
    api.update(
        {
            "3": {"class_type": "CustomConditioning", "inputs": {"strength": 1.0}},
            "5": {"class_type": "ConditioningCombine", "inputs": {"conditioning_1": ["3", 0]}},
        }
    )

    workflow_map, comments = _infer_workflow_map(api, [])

    assert workflow_map is None
    assert any("node 3 (CustomConditioning)" in comment for comment in comments)
    assert any("no writable text input" in comment for comment in comments)


def _inference_api_with_negative(negative_id: str) -> dict[str, object]:
    api = _inference_api("3")
    sampler = api["71"]
    assert isinstance(sampler, dict)
    sampler_inputs = sampler["inputs"]
    assert isinstance(sampler_inputs, dict)
    sampler_inputs["negative"] = [negative_id, 0]
    api["3"] = {"class_type": "CLIPTextEncode", "inputs": {"text": "positive"}}
    return api


def test_negative_prompt_traces_through_timestep_range() -> None:
    api = _inference_api_with_negative("5")
    api.update(
        {
            "5": {
                "class_type": "ConditioningSetTimestepRange",
                "inputs": {"conditioning": ["6", 0], "start": 0.0, "end": 1.0},
            },
            "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "negative"}},
        }
    )

    workflow_map, comments = _infer_workflow_map(api, [])

    assert workflow_map is not None
    assert workflow_map.nodes[WorkflowRole.NEGATIVE_PROMPT].id == "6"
    assert any("negative_prompt traced through ConditioningSetTimestepRange" in c for c in comments)


def test_negative_multi_source_combiner_omits_only_negative_role() -> None:
    api = _inference_api_with_negative("5")
    api.update(
        {
            "5": {
                "class_type": "ConditioningCombine",
                "inputs": {"conditioning_1": ["6", 0], "conditioning_2": ["7", 0]},
            },
            "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "negative 1"}},
            "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "negative 2"}},
        }
    )

    workflow_map, comments = _infer_workflow_map(api, [])

    assert workflow_map is not None
    assert WorkflowRole.NEGATIVE_PROMPT not in workflow_map.nodes
    assert any(
        comment.startswith("TODO: negative_prompt") and "6" in comment and "7" in comment
        for comment in comments
    )


def test_controlnet_advanced_output_slots_trace_distinct_prompt_roles() -> None:
    api = _inference_api_with_negative("5")
    sampler = api["71"]
    assert isinstance(sampler, dict)
    sampler_inputs = sampler["inputs"]
    assert isinstance(sampler_inputs, dict)
    sampler_inputs["positive"] = ["5", 0]
    sampler_inputs["negative"] = ["5", 1]
    api.update(
        {
            "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "positive"}},
            "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "negative"}},
            "5": {
                "class_type": "ControlNetApplyAdvanced",
                "inputs": {"positive": ["3", 0], "negative": ["6", 0]},
            },
        }
    )

    workflow_map, _ = _infer_workflow_map(api, [])

    assert workflow_map is not None
    assert workflow_map.nodes[WorkflowRole.POSITIVE_PROMPT].id == "3"
    assert workflow_map.nodes[WorkflowRole.NEGATIVE_PROMPT].id == "6"


def test_positive_multi_source_combiner_declines_map_with_all_origins() -> None:
    api = _inference_api("5")
    api.update(
        {
            "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "first"}},
            "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "second"}},
            "5": {
                "class_type": "ConditioningCombine",
                "inputs": {"conditioning_1": ["3", 0], "conditioning_2": ["6", 0]},
            },
        }
    )

    workflow_map, comments = _infer_workflow_map(api, [])

    assert workflow_map is None
    assert any(
        comment.startswith("TODO: positive_prompt") and "3" in comment and "6" in comment
        for comment in comments
    )


def test_conditioning_trace_depth_exhaustion_is_explicit() -> None:
    api = _inference_api("5")
    for index in range(5, 14):
        api[str(index)] = {
            "class_type": "ConditioningSetArea",
            "inputs": {"conditioning": [str(index + 1), 0]},
        }
    api["14"] = {"class_type": "CLIPTextEncode", "inputs": {"text": "cat"}}

    workflow_map, comments = _infer_workflow_map(api, [])

    assert workflow_map is None
    assert any("conditioning trace reached maximum depth" in comment for comment in comments)


def test_conditioning_passthrough_cycle_terminates_without_emitting_map() -> None:
    api = _inference_api("5")
    api.update(
        {
            "5": {"class_type": "ConditioningCombine", "inputs": {"conditioning_1": ["6", 0]}},
            "6": {"class_type": "ConditioningCombine", "inputs": {"conditioning_1": ["5", 0]}},
        }
    )

    workflow_map, comments = _infer_workflow_map(api, [])

    assert workflow_map is None
    assert any("no writable text input" in comment for comment in comments)


def test_infer_workflow_map_degrades_on_blank_class_type_instead_of_raising() -> None:
    api = {
        "71": {
            "class_type": "KSampler",
            "inputs": {"positive": ["69", 0], "latent_image": ["65", 0], "steps": 8},
        },
        "69": {"class_type": "", "inputs": {}},
        "65": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024}},
    }

    workflow_map, comments = _infer_workflow_map(api, [])

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

    workflow_map, comments = _infer_workflow_map(api, [])

    assert workflow_map is None
    assert any("failed workflow map validation" in comment for comment in comments)


def test_generated_workflow_comments_are_ascii() -> None:
    _, comments = _infer_workflow_map({}, [])
    assert comments
    assert all(comment.isascii() for comment in comments)


def test_every_inference_comment_branch_round_trips_through_bundle_yaml() -> None:
    base = _inference_api("3")
    base.update({"3": {"class_type": "CLIPTextEncode", "inputs": {"text": "cat"}}})
    model = ModelConfig(
        name="checkpoint",
        model_type="checkpoints",
        files=[ModelFileConfig(name="checkpoint", url="", filename="model.safetensors")],
    )
    invalid_image = copy.deepcopy(base)
    _api_inputs(invalid_image, "3")["image"] = ["", 0]
    invalid_image[""] = {"class_type": "LoadImage", "inputs": {"image": "input.png"}}
    invalid_model = copy.deepcopy(base)
    invalid_model[""] = {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": "model.safetensors"},
    }
    structurally_invalid = copy.deepcopy(base)
    _api_inputs(structurally_invalid, "71")["latent_image"] = ["65", 0]
    structurally_invalid["65"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": "cat", "width": 1024},
    }
    _api_inputs(structurally_invalid, "71")["positive"] = ["65", 0]
    two_samplers = copy.deepcopy(base)
    two_samplers["72"] = {"class_type": "KSampler", "inputs": {}}
    blank_class = copy.deepcopy(base)
    blank_class["3"] = {"class_type": "", "inputs": {}}
    empty_node_id = copy.deepcopy(base)
    _api_inputs(empty_node_id, "71")["positive"] = ["", 0]
    empty_node_id[""] = {"class_type": "CLIPTextEncode", "inputs": {"text": "cat"}}
    unwritable = copy.deepcopy(base)
    unwritable["3"] = {"class_type": "CustomConditioning", "inputs": {"strength": 1.0}}
    multi_source = copy.deepcopy(base)
    multi_source["3"] = {"class_type": "CLIPTextEncode", "inputs": {"text": "first"}}
    multi_source["6"] = {"class_type": "CLIPTextEncode", "inputs": {"text": "second"}}
    multi_source["5"] = {
        "class_type": "ConditioningCombine",
        "inputs": {"conditioning_1": ["3", 0], "conditioning_2": ["6", 0]},
    }
    _api_inputs(multi_source, "71")["positive"] = ["5", 0]
    cases: tuple[tuple[str, dict[str, object], list[ModelConfig]], ...] = (
        ("no sampler", {}, []),
        ("several samplers", two_samplers, []),
        ("blank class", blank_class, []),
        ("empty node id", empty_node_id, []),
        ("unwritable positive", unwritable, []),
        ("multi-source combiner", multi_source, []),
        ("invalid image input", invalid_image, []),
        ("invalid model input", invalid_model, [model]),
        ("structurally invalid map", structurally_invalid, []),
    )
    config = BundleConfig.model_validate(_raw_bundle())
    expected = config.model_dump(mode="json", by_alias=True, exclude_none=True)

    for label, api, models in cases:
        _workflow_map, comments = _infer_workflow_map(api, models)
        assert comments, label
        rendered = _render_bundle_yaml(config, workflow_comments=comments)
        assert yaml.safe_load(rendered) == expected, label
