"""Coverage for inferring a workflow map from a converted API graph."""

from __future__ import annotations

import copy

import pytest
import yaml

from ai_content_service.config import BundleConfig, ModelConfig, ModelFileConfig, WorkflowRole
from ai_content_service.snapshot import _render_bundle_yaml
from ai_content_service.workflow_map import infer_workflow_map
from tests.workflow_map_helpers import _api_inputs, _raw_bundle


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

    workflow_map, comments = infer_workflow_map(api, [])

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

    workflow_map, comments = infer_workflow_map(api, [])

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

    workflow_map, comments = infer_workflow_map(api, [])

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

    workflow_map, comments = infer_workflow_map(api, [])

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

    workflow_map, _ = infer_workflow_map(api, [])

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

    workflow_map, comments = infer_workflow_map(api, [])

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

    workflow_map, comments = infer_workflow_map(api, [])

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

    workflow_map, comments = infer_workflow_map(api, [])

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
    cases = (
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
        _workflow_map, comments = infer_workflow_map(api, models)
        assert comments, label
        rendered = _render_bundle_yaml(config, workflow_comments=comments)
        assert yaml.safe_load(rendered) == expected, label
