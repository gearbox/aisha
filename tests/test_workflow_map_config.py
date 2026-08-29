"""Schema-level validation for workflow map config models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ai_content_service.config import (
    BundleConfig,
    WorkflowModelInputConfig,
    WorkflowNodeConfig,
)
from tests.workflow_map_helpers import _raw_bundle


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


def test_workflow_map_requires_contract_version_and_media() -> None:
    for field in ("contract_version", "media"):
        raw = _raw_bundle()
        workflow = raw["workflow"]
        assert isinstance(workflow, dict)
        del workflow[field]

        with pytest.raises(ValidationError, match=field):
            BundleConfig.model_validate(raw)


@pytest.mark.parametrize("version", (1, 3))
def test_workflow_map_rejects_unsupported_contract_versions(version: int) -> None:
    raw = _raw_bundle()
    workflow = raw["workflow"]
    assert isinstance(workflow, dict)
    workflow["contract_version"] = version

    with pytest.raises(ValidationError, match="supported version is 2"):
        BundleConfig.model_validate(raw)


def test_image_media_rejects_video_only_parameters_but_video_allows_them() -> None:
    image_raw = _raw_bundle()
    image_workflow = image_raw["workflow"]
    assert isinstance(image_workflow, dict)
    image_nodes = image_workflow["nodes"]
    assert isinstance(image_nodes, dict)
    image_latent = image_nodes["latent"]
    assert isinstance(image_latent, dict)
    image_latent["inputs"] = {"length": "length"}

    with pytest.raises(ValidationError, match=r"length.*image"):
        BundleConfig.model_validate(image_raw)

    video_raw = _raw_bundle()
    video_workflow = video_raw["workflow"]
    assert isinstance(video_workflow, dict)
    video_workflow["media"] = "video"
    video_nodes = video_workflow["nodes"]
    assert isinstance(video_nodes, dict)
    video_latent = video_nodes["latent"]
    assert isinstance(video_latent, dict)
    video_latent["inputs"] = {"length": "length"}

    assert BundleConfig.model_validate(video_raw).workflow is not None


def test_model_sampling_accepts_only_shift() -> None:
    raw = _raw_bundle()
    workflow = raw["workflow"]
    assert isinstance(workflow, dict)
    nodes = workflow["nodes"]
    assert isinstance(nodes, dict)
    nodes["model_sampling"] = {
        "id": 13,
        "class": "ModelSamplingSD3",
        "inputs": {"shift": "shift"},
    }

    assert BundleConfig.model_validate(raw).workflow is not None

    nodes["model_sampling"] = {
        "id": 13,
        "class": "ModelSamplingSD3",
        "inputs": {"cfg": "cfg"},
    }
    with pytest.raises(ValidationError, match="unsupported parameter"):
        BundleConfig.model_validate(raw)


def test_media_inputs_validate_targets_slots_and_node_ids() -> None:
    raw = _raw_bundle()
    workflow = raw["workflow"]
    assert isinstance(workflow, dict)
    workflow["media_inputs"] = [
        {
            "id": 4,
            "class": "LoadImage",
            "kind": "image",
            "slot": "reference",
            "target_input": "image1",
        }
    ]
    config = BundleConfig.model_validate(raw)
    assert config.workflow is not None
    media_input = config.workflow.media_inputs[0]
    assert media_input.input == "image"
    assert media_input.target_role.value == "positive_prompt"
    dumped = config.model_dump(mode="json", by_alias=True)
    dumped_workflow = dumped["workflow"]
    assert isinstance(dumped_workflow, dict)
    dumped_media_inputs = dumped_workflow["media_inputs"]
    assert isinstance(dumped_media_inputs, list)
    assert dumped_media_inputs[0]["input"] == "image"

    workflow["media_inputs"] = [
        {
            "id": 4,
            "class": "LoadImage",
            "kind": "image",
            "slot": "reference",
            "target_role": "save",
            "target_input": "images",
        }
    ]
    with pytest.raises(ValidationError, match=r"target_role.*not present"):
        BundleConfig.model_validate(raw)

    workflow["media_inputs"] = [
        {
            "id": 4,
            "class": "LoadImage",
            "kind": "image",
            "slot": "reference",
            "target_input": "prompt",
        }
    ]
    with pytest.raises(ValidationError, match="collides with a mapped parameter"):
        BundleConfig.model_validate(raw)

    workflow["media_inputs"] = [
        {
            "id": 4,
            "class": "LoadImage",
            "kind": "image",
            "slot": "first_frame",
            "target_input": "image1",
        },
        {
            "id": 5,
            "class": "LoadImage",
            "kind": "image",
            "slot": "first_frame",
            "target_input": "image2",
        },
    ]
    with pytest.raises(ValidationError, match=r"more than one.*first_frame"):
        BundleConfig.model_validate(raw)

    workflow["media_inputs"] = [
        {
            "id": 4,
            "class": "LoadImage",
            "kind": "image",
            "slot": "reference",
            "target_input": "image1",
        },
        {
            "id": 5,
            "class": "LoadImage",
            "kind": "image",
            "slot": "reference",
            "target_input": "image2",
        },
    ]
    assert BundleConfig.model_validate(raw).workflow is not None


def test_media_inputs_enforce_media_slot_order_and_global_node_ids() -> None:
    raw = _raw_bundle()
    workflow = raw["workflow"]
    assert isinstance(workflow, dict)
    workflow["media_inputs"] = [
        {
            "id": 4,
            "class": "LoadImage",
            "kind": "image",
            "slot": "first_frame",
            "target_input": "image1",
        }
    ]
    with pytest.raises(ValidationError, match="requires media: video"):
        BundleConfig.model_validate(raw)

    workflow["media"] = "video"
    workflow["media_inputs"] = [
        {
            "id": 4,
            "class": "LoadImage",
            "kind": "image",
            "slot": "last_frame",
            "target_input": "image1",
        }
    ]
    with pytest.raises(ValidationError, match=r"last_frame.*first_frame"):
        BundleConfig.model_validate(raw)

    workflow["media_inputs"] = [
        {
            "id": 12,
            "class": "LoadImage",
            "kind": "image",
            "slot": "first_frame",
            "target_input": "image1",
        }
    ]
    workflow["model_inputs"] = [
        {
            "id": 12,
            "class": "CheckpointLoaderSimple",
            "input": "ckpt_name",
            "filename": "model.safetensors",
        }
    ]
    with pytest.raises(ValidationError, match=r"node id.*reused"):
        BundleConfig.model_validate(raw)


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


@pytest.mark.parametrize("role", ("positive_prompt", "negative_prompt"))
def test_workflow_map_rejects_prompt_role_without_text_input(role: str) -> None:
    raw = _raw_bundle()
    workflow = raw["workflow"]
    assert isinstance(workflow, dict)
    nodes = workflow["nodes"]
    assert isinstance(nodes, dict)
    if role == "negative_prompt":
        nodes[role] = {"id": 4, "class": "CLIPTextEncode", "inputs": {}}
    else:
        node = nodes[role]
        assert isinstance(node, dict)
        node["inputs"] = {}

    with pytest.raises(ValidationError, match="must include 'text'"):
        BundleConfig.model_validate(raw)


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
