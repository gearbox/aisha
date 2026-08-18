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
