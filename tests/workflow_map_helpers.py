"""Shared graph-builder helpers for the workflow-map test suite (not pytest fixtures)."""

from __future__ import annotations

from typing import cast


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
        },
        "readiness_marker": {"node_class": "KSampler"},
        "workflow_file": "workflow.json",
        "workflow_api_file": "workflow.api.json",
        "workflow": {
            "contract_version": 2,
            "media": "image",
            "nodes": {
                "latent": {"id": 9, "class": "EmptyLatentImage", "inputs": {"width": "width"}},
                "positive_prompt": {
                    "id": 3,
                    "class": "TextEncodeQwenImageEditPlus",
                    "inputs": {"text": "prompt"},
                },
                "sampler": {"id": 2, "class": "KSampler", "inputs": {"steps": "steps"}},
            },
        },
    }


def _gui_graph() -> dict[str, object]:
    return {
        "nodes": [
            {"id": 9, "type": "EmptyLatentImage", "inputs": [], "widgets_values": []},
            {
                "id": 3,
                "type": "TextEncodeQwenImageEditPlus",
                "inputs": [{"name": "prompt", "widget": {}}],
                "widgets_values": ["hello"],
            },
            {
                "id": 2,
                "type": "KSampler",
                "inputs": [{"name": "steps", "widget": {}}],
                "widgets_values": [8],
            },
        ],
        "links": [],
    }


def _api_graph() -> dict[str, object]:
    return {
        "9": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024}},
        "3": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"prompt": "hello"}},
        "2": {"class_type": "KSampler", "inputs": {"steps": 8}},
    }


def _api_inputs(api_graph: dict[str, object], node_id: str) -> dict[str, object]:
    node = api_graph[node_id]
    assert isinstance(node, dict)
    inputs = node.get("inputs")
    assert isinstance(inputs, dict)
    return cast("dict[str, object]", inputs)
