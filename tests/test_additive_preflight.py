"""Tests for pure additive deployment collision checks."""

from __future__ import annotations

from datetime import UTC, datetime

from ai_content_service.additive_preflight import check_additive
from ai_content_service.config import (
    BundleConfig,
    BundleMetadata,
    ComfyUIConfig,
    CustomNode,
    ModelConfig,
    ModelFileConfig,
)
from ai_content_service.residency import (
    ResidentBundle,
    ResidentCustomNode,
    ResidentModelFile,
)


def _bundle(**changes: object) -> BundleConfig:
    values: dict[str, object] = {
        "metadata": BundleMetadata(
            name="incoming", version="260901-01", created_at=datetime.now(UTC)
        ),
        "models": [
            ModelConfig(
                name="Model",
                model_type="diffusion_models",
                files=[
                    ModelFileConfig(
                        name="weight",
                        url="https://example.test/weight.safetensors",
                        filename="weight.safetensors",
                        sha256="a" * 64,
                    )
                ],
            )
        ],
    }
    values.update(changes)
    return BundleConfig.model_validate(values)


def _resident(
    *,
    nodes: tuple[ResidentCustomNode, ...] = (),
    files: tuple[ResidentModelFile, ...] = (),
) -> dict[str, ResidentBundle]:
    return {
        "resident": ResidentBundle(
            name="resident",
            version="260901-01",
            registry=None,
            mode="full",
            deployed_at="2026-09-01T00:00:00+00:00",
            model_files=files,
            custom_nodes=nodes,
            workflow_filename=None,
            readiness_node_class=None,
            pending_restart=False,
        )
    }


def _codes(report: object, attribute: str) -> list[str]:
    return [finding.code for finding in getattr(report, attribute)]


def test_full_lock_without_overlay_blocks() -> None:
    report = check_additive(
        _bundle(requirements_lock_file="requirements.lock"),
        resident={},
        current_comfyui_commit="a" * 40,
    )

    assert _codes(report, "blocking") == ["requirements_full_lock"]
    assert "requirements_lock_file" in report.blocking[0].detail


def test_overlay_only_bundle_passes() -> None:
    report = check_additive(
        _bundle(requirements_overlay_file="requirements.overlay.txt"),
        resident={},
        current_comfyui_commit="a" * 40,
    )

    assert report.ok is True


def test_same_node_different_commit_blocks() -> None:
    report = check_additive(
        _bundle(
            custom_nodes=[
                CustomNode(name="Node", git_url="https://example.test/node", commit_sha="b" * 40)
            ]
        ),
        resident=_resident(nodes=(ResidentCustomNode("Node", "git", "a" * 40),)),
        current_comfyui_commit="a" * 40,
    )

    assert _codes(report, "blocking") == ["custom_node_pin_conflict"]


def test_same_node_different_source_blocks() -> None:
    report = check_additive(
        _bundle(
            custom_nodes=[
                CustomNode(name="Node", node_id="node-id", source="registry", version="1.0.0")
            ]
        ),
        resident=_resident(nodes=(ResidentCustomNode("Node", "git", "1.0.0"),)),
        current_comfyui_commit="a" * 40,
    )

    assert _codes(report, "blocking") == ["custom_node_pin_conflict"]


def test_same_node_same_pin_passes() -> None:
    report = check_additive(
        _bundle(
            custom_nodes=[
                CustomNode(name="Node", git_url="https://example.test/node", commit_sha="a" * 40)
            ]
        ),
        resident=_resident(nodes=(ResidentCustomNode("Node", "git", "a" * 40),)),
        current_comfyui_commit="a" * 40,
    )

    assert report.ok is True


def test_same_path_different_sha_blocks() -> None:
    report = check_additive(
        _bundle(),
        resident=_resident(
            files=(ResidentModelFile("diffusion_models/weight.safetensors", "b" * 64, 1),)
        ),
        current_comfyui_commit="a" * 40,
    )

    assert _codes(report, "blocking") == ["model_sha_collision"]


def test_same_path_same_sha_produces_no_finding() -> None:
    report = check_additive(
        _bundle(),
        resident=_resident(
            files=(ResidentModelFile("diffusion_models/weight.safetensors", "a" * 64, 1),)
        ),
        current_comfyui_commit="a" * 40,
    )

    assert report.ok is True
    assert report.advisory == ()


def test_missing_sha_is_advisory_not_blocking() -> None:
    report = check_additive(
        _bundle(),
        resident=_resident(
            files=(ResidentModelFile("diffusion_models/weight.safetensors", None, 1),)
        ),
        current_comfyui_commit="a" * 40,
    )

    assert report.ok is True
    assert _codes(report, "advisory") == ["model_path_unverifiable"]


def test_comfyui_override_mismatch_blocks_and_none_commit_is_advisory() -> None:
    bundle = _bundle(comfyui=ComfyUIConfig(commit="b" * 40))

    blocking = check_additive(bundle, resident={}, current_comfyui_commit="a" * 40)
    advisory = check_additive(bundle, resident={}, current_comfyui_commit=None)

    assert _codes(blocking, "blocking") == ["comfyui_revision_mismatch"]
    assert _codes(advisory, "advisory") == ["comfyui_revision_mismatch"]


def test_report_lists_every_finding_not_just_the_first() -> None:
    report = check_additive(
        _bundle(
            requirements_lock_file="requirements.lock",
            comfyui=ComfyUIConfig(commit="b" * 40),
            custom_nodes=[
                CustomNode(name="Node", git_url="https://example.test/node", commit_sha="b" * 40)
            ],
        ),
        resident=_resident(
            nodes=(ResidentCustomNode("Node", "git", "a" * 40),),
            files=(ResidentModelFile("diffusion_models/weight.safetensors", "b" * 64, 1),),
        ),
        current_comfyui_commit="a" * 40,
    )

    assert _codes(report, "blocking") == [
        "requirements_full_lock",
        "comfyui_revision_mismatch",
        "custom_node_pin_conflict",
        "model_sha_collision",
    ]
