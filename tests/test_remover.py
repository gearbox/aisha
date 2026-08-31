"""Tests for reference-counted resident bundle removal."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from ai_content_service.config import Settings
from ai_content_service.remover import BundleRemover, RemovalError
from ai_content_service.residency import ResidencyStore, ResidentBundle, ResidentModelFile
from ai_content_service.workflows import WorkflowError, WorkflowManager

if TYPE_CHECKING:
    from pathlib import Path


def _bundle(name: str, paths: tuple[str, ...]) -> ResidentBundle:
    return ResidentBundle(
        name=name,
        version="260901-01",
        registry=None,
        mode="additive",
        deployed_at="2026-09-01T00:00:00+00:00",
        model_files=tuple(ResidentModelFile(path, "a" * 64, None) for path in paths),
        custom_nodes=(),
        workflow_filename=None,
        readiness_node_class=None,
        pending_restart=True,
    )


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(comfyui_path=tmp_path / "ComfyUI", cache_path=tmp_path / "cache")


@pytest.fixture
def store(settings: Settings) -> ResidencyStore:
    return ResidencyStore(settings.residency_path)


@pytest.fixture
def workflow_manager() -> MagicMock:
    return MagicMock(spec=WorkflowManager)


def _remover(
    settings: Settings, store: ResidencyStore, workflow_manager: MagicMock
) -> BundleRemover:
    return BundleRemover(settings, residency=store, workflow_manager=workflow_manager)


async def test_shared_file_is_retained_for_the_other_bundle(
    settings: Settings, store: ResidencyStore, workflow_manager: MagicMock
) -> None:
    store.record(_bundle("first", ("checkpoints/shared.safetensors",)))
    store.record(_bundle("second", ("checkpoints/shared.safetensors",)))
    path = settings.models_path / "checkpoints" / "shared.safetensors"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"shared")

    result = await _remover(settings, store, workflow_manager).remove("second")

    assert path.exists()
    assert result.files_retained == ("checkpoints/shared.safetensors",)
    assert result.files_removed == ()


async def test_exclusive_file_is_removed(
    settings: Settings, store: ResidencyStore, workflow_manager: MagicMock
) -> None:
    store.record(_bundle("target", ("checkpoints/exclusive.safetensors",)))
    path = settings.models_path / "checkpoints" / "exclusive.safetensors"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"bytes")

    result = await _remover(settings, store, workflow_manager).remove("target")

    assert not path.exists()
    assert result.bytes_freed == 5


async def test_missing_file_on_disk_is_not_an_error(
    settings: Settings, store: ResidencyStore, workflow_manager: MagicMock
) -> None:
    store.record(_bundle("target", ("checkpoints/missing.safetensors",)))

    result = await _remover(settings, store, workflow_manager).remove("target")

    assert result.files_removed == ("checkpoints/missing.safetensors",)
    assert result.bytes_freed == 0


async def test_custom_nodes_directory_is_untouched(
    settings: Settings, store: ResidencyStore, workflow_manager: MagicMock
) -> None:
    store.record(_bundle("target", ("checkpoints/weight.safetensors",)))
    node_file = settings.custom_nodes_path / "Node" / "node.py"
    node_file.parent.mkdir(parents=True)
    node_file.write_text("keep")

    await _remover(settings, store, workflow_manager).remove("target")

    assert node_file.exists()
    assert workflow_manager.method_calls == []


async def test_models_root_is_never_pruned(
    settings: Settings, store: ResidencyStore, workflow_manager: MagicMock
) -> None:
    store.record(_bundle("target", ("only.safetensors",)))
    path = settings.models_path / "only.safetensors"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"x")

    await _remover(settings, store, workflow_manager).remove("target")

    assert settings.models_path.exists()


async def test_empty_subdirectory_is_pruned(
    settings: Settings, store: ResidencyStore, workflow_manager: MagicMock
) -> None:
    store.record(_bundle("target", ("checkpoints/nested/weight.safetensors",)))
    path = settings.models_path / "checkpoints" / "nested" / "weight.safetensors"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"x")

    result = await _remover(settings, store, workflow_manager).remove("target")

    assert not (settings.models_path / "checkpoints").exists()
    assert result.directories_pruned == ("checkpoints/nested", "checkpoints")


async def test_retain_mismatch_with_known_bundles_uses_union_and_logs(
    settings: Settings,
    store: ResidencyStore,
    workflow_manager: MagicMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    store.record(_bundle("target", ("checkpoints/shared.safetensors",)))
    store.record(_bundle("apex-only", ("checkpoints/shared.safetensors",)))
    store.record(_bundle("manifest-only", ()))
    path = settings.models_path / "checkpoints" / "shared.safetensors"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"shared")

    result = await _remover(settings, store, workflow_manager).remove(
        "target", retain_bundles=("apex-only",)
    )

    assert path.exists()
    assert result.files_retained == ("checkpoints/shared.safetensors",)
    assert "residency.retain_mismatch" in caplog.text


async def test_retain_naming_unknown_bundle_refuses_removal(
    settings: Settings,
    store: ResidencyStore,
    workflow_manager: MagicMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    store.record(_bundle("target", ("checkpoints/weight.safetensors",)))
    path = settings.models_path / "checkpoints" / "weight.safetensors"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"weight")

    with pytest.raises(RemovalError, match=r"apex.*unknown"):
        await _remover(settings, store, workflow_manager).remove(
            "target", retain_bundles=("unknown",)
        )

    assert path.exists()
    assert "target" in store.load()
    assert "residency.retain_unresolvable" in caplog.text


async def test_retain_omitting_a_manifest_bundle_still_retains_it(
    settings: Settings, store: ResidencyStore, workflow_manager: MagicMock
) -> None:
    store.record(_bundle("target", ("checkpoints/shared.safetensors",)))
    store.record(_bundle("manifest-only", ("checkpoints/shared.safetensors",)))
    store.record(_bundle("apex-known", ()))
    path = settings.models_path / "checkpoints" / "shared.safetensors"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"shared")

    result = await _remover(settings, store, workflow_manager).remove(
        "target", retain_bundles=("apex-known",)
    )

    assert path.exists()
    assert result.files_retained == ("checkpoints/shared.safetensors",)


async def test_dry_run_prune_projection_matches_actual_prune(
    settings: Settings, store: ResidencyStore, workflow_manager: MagicMock
) -> None:
    store.record(_bundle("target", ("checkpoints/nested/weight.safetensors",)))
    path = settings.models_path / "checkpoints" / "nested" / "weight.safetensors"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"weight")
    remover = _remover(settings, store, workflow_manager)

    projected = await remover.remove("target", dry_run=True)
    actual = await remover.remove("target")

    assert projected.directories_pruned == actual.directories_pruned


async def test_workflow_removed_is_false_when_the_file_is_absent(
    settings: Settings, store: ResidencyStore, workflow_manager: MagicMock
) -> None:
    bundle = _bundle("target", ())
    store.record(
        ResidentBundle(
            name=bundle.name,
            version=bundle.version,
            registry=bundle.registry,
            mode=bundle.mode,
            deployed_at=bundle.deployed_at,
            model_files=bundle.model_files,
            custom_nodes=bundle.custom_nodes,
            workflow_filename="target_workflow.json",
            readiness_node_class=bundle.readiness_node_class,
            pending_restart=bundle.pending_restart,
        )
    )
    workflow_manager.remove_workflow.side_effect = WorkflowError("missing")
    workflow_manager.list_workflows.return_value = []

    dry_run = await _remover(settings, store, workflow_manager).remove("target", dry_run=True)
    result = await _remover(settings, store, workflow_manager).remove("target")

    assert dry_run.workflow_removed is False
    assert result.workflow_removed is False


async def test_dry_run_changes_nothing(
    settings: Settings, store: ResidencyStore, workflow_manager: MagicMock
) -> None:
    store.record(_bundle("target", ("checkpoints/weight.safetensors",)))
    path = settings.models_path / "checkpoints" / "weight.safetensors"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"bytes")
    before = settings.residency_path.read_bytes()

    result = await _remover(settings, store, workflow_manager).remove("target", dry_run=True)

    assert result.files_removed == ("checkpoints/weight.safetensors",)
    assert path.exists()
    assert settings.residency_path.read_bytes() == before
    workflow_manager.remove_workflow.assert_not_called()


async def test_removing_unknown_bundle_raises_and_names_residents(
    settings: Settings, store: ResidencyStore, workflow_manager: MagicMock
) -> None:
    store.record(_bundle("known", ()))

    with pytest.raises(RemovalError, match="known"):
        await _remover(settings, store, workflow_manager).remove("unknown")
