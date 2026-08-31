"""Tests for the node-local bundle residency manifest."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_content_service.residency import (
    RESIDENCY_SCHEMA_VERSION,
    ResidencyError,
    ResidencyStore,
    ResidentBundle,
    ResidentCustomNode,
    ResidentModelFile,
)


def _bundle(name: str = "image") -> ResidentBundle:
    return ResidentBundle(
        name=name,
        version="260901-01",
        registry="local",
        mode="full",
        deployed_at="2026-09-01T00:00:00+00:00",
        model_files=(
            ResidentModelFile(path="checkpoints/base.safetensors", sha256="a" * 64, size_bytes=42),
        ),
        custom_nodes=(ResidentCustomNode(name="Node", source="git", pin="b" * 40),),
        workflow_filename="image_workflow.json",
        readiness_node_class="ImageNode",
        pending_restart=True,
    )


def test_load_missing_file_returns_empty(tmp_path: Path) -> None:
    assert ResidencyStore(tmp_path / "residency.json").load() == {}


def test_corrupt_manifest_raises_and_does_not_rewrite(tmp_path: Path) -> None:
    path = tmp_path / "residency.json"
    path.write_text("not json", encoding="utf-8")
    original = path.read_bytes()

    with pytest.raises(ResidencyError, match="delete the file"):
        ResidencyStore(path).load()

    assert path.read_bytes() == original


def test_schema_version_mismatch_raises(tmp_path: Path) -> None:
    path = tmp_path / "residency.json"
    path.write_text(json.dumps({"schema_version": 0, "bundles": {}}), encoding="utf-8")

    with pytest.raises(ResidencyError, match="schema_version"):
        ResidencyStore(path).load()


def test_record_replaces_entry_for_same_bundle_name(tmp_path: Path) -> None:
    store = ResidencyStore(tmp_path / "residency.json")
    store.record(_bundle())
    store.record(_bundle("image"))

    assert store.load() == {"image": _bundle("image")}


def test_save_is_atomic_via_tmp_and_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "residency.json"
    calls: list[tuple[Path, Path]] = []

    def capture_replace(source: Path, target: Path) -> None:
        calls.append((source, target))
        source.rename(target)

    monkeypatch.setattr(Path, "replace", capture_replace)
    ResidencyStore(path).record(_bundle())

    assert calls == [(path.with_name("residency.json.tmp"), path)]
    assert (
        json.loads(path.read_text(encoding="utf-8"))["schema_version"] == RESIDENCY_SCHEMA_VERSION
    )


def test_mark_all_restarted_clears_pending_flags(tmp_path: Path) -> None:
    store = ResidencyStore(tmp_path / "residency.json")
    store.record(_bundle())
    store.mark_all_restarted()

    assert store.load()["image"].pending_restart is False
