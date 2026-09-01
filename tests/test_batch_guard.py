"""Tests for conservative provisioning batch headroom checks."""

from __future__ import annotations

from collections import namedtuple
from pathlib import Path
from typing import TYPE_CHECKING

from ai_content_service.batch_guard import check_batch_headroom

if TYPE_CHECKING:
    import pytest

DiskUsage = namedtuple("DiskUsage", "total used free")


def test_sufficient_free_space_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "ai_content_service.batch_guard.shutil.disk_usage", lambda _path: DiskUsage(0, 0, 106)
    )

    verdict = check_batch_headroom(models_path=Path("/models"), declared_bytes=100, margin=0.05)

    assert verdict.ok is True
    assert verdict.required_bytes == 105


def test_margin_is_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "ai_content_service.batch_guard.shutil.disk_usage", lambda _path: DiskUsage(0, 0, 104)
    )

    verdict = check_batch_headroom(models_path=Path("/models"), declared_bytes=100, margin=0.05)

    assert verdict.ok is False


def test_verdict_detail_names_free_and_required_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "ai_content_service.batch_guard.shutil.disk_usage", lambda _path: DiskUsage(0, 0, 50)
    )

    verdict = check_batch_headroom(models_path=Path("/models"), declared_bytes=100, margin=0.0)

    assert "50 bytes free" in verdict.detail
    assert "100 bytes required" in verdict.detail
