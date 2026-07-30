"""Tests for scripts/quick_deploy.py."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

from quick_deploy import _download_progress

if TYPE_CHECKING:
    from pathlib import Path


def test_download_progress_writes_all_chunks(tmp_path: Path) -> None:
    payload = b"x" * (20 * 1024 * 1024)
    temp_path = tmp_path / "model.gguf.tmp"
    _download_progress(io.BytesIO(payload), len(payload), temp_path)
    assert temp_path.read_bytes() == payload


def test_download_progress_handles_unknown_length(tmp_path: Path) -> None:
    temp_path = tmp_path / "model.gguf.tmp"
    _download_progress(io.BytesIO(b"abc"), 0, temp_path)
    assert temp_path.read_bytes() == b"abc"
