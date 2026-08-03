"""Tests for public local-file hashing utilities."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import pytest

from ai_content_service.file_hashes import compute_file_sha256

if TYPE_CHECKING:
    from pathlib import Path


def test_compute_file_sha256_reads_file_in_chunks(tmp_path: Path) -> None:
    path = tmp_path / "weights.bin"
    content = b"weights" * 200_000
    path.write_bytes(content)

    assert compute_file_sha256(path) == hashlib.sha256(content).hexdigest()


def test_compute_file_sha256_propagates_os_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        compute_file_sha256(tmp_path / "missing.bin")
