"""Shared test helper utilities (not pytest fixtures)."""

from __future__ import annotations

import stat
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def make_path_stubs(tmp_path: Path, names: list[str]) -> Path:
    """Create executable no-op stubs for each named binary in tmp_path/bin/.

    Each stub is a shell script that exits 0, so callers that check exit codes
    see success without touching the real system.  Prepend the returned dir to
    PATH in any subprocess env to shadow real binaries.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    for name in names:
        stub = bin_dir / name
        stub.write_text("#!/bin/sh\nexit 0\n")
        stub.chmod(stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return bin_dir
