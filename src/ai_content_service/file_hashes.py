"""Small, shared filesystem hashing utilities."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


_HASH_CHUNK_SIZE = 1024 * 1024


def compute_file_sha256(path: Path) -> str:
    """Return the lowercase SHA-256 digest for *path*.

    Filesystem errors deliberately propagate to the operation that owns the
    user-facing error boundary.  That keeps this utility usable for both
    strict operations and no-fail-fast report generation.
    """
    hasher = hashlib.sha256()
    with path.open("rb") as file_handle:
        while chunk := file_handle.read(_HASH_CHUNK_SIZE):
            hasher.update(chunk)
    return hasher.hexdigest()
