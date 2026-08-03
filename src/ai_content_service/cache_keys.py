"""The one place the R2 model-cache key layout is defined.

Content-addressed: the key is the integrity check, and a weight shared by two
bundles is stored once. Both the read path and every write path derive keys
here so the two cannot drift.
"""

from __future__ import annotations

import re
from typing import Final

MODEL_CACHE_KEY_PREFIX: Final = "models/by-sha256"

_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}")


def cache_key_for_sha256(sha256: str) -> str:
    """Return the R2 object key for a weight with digest *sha256*.

    Raises ValueError when the digest is not a 64-character lowercase hex
    string. A malformed key would otherwise create an unreachable object.
    """
    if not _SHA256_RE.fullmatch(sha256):
        msg = f"sha256 must be 64 lowercase hexadecimal characters, got {sha256!r}"
        raise ValueError(msg)
    return f"{MODEL_CACHE_KEY_PREFIX}/{sha256}"
