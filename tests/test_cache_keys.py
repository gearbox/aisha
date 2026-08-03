"""Tests for the single model-cache key derivation function."""

import pytest

from ai_content_service.cache_keys import cache_key_for_sha256


def test_cache_key_uses_content_addressed_layout() -> None:
    digest = "a" * 64
    assert cache_key_for_sha256(digest) == f"models/by-sha256/{digest}"


@pytest.mark.parametrize("digest", ["A" * 64, "a" * 63, "g" * 64])
def test_cache_key_rejects_noncanonical_digest(digest: str) -> None:
    with pytest.raises(ValueError, match="lowercase hexadecimal"):
        cache_key_for_sha256(digest)
