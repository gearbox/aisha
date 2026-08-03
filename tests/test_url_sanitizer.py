"""Tests for safe presentation of source URLs."""

from __future__ import annotations

import pytest

from ai_content_service.url_sanitizer import sanitize_civitai_url_for_output


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (
            "https://civitai.com/api/download/models/1?token=secret",
            "https://civitai.com/api/download/models/1",
        ),
        (
            "https://civitai.red/api/download/models/1?a=1&TOKEN=secret&b=2",
            "https://civitai.red/api/download/models/1?a=1&b=2",
        ),
        (
            "https://www.civitai.green/api/download/models/1?a=one&a=two&token=encoded%2Bvalue",
            "https://www.civitai.green/api/download/models/1?a=one&a=two",
        ),
        (
            "https://civitai.com/api/download/models/1?a=1&a=2",
            "https://civitai.com/api/download/models/1?a=1&a=2",
        ),
        (
            "https://storage.example.com/file?token=signed-token&X-Amz-Signature=keep",
            "https://storage.example.com/file?token=signed-token&X-Amz-Signature=keep",
        ),
    ],
)
def test_sanitize_civitai_url_for_output_removes_only_civitai_tokens(
    url: str, expected: str
) -> None:
    assert sanitize_civitai_url_for_output(url) == expected
