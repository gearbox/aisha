"""Tests for callback delivery policy and diagnostics."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from tenacity import wait_none

from ai_content_service.callback_client import CallbackClient

if TYPE_CHECKING:
    import pytest


def _response(status: int, content_type: str = "application/json") -> httpx.Response:
    return httpx.Response(status, headers={"content-type": content_type})


async def test_disabled_client_never_builds_httpx_client() -> None:
    client = CallbackClient.disabled()
    with patch("ai_content_service.callback_client.httpx.AsyncClient") as mock_client:
        assert await client.post_retried("/events", {}) is True
        await client.post_best_effort("/events", {})

    mock_client.assert_not_called()


async def test_post_retried_retries_5xx_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    http_client = MagicMock()
    http_client.post = AsyncMock(side_effect=[_response(500), _response(200)])
    monkeypatch.setattr(
        "ai_content_service.callback_client.wait_exponential", lambda **_kwargs: wait_none()
    )
    client = CallbackClient(base_url="https://apex.test", token="token", enabled=True)
    with patch("ai_content_service.callback_client.httpx.AsyncClient", return_value=http_client):
        assert await client.post_retried("/events", {}) is True

    assert http_client.post.await_count == 2


async def test_post_retried_does_not_retry_4xx() -> None:
    http_client = MagicMock()
    http_client.post = AsyncMock(return_value=_response(400))
    client = CallbackClient(base_url="https://apex.test", token="token", enabled=True)
    with patch("ai_content_service.callback_client.httpx.AsyncClient", return_value=http_client):
        assert await client.post_retried("/events", {}) is False

    http_client.post.assert_awaited_once()


async def test_post_retried_returns_false_after_exhaustion_without_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    http_client = MagicMock()
    http_client.post = AsyncMock(return_value=_response(503))
    monkeypatch.setattr(
        "ai_content_service.callback_client.wait_exponential", lambda **_kwargs: wait_none()
    )
    client = CallbackClient(base_url="https://apex.test", token="token", enabled=True)
    with patch("ai_content_service.callback_client.httpx.AsyncClient", return_value=http_client):
        assert await client.post_retried("/events", {}) is False

    assert http_client.post.await_count == 5


async def test_non_json_success_is_diagnosed_as_failure(caplog: pytest.LogCaptureFixture) -> None:
    http_client = MagicMock()
    http_client.post = AsyncMock(return_value=_response(200, "text/html"))
    client = CallbackClient(base_url="https://apex.test", token="token", enabled=True)
    with (
        patch("ai_content_service.callback_client.httpx.AsyncClient", return_value=http_client),
        caplog.at_level(logging.WARNING),
    ):
        assert await client.post_best_effort("/events", {}) is None

    assert "APEX_CALLBACK_URL may not point at the Apex API" in caplog.text


async def test_token_never_logged(caplog: pytest.LogCaptureFixture) -> None:
    token = "callback-secret"
    http_client = MagicMock()
    http_client.post = AsyncMock(return_value=_response(400))
    client = CallbackClient(base_url="https://apex.test", token=token, enabled=True)
    with (
        patch("ai_content_service.callback_client.httpx.AsyncClient", return_value=http_client),
        caplog.at_level(logging.DEBUG),
    ):
        await client.post_best_effort("/events", {"secret": token})

    assert token not in caplog.text
