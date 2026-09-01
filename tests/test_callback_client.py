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


def _response(
    status: int,
    content_type: str = "application/json",
    *,
    location: str | None = None,
) -> httpx.Response:
    headers = {"content-type": content_type}
    if location is not None:
        headers["location"] = location
    return httpx.Response(status, headers=headers)


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


def test_200_html_warns_with_frontend_hint() -> None:
    detail = CallbackClient._diagnose(_response(200, "text/html"), "text/html")
    assert "frontend/static host" in detail


def test_200_missing_content_type_emits_non_api_hint() -> None:
    detail = CallbackClient._diagnose(_response(200, ""), "")
    assert "non-API response" in detail


def test_200_text_plain_emits_non_api_hint() -> None:
    detail = CallbackClient._diagnose(_response(200, "text/plain"), "text/plain")
    assert "APEX_CALLBACK_URL may not point at the Apex API" in detail


def test_405_warns_with_static_host_hint() -> None:
    detail = CallbackClient._diagnose(_response(405), "application/json")
    assert "static host" in detail


def test_302_cloudflare_access_warns_with_auth_proxy_hint() -> None:
    detail = CallbackClient._diagnose(
        _response(302, location="https://team.cloudflareaccess.com/cdn-cgi/access/login"),
        "application/json",
    )
    assert "auth proxy" in detail


def test_403_html_warns_with_auth_proxy_hint() -> None:
    detail = CallbackClient._diagnose(_response(403, "text/html"), "text/html")
    assert "Cloudflare Access" in detail


def test_401_json_warns_without_auth_proxy_hint() -> None:
    detail = CallbackClient._diagnose(_response(401), "application/json")
    assert detail == "HTTP 401"


def test_403_json_is_generic_not_auth_proxy() -> None:
    detail = CallbackClient._diagnose(_response(403), "application/json")
    assert detail == "HTTP 403"


def test_302_non_cloudflare_redirect_is_generic() -> None:
    detail = CallbackClient._diagnose(
        _response(302, location="https://example.test/login"), "application/json"
    )
    assert detail == "HTTP 302"


def test_trailing_slash_in_callback_url_is_stripped() -> None:
    client = CallbackClient(base_url="https://apex.test/", token="token", enabled=True)
    assert client._base_url == "https://apex.test"


async def test_first_failure_warns_then_debug(caplog: pytest.LogCaptureFixture) -> None:
    http_client = MagicMock()
    http_client.post = AsyncMock(side_effect=httpx.ConnectError("offline"))
    client = CallbackClient(base_url="https://apex.test", token="token", enabled=True)
    with (
        patch("ai_content_service.callback_client.httpx.AsyncClient", return_value=http_client),
        caplog.at_level(logging.DEBUG),
    ):
        await client.post_best_effort("/events", {})
        await client.post_best_effort("/events", {})

    failures = [record for record in caplog.records if record.name.endswith("callback_client")]
    assert [record.levelname for record in failures] == ["WARNING", "DEBUG"]


async def test_recovery_rearms_warning(caplog: pytest.LogCaptureFixture) -> None:
    http_client = MagicMock()
    http_client.post = AsyncMock(
        side_effect=[httpx.ConnectError("offline"), _response(200), httpx.ConnectError("offline")]
    )
    client = CallbackClient(base_url="https://apex.test", token="token", enabled=True)
    with (
        patch("ai_content_service.callback_client.httpx.AsyncClient", return_value=http_client),
        caplog.at_level(logging.WARNING),
    ):
        await client.post_best_effort("/events", {})
        await client.post_best_effort("/events", {})
        await client.post_best_effort("/events", {})

    assert sum(record.levelname == "WARNING" for record in caplog.records) == 2


async def test_reuses_single_client() -> None:
    http_client = MagicMock()
    http_client.post = AsyncMock(return_value=_response(200))
    client = CallbackClient(base_url="https://apex.test", token="token", enabled=True)
    with patch(
        "ai_content_service.callback_client.httpx.AsyncClient", return_value=http_client
    ) as factory:
        await client.post_best_effort("/events", {})
        await client.post_best_effort("/events", {})

    factory.assert_called_once()


async def test_closes_client_via_context_manager() -> None:
    http_client = MagicMock()
    http_client.post = AsyncMock(return_value=_response(200))
    http_client.aclose = AsyncMock()
    client = CallbackClient(base_url="https://apex.test", token="token", enabled=True)
    with patch("ai_content_service.callback_client.httpx.AsyncClient", return_value=http_client):
        async with client:
            await client.post_best_effort("/events", {})

    http_client.aclose.assert_awaited_once()


async def test_post_after_close_recreates_client_and_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    first = MagicMock(post=AsyncMock(return_value=_response(200)), aclose=AsyncMock())
    second = MagicMock(post=AsyncMock(return_value=_response(200)))
    client = CallbackClient(base_url="https://apex.test", token="token", enabled=True)
    with (
        patch("ai_content_service.callback_client.httpx.AsyncClient", side_effect=[first, second]),
        caplog.at_level(logging.WARNING),
    ):
        await client.post_best_effort("/events", {})
        await client.aclose()
        await client.post_best_effort("/events", {})

    assert "provisioning.client.recreated_after_close" in caplog.text


async def test_aclose_is_a_noop_when_never_posted() -> None:
    client = CallbackClient(base_url="https://apex.test", token="token", enabled=True)
    with patch("ai_content_service.callback_client.httpx.AsyncClient") as factory:
        await client.aclose()

    factory.assert_not_called()


async def test_network_error_does_not_propagate() -> None:
    http_client = MagicMock(post=AsyncMock(side_effect=httpx.ConnectError("offline")))
    client = CallbackClient(base_url="https://apex.test", token="token", enabled=True)
    with patch("ai_content_service.callback_client.httpx.AsyncClient", return_value=http_client):
        await client.post_best_effort("/events", {})


async def test_http_500_does_not_propagate() -> None:
    http_client = MagicMock(post=AsyncMock(return_value=_response(500)))
    client = CallbackClient(base_url="https://apex.test", token="token", enabled=True)
    with patch("ai_content_service.callback_client.httpx.AsyncClient", return_value=http_client):
        await client.post_best_effort("/events", {})


async def test_timeout_does_not_propagate() -> None:
    http_client = MagicMock(post=AsyncMock(side_effect=httpx.ReadTimeout("slow")))
    client = CallbackClient(base_url="https://apex.test", token="token", enabled=True)
    with patch("ai_content_service.callback_client.httpx.AsyncClient", return_value=http_client):
        await client.post_best_effort("/events", {})


async def test_post_json_returns_decoded_body() -> None:
    http_client = MagicMock()
    http_client.post = AsyncMock(return_value=httpx.Response(200, json={"command_id": "cmd-1"}))
    client = CallbackClient(base_url="https://apex.test", token="token", enabled=True)

    with patch("ai_content_service.callback_client.httpx.AsyncClient", return_value=http_client):
        assert await client.post_json("/claim", {}) == (200, {"command_id": "cmd-1"})


async def test_post_json_204_returns_none_body() -> None:
    http_client = MagicMock(post=AsyncMock(return_value=_response(204)))
    client = CallbackClient(base_url="https://apex.test", token="token", enabled=True)

    with patch("ai_content_service.callback_client.httpx.AsyncClient", return_value=http_client):
        assert await client.post_json("/claim", {}) == (204, None)


async def test_post_json_non_json_200_is_diagnosed(caplog: pytest.LogCaptureFixture) -> None:
    http_client = MagicMock(post=AsyncMock(return_value=_response(200, "text/html")))
    client = CallbackClient(base_url="https://apex.test", token="token", enabled=True)

    with (
        patch("ai_content_service.callback_client.httpx.AsyncClient", return_value=http_client),
        caplog.at_level(logging.ERROR),
    ):
        assert await client.post_json("/claim", {}) == (200, None)

    assert "APEX_CALLBACK_URL may not point at the Apex API" in caplog.text


async def test_post_json_never_raises_on_exhausted_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    http_client = MagicMock(post=AsyncMock(return_value=_response(503)))
    monkeypatch.setattr(
        "ai_content_service.callback_client.wait_exponential", lambda **_kwargs: wait_none()
    )
    client = CallbackClient(base_url="https://apex.test", token="token", enabled=True)

    with patch("ai_content_service.callback_client.httpx.AsyncClient", return_value=http_client):
        assert await client.post_json("/claim", {}) == (0, None)


async def test_disabled_client_claim_returns_204() -> None:
    assert await CallbackClient.disabled().claim_command("session", "agent") == (204, None)
