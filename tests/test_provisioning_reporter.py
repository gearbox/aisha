"""Tests for ProvisioningReporter + deployer/downloader integration."""

from __future__ import annotations

import itertools
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from pydantic import SecretStr

from ai_content_service.config import (
    BundleConfig,
    BundleMetadata,
    CustomNodeConfig,
    ModelConfig,
    ModelFileConfig,
    Settings,
)
from ai_content_service.deployer import Deployer
from ai_content_service.downloader import ModelDownloader
from ai_content_service.provisioning_reporter import ProvisioningReporter

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_reporter(
    session_id: str = "sid-123",
    callback_url: str = "https://apex.example.com",
    callback_token: str = "tok-xyz",
    enabled: bool = True,
    start_ts: float = 0.0,
) -> ProvisioningReporter:
    return ProvisioningReporter(
        session_id=session_id,
        callback_url=callback_url,
        callback_token=callback_token,
        enabled=enabled,
        start_ts=start_ts,
    )


def _make_async_http_client() -> AsyncMock:
    """Return a mock httpx async client with a `.post` spy."""
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=MagicMock())
    return mock_client


def _resp(
    status_code: int,
    content_type: str = "",
    content: bytes = b"",
    location: str = "",
) -> httpx.Response:
    headers: dict[str, str] = {}
    if content_type:
        headers["content-type"] = content_type
    if location:
        headers["location"] = location
    return httpx.Response(status_code=status_code, headers=headers, content=content)


def _patch_post(mock_cls: MagicMock, response: httpx.Response) -> None:
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=response)
    mock_cls.return_value = mock_client


# ---------------------------------------------------------------------------
# 1. Disabled when env unset
# ---------------------------------------------------------------------------


class TestFromEnvDisabled:
    """from_env() returns a disabled reporter when any required var is missing."""

    def test_all_vars_missing(self) -> None:
        # conftest wipes ACS_* env vars automatically
        reporter = ProvisioningReporter.from_env()
        assert not reporter._enabled

    def test_session_id_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ACS_APEX_CALLBACK_URL", "https://apex.example.com")
        monkeypatch.setenv("ACS_APEX_CALLBACK_TOKEN", "tok")
        reporter = ProvisioningReporter.from_env()
        assert not reporter._enabled

    def test_callback_url_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ACS_APEX_SESSION_ID", "sid")
        monkeypatch.setenv("ACS_APEX_CALLBACK_TOKEN", "tok")
        reporter = ProvisioningReporter.from_env()
        assert not reporter._enabled

    def test_token_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ACS_APEX_SESSION_ID", "sid")
        monkeypatch.setenv("ACS_APEX_CALLBACK_URL", "https://apex.example.com")
        reporter = ProvisioningReporter.from_env()
        assert not reporter._enabled

    async def test_disabled_no_http_calls(self) -> None:
        reporter = ProvisioningReporter.disabled()
        with patch("ai_content_service.provisioning_reporter.httpx.AsyncClient") as mock_cls:
            await reporter.phase("comfyui")
            await reporter.download_progress(0, 100, 0, 1)
            await reporter.ready()
            await reporter.failed("err")
            mock_cls.assert_not_called()


# ---------------------------------------------------------------------------
# 2. Enabled posts correct shape
# ---------------------------------------------------------------------------


class TestEnabledPostsCorrectShape:
    async def test_phase_posts_correct_url_and_headers(self) -> None:
        reporter = _make_reporter()
        mock_client = _make_async_http_client()

        with patch("ai_content_service.provisioning_reporter.httpx.AsyncClient") as mock_cls:
            mock_cls.return_value = mock_client
            await reporter.phase("downloading", "Downloading 1 model files")

        mock_client.post.assert_called_once()
        url: str = mock_client.post.call_args.args[0]
        headers: dict[str, str] = mock_client.post.call_args.kwargs["headers"]
        body: dict[str, object] = mock_client.post.call_args.kwargs["json"]

        assert url == "https://apex.example.com/v1/internal/gpu-sessions/sid-123/provisioning"
        assert headers["Authorization"] == "Bearer tok-xyz"
        assert headers["Content-Type"] == "application/json"
        assert body["session_id"] == "sid-123"
        assert body["phase"] == "downloading"
        assert body["message"] == "Downloading 1 model files"
        assert "elapsed_seconds" in body
        assert "ts" in body

    async def test_trailing_slash_in_callback_url_is_stripped(self) -> None:
        reporter = _make_reporter(callback_url="https://apex.example.com/")
        mock_client = _make_async_http_client()

        with patch("ai_content_service.provisioning_reporter.httpx.AsyncClient") as mock_cls:
            mock_cls.return_value = mock_client
            await reporter.phase("comfyui")

        url: str = mock_client.post.call_args.args[0]
        assert "//v1" not in url
        assert "/v1/internal/gpu-sessions/sid-123/provisioning" in url

    async def test_ready_sends_ready_phase(self) -> None:
        reporter = _make_reporter()
        with patch.object(reporter, "_post") as mock_post:
            await reporter.ready()
        mock_post.assert_called_once()
        payload: dict[str, object] = mock_post.call_args.args[0]
        assert payload["phase"] == "ready"
        assert payload["error"] is None
        assert payload["download"] is None

    async def test_failed_sends_error(self) -> None:
        reporter = _make_reporter()
        with patch.object(reporter, "_post") as mock_post:
            await reporter.failed("out of disk space")
        mock_post.assert_called_once()
        payload: dict[str, object] = mock_post.call_args.args[0]
        assert payload["phase"] == "failed"
        assert payload["error"] == "out of disk space"

    async def test_download_progress_includes_download_stats(self) -> None:
        reporter = _make_reporter()
        # Force pct_ok = True so throttle passes unconditionally
        reporter._last_progress_pct = -100.0

        with (
            patch("ai_content_service.provisioning_reporter.time") as mock_time,
            patch.object(reporter, "_post") as mock_post,
        ):
            mock_time.monotonic.return_value = 100.0  # time_ok = True (100 - 0 >= 3)
            await reporter.download_progress(1_000, 10_000, 0, 1)

        mock_post.assert_called_once()
        payload: dict[str, object] = mock_post.call_args.args[0]
        assert payload["phase"] == "downloading"
        dl = payload["download"]
        assert isinstance(dl, dict)
        assert dl["bytes_done"] == 1_000
        assert dl["bytes_total"] == 10_000
        assert dl["files_done"] == 0
        assert dl["files_total"] == 1


# ---------------------------------------------------------------------------
# New: from_settings + SecretStr unwrapping (Fix #1)
# ---------------------------------------------------------------------------


class TestFromSettings:
    def _settings_enabled(self) -> Settings:
        return Settings(
            apex_session_id="sid-42",
            apex_callback_url="https://apex.example.com",
            apex_callback_token=SecretStr("secret-token"),
        )

    async def test_from_settings_enabled_posts_with_token(self) -> None:
        reporter = ProvisioningReporter.from_settings(self._settings_enabled())
        assert reporter._enabled

        mock_client = _make_async_http_client()
        with patch("ai_content_service.provisioning_reporter.httpx.AsyncClient") as mock_cls:
            mock_cls.return_value = mock_client
            await reporter.phase("comfyui")

        mock_client.post.assert_called_once()
        headers: dict[str, str] = mock_client.post.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer secret-token"

    def test_from_settings_disabled_when_token_missing(self) -> None:
        settings = Settings(
            apex_session_id="sid",
            apex_callback_url="https://apex.example.com",
            apex_callback_token=None,
        )
        reporter = ProvisioningReporter.from_settings(settings)
        assert not reporter._enabled

    def test_from_settings_unwraps_secretstr(self) -> None:
        reporter = ProvisioningReporter.from_settings(self._settings_enabled())
        assert isinstance(reporter._callback_token, str)
        assert not isinstance(reporter._callback_token, SecretStr)
        assert reporter._callback_token == "secret-token"


# ---------------------------------------------------------------------------
# New: first failure visible at WARNING (Fix #2)
# ---------------------------------------------------------------------------


class TestFirstFailureVisibility:
    async def test_first_failure_warns_then_debug(self, caplog: pytest.LogCaptureFixture) -> None:
        reporter = _make_reporter()
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=ConnectionError("refused"))

        with (
            caplog.at_level(logging.DEBUG, logger="ai_content_service.provisioning_reporter"),
            patch("ai_content_service.provisioning_reporter.httpx.AsyncClient") as mock_cls,
        ):
            mock_cls.return_value = mock_client
            await reporter.phase("comfyui")  # first failure → WARNING
            await reporter.phase("workflow")  # second failure → DEBUG

        records = [
            r for r in caplog.records if r.name == "ai_content_service.provisioning_reporter"
        ]
        levels = [r.levelno for r in records]
        assert logging.WARNING in levels, "first failure must log at WARNING"
        assert logging.DEBUG in levels, "second failure must log at DEBUG"
        assert levels.index(logging.WARNING) < levels.index(logging.DEBUG)

    async def test_recovery_rearms_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """After a success, the next failure should warn again."""
        reporter = _make_reporter()
        mock_client = AsyncMock()

        with (
            caplog.at_level(logging.DEBUG, logger="ai_content_service.provisioning_reporter"),
            patch("ai_content_service.provisioning_reporter.httpx.AsyncClient") as mock_cls,
        ):
            mock_cls.return_value = mock_client

            # First: success → _callback_ok stays True
            mock_client.post = AsyncMock(return_value=MagicMock())
            await reporter.phase("comfyui")

            # Then: failure → WARNING (re-armed)
            mock_client.post = AsyncMock(side_effect=ConnectionError("refused"))
            await reporter.phase("workflow")

        records = [
            r for r in caplog.records if r.name == "ai_content_service.provisioning_reporter"
        ]
        assert any(r.levelno == logging.WARNING for r in records)


# ---------------------------------------------------------------------------
# New: _post response inspection hardening
# ---------------------------------------------------------------------------


class TestPostHardening:
    def _setup(self, mock_cls: MagicMock, response: httpx.Response) -> ProvisioningReporter:
        reporter = _make_reporter()
        _patch_post(mock_cls, response)
        return reporter

    async def test_200_json_sets_callback_ok_and_logs_info_once(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with (
            caplog.at_level(logging.DEBUG, logger="ai_content_service.provisioning_reporter"),
            patch("ai_content_service.provisioning_reporter.httpx.AsyncClient") as mock_cls,
        ):
            reporter = self._setup(mock_cls, _resp(200, "application/json", b'{"ok": true}'))
            await reporter.phase("comfyui")
            await reporter.phase("workflow")  # second success — no extra INFO

        records = [
            r for r in caplog.records if r.name == "ai_content_service.provisioning_reporter"
        ]
        assert reporter._callback_ok is True
        info_records = [r for r in records if r.levelno == logging.INFO]
        assert len(info_records) == 1
        assert "reaching Apex" in info_records[0].message
        assert all(r.levelno != logging.WARNING for r in records)

    async def test_200_html_warns_with_frontend_hint(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with (
            caplog.at_level(logging.DEBUG, logger="ai_content_service.provisioning_reporter"),
            patch("ai_content_service.provisioning_reporter.httpx.AsyncClient") as mock_cls,
        ):
            reporter = self._setup(mock_cls, _resp(200, "text/html", b"<html>"))
            await reporter.phase("comfyui")

        records = [
            r for r in caplog.records if r.name == "ai_content_service.provisioning_reporter"
        ]
        assert reporter._callback_ok is False
        assert any(r.levelno == logging.WARNING for r in records)
        warn_msg = next(r.message for r in records if r.levelno == logging.WARNING)
        assert "not JSON" in warn_msg
        assert "frontend" in warn_msg or "static host" in warn_msg

    async def test_405_warns_with_static_host_hint(self, caplog: pytest.LogCaptureFixture) -> None:
        with (
            caplog.at_level(logging.DEBUG, logger="ai_content_service.provisioning_reporter"),
            patch("ai_content_service.provisioning_reporter.httpx.AsyncClient") as mock_cls,
        ):
            reporter = self._setup(mock_cls, _resp(405))
            await reporter.phase("comfyui")

        records = [
            r for r in caplog.records if r.name == "ai_content_service.provisioning_reporter"
        ]
        assert reporter._callback_ok is False
        warn_msg = next(r.message for r in records if r.levelno == logging.WARNING)
        assert "405 Method Not Allowed" in warn_msg
        assert "static host" in warn_msg or "APEX_CALLBACK_URL" in warn_msg

    async def test_302_cloudflare_access_warns_with_auth_proxy_hint(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with (
            caplog.at_level(logging.DEBUG, logger="ai_content_service.provisioning_reporter"),
            patch("ai_content_service.provisioning_reporter.httpx.AsyncClient") as mock_cls,
        ):
            reporter = self._setup(
                mock_cls,
                _resp(302, location="https://team.cloudflareaccess.com/cdn-cgi/access/login"),
            )
            await reporter.phase("comfyui")

        records = [
            r for r in caplog.records if r.name == "ai_content_service.provisioning_reporter"
        ]
        assert reporter._callback_ok is False
        warn_msg = next(r.message for r in records if r.levelno == logging.WARNING)
        assert "HTTP 302" in warn_msg
        assert "auth proxy" in warn_msg

    async def test_401_json_warns_without_auth_proxy_hint(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Genuine Apex 401 must not be mislabeled as auth-proxy."""
        with (
            caplog.at_level(logging.DEBUG, logger="ai_content_service.provisioning_reporter"),
            patch("ai_content_service.provisioning_reporter.httpx.AsyncClient") as mock_cls,
        ):
            reporter = self._setup(
                mock_cls, _resp(401, "application/json", b'{"error": "unauthorized"}')
            )
            await reporter.phase("comfyui")

        records = [
            r for r in caplog.records if r.name == "ai_content_service.provisioning_reporter"
        ]
        assert reporter._callback_ok is False
        warn_msg = next(r.message for r in records if r.levelno == logging.WARNING)
        assert "HTTP 401" in warn_msg
        assert "auth proxy" not in warn_msg

    async def test_302_non_cloudflare_redirect_is_generic(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A redirect without Cloudflare/HTML indicators is generic, not an auth-proxy hint."""
        with (
            caplog.at_level(logging.DEBUG, logger="ai_content_service.provisioning_reporter"),
            patch("ai_content_service.provisioning_reporter.httpx.AsyncClient") as mock_cls,
        ):
            reporter = self._setup(
                mock_cls,
                _resp(302, "application/json", location="https://example.com/elsewhere"),
            )
            await reporter.phase("comfyui")

        records = [
            r for r in caplog.records if r.name == "ai_content_service.provisioning_reporter"
        ]
        assert reporter._callback_ok is False
        warn_msg = next(r.message for r in records if r.levelno == logging.WARNING)
        assert "HTTP 302" in warn_msg
        assert "auth proxy" not in warn_msg

    async def test_403_json_is_generic_not_auth_proxy(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """403 JSON (no Cloudflare/HTML) is a plain failure, symmetric with the 401 JSON case."""
        with (
            caplog.at_level(logging.DEBUG, logger="ai_content_service.provisioning_reporter"),
            patch("ai_content_service.provisioning_reporter.httpx.AsyncClient") as mock_cls,
        ):
            reporter = self._setup(
                mock_cls, _resp(403, "application/json", b'{"error": "forbidden"}')
            )
            await reporter.phase("comfyui")

        records = [
            r for r in caplog.records if r.name == "ai_content_service.provisioning_reporter"
        ]
        warn_msg = next(r.message for r in records if r.levelno == logging.WARNING)
        assert "HTTP 403" in warn_msg
        assert "auth proxy" not in warn_msg

    async def test_200_missing_content_type_emits_non_api_hint(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """2xx with no content-type still fails as a non-API response (covers 204/empty-ctype)."""
        with (
            caplog.at_level(logging.DEBUG, logger="ai_content_service.provisioning_reporter"),
            patch("ai_content_service.provisioning_reporter.httpx.AsyncClient") as mock_cls,
        ):
            reporter = self._setup(mock_cls, _resp(200))  # no content_type
            await reporter.phase("comfyui")

        records = [
            r for r in caplog.records if r.name == "ai_content_service.provisioning_reporter"
        ]
        assert reporter._callback_ok is False
        warn_msg = next(r.message for r in records if r.levelno == logging.WARNING)
        assert "not JSON" in warn_msg
        assert "APEX_CALLBACK_URL" in warn_msg

    async def test_200_text_plain_emits_non_api_hint(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """2xx with a non-JSON, non-HTML content-type still fails as a non-API response."""
        with (
            caplog.at_level(logging.DEBUG, logger="ai_content_service.provisioning_reporter"),
            patch("ai_content_service.provisioning_reporter.httpx.AsyncClient") as mock_cls,
        ):
            reporter = self._setup(mock_cls, _resp(200, "text/plain", b"ok"))
            await reporter.phase("comfyui")

        records = [
            r for r in caplog.records if r.name == "ai_content_service.provisioning_reporter"
        ]
        assert reporter._callback_ok is False
        warn_msg = next(r.message for r in records if r.levelno == logging.WARNING)
        assert "not JSON" in warn_msg
        assert "APEX_CALLBACK_URL" in warn_msg

    async def test_403_html_warns_with_auth_proxy_hint(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with (
            caplog.at_level(logging.DEBUG, logger="ai_content_service.provisioning_reporter"),
            patch("ai_content_service.provisioning_reporter.httpx.AsyncClient") as mock_cls,
        ):
            reporter = self._setup(mock_cls, _resp(403, "text/html", b"<html>"))
            await reporter.phase("comfyui")

        records = [
            r for r in caplog.records if r.name == "ai_content_service.provisioning_reporter"
        ]
        assert reporter._callback_ok is False
        warn_msg = next(r.message for r in records if r.levelno == logging.WARNING)
        assert "HTTP 403" in warn_msg
        assert "auth proxy" in warn_msg

    async def test_500_warns_generic(self, caplog: pytest.LogCaptureFixture) -> None:
        with (
            caplog.at_level(logging.DEBUG, logger="ai_content_service.provisioning_reporter"),
            patch("ai_content_service.provisioning_reporter.httpx.AsyncClient") as mock_cls,
        ):
            reporter = self._setup(mock_cls, _resp(500))
            await reporter.phase("comfyui")

        records = [
            r for r in caplog.records if r.name == "ai_content_service.provisioning_reporter"
        ]
        assert reporter._callback_ok is False
        warn_msg = next(r.message for r in records if r.levelno == logging.WARNING)
        assert "HTTP 500" in warn_msg

    async def test_transport_exception_warns_request_error(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        reporter = _make_reporter()
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("timed out"))

        with (
            caplog.at_level(logging.DEBUG, logger="ai_content_service.provisioning_reporter"),
            patch("ai_content_service.provisioning_reporter.httpx.AsyncClient") as mock_cls,
        ):
            mock_cls.return_value = mock_client
            await reporter.phase("comfyui")

        records = [
            r for r in caplog.records if r.name == "ai_content_service.provisioning_reporter"
        ]
        assert reporter._callback_ok is False
        warn_msg = next(r.message for r in records if r.levelno == logging.WARNING)
        assert "request error" in warn_msg

    async def test_warn_once_then_debug_on_consecutive_failures(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        reporter = _make_reporter()
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=_resp(500))

        with (
            caplog.at_level(logging.DEBUG, logger="ai_content_service.provisioning_reporter"),
            patch("ai_content_service.provisioning_reporter.httpx.AsyncClient") as mock_cls,
        ):
            mock_cls.return_value = mock_client
            await reporter.phase("comfyui")  # first failure → WARNING
            await reporter.phase("workflow")  # second failure → DEBUG

        records = [
            r for r in caplog.records if r.name == "ai_content_service.provisioning_reporter"
        ]
        levels = [r.levelno for r in records]
        assert levels.count(logging.WARNING) == 1
        assert logging.DEBUG in levels
        assert levels.index(logging.WARNING) < levels.index(logging.DEBUG)

    async def test_one_time_success_info(self, caplog: pytest.LogCaptureFixture) -> None:
        reporter = _make_reporter()
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=_resp(200, "application/json", b'{"ok":true}'))

        with (
            caplog.at_level(logging.DEBUG, logger="ai_content_service.provisioning_reporter"),
            patch("ai_content_service.provisioning_reporter.httpx.AsyncClient") as mock_cls,
        ):
            mock_cls.return_value = mock_client
            await reporter.phase("comfyui")
            await reporter.phase("workflow")

        records = [
            r for r in caplog.records if r.name == "ai_content_service.provisioning_reporter"
        ]
        info_records = [r for r in records if r.levelno == logging.INFO]
        assert len(info_records) == 1

    async def test_token_never_appears_in_logs(self, caplog: pytest.LogCaptureFixture) -> None:
        reporter = _make_reporter(callback_token="super-secret-token")
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=_resp(401, "application/json", b"{}"))

        with (
            caplog.at_level(logging.DEBUG, logger="ai_content_service.provisioning_reporter"),
            patch("ai_content_service.provisioning_reporter.httpx.AsyncClient") as mock_cls,
        ):
            mock_cls.return_value = mock_client
            await reporter.phase("comfyui")

        for record in caplog.records:
            assert "super-secret-token" not in record.getMessage()


# ---------------------------------------------------------------------------
# 3. Failures are swallowed
# ---------------------------------------------------------------------------


class TestFailuresSwallowed:
    async def test_network_error_does_not_propagate(self) -> None:
        reporter = _make_reporter()
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=ConnectionError("refused"))

        with patch("ai_content_service.provisioning_reporter.httpx.AsyncClient") as mock_cls:
            mock_cls.return_value = mock_client
            # Must not raise
            await reporter.phase("comfyui")
            await reporter.ready()

    async def test_http_500_does_not_propagate(self) -> None:
        reporter = _make_reporter()
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=_resp(500))

        with patch("ai_content_service.provisioning_reporter.httpx.AsyncClient") as mock_cls:
            mock_cls.return_value = mock_client
            await reporter.phase("comfyui")  # must not raise

    async def test_timeout_does_not_propagate(self) -> None:
        reporter = _make_reporter()
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("timed out"))

        with patch("ai_content_service.provisioning_reporter.httpx.AsyncClient") as mock_cls:
            mock_cls.return_value = mock_client
            await reporter.ready()  # must not raise


# ---------------------------------------------------------------------------
# 4. Throttling
# ---------------------------------------------------------------------------


class TestDownloadProgressThrottling:
    def _reporter_at_50pct(self) -> ProvisioningReporter:
        r = _make_reporter()
        r._last_progress_ts = 1000.0
        r._last_progress_pct = 50.0
        return r

    async def test_rapid_calls_suppressed_within_time_and_pct_window(self) -> None:
        reporter = self._reporter_at_50pct()

        with (
            patch("ai_content_service.provisioning_reporter.time") as mock_time,
            patch.object(reporter, "_post") as mock_post,
        ):
            mock_time.monotonic.return_value = 1001.0  # 1s elapsed < 3s
            # 52% — only 2% jump, less than 5% threshold
            await reporter.download_progress(520, 1000, 0, 1)
            mock_post.assert_not_called()

    async def test_time_window_triggers_emit(self) -> None:
        reporter = self._reporter_at_50pct()

        with (
            patch("ai_content_service.provisioning_reporter.time") as mock_time,
            patch.object(reporter, "_post") as mock_post,
        ):
            mock_time.monotonic.return_value = 1004.0  # 4s elapsed >= 3s
            await reporter.download_progress(520, 1000, 0, 1)
            mock_post.assert_called_once()

    async def test_percent_jump_triggers_emit(self) -> None:
        reporter = self._reporter_at_50pct()

        with (
            patch("ai_content_service.provisioning_reporter.time") as mock_time,
            patch.object(reporter, "_post") as mock_post,
        ):
            mock_time.monotonic.return_value = 1001.0  # within 3s
            # 57% — 7% jump >= 5% threshold
            await reporter.download_progress(570, 1000, 0, 1)
            mock_post.assert_called_once()

    async def test_final_100pct_always_emits(self) -> None:
        r = _make_reporter()
        r._last_progress_ts = 1000.0
        r._last_progress_pct = 99.5  # only 0.5% away from 100%

        with (
            patch("ai_content_service.provisioning_reporter.time") as mock_time,
            patch.object(r, "_post") as mock_post,
        ):
            mock_time.monotonic.return_value = 1001.0  # within 3s; pct jump < 5%
            await r.download_progress(1000, 1000, 1, 1)
            mock_post.assert_called_once()


# ---------------------------------------------------------------------------
# 5. Elapsed seconds
# ---------------------------------------------------------------------------


class TestElapsedSeconds:
    async def test_elapsed_increases_across_events(self) -> None:
        reporter = _make_reporter(start_ts=0.0)
        reporter._last_progress_pct = -100.0  # bypass pct throttle

        payloads: list[dict[str, object]] = []

        async def capture_post(p: dict[str, object]) -> None:
            payloads.append(p)

        times = iter([10.0, 20.0, 50.0])

        with (
            patch("ai_content_service.provisioning_reporter.time") as mock_time,
            patch.object(reporter, "_post", side_effect=capture_post),
        ):
            mock_time.monotonic.side_effect = lambda: next(times)
            await reporter.phase("comfyui")
            await reporter.phase("workflow")
            await reporter.ready()

        elapsed_values: list[int] = []
        for p in payloads:
            es = p["elapsed_seconds"]
            assert isinstance(es, int)
            elapsed_values.append(es)

        assert elapsed_values == sorted(elapsed_values), "elapsed_seconds should be non-decreasing"
        assert elapsed_values[-1] > elapsed_values[0]


# ---------------------------------------------------------------------------
# 5b. HTTP client reuse and lifecycle (Fix #6 / P1-7)
# ---------------------------------------------------------------------------


class TestClientReuse:
    async def test_reporter_reuses_single_client(self) -> None:
        reporter = _make_reporter()
        mock_client = _make_async_http_client()

        with patch("ai_content_service.provisioning_reporter.httpx.AsyncClient") as mock_cls:
            mock_cls.return_value = mock_client
            await reporter.phase("comfyui")
            await reporter.phase("requirements")
            await reporter.phase("workflow")

        assert mock_cls.call_count == 1
        assert mock_client.post.call_count == 3

    async def test_reporter_closes_client_via_context_manager(self) -> None:
        reporter = _make_reporter()
        mock_client = _make_async_http_client()

        with patch("ai_content_service.provisioning_reporter.httpx.AsyncClient") as mock_cls:
            mock_cls.return_value = mock_client
            async with reporter:
                await reporter.phase("comfyui")
                await reporter.ready()

        mock_client.aclose.assert_called_once()
        assert reporter._client is None

    async def test_post_after_close_recreates_client_and_warns(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        reporter = _make_reporter()
        first_client = _make_async_http_client()
        second_client = _make_async_http_client()

        with (
            caplog.at_level("WARNING", logger="ai_content_service.provisioning_reporter"),
            patch("ai_content_service.provisioning_reporter.httpx.AsyncClient") as mock_cls,
        ):
            mock_cls.side_effect = [first_client, second_client]
            await reporter.phase("comfyui")
            await reporter.aclose()
            await reporter.phase("workflow")  # unexpected post-close event

        assert mock_cls.call_count == 2
        first_client.aclose.assert_called_once()
        second_client.post.assert_called_once()
        assert any(
            "after close" in r.message
            for r in caplog.records
            if r.name == "ai_content_service.provisioning_reporter"
        )

    async def test_aclose_is_a_noop_when_never_posted(self) -> None:
        reporter = _make_reporter()
        with patch("ai_content_service.provisioning_reporter.httpx.AsyncClient") as mock_cls:
            await reporter.aclose()
        mock_cls.assert_not_called()


# ---------------------------------------------------------------------------
# 6. Deployer phase sequence
# ---------------------------------------------------------------------------


@pytest.fixture
def full_bundle() -> BundleConfig:
    return BundleConfig(
        metadata=BundleMetadata(
            name="full_bundle",
            version="260101-01",
            description="Test",
            created_at=datetime.now(timezone.utc),
        ),
        custom_nodes=[
            CustomNodeConfig(
                name="TestNode",
                git_url="https://github.com/test/node",
                commit_sha="abc123",
            )
        ],
        models=[
            ModelConfig(
                name="Test Model",
                model_type="checkpoints",
                files=[
                    ModelFileConfig(
                        name="Checkpoint",
                        url="https://huggingface.co/test/model.safetensors",
                        filename="model.safetensors",
                    )
                ],
            )
        ],
        workflow_file="workflow.json",
    )


class TestDeployerPhaseSequence:
    async def test_full_deploy_emits_ordered_phases_ending_in_ready(
        self, full_bundle: BundleConfig
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = Path(tmpdir) / "bundle"
            bundle_path.mkdir()
            (bundle_path / "workflow.json").write_text("{}")

            settings = Settings(comfyui_path=Path(tmpdir) / "comfyui")

            mock_bundle_mgr = MagicMock()
            mock_bundle_mgr.load_bundle_config_from_path.return_value = full_bundle

            mock_comfyui = AsyncMock()
            mock_comfyui.verify = AsyncMock(return_value=True)

            mock_downloader = AsyncMock()
            mock_downloader.download_all = AsyncMock(return_value=1)

            mock_workflow = AsyncMock()

            emitted: list[str] = []

            class SpyReporter(ProvisioningReporter):
                async def phase(self, name: str, message: str = "") -> None:  # noqa: ARG002
                    emitted.append(name)

                async def ready(self) -> None:
                    emitted.append("ready")

                async def failed(self, error: str) -> None:
                    emitted.append(f"failed:{error}")

            spy = SpyReporter(
                session_id="s", callback_url="http://x", callback_token="t", enabled=True
            )

            deployer = Deployer(
                settings=settings,
                bundle_manager=mock_bundle_mgr,
                comfyui_manager=mock_comfyui,
                model_downloader=mock_downloader,
                workflow_manager=mock_workflow,
                reporter=spy,
            )

            result = await deployer.deploy_from_path(bundle_path)

        assert result.success is True
        assert emitted[-1] == "ready"
        assert "failed" not in " ".join(emitted)
        # Phase order: downloading before workflow, workflow before verifying
        assert emitted.index("downloading") < emitted.index("workflow")
        assert emitted.index("workflow") < emitted.index("verifying")
        assert emitted.index("verifying") < emitted.index("ready")

    async def test_failing_deploy_emits_failed_not_ready(self, full_bundle: BundleConfig) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = Path(tmpdir) / "bundle"
            bundle_path.mkdir()

            settings = Settings(comfyui_path=Path(tmpdir) / "comfyui")

            mock_bundle_mgr = MagicMock()
            mock_bundle_mgr.load_bundle_config_from_path.return_value = full_bundle

            mock_comfyui = AsyncMock()
            mock_downloader = AsyncMock()
            mock_downloader.download_all = AsyncMock(side_effect=RuntimeError("disk full"))
            mock_workflow = AsyncMock()

            emitted: list[str] = []

            class SpyReporter(ProvisioningReporter):
                async def phase(self, name: str, message: str = "") -> None:  # noqa: ARG002
                    emitted.append(name)

                async def ready(self) -> None:
                    emitted.append("ready")

                async def failed(self, error: str) -> None:
                    emitted.append(f"failed:{error}")

            spy = SpyReporter(
                session_id="s", callback_url="http://x", callback_token="t", enabled=True
            )

            deployer = Deployer(
                settings=settings,
                bundle_manager=mock_bundle_mgr,
                comfyui_manager=mock_comfyui,
                model_downloader=mock_downloader,
                workflow_manager=mock_workflow,
                reporter=spy,
            )

            result = await deployer.deploy_from_path(bundle_path)

        assert result.success is False
        assert any(e.startswith("failed:") for e in emitted)
        assert "ready" not in emitted

    async def test_skipped_steps_do_not_emit_phase(self, full_bundle: BundleConfig) -> None:
        """MODELS_ONLY mode must not emit comfyui/requirements/custom_nodes phases."""
        from ai_content_service.config import DeployMode

        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = Path(tmpdir) / "bundle"
            bundle_path.mkdir()
            (bundle_path / "workflow.json").write_text("{}")

            settings = Settings(comfyui_path=Path(tmpdir) / "comfyui")

            mock_bundle_mgr = MagicMock()
            mock_bundle_mgr.load_bundle_config_from_path.return_value = full_bundle

            mock_comfyui = AsyncMock()
            mock_comfyui.verify = AsyncMock(return_value=True)
            mock_downloader = AsyncMock()
            mock_downloader.download_all = AsyncMock(return_value=1)
            mock_workflow = AsyncMock()

            emitted: list[str] = []

            class SpyReporter(ProvisioningReporter):
                async def phase(self, name: str, message: str = "") -> None:  # noqa: ARG002
                    emitted.append(name)

                async def ready(self) -> None:
                    emitted.append("ready")

                async def failed(self, error: str) -> None:
                    emitted.append(f"failed:{error}")

            spy = SpyReporter(
                session_id="s", callback_url="http://x", callback_token="t", enabled=True
            )

            deployer = Deployer(
                settings=settings,
                bundle_manager=mock_bundle_mgr,
                comfyui_manager=mock_comfyui,
                model_downloader=mock_downloader,
                workflow_manager=mock_workflow,
                reporter=spy,
            )

            await deployer.deploy_from_path(bundle_path, mode=DeployMode.MODELS_ONLY)

        assert "comfyui" not in emitted
        assert "requirements" not in emitted
        assert "custom_nodes" not in emitted
        assert "downloading" in emitted


# ---------------------------------------------------------------------------
# 7. Downloader callback
# ---------------------------------------------------------------------------


class TestDownloaderCallback:
    async def test_on_progress_called_with_increasing_bytes(self) -> None:
        settings = Settings()
        downloader = ModelDownloader(settings)

        model = ModelConfig(
            name="test",
            model_type="checkpoints",
            files=[
                ModelFileConfig(
                    name="f1",
                    url="https://example.com/f1.safetensors",
                    filename="f1.safetensors",
                    size_bytes=1000,
                ),
                ModelFileConfig(
                    name="f2",
                    url="https://example.com/f2.safetensors",
                    filename="f2.safetensors",
                    size_bytes=2000,
                ),
            ],
        )

        progress_calls: list[tuple[int, int, int, int]] = []

        async def on_progress(
            bytes_done: int, bytes_total: int, files_done: int, files_total: int
        ) -> None:
            progress_calls.append((bytes_done, bytes_total, files_done, files_total))

        async def fake_download_file(
            file: ModelFileConfig,
            _path: Path,
            _progress_obj: object,
            _task_id: object,
            on_bytes: Callable[[int], Awaitable[None]] | None = None,
        ) -> None:
            if on_bytes is not None:
                await on_bytes(file.size_bytes or 0)

        with (
            patch.object(downloader, "_download_file", side_effect=fake_download_file),
            tempfile.TemporaryDirectory() as tmpdir,
        ):
            result = await downloader.download_all([model], Path(tmpdir), on_progress=on_progress)

        assert result == 2
        assert progress_calls

        bytes_done_seq = [c[0] for c in progress_calls]
        assert all(a <= b for a, b in itertools.pairwise(bytes_done_seq)), (
            "bytes_done must be non-decreasing"
        )
        assert all(c[1] == 3000 for c in progress_calls)  # bytes_total = 1000 + 2000
        assert all(c[3] == 2 for c in progress_calls)  # files_total is always 2

    async def test_on_progress_none_behaves_as_today(self) -> None:
        settings = Settings()
        downloader = ModelDownloader(settings)

        model = ModelConfig(
            name="test",
            model_type="checkpoints",
            files=[
                ModelFileConfig(
                    name="f1",
                    url="https://example.com/f1.safetensors",
                    filename="f1.safetensors",
                )
            ],
        )

        async def fake_download_file(
            file: ModelFileConfig,
            path: Path,
            progress_obj: object,
            task_id: object,
            on_bytes: Callable[[int], Awaitable[None]] | None = None,
        ) -> None:
            pass

        with (
            patch.object(downloader, "_download_file", side_effect=fake_download_file),
            tempfile.TemporaryDirectory() as tmpdir,
        ):
            result = await downloader.download_all([model], Path(tmpdir))

        assert result == 1

    async def test_files_done_increments_after_each_file(self) -> None:
        settings = Settings()
        downloader = ModelDownloader(settings)

        model = ModelConfig(
            name="test",
            model_type="checkpoints",
            files=[
                ModelFileConfig(
                    name="f1",
                    url="https://example.com/f1.safetensors",
                    filename="f1.safetensors",
                    size_bytes=500,
                ),
                ModelFileConfig(
                    name="f2",
                    url="https://example.com/f2.safetensors",
                    filename="f2.safetensors",
                    size_bytes=500,
                ),
            ],
        )

        files_done_values: list[int] = []

        async def on_progress(
            _bytes_done: int, _bytes_total: int, files_done: int, _files_total: int
        ) -> None:
            files_done_values.append(files_done)

        async def fake_download_file(
            file: ModelFileConfig,
            _path: Path,
            _progress_obj: object,
            _task_id: object,
            on_bytes: Callable[[int], Awaitable[None]] | None = None,
        ) -> None:
            if on_bytes is not None:
                await on_bytes(file.size_bytes or 0)

        with (
            patch.object(downloader, "_download_file", side_effect=fake_download_file),
            tempfile.TemporaryDirectory() as tmpdir,
        ):
            await downloader.download_all([model], Path(tmpdir), on_progress=on_progress)

        # The per-file "file-complete" on_progress call should reach 2
        assert max(files_done_values) == 2
