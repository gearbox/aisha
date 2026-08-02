"""Tests for preflight (Typer-free core of `acs models check`, D14).

R1a: the probe streams and never reads a body, so a Range-ignoring host costs
one round trip, not a multi-GB download. Mocks therefore patch
``httpx.AsyncClient().stream`` (an async context manager), not ``.get()``.
"""

from __future__ import annotations

import asyncio
import math
import time
import tracemalloc
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import yaml
from pydantic import SecretStr
from pydantic_core import ValidationError as PydanticValidationError
from rich.console import Console

from ai_content_service import preflight as preflight_module
from ai_content_service.config import (
    BundleConfig,
    BundleMetadata,
    ModelConfig,
    ModelFileConfig,
    Settings,
)
from ai_content_service.download_auth import (
    BoundCredential,
    CredentialEgressError,
    build_credentials,
    build_registry,
)
from ai_content_service.downloader import _can_obtain_without_url
from ai_content_service.preflight import (
    BundleCheckResult,
    MultiBundleReport,
    _ProbeResult,
    _row_style,
    check_all_bundles,
    check_bundle,
    check_bundle_path,
    multi_report_to_dict,
    render_report,
    report_to_dict,
)

if TYPE_CHECKING:
    from contextlib import AbstractContextManager
    from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bundle_with_files(files: list[tuple[str, str]]) -> BundleConfig:
    """Build a single-model bundle from (filename, url) pairs."""
    return BundleConfig(
        metadata=BundleMetadata(
            name="test_bundle",
            version="260101-01",
            created_at=datetime.now(timezone.utc),
        ),
        models=[
            ModelConfig(
                name="Test Model",
                model_type="checkpoints",
                files=[
                    ModelFileConfig(name=filename, url=url, filename=filename)
                    for filename, url in files
                ],
            )
        ],
    )


_R2_SHA256 = "a" * 64


def _r2_settings() -> Settings:
    return Settings(
        r2_s3_endpoint="https://account.r2.cloudflarestorage.com",
        r2_readonly_access_key_id="READKEY",
        r2_readonly_secret_access_key=SecretStr("readsecret"),
    )


def _bundle_with_model_file(file: ModelFileConfig) -> BundleConfig:
    return BundleConfig(
        metadata=BundleMetadata(
            name="test_bundle",
            version="260101-01",
            created_at=datetime.now(timezone.utc),
        ),
        models=[
            ModelConfig(
                name="Test Model",
                model_type="checkpoints",
                files=[file],
            )
        ],
    )


def _mock_stream_response(
    status_code: int = 200,
    content_type: str = "",
    content_disposition: str | None = None,
    content_length: int | None = None,
    content_range: str | None = None,
    body_chunk: bytes | None = None,
) -> MagicMock:
    """Build a headers-only httpx-response mock for `client.stream(...)`.

    If *body_chunk* is set, `aiter_bytes`/`aread` would return it -- but R1a's
    probe must never call either, so tests can use this to prove the body is
    never touched.
    """
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    headers: dict[str, str] = {}
    if content_type:
        headers["content-type"] = content_type
    if content_disposition is not None:
        headers["content-disposition"] = content_disposition
    if content_length is not None:
        headers["content-length"] = str(content_length)
    if content_range is not None:
        headers["content-range"] = content_range
    resp.headers = headers

    if body_chunk is not None:

        async def _aiter_bytes(chunk_size: int | None = None) -> object:  # noqa: ARG001
            yield body_chunk

        async def _aread() -> bytes:
            return body_chunk

        resp.aiter_bytes = _aiter_bytes
        resp.aread = _aread

    return resp


def _make_async_cm(return_value: object) -> MagicMock:
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=return_value)
    cm.__aexit__ = AsyncMock(return_value=None)
    return cm


def _patch_client(mock_client: MagicMock) -> AbstractContextManager[MagicMock]:
    return patch(
        "ai_content_service.preflight.httpx.AsyncClient", return_value=_make_async_cm(mock_client)
    )


def _stream_client(*responses: object) -> MagicMock:
    """A mock httpx.AsyncClient whose `.stream(...)` yields *responses* in order.

    Each entry is either a response mock (wrapped in an async CM) or an
    Exception instance (raised synchronously by the `.stream()` call itself,
    mirroring a real connection/URL error before any request is sent).
    """
    mock_client = MagicMock()
    mock_client.stream = MagicMock(
        side_effect=[r if isinstance(r, Exception) else _make_async_cm(r) for r in responses]
    )
    return mock_client


# ---------------------------------------------------------------------------
# check_bundle
# ---------------------------------------------------------------------------


class TestCheckBundle:
    async def test_missing_snapshot_url_is_actionable_without_a_request(self) -> None:
        bundle = _bundle_with_files([("model.safetensors", "")])
        settings = Settings()
        mock_client = _stream_client()

        with _patch_client(mock_client):
            report = await check_bundle(bundle, settings)

        assert report.ok is False
        result = report.results[0]
        assert result.status == "MISSING URL"
        assert result.flag is not None
        assert "source URL" in result.flag
        mock_client.stream.assert_not_called()

    async def test_mixed_statuses_produce_correct_rows(self) -> None:
        bundle = _bundle_with_files(
            [
                ("ok.safetensors", "https://example.com/ok"),
                ("missing.safetensors", "https://example.com/missing"),
            ]
        )
        settings = Settings()

        mock_client = _stream_client(
            _mock_stream_response(200, content_type="application/octet-stream", content_length=123),
            _mock_stream_response(404),
        )

        with _patch_client(mock_client):
            report = await check_bundle(bundle, settings)

        assert report.ok is False
        assert len(report.results) == 2

        ok_result, missing_result = report.results
        assert ok_result.ok is True
        assert ok_result.status == "200"
        assert ok_result.content_length == 123

        assert missing_result.ok is False
        assert missing_result.status == "404"
        assert missing_result.flag is not None

    async def test_html_response_flagged_as_auth_domain_problem(self) -> None:
        bundle = _bundle_with_files(
            [("model.safetensors", "https://civitai.red/api/download/models/1")]
        )
        settings = Settings()

        mock_client = _stream_client(
            _mock_stream_response(200, content_type="text/html; charset=utf-8")
        )

        with _patch_client(mock_client):
            report = await check_bundle(bundle, settings)

        assert report.ok is False
        result = report.results[0]
        assert result.ok is False
        assert result.flag is not None
        assert "html" in result.flag.lower() or "auth" in result.flag.lower()

    async def test_civitai_404_flags_nsfw_suggestion(self) -> None:
        bundle = _bundle_with_files(
            [("model.safetensors", "https://civitai.com/api/download/models/1")]
        )
        settings = Settings()

        mock_client = _stream_client(_mock_stream_response(404))

        with _patch_client(mock_client):
            report = await check_bundle(bundle, settings)

        assert report.results[0].flag is not None
        assert "civitai.red" in report.results[0].flag

    async def test_all_ok_reports_ok_true(self) -> None:
        bundle = _bundle_with_files(
            [
                ("a.safetensors", "https://example.com/a"),
                ("b.safetensors", "https://example.com/b"),
            ]
        )
        settings = Settings()

        mock_client = _stream_client(
            _mock_stream_response(206, content_type="application/octet-stream"),
            _mock_stream_response(206, content_type="application/octet-stream"),
        )

        with _patch_client(mock_client):
            report = await check_bundle(bundle, settings)

        assert report.ok is True
        assert all(r.ok for r in report.results)
        assert all(r.range_supported for r in report.results)

    async def test_civitai_401_falls_back_to_query_token(self) -> None:
        bundle = _bundle_with_files(
            [("model.safetensors", "https://civitai.red/api/download/models/1")]
        )
        settings = Settings(
            civitai_api_token="secret-token",  # type: ignore[arg-type]
            civitai_allow_query_token_fallback=True,
        )

        mock_client = _stream_client(
            _mock_stream_response(401),
            _mock_stream_response(200, content_type="application/octet-stream"),
        )

        with _patch_client(mock_client):
            report = await check_bundle(bundle, settings)

        assert mock_client.stream.call_count == 2
        first_headers = mock_client.stream.call_args_list[0].kwargs["headers"]
        second_url = mock_client.stream.call_args_list[1].args[1]
        assert "Authorization" in first_headers
        assert "token=secret-token" in second_url
        assert report.ok is True

    async def test_result_url_is_redacted(self) -> None:
        bundle = _bundle_with_files(
            [("model.safetensors", "https://example.com/x?token=should_not_appear")]
        )
        settings = Settings()

        mock_client = _stream_client(_mock_stream_response(200))

        with _patch_client(mock_client):
            report = await check_bundle(bundle, settings)

        assert "should_not_appear" not in report.results[0].url

    async def test_request_error_reported_as_failure(self) -> None:
        bundle = _bundle_with_files([("model.safetensors", "https://example.com/x")])
        settings = Settings()

        mock_client = _stream_client(httpx.ConnectError("boom"))

        with _patch_client(mock_client):
            report = await check_bundle(bundle, settings)

        assert report.ok is False
        assert report.results[0].status.startswith("ERROR")
        assert report.results[0].flag == "request failed"


class TestR2ExpectedRows:
    @pytest.mark.parametrize(
        ("r2_configured", "sha256", "expected_status", "expected_ok"),
        [
            (True, _R2_SHA256, "R2 BY SHA256", True),
            (True, None, "MISSING URL", False),
            (False, _R2_SHA256, "MISSING URL", False),
        ],
    )
    @pytest.mark.parametrize("offline", [False, True])
    async def test_empty_url_uses_the_same_r2_rule_online_and_offline(
        self,
        r2_configured: bool,
        sha256: str | None,
        expected_status: str,
        expected_ok: bool,
        offline: bool,
    ) -> None:
        file = ModelFileConfig(
            name="model.safetensors",
            url="",
            filename="model.safetensors",
            sha256=sha256,
        )
        bundle = _bundle_with_model_file(file)
        settings = _r2_settings() if r2_configured else Settings()
        mock_client = _stream_client()

        with _patch_client(mock_client):
            report = await check_bundle(bundle, settings, offline=offline)

        result = report.results[0]
        assert result.status == expected_status
        assert result.ok is expected_ok
        assert report.ok is expected_ok
        assert result.range_supported is False
        assert not result.url
        mock_client.stream.assert_not_called()

    async def test_r2_lookup_is_resolved_once_per_run(self) -> None:
        bundle = _bundle_with_model_file(
            ModelFileConfig(
                name="model.safetensors",
                url="",
                filename="model.safetensors",
                sha256=_R2_SHA256,
            )
        )
        settings = _r2_settings()

        with patch.object(
            preflight_module,
            "read_creds_from_settings",
            wraps=preflight_module.read_creds_from_settings,
        ) as read_creds:
            report = await check_bundle(bundle, settings, offline=True)

        assert report.ok is True
        read_creds.assert_called_once_with(settings)

    @pytest.mark.parametrize(
        ("url", "sha256", "r2_configured"),
        [
            ("", None, False),
            ("", _R2_SHA256, False),
            ("", _R2_SHA256, True),
            ("https://example.com/model", None, False),
        ],
    )
    async def test_pass_fail_matches_downloader_obtainability_rule(
        self,
        tmp_path: Path,
        url: str,
        sha256: str | None,
        r2_configured: bool,
    ) -> None:
        """The no-URL half of preflight stays aligned with the downloader.

        The target is deliberately absent, so the downloader's on-disk skip
        route cannot affect the comparison.
        """
        file = ModelFileConfig(
            name="model.safetensors",
            url=url,
            filename="model.safetensors",
            sha256=sha256,
        )
        bundle = _bundle_with_model_file(file)
        settings = _r2_settings() if r2_configured else Settings()
        downloader_verdict = bool(url) or _can_obtain_without_url(
            file,
            tmp_path / "not-present.safetensors",
            skip_existing=True,
            r2_configured=r2_configured,
        )

        responses = (_mock_stream_response(206, content_type="application/octet-stream"),)
        with _patch_client(_stream_client(*responses)):
            report = await check_bundle(bundle, settings)

        assert report.ok is downloader_verdict


# ---------------------------------------------------------------------------
# R1a: the probe never reads the response body
# ---------------------------------------------------------------------------


class TestProbeNeverReadsBody:
    async def test_range_ignoring_host_never_drains_the_body(self) -> None:
        """A host that ignores Range and answers 200 with a large body must
        cost one round trip, not a multi-GB read -- and the real size must
        still be reported from Content-Length on that 200."""
        big_body = b"x" * (40 * 1024 * 1024)
        response = _mock_stream_response(
            200,
            content_type="application/octet-stream",
            content_length=len(big_body),
            body_chunk=big_body,
        )
        bundle = _bundle_with_files([("model.safetensors", "https://example.com/model")])
        settings = Settings()

        tracemalloc.start()
        try:
            with _patch_client(_stream_client(response)):
                report = await check_bundle(bundle, settings)
        finally:
            _current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

        assert peak < 2 * 1024 * 1024, f"probe allocated {peak} bytes; body must never be read"
        result = report.results[0]
        assert result.ok is True
        assert result.range_supported is False
        assert result.content_length == len(big_body)
        assert result.flag == "Range ignored — no resume support"

    async def test_probe_uses_stream_not_get(self) -> None:
        mock_client = _stream_client(_mock_stream_response(200))
        bundle = _bundle_with_files([("model.safetensors", "https://example.com/model")])
        settings = Settings()

        with _patch_client(mock_client):
            await check_bundle(bundle, settings)

        mock_client.stream.assert_called_once()
        mock_client.get.assert_not_called()


# ---------------------------------------------------------------------------
# R2a / M5a: size resolution from Content-Range / Content-Length
# ---------------------------------------------------------------------------


class TestSizeResolution:
    async def test_content_range_total_used_over_probe_slice_length(self) -> None:
        """The regression test for the bug: Content-Length on a 206 is the
        1-byte probe slice, not the file. Content-Range's total is correct."""
        response = _mock_stream_response(
            206,
            content_type="application/octet-stream",
            content_length=1,
            content_range="bytes 0-0/14203980000",
        )
        bundle = _bundle_with_files([("model.safetensors", "https://example.com/model")])
        settings = Settings()

        with _patch_client(_stream_client(response)):
            report = await check_bundle(bundle, settings)

        assert report.results[0].content_length == 14203980000

    async def test_content_range_wildcard_total_is_unknown(self) -> None:
        response = _mock_stream_response(
            206, content_type="application/octet-stream", content_range="bytes 0-0/*"
        )
        bundle = _bundle_with_files([("model.safetensors", "https://example.com/model")])
        settings = Settings()

        with _patch_client(_stream_client(response)):
            report = await check_bundle(bundle, settings)

        assert report.results[0].content_length is None

    async def test_malformed_headers_do_not_raise(self) -> None:
        """M5a: a malformed Content-Length/Content-Range must produce a dash,
        not an exception, in a tool whose job is to detect malformed input."""
        response = _mock_stream_response(
            206, content_type="application/octet-stream", content_range="bytes 0-0/banana"
        )
        response.headers["content-length"] = "banana"
        bundle = _bundle_with_files([("model.safetensors", "https://example.com/model")])
        settings = Settings()

        with _patch_client(_stream_client(response)):
            report = await check_bundle(bundle, settings)  # must not raise

        assert report.results[0].content_length is None


# ---------------------------------------------------------------------------
# M1a: a malformed URL red-flags one row instead of aborting the whole run
# ---------------------------------------------------------------------------


class TestMalformedUrlNeverAbortsRun:
    async def test_bad_url_produces_red_row_others_still_probed(self) -> None:
        """A syntactically valid but unresolvable/unconnectable URL: the error
        surfaces from `_probe` itself (mocked here), not from URL parsing --
        MY-6a now rejects malformed URLs earlier, at bundle-parse time
        (see TestMalformedUrlRejectedAtBoundary below)."""
        bundle = _bundle_with_files(
            [
                ("bad.safetensors", "https://bad.example/x"),
                ("ok.safetensors", "https://example.com/ok"),
            ]
        )
        settings = Settings()

        mock_client = _stream_client(
            ValueError("Invalid URL component: host"),
            _mock_stream_response(200, content_type="application/octet-stream"),
        )

        with _patch_client(mock_client):
            report = await check_bundle(bundle, settings)  # must not raise

        assert len(report.results) == 2
        bad, ok = report.results
        assert bad.ok is False
        assert bad.flag == "request failed"
        assert ok.ok is True

    async def test_invalid_url_not_a_subclass_of_http_error_is_still_caught(self) -> None:
        """httpx.InvalidURL is not an httpx.HTTPError subclass (the actual
        finding) -- M1a catches bare Exception, not httpx.HTTPError."""
        assert not issubclass(httpx.InvalidURL, httpx.HTTPError)

        bundle = _bundle_with_files([("model.safetensors", "https://bad")])
        settings = Settings()
        mock_client = _stream_client(httpx.InvalidURL("no host"))

        with _patch_client(mock_client):
            report = await check_bundle(bundle, settings)  # must not raise

        assert report.results[0].flag == "request failed"


# ---------------------------------------------------------------------------
# MY-6a: malformed URLs are now rejected at the bundle-config boundary
# ---------------------------------------------------------------------------


class TestMalformedUrlRejectedAtBoundary:
    def test_ipv6_malformed_url_rejected_by_model_file_config(self) -> None:
        with pytest.raises(PydanticValidationError):
            ModelFileConfig(name="bad", url="http://[::1", filename="bad.safetensors")


# ---------------------------------------------------------------------------
# E2 / M2a: _check_file is total by construction -- defence in depth for a
# file that somehow still reaches check_bundle with a broken URL (e.g. a
# legacy bundle loaded via `model_construct`, bypassing MY-6a's validator).
# ---------------------------------------------------------------------------


class TestCheckFileTotalByConstruction:
    async def test_malformed_url_bypassing_validation_produces_red_row_others_still_probed(
        self,
    ) -> None:
        """E2 regression: three files, the middle one carries a URL that would
        fail `resolve_policy`'s urlparse (the actual 01r escape) -- built via
        `model_construct` since MY-6a now blocks this at normal construction.
        `check_bundle` must still return three rows and probe the other two."""
        bad_file = ModelFileConfig.model_construct(
            name="bad", url="http://[::1", filename="bad.safetensors", sha256=None, size_bytes=None
        )
        first_file = ModelFileConfig(
            name="first", url="https://example.com/first", filename="first.safetensors"
        )
        third_file = ModelFileConfig(
            name="third", url="https://example.com/third", filename="third.safetensors"
        )
        bundle = BundleConfig(
            metadata=BundleMetadata(
                name="test_bundle", version="260101-01", created_at=datetime.now(timezone.utc)
            ),
            models=[
                ModelConfig(
                    name="Test Model",
                    model_type="checkpoints",
                    files=[first_file, bad_file, third_file],
                )
            ],
        )
        settings = Settings()

        mock_client = _stream_client(
            _mock_stream_response(200, content_type="application/octet-stream"),
            _mock_stream_response(200, content_type="application/octet-stream"),
        )

        with _patch_client(mock_client):
            report = await check_bundle(bundle, settings)  # must not raise

        assert len(report.results) == 3
        first, bad, third = report.results
        assert first.ok is True
        assert bad.ok is False
        assert bad.flag == "invalid URL or unresolvable host"
        assert third.ok is True

    async def test_check_file_totality_survives_arbitrary_exception_above_the_probe(self) -> None:
        """Totality is by construction, not by catching a specific known
        exception type -- a bare RuntimeError anywhere above `_probe` must
        still degrade to one red row instead of aborting `check_bundle`."""
        bundle = _bundle_with_files([("model.safetensors", "https://example.com/model")])
        settings = Settings()

        with patch("ai_content_service.preflight._probe", side_effect=RuntimeError("boom")):
            report = await check_bundle(bundle, settings)  # must not raise

        assert len(report.results) == 1
        assert report.results[0].ok is False


# ---------------------------------------------------------------------------
# R3b/R3c: the preflight egress guard is wired with the same token set the
# downloader uses.
# ---------------------------------------------------------------------------


class TestPreflightEgressGuardWiredWithSecrets:
    async def test_guard_receives_configured_tokens(self) -> None:
        settings = Settings(civitai_api_token="secret-token")  # type: ignore[arg-type]
        bundle = _bundle_with_files([("model.safetensors", "https://example.com/model")])

        captured_credentials: list[tuple[BoundCredential, ...]] = []
        real_make_guard = preflight_module._make_egress_guard

        def spy_make_guard(credentials: tuple[BoundCredential, ...]) -> object:
            captured_credentials.append(credentials)
            return real_make_guard(credentials)

        mock_client = _stream_client(_mock_stream_response(200))
        with (
            _patch_client(mock_client),
            patch("ai_content_service.preflight._make_egress_guard", side_effect=spy_make_guard),
        ):
            await check_bundle(bundle, settings)

        assert len(captured_credentials) == 1
        (credentials,) = captured_credentials
        assert [c.token for c in credentials] == ["secret-token"]
        assert [c.policy.name for c in credentials] == ["civitai"]

    async def test_make_egress_guard_forwards_credentials_to_assert_no_credential_egress(
        self,
    ) -> None:
        """Unit-level: the guard closure itself carries the credential set
        into every call, not just at construction time."""
        registry = build_registry(Settings())
        credentials = build_credentials(registry, {"civitai": "live-secret"})
        guard = preflight_module._make_egress_guard(credentials)
        request = httpx.Request(
            "GET", "https://evil.example/x", headers={"Authorization": "Bearer live-secret"}
        )
        with pytest.raises(CredentialEgressError):
            await guard(request)


# ---------------------------------------------------------------------------
# E3: a blank configured token behaves like no token at all
# ---------------------------------------------------------------------------


class TestBlankTokensBehaveAsAnonymous:
    async def test_empty_civitai_token_probes_anonymously_and_run_completes(self) -> None:
        settings = Settings(civitai_api_token="")  # type: ignore[arg-type]
        bundle = _bundle_with_files(
            [("model.safetensors", "https://civitai.red/api/download/models/1")]
        )

        mock_client = _stream_client(
            _mock_stream_response(200, content_type="application/octet-stream")
        )

        with _patch_client(mock_client):
            report = await check_bundle(bundle, settings)

        assert report.ok is True
        assert mock_client.stream.call_count == 1
        sent_headers = mock_client.stream.call_args.kwargs["headers"]
        assert "Authorization" not in sent_headers

    async def test_empty_civitai_token_yields_no_egress_guard_credentials(self) -> None:
        settings = Settings(civitai_api_token="")  # type: ignore[arg-type]
        bundle = _bundle_with_files([("model.safetensors", "https://example.com/model")])

        captured_credentials: list[tuple[BoundCredential, ...]] = []
        real_make_guard = preflight_module._make_egress_guard

        def spy_make_guard(credentials: tuple[BoundCredential, ...]) -> object:
            captured_credentials.append(credentials)
            return real_make_guard(credentials)

        mock_client = _stream_client(_mock_stream_response(200))
        with (
            _patch_client(mock_client),
            patch("ai_content_service.preflight._make_egress_guard", side_effect=spy_make_guard),
        ):
            await check_bundle(bundle, settings)

        assert captured_credentials == [()]


# ---------------------------------------------------------------------------
# M2a: ok means reachable, not healthy; range_supported carries resume signal
# ---------------------------------------------------------------------------


class TestRangeSupported:
    async def test_200_on_range_probe_is_ok_but_flags_no_resume(self) -> None:
        response = _mock_stream_response(200, content_type="application/octet-stream")
        bundle = _bundle_with_files([("model.safetensors", "https://example.com/model")])
        settings = Settings()

        with _patch_client(_stream_client(response)):
            report = await check_bundle(bundle, settings)

        result = report.results[0]
        assert result.ok is True
        assert result.range_supported is False
        assert result.flag is not None and "resume" in result.flag.lower()

    async def test_206_sets_range_supported_true_with_no_flag(self) -> None:
        response = _mock_stream_response(206, content_type="application/octet-stream")
        bundle = _bundle_with_files([("model.safetensors", "https://example.com/model")])
        settings = Settings()

        with _patch_client(_stream_client(response)):
            report = await check_bundle(bundle, settings)

        result = report.results[0]
        assert result.ok is True
        assert result.range_supported is True
        assert result.flag is None

    async def test_html_flag_takes_precedence_over_range_ignored(self) -> None:
        """A 200 that is *also* HTML must keep the HTML flag, not resume-noise."""
        response = _mock_stream_response(200, content_type="text/html; charset=utf-8")
        bundle = _bundle_with_files([("model.safetensors", "https://example.com/model")])
        settings = Settings()

        with _patch_client(_stream_client(response)):
            report = await check_bundle(bundle, settings)

        result = report.results[0]
        assert result.ok is False
        assert result.flag is not None
        assert "html" in result.flag.lower()


# ---------------------------------------------------------------------------
# M3a: bounded concurrency, order preserved
# ---------------------------------------------------------------------------


class TestBoundedConcurrency:
    async def test_wall_clock_bounded_by_semaphore_not_serial(self) -> None:
        file_count = 10
        settings = Settings(max_concurrent_downloads=3)
        bundle = _bundle_with_files(
            [(f"f{i}.safetensors", f"https://example.com/{i}") for i in range(file_count)]
        )

        async def slow_probe(_client: object, _url: str, _headers: dict[str, str]) -> _ProbeResult:
            await asyncio.sleep(0.1)
            return _ProbeResult(
                status_code=200,
                headers={"content-type": "application/octet-stream", "content-length": "1"},
            )

        start = time.monotonic()
        with patch("ai_content_service.preflight._probe", side_effect=slow_probe):
            await check_bundle(bundle, settings)
        elapsed = time.monotonic() - start

        expected_batches = math.ceil(file_count / settings.max_concurrent_downloads)
        assert elapsed < expected_batches * 0.1 + 0.5
        assert elapsed >= (expected_batches - 1) * 0.1

    async def test_report_order_matches_bundle_declaration_order(self) -> None:
        file_count = 10
        settings = Settings(max_concurrent_downloads=3)
        bundle = _bundle_with_files(
            [(f"f{i}.safetensors", f"https://example.com/{i}") for i in range(file_count)]
        )

        async def slow_probe(_client: object, _url: str, _headers: dict[str, str]) -> _ProbeResult:
            await asyncio.sleep(0.01 * (file_count - len(_url)))
            return _ProbeResult(
                status_code=200,
                headers={"content-type": "application/octet-stream", "content-length": "1"},
            )

        with patch("ai_content_service.preflight._probe", side_effect=slow_probe):
            report = await check_bundle(bundle, settings)

        assert [r.filename for r in report.results] == [
            f"f{i}.safetensors" for i in range(file_count)
        ]


# ---------------------------------------------------------------------------
# render_report
# ---------------------------------------------------------------------------


class TestRenderReport:
    async def test_renders_without_error_and_includes_filenames(self) -> None:
        bundle = _bundle_with_files([("a.safetensors", "https://example.com/a")])
        settings = Settings()

        mock_client = _stream_client(
            _mock_stream_response(200, content_type="application/octet-stream")
        )

        with _patch_client(mock_client):
            report = await check_bundle(bundle, settings)

        console = Console(record=True, width=200)
        render_report(report, console)
        output = console.export_text()

        assert "a.safetensors" in output
        assert "200" in output

    async def test_render_report_includes_resume_column(self) -> None:
        bundle = _bundle_with_files([("a.safetensors", "https://example.com/a")])
        settings = Settings()

        mock_client = _stream_client(
            _mock_stream_response(206, content_type="application/octet-stream")
        )

        with _patch_client(mock_client):
            report = await check_bundle(bundle, settings)

        console = Console(record=True, width=200)
        render_report(report, console)
        output = console.export_text()

        assert "Resume" in output
        assert "yes" in output.lower()

    async def test_r2_expected_rows_are_not_rendered_green(self) -> None:
        file = ModelFileConfig(
            name="cached.safetensors",
            url="",
            filename="cached.safetensors",
            sha256=_R2_SHA256,
        )
        report = await check_bundle(_bundle_with_model_file(file), _r2_settings())

        assert report.ok is True
        assert report.results[0].status == "R2 BY SHA256"
        assert _row_style(report.results[0]) == "yellow"

        healthy_bundle = _bundle_with_files([("healthy.safetensors", "https://example.com/a")])
        mock_client = _stream_client(
            _mock_stream_response(206, content_type="application/octet-stream")
        )
        with _patch_client(mock_client):
            healthy_report = await check_bundle(healthy_bundle, Settings())

        assert _row_style(healthy_report.results[0]) == "green"


# ---------------------------------------------------------------------------
# C4c: --offline validates parsing/required fields without any network request
# ---------------------------------------------------------------------------


class TestOfflineMode:
    async def test_offline_mode_makes_zero_network_calls(self) -> None:
        bundle = _bundle_with_files([("a.safetensors", "https://example.com/a")])
        settings = Settings()

        with patch("ai_content_service.preflight.httpx.AsyncClient") as mock_client_cls:
            report = await check_bundle(bundle, settings, offline=True)

        mock_client_cls.assert_not_called()
        result = report.results[0]
        assert result.status == "OFFLINE"
        assert result.ok is True
        assert result.flag == "not checked (offline)"

    async def test_offline_mode_still_flags_missing_url(self) -> None:
        """A missing URL requires no network access to detect -- still reported (C4c)."""
        bundle = _bundle_with_files([("a.safetensors", "")])
        settings = Settings()

        report = await check_bundle(bundle, settings, offline=True)

        assert report.ok is False
        assert report.results[0].status == "MISSING URL"

    async def test_offline_mode_redacts_url(self) -> None:
        bundle = _bundle_with_files(
            [("a.safetensors", "https://example.com/x?token=should_not_appear")]
        )
        settings = Settings()

        report = await check_bundle(bundle, settings, offline=True)

        assert "should_not_appear" not in report.results[0].url


# ---------------------------------------------------------------------------
# C4a/C4b: `--all` across many bundles, one unparseable
# ---------------------------------------------------------------------------


def _write_bundle_yaml(
    bundle_dir: Path,
    *,
    name: str,
    files: list[tuple[str, str]] | None = None,
    raw_text: str | None = None,
) -> None:
    """Write a bundle.yaml under *bundle_dir*, either from *raw_text* verbatim
    or built from a (filename, url) list."""
    bundle_dir.mkdir(parents=True, exist_ok=True)
    if raw_text is not None:
        (bundle_dir / "bundle.yaml").write_text(raw_text)
        return
    data = {
        "metadata": {"name": name, "version": "260101-01"},
        "models": [
            {
                "name": "Test Model",
                "model_type": "checkpoints",
                "files": [
                    {"name": filename, "url": url, "filename": filename}
                    for filename, url in (files or [])
                ],
            }
        ]
        if files
        else [],
    }
    (bundle_dir / "bundle.yaml").write_text(yaml.safe_dump(data))


class TestCheckBundlePath:
    async def test_parseable_bundle_is_probed(self, tmp_path: Path) -> None:
        bundle_dir = tmp_path / "good"
        _write_bundle_yaml(bundle_dir, name="good", files=[("f.safetensors", "https://x/f")])
        settings = Settings()

        mock_client = _stream_client(
            _mock_stream_response(200, content_type="application/octet-stream")
        )
        with _patch_client(mock_client):
            result = await check_bundle_path("good", bundle_dir, settings)

        assert result.parse_error is None
        assert result.ok is True
        assert len(result.file_results) == 1

    async def test_unparseable_bundle_is_a_result_not_an_exception(self, tmp_path: Path) -> None:
        """The C1b/C4b regression test: `extra='forbid'` rejecting an unknown
        key must be a reported row, not a raised exception."""
        bundle_dir = tmp_path / "bad"
        _write_bundle_yaml(
            bundle_dir,
            name="bad",
            raw_text="metadata:\n  name: bad\n  version: '260101-01'\n"
            "unknown_top_level_key: true\n",
        )
        settings = Settings()

        result = await check_bundle_path("bad", bundle_dir, settings)  # must not raise

        assert result.ok is False
        assert result.parse_error is not None
        assert "unknown_top_level_key" in result.parse_error
        assert result.file_results == ()

    async def test_missing_bundle_yaml_is_a_result_not_an_exception(self, tmp_path: Path) -> None:
        bundle_dir = tmp_path / "empty"
        bundle_dir.mkdir()
        settings = Settings()

        result = await check_bundle_path("empty", bundle_dir, settings)  # must not raise

        assert result.ok is False
        assert result.parse_error is not None


class TestCheckAllBundles:
    async def test_empty_entries_are_ok_but_reported_as_empty(self) -> None:
        report = await check_all_bundles([], Settings())

        assert report.ok is True
        assert report.is_empty is True

    async def test_nonempty_report_with_failure_is_not_ok(self) -> None:
        report = MultiBundleReport(
            results=(BundleCheckResult(bundle_name="bad", parse_error="broken"),)
        )

        assert report.ok is False
        assert report.is_empty is False

    async def test_three_bundles_one_unparseable(self, tmp_path: Path) -> None:
        good_a = tmp_path / "a"
        good_b = tmp_path / "b"
        bad = tmp_path / "bad"
        _write_bundle_yaml(good_a, name="a", files=[("f.safetensors", "https://x/f")])
        _write_bundle_yaml(good_b, name="b", files=[("g.safetensors", "https://x/g")])
        _write_bundle_yaml(
            bad,
            name="bad",
            raw_text="metadata:\n  name: bad\n  version: '260101-01'\nsize_gb: 14.5\n",
        )
        settings = Settings()

        mock_client = _stream_client(
            _mock_stream_response(200, content_type="application/octet-stream"),
            _mock_stream_response(200, content_type="application/octet-stream"),
        )
        with _patch_client(mock_client):
            report = await check_all_bundles([("a", good_a), ("bad", bad), ("b", good_b)], settings)

        assert len(report.results) == 3
        assert report.ok is False
        by_name = {r.bundle_name: r for r in report.results}
        assert by_name["a"].ok is True
        assert by_name["b"].ok is True
        assert by_name["bad"].parse_error is not None
        assert mock_client.stream.call_count == 2

    async def test_all_clean_reports_ok_true(self, tmp_path: Path) -> None:
        good_a = tmp_path / "a"
        good_b = tmp_path / "b"
        _write_bundle_yaml(good_a, name="a", files=[("f.safetensors", "https://x/f")])
        _write_bundle_yaml(good_b, name="b", files=[("g.safetensors", "https://x/g")])
        settings = Settings()

        mock_client = _stream_client(
            _mock_stream_response(200, content_type="application/octet-stream"),
            _mock_stream_response(200, content_type="application/octet-stream"),
        )
        with _patch_client(mock_client):
            report = await check_all_bundles([("a", good_a), ("b", good_b)], settings)

        assert report.ok is True

    async def test_all_snapshot_authored_bundle_with_r2_is_ok(self, tmp_path: Path) -> None:
        bundle_dir = tmp_path / "snapshot"
        bundle_dir.mkdir()
        data = {
            "metadata": {"name": "snapshot", "version": "260101-01"},
            "models": [
                {
                    "name": "Test Model",
                    "model_type": "checkpoints",
                    "files": [
                        {
                            "name": "cached.safetensors",
                            "url": "",
                            "filename": "cached.safetensors",
                            "sha256": _R2_SHA256,
                        }
                    ],
                }
            ],
        }
        (bundle_dir / "bundle.yaml").write_text(yaml.safe_dump(data))

        report = await check_all_bundles([("snapshot", bundle_dir)], _r2_settings())

        assert report.ok is True
        assert report.results[0].ok is True
        assert report.results[0].file_results[0].status == "R2 BY SHA256"

    async def test_offline_propagates_to_every_bundle(self, tmp_path: Path) -> None:
        bundle_dir = tmp_path / "a"
        _write_bundle_yaml(bundle_dir, name="a", files=[("f.safetensors", "https://x/f")])
        settings = Settings()

        with patch("ai_content_service.preflight.httpx.AsyncClient") as mock_client_cls:
            report = await check_all_bundles([("a", bundle_dir)], settings, offline=True)

        mock_client_cls.assert_not_called()
        assert report.results[0].file_results[0].status == "OFFLINE"

    async def test_resolver_programming_error_propagates(self) -> None:
        async def resolve(_name: str) -> Path:
            raise AttributeError("broken resolver wiring")

        with pytest.raises(AttributeError, match="broken resolver wiring"):
            await check_all_bundles(["broken"], Settings(), resolve_bundle_path=resolve)

    async def test_resolver_value_error_becomes_parse_error(self) -> None:
        async def resolve(_name: str) -> Path:
            raise ValueError("bundle not found")

        report = await check_all_bundles(["missing"], Settings(), resolve_bundle_path=resolve)

        assert report.results[0].parse_error == "bundle not found"


# ---------------------------------------------------------------------------
# C4d: --json machine-readable output
# ---------------------------------------------------------------------------


class TestJsonSerialization:
    async def test_multi_report_to_dict_marks_empty_reports(self) -> None:
        report = await check_all_bundles([], Settings())

        data = multi_report_to_dict(report)

        assert data["ok"] is True
        assert data["is_empty"] is True

    async def test_report_to_dict_includes_per_file_status_and_overall_result(self) -> None:
        bundle = _bundle_with_files([("a.safetensors", "https://example.com/a")])
        settings = Settings()

        mock_client = _stream_client(
            _mock_stream_response(200, content_type="application/octet-stream")
        )
        with _patch_client(mock_client):
            report = await check_bundle(bundle, settings)

        data = report_to_dict(report)

        assert data["ok"] is True
        files = data["files"]
        assert isinstance(files, list)
        assert files[0]["filename"] == "a.safetensors"
        assert files[0]["status"] == "200"
        assert files[0]["ok"] is True

    async def test_multi_report_to_dict_includes_parse_error_and_overall_result(
        self, tmp_path: Path
    ) -> None:
        bad = tmp_path / "bad"
        _write_bundle_yaml(
            bad, name="bad", raw_text="metadata:\n  name: bad\n  version: '260101-01'\nx: 1\n"
        )
        settings = Settings()

        report = await check_all_bundles([("bad", bad)], settings)
        data = multi_report_to_dict(report)

        assert data["ok"] is False
        bundles = data["bundles"]
        assert isinstance(bundles, list)
        assert bundles[0]["bundle"] == "bad"
        assert bundles[0]["parse_error"] is not None
        assert bundles[0]["files"] == []
