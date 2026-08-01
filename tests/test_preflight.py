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
from pydantic import ValidationError
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
from ai_content_service.preflight import _ProbeResult, check_bundle, render_report

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

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
        with pytest.raises(ValidationError):
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
