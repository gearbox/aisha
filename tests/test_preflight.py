"""Tests for preflight (Typer-free core of `acs models check`, D14)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from rich.console import Console

from ai_content_service.config import (
    BundleConfig,
    BundleMetadata,
    ModelConfig,
    ModelFileConfig,
    Settings,
)
from ai_content_service.preflight import check_bundle, render_report

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


def _mock_response(
    status_code: int = 200,
    content_type: str = "",
    content_disposition: str | None = None,
    content_length: int | None = None,
) -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    headers: dict[str, str] = {"content-type": content_type}
    if content_disposition is not None:
        headers["content-disposition"] = content_disposition
    if content_length is not None:
        headers["content-length"] = str(content_length)
    resp.headers = headers
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

        mock_client = MagicMock()
        mock_client.get = AsyncMock(
            side_effect=[
                _mock_response(200, content_type="application/octet-stream", content_length=123),
                _mock_response(404),
            ]
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

        mock_client = MagicMock()
        mock_client.get = AsyncMock(
            return_value=_mock_response(200, content_type="text/html; charset=utf-8")
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

        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=_mock_response(404))

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

        mock_client = MagicMock()
        mock_client.get = AsyncMock(
            return_value=_mock_response(206, content_type="application/octet-stream")
        )

        with _patch_client(mock_client):
            report = await check_bundle(bundle, settings)

        assert report.ok is True
        assert all(r.ok for r in report.results)

    async def test_civitai_401_falls_back_to_query_token(self) -> None:
        bundle = _bundle_with_files(
            [("model.safetensors", "https://civitai.red/api/download/models/1")]
        )
        settings = Settings(civitai_api_token="secret-token")  # type: ignore[arg-type]

        mock_client = MagicMock()
        mock_client.get = AsyncMock(
            side_effect=[
                _mock_response(401),
                _mock_response(200, content_type="application/octet-stream"),
            ]
        )

        with _patch_client(mock_client):
            report = await check_bundle(bundle, settings)

        assert mock_client.get.call_count == 2
        first_headers = mock_client.get.call_args_list[0].kwargs["headers"]
        second_url = mock_client.get.call_args_list[1].args[0]
        assert "Authorization" in first_headers
        assert "token=secret-token" in second_url
        assert report.ok is True

    async def test_result_url_is_redacted(self) -> None:
        bundle = _bundle_with_files(
            [("model.safetensors", "https://example.com/x?token=should_not_appear")]
        )
        settings = Settings()

        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=_mock_response(200))

        with _patch_client(mock_client):
            report = await check_bundle(bundle, settings)

        assert "should_not_appear" not in report.results[0].url

    async def test_request_error_reported_as_failure(self) -> None:
        bundle = _bundle_with_files([("model.safetensors", "https://example.com/x")])
        settings = Settings()

        mock_client = MagicMock()
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("boom"))

        with _patch_client(mock_client):
            report = await check_bundle(bundle, settings)

        assert report.ok is False
        assert report.results[0].status.startswith("ERROR")


# ---------------------------------------------------------------------------
# render_report
# ---------------------------------------------------------------------------


class TestRenderReport:
    async def test_renders_without_error_and_includes_filenames(self) -> None:
        bundle = _bundle_with_files([("a.safetensors", "https://example.com/a")])
        settings = Settings()

        mock_client = MagicMock()
        mock_client.get = AsyncMock(
            return_value=_mock_response(200, content_type="application/octet-stream")
        )

        with _patch_client(mock_client):
            report = await check_bundle(bundle, settings)

        console = Console(record=True, width=200)
        render_report(report, console)
        output = console.export_text()

        assert "a.safetensors" in output
        assert "200" in output
