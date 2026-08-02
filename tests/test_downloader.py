"""Tests for model downloader including Civitai support."""

import hashlib
import inspect
import logging
from contextlib import AbstractContextManager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from rich.progress import TaskID

from ai_content_service.config import ModelConfig, ModelFileConfig, Settings
from ai_content_service.downloader import (
    DownloadError,
    ModelDownloader,
    _part_size,
    _ProgressTracker,
    _RetryWait,
    _StalePartError,
    _TruncatedTransferError,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# Captured before any test patches `ai_content_service.downloader.httpx.AsyncClient`
# -- that patch replaces the attribute on this same shared `httpx` module object,
# so a fresh `httpx.AsyncClient` lookup inside a side_effect would recurse into
# the mock itself. Real construction must go through this saved reference.
_REAL_ASYNC_CLIENT = httpx.AsyncClient


def _client_factory_with_transport(
    handler: object,
) -> object:
    """A `side_effect` for patching `httpx.AsyncClient` that injects a
    MockTransport into otherwise-real client construction, so `download_all`'s
    real code path (redirects, event hooks) runs end-to-end against a fake
    handler instead of the network."""

    def _make(*args: object, **kwargs: object) -> httpx.AsyncClient:
        kwargs["transport"] = httpx.MockTransport(handler)  # type: ignore[arg-type]
        return _REAL_ASYNC_CLIENT(*args, **kwargs)  # type: ignore[arg-type]

    return _make


def _file_cfg(
    filename: str,
    url: str,
    sha256: str | None = None,
    size_bytes: int | None = None,
) -> ModelFileConfig:
    return ModelFileConfig(
        name=filename, url=url, filename=filename, sha256=sha256, size_bytes=size_bytes
    )


def _model_cfg(
    name: str,
    model_type: str,
    files: list[ModelFileConfig],
    subdirectory: str | None = None,
) -> ModelConfig:
    return ModelConfig(name=name, model_type=model_type, files=files, subdirectory=subdirectory)


def _make_async_cm(return_value: object) -> MagicMock:
    """Wrap *return_value* in an async context-manager mock."""
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=return_value)
    cm.__aexit__ = AsyncMock(return_value=None)
    return cm


def _mock_http_response(
    chunks: list[bytes], content_length: str = "", status_code: int = 200
) -> MagicMock:
    """Build a minimal httpx-response mock for streaming tests."""
    response = MagicMock()
    response.status_code = status_code
    response.headers = {"content-length": content_length} if content_length else {}

    def _raise_for_status() -> None:
        if status_code >= 400:
            raise httpx.HTTPStatusError(
                f"HTTP {status_code}", request=MagicMock(), response=response
            )

    response.raise_for_status = MagicMock(side_effect=_raise_for_status)

    async def _aiter_bytes(chunk_size: int):  # noqa: ARG001
        for chunk in chunks:
            yield chunk

    response.aiter_bytes = _aiter_bytes
    return response


def _patch_http(
    chunks: list[bytes], content_length: str = "", status_code: int = 200
) -> tuple[AbstractContextManager[MagicMock], MagicMock, MagicMock]:
    """Patch httpx.AsyncClient only; aiofiles performs real I/O against tmp_path.

    Returns (patch_context, mock_client, response) so callers can inspect the
    request (e.g. the Range header) or the response mock.
    """
    response = _mock_http_response(chunks, content_length, status_code)
    mock_client = MagicMock()
    mock_client.stream = MagicMock(return_value=_make_async_cm(response))
    http_cm = _make_async_cm(mock_client)

    http_patch = patch("ai_content_service.downloader.httpx.AsyncClient", return_value=http_cm)
    return http_patch, mock_client, response


@pytest.fixture
def settings() -> Settings:
    """Create settings for testing."""
    return Settings(
        civitai_api_token="test_civitai_token_123",  # type: ignore[arg-type]
        hf_token="test_hf_token_456",  # type: ignore[arg-type]
        civitai_allow_query_token_fallback=True,
    )


@pytest.fixture
def settings_no_tokens() -> Settings:
    """Create settings without API tokens."""
    return Settings()


@pytest.fixture
def downloader(settings: Settings) -> ModelDownloader:
    """Create a downloader instance with tokens."""
    return ModelDownloader(settings)


@pytest.fixture
def downloader_no_tokens(settings_no_tokens: Settings) -> ModelDownloader:
    """Create a downloader instance without tokens."""
    return ModelDownloader(settings_no_tokens)


@pytest.fixture
def settings_blank_tokens() -> Settings:
    """Create settings with tokens configured but blank -- the E3 scenario:
    `ACS_CIVITAI_API_TOKEN=` in a `.env`, distinct from the var being unset."""
    return Settings(
        civitai_api_token="",  # type: ignore[arg-type]
        hf_token="   ",  # type: ignore[arg-type]
    )


@pytest.fixture
def downloader_blank_tokens(settings_blank_tokens: Settings) -> ModelDownloader:
    """Create a downloader instance with blank (not unset) tokens configured."""
    return ModelDownloader(settings_blank_tokens)


class TestCivitaiAuthTransport:
    """Tests for Change 3 (D1-D4, D6): auth via the download_auth registry.

    Civitai now uses the ``Authorization: Bearer`` header first (D3), with a
    single query-token fallback on 401/403. Domain-matching itself (including
    the look-alike-host security guard) is unit-tested directly against
    download_auth in test_download_auth.py; these tests confirm the
    downloader actually wires it up end to end.
    """

    @pytest.fixture
    def progress(self) -> MagicMock:
        p = MagicMock()
        p.add_task.return_value = 0
        return p

    async def test_civitai_red_gets_bearer_header_no_query_token(
        self, tmp_path: Path, downloader: ModelDownloader, progress: MagicMock
    ) -> None:
        """B1: civitai.red — the presenting bug — is now authenticated like civitai.com."""
        file_cfg = _file_cfg("model.safetensors", "https://civitai.red/api/download/models/123")
        path = tmp_path / "model.safetensors"

        http_p, mock_client, _resp = _patch_http([b"data"])
        with http_p:
            await downloader._download_file(
                file_cfg, path, progress, task_id=TaskID(0), client=mock_client
            )

        sent_url = mock_client.stream.call_args.args[1]
        sent_headers = mock_client.stream.call_args.kwargs["headers"]
        assert sent_headers["Authorization"] == "Bearer test_civitai_token_123"
        assert "token" not in sent_url

    async def test_lookalike_domain_gets_no_token_anywhere(
        self, tmp_path: Path, downloader: ModelDownloader, progress: MagicMock
    ) -> None:
        """Security guard: civitai.com.evil.com must never receive a token (D1 pitfall #7)."""
        file_cfg = _file_cfg("model.safetensors", "https://civitai.com.evil.com/x")
        path = tmp_path / "model.safetensors"

        http_p, mock_client, _resp = _patch_http([b"data"])
        with http_p:
            await downloader._download_file(
                file_cfg, path, progress, task_id=TaskID(0), client=mock_client
            )

        sent_headers = mock_client.stream.call_args.kwargs["headers"]
        sent_url = mock_client.stream.call_args.args[1]
        assert "Authorization" not in sent_headers
        assert "token" not in sent_url

    async def test_no_token_configured_no_auth_attached(
        self, tmp_path: Path, downloader_no_tokens: ModelDownloader, progress: MagicMock
    ) -> None:
        file_cfg = _file_cfg("model.safetensors", "https://civitai.com/api/download/models/123")
        path = tmp_path / "model.safetensors"

        http_p, mock_client, _resp = _patch_http([b"data"])
        with http_p:
            await downloader_no_tokens._download_file(
                file_cfg, path, progress, task_id=TaskID(0), client=mock_client
            )

        sent_headers = mock_client.stream.call_args.kwargs["headers"]
        sent_url = mock_client.stream.call_args.args[1]
        assert "Authorization" not in sent_headers
        assert "token" not in sent_url

    async def test_huggingface_still_gets_bearer_header(
        self, tmp_path: Path, downloader: ModelDownloader, progress: MagicMock
    ) -> None:
        file_cfg = _file_cfg("model.safetensors", "https://huggingface.co/model/download")
        path = tmp_path / "model.safetensors"

        http_p, mock_client, _resp = _patch_http([b"data"])
        with http_p:
            await downloader._download_file(
                file_cfg, path, progress, task_id=TaskID(0), client=mock_client
            )

        sent_headers = mock_client.stream.call_args.kwargs["headers"]
        assert sent_headers["Authorization"] == "Bearer test_hf_token_456"

    async def test_401_then_200_falls_back_to_query_token_exactly_once(
        self, tmp_path: Path, downloader: ModelDownloader, progress: MagicMock
    ) -> None:
        """D3: header fails once -> single query-token retry -> success. Never a third request."""
        file_cfg = _file_cfg("model.safetensors", "https://civitai.red/api/download/models/123")
        path = tmp_path / "model.safetensors"
        part_path = path.with_name(f"{path.name}.part")

        responses = [
            _mock_http_response([], status_code=401),
            _mock_http_response([b"data"], status_code=200),
        ]
        mock_client = MagicMock()
        mock_client.stream = MagicMock(side_effect=[_make_async_cm(r) for r in responses])
        http_cm = _make_async_cm(mock_client)

        with patch("ai_content_service.downloader.httpx.AsyncClient", return_value=http_cm):
            await downloader._stream_to_part(
                file_cfg, path, part_path, progress, TaskID(0), None, mock_client
            )

        assert mock_client.stream.call_count == 2
        first_url = mock_client.stream.call_args_list[0].args[1]
        first_headers = mock_client.stream.call_args_list[0].kwargs["headers"]
        second_url = mock_client.stream.call_args_list[1].args[1]

        assert "token" not in first_url
        assert first_headers["Authorization"] == "Bearer test_civitai_token_123"
        assert "token=test_civitai_token_123" in second_url
        assert path.read_bytes() == b"data"

    async def test_401_then_401_raises_download_error_exactly_two_requests(
        self, tmp_path: Path, downloader: ModelDownloader, progress: MagicMock
    ) -> None:
        """D3: fallback must fire at most once per file (pitfall #2), not loop forever."""
        file_cfg = _file_cfg("model.safetensors", "https://civitai.red/api/download/models/123")
        path = tmp_path / "model.safetensors"
        part_path = path.with_name(f"{path.name}.part")

        responses = [
            _mock_http_response([], status_code=401),
            _mock_http_response([], status_code=401),
        ]
        mock_client = MagicMock()
        mock_client.stream = MagicMock(side_effect=[_make_async_cm(r) for r in responses])
        http_cm = _make_async_cm(mock_client)

        with (
            patch("ai_content_service.downloader.httpx.AsyncClient", return_value=http_cm),
            pytest.raises(DownloadError, match="authentication failed"),
        ):
            await downloader._stream_to_part(
                file_cfg, path, part_path, progress, TaskID(0), None, mock_client
            )

        assert mock_client.stream.call_count == 2

    async def test_500_does_not_trigger_query_fallback(
        self, tmp_path: Path, downloader: ModelDownloader, progress: MagicMock
    ) -> None:
        """D3: a 500 goes through tenacity's retry path only, never the auth fallback."""
        file_cfg = _file_cfg("model.safetensors", "https://civitai.red/api/download/models/123")
        path = tmp_path / "model.safetensors"
        part_path = path.with_name(f"{path.name}.part")

        http_p, mock_client, _resp = _patch_http([], status_code=500)
        with http_p, pytest.raises(httpx.HTTPStatusError):
            await downloader._stream_to_part(
                file_cfg, path, part_path, progress, TaskID(0), None, mock_client
            )

        assert mock_client.stream.call_count == 1
        assert "token=" not in mock_client.stream.call_args.args[1]

    async def test_user_agent_set_on_every_request(
        self, tmp_path: Path, downloader: ModelDownloader, progress: MagicMock
    ) -> None:
        """D6: Cloudflare challenges default library UAs, so every request needs a browser UA."""
        file_cfg = _file_cfg("model.safetensors", "https://example.com/model.safetensors")
        path = tmp_path / "model.safetensors"

        http_p, mock_client, _resp = _patch_http([b"data"])
        with http_p:
            await downloader._download_file(
                file_cfg, path, progress, task_id=TaskID(0), client=mock_client
            )

        sent_headers = mock_client.stream.call_args.kwargs["headers"]
        assert sent_headers["User-Agent"] == downloader._user_agent
        assert "Mozilla" in sent_headers["User-Agent"]

    async def test_html_response_raises_and_writes_no_part_file(
        self, tmp_path: Path, downloader: ModelDownloader, progress: MagicMock
    ) -> None:
        """D7: an HTML login page (civitai.red without/invalid token) fails loud before any bytes."""
        file_cfg = _file_cfg("model.safetensors", "https://civitai.red/api/download/models/123")
        path = tmp_path / "model.safetensors"
        part_path = path.with_name(f"{path.name}.part")

        response = _mock_http_response([b"<html>login</html>"], status_code=200)
        response.headers = {"content-length": "0", "content-type": "text/html; charset=utf-8"}
        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=_make_async_cm(response))
        http_cm = _make_async_cm(mock_client)

        with (
            patch("ai_content_service.downloader.httpx.AsyncClient", return_value=http_cm),
            pytest.raises(DownloadError, match=r"civitai\.red"),
        ):
            await downloader._download_file(
                file_cfg, path, progress, task_id=TaskID(0), client=mock_client
            )

        assert not part_path.exists()
        assert not path.exists()

    async def test_token_never_leaks_to_logs_console_or_exception(
        self,
        tmp_path: Path,
        downloader: ModelDownloader,
        caplog: pytest.LogCaptureFixture,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """D12: force a 403 with the query fallback active; the raw token must appear nowhere."""
        file_cfg = _file_cfg("model.safetensors", "https://civitai.red/api/download/models/123")
        model = _model_cfg("m", "diffusion_models", [file_cfg])

        responses = [
            _mock_http_response([], status_code=403),
            _mock_http_response([], status_code=403),
        ]
        mock_client = MagicMock()
        mock_client.stream = MagicMock(side_effect=[_make_async_cm(r) for r in responses])
        http_cm = _make_async_cm(mock_client)

        token = "test_civitai_token_123"
        with (
            caplog.at_level(logging.DEBUG, logger="ai_content_service.downloader"),
            patch("ai_content_service.downloader.httpx.AsyncClient", return_value=http_cm),
        ):
            report = await downloader.download_all([model], tmp_path)

        assert report.ok is False
        assert token not in report.failed[0].reason
        assert token not in report.failed[0].url
        for record in caplog.records:
            assert token not in record.getMessage()
        captured = capsys.readouterr()
        assert token not in captured.out
        assert token not in captured.err


class TestBlankTokenBehavesAsAnonymous:
    """E3: a blank configured token must behave exactly like an unset one --
    not like a malformed credential that trips the auth-retry path."""

    @pytest.fixture
    def progress(self) -> MagicMock:
        p = MagicMock()
        p.add_task.return_value = 0
        return p

    async def test_blank_civitai_token_sends_no_authorization_header(
        self, tmp_path: Path, downloader_blank_tokens: ModelDownloader, progress: MagicMock
    ) -> None:
        file_cfg = _file_cfg("model.safetensors", "https://civitai.red/api/download/models/123")
        path = tmp_path / "model.safetensors"

        http_p, mock_client, _resp = _patch_http([b"data"])
        with http_p:
            await downloader_blank_tokens._download_file(
                file_cfg, path, progress, task_id=TaskID(0), client=mock_client
            )

        sent_headers = mock_client.stream.call_args.kwargs["headers"]
        sent_url = mock_client.stream.call_args.args[1]
        assert "Authorization" not in sent_headers, "must be absent, not merely empty"
        assert "token" not in sent_url

    async def test_blank_hf_token_sends_no_authorization_header(
        self, tmp_path: Path, downloader_blank_tokens: ModelDownloader, progress: MagicMock
    ) -> None:
        file_cfg = _file_cfg("model.safetensors", "https://huggingface.co/model/download")
        path = tmp_path / "model.safetensors"

        http_p, mock_client, _resp = _patch_http([b"data"])
        with http_p:
            await downloader_blank_tokens._download_file(
                file_cfg, path, progress, task_id=TaskID(0), client=mock_client
            )

        sent_headers = mock_client.stream.call_args.kwargs["headers"]
        assert "Authorization" not in sent_headers, "must be absent, not merely empty"

    async def test_blank_token_does_not_trigger_query_fallback_on_401(
        self, tmp_path: Path, downloader_blank_tokens: ModelDownloader, progress: MagicMock
    ) -> None:
        """A blank token must not burn the query-token fallback on a 401 --
        that retry exists for a real credential, not a no-op one."""
        file_cfg = _file_cfg("model.safetensors", "https://civitai.red/api/download/models/123")
        path = tmp_path / "model.safetensors"
        part_path = path.with_name(f"{path.name}.part")

        http_p, mock_client, _resp = _patch_http([], status_code=401)
        with http_p, pytest.raises(DownloadError, match="authentication failed"):
            await downloader_blank_tokens._stream_to_part(
                file_cfg, path, part_path, progress, TaskID(0), None, mock_client
            )

        assert mock_client.stream.call_count == 1


class TestSettingsCivitaiToken:
    """Tests for Civitai token in settings."""

    def test_civitai_token_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that CIVITAI_API_TOKEN env var is read."""
        monkeypatch.setenv("ACS_CIVITAI_API_TOKEN", "env_token_xyz")
        settings = Settings()
        assert settings.civitai_api_token is not None
        assert settings.civitai_api_token.get_secret_value() == "env_token_xyz"

    def test_civitai_token_default_none(self) -> None:
        """Test that Civitai token defaults to None."""
        settings = Settings()
        assert settings.civitai_api_token is None


# ---------------------------------------------------------------------------
# _part_size
# ---------------------------------------------------------------------------


class TestPartSize:
    """Tests for the module-level _part_size helper (single-stat TOCTOU fix)."""

    def test_part_size_absent_file_returns_zero(self, tmp_path: Path) -> None:
        assert _part_size(tmp_path / "nope.part") == 0

    def test_part_size_returns_existing_size(self, tmp_path: Path) -> None:
        part = tmp_path / "m.part"
        part.write_bytes(b"0123456789")
        assert _part_size(part) == 10

    def test_part_size_treats_stat_errors_as_absent(self, tmp_path: Path) -> None:
        with patch("ai_content_service.downloader.os.stat", side_effect=OSError):
            assert _part_size(tmp_path / "m.part") == 0


class TestTransferFailureMessages:
    def test_stale_part_error_includes_filename_and_offset(self) -> None:
        error = _StalePartError("model.safetensors", 4096)
        assert error.filename == "model.safetensors"
        assert error.offset == 4096
        assert "4096 bytes" in str(error)

    def test_truncated_transfer_error_includes_byte_counts(self) -> None:
        error = _TruncatedTransferError("model.safetensors", 400, 1000, True)
        assert error.filename == "model.safetensors"
        assert error.written == 400
        assert error.expected == 1000
        assert error.part_preserved is True
        assert "400 of 1000 bytes" in str(error)


# ---------------------------------------------------------------------------
# _verify_checksum
# ---------------------------------------------------------------------------


class TestVerifyChecksum:
    """Tests for ModelDownloader._verify_checksum."""

    async def test_correct_checksum_returns_true(
        self, tmp_path: Path, downloader: ModelDownloader
    ) -> None:
        content = b"hello world"
        expected = hashlib.sha256(content).hexdigest()
        path = tmp_path / "model.safetensors"
        path.write_bytes(content)

        assert await downloader._verify_checksum(path, expected) is True

    async def test_wrong_checksum_returns_false(
        self, tmp_path: Path, downloader: ModelDownloader
    ) -> None:
        path = tmp_path / "model.safetensors"
        path.write_bytes(b"hello world")

        assert await downloader._verify_checksum(path, "0" * 64) is False

    async def test_large_file_chunked_correctly(
        self, tmp_path: Path, downloader: ModelDownloader
    ) -> None:
        """Verifier handles files larger than CHUNK_SIZE."""
        content = b"x" * (ModelDownloader.CHUNK_SIZE + 1)
        expected = hashlib.sha256(content).hexdigest()
        path = tmp_path / "big.safetensors"
        path.write_bytes(content)

        assert await downloader._verify_checksum(path, expected) is True


# ---------------------------------------------------------------------------
# _download_file
# ---------------------------------------------------------------------------


class TestDownloadFile:
    """Tests for ModelDownloader._download_file."""

    @pytest.fixture
    def progress(self) -> MagicMock:
        p = MagicMock()
        p.add_task.return_value = 0
        return p

    async def test_mismatched_snapshot_file_with_empty_url_has_actionable_error(
        self, tmp_path: Path, progress: MagicMock
    ) -> None:
        path = tmp_path / "model.safetensors"
        path.write_bytes(b"wrong")
        file_cfg = _file_cfg(
            path.name,
            "",
            sha256=hashlib.sha256(b"expected").hexdigest(),
            size_bytes=path.stat().st_size,
        )
        downloader = ModelDownloader(Settings())

        with pytest.raises(DownloadError, match="no source URL"):
            await downloader._download_file(
                file_cfg, path, progress, task_id=TaskID(0), client=MagicMock()
            )

    async def test_writes_chunks_to_file(
        self, tmp_path: Path, downloader: ModelDownloader, progress: MagicMock
    ) -> None:
        chunks = [b"hello ", b"world"]
        file_cfg = _file_cfg("model.safetensors", "https://example.com/model.safetensors")
        path = tmp_path / "model.safetensors"

        http_p, mock_client, _resp = _patch_http(chunks)
        with http_p:
            await downloader._download_file(
                file_cfg, path, progress, task_id=TaskID(0), client=mock_client
            )

        assert path.read_bytes() == b"hello world"

    async def test_progress_updated_per_chunk(
        self, tmp_path: Path, downloader: ModelDownloader, progress: MagicMock
    ) -> None:
        chunks = [b"aa", b"bbb"]
        file_cfg = _file_cfg("model.safetensors", "https://example.com/model.safetensors")
        path = tmp_path / "model.safetensors"

        http_p, mock_client, _resp = _patch_http(chunks)
        with http_p:
            await downloader._download_file(
                file_cfg, path, progress, task_id=TaskID(0), client=mock_client
            )

        advance_calls = [c for c in progress.update.call_args_list if "advance" in c.kwargs]
        assert len(advance_calls) == 2
        assert advance_calls[0].kwargs["advance"] == 2
        assert advance_calls[1].kwargs["advance"] == 3

    async def test_content_length_sets_task_total(
        self, tmp_path: Path, downloader: ModelDownloader, progress: MagicMock
    ) -> None:
        file_cfg = _file_cfg("model.safetensors", "https://example.com/model.safetensors")
        path = tmp_path / "model.safetensors"

        http_p, mock_client, _resp = _patch_http([b"data"], content_length="4")
        with http_p:
            await downloader._download_file(
                file_cfg, path, progress, task_id=TaskID(0), client=mock_client
            )

        total_calls = [c for c in progress.update.call_args_list if c.kwargs.get("total") == 4]
        assert total_calls, "progress.update(total=4) should have been called"

    async def test_on_bytes_called_per_chunk(
        self, tmp_path: Path, downloader: ModelDownloader, progress: MagicMock
    ) -> None:
        chunks = [b"aa", b"bbb"]
        file_cfg = _file_cfg("model.safetensors", "https://example.com/model.safetensors")
        path = tmp_path / "model.safetensors"
        on_bytes = AsyncMock()

        http_p, mock_client, _resp = _patch_http(chunks)
        with http_p:
            await downloader._download_file(
                file_cfg, path, progress, task_id=TaskID(0), client=mock_client, on_bytes=on_bytes
            )

        # D9: absolute cumulative bytes, not per-chunk deltas.
        on_bytes.assert_any_call(2)
        on_bytes.assert_any_call(5)

    async def test_checksum_match_no_error(self, tmp_path: Path, progress: MagicMock) -> None:
        content = b"good content"
        sha256 = hashlib.sha256(content).hexdigest()
        file_cfg = _file_cfg(
            "model.safetensors", "https://example.com/model.safetensors", sha256=sha256
        )
        path = tmp_path / "model.safetensors"
        settings = Settings(verify_checksums=True)
        dl = ModelDownloader(settings)

        http_p, mock_client, _resp = _patch_http([content])
        with http_p:
            await dl._download_file(
                file_cfg, path, progress, task_id=TaskID(0), client=mock_client
            )  # must not raise

        assert path.read_bytes() == content

    async def test_no_hasher_when_verify_disabled(
        self, tmp_path: Path, progress: MagicMock
    ) -> None:
        sha256 = "0" * 64
        file_cfg = _file_cfg(
            "model.safetensors", "https://example.com/model.safetensors", sha256=sha256
        )
        path = tmp_path / "model.safetensors"
        settings = Settings(verify_checksums=False)
        dl = ModelDownloader(settings)

        http_p, mock_client, _resp = _patch_http([b"any content"])
        with http_p:
            await dl._download_file(
                file_cfg, path, progress, task_id=TaskID(0), client=mock_client
            )  # wrong hash but no check

    async def test_skip_existing_valid_checksum(self, tmp_path: Path, progress: MagicMock) -> None:
        content = b"already here"
        sha256 = hashlib.sha256(content).hexdigest()
        path = tmp_path / "model.safetensors"
        path.write_bytes(content)
        file_cfg = _file_cfg(
            "model.safetensors", "https://example.com/model.safetensors", sha256=sha256
        )
        settings = Settings(skip_existing=True)
        dl = ModelDownloader(settings)

        with patch("ai_content_service.downloader.httpx.AsyncClient") as mock_async_client_cls:
            await dl._download_file(file_cfg, path, progress, task_id=TaskID(0), client=MagicMock())
            mock_async_client_cls.assert_not_called()

    async def test_skip_existing_calls_on_bytes(self, tmp_path: Path, progress: MagicMock) -> None:
        content = b"already here"
        sha256 = hashlib.sha256(content).hexdigest()
        path = tmp_path / "model.safetensors"
        path.write_bytes(content)
        file_cfg = _file_cfg(
            "model.safetensors", "https://example.com/model.safetensors", sha256=sha256
        )
        settings = Settings(skip_existing=True)
        dl = ModelDownloader(settings)
        on_bytes = AsyncMock()

        with patch("ai_content_service.downloader.httpx.AsyncClient"):
            await dl._download_file(
                file_cfg, path, progress, task_id=TaskID(0), client=MagicMock(), on_bytes=on_bytes
            )

        on_bytes.assert_called_once_with(len(content))

    async def test_does_not_skip_when_checksum_absent(
        self, tmp_path: Path, progress: MagicMock
    ) -> None:
        """skip_existing=True but no sha256 → file is re-downloaded."""
        path = tmp_path / "model.safetensors"
        path.write_bytes(b"old content")
        file_cfg = _file_cfg("model.safetensors", "https://example.com/model.safetensors")
        settings = Settings(skip_existing=True)
        dl = ModelDownloader(settings)

        http_p, mock_client, _resp = _patch_http([b"new content"])
        with http_p:
            await dl._download_file(file_cfg, path, progress, task_id=TaskID(0), client=mock_client)

        assert path.read_bytes() == b"new content"


# ---------------------------------------------------------------------------
# Atomic writes, resume, and retry (P1-1)
# ---------------------------------------------------------------------------


class TestAtomicDownload:
    """Tests for atomic .part-then-rename semantics on the HTTP path."""

    @pytest.fixture
    def progress(self) -> MagicMock:
        p = MagicMock()
        p.add_task.return_value = 0
        return p

    async def test_download_atomic_rename_on_success(
        self, tmp_path: Path, downloader: ModelDownloader, progress: MagicMock
    ) -> None:
        file_cfg = _file_cfg("model.safetensors", "https://example.com/model.safetensors")
        path = tmp_path / "model.safetensors"
        part_path = path.with_name(f"{path.name}.part")

        async def _aiter_bytes(_chunk_size: int):
            for chunk in [b"hello ", b"world"]:
                assert not path.exists(), "final path must not exist mid-stream"
                yield chunk

        response = _mock_http_response([])
        response.aiter_bytes = _aiter_bytes
        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=_make_async_cm(response))
        http_cm = _make_async_cm(mock_client)

        with patch("ai_content_service.downloader.httpx.AsyncClient", return_value=http_cm):
            await downloader._download_file(
                file_cfg, path, progress, task_id=TaskID(0), client=mock_client
            )

        assert path.read_bytes() == b"hello world"
        assert not part_path.exists()

    async def test_checksum_mismatch_unlinks_part_not_final(
        self, tmp_path: Path, progress: MagicMock
    ) -> None:
        """A pre-existing valid file at `path` must survive a failed re-download."""
        previous_content = b"previously good file"
        path = tmp_path / "model.safetensors"
        path.write_bytes(previous_content)

        bad_sha256 = "0" * 64  # deliberately wrong for the new download
        file_cfg = _file_cfg(
            "model.safetensors", "https://example.com/model.safetensors", sha256=bad_sha256
        )
        settings = Settings(verify_checksums=True, skip_existing=False, download_max_attempts=1)
        dl = ModelDownloader(settings)

        http_p, mock_client, _resp = _patch_http([b"bad new content"])
        with http_p, pytest.raises(DownloadError, match="Checksum mismatch"):
            await dl._download_file(file_cfg, path, progress, task_id=TaskID(0), client=mock_client)

        assert path.read_bytes() == previous_content
        assert not path.with_name(f"{path.name}.part").exists()


class TestIntegrityGates:
    """Server length catches truncation; SHA-256 establishes content identity."""

    @pytest.fixture
    def progress(self) -> MagicMock:
        progress = MagicMock()
        progress.add_task.return_value = 0
        return progress

    @staticmethod
    def _events(caplog: pytest.LogCaptureFixture, event: str) -> list[dict[str, object]]:
        return [
            record.msg
            for record in caplog.records
            if isinstance(record.msg, dict) and record.msg.get("event") == event
        ]

    async def test_chunked_stale_declared_size_with_correct_sha_succeeds(
        self, tmp_path: Path, progress: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        content = b"complete model"
        file = _file_cfg(
            "model.safetensors",
            "https://example.com/model",
            sha256=hashlib.sha256(content).hexdigest(),
            size_bytes=len(content) + 3,
        )
        path = tmp_path / file.filename
        response = _mock_http_response([content])  # chunked: no length headers
        client = MagicMock()
        client.stream = MagicMock(return_value=_make_async_cm(response))

        with caplog.at_level(logging.WARNING, logger="ai_content_service.downloader"):
            await ModelDownloader(Settings(download_max_attempts=2))._download_file(
                file, path, progress, task_id=TaskID(0), client=client
            )

        assert path.read_bytes() == content
        assert client.stream.call_count == 1
        assert not self._events(caplog, "download.size.declared_mismatch")
        assert not self._events(caplog, "download.unverified")

    async def test_chunked_stale_declared_size_without_sha_succeeds(
        self, tmp_path: Path, progress: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        content = b"complete model"
        file = _file_cfg(
            "model.safetensors", "https://example.com/model", size_bytes=len(content) + 3
        )
        path = tmp_path / file.filename
        response = _mock_http_response([content])  # chunked: no length headers
        client = MagicMock()
        client.stream = MagicMock(return_value=_make_async_cm(response))

        with caplog.at_level(logging.WARNING, logger="ai_content_service.downloader"):
            await ModelDownloader(Settings(download_max_attempts=2))._download_file(
                file, path, progress, task_id=TaskID(0), client=client
            )

        assert path.read_bytes() == content
        assert client.stream.call_count == 1
        assert not self._events(caplog, "download.truncated")
        unverified = self._events(caplog, "download.unverified")
        assert len(unverified) == 1
        assert unverified[0]["reason"] == "no sha256"

    async def test_short_server_length_retries_and_keeps_partial_file(
        self, tmp_path: Path, progress: MagicMock
    ) -> None:
        file = _file_cfg("model.safetensors", "https://example.com/model")
        path = tmp_path / file.filename
        first = _mock_http_response([b"abcd"], content_length="10")
        second = _mock_http_response([b"ef"], content_length="6", status_code=206)
        second.headers["content-range"] = "bytes 4-9/10"
        client = MagicMock()
        client.stream = MagicMock(side_effect=[_make_async_cm(first), _make_async_cm(second)])

        with (
            patch("ai_content_service.downloader._RetryWait", return_value=lambda _: 0.0),
            pytest.raises(DownloadError, match="transfer truncated on every one of 2 attempts"),
        ):
            await ModelDownloader(Settings(download_max_attempts=2))._download_file(
                file, path, progress, task_id=TaskID(0), client=client
            )

        assert client.stream.call_count == 2
        assert client.stream.call_args_list[1].kwargs["headers"]["Range"] == "bytes=4-"
        assert path.with_name(f"{path.name}.part").read_bytes() == b"abcdef"

    async def test_server_size_mismatch_is_a_warning_not_a_failure(
        self, tmp_path: Path, progress: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        content = b"correct body"
        file = _file_cfg(
            "model.safetensors", "https://example.com/model", size_bytes=len(content) + 3
        )
        path = tmp_path / file.filename
        response = _mock_http_response([content], content_length=str(len(content)))
        client = MagicMock()
        client.stream = MagicMock(return_value=_make_async_cm(response))

        with caplog.at_level(logging.WARNING, logger="ai_content_service.downloader"):
            await ModelDownloader(Settings())._download_file(
                file, path, progress, task_id=TaskID(0), client=client
            )

        assert path.read_bytes() == content
        mismatches = self._events(caplog, "download.size.declared_mismatch")
        assert len(mismatches) == 1
        assert mismatches[0]["declared"] == len(content) + 3
        assert mismatches[0]["server"] == len(content)
        assert not self._events(caplog, "download.unverified")

    async def test_checksum_mismatch_remains_fatal_when_server_length_matches(
        self, tmp_path: Path, progress: MagicMock
    ) -> None:
        content = b"wrong content"
        file = _file_cfg(
            "model.safetensors",
            "https://example.com/model",
            sha256=hashlib.sha256(b"expected content").hexdigest(),
            size_bytes=len(content),
        )
        path = tmp_path / file.filename
        response = _mock_http_response([content], content_length=str(len(content)))
        client = MagicMock()
        client.stream = MagicMock(return_value=_make_async_cm(response))

        with pytest.raises(DownloadError, match="Checksum mismatch"):
            await ModelDownloader(Settings(download_max_attempts=1))._download_file(
                file, path, progress, task_id=TaskID(0), client=client
            )

        assert not path.exists()
        assert not path.with_name(f"{path.name}.part").exists()

    async def test_progressing_truncation_resumes_with_range_header(
        self, tmp_path: Path, progress: MagicMock
    ) -> None:
        content = b"abcdefghij"
        file = _file_cfg(
            "model.safetensors",
            "https://example.com/model",
            sha256=hashlib.sha256(content).hexdigest(),
            size_bytes=len(content),
        )
        path = tmp_path / file.filename
        first = _mock_http_response([content[:4]], content_length=str(len(content)))
        second = _mock_http_response(
            [content[4:]], content_length=str(len(content) - 4), status_code=206
        )
        second.headers["content-range"] = "bytes 4-9/10"
        client = MagicMock()
        client.stream = MagicMock(side_effect=[_make_async_cm(first), _make_async_cm(second)])

        with patch("ai_content_service.downloader._RetryWait", return_value=lambda _: 0.0):
            await ModelDownloader(Settings(download_max_attempts=2))._download_file(
                file, path, progress, task_id=TaskID(0), client=client
            )

        assert path.read_bytes() == content
        assert client.stream.call_args_list[1].kwargs["headers"]["Range"] == "bytes=4-"
        assert len(content[:4]) + len(content[4:]) < 2 * len(content)

    async def test_stuck_resumed_truncation_discards_partial_and_restarts(
        self, tmp_path: Path, progress: MagicMock
    ) -> None:
        content = b"abcdef"
        file = _file_cfg(
            "model.safetensors",
            "https://example.com/model",
            sha256=hashlib.sha256(content).hexdigest(),
            size_bytes=len(content),
        )
        path = tmp_path / file.filename
        part_path = path.with_name(f"{path.name}.part")
        part_path.write_bytes(content[:3])
        stuck = _mock_http_response([], content_length="3", status_code=206)
        stuck.headers["content-range"] = "bytes 3-5/6"
        complete = _mock_http_response([content], content_length=str(len(content)))
        client = MagicMock()
        client.stream = MagicMock(side_effect=[_make_async_cm(stuck), _make_async_cm(complete)])

        with patch("ai_content_service.downloader._RetryWait", return_value=lambda _: 0.0):
            await ModelDownloader(Settings(download_max_attempts=2))._download_file(
                file, path, progress, task_id=TaskID(0), client=client
            )

        assert path.read_bytes() == content
        assert client.stream.call_args_list[0].kwargs["headers"]["Range"] == "bytes=3-"
        assert "Range" not in client.stream.call_args_list[1].kwargs["headers"]

    async def test_checksum_disabled_chunked_download_is_warned_as_unverified(
        self, tmp_path: Path, progress: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        content = b"content"
        file = _file_cfg(
            "model.safetensors",
            "https://example.com/model",
            sha256=hashlib.sha256(content).hexdigest(),
        )
        path = tmp_path / file.filename
        response = _mock_http_response([content])
        client = MagicMock()
        client.stream = MagicMock(return_value=_make_async_cm(response))

        with caplog.at_level(logging.WARNING, logger="ai_content_service.downloader"):
            await ModelDownloader(Settings(verify_checksums=False))._download_file(
                file, path, progress, task_id=TaskID(0), client=client
            )

        unverified = self._events(caplog, "download.unverified")
        assert len(unverified) == 1
        assert unverified[0]["reason"] == "checksum verification disabled"

    async def test_unverified_warning_is_emitted_once_after_retries(
        self, tmp_path: Path, progress: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        content = b"abcdefghij"
        file = _file_cfg("model.safetensors", "https://example.com/model")
        path = tmp_path / file.filename
        first = _mock_http_response([content[:3]], content_length="10")
        second = _mock_http_response([content[3:6]], content_length="7", status_code=206)
        second.headers["content-range"] = "bytes 3-9/10"
        third = _mock_http_response([content[6:]], status_code=206)
        client = MagicMock()
        client.stream = MagicMock(
            side_effect=[_make_async_cm(first), _make_async_cm(second), _make_async_cm(third)]
        )

        with (
            caplog.at_level(logging.WARNING, logger="ai_content_service.downloader"),
            patch("ai_content_service.downloader._RetryWait", return_value=lambda _: 0.0),
        ):
            await ModelDownloader(Settings(download_max_attempts=3))._download_file(
                file, path, progress, task_id=TaskID(0), client=client
            )

        assert path.read_bytes() == content
        assert client.stream.call_count == 3
        assert len(self._events(caplog, "download.unverified")) == 1


class TestResume:
    """Tests for HTTP Range-based resume of a seeded .part file."""

    @pytest.fixture
    def progress(self) -> MagicMock:
        p = MagicMock()
        p.add_task.return_value = 0
        return p

    async def test_partial_file_resumed_with_range_header(
        self, tmp_path: Path, downloader: ModelDownloader, progress: MagicMock
    ) -> None:
        existing_bytes = b"AAAA"
        new_bytes = b"BBBBBB"
        full_content = existing_bytes + new_bytes
        sha256 = hashlib.sha256(full_content).hexdigest()

        path = tmp_path / "model.safetensors"
        part_path = path.with_name(f"{path.name}.part")
        part_path.write_bytes(existing_bytes)

        file_cfg = _file_cfg(
            "model.safetensors", "https://example.com/model.safetensors", sha256=sha256
        )
        http_p, mock_client, _resp = _patch_http([new_bytes], status_code=206)
        with http_p:
            await downloader._download_file(
                file_cfg, path, progress, task_id=TaskID(0), client=mock_client
            )

        sent_headers = mock_client.stream.call_args.kwargs["headers"]
        assert sent_headers["Range"] == f"bytes={len(existing_bytes)}-"
        assert path.read_bytes() == full_content
        assert not part_path.exists()

    async def test_no_part_file_omits_range_header(
        self, tmp_path: Path, downloader: ModelDownloader, progress: MagicMock
    ) -> None:
        """No pre-existing .part file -> no Range header is sent."""
        content = b"full fresh content"
        file_cfg = _file_cfg("model.safetensors", "https://example.com/model.safetensors")
        path = tmp_path / "model.safetensors"

        http_p, mock_client, _resp = _patch_http([content], status_code=200)
        with http_p:
            await downloader._download_file(
                file_cfg, path, progress, task_id=TaskID(0), client=mock_client
            )

        sent_headers = mock_client.stream.call_args.kwargs["headers"]
        assert "Range" not in sent_headers
        assert path.read_bytes() == content

    async def test_resume_falls_back_on_200(
        self, tmp_path: Path, downloader: ModelDownloader, progress: MagicMock
    ) -> None:
        """Server ignores Range and returns 200 -> truncate and restart from zero."""
        stale_partial = b"stale-partial-data-from-a-previous-attempt"
        full_new_content = b"complete fresh content"
        sha256 = hashlib.sha256(full_new_content).hexdigest()

        path = tmp_path / "model.safetensors"
        part_path = path.with_name(f"{path.name}.part")
        part_path.write_bytes(stale_partial)

        file_cfg = _file_cfg(
            "model.safetensors", "https://example.com/model.safetensors", sha256=sha256
        )
        http_p, mock_client, _resp = _patch_http([full_new_content], status_code=200)
        with http_p:
            await downloader._download_file(
                file_cfg, path, progress, task_id=TaskID(0), client=mock_client
            )

        assert path.read_bytes() == full_new_content
        assert not part_path.exists()


class TestRetry:
    """Tests for tenacity-driven retry of the HTTP transfer."""

    @pytest.fixture
    def progress(self) -> MagicMock:
        p = MagicMock()
        p.add_task.return_value = 0
        return p

    async def test_transport_error_retried_then_succeeds(
        self, tmp_path: Path, downloader: ModelDownloader, progress: MagicMock
    ) -> None:
        file_cfg = _file_cfg("model.safetensors", "https://example.com/model.safetensors")
        path = tmp_path / "model.safetensors"
        calls = 0

        async def flaky(
            _file: object,
            _path: object,
            _part_path: object,
            _progress: object,
            _task_id: object,
            _on_bytes: object,
            _client: object = None,
        ) -> None:
            nonlocal calls
            calls += 1
            if calls < 3:
                raise httpx.ConnectError("boom", request=MagicMock())
            path.write_bytes(b"ok")

        with patch.object(downloader, "_stream_to_part", side_effect=flaky):
            await downloader._download_http(
                file_cfg, path, progress, task_id=TaskID(0), on_bytes=None, client=MagicMock()
            )

        assert calls == 3
        assert path.read_bytes() == b"ok"

    async def test_4xx_not_retried(
        self, tmp_path: Path, downloader: ModelDownloader, progress: MagicMock
    ) -> None:
        file_cfg = _file_cfg("model.safetensors", "https://example.com/model.safetensors")
        path = tmp_path / "model.safetensors"
        calls = 0

        async def fail_4xx(
            _file: object,
            _path: object,
            _part_path: object,
            _progress: object,
            _task_id: object,
            _on_bytes: object,
            _client: object = None,
        ) -> None:
            nonlocal calls
            calls += 1
            raise httpx.HTTPStatusError(
                "404", request=MagicMock(), response=MagicMock(status_code=404)
            )

        with (
            patch.object(downloader, "_stream_to_part", side_effect=fail_4xx),
            pytest.raises(httpx.HTTPStatusError),
        ):
            await downloader._download_http(
                file_cfg, path, progress, task_id=TaskID(0), on_bytes=None, client=MagicMock()
            )

        assert calls == 1


class TestFinalRemediation:
    @pytest.fixture
    def progress(self) -> MagicMock:
        p = MagicMock()
        p.add_task.return_value = 0
        return p

    @pytest.mark.parametrize("content_length", ["", "1234, 1234"])
    async def test_malformed_content_length_does_not_abort_download(
        self,
        content_length: str,
        tmp_path: Path,
        downloader: ModelDownloader,
        progress: MagicMock,
    ) -> None:
        file_cfg = _file_cfg("model.safetensors", "https://example.com/model.safetensors")
        path = tmp_path / "model.safetensors"
        http_p, mock_client, _resp = _patch_http([b"data"], content_length=content_length)

        with http_p:
            await downloader._download_file(
                file_cfg, path, progress, task_id=TaskID(0), client=mock_client
            )

        assert path.read_bytes() == b"data"

    async def test_429_then_200_is_retried(self, tmp_path: Path, progress: MagicMock) -> None:
        settings = Settings(download_max_attempts=2)
        dl = ModelDownloader(settings)
        file_cfg = _file_cfg("model.safetensors", "https://example.com/model.safetensors")
        path = tmp_path / "model.safetensors"
        rate_limited = _mock_http_response([], status_code=429)
        rate_limited.headers["retry-after"] = "0"
        success = _mock_http_response([b"data"])
        mock_client = MagicMock()
        mock_client.stream = MagicMock(
            side_effect=[_make_async_cm(rate_limited), _make_async_cm(success)]
        )

        await dl._download_file(file_cfg, path, progress, task_id=TaskID(0), client=mock_client)

        assert mock_client.stream.call_count == 2
        assert path.read_bytes() == b"data"

    async def test_persistent_429_stops_at_configured_attempt_count(
        self, tmp_path: Path, progress: MagicMock
    ) -> None:
        settings = Settings(download_max_attempts=2)
        dl = ModelDownloader(settings)
        file_cfg = _file_cfg("model.safetensors", "https://example.com/model.safetensors")
        response = _mock_http_response([], status_code=429)
        response.headers["retry-after"] = "0"
        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=_make_async_cm(response))

        with pytest.raises(httpx.HTTPStatusError):
            await dl._download_file(
                file_cfg,
                tmp_path / "model.safetensors",
                progress,
                task_id=TaskID(0),
                client=mock_client,
            )

        assert mock_client.stream.call_count == 2

    def test_retry_after_wait_prefers_header_and_clamps(self) -> None:
        response = _mock_http_response([], status_code=429)
        response.headers["retry-after"] = "999999"
        error = httpx.HTTPStatusError("429", request=MagicMock(), response=response)
        state = MagicMock()
        state.outcome.failed = True
        state.outcome.exception.return_value = error

        assert _RetryWait(120)(state) == 120

        response.headers["retry-after"] = "5"
        assert _RetryWait(120)(state) == 5

    async def test_429_raises_with_no_partial_file(
        self, tmp_path: Path, progress: MagicMock
    ) -> None:
        response = _mock_http_response([], status_code=429)
        response.headers["retry-after"] = "0"
        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=_make_async_cm(response))
        file_cfg = _file_cfg("model.safetensors", "https://example.com/model.safetensors")
        path = tmp_path / "model.safetensors"

        with pytest.raises(httpx.HTTPStatusError):
            await ModelDownloader(Settings(download_max_attempts=1))._stream_to_part(
                file_cfg,
                path,
                path.with_name(f"{path.name}.part"),
                progress,
                TaskID(0),
                None,
                mock_client,
            )

        assert response.raise_for_status.called

    async def test_429_with_resume_range_also_raises(
        self, tmp_path: Path, progress: MagicMock
    ) -> None:
        response = _mock_http_response([], status_code=429)
        response.headers["retry-after"] = "0"
        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=_make_async_cm(response))
        file_cfg = _file_cfg("model.safetensors", "https://example.com/model.safetensors")
        path = tmp_path / "model.safetensors"
        part_path = path.with_name(f"{path.name}.part")
        part_path.write_bytes(b"partial")

        with pytest.raises(httpx.HTTPStatusError):
            await ModelDownloader(Settings(download_max_attempts=1))._stream_to_part(
                file_cfg, path, part_path, progress, TaskID(0), None, mock_client
            )

        assert response.raise_for_status.called
        assert part_path.exists()

    async def test_truncated_response_is_not_installed_without_checksum(
        self, tmp_path: Path, progress: MagicMock
    ) -> None:
        response = _mock_http_response([b"x" * 400], content_length="1000")
        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=_make_async_cm(response))
        file_cfg = _file_cfg("model.safetensors", "https://example.com/model.safetensors")
        path = tmp_path / "model.safetensors"
        dl = ModelDownloader(
            Settings(verify_checksums=False, skip_existing=False, download_max_attempts=1)
        )

        with pytest.raises(
            DownloadError,
            match=r"model\.safetensors: transfer truncated on every one of 1 attempts "
            r"\(last: 400 of 1000 bytes; resuming preserved partial files\)",
        ):
            await dl._download_file(file_cfg, path, progress, task_id=TaskID(0), client=mock_client)

        assert not path.exists()
        assert path.with_name(f"{path.name}.part").exists()

    async def test_truncated_resumed_response_is_not_installed(
        self, tmp_path: Path, progress: MagicMock
    ) -> None:
        response = _mock_http_response([b"y" * 100], content_length="500", status_code=206)
        response.headers["content-range"] = "bytes 500-999/1000"
        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=_make_async_cm(response))
        file_cfg = _file_cfg("model.safetensors", "https://example.com/model.safetensors")
        path = tmp_path / "model.safetensors"
        part_path = path.with_name(f"{path.name}.part")
        part_path.write_bytes(b"x" * 500)
        dl = ModelDownloader(Settings(verify_checksums=False, download_max_attempts=1))

        with pytest.raises(
            DownloadError,
            match=r"model\.safetensors: transfer truncated on every one of 1 attempts "
            r"\(last: 600 of 1000 bytes; resuming preserved partial files\)",
        ):
            await dl._download_file(file_cfg, path, progress, task_id=TaskID(0), client=mock_client)

        assert not path.exists()
        assert part_path.exists()

    async def test_stale_part_is_translated_with_offset_and_attempt_count(
        self, tmp_path: Path, progress: MagicMock
    ) -> None:
        dl = ModelDownloader(Settings(download_max_attempts=1))
        file_cfg = _file_cfg("model.safetensors", "https://example.com/model.safetensors")

        with (
            patch.object(
                dl,
                "_stream_to_part",
                side_effect=_StalePartError("model.safetensors", 4096),
            ),
            pytest.raises(
                DownloadError,
                match=r"could not restart after discarding a stale partial download "
                r"\(4096 bytes\) across 1 attempts",
            ),
        ):
            await dl._download_http(
                file_cfg,
                tmp_path / file_cfg.filename,
                progress,
                task_id=TaskID(0),
                on_bytes=None,
                client=MagicMock(),
            )

    async def test_download_all_failure_reason_keeps_transfer_context(self, tmp_path: Path) -> None:
        response = _mock_http_response([b"x" * 400], content_length="1000")
        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=_make_async_cm(response))
        dl = ModelDownloader(Settings(verify_checksums=False, download_max_attempts=1))
        file_cfg = _file_cfg("model.safetensors", "https://example.com/model.safetensors")
        model = _model_cfg("m", "diffusion_models", [file_cfg])

        with patch.object(dl, "_build_client", return_value=_make_async_cm(mock_client)):
            report = await dl.download_all([model], tmp_path / "models")

        assert report.ok is False
        assert report.failed[0].reason == (
            "model.safetensors: transfer truncated on every one of 1 attempts "
            "(last: 400 of 1000 bytes; resuming preserved partial files)"
        )


class TestSpaceRequirement:
    @pytest.fixture
    def progress(self) -> MagicMock:
        p = MagicMock()
        p.add_task.return_value = 0
        return p

    @staticmethod
    def _space_event(caplog: pytest.LogCaptureFixture) -> dict[str, object]:
        events = [
            record.msg
            for record in caplog.records
            if isinstance(record.msg, dict) and record.msg.get("event") == "download.space.check"
        ]
        assert len(events) == 1
        return events[0]

    async def test_existing_declared_files_are_idempotent_when_space_is_low(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        content_a = b"aaaa"
        content_b = b"bbbbb"
        files = [
            _file_cfg(
                "a.safetensors",
                "https://example.com/a",
                sha256=hashlib.sha256(content_a).hexdigest(),
                size_bytes=len(content_a),
            ),
            _file_cfg(
                "b.safetensors",
                "https://example.com/b",
                sha256=hashlib.sha256(content_b).hexdigest(),
                size_bytes=len(content_b),
            ),
        ]
        model = _model_cfg("m", "diffusion_models", files)
        models_path = tmp_path / "models"
        model_path = models_path / model.target_subpath
        model_path.mkdir(parents=True)
        (model_path / files[0].filename).write_bytes(content_a)
        (model_path / files[1].filename).write_bytes(content_b)
        dl = ModelDownloader(Settings(skip_existing=True))
        client = MagicMock()

        with (
            caplog.at_level(logging.DEBUG, logger="ai_content_service.downloader"),
            patch(
                "ai_content_service.downloader.shutil.disk_usage", return_value=MagicMock(free=1)
            ),
            patch.object(dl, "_build_client", return_value=_make_async_cm(client)),
        ):
            report = await dl.download_all([model], models_path)

        assert report.ok
        assert client.stream.call_count == 0
        event = self._space_event(caplog)
        assert event["declared"] == 9
        assert event["required"] == 0
        assert event["skipped"] == 9

    async def test_only_absent_file_counts_toward_requirement(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        files = [
            _file_cfg(
                "present",
                "https://example.com/present",
                sha256=hashlib.sha256(b"1234").hexdigest(),
                size_bytes=4,
            ),
            _file_cfg("absent", "https://example.com/absent", size_bytes=6),
        ]
        model = _model_cfg("m", "diffusion_models", files)
        models_path = tmp_path / "models"
        model_path = models_path / model.target_subpath
        model_path.mkdir(parents=True)
        (model_path / files[0].filename).write_bytes(b"1234")
        dl = ModelDownloader(Settings())

        with (
            caplog.at_level(logging.DEBUG, logger="ai_content_service.downloader"),
            patch(
                "ai_content_service.downloader.shutil.disk_usage", return_value=MagicMock(free=7)
            ),
            patch.object(dl, "_download_file", new_callable=AsyncMock),
        ):
            report = await dl.download_all([model], models_path)

        assert report.ok
        event = self._space_event(caplog)
        assert event["required"] == 6
        assert event["skipped"] == 4
        assert event["pending_existing"] == 0

    async def test_size_matched_unhashed_files_block_before_any_request(
        self, tmp_path: Path
    ) -> None:
        hashed = b"good"
        unhashed_a = b"unhashed-a"
        unhashed_b = b"unhashed-bb"
        files = [
            _file_cfg(
                "hashed",
                "https://example.com/hashed",
                sha256=hashlib.sha256(hashed).hexdigest(),
                size_bytes=len(hashed),
            ),
            _file_cfg("unhashed-a", "https://example.com/unhashed-a", size_bytes=len(unhashed_a)),
            _file_cfg("unhashed-b", "https://example.com/unhashed-b", size_bytes=len(unhashed_b)),
        ]
        model = _model_cfg("m", "diffusion_models", files)
        models_path = tmp_path / "models"
        model_path = models_path / model.target_subpath
        model_path.mkdir(parents=True)
        for file, content in zip(files, [hashed, unhashed_a, unhashed_b], strict=True):
            (model_path / file.filename).write_bytes(content)
        downloader = ModelDownloader(Settings(skip_existing=True))

        with (
            patch(
                "ai_content_service.downloader.shutil.disk_usage", return_value=MagicMock(free=10)
            ),
            patch.object(downloader, "_build_client") as build_client,
            pytest.raises(DownloadError, match="insufficient disk space"),
        ):
            await downloader.download_all([model], models_path)

        build_client.assert_not_called()

    async def test_partial_file_bytes_are_credited(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        file = _file_cfg("pending", "https://example.com/pending", size_bytes=100)
        model = _model_cfg("m", "diffusion_models", [file])
        models_path = tmp_path / "models"
        model_path = models_path / model.target_subpath
        model_path.mkdir(parents=True)
        (model_path / f"{file.filename}.part").write_bytes(b"x" * 60)
        dl = ModelDownloader(Settings())

        with (
            caplog.at_level(logging.DEBUG, logger="ai_content_service.downloader"),
            patch(
                "ai_content_service.downloader.shutil.disk_usage", return_value=MagicMock(free=50)
            ),
            patch.object(dl, "_download_file", new_callable=AsyncMock),
        ):
            report = await dl.download_all([model], models_path)

        assert report.ok
        assert self._space_event(caplog)["required"] == 40

    async def test_wrong_sized_existing_file_is_pending(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        file = _file_cfg("model", "https://example.com/model", size_bytes=5)
        model = _model_cfg("m", "diffusion_models", [file])
        models_path = tmp_path / "models"
        model_path = models_path / model.target_subpath
        model_path.mkdir(parents=True)
        (model_path / file.filename).write_bytes(b"1234")
        dl = ModelDownloader(Settings())

        with (
            caplog.at_level(logging.DEBUG, logger="ai_content_service.downloader"),
            patch(
                "ai_content_service.downloader.shutil.disk_usage", return_value=MagicMock(free=6)
            ),
            patch.object(dl, "_download_file", new_callable=AsyncMock),
        ):
            report = await dl.download_all([model], models_path)

        assert report.ok
        event = self._space_event(caplog)
        assert event["required"] == 5
        assert event["skipped"] == 0
        assert event["pending_existing"] == 4

    async def test_skip_existing_false_requires_present_files_too(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        files = [
            _file_cfg(
                "a",
                "https://example.com/a",
                sha256=hashlib.sha256(b"x" * 4).hexdigest(),
                size_bytes=4,
            ),
            _file_cfg(
                "b",
                "https://example.com/b",
                sha256=hashlib.sha256(b"x" * 6).hexdigest(),
                size_bytes=6,
            ),
        ]
        model = _model_cfg("m", "diffusion_models", files)
        models_path = tmp_path / "models"
        model_path = models_path / model.target_subpath
        model_path.mkdir(parents=True)
        for file in files:
            (model_path / file.filename).write_bytes(b"x" * (file.size_bytes or 0))
        dl = ModelDownloader(Settings(skip_existing=False))

        with (
            caplog.at_level(logging.DEBUG, logger="ai_content_service.downloader"),
            patch(
                "ai_content_service.downloader.shutil.disk_usage", return_value=MagicMock(free=11)
            ),
            patch.object(dl, "_download_file", new_callable=AsyncMock),
        ):
            report = await dl.download_all([model], models_path)

        assert report.ok
        event = self._space_event(caplog)
        assert event["required"] == 10
        assert event["skipped"] == 0

    async def test_insufficient_pending_space_reports_pending_and_skipped_totals(
        self, tmp_path: Path
    ) -> None:
        files = [
            _file_cfg(
                "present",
                "https://example.com/present",
                sha256="0" * 64,
                size_bytes=1_000_000_000,
            ),
            _file_cfg("pending", "https://example.com/pending", size_bytes=2_000_000_000),
        ]
        model = _model_cfg("m", "diffusion_models", files)
        models_path = tmp_path / "models"
        model_path = models_path / model.target_subpath
        model_path.mkdir(parents=True)
        present_path = model_path / files[0].filename
        present_path.touch()
        with present_path.open("r+b") as present_file:
            present_file.truncate(files[0].size_bytes or 0)
        dl = ModelDownloader(Settings())

        with (
            patch(
                "ai_content_service.downloader.shutil.disk_usage", return_value=MagicMock(free=0)
            ),
            pytest.raises(
                DownloadError,
                match=r"need ~2\.1 GB for 1 pending file\(s\) \(1\.0 GB already present\)",
            ),
        ):
            await dl.download_all([model], models_path)

        assert model_path.exists()

    async def test_unknown_sizes_skip_space_check(self, tmp_path: Path) -> None:
        file = _file_cfg("model", "https://example.com/model")
        model = _model_cfg("m", "diffusion_models", [file])
        dl = ModelDownloader(Settings())
        with (
            patch("ai_content_service.downloader.shutil.disk_usage") as disk_usage,
            patch.object(dl, "_download_file", new_callable=AsyncMock),
        ):
            report = await dl.download_all([model], tmp_path / "models")

        assert report.ok
        disk_usage.assert_not_called()

    async def test_downloads_request_identity_encoding(
        self, tmp_path: Path, downloader: ModelDownloader, progress: MagicMock
    ) -> None:
        file_cfg = _file_cfg("model.safetensors", "https://example.com/model.safetensors")
        mock_p, mock_client, _response = _patch_http([b"data"])

        with mock_p:
            await downloader._download_file(
                file_cfg,
                tmp_path / file_cfg.filename,
                progress,
                task_id=TaskID(0),
                client=mock_client,
            )

        assert mock_client.stream.call_args.kwargs["headers"]["Accept-Encoding"] == "identity"

    async def test_checksum_mismatch_retries_from_clean_file_then_succeeds(
        self, tmp_path: Path, progress: MagicMock
    ) -> None:
        content = b"correct content"
        file_cfg = _file_cfg(
            "model.safetensors",
            "https://example.com/model.safetensors",
            sha256=hashlib.sha256(content).hexdigest(),
        )
        path = tmp_path / "model.safetensors"
        bad = _mock_http_response([b"wrong content"])
        good = _mock_http_response([content])
        mock_client = MagicMock()
        mock_client.stream = MagicMock(side_effect=[_make_async_cm(bad), _make_async_cm(good)])
        dl = ModelDownloader(Settings(skip_existing=False, download_max_attempts=2))

        await dl._download_file(file_cfg, path, progress, task_id=TaskID(0), client=mock_client)

        assert mock_client.stream.call_count == 2
        assert path.read_bytes() == content

    async def test_persistent_checksum_mismatch_names_attempt_count(
        self, tmp_path: Path, progress: MagicMock
    ) -> None:
        content = b"wrong content"
        file_cfg = _file_cfg(
            "model.safetensors",
            "https://example.com/model.safetensors",
            sha256="0" * 64,
        )
        path = tmp_path / "model.safetensors"
        response = _mock_http_response([content])
        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=_make_async_cm(response))
        dl = ModelDownloader(Settings(skip_existing=False, download_max_attempts=2))

        with pytest.raises(DownloadError, match=r"checksum never matched after 2 attempts"):
            await dl._download_file(file_cfg, path, progress, task_id=TaskID(0), client=mock_client)

        assert not path.exists()
        assert not path.with_name(f"{path.name}.part").exists()

    async def test_default_civitai_auth_failure_points_to_query_fallback(
        self, tmp_path: Path, progress: MagicMock
    ) -> None:
        settings = Settings(civitai_api_token="civitai-token", download_max_attempts=1)  # type: ignore[arg-type]
        dl = ModelDownloader(settings)
        response = _mock_http_response([], status_code=401)
        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=_make_async_cm(response))
        file_cfg = _file_cfg("model.safetensors", "https://civitai.red/api/download/models/1")

        with pytest.raises(DownloadError, match="ACS_CIVITAI_ALLOW_QUERY_TOKEN_FALLBACK"):
            await dl._download_file(
                file_cfg,
                tmp_path / file_cfg.filename,
                progress,
                task_id=TaskID(0),
                client=mock_client,
            )

        assert mock_client.stream.call_count == 1

    async def test_disk_space_is_checked_before_mkdir(
        self, tmp_path: Path, downloader: ModelDownloader
    ) -> None:
        model = _model_cfg(
            "m",
            "diffusion_models",
            [_file_cfg("model.safetensors", "https://example.com/model", size_bytes=100)],
        )

        with (
            patch(
                "ai_content_service.downloader.shutil.disk_usage", return_value=MagicMock(free=0)
            ),
            patch.object(downloader, "_download_file", new_callable=AsyncMock) as download,
            pytest.raises(DownloadError, match="insufficient disk space"),
        ):
            await downloader.download_all([model], tmp_path / "models")

        download.assert_not_called()
        assert not (tmp_path / "models").exists()


class TestR2PullAtomic:
    """Tests for atomic temp-then-rename semantics on the R2 cache-pull path."""

    @pytest.fixture
    def progress(self) -> MagicMock:
        p = MagicMock()
        p.add_task.return_value = 0
        return p

    def _r2_settings(self) -> Settings:
        return Settings(
            r2_s3_endpoint="https://example.r2.cloudflarestorage.com",
            r2_readonly_access_key_id="key",
            r2_readonly_secret_access_key="secret",  # type: ignore[arg-type]
        )

    async def test_r2_pull_uses_temp_and_renames(self, tmp_path: Path, progress: MagicMock) -> None:
        content = b"cached weights"
        sha256 = hashlib.sha256(content).hexdigest()
        dl = ModelDownloader(self._r2_settings())
        file_cfg = _file_cfg(
            "model.safetensors", "https://example.com/model.safetensors", sha256=sha256
        )
        path = tmp_path / "model.safetensors"
        tmp_path_r2 = path.with_name(f"{path.name}.r2tmp")

        def fake_pull(*, dest_path: Path, **_kwargs: object) -> None:
            dest_path.write_bytes(content)

        with patch("ai_content_service.downloader.r2_transfer.pull", side_effect=fake_pull):
            await dl._download_file(file_cfg, path, progress, task_id=TaskID(0), client=MagicMock())

        assert path.read_bytes() == content
        assert not tmp_path_r2.exists()

    async def test_r2_pull_corrupt_leaves_canonical_path_untouched(
        self, tmp_path: Path, progress: MagicMock
    ) -> None:
        expected_sha256 = hashlib.sha256(b"expected content").hexdigest()
        dl = ModelDownloader(self._r2_settings())
        file_cfg = _file_cfg(
            "model.safetensors", "https://example.com/model.safetensors", sha256=expected_sha256
        )
        path = tmp_path / "model.safetensors"
        tmp_path_r2 = path.with_name(f"{path.name}.r2tmp")

        def fake_pull_corrupt(*, dest_path: Path, **_kwargs: object) -> None:
            dest_path.write_bytes(b"corrupted data")

        with (
            patch("ai_content_service.downloader.r2_transfer.pull", side_effect=fake_pull_corrupt),
            patch.object(dl, "_download_http", new_callable=AsyncMock),
        ):
            await dl._download_file(file_cfg, path, progress, task_id=TaskID(0), client=MagicMock())

        assert not path.exists()
        assert not tmp_path_r2.exists()


# ---------------------------------------------------------------------------
# download_all
# ---------------------------------------------------------------------------


class TestDownloadAll:
    """Tests for ModelDownloader.download_all."""

    async def test_present_hashed_snapshot_files_with_empty_urls_are_skipped(
        self, tmp_path: Path, downloader: ModelDownloader
    ) -> None:
        content = b"snapshot model"
        digest = hashlib.sha256(content).hexdigest()
        models_path = tmp_path / "models"
        destination = models_path / "checkpoints" / "model.safetensors"
        destination.parent.mkdir(parents=True)
        destination.write_bytes(content)
        model = _model_cfg(
            "m",
            "checkpoints",
            [_file_cfg(destination.name, "", sha256=digest, size_bytes=len(content))],
        )
        client = MagicMock()
        with patch.object(downloader, "_build_client", return_value=_make_async_cm(client)):
            report = await downloader.download_all([model], models_path)

        assert report.ok is True
        client.stream.assert_not_called()

    async def test_missing_url_guard_names_only_absent_files(
        self, tmp_path: Path, downloader: ModelDownloader
    ) -> None:
        content = b"present"
        present = tmp_path / "models" / "checkpoints" / "present.safetensors"
        present.parent.mkdir(parents=True)
        present.write_bytes(content)
        model = _model_cfg(
            "m",
            "checkpoints",
            [
                _file_cfg(
                    present.name,
                    "",
                    sha256=hashlib.sha256(content).hexdigest(),
                    size_bytes=len(content),
                ),
                _file_cfg("absent.safetensors", "", sha256="0" * 64, size_bytes=10),
            ],
        )

        with pytest.raises(DownloadError) as exc_info:
            await downloader.download_all([model], tmp_path / "models")

        assert "absent.safetensors" in str(exc_info.value)
        assert "present.safetensors" not in str(exc_info.value)

    async def test_empty_urls_are_all_rejected_when_skip_existing_is_disabled(
        self, tmp_path: Path
    ) -> None:
        dl = ModelDownloader(Settings(skip_existing=False))
        model = _model_cfg(
            "m",
            "checkpoints",
            [_file_cfg("a.safetensors", ""), _file_cfg("b.safetensors", "")],
        )

        with pytest.raises(DownloadError) as exc_info:
            await dl.download_all([model], tmp_path / "models")

        message = str(exc_info.value)
        assert "a.safetensors" in message
        assert "b.safetensors" in message

    async def test_present_file_without_checksum_is_not_exempt(
        self, tmp_path: Path, downloader: ModelDownloader
    ) -> None:
        models_path = tmp_path / "models"
        destination = models_path / "checkpoints" / "model.safetensors"
        destination.parent.mkdir(parents=True)
        destination.write_bytes(b"present")
        model = _model_cfg("m", "checkpoints", [_file_cfg(destination.name, "", size_bytes=7)])

        with pytest.raises(DownloadError, match=r"model\.safetensors"):
            await downloader.download_all([model], models_path)

    async def test_absent_empty_url_fails_before_mkdir(
        self, tmp_path: Path, downloader: ModelDownloader
    ) -> None:
        model = _model_cfg(
            "m", "checkpoints", [_file_cfg("missing.safetensors", "", sha256="0" * 64)]
        )
        models_path = tmp_path / "models"

        with pytest.raises(DownloadError):
            await downloader.download_all([model], models_path)

        assert not models_path.exists()

    async def test_returns_count_of_successful_downloads(
        self, tmp_path: Path, downloader: ModelDownloader
    ) -> None:
        files = [
            _file_cfg("a.safetensors", "https://example.com/a"),
            _file_cfg("b.safetensors", "https://example.com/b"),
        ]
        model = _model_cfg("m", "diffusion_models", files)

        with patch.object(downloader, "_download_file", new_callable=AsyncMock):
            result = await downloader.download_all([model], tmp_path)

        assert result.succeeded == 2
        assert result.ok is True
        assert result.failed == ()

    async def test_empty_model_list_returns_zero(
        self, tmp_path: Path, downloader: ModelDownloader
    ) -> None:
        result = await downloader.download_all([], tmp_path)
        assert result.succeeded == 0
        assert result.ok is True

    async def test_failed_download_not_counted(
        self, tmp_path: Path, downloader: ModelDownloader
    ) -> None:
        files = [
            _file_cfg("ok.safetensors", "https://example.com/ok"),
            _file_cfg("fail.safetensors", "https://example.com/fail"),
        ]
        model = _model_cfg("m", "diffusion_models", files)
        call_count = 0

        async def _fail_first(*_args: object, **_kwargs: object) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise DownloadError("Network error")

        with patch.object(downloader, "_download_file", side_effect=_fail_first):
            result = await downloader.download_all([model], tmp_path)

        assert result.succeeded == 1
        assert result.ok is False
        assert len(result.failed) == 1

    async def test_subdirectory_creates_nested_path(
        self, tmp_path: Path, downloader: ModelDownloader
    ) -> None:
        files = [_file_cfg("model.safetensors", "https://example.com/model")]
        model = _model_cfg("m", "clip", files, subdirectory="sdxl")

        with patch.object(downloader, "_download_file", new_callable=AsyncMock):
            await downloader.download_all([model], tmp_path)

        assert (tmp_path / "clip" / "sdxl").is_dir()

    async def test_no_subdirectory_flat_type_dir(
        self, tmp_path: Path, downloader: ModelDownloader
    ) -> None:
        files = [_file_cfg("model.safetensors", "https://example.com/model")]
        model = _model_cfg("m", "vae", files)

        with patch.object(downloader, "_download_file", new_callable=AsyncMock):
            await downloader.download_all([model], tmp_path)

        assert (tmp_path / "vae").is_dir()

    async def test_on_progress_called_after_each_file(
        self, tmp_path: Path, downloader: ModelDownloader
    ) -> None:
        files = [
            _file_cfg("a.safetensors", "https://example.com/a", size_bytes=50),
            _file_cfg("b.safetensors", "https://example.com/b", size_bytes=50),
        ]
        model = _model_cfg("m", "diffusion_models", files)
        on_progress = AsyncMock()

        with patch.object(downloader, "_download_file", new_callable=AsyncMock):
            await downloader.download_all([model], tmp_path, on_progress=on_progress)

        # Called once per completed file
        assert on_progress.call_count == 2
        # Final call: 2/2 files done
        _bytes_done, _bytes_total, files_done, files_total = on_progress.call_args.args
        assert files_done == 2
        assert files_total == 2

    async def test_multiple_models_all_counted(
        self, tmp_path: Path, downloader: ModelDownloader
    ) -> None:
        m1 = _model_cfg(
            "m1", "diffusion_models", [_file_cfg("a.safetensors", "https://example.com/a")]
        )
        m2 = _model_cfg(
            "m2",
            "vae",
            [
                _file_cfg("b.safetensors", "https://example.com/b"),
                _file_cfg("c.safetensors", "https://example.com/c"),
            ],
        )

        with patch.object(downloader, "_download_file", new_callable=AsyncMock):
            result = await downloader.download_all([m1, m2], tmp_path)

        assert result.succeeded == 3

    async def test_downloader_containment_with_symlinked_models_dir(
        self, tmp_path: Path, downloader: ModelDownloader
    ) -> None:
        """Containment check must compare against the *resolved* base for symlinked models dirs."""
        real_base = tmp_path / "real_models"
        real_base.mkdir()
        symlinked_base = tmp_path / "models_link"
        symlinked_base.symlink_to(real_base)

        files = [_file_cfg("model.safetensors", "https://example.com/model")]
        model = _model_cfg("m", "vae", files)

        with patch.object(downloader, "_download_file", new_callable=AsyncMock):
            result = await downloader.download_all([model], symlinked_base)

        assert result.succeeded == 1

    async def test_containment_raises_for_crafted_escape(
        self, tmp_path: Path, downloader: ModelDownloader
    ) -> None:
        """A filename that escapes the models dir must be rejected even if it bypassed
        ModelFileConfig's own validator (e.g. via model_construct)."""
        real_base = tmp_path / "real_models"
        real_base.mkdir()
        symlinked_base = tmp_path / "models_link"
        symlinked_base.symlink_to(real_base)

        evil_file = ModelFileConfig.model_construct(
            name="evil",
            url="https://example.com/evil",
            filename="../../etc/passwd",
            sha256=None,
            size_bytes=None,
        )
        model = _model_cfg("m", "vae", [evil_file])

        with (
            patch.object(downloader, "_download_file", new_callable=AsyncMock),
            pytest.raises(DownloadError, match="outside models dir"),
        ):
            await downloader.download_all([model], symlinked_base)

    async def test_path_traversal_check_runs_before_mkdir(
        self, tmp_path: Path, downloader: ModelDownloader
    ) -> None:
        """D11: validate-then-mutate — a rejected model creates no directories."""
        evil_file = ModelFileConfig.model_construct(
            name="evil",
            url="https://example.com/evil",
            filename="../../etc/passwd",
            sha256=None,
            size_bytes=None,
        )
        model = _model_cfg("m", "diffusion_models", [evil_file])

        with pytest.raises(DownloadError, match="outside models dir"):
            await downloader.download_all([model], tmp_path)

        assert not list(tmp_path.iterdir())

    async def test_tracker_not_used_when_on_progress_none(
        self, tmp_path: Path, downloader: ModelDownloader
    ) -> None:
        """_download_file receives on_bytes=None when on_progress is not provided."""
        files = [_file_cfg("a.safetensors", "https://example.com/a", size_bytes=100)]
        model = _model_cfg("m", "diffusion_models", files)
        captured_on_bytes: list[object] = []

        async def capture_download(
            _file: object,
            _path: object,
            _progress: object,
            _task_id: object,
            on_bytes: object = None,
            client: object = None,  # noqa: ARG001 -- must be named `client`, download_all passes it as a kwarg
        ) -> None:
            captured_on_bytes.append(on_bytes)

        with patch.object(downloader, "_download_file", side_effect=capture_download):
            await downloader.download_all([model], tmp_path)

        assert captured_on_bytes == [None]


class TestEmptyUrlGuard:
    """Tests for C3 -- an empty `url` (the snapshot placeholder) must fail
    before any request or directory creation, naming every offending file."""

    async def test_two_missing_urls_raise_naming_both_before_any_request(
        self, tmp_path: Path, downloader: ModelDownloader
    ) -> None:
        files = [
            _file_cfg("a.safetensors", ""),
            _file_cfg("b.safetensors", ""),
        ]
        model = _model_cfg("m", "diffusion_models", files)

        with (
            patch.object(downloader, "_download_file", new_callable=AsyncMock) as mock_download,
            pytest.raises(DownloadError, match="2 model file"),
        ):
            await downloader.download_all([model], tmp_path)

        mock_download.assert_not_called()
        assert not list(tmp_path.iterdir())

    async def test_error_names_every_missing_file(
        self, tmp_path: Path, downloader: ModelDownloader
    ) -> None:
        files = [_file_cfg("a.safetensors", ""), _file_cfg("b.safetensors", "")]
        model = _model_cfg("m", "diffusion_models", files)

        with pytest.raises(DownloadError) as exc_info:
            await downloader.download_all([model], tmp_path)

        assert "a.safetensors" in str(exc_info.value)
        assert "b.safetensors" in str(exc_info.value)

    async def test_mixed_empty_and_valid_urls_still_raises_nothing_downloaded(
        self, tmp_path: Path, downloader: ModelDownloader
    ) -> None:
        files = [
            _file_cfg("ok.safetensors", "https://example.com/ok"),
            _file_cfg("missing.safetensors", ""),
        ]
        model = _model_cfg("m", "diffusion_models", files)

        with (
            patch.object(downloader, "_download_file", new_callable=AsyncMock) as mock_download,
            pytest.raises(DownloadError, match=r"missing\.safetensors"),
        ):
            await downloader.download_all([model], tmp_path)

        mock_download.assert_not_called()
        assert not list(tmp_path.iterdir())

    async def test_guard_runs_before_disk_space_check(
        self, tmp_path: Path, downloader: ModelDownloader
    ) -> None:
        """The guard must fire even when disk-space accounting would otherwise
        run first (declared size > 0)."""
        files = [_file_cfg("missing.safetensors", "", size_bytes=1_000_000_000)]
        model = _model_cfg("m", "diffusion_models", files)

        with (
            patch("ai_content_service.downloader.shutil.disk_usage") as mock_disk_usage,
            pytest.raises(DownloadError, match=r"missing\.safetensors"),
        ):
            await downloader.download_all([model], tmp_path)

        mock_disk_usage.assert_not_called()


# ---------------------------------------------------------------------------
# B2/B3: sha256 normalization (config.py validator) and its downstream effects
# ---------------------------------------------------------------------------


class TestSha256Normalization:
    """Tests for D5 (config.py normalize_sha256) as observed from the downloader."""

    def test_uppercase_sha256_is_normalized_to_lowercase(self) -> None:
        content = b"model bytes"
        digest = hashlib.sha256(content).hexdigest()
        cfg = _file_cfg("m.safetensors", "https://example.com/m", sha256=digest.upper())
        assert cfg.sha256 == digest

    async def test_verify_checksum_passes_against_normalized_hash(
        self, tmp_path: Path, downloader: ModelDownloader
    ) -> None:
        content = b"model bytes"
        digest = hashlib.sha256(content).hexdigest()
        cfg = _file_cfg("m.safetensors", "https://example.com/m", sha256=digest.upper())
        path = tmp_path / "m.safetensors"
        path.write_bytes(content)

        assert await downloader._verify_checksum(path, cfg.sha256) is True  # type: ignore[arg-type]

    def test_invalid_sha256_raises_validation_error(self) -> None:
        with pytest.raises(ValueError, match="64 hexadecimal"):
            _file_cfg("m.safetensors", "https://example.com/m", sha256="not-a-hash")

    def test_63_char_sha256_raises_validation_error(self) -> None:
        with pytest.raises(ValueError, match="64 hexadecimal"):
            _file_cfg("m.safetensors", "https://example.com/m", sha256="a" * 63)

    async def test_r2_pull_key_identical_regardless_of_authored_case(self, tmp_path: Path) -> None:
        """B3: an uppercase-authored bundle must resolve to the same R2 key as a lowercase one."""
        content = b"cached weights"
        digest = hashlib.sha256(content).hexdigest()

        settings = Settings(
            r2_s3_endpoint="https://example.r2.cloudflarestorage.com",
            r2_readonly_access_key_id="key",
            r2_readonly_secret_access_key="secret",  # type: ignore[arg-type]
        )
        dl = ModelDownloader(settings)
        progress = MagicMock()
        progress.add_task.return_value = 0

        upper_cfg = _file_cfg("m.safetensors", "https://example.com/m", sha256=digest.upper())
        path = tmp_path / "m.safetensors"

        captured_keys: list[str] = []

        def fake_pull(*, key: str, dest_path: Path, **_kwargs: object) -> None:
            captured_keys.append(key)
            dest_path.write_bytes(content)

        with patch("ai_content_service.downloader.r2_transfer.pull", side_effect=fake_pull):
            await dl._download_file(
                upper_cfg, path, progress, task_id=TaskID(0), client=MagicMock()
            )

        assert captured_keys == [f"models/by-sha256/{digest}"]


# ---------------------------------------------------------------------------
# B5/D8: byte-complete .part + 416 self-heals instead of bricking the download
# ---------------------------------------------------------------------------


class TestStalePartDiscard:
    @pytest.fixture
    def progress(self) -> MagicMock:
        p = MagicMock()
        p.add_task.return_value = 0
        return p

    async def test_stale_complete_part_with_416_is_discarded_and_retried(
        self, tmp_path: Path, downloader: ModelDownloader, progress: MagicMock
    ) -> None:
        content = b"complete file contents from a previous, interrupted run"
        sha256 = hashlib.sha256(content).hexdigest()
        path = tmp_path / "model.safetensors"
        part_path = path.with_name(f"{path.name}.part")
        part_path.write_bytes(content)  # byte-complete already

        file_cfg = _file_cfg(
            "model.safetensors", "https://example.com/model.safetensors", sha256=sha256
        )

        responses = [
            _mock_http_response([], status_code=416),
            _mock_http_response([content], status_code=200),
        ]
        mock_client = MagicMock()
        mock_client.stream = MagicMock(side_effect=[_make_async_cm(r) for r in responses])
        http_cm = _make_async_cm(mock_client)

        with patch("ai_content_service.downloader.httpx.AsyncClient", return_value=http_cm):
            await downloader._download_file(
                file_cfg, path, progress, task_id=TaskID(0), client=mock_client
            )

        assert path.read_bytes() == content
        assert not part_path.exists()
        assert mock_client.stream.call_count == 2
        second_headers = mock_client.stream.call_args_list[1].kwargs["headers"]
        assert "Range" not in second_headers  # retried from zero, not resumed

    async def test_416_without_offset_is_not_special_cased(
        self, tmp_path: Path, downloader: ModelDownloader, progress: MagicMock
    ) -> None:
        """A 416 with no prior .part (offset==0) is just a normal HTTP error."""
        file_cfg = _file_cfg("model.safetensors", "https://example.com/model.safetensors")
        path = tmp_path / "model.safetensors"
        part_path = path.with_name(f"{path.name}.part")

        http_p, mock_client, _resp = _patch_http([], status_code=416)
        with http_p, pytest.raises(httpx.HTTPStatusError):
            await downloader._stream_to_part(
                file_cfg, path, part_path, progress, TaskID(0), None, mock_client
            )


# ---------------------------------------------------------------------------
# D9: _ProgressTracker is absolute per-file, not delta — idempotent across retries
# ---------------------------------------------------------------------------


class TestProgressTracker:
    async def test_set_file_bytes_overwrites_not_accumulates(self) -> None:
        on_progress = AsyncMock()
        tracker = _ProgressTracker(bytes_total=200, files_total=2, on_progress=on_progress)

        await tracker.set_file_bytes("a", 50)
        await tracker.set_file_bytes("b", 30)
        await tracker.set_file_bytes("a", 100)  # a later, larger absolute value replaces the first

        assert tracker.bytes_done == 130  # 100 (a) + 30 (b) -- not 50 + 30 + 100

    async def test_resume_does_not_double_count(self) -> None:
        """A failed attempt's partial progress plus a resumed attempt's total must not stack."""
        on_progress = AsyncMock()
        tracker = _ProgressTracker(bytes_total=100, files_total=1, on_progress=on_progress)

        await tracker.set_file_bytes("model.safetensors", 50)  # attempt 1 died at 50/100
        await tracker.set_file_bytes("model.safetensors", 100)  # attempt 2 resumed and finished

        assert tracker.bytes_done == 100

    async def test_on_file_done_increments_and_emits_final_state(self) -> None:
        calls: list[tuple[int, int, int, int]] = []

        async def on_progress(bd: int, bt: int, fd: int, ft: int) -> None:
            calls.append((bd, bt, fd, ft))

        tracker = _ProgressTracker(bytes_total=10, files_total=1, on_progress=on_progress)
        await tracker.set_file_bytes("a", 10)
        await tracker.on_file_done()

        assert calls[-1] == (10, 10, 1, 1)

    async def test_resume_progress_never_exceeds_total_end_to_end(
        self, tmp_path: Path, downloader: ModelDownloader
    ) -> None:
        """B6/D9: interrupt at 50%, resume -> final bytes_done == file size, never more."""
        first_half = b"A" * 50
        second_half = b"B" * 50
        full_content = first_half + second_half
        sha256 = hashlib.sha256(full_content).hexdigest()

        path = tmp_path / "model.safetensors"
        part_path = path.with_name(f"{path.name}.part")
        part_path.write_bytes(first_half)  # simulates attempt 1 dying at 50%

        file_cfg = _file_cfg(
            "model.safetensors", "https://example.com/model.safetensors", sha256=sha256
        )
        progress = MagicMock()
        progress.add_task.return_value = 0

        emitted: list[int] = []

        async def on_bytes(n: int) -> None:
            emitted.append(n)
            assert n <= len(full_content)

        http_p, mock_client, _resp = _patch_http([second_half], status_code=206)
        with http_p:
            await downloader._stream_to_part(
                file_cfg, path, part_path, progress, TaskID(0), on_bytes, mock_client
            )

        # Absolute, not delta: the resumed attempt reports the true cumulative total
        # directly -- it never re-emits the bare pre-existing offset (the old bug).
        assert emitted == [100]
        assert path.read_bytes() == full_content


# ---------------------------------------------------------------------------
# B7/D10: download_all returns a DownloadReport; failures are never silently ok
# ---------------------------------------------------------------------------


class TestDownloadReport:
    async def test_partial_failure_reports_redacted_url_and_reason(
        self, tmp_path: Path, downloader: ModelDownloader
    ) -> None:
        ok_file = _file_cfg("ok.safetensors", "https://example.com/ok")
        bad_file = _file_cfg("bad.safetensors", "https://civitai.com/api/download/models/1")
        model = _model_cfg("m", "diffusion_models", [ok_file, bad_file])

        async def fake_download(file: ModelFileConfig, *_args: object, **_kwargs: object) -> None:
            if file.filename == "bad.safetensors":
                raise DownloadError(
                    "404 for url 'https://civitai.com/api/download/models/1?token=leaked_secret'"
                )

        with patch.object(downloader, "_download_file", side_effect=fake_download):
            report = await downloader.download_all([model], tmp_path)

        assert report.succeeded == 1
        assert report.ok is False
        assert len(report.failed) == 1
        failure = report.failed[0]
        assert failure.filename == "bad.safetensors"
        assert "leaked_secret" not in failure.url
        assert "leaked_secret" not in failure.reason
        assert "***" in failure.reason


# ---------------------------------------------------------------------------
# D13: Content-Disposition is a warning-only cross-check; bundle filename wins
# ---------------------------------------------------------------------------


class TestContentDispositionCrossCheck:
    async def test_absent_server_filename_logs_debug_not_warning(
        self,
        tmp_path: Path,
        downloader: ModelDownloader,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        progress = MagicMock()
        progress.add_task.return_value = 0
        file_cfg = _file_cfg("bundle_name.safetensors", "https://example.com/model")
        response = _mock_http_response([b"data"])
        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=_make_async_cm(response))

        with caplog.at_level(logging.DEBUG, logger="ai_content_service.downloader"):
            await downloader._download_file(
                file_cfg,
                tmp_path / file_cfg.filename,
                progress,
                task_id=TaskID(0),
                client=mock_client,
            )

        messages = [(r.levelname, r.getMessage()) for r in caplog.records]
        assert not any(level == "WARNING" and "filename.mismatch" in msg for level, msg in messages)
        assert any(level == "DEBUG" and "filename.absent" in msg for level, msg in messages)

    async def test_same_extension_rename_logs_debug_not_warning(
        self,
        tmp_path: Path,
        downloader: ModelDownloader,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """M6a: a deliberate rename (same extension) is guaranteed noise on Civitai
        bundles — Civitai returns the uploader's original filename, while our
        bundles deliberately rename. Demote to debug so the real signal (an
        extension change) doesn't drown."""
        progress = MagicMock()
        progress.add_task.return_value = 0
        file_cfg = _file_cfg("bundle_name.safetensors", "https://example.com/model.safetensors")
        path = tmp_path / "bundle_name.safetensors"

        response = _mock_http_response([b"data"], status_code=200)
        response.headers = {
            "content-length": "4",
            "content-disposition": 'attachment; filename="server_name.safetensors"',
        }
        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=_make_async_cm(response))
        http_cm = _make_async_cm(mock_client)

        with (
            caplog.at_level(logging.DEBUG, logger="ai_content_service.downloader"),
            patch("ai_content_service.downloader.httpx.AsyncClient", return_value=http_cm),
        ):
            await downloader._download_file(
                file_cfg, path, progress, task_id=TaskID(0), client=mock_client
            )

        assert path.exists()
        assert path.name == "bundle_name.safetensors"
        messages = [(r.levelname, r.getMessage()) for r in caplog.records]
        assert not any(level == "WARNING" and "filename.mismatch" in msg for level, msg in messages)
        assert any(level == "DEBUG" and "filename.renamed" in msg for level, msg in messages)

    async def test_extension_change_logs_warning(
        self,
        tmp_path: Path,
        downloader: ModelDownloader,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """M6a: an extension change (.safetensors -> .zip) is the real D13 signal —
        a modelVersionId silently resolving to a different artefact — and must
        still warn."""
        progress = MagicMock()
        progress.add_task.return_value = 0
        file_cfg = _file_cfg("bundle_name.safetensors", "https://example.com/model.safetensors")
        path = tmp_path / "bundle_name.safetensors"

        response = _mock_http_response([b"data"], status_code=200)
        response.headers = {
            "content-length": "4",
            "content-disposition": 'attachment; filename="server_name.zip"',
        }
        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=_make_async_cm(response))
        http_cm = _make_async_cm(mock_client)

        with (
            caplog.at_level(logging.WARNING, logger="ai_content_service.downloader"),
            patch("ai_content_service.downloader.httpx.AsyncClient", return_value=http_cm),
        ):
            await downloader._download_file(
                file_cfg, path, progress, task_id=TaskID(0), client=mock_client
            )

        messages = [r.getMessage() for r in caplog.records]
        assert any("download.filename.mismatch" in m for m in messages)

    async def test_matching_filename_logs_no_warning(
        self,
        tmp_path: Path,
        downloader: ModelDownloader,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        progress = MagicMock()
        progress.add_task.return_value = 0
        file_cfg = _file_cfg("model.safetensors", "https://example.com/model.safetensors")
        path = tmp_path / "model.safetensors"

        response = _mock_http_response([b"data"], status_code=200)
        response.headers = {
            "content-length": "4",
            "content-disposition": 'attachment; filename="model.safetensors"',
        }
        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=_make_async_cm(response))
        http_cm = _make_async_cm(mock_client)

        with (
            caplog.at_level(logging.WARNING, logger="ai_content_service.downloader"),
            patch("ai_content_service.downloader.httpx.AsyncClient", return_value=http_cm),
        ):
            await downloader._download_file(
                file_cfg, path, progress, task_id=TaskID(0), client=mock_client
            )

        messages = [r.getMessage() for r in caplog.records]
        assert all("download.filename.mismatch" not in m for m in messages)


# ---------------------------------------------------------------------------
# M7a: one httpx.AsyncClient per download_all call, reused across files/retries
# ---------------------------------------------------------------------------


class TestHoistedClient:
    async def test_one_client_per_download_all_call_reused_across_files_and_retries(
        self, tmp_path: Path, downloader: ModelDownloader
    ) -> None:
        files = [
            _file_cfg("a.safetensors", "https://example.com/a"),
            _file_cfg("b.safetensors", "https://example.com/b"),
        ]
        model = _model_cfg("m", "diffusion_models", files)

        seen_clients: list[object] = []
        calls_for_a = 0

        async def fake_stream_to_part(
            file: ModelFileConfig,
            path: Path,
            _part_path: object,
            _progress: object,
            _task_id: object,
            _on_bytes: object,
            client: object = None,
        ) -> None:
            nonlocal calls_for_a
            seen_clients.append(client)
            if file.filename == "a.safetensors":
                calls_for_a += 1
                if calls_for_a == 1:
                    raise httpx.ConnectError("boom", request=MagicMock())
            path.write_bytes(b"ok")

        construct_count = 0

        def _counting_client(*args: object, **kwargs: object) -> httpx.AsyncClient:
            nonlocal construct_count
            construct_count += 1
            return _REAL_ASYNC_CLIENT(*args, **kwargs)  # type: ignore[arg-type]

        with (
            patch.object(downloader, "_stream_to_part", side_effect=fake_stream_to_part),
            patch("ai_content_service.downloader.httpx.AsyncClient", side_effect=_counting_client),
        ):
            result = await downloader.download_all([model], tmp_path)

        assert result.succeeded == 2
        assert construct_count == 1, "exactly one AsyncClient for the whole download_all call"
        assert all(c is not None for c in seen_clients)
        assert len({id(c) for c in seen_clients}) == 1, "same client instance reused everywhere"


# ---------------------------------------------------------------------------
# R3a: the egress guard is wired to the downloader's hoisted client
# ---------------------------------------------------------------------------


class TestEgressGuardWiredToClient:
    async def test_guard_fires_on_original_request_and_on_the_redirect_hop(
        self, tmp_path: Path, downloader: ModelDownloader
    ) -> None:
        """The event hook must fire for every hop, not just the first request --
        the redirect hop is the whole point (pitfall #1)."""
        file_cfg = _file_cfg("model.safetensors", "https://civitai.red/api/download/models/123")
        model = _model_cfg("m", "diffusion_models", [file_cfg])

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.host == "civitai.red":
                return httpx.Response(302, headers={"Location": "https://cdn.evil.example/blob"})
            return httpx.Response(200, content=b"data")

        seen_hosts: list[str | None] = []
        real_guard = downloader._guard_egress

        async def spy_guard(request: httpx.Request) -> None:
            seen_hosts.append(request.url.host)
            await real_guard(request)

        with (
            patch.object(downloader, "_guard_egress", side_effect=spy_guard),
            patch(
                "ai_content_service.downloader.httpx.AsyncClient",
                side_effect=_client_factory_with_transport(handler),
            ),
        ):
            result = await downloader.download_all([model], tmp_path)

        assert result.ok is True
        assert "civitai.red" in seen_hosts
        assert "cdn.evil.example" in seen_hosts


# ---------------------------------------------------------------------------
# MY-1a: client is a required parameter, not an opt-in one
# ---------------------------------------------------------------------------


class TestClientRequiredByType:
    """The unguarded default is gone: `_download_file`, `_download_http`, and
    `_stream_to_part` cannot be called without a client, and `_build_client`
    is the only place one is constructed."""

    async def test_download_file_requires_client(
        self, tmp_path: Path, downloader: ModelDownloader
    ) -> None:
        file_cfg = _file_cfg("model.safetensors", "https://example.com/model.safetensors")
        path = tmp_path / "model.safetensors"
        progress = MagicMock()

        with pytest.raises(TypeError):
            await downloader._download_file(  # type: ignore[call-arg]
                file_cfg, path, progress, task_id=TaskID(0)
            )

    async def test_download_http_requires_client(
        self, tmp_path: Path, downloader: ModelDownloader
    ) -> None:
        file_cfg = _file_cfg("model.safetensors", "https://example.com/model.safetensors")
        path = tmp_path / "model.safetensors"
        progress = MagicMock()

        with pytest.raises(TypeError):
            await downloader._download_http(  # type: ignore[call-arg]
                file_cfg, path, progress, task_id=TaskID(0), on_bytes=None
            )

    async def test_stream_to_part_requires_client(
        self, tmp_path: Path, downloader: ModelDownloader
    ) -> None:
        file_cfg = _file_cfg("model.safetensors", "https://example.com/model.safetensors")
        path = tmp_path / "model.safetensors"
        part_path = path.with_name(f"{path.name}.part")
        progress = MagicMock()

        with pytest.raises(TypeError):
            await downloader._stream_to_part(  # type: ignore[call-arg]
                file_cfg, path, part_path, progress, TaskID(0), None
            )

    async def test_build_client_carries_the_egress_hook(self, downloader: ModelDownloader) -> None:
        async with downloader._build_client() as client:
            assert downloader._guard_egress in client.event_hooks["request"]

    def test_exactly_one_async_client_construction_site_in_module(self) -> None:
        """Grep-equivalent: `httpx.AsyncClient(` must appear exactly once in
        the module -- in `_build_client` -- so the guard can never be
        bypassed by a second, unhooked construction path."""
        source_file = inspect.getsourcefile(ModelDownloader)
        assert source_file is not None
        source = Path(source_file).read_text()
        assert source.count("httpx.AsyncClient(") == 1


# ---------------------------------------------------------------------------
# MY-5a: value-based redaction -- a token we hold is masked under any name
# ---------------------------------------------------------------------------


class TestTokenRedactionIsValueBased:
    async def test_token_never_leaks_under_non_standard_param_name(
        self,
        tmp_path: Path,
        downloader: ModelDownloader,
        caplog: pytest.LogCaptureFixture,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Extends the `01r` D12 test: redaction used to key on a fixed list
        of param names (`token`, `api_key`, `access_token`). A token leaking
        into an error message under any other name (`key=`) must now also be
        masked, because MY-5a masks by value first."""
        file_cfg = _file_cfg("model.safetensors", "https://civitai.red/api/download/models/123")
        model = _model_cfg("m", "diffusion_models", [file_cfg])
        token = "test_civitai_token_123"

        async def raise_with_leaked_token(*_args: object, **_kwargs: object) -> None:
            raise DownloadError(
                f"upstream redirected with an unexpected credential: "
                f"https://cdn.example/blob?key={token}"
            )

        with (
            caplog.at_level(logging.DEBUG, logger="ai_content_service.downloader"),
            patch.object(downloader, "_download_file", side_effect=raise_with_leaked_token),
        ):
            report = await downloader.download_all([model], tmp_path)

        assert report.ok is False
        assert token not in report.failed[0].reason
        assert token not in report.failed[0].url
        for record in caplog.records:
            assert token not in record.getMessage()
        captured = capsys.readouterr()
        assert token not in captured.out
        assert token not in captured.err


# ---------------------------------------------------------------------------
# MY-3a: no closed httpx.Response escapes the streaming context
# ---------------------------------------------------------------------------


class _ClosingResponse:
    """A response stand-in that raises if `.status_code`/`.headers` are read
    after the streaming context exits -- proves the downloader captures a
    `_StreamOutcome` while the response is still open, never reading from a
    closed one afterward."""

    def __init__(self, status_code: int, headers: dict[str, str], chunks: list[bytes]) -> None:
        self._status_code = status_code
        self._headers = headers
        self._chunks = chunks
        self.closed = False

    @property
    def status_code(self) -> int:
        if self.closed:
            raise RuntimeError("status_code read after response closed")
        return self._status_code

    @property
    def headers(self) -> dict[str, str]:
        if self.closed:
            raise RuntimeError("headers read after response closed")
        return self._headers

    def raise_for_status(self) -> None:
        return None

    async def aiter_bytes(self, _chunk_size: int) -> object:
        for chunk in self._chunks:
            yield chunk


class TestStreamOutcomeNeverEscapesClosedResponse:
    async def test_download_succeeds_even_when_response_is_unreadable_after_close(
        self, tmp_path: Path, downloader: ModelDownloader
    ) -> None:
        response = _ClosingResponse(
            status_code=200,
            headers={
                "content-length": "4",
                "content-disposition": 'attachment; filename="model.safetensors"',
            },
            chunks=[b"data"],
        )

        async def _mark_closed(*_args: object) -> None:
            response.closed = True

        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=response)
        cm.__aexit__ = AsyncMock(side_effect=_mark_closed)

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=cm)

        file_cfg = _file_cfg("model.safetensors", "https://example.com/model.safetensors")
        path = tmp_path / "model.safetensors"
        progress = MagicMock()
        progress.add_task.return_value = 0

        await downloader._download_file(
            file_cfg, path, progress, task_id=TaskID(0), client=mock_client
        )

        assert path.read_bytes() == b"data"

    async def test_hf_redirect_to_foreign_cdn_does_not_false_positive(
        self, tmp_path: Path, downloader: ModelDownloader
    ) -> None:
        """Pitfall #5: HF may redirect to CloudFront or another foreign host.
        Authorization is already stripped by httpx by the time the guard sees
        it, so the guard must stay silent -- not reject a legitimate download."""
        file_cfg = _file_cfg("model.safetensors", "https://huggingface.co/model/download")
        model = _model_cfg("m", "diffusion_models", [file_cfg])

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.host == "huggingface.co":
                return httpx.Response(
                    302, headers={"Location": "https://d1.cloudfront.example/blob"}
                )
            return httpx.Response(200, content=b"weights")

        with patch(
            "ai_content_service.downloader.httpx.AsyncClient",
            side_effect=_client_factory_with_transport(handler),
        ):
            result = await downloader.download_all([model], tmp_path)

        assert result.ok is True
        assert (tmp_path / "diffusion_models" / "model.safetensors").read_bytes() == b"weights"
