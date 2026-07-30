"""Tests for model downloader including Civitai support."""

import hashlib
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
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


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
    chunks: list[bytes], content_length: str = "0", status_code: int = 200
) -> MagicMock:
    """Build a minimal httpx-response mock for streaming tests."""
    response = MagicMock()
    response.status_code = status_code
    response.headers = {"content-length": content_length}

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
    chunks: list[bytes], content_length: str = "0", status_code: int = 200
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
            await downloader._download_file(file_cfg, path, progress, task_id=TaskID(0))

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
            await downloader._download_file(file_cfg, path, progress, task_id=TaskID(0))

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
            await downloader_no_tokens._download_file(file_cfg, path, progress, task_id=TaskID(0))

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
            await downloader._download_file(file_cfg, path, progress, task_id=TaskID(0))

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
            await downloader._stream_to_part(file_cfg, path, part_path, progress, TaskID(0), None)

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
            await downloader._stream_to_part(file_cfg, path, part_path, progress, TaskID(0), None)

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
            await downloader._stream_to_part(file_cfg, path, part_path, progress, TaskID(0), None)

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
            await downloader._download_file(file_cfg, path, progress, task_id=TaskID(0))

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
            await downloader._download_file(file_cfg, path, progress, task_id=TaskID(0))

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

    async def test_writes_chunks_to_file(
        self, tmp_path: Path, downloader: ModelDownloader, progress: MagicMock
    ) -> None:
        chunks = [b"hello ", b"world"]
        file_cfg = _file_cfg("model.safetensors", "https://example.com/model.safetensors")
        path = tmp_path / "model.safetensors"

        http_p, _client, _resp = _patch_http(chunks)
        with http_p:
            await downloader._download_file(file_cfg, path, progress, task_id=TaskID(0))

        assert path.read_bytes() == b"hello world"

    async def test_progress_updated_per_chunk(
        self, tmp_path: Path, downloader: ModelDownloader, progress: MagicMock
    ) -> None:
        chunks = [b"aa", b"bbb"]
        file_cfg = _file_cfg("model.safetensors", "https://example.com/model.safetensors")
        path = tmp_path / "model.safetensors"

        http_p, _client, _resp = _patch_http(chunks)
        with http_p:
            await downloader._download_file(file_cfg, path, progress, task_id=TaskID(0))

        advance_calls = [c for c in progress.update.call_args_list if "advance" in c.kwargs]
        assert len(advance_calls) == 2
        assert advance_calls[0].kwargs["advance"] == 2
        assert advance_calls[1].kwargs["advance"] == 3

    async def test_content_length_sets_task_total(
        self, tmp_path: Path, downloader: ModelDownloader, progress: MagicMock
    ) -> None:
        file_cfg = _file_cfg("model.safetensors", "https://example.com/model.safetensors")
        path = tmp_path / "model.safetensors"

        http_p, _client, _resp = _patch_http([b"data"], content_length="4")
        with http_p:
            await downloader._download_file(file_cfg, path, progress, task_id=TaskID(0))

        total_calls = [c for c in progress.update.call_args_list if c.kwargs.get("total") == 4]
        assert total_calls, "progress.update(total=4) should have been called"

    async def test_on_bytes_called_per_chunk(
        self, tmp_path: Path, downloader: ModelDownloader, progress: MagicMock
    ) -> None:
        chunks = [b"aa", b"bbb"]
        file_cfg = _file_cfg("model.safetensors", "https://example.com/model.safetensors")
        path = tmp_path / "model.safetensors"
        on_bytes = AsyncMock()

        http_p, _client, _resp = _patch_http(chunks)
        with http_p:
            await downloader._download_file(
                file_cfg, path, progress, task_id=TaskID(0), on_bytes=on_bytes
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

        http_p, _client, _resp = _patch_http([content])
        with http_p:
            await dl._download_file(file_cfg, path, progress, task_id=TaskID(0))  # must not raise

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

        http_p, _client, _resp = _patch_http([b"any content"])
        with http_p:
            await dl._download_file(
                file_cfg, path, progress, task_id=TaskID(0)
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

        with patch("ai_content_service.downloader.httpx.AsyncClient") as mock_client:
            await dl._download_file(file_cfg, path, progress, task_id=TaskID(0))
            mock_client.assert_not_called()

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
            await dl._download_file(file_cfg, path, progress, task_id=TaskID(0), on_bytes=on_bytes)

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

        http_p, _client, _resp = _patch_http([b"new content"])
        with http_p:
            await dl._download_file(file_cfg, path, progress, task_id=TaskID(0))

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
            await downloader._download_file(file_cfg, path, progress, task_id=TaskID(0))

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
        settings = Settings(verify_checksums=True, skip_existing=False)
        dl = ModelDownloader(settings)

        http_p, _client, _resp = _patch_http([b"bad new content"])
        with http_p, pytest.raises(DownloadError, match="Checksum mismatch"):
            await dl._download_file(file_cfg, path, progress, task_id=TaskID(0))

        assert path.read_bytes() == previous_content
        assert not path.with_name(f"{path.name}.part").exists()


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
            await downloader._download_file(file_cfg, path, progress, task_id=TaskID(0))

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
            await downloader._download_file(file_cfg, path, progress, task_id=TaskID(0))

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
        http_p, _client, _resp = _patch_http([full_new_content], status_code=200)
        with http_p:
            await downloader._download_file(file_cfg, path, progress, task_id=TaskID(0))

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
        ) -> None:
            nonlocal calls
            calls += 1
            if calls < 3:
                raise httpx.ConnectError("boom", request=MagicMock())
            path.write_bytes(b"ok")

        with patch.object(downloader, "_stream_to_part", side_effect=flaky):
            await downloader._download_http(
                file_cfg, path, progress, task_id=TaskID(0), on_bytes=None
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
                file_cfg, path, progress, task_id=TaskID(0), on_bytes=None
            )

        assert calls == 1


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
            await dl._download_file(file_cfg, path, progress, task_id=TaskID(0))

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
            await dl._download_file(file_cfg, path, progress, task_id=TaskID(0))

        assert not path.exists()
        assert not tmp_path_r2.exists()


# ---------------------------------------------------------------------------
# download_all
# ---------------------------------------------------------------------------


class TestDownloadAll:
    """Tests for ModelDownloader.download_all."""

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

        assert list(tmp_path.iterdir()) == []

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
        ) -> None:
            captured_on_bytes.append(on_bytes)

        with patch.object(downloader, "_download_file", side_effect=capture_download):
            await downloader.download_all([model], tmp_path)

        assert captured_on_bytes == [None]


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
            await dl._download_file(upper_cfg, path, progress, task_id=TaskID(0))

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
            await downloader._download_file(file_cfg, path, progress, task_id=TaskID(0))

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

        http_p, _client, _resp = _patch_http([], status_code=416)
        with http_p, pytest.raises(httpx.HTTPStatusError):
            await downloader._stream_to_part(file_cfg, path, part_path, progress, TaskID(0), None)


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

        http_p, _client, _resp = _patch_http([second_half], status_code=206)
        with http_p:
            await downloader._stream_to_part(
                file_cfg, path, part_path, progress, TaskID(0), on_bytes
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
    async def test_mismatch_logs_warning_but_keeps_bundle_filename(
        self,
        tmp_path: Path,
        downloader: ModelDownloader,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
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
            caplog.at_level(logging.WARNING, logger="ai_content_service.downloader"),
            patch("ai_content_service.downloader.httpx.AsyncClient", return_value=http_cm),
        ):
            await downloader._download_file(file_cfg, path, progress, task_id=TaskID(0))

        assert path.exists()
        assert path.name == "bundle_name.safetensors"
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
            await downloader._download_file(file_cfg, path, progress, task_id=TaskID(0))

        messages = [r.getMessage() for r in caplog.records]
        assert not any("download.filename.mismatch" in m for m in messages)
