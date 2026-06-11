"""Tests for model downloader including Civitai support."""

import hashlib
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rich.progress import TaskID

from ai_content_service.config import ModelConfig, ModelFileConfig, Settings
from ai_content_service.downloader import DownloadError, ModelDownloader

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


def _mock_http_response(chunks: list[bytes], content_length: str = "0") -> MagicMock:
    """Build a minimal httpx-response mock for streaming tests."""
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.headers = {"content-length": content_length}

    async def _aiter_bytes(chunk_size: int):  # noqa: ARG001
        for chunk in chunks:
            yield chunk

    response.aiter_bytes = _aiter_bytes
    return response


def _patch_http(chunks: list[bytes], content_length: str = "0"):
    """Return a patch context for httpx.AsyncClient + aiofiles.open."""
    response = _mock_http_response(chunks, content_length)
    mock_client = MagicMock()
    mock_client.stream.return_value = _make_async_cm(response)
    http_cm = _make_async_cm(mock_client)

    mock_file = AsyncMock()
    file_cm = _make_async_cm(mock_file)

    http_patch = patch("ai_content_service.downloader.httpx.AsyncClient", return_value=http_cm)
    file_patch = patch("ai_content_service.downloader.aiofiles.open", return_value=file_cm)
    return http_patch, file_patch, response, mock_file


@pytest.fixture
def settings() -> Settings:
    """Create settings for testing."""
    return Settings(
        civitai_api_token="test_civitai_token_123",
        hf_token="test_hf_token_456",
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


class TestCivitaiUrlDetection:
    """Tests for Civitai URL detection."""

    def test_civitai_url_detected(self, downloader: ModelDownloader) -> None:
        """Test that civitai.com URLs are detected."""
        assert downloader._is_civitai_url("https://civitai.com/api/download/models/123")
        assert downloader._is_civitai_url("https://www.civitai.com/api/download/models/456")

    def test_civitai_url_case_insensitive(self, downloader: ModelDownloader) -> None:
        """Test that detection is case insensitive."""
        assert downloader._is_civitai_url("https://CIVITAI.COM/api/download/models/123")
        assert downloader._is_civitai_url("https://CivitAI.com/api/download/models/123")

    def test_non_civitai_urls_not_detected(self, downloader: ModelDownloader) -> None:
        """Test that non-Civitai URLs are not detected."""
        assert not downloader._is_civitai_url("https://huggingface.co/model/download")
        assert not downloader._is_civitai_url("https://example.com/civitai.com/fake")
        assert not downloader._is_civitai_url("https://notcivitai.com/models/123")


class TestCivitaiUrlPreparation:
    """Tests for Civitai URL preparation with token."""

    def test_civitai_url_gets_token_appended(self, downloader: ModelDownloader) -> None:
        """Test that token is appended to Civitai URLs."""
        url = "https://civitai.com/api/download/models/123"
        prepared = downloader._prepare_download_url(url)

        assert "token=test_civitai_token_123" in prepared
        assert prepared.startswith("https://civitai.com/api/download/models/123")

    def test_civitai_url_with_existing_query_params(self, downloader: ModelDownloader) -> None:
        """Test that token is added to URLs with existing query params."""
        url = "https://civitai.com/api/download/models/123?type=Model"
        prepared = downloader._prepare_download_url(url)

        assert "token=test_civitai_token_123" in prepared
        assert "type=Model" in prepared

    def test_civitai_url_token_overwrites_existing(self, downloader: ModelDownloader) -> None:
        """Test that existing token is overwritten."""
        url = "https://civitai.com/api/download/models/123?token=old_token"
        prepared = downloader._prepare_download_url(url)

        assert "token=test_civitai_token_123" in prepared
        assert "old_token" not in prepared

    def test_civitai_url_without_token_unchanged(
        self, downloader_no_tokens: ModelDownloader
    ) -> None:
        """Test that Civitai URLs without token setting are unchanged."""
        url = "https://civitai.com/api/download/models/123"
        prepared = downloader_no_tokens._prepare_download_url(url)

        assert prepared == url
        assert "token=" not in prepared

    def test_non_civitai_url_unchanged(self, downloader: ModelDownloader) -> None:
        """Test that non-Civitai URLs are not modified."""
        url = "https://huggingface.co/model/download"
        prepared = downloader._prepare_download_url(url)

        assert prepared == url
        assert "token=" not in prepared


class TestAuthHeaders:
    """Tests for authentication headers."""

    def test_huggingface_gets_auth_header(self, downloader: ModelDownloader) -> None:
        """Test that HuggingFace URLs get Authorization header."""
        url = "https://huggingface.co/model/download"
        headers = downloader._get_auth_headers(url)

        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer test_hf_token_456"

    def test_civitai_no_auth_header(self, downloader: ModelDownloader) -> None:
        """Test that Civitai URLs don't get auth header (uses URL token)."""
        url = "https://civitai.com/api/download/models/123"
        headers = downloader._get_auth_headers(url)

        # Civitai uses URL token, not header
        assert "Authorization" not in headers

    def test_other_urls_no_auth_header(self, downloader: ModelDownloader) -> None:
        """Test that other URLs don't get auth headers."""
        url = "https://example.com/model.safetensors"
        headers = downloader._get_auth_headers(url)

        assert headers == {}

    def test_no_token_no_auth_header(self, downloader_no_tokens: ModelDownloader) -> None:
        """Test that no header is added when no token is configured."""
        url = "https://huggingface.co/model/download"
        headers = downloader_no_tokens._get_auth_headers(url)

        assert headers == {}


class TestContentDispositionParsing:
    """Tests for Content-Disposition header parsing."""

    def test_simple_filename(self) -> None:
        """Test parsing simple filename."""
        header = 'attachment; filename="model.safetensors"'
        result = ModelDownloader._parse_content_disposition(header)
        assert result == "model.safetensors"

    def test_filename_without_quotes(self) -> None:
        """Test parsing filename without quotes."""
        header = "attachment; filename=model.safetensors"
        result = ModelDownloader._parse_content_disposition(header)
        assert result == "model.safetensors"

    def test_utf8_encoded_filename(self) -> None:
        """Test parsing UTF-8 encoded filename."""
        header = "attachment; filename*=UTF-8''model%20name.safetensors"
        result = ModelDownloader._parse_content_disposition(header)
        assert result == "model name.safetensors"

    def test_utf8_lowercase(self) -> None:
        """Test parsing utf-8 lowercase encoding."""
        header = "attachment; filename*=utf-8''test.safetensors"
        result = ModelDownloader._parse_content_disposition(header)
        assert result == "test.safetensors"

    def test_none_header(self) -> None:
        """Test handling None header."""
        result = ModelDownloader._parse_content_disposition(None)
        assert result is None

    def test_empty_header(self) -> None:
        """Test handling empty header."""
        result = ModelDownloader._parse_content_disposition("")
        assert result is None

    def test_header_without_filename(self) -> None:
        """Test handling header without filename."""
        header = "attachment"
        result = ModelDownloader._parse_content_disposition(header)
        assert result is None

    def test_complex_civitai_header(self) -> None:
        """Test parsing realistic Civitai Content-Disposition header."""
        header = "attachment; filename=\"v1-5-pruned-emaonly.safetensors\"; filename*=UTF-8''v1-5-pruned-emaonly.safetensors"
        result = ModelDownloader._parse_content_disposition(header)
        # Should prefer filename*= (UTF-8 encoded)
        assert result == "v1-5-pruned-emaonly.safetensors"


class TestSettingsCivitaiToken:
    """Tests for Civitai token in settings."""

    def test_civitai_token_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that CIVITAI_API_TOKEN env var is read."""
        monkeypatch.setenv("ACS_CIVITAI_API_TOKEN", "env_token_xyz")
        settings = Settings()
        assert settings.civitai_api_token == "env_token_xyz"

    def test_civitai_token_default_none(self) -> None:
        """Test that Civitai token defaults to None."""
        settings = Settings()
        assert settings.civitai_api_token is None


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

        http_p, file_p, _resp, mock_file = _patch_http(chunks)
        with http_p, file_p:
            await downloader._download_file(file_cfg, path, progress, task_id=TaskID(0))

        assert mock_file.write.call_count == len(chunks)

    async def test_progress_updated_per_chunk(
        self, tmp_path: Path, downloader: ModelDownloader, progress: MagicMock
    ) -> None:
        chunks = [b"aa", b"bbb"]
        file_cfg = _file_cfg("model.safetensors", "https://example.com/model.safetensors")
        path = tmp_path / "model.safetensors"

        http_p, file_p, _resp, _file = _patch_http(chunks)
        with http_p, file_p:
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

        http_p, file_p, _resp, _file = _patch_http([b"data"], content_length="4")
        with http_p, file_p:
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

        http_p, file_p, _resp, _file = _patch_http(chunks)
        with http_p, file_p:
            await downloader._download_file(
                file_cfg, path, progress, task_id=TaskID(0), on_bytes=on_bytes
            )

        on_bytes.assert_any_call(2)
        on_bytes.assert_any_call(3)

    async def test_checksum_match_no_error(self, tmp_path: Path, progress: MagicMock) -> None:
        content = b"good content"
        sha256 = hashlib.sha256(content).hexdigest()
        file_cfg = _file_cfg(
            "model.safetensors", "https://example.com/model.safetensors", sha256=sha256
        )
        path = tmp_path / "model.safetensors"
        settings = Settings(verify_checksums=True)
        dl = ModelDownloader(settings)

        http_p, file_p, _resp, _file = _patch_http([content])
        with http_p, file_p:
            await dl._download_file(file_cfg, path, progress, task_id=TaskID(0))  # must not raise

    async def test_checksum_mismatch_raises_and_deletes_file(
        self, tmp_path: Path, progress: MagicMock
    ) -> None:
        sha256 = "0" * 64  # deliberately wrong
        file_cfg = _file_cfg(
            "model.safetensors", "https://example.com/model.safetensors", sha256=sha256
        )
        path = tmp_path / "model.safetensors"
        path.touch()  # must exist so unlink() succeeds after the mismatch
        # skip_existing=False so the existing file is re-downloaded (not skipped)
        settings = Settings(verify_checksums=True, skip_existing=False)
        dl = ModelDownloader(settings)

        http_p, file_p, _resp, _file = _patch_http([b"bad content"])
        with http_p, file_p, pytest.raises(DownloadError, match="Checksum mismatch"):
            await dl._download_file(file_cfg, path, progress, task_id=TaskID(0))

        assert not path.exists()

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

        http_p, file_p, _resp, _file = _patch_http([b"any content"])
        with http_p, file_p:
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

        http_p, file_p, _resp, mock_file = _patch_http([b"new content"])
        with http_p, file_p:
            await dl._download_file(file_cfg, path, progress, task_id=TaskID(0))

        assert mock_file.write.called


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

        assert result == 2

    async def test_empty_model_list_returns_zero(
        self, tmp_path: Path, downloader: ModelDownloader
    ) -> None:
        result = await downloader.download_all([], tmp_path)
        assert result == 0

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

        assert result == 1

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

        assert result == 3
