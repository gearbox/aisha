"""Tests for model downloader including Civitai support."""

import hashlib
from contextlib import AbstractContextManager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import parse_qs, urlparse

import httpx
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


class TestNetlocMatches:
    """Tests for ModelDownloader._netloc_matches (host allowlist matching)."""

    def test_civitai_url_detected(self) -> None:
        """Test that civitai.com URLs are detected."""
        assert ModelDownloader._netloc_matches("civitai.com", ModelDownloader.CIVITAI_DOMAINS)
        assert ModelDownloader._netloc_matches("www.civitai.com", ModelDownloader.CIVITAI_DOMAINS)

    def test_civitai_url_case_insensitive(self) -> None:
        """Test that detection is case insensitive."""
        assert ModelDownloader._netloc_matches("CIVITAI.COM", ModelDownloader.CIVITAI_DOMAINS)
        assert ModelDownloader._netloc_matches("CivitAI.com", ModelDownloader.CIVITAI_DOMAINS)

    def test_non_civitai_urls_not_detected(self) -> None:
        """Test that non-Civitai URLs are not detected."""
        assert not ModelDownloader._netloc_matches(
            "huggingface.co", ModelDownloader.CIVITAI_DOMAINS
        )
        assert not ModelDownloader._netloc_matches(
            "notcivitai.com", ModelDownloader.CIVITAI_DOMAINS
        )

    def test_hf_token_not_sent_to_lookalike_domain(self, downloader: ModelDownloader) -> None:
        """Substring lookalike domains must not match (token exfiltration guard)."""
        headers = downloader._get_auth_headers("https://huggingface.co.evil.com/x")
        assert headers == {}

    def test_hf_token_sent_to_subdomain(self, downloader: ModelDownloader) -> None:
        headers = downloader._get_auth_headers("https://cdn-lfs.huggingface.co/x")
        assert "Authorization" in headers

    def test_hf_token_host_with_port_and_userinfo(self, downloader: ModelDownloader) -> None:
        headers = downloader._get_auth_headers("https://foo@huggingface.co:443/x")
        assert "Authorization" in headers

        headers = downloader._get_auth_headers("https://huggingface.co@evil.com/x")
        assert "Authorization" not in headers

    def test_civitai_token_not_appended_to_lookalike(self, downloader: ModelDownloader) -> None:
        prepared = downloader._prepare_download_url("https://civitai.com.evil.com/x")
        assert "token" not in parse_qs(urlparse(prepared).query)


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

        assert result == 1

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
