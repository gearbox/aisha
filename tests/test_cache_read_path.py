"""Tests for R2 cache read-path integration in ModelDownloader."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rich.progress import TaskID

from ai_content_service.config import ModelFileConfig, Settings
from ai_content_service.downloader import ModelDownloader
from ai_content_service.r2_transfer import CachePullError

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _r2_settings(**overrides: object) -> Settings:
    """Return Settings with R2 cache fully configured."""
    base = {
        "r2_s3_endpoint": "https://account.r2.cloudflarestorage.com",
        "r2_readonly_access_key_id": "READKEY",
        "r2_readonly_secret_access_key": "readsecret",
        "r2_model_cache_bucket": "apex-model-cache",
    } | overrides
    return Settings(**base)  # type: ignore[arg-type]


def _file_cfg(filename: str, sha256: str | None = None) -> ModelFileConfig:
    return ModelFileConfig(
        name=filename,
        url=f"https://example.com/{filename}",
        filename=filename,
        sha256=sha256,
    )


def _make_async_cm(return_value: object) -> MagicMock:
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=return_value)
    cm.__aexit__ = AsyncMock(return_value=None)
    return cm


def _mock_http_response(chunks: list[bytes]) -> MagicMock:
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.headers = {"content-length": "0"}

    async def _aiter_bytes(chunk_size: int):  # noqa: ARG001
        for chunk in chunks:
            yield chunk

    response.aiter_bytes = _aiter_bytes
    return response


@pytest.fixture
def progress() -> MagicMock:
    p = MagicMock()
    p.add_task.return_value = TaskID(0)
    return p


# ---------------------------------------------------------------------------
# Cache hit — rclone pull succeeds + checksum matches
# ---------------------------------------------------------------------------


class TestCacheHit:
    async def test_hit_skips_upstream_fetch(self, tmp_path: Path, progress: MagicMock) -> None:
        content = b"model weights"
        sha256 = hashlib.sha256(content).hexdigest()
        dest = tmp_path / "model.safetensors"
        file_cfg = _file_cfg("model.safetensors", sha256=sha256)

        def _fake_pull(**kwargs: object) -> None:
            dest.write_bytes(content)

        settings = _r2_settings()
        dl = ModelDownloader(settings)

        with (
            patch("ai_content_service.downloader.r2_transfer.pull", side_effect=_fake_pull),
            patch("ai_content_service.downloader.httpx.AsyncClient") as mock_http,
        ):
            await dl._download_file(file_cfg, dest, progress, task_id=TaskID(0))

        mock_http.assert_not_called()

    async def test_hit_logs_cache_hit(
        self, tmp_path: Path, progress: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        content = b"model weights"
        sha256 = hashlib.sha256(content).hexdigest()
        dest = tmp_path / "model.safetensors"
        file_cfg = _file_cfg("model.safetensors", sha256=sha256)

        def _fake_pull(**kwargs: object) -> None:
            dest.write_bytes(content)

        settings = _r2_settings()
        dl = ModelDownloader(settings)

        with (
            patch("ai_content_service.downloader.r2_transfer.pull", side_effect=_fake_pull),
            patch("ai_content_service.downloader.httpx.AsyncClient"),
            caplog.at_level("INFO", logger="ai_content_service.downloader"),
        ):
            await dl._download_file(file_cfg, dest, progress, task_id=TaskID(0))

        assert any("cache.pull.hit" in r.message for r in caplog.records)

    async def test_hit_calls_on_bytes_with_file_size(
        self, tmp_path: Path, progress: MagicMock
    ) -> None:
        content = b"model weights data"
        sha256 = hashlib.sha256(content).hexdigest()
        dest = tmp_path / "model.safetensors"
        file_cfg = _file_cfg("model.safetensors", sha256=sha256)
        on_bytes = AsyncMock()

        def _fake_pull(**kwargs: object) -> None:
            dest.write_bytes(content)

        settings = _r2_settings()
        dl = ModelDownloader(settings)

        with (
            patch("ai_content_service.downloader.r2_transfer.pull", side_effect=_fake_pull),
            patch("ai_content_service.downloader.httpx.AsyncClient"),
        ):
            await dl._download_file(file_cfg, dest, progress, task_id=TaskID(0), on_bytes=on_bytes)

        on_bytes.assert_called_once_with(len(content))

    async def test_pull_called_with_correct_key(self, tmp_path: Path, progress: MagicMock) -> None:
        sha256 = "a" * 64
        dest = tmp_path / "model.safetensors"
        file_cfg = _file_cfg("model.safetensors", sha256=sha256)

        # verify_checksums=False so the upstream mock's bytes don't trigger path.unlink()
        settings = _r2_settings(verify_checksums=False)
        dl = ModelDownloader(settings)

        response = _mock_http_response([b"x"])
        mock_client = MagicMock()
        mock_client.stream.return_value = _make_async_cm(response)
        http_cm = _make_async_cm(mock_client)
        file_cm = _make_async_cm(AsyncMock())

        with (
            patch(
                "ai_content_service.downloader.r2_transfer.pull",
                side_effect=CachePullError("miss"),
            ) as mock_pull,
            patch("ai_content_service.downloader.httpx.AsyncClient", return_value=http_cm),
            patch("ai_content_service.downloader.aiofiles.open", return_value=file_cm),
        ):
            await dl._download_file(file_cfg, dest, progress, task_id=TaskID(0))

        assert mock_pull.call_args.kwargs["key"] == f"models/by-sha256/{sha256}"


# ---------------------------------------------------------------------------
# Corrupt pull — rclone succeeds but checksum mismatches
# ---------------------------------------------------------------------------


class TestCorruptPull:
    async def test_corrupt_deletes_file_and_falls_back_to_upstream(
        self, tmp_path: Path, progress: MagicMock
    ) -> None:
        sha256 = "corrupt_sha256" + "a" * 50
        dest = tmp_path / "model.safetensors"
        file_cfg = _file_cfg("model.safetensors", sha256=sha256)

        settings = _r2_settings(verify_checksums=False)
        dl = ModelDownloader(settings)

        mock_file = AsyncMock()
        response = _mock_http_response([b"upstream bytes"])
        mock_client = MagicMock()
        mock_client.stream.return_value = _make_async_cm(response)

        with (
            patch("ai_content_service.downloader.r2_transfer.pull"),
            patch.object(dl, "_verify_checksum", new=AsyncMock(return_value=False)),
            patch(
                "ai_content_service.downloader.httpx.AsyncClient",
                return_value=_make_async_cm(mock_client),
            ),
            patch(
                "ai_content_service.downloader.aiofiles.open",
                return_value=_make_async_cm(mock_file),
            ),
        ):
            await dl._download_file(file_cfg, dest, progress, task_id=TaskID(0))

        # Upstream was called after corrupt R2 pull
        assert mock_file.write.called

    async def test_corrupt_logs_warning(
        self, tmp_path: Path, progress: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        sha256 = "b" * 64
        dest = tmp_path / "model.safetensors"
        file_cfg = _file_cfg("model.safetensors", sha256=sha256)

        settings = _r2_settings(verify_checksums=False)
        dl = ModelDownloader(settings)

        mock_file = AsyncMock()
        response = _mock_http_response([b"x"])
        mock_client = MagicMock()
        mock_client.stream.return_value = _make_async_cm(response)

        with (
            patch("ai_content_service.downloader.r2_transfer.pull"),
            patch.object(dl, "_verify_checksum", new=AsyncMock(return_value=False)),
            patch(
                "ai_content_service.downloader.httpx.AsyncClient",
                return_value=_make_async_cm(mock_client),
            ),
            patch(
                "ai_content_service.downloader.aiofiles.open",
                return_value=_make_async_cm(mock_file),
            ),
            caplog.at_level("WARNING", logger="ai_content_service.downloader"),
        ):
            await dl._download_file(file_cfg, dest, progress, task_id=TaskID(0))

        assert any("cache.pull.corrupt" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Cache miss / CachePullError — falls back to upstream
# ---------------------------------------------------------------------------


class TestCacheMiss:
    async def test_miss_falls_back_to_upstream(self, tmp_path: Path, progress: MagicMock) -> None:
        sha256 = "c" * 64
        dest = tmp_path / "model.safetensors"
        file_cfg = _file_cfg("model.safetensors", sha256=sha256)

        # verify_checksums=False: skip upstream checksum so path.unlink() isn't called
        settings = _r2_settings(verify_checksums=False)
        dl = ModelDownloader(settings)

        mock_file = AsyncMock()
        response = _mock_http_response([b"upstream content"])
        mock_client = MagicMock()
        mock_client.stream.return_value = _make_async_cm(response)

        with (
            patch(
                "ai_content_service.downloader.r2_transfer.pull",
                side_effect=CachePullError("404"),
            ),
            patch(
                "ai_content_service.downloader.httpx.AsyncClient",
                return_value=_make_async_cm(mock_client),
            ),
            patch(
                "ai_content_service.downloader.aiofiles.open",
                return_value=_make_async_cm(mock_file),
            ),
        ):
            await dl._download_file(file_cfg, dest, progress, task_id=TaskID(0))

        assert mock_file.write.called

    async def test_miss_logs_fallback(
        self, tmp_path: Path, progress: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        sha256 = "d" * 64
        dest = tmp_path / "model.safetensors"
        file_cfg = _file_cfg("model.safetensors", sha256=sha256)

        # verify_checksums=False: skip upstream checksum so path.unlink() isn't called
        settings = _r2_settings(verify_checksums=False)
        dl = ModelDownloader(settings)

        response = _mock_http_response([b"x"])
        mock_client = MagicMock()
        mock_client.stream.return_value = _make_async_cm(response)

        with (
            patch(
                "ai_content_service.downloader.r2_transfer.pull",
                side_effect=CachePullError("miss"),
            ),
            patch(
                "ai_content_service.downloader.httpx.AsyncClient",
                return_value=_make_async_cm(mock_client),
            ),
            patch(
                "ai_content_service.downloader.aiofiles.open",
                return_value=_make_async_cm(AsyncMock()),
            ),
            caplog.at_level("DEBUG", logger="ai_content_service.downloader"),
        ):
            await dl._download_file(file_cfg, dest, progress, task_id=TaskID(0))

        assert any("cache.pull.fallback" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# No sha256 — cache skipped entirely
# ---------------------------------------------------------------------------


class TestNoSha256:
    async def test_cache_skipped_when_no_sha256(self, tmp_path: Path, progress: MagicMock) -> None:
        dest = tmp_path / "model.safetensors"
        file_cfg = _file_cfg("model.safetensors", sha256=None)

        settings = _r2_settings()
        dl = ModelDownloader(settings)

        response = _mock_http_response([b"data"])
        mock_client = MagicMock()
        mock_client.stream.return_value = _make_async_cm(response)

        with (
            patch("ai_content_service.downloader.r2_transfer.pull") as mock_pull,
            patch(
                "ai_content_service.downloader.httpx.AsyncClient",
                return_value=_make_async_cm(mock_client),
            ),
            patch(
                "ai_content_service.downloader.aiofiles.open",
                return_value=_make_async_cm(AsyncMock()),
            ),
        ):
            await dl._download_file(file_cfg, dest, progress, task_id=TaskID(0))

        mock_pull.assert_not_called()

    async def test_no_sha256_logs_skip(
        self, tmp_path: Path, progress: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        dest = tmp_path / "model.safetensors"
        file_cfg = _file_cfg("model.safetensors", sha256=None)

        settings = _r2_settings()
        dl = ModelDownloader(settings)

        response = _mock_http_response([b"x"])
        mock_client = MagicMock()
        mock_client.stream.return_value = _make_async_cm(response)

        with (
            patch("ai_content_service.downloader.r2_transfer.pull"),
            patch(
                "ai_content_service.downloader.httpx.AsyncClient",
                return_value=_make_async_cm(mock_client),
            ),
            patch(
                "ai_content_service.downloader.aiofiles.open",
                return_value=_make_async_cm(AsyncMock()),
            ),
            caplog.at_level("DEBUG", logger="ai_content_service.downloader"),
        ):
            await dl._download_file(file_cfg, dest, progress, task_id=TaskID(0))

        assert any("cache.skip.no_sha256" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# R2 disabled — pull never called
# ---------------------------------------------------------------------------


class TestR2Disabled:
    async def test_cache_not_attempted_when_r2_not_configured(
        self, tmp_path: Path, progress: MagicMock
    ) -> None:
        sha256 = "e" * 64
        dest = tmp_path / "model.safetensors"
        file_cfg = _file_cfg("model.safetensors", sha256=sha256)

        # No R2 settings configured; verify_checksums=False avoids path.unlink() on mock content
        settings = Settings(verify_checksums=False)
        dl = ModelDownloader(settings)

        response = _mock_http_response([b"data"])
        mock_client = MagicMock()
        mock_client.stream.return_value = _make_async_cm(response)

        with (
            patch("ai_content_service.downloader.r2_transfer.pull") as mock_pull,
            patch(
                "ai_content_service.downloader.httpx.AsyncClient",
                return_value=_make_async_cm(mock_client),
            ),
            patch(
                "ai_content_service.downloader.aiofiles.open",
                return_value=_make_async_cm(AsyncMock()),
            ),
        ):
            await dl._download_file(file_cfg, dest, progress, task_id=TaskID(0))

        mock_pull.assert_not_called()


# ---------------------------------------------------------------------------
# Deploy never fails on cache errors
# ---------------------------------------------------------------------------


class TestDeployNeverFailsOnCache:
    async def test_cache_error_does_not_propagate(
        self, tmp_path: Path, progress: MagicMock
    ) -> None:
        sha256 = "f" * 64
        dest = tmp_path / "model.safetensors"
        file_cfg = _file_cfg("model.safetensors", sha256=sha256)

        # verify_checksums=False avoids path.unlink() on mock upstream content
        settings = _r2_settings(verify_checksums=False)
        dl = ModelDownloader(settings)

        response = _mock_http_response([b"fallback"])
        mock_client = MagicMock()
        mock_client.stream.return_value = _make_async_cm(response)

        with (
            patch(
                "ai_content_service.downloader.r2_transfer.pull",
                side_effect=CachePullError("network error"),
            ),
            patch(
                "ai_content_service.downloader.httpx.AsyncClient",
                return_value=_make_async_cm(mock_client),
            ),
            patch(
                "ai_content_service.downloader.aiofiles.open",
                return_value=_make_async_cm(AsyncMock()),
            ),
        ):
            # Must not raise
            await dl._download_file(file_cfg, dest, progress, task_id=TaskID(0))

    async def test_rclone_missing_falls_back_to_upstream(
        self, tmp_path: Path, progress: MagicMock
    ) -> None:
        sha256 = "a" * 64
        dest = tmp_path / "model.safetensors"
        file_cfg = _file_cfg("model.safetensors", sha256=sha256)

        # verify_checksums=False avoids path.unlink() on mock upstream content
        settings = _r2_settings(verify_checksums=False)
        dl = ModelDownloader(settings)

        response = _mock_http_response([b"upstream"])
        mock_client = MagicMock()
        mock_client.stream.return_value = _make_async_cm(response)
        mock_file = AsyncMock()

        with (
            patch(
                "ai_content_service.downloader.r2_transfer.pull",
                side_effect=RuntimeError("rclone not found"),
            ),
            patch(
                "ai_content_service.downloader.httpx.AsyncClient",
                return_value=_make_async_cm(mock_client),
            ),
            patch(
                "ai_content_service.downloader.aiofiles.open",
                return_value=_make_async_cm(mock_file),
            ),
        ):
            # Must not raise — RuntimeError from missing rclone must degrade gracefully
            await dl._download_file(file_cfg, dest, progress, task_id=TaskID(0))

        assert mock_file.write.called
