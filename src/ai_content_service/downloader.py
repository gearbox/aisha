"""Model downloader for AI Content Service."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import parse_qs, urlencode, urlparse

import aiofiles
import httpx
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TaskID,
    TextColumn,
    TransferSpeedColumn,
)

from . import r2_transfer
from .content_disposition_utils import parse_content_disposition as _parse_content_disposition
from .r2_transfer import R2ReadCreds

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

    from .config import ModelConfig, ModelFileConfig, Settings

console = Console()
log = logging.getLogger(__name__)


@dataclass
class _ProgressTracker:
    bytes_done: int
    bytes_total: int
    files_done: int
    files_total: int
    on_progress: Callable[[int, int, int, int], Awaitable[None]]

    async def on_bytes(self, n: int) -> None:
        self.bytes_done += n
        await self._emit()

    async def on_file_done(self) -> None:
        self.files_done += 1
        await self._emit()

    async def _emit(self) -> None:
        await self.on_progress(self.bytes_done, self.bytes_total, self.files_done, self.files_total)


class DownloadError(Exception):
    """Raised when download fails."""

    pass


class ModelDownloader:
    """Async model downloader with progress tracking and verification.

    Supports downloading from:
    - Hugging Face (with optional token for private/gated models)
    - Civitai (with API token)
    - Direct URLs
    """

    CHUNK_SIZE = 1024 * 1024  # 1MB chunks
    HF_DOMAINS = ("huggingface.co", "hf.co")
    CIVITAI_DOMAINS = ("civitai.com",)

    def __init__(self, settings: Settings) -> None:
        self._max_concurrent = settings.max_concurrent_downloads
        self._hf_token = settings.hf_token
        self._civitai_token = settings.civitai_api_token
        self._verify_checksums = settings.verify_checksums
        self._skip_existing = settings.skip_existing
        self._semaphore = asyncio.Semaphore(self._max_concurrent)

        # R2 cache — read path
        self._r2_enabled: bool = bool(
            settings.r2_s3_endpoint
            and settings.r2_readonly_access_key_id
            and settings.r2_readonly_secret_access_key
        )
        self._r2_creds: R2ReadCreds | None = (
            R2ReadCreds(
                access_key_id=settings.r2_readonly_access_key_id,  # type: ignore[arg-type]
                secret_access_key=settings.r2_readonly_secret_access_key.get_secret_value(),  # type: ignore[union-attr]
            )
            if self._r2_enabled
            else None
        )
        self._r2_bucket = settings.r2_model_cache_bucket
        self._r2_endpoint: str = settings.r2_s3_endpoint or ""
        self._rclone_path = settings.rclone_path
        self._rclone_multi_thread_streams = settings.rclone_multi_thread_streams
        self._rclone_max_transfer_seconds = settings.rclone_max_transfer_seconds

    async def download_all(
        self,
        models: list[ModelConfig],
        models_base_path: Path,
        on_progress: Callable[[int, int, int, int], Awaitable[None]] | None = None,
    ) -> int:
        """Download all models with concurrent limit.

        Args:
            models: Model groups to download.
            models_base_path: Root directory for model files.
            on_progress: Optional async callback ``(bytes_done, bytes_total,
                files_done, files_total)`` invoked after each chunk and file
                completion. Caller is responsible for throttling if needed.

        Returns:
            Number of files successfully downloaded.
        """
        tasks: list[tuple[ModelConfig, ModelFileConfig, Path]] = []

        for model in models:
            model_dir = models_base_path / model.target_subpath
            model_dir.mkdir(parents=True, exist_ok=True)

            for file in model.files:
                file_path = model_dir / file.filename
                resolved = file_path.resolve()
                if models_base_path.resolve() not in resolved.parents:
                    raise DownloadError(f"Refusing path outside models dir: {file.filename!r}")
                tasks.append((model, file, file_path))

        files_total = len(tasks)
        bytes_total_all = sum(f.size_bytes or 0 for _, f, _ in tasks)
        tracker = (
            _ProgressTracker(0, bytes_total_all, 0, files_total, on_progress)
            if on_progress is not None
            else None
        )

        downloaded = 0

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            console=console,
        ) as progress:

            async def download_with_progress(
                _model: ModelConfig,
                file: ModelFileConfig,
                path: Path,
            ) -> bool:
                async with self._semaphore:
                    task_id = progress.add_task(
                        f"[cyan]{file.filename}",
                        total=file.size_bytes or 0,
                    )
                    try:
                        await self._download_file(
                            file,
                            path,
                            progress,
                            task_id,
                            on_bytes=tracker.on_bytes if tracker is not None else None,
                        )
                        if tracker is not None:
                            await tracker.on_file_done()
                        progress.update(task_id, description=f"[green]✓ {file.filename}")
                        return True
                    except Exception as e:
                        progress.update(task_id, description=f"[red]✗ {file.filename}")
                        console.print(f"[red]Error downloading {file.filename}: {e}[/red]")
                        return False

            results = await asyncio.gather(*[download_with_progress(m, f, p) for m, f, p in tasks])
            downloaded = sum(results)

        return downloaded

    async def _download_file(
        self,
        file: ModelFileConfig,
        path: Path,
        progress: Progress,
        task_id: TaskID,
        on_bytes: Callable[[int], Awaitable[None]] | None = None,
    ) -> None:
        """Download a single file with progress tracking."""
        if (
            self._skip_existing
            and path.exists()
            and file.sha256
            and await self._verify_checksum(path, file.sha256)
        ):
            file_size = path.stat().st_size
            progress.update(task_id, completed=file_size)
            if on_bytes is not None:
                await on_bytes(file_size)
            return

        # Attempt R2 cache pull before falling back to upstream download.
        # Any failure (miss, corrupt, rclone error) degrades gracefully.
        if self._r2_enabled and self._r2_creds is not None:
            if file.sha256:
                key = f"models/by-sha256/{file.sha256}"
                try:
                    await asyncio.to_thread(
                        r2_transfer.pull,
                        key=key,
                        dest_path=path,
                        creds=self._r2_creds,
                        bucket=self._r2_bucket,
                        endpoint=self._r2_endpoint,
                        rclone_path=self._rclone_path,
                        multi_thread_streams=self._rclone_multi_thread_streams,
                        size_bytes=file.size_bytes,
                        max_timeout_s=self._rclone_max_transfer_seconds,
                    )
                    if await self._verify_checksum(path, file.sha256):
                        log.info("cache.pull.hit filename=%s", file.filename)
                        console.print(f"  [green]cache hit[/green]  {file.filename}")
                        file_size = path.stat().st_size
                        progress.update(task_id, completed=file_size)
                        if on_bytes is not None:
                            await on_bytes(file_size)
                        return
                    else:
                        log.warning("cache.pull.corrupt filename=%s", file.filename)
                        console.print(
                            f"  [yellow]cache corrupt[/yellow] {file.filename} — fetching upstream"
                        )
                        path.unlink(missing_ok=True)
                except Exception as exc:
                    log.warning("cache.pull.fallback filename=%s exc=%s", file.filename, exc)
                    console.print(
                        f"  [yellow]cache miss[/yellow] {file.filename} — fetching upstream"
                    )
                    path.unlink(missing_ok=True)

            else:
                log.debug("cache.skip.no_sha256 filename=%s", file.filename)
        url = self._prepare_download_url(file.url)
        headers = self._get_auth_headers(file.url)

        async with (
            httpx.AsyncClient(follow_redirects=True) as client,
            client.stream("GET", url, headers=headers, timeout=300.0) as response,
        ):
            response.raise_for_status()

            if total := int(response.headers.get("content-length", 0)):
                progress.update(task_id, total=total)

            hasher = hashlib.sha256() if (self._verify_checksums and file.sha256) else None

            async with aiofiles.open(path, "wb") as f:
                async for chunk in response.aiter_bytes(self.CHUNK_SIZE):
                    await f.write(chunk)
                    if hasher:
                        hasher.update(chunk)
                    progress.update(task_id, advance=len(chunk))
                    if on_bytes is not None:
                        await on_bytes(len(chunk))

            if file.sha256 and hasher:
                actual_hash = hasher.hexdigest()
                if actual_hash != file.sha256:
                    path.unlink()  # Remove corrupted file
                    raise DownloadError(
                        f"Checksum mismatch for {file.filename}: "
                        f"expected {file.sha256}, got {actual_hash}"
                    )

    @staticmethod
    def _netloc_matches(netloc: str, domains: tuple[str, ...]) -> bool:
        """Exact-or-subdomain host match; strips userinfo and port."""
        host = netloc.lower().rsplit("@", 1)[-1].split(":", 1)[0]
        return any(host == d or host.endswith(f".{d}") for d in domains)

    def _prepare_download_url(self, url: str) -> str:
        """Prepare URL with authentication tokens if needed."""
        parsed = urlparse(url)

        if self._civitai_token and self._netloc_matches(parsed.netloc, self.CIVITAI_DOMAINS):
            query = parse_qs(parsed.query)
            query["token"] = [self._civitai_token]
            new_query = urlencode(query, doseq=True)
            return parsed._replace(query=new_query).geturl()

        return url

    def _get_auth_headers(self, url: str) -> dict[str, str]:
        """Return auth headers for the given URL."""
        headers: dict[str, str] = {}
        parsed = urlparse(url)

        if self._netloc_matches(parsed.netloc, self.HF_DOMAINS) and self._hf_token:
            headers["Authorization"] = f"Bearer {self._hf_token}"

        return headers

    @staticmethod
    def _parse_content_disposition(header: str | None) -> str | None:
        return _parse_content_disposition(header)

    async def _verify_checksum(self, path: Path, expected_sha256: str) -> bool:
        """Verify file checksum."""
        hasher = hashlib.sha256()

        async with aiofiles.open(path, "rb") as f:
            while chunk := await f.read(self.CHUNK_SIZE):
                hasher.update(chunk)

        return hasher.hexdigest() == expected_sha256
