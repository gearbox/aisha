"""Async model downloader with progress tracking, resumable downloads, and verification."""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import aiofiles
import httpx
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from ai_content_service.config import ModelFile, Settings


class DownloadError(Exception):
    """Base exception for download errors."""


class ChecksumMismatchError(DownloadError):
    """Raised when file checksum doesn't match expected value."""


class DownloadReporter(Protocol):
    """Protocol for download progress reporting."""

    def start_file(self, filename: str, total_size: int | None) -> TaskID: ...
    def update_progress(self, task_id: TaskID, advance: int) -> None: ...
    def complete_file(self, task_id: TaskID) -> None: ...
    def fail_file(self, task_id: TaskID, error: str) -> None: ...


@dataclass
class DownloadResult:
    """Result of a download operation."""

    filename: str
    path: Path
    success: bool
    error: str | None = None
    sha256: str | None = None
    size_bytes: int = 0


class RichDownloadReporter:
    """Rich-based progress reporter for downloads."""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()
        self.progress = Progress(
            TextColumn("[bold blue]{task.fields[filename]}", justify="right"),
            BarColumn(bar_width=40),
            "[progress.percentage]{task.percentage:>3.1f}%",
            "•",
            DownloadColumn(),
            "•",
            TransferSpeedColumn(),
            "•",
            TimeRemainingColumn(),
            console=self.console,
            transient=False,
        )
        self._started = False

    def __enter__(self) -> RichDownloadReporter:
        self.progress.start()
        self._started = True
        return self

    def __exit__(self, *args: object) -> None:
        if self._started:
            self.progress.stop()
            self._started = False

    def start_file(self, filename: str, total_size: int | None) -> TaskID:
        """Start tracking a new file download."""
        return self.progress.add_task(
            f"Downloading {filename}",
            total=total_size,
            filename=filename,
        )

    def update_progress(self, task_id: TaskID, advance: int) -> None:
        """Update download progress."""
        self.progress.update(task_id, advance=advance)

    def complete_file(self, task_id: TaskID) -> None:
        """Mark file download as complete."""
        self.progress.update(task_id, completed=self.progress.tasks[task_id].total)

    def fail_file(self, task_id: TaskID, error: str) -> None:
        """Mark file download as failed."""
        self.progress.console.print(f"[red]✗ Task: {task_id}. Failed: {error}[/red]")


class ModelDownloader:
    """Handles downloading model files with async I/O and progress tracking."""

    def __init__(
        self,
        settings: Settings,
        reporter: DownloadReporter | None = None,
    ) -> None:
        self._settings = settings
        self._reporter = reporter
        self._semaphore = asyncio.Semaphore(settings.max_concurrent_downloads)

    def _get_auth_headers(self) -> dict[str, str]:
        """Get authentication headers if HF token is configured."""
        if self._settings.hf_token:
            return {"Authorization": f"Bearer {self._settings.hf_token}"}
        return {}

    async def _get_file_size(
        self,
        client: httpx.AsyncClient,
        url: str,
    ) -> int | None:
        """Get file size from HEAD request."""
        try:
            response = await client.head(url, follow_redirects=True)
            response.raise_for_status()
            content_length = response.headers.get("content-length")
            return int(content_length) if content_length else None
        except httpx.HTTPError:
            return None

    async def _stream_download(
        self,
        client: httpx.AsyncClient,
        url: str,
    ) -> AsyncIterator[bytes]:
        """Stream file content with retry logic."""
        async with client.stream(
            "GET",
            url,
            follow_redirects=True,
            timeout=httpx.Timeout(self._settings.download_timeout, connect=30.0),
        ) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes(chunk_size=self._settings.download_chunk_size):
                yield chunk

    @retry(
        retry=retry_if_exception_type(httpx.HTTPError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=30),
        reraise=True,
    )
    async def _download_file(
        self,
        client: httpx.AsyncClient,
        model_file: ModelFile,
        target_dir: Path,
    ) -> DownloadResult:
        """Download a single file with progress tracking."""
        target_path = target_dir / model_file.filename
        temp_path = target_path.with_suffix(target_path.suffix + ".tmp")

        # Check if file exists and matches expected size
        if self._settings.skip_existing and target_path.exists():
            if model_file.size_bytes:
                if target_path.stat().st_size == model_file.size_bytes:
                    return DownloadResult(
                        filename=model_file.filename,
                        path=target_path,
                        success=True,
                        size_bytes=model_file.size_bytes,
                    )
            else:
                # No expected size, assume existing file is valid
                return DownloadResult(
                    filename=model_file.filename,
                    path=target_path,
                    success=True,
                    size_bytes=target_path.stat().st_size,
                )

        # Get file size for progress tracking
        url = str(model_file.url)
        total_size = model_file.size_bytes or await self._get_file_size(client, url)

        # Initialize progress tracking
        task_id: TaskID | None = None
        if self._reporter:
            task_id = self._reporter.start_file(model_file.filename, total_size)

        hasher = hashlib.sha256()
        bytes_downloaded = 0

        try:
            # Ensure target directory exists
            target_dir.mkdir(parents=True, exist_ok=True)

            # Stream download to temp file
            async with aiofiles.open(temp_path, "wb") as f:
                async for chunk in self._stream_download(client, url):
                    await f.write(chunk)
                    hasher.update(chunk)
                    bytes_downloaded += len(chunk)
                    if self._reporter and task_id is not None:
                        self._reporter.update_progress(task_id, len(chunk))

            # Verify checksum if provided
            computed_sha256 = hasher.hexdigest()
            if (
                self._settings.verify_checksums
                and model_file.sha256
                and computed_sha256.lower() != model_file.sha256.lower()
            ):
                temp_path.unlink(missing_ok=True)
                raise ChecksumMismatchError(
                    f"Checksum mismatch for {model_file.filename}: "
                    f"expected {model_file.sha256}, got {computed_sha256}"
                )

            # Atomic rename to final location
            temp_path.rename(target_path)

            if self._reporter and task_id is not None:
                self._reporter.complete_file(task_id)

            return DownloadResult(
                filename=model_file.filename,
                path=target_path,
                success=True,
                sha256=computed_sha256,
                size_bytes=bytes_downloaded,
            )

        except Exception as e:
            temp_path.unlink(missing_ok=True)
            if self._reporter and task_id is not None:
                self._reporter.fail_file(task_id, str(e))
            raise

    async def download_model_file(
        self,
        model_file: ModelFile,
        target_dir: Path,
    ) -> DownloadResult:
        """Download a single model file with concurrency limiting."""
        async with self._semaphore, httpx.AsyncClient(
            headers=self._get_auth_headers(),
            http2=True,
        ) as client:
            try:
                return await self._download_file(client, model_file, target_dir)
            except Exception as e:
                return DownloadResult(
                    filename=model_file.filename,
                    path=target_dir / model_file.filename,
                    success=False,
                    error=str(e),
                )

    async def download_model_files(
        self,
        files: list[ModelFile],
        target_dir: Path,
    ) -> list[DownloadResult]:
        """Download multiple model files concurrently."""
        tasks = [self.download_model_file(f, target_dir) for f in files]
        return await asyncio.gather(*tasks)
