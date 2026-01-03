"""Async model downloader with progress tracking, resumable downloads, and verification."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Protocol
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

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

logger = logging.getLogger(__name__)


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

    # Known Civitai domains
    CIVITAI_DOMAINS: ClassVar[set[str]] = {"civitai.com", "www.civitai.com"}

    def __init__(
        self,
        settings: Settings,
        reporter: DownloadReporter | None = None,
    ) -> None:
        self._settings = settings
        self._reporter = reporter
        self._semaphore = asyncio.Semaphore(settings.max_concurrent_downloads)

    def _get_auth_headers(self, url: str) -> dict[str, str]:
        """Get authentication headers based on URL domain."""
        parsed = urlparse(url)
        headers: dict[str, str] = {}

        # Hugging Face authentication via header
        if "huggingface.co" in parsed.netloc and self._settings.hf_token:
            headers["Authorization"] = f"Bearer {self._settings.hf_token}"

        return headers

    def _is_civitai_url(self, url: str) -> bool:
        """Check if URL is from Civitai."""
        parsed = urlparse(url)
        return parsed.netloc.lower() in self.CIVITAI_DOMAINS

    def _prepare_download_url(self, url: str) -> str:
        """Prepare URL for download, adding authentication if needed.

        For Civitai URLs, appends the API token as a query parameter.
        """
        if not self._is_civitai_url(url):
            return url

        # Check if we have a Civitai token
        if not self._settings.civitai_api_token:
            # Return URL as-is, download may fail or be limited
            return url

        # Parse URL and add token
        parsed = urlparse(url)
        query_params = parse_qs(parsed.query, keep_blank_values=True)

        # Add token (overwrite if exists)
        query_params["token"] = [self._settings.civitai_api_token]

        # Rebuild URL with token
        new_query = urlencode(query_params, doseq=True)
        new_parsed = parsed._replace(query=new_query)

        return urlunparse(new_parsed)

    @staticmethod
    def _parse_content_disposition(header: str | None) -> str | None:
        """Parse filename from Content-Disposition header.

        Handles both:
        - attachment; filename="model.safetensors"
        - attachment; filename*=UTF-8''model.safetensors
        """
        if not header:
            return None

        # Try filename*= first (RFC 5987 encoded)
        match = re.search(r"filename\*=(?:UTF-8''|utf-8'')(.+?)(?:;|$)", header, re.IGNORECASE)
        if match:
            from urllib.parse import unquote

            return unquote(match[1].strip())

        # Try regular filename=
        match = re.search(r'filename="?([^";]+)"?', header, re.IGNORECASE)
        return match[1].strip() if match else None

    async def _stream_download(
        self,
        client: httpx.AsyncClient,
        url: str,
    ) -> AsyncIterator[bytes]:
        """Stream file content with retry logic."""
        # Prepare URL (adds Civitai token if needed)
        prepared_url = self._prepare_download_url(url)
        # Get auth headers for this URL (e.g., HuggingFace token)
        headers = self._get_auth_headers(url)

        async with client.stream(
            "GET",
            prepared_url,
            headers=headers,
            follow_redirects=True,
            timeout=httpx.Timeout(self._settings.download_timeout, connect=30.0),
        ) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes(chunk_size=self._settings.download_chunk_size):
                yield chunk

    async def _get_download_info(
        self,
        client: httpx.AsyncClient,
        url: str,
    ) -> tuple[int | None, str | None]:
        """Get file size and real filename from HEAD request.

        Returns:
            Tuple of (file_size, real_filename) where real_filename is from
            Content-Disposition header if present.
        """
        prepared_url = self._prepare_download_url(url)
        headers = self._get_auth_headers(url)

        try:
            response = await client.head(
                prepared_url,
                headers=headers,
                follow_redirects=True,
            )
            response.raise_for_status()

            # Get file size
            content_length = response.headers.get("content-length")
            file_size = int(content_length) if content_length else None

            # Get real filename from Content-Disposition (important for Civitai)
            content_disposition = response.headers.get("content-disposition")
            real_filename = self._parse_content_disposition(content_disposition)

            return file_size, real_filename

        except httpx.HTTPError:
            return None, None

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
        temp_path = target_path.with_suffix(f"{target_path.suffix}.tmp")

        # Check if file exists and matches expected size
        if self._settings.skip_existing and target_path.exists():
            if not model_file.size_bytes:
                # No expected size, assume existing file is valid
                return DownloadResult(
                    filename=model_file.filename,
                    path=target_path,
                    success=True,
                    size_bytes=target_path.stat().st_size,
                )

            if target_path.stat().st_size == model_file.size_bytes:
                return DownloadResult(
                    filename=model_file.filename,
                    path=target_path,
                    success=True,
                    size_bytes=model_file.size_bytes,
                )
        # Get file info (size and real filename from Content-Disposition)
        url = str(model_file.url)
        file_size, real_filename = await self._get_download_info(client, url)
        total_size = model_file.size_bytes or file_size

        # Log if Content-Disposition filename differs (useful for debugging)
        if real_filename and real_filename != model_file.filename:
            # This is informational - we still use the configured filename
            logger.info(
                f"Content-Disposition filename differs: configured={model_file.filename}, actual={real_filename}"
            )

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
        async with (
            self._semaphore,
            httpx.AsyncClient(
                http2=True,
            ) as client,
        ):
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
