"""Model downloader for AI Content Service."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import os
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import aiofiles
import httpx
import structlog
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TaskID,
    TextColumn,
    TransferSpeedColumn,
)
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential

from . import r2_transfer
from .config import unwrap_secret
from .content_disposition_utils import parse_content_disposition
from .download_auth import AuthTransport, apply_auth, build_registry, redact_url, resolve_policy
from .r2_transfer import read_creds_from_settings

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

    from .config import ModelConfig, ModelFileConfig, Settings
    from .download_auth import HostAuthPolicy
    from .r2_transfer import R2ReadCreds

console = Console()
log = structlog.get_logger()

_RETRYABLE_STATUS_FLOOR = 500
_AUTH_RETRY_STATUSES = (HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN)


class DownloadError(Exception):
    """Raised when download fails."""


class _StalePartError(Exception):
    """A `.part` file is byte-complete and the server answered 416; discard and retry from zero."""


def _is_retryable_transfer_error(exc: BaseException) -> bool:
    """Retry on transport failures, 5xx responses, and stale-`.part` 416s only; 4xx is not retried."""
    if isinstance(exc, _StalePartError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= _RETRYABLE_STATUS_FLOOR
    return isinstance(exc, httpx.TransportError)


@dataclass
class _ProgressTracker:
    """Absolute per-file byte tracking, so progress is idempotent across retries.

    A delta-based tracker cannot be idempotent: a failed attempt's already-emitted
    contribution is unknowable to a later retry. Tracking each file's *absolute*
    bytes-done and summing avoids double-counting on resume.
    """

    bytes_total: int
    files_total: int
    on_progress: Callable[[int, int, int, int], Awaitable[None]]
    _per_file: dict[str, int] = field(default_factory=dict)
    _files_done: int = 0

    @property
    def bytes_done(self) -> int:
        return sum(self._per_file.values())

    async def set_file_bytes(self, key: str, absolute: int) -> None:
        self._per_file[key] = absolute
        await self._emit()

    async def on_file_done(self) -> None:
        self._files_done += 1
        await self._emit()

    async def _emit(self) -> None:
        await self.on_progress(
            self.bytes_done, self.bytes_total, self._files_done, self.files_total
        )


@dataclass(frozen=True, slots=True)
class FileFailure:
    """A single model file that failed to download."""

    filename: str
    url: str  # already redacted by the caller
    reason: str


@dataclass(frozen=True, slots=True)
class DownloadReport:
    """Aggregate outcome of a `download_all` call."""

    succeeded: int
    failed: tuple[FileFailure, ...]

    @property
    def ok(self) -> bool:
        return not self.failed


def _part_size(part_path: Path) -> int:
    """Size of an existing partial download, or 0 if absent.

    Single stat syscall — avoids the exists()/stat() TOCTOU window.
    """
    try:
        # single syscall, no Path object churn
        return os.stat(part_path).st_size  # noqa: PTH116
    except FileNotFoundError:
        return 0


class ModelDownloader:
    """Async model downloader with progress tracking and verification.

    Supports downloading from:
    - Hugging Face (with optional token for private/gated models)
    - Civitai (with API token; domains configurable via `Settings.civitai_domains`)
    - Direct URLs
    """

    CHUNK_SIZE = 1024 * 1024  # 1MB chunks

    def __init__(self, settings: Settings) -> None:
        self._max_concurrent = settings.max_concurrent_downloads
        self._verify_checksums = settings.verify_checksums
        self._skip_existing = settings.skip_existing
        self._semaphore = asyncio.Semaphore(self._max_concurrent)
        self._user_agent = settings.download_user_agent

        self._auth_registry: tuple[HostAuthPolicy, ...] = build_registry(settings)
        self._tokens: dict[str, str | None] = {
            "huggingface": unwrap_secret(settings.hf_token),
            "civitai": unwrap_secret(settings.civitai_api_token),
        }

        # R2 cache — read path
        self._r2_creds: R2ReadCreds | None = read_creds_from_settings(settings)
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
    ) -> DownloadReport:
        """Download all models with concurrent limit.

        Args:
            models: Model groups to download.
            models_base_path: Root directory for model files.
            on_progress: Optional async callback ``(bytes_done, bytes_total,
                files_done, files_total)`` invoked after each chunk and file
                completion. Caller is responsible for throttling if needed.

        Returns:
            DownloadReport with the count of successes and the list of
            per-file failures (empty when everything succeeded).
        """
        tasks: list[tuple[ModelConfig, ModelFileConfig, Path]] = []

        base_resolved = models_base_path.resolve()

        for model in models:
            model_dir = models_base_path / model.target_subpath
            planned = [(model_dir / f.filename, f) for f in model.files]
            for file_path, file in planned:
                if base_resolved not in file_path.resolve().parents:
                    raise DownloadError(f"Refusing path outside models dir: {file.filename!r}")
            model_dir.mkdir(parents=True, exist_ok=True)  # mutate only after validation
            tasks.extend((model, f, p) for p, f in planned)

        files_total = len(tasks)
        bytes_total_all = sum(f.size_bytes or 0 for _, f, _ in tasks)
        tracker = (
            _ProgressTracker(bytes_total_all, files_total, on_progress)
            if on_progress is not None
            else None
        )

        failures: list[FileFailure] = []

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
                    key = str(path)

                    async def _on_bytes(absolute: int) -> None:
                        if tracker is not None:
                            await tracker.set_file_bytes(key, absolute)

                    try:
                        await self._download_file(
                            file,
                            path,
                            progress,
                            task_id,
                            on_bytes=_on_bytes if tracker is not None else None,
                        )
                        if tracker is not None:
                            await tracker.on_file_done()
                        progress.update(task_id, description=f"[green]✓ {file.filename}")
                        return True
                    except Exception as e:
                        progress.update(task_id, description=f"[red]✗ {file.filename}")
                        reason = redact_url(str(e))
                        console.print(f"[red]Error downloading {file.filename}: {reason}[/red]")
                        failures.append(FileFailure(file.filename, redact_url(file.url), reason))
                        return False

            results = await asyncio.gather(*[download_with_progress(m, f, p) for m, f, p in tasks])
            return DownloadReport(succeeded=sum(results), failed=tuple(failures))

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
        if self._r2_creds is not None:
            if file.sha256:
                tmp_path = path.with_name(f"{path.name}.r2tmp")
                try:
                    await asyncio.to_thread(
                        r2_transfer.pull,
                        key=f"models/by-sha256/{file.sha256}",
                        dest_path=tmp_path,
                        creds=self._r2_creds,
                        bucket=self._r2_bucket,
                        endpoint=self._r2_endpoint,
                        rclone_path=self._rclone_path,
                        multi_thread_streams=self._rclone_multi_thread_streams,
                        size_bytes=file.size_bytes,
                        max_timeout_s=self._rclone_max_transfer_seconds,
                    )
                    if await self._verify_checksum(tmp_path, file.sha256):
                        await asyncio.to_thread(tmp_path.replace, path)
                        log.info("cache.pull.hit", filename=file.filename)
                        console.print(f"  [green]cache hit[/green]  {file.filename}")
                        file_size = path.stat().st_size
                        progress.update(task_id, completed=file_size)
                        if on_bytes is not None:
                            await on_bytes(file_size)
                        return
                    log.warning("cache.pull.corrupt", filename=file.filename)
                    console.print(
                        f"  [yellow]cache corrupt[/yellow] {file.filename} — fetching upstream"
                    )
                except Exception as exc:
                    log.warning("cache.pull.fallback", filename=file.filename, error=str(exc))
                    console.print(
                        f"  [yellow]cache miss[/yellow] {file.filename} — fetching upstream"
                    )
                finally:
                    with contextlib.suppress(FileNotFoundError):
                        tmp_path.unlink()
            else:
                log.debug("cache.skip.no_sha256", filename=file.filename)

        await self._download_http(file, path, progress, task_id, on_bytes)

    async def _download_http(
        self,
        file: ModelFileConfig,
        path: Path,
        progress: Progress,
        task_id: TaskID,
        on_bytes: Callable[[int], Awaitable[None]] | None,
    ) -> None:
        """Stream *file* to *path* atomically, retrying transient failures.

        Each attempt resumes from a previous ``.part`` file when the server
        honours ``Range``, so a retry after a transport error or 5xx does not
        re-download bytes already on disk.
        """
        part_path = path.with_name(f"{path.name}.part")
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, max=30),
            retry=retry_if_exception(_is_retryable_transfer_error),
            reraise=True,
        ):
            with attempt:
                await self._stream_to_part(file, path, part_path, progress, task_id, on_bytes)

    async def _stream_to_part(
        self,
        file: ModelFileConfig,
        path: Path,
        part_path: Path,
        progress: Progress,
        task_id: TaskID,
        on_bytes: Callable[[int], Awaitable[None]] | None,
    ) -> None:
        """Perform a single streaming attempt into *part_path*, then rename atomically.

        Resolves the host's auth policy once. On 401/403 with a configured
        fallback transport, retries exactly once with that transport before
        giving up — this never loops more than twice per call.
        """
        policy = resolve_policy(self._auth_registry, file.url)
        token = self._tokens.get(policy.name) if policy is not None else None
        transport = policy.primary if policy is not None else AuthTransport.NONE

        offset = _part_size(part_path)
        hasher = hashlib.sha256() if (self._verify_checksums and file.sha256) else None
        resuming = False
        content_disposition: str | None = None

        while True:
            url, headers = (
                apply_auth(policy, transport, file.url, {}, token)
                if policy is not None
                else (file.url, {})
            )
            headers = {**headers, "User-Agent": self._user_agent}
            if offset > 0:
                headers = {**headers, "Range": f"bytes={offset}-"}

            async with (
                httpx.AsyncClient(follow_redirects=True) as client,
                client.stream("GET", url, headers=headers, timeout=300.0) as response,
            ):
                if (
                    response.status_code in _AUTH_RETRY_STATUSES
                    and policy is not None
                    and policy.fallback is not None
                    and transport != policy.fallback
                ):
                    host = urlparse(file.url).netloc
                    log.warning(
                        "civitai.auth.query_fallback", host=host, status=response.status_code
                    )
                    transport = policy.fallback
                    continue

                if response.status_code in _AUTH_RETRY_STATUSES:
                    raise DownloadError(
                        f"{file.filename}: authentication failed ({response.status_code}) "
                        f"for {redact_url(url)}"
                    )

                resuming = offset > 0 and response.status_code == HTTPStatus.PARTIAL_CONTENT
                if not resuming:
                    if (
                        offset > 0
                        and response.status_code == HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE
                    ):
                        part_path.unlink(missing_ok=True)
                        log.warning(
                            "download.part.stale_discarded", filename=file.filename, offset=offset
                        )
                        raise _StalePartError(file.filename)
                    offset = 0
                    response.raise_for_status()

                content_type = response.headers.get("content-type", "")
                if content_type.split(";", 1)[0].strip().lower() == "text/html":
                    raise DownloadError(
                        f"{file.filename}: {urlparse(url).netloc} returned an HTML page "
                        f"(content-type={content_type!r}). The API token is likely missing, "
                        f"invalid, or the model version is not available on this domain. "
                        f"NSFW model versions are only reachable via civitai.red."
                    )
                content_disposition = response.headers.get("content-disposition")

                if resuming:
                    if hasher is not None:
                        async with aiofiles.open(part_path, "rb") as existing:
                            while chunk := await existing.read(self.CHUNK_SIZE):
                                hasher.update(chunk)
                    progress.update(task_id, completed=offset)

                if content_length := int(response.headers.get("content-length", 0)):
                    progress.update(task_id, total=content_length + offset)

                written = offset
                async with aiofiles.open(part_path, "ab" if resuming else "wb") as f:
                    async for chunk in response.aiter_bytes(self.CHUNK_SIZE):
                        await f.write(chunk)
                        if hasher:
                            hasher.update(chunk)
                        written += len(chunk)
                        progress.update(task_id, advance=len(chunk))
                        if on_bytes is not None:
                            await on_bytes(written)
                break

        if file.sha256 and hasher:
            actual_hash = hasher.hexdigest()
            if actual_hash != file.sha256:
                part_path.unlink()  # Remove corrupted partial file
                raise DownloadError(
                    f"Checksum mismatch for {file.filename}: "
                    f"expected {file.sha256}, got {actual_hash}"
                )

        server_filename = parse_content_disposition(content_disposition)
        if server_filename is not None and server_filename != file.filename:
            log.warning(
                "download.filename.mismatch", expected=file.filename, server=server_filename
            )

        await asyncio.to_thread(part_path.replace, path)

    async def _verify_checksum(self, path: Path, expected_sha256: str) -> bool:
        """Verify file checksum."""
        hasher = hashlib.sha256()

        async with aiofiles.open(path, "rb") as f:
            while chunk := await f.read(self.CHUNK_SIZE):
                hasher.update(chunk)

        return hasher.hexdigest() == expected_sha256
