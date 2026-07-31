"""Model downloader for AI Content Service."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http import HTTPStatus
from pathlib import Path
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
from tenacity import AsyncRetrying, RetryCallState, retry_if_exception, stop_after_attempt
from tenacity.wait import wait_base, wait_exponential

from . import r2_transfer
from .config import unwrap_secret
from .content_disposition_utils import parse_content_disposition
from .download_auth import (
    AUTH_RETRY_STATUSES,
    AuthTransport,
    apply_auth,
    assert_no_credential_egress,
    attempt_with_auth,
    build_credentials,
    build_registry,
    redact_url,
    resolve_policy,
)
from .http_utils import parse_content_length, parse_content_range_total, parse_retry_after
from .r2_transfer import read_creds_from_settings

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from .config import ModelConfig, ModelFileConfig, Settings
    from .download_auth import BoundCredential, HostAuthPolicy
    from .r2_transfer import R2ReadCreds

console = Console()
log = structlog.get_logger()

_RETRYABLE_STATUS_FLOOR = 500


class DownloadError(Exception):
    """Raised when download fails."""


class _StalePartError(Exception):
    """A `.part` file is byte-complete and the server answered 416; discard and retry from zero."""

    def __init__(self, filename: str, offset: int) -> None:
        self.filename = filename
        self.offset = offset
        super().__init__(
            f"{filename}: discarded a byte-complete partial download "
            f"({offset} bytes) after HTTP 416"
        )


class _TruncatedTransferError(Exception):
    """The response ended before its advertised total length."""

    def __init__(self, filename: str, written: int, expected: int) -> None:
        self.filename = filename
        self.written = written
        self.expected = expected
        super().__init__(
            f"{filename}: transfer ended early — {written} of {expected} bytes "
            f"({expected - written} short)"
        )


class _ChecksumMismatchError(Exception):
    """The downloaded bytes did not match the bundle's expected checksum."""

    def __init__(self, filename: str, expected: str, actual: str) -> None:
        super().__init__(filename, expected, actual)
        self.filename = filename
        self.expected = expected
        self.actual = actual


def _is_retryable_transfer_error(exc: BaseException) -> bool:
    """Retry on transient transfer failures while keeping other 4xx terminal."""
    if isinstance(exc, (_StalePartError, _TruncatedTransferError, _ChecksumMismatchError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status == HTTPStatus.TOO_MANY_REQUESTS or status >= _RETRYABLE_STATUS_FLOOR
    return isinstance(exc, httpx.TransportError)


class _RetryWait(wait_base):
    """Use a bounded server-provided delay for 429, otherwise exponential backoff."""

    def __init__(self, max_retry_after_seconds: float) -> None:
        self._max_retry_after_seconds = max_retry_after_seconds
        self._fallback = wait_exponential(multiplier=1, max=30)

    def __call__(self, retry_state: RetryCallState) -> float:
        outcome = retry_state.outcome
        exception = outcome.exception() if outcome is not None and outcome.failed else None
        if isinstance(exception, httpx.HTTPStatusError):
            response = exception.response
            if response.status_code == HTTPStatus.TOO_MANY_REQUESTS:
                retry_after = parse_retry_after(
                    response.headers.get("retry-after"),
                    now=datetime.now(timezone.utc),
                    max_seconds=self._max_retry_after_seconds,
                )
                if retry_after is not None:
                    return retry_after
        return self._fallback(retry_state)


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
class _StreamOutcome:
    """Headers-only outcome of a single streaming attempt, captured inside the
    `stream()` context. MY-3a: a closed `httpx.Response` must never escape the
    context that produced it -- this is the value that does instead."""

    status_code: int
    headers: dict[str, str]


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
    except OSError:
        return 0


def _part_path(path: Path) -> Path:
    """Return the atomic-download partial path for *path*."""
    return path.with_name(f"{path.name}.part")


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
        self._max_attempts = settings.download_max_attempts
        self._max_retry_after_seconds = settings.download_max_retry_after_seconds
        self._verify_checksums = settings.verify_checksums
        self._skip_existing = settings.skip_existing
        self._semaphore = asyncio.Semaphore(self._max_concurrent)
        self._user_agent = settings.download_user_agent

        self._auth_registry: tuple[HostAuthPolicy, ...] = build_registry(settings)
        self._tokens: dict[str, str | None] = {
            "huggingface": unwrap_secret(settings.hf_token),
            "civitai": unwrap_secret(settings.civitai_api_token),
        }
        self._credentials: tuple[BoundCredential, ...] = build_credentials(
            self._auth_registry, self._tokens
        )
        self._secret_values: tuple[str, ...] = tuple(c.token for c in self._credentials)

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

        model_dirs: list[Path] = []
        for model in models:
            model_dir = models_base_path / model.target_subpath
            planned = [(model_dir / f.filename, f) for f in model.files]
            for file_path, file in planned:
                if base_resolved not in file_path.resolve().parents:
                    raise DownloadError(f"Refusing path outside models dir: {file.filename!r}")
            model_dirs.append(model_dir)
            tasks.extend((model, f, p) for p, f in planned)

        files_total = len(tasks)
        bytes_total_all = sum(f.size_bytes or 0 for _, f, _ in tasks)
        if bytes_total_all > 0:
            space_path = models_base_path
            while not space_path.exists() and space_path != space_path.parent:
                space_path = space_path.parent
            if space_path.exists():
                free = shutil.disk_usage(space_path).free
                required = 0
                skipped = 0
                n_pending = 0
                for _model, file, path in tasks:
                    declared = file.size_bytes or 0
                    if declared <= 0:
                        continue

                    on_disk: int | None = None
                    if self._skip_existing:
                        with contextlib.suppress(OSError):
                            on_disk = path.stat().st_size
                    if on_disk is not None and (
                        file.size_bytes is None or on_disk == file.size_bytes
                    ):
                        skipped += declared
                        continue

                    n_pending += 1
                    required += max(declared - _part_size(_part_path(path)), 0)

                log.debug(
                    "download.space.check",
                    declared=bytes_total_all,
                    required=required,
                    skipped=skipped,
                    free=free,
                )
                if required > 0 and free < required * 1.05:
                    raise DownloadError(
                        f"insufficient disk space at {models_base_path}: "
                        f"{free / 1e9:.1f} GB free, "
                        f"need ~{required * 1.05 / 1e9:.1f} GB for {n_pending} pending file(s) "
                        f"({skipped / 1e9:.1f} GB already present)"
                    )

        for model_dir in model_dirs:
            model_dir.mkdir(parents=True, exist_ok=True)  # mutate only after validation

        tracker = (
            _ProgressTracker(bytes_total_all, files_total, on_progress)
            if on_progress is not None
            else None
        )

        failures: list[FileFailure] = []

        async with self._build_client() as client:
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
                                client=client,
                            )
                            if tracker is not None:
                                await tracker.on_file_done()
                            progress.update(task_id, description=f"[green]✓ {file.filename}")
                            return True
                        except Exception as e:
                            progress.update(task_id, description=f"[red]✗ {file.filename}")
                            reason = redact_url(str(e), secrets=self._secret_values)
                            console.print(f"[red]Error downloading {file.filename}: {reason}[/red]")
                            failures.append(
                                FileFailure(
                                    file.filename,
                                    redact_url(file.url, secrets=self._secret_values),
                                    reason,
                                )
                            )
                            return False

                results = await asyncio.gather(
                    *[download_with_progress(m, f, p) for m, f, p in tasks]
                )
                return DownloadReport(succeeded=sum(results), failed=tuple(failures))

    def _build_client(self) -> httpx.AsyncClient:
        """The only place a download client is constructed. The egress hook is
        not optional."""
        return httpx.AsyncClient(
            follow_redirects=True,
            timeout=httpx.Timeout(connect=15.0, read=300.0, write=60.0, pool=60.0),
            event_hooks={"request": [self._guard_egress]},
        )

    async def _guard_egress(self, request: httpx.Request) -> None:
        """R3a event hook: fires on every request and every redirect hop."""
        assert_no_credential_egress(str(request.url), request.headers, self._credentials)

    async def _download_file(
        self,
        file: ModelFileConfig,
        path: Path,
        progress: Progress,
        task_id: TaskID,
        client: httpx.AsyncClient,
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
                        tmp_path.replace(path)
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

        await self._download_http(file, path, progress, task_id, on_bytes, client)

    async def _download_http(
        self,
        file: ModelFileConfig,
        path: Path,
        progress: Progress,
        task_id: TaskID,
        on_bytes: Callable[[int], Awaitable[None]] | None,
        client: httpx.AsyncClient,
    ) -> None:
        """Stream *file* to *path* atomically, retrying transient failures.

        Each attempt resumes from a previous ``.part`` file when the server
        honours ``Range``, so a retry after a transport error or 5xx does not
        re-download bytes already on disk.
        """
        part_path = _part_path(path)
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self._max_attempts),
                wait=_RetryWait(self._max_retry_after_seconds),
                retry=retry_if_exception(_is_retryable_transfer_error),
                reraise=True,
            ):
                with attempt:
                    await self._stream_to_part(
                        file, path, part_path, progress, task_id, on_bytes, client
                    )
        except _ChecksumMismatchError as exc:
            raise DownloadError(
                f"Checksum mismatch for {exc.filename}: checksum never matched after "
                f"{self._max_attempts} attempts (expected {exc.expected}, last {exc.actual})"
            ) from exc
        except _TruncatedTransferError as exc:
            raise DownloadError(
                f"{exc.filename}: transfer truncated on every one of "
                f"{self._max_attempts} attempts (last: {exc.written} of {exc.expected} bytes)"
            ) from exc
        except _StalePartError as exc:
            raise DownloadError(
                f"{exc.filename}: could not restart after discarding a stale partial "
                f"download ({exc.offset} bytes) across {self._max_attempts} attempts"
            ) from exc

    async def _stream_to_part(
        self,
        file: ModelFileConfig,
        path: Path,
        part_path: Path,
        progress: Progress,
        task_id: TaskID,
        on_bytes: Callable[[int], Awaitable[None]] | None,
        client: httpx.AsyncClient,
    ) -> None:
        """Perform a single streaming attempt into *part_path*, then rename atomically.

        Resolves the host's auth policy once and delegates the primary/fallback
        dance to `attempt_with_auth`, which never issues more than two attempts.
        """
        policy = resolve_policy(self._auth_registry, file.url)
        token = self._tokens.get(policy.name) if policy is not None else None

        offset = _part_size(part_path)
        hasher = hashlib.sha256() if (self._verify_checksums and file.sha256) else None

        base_headers = {"User-Agent": self._user_agent, "Accept-Encoding": "identity"}
        if offset > 0:
            base_headers = {**base_headers, "Range": f"bytes={offset}-"}

        async def _send(
            active_client: httpx.AsyncClient, url: str, headers: dict[str, str]
        ) -> _StreamOutcome:
            async with active_client.stream("GET", url, headers=headers) as response:
                if response.status_code in AUTH_RETRY_STATUSES:
                    return _StreamOutcome(response.status_code, dict(response.headers))

                resuming = offset > 0 and response.status_code == HTTPStatus.PARTIAL_CONTENT
                effective_offset = offset if resuming else 0
                if (
                    offset > 0
                    and response.status_code == HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE
                ):
                    part_path.unlink(missing_ok=True)
                    log.warning(
                        "download.part.stale_discarded", filename=file.filename, offset=offset
                    )
                    raise _StalePartError(file.filename, offset)
                # Raise for every non-auth status, including 429 on a request
                # carrying a Range header.  A response is resuming only after
                # it has been established as a successful 206.
                response.raise_for_status()

                content_type = response.headers.get("content-type", "")
                if content_type.split(";", 1)[0].strip().lower() == "text/html":
                    raise DownloadError(
                        f"{file.filename}: {urlparse(url).netloc} returned an HTML page "
                        f"(content-type={content_type!r}). The API token is likely missing, "
                        f"invalid, or the model version is not available on this domain. "
                        f"NSFW model versions are only reachable via civitai.red."
                    )

                if resuming:
                    if hasher is not None:
                        async with aiofiles.open(part_path, "rb") as existing:
                            while chunk := await existing.read(self.CHUNK_SIZE):
                                hasher.update(chunk)
                    progress.update(task_id, completed=effective_offset)

                content_length = parse_content_length(response.headers)
                content_range_total = parse_content_range_total(response.headers)
                expected_total: int | None
                if content_range_total is not None:
                    expected_total = content_range_total
                elif content_length is not None:
                    expected_total = content_length + effective_offset
                else:
                    expected_total = file.size_bytes
                if expected_total is not None:
                    progress.update(task_id, total=expected_total)

                written = effective_offset
                async with aiofiles.open(part_path, "ab" if resuming else "wb") as f:
                    async for chunk in response.aiter_bytes(self.CHUNK_SIZE):
                        await f.write(chunk)
                        if hasher:
                            hasher.update(chunk)
                        written += len(chunk)
                        progress.update(task_id, advance=len(chunk))
                        if on_bytes is not None:
                            await on_bytes(written)

                if expected_total is not None and written != expected_total:
                    part_path.unlink(missing_ok=True)
                    log.warning(
                        "download.truncated",
                        filename=file.filename,
                        written=written,
                        expected=expected_total,
                    )
                    raise _TruncatedTransferError(file.filename, written, expected_total)

                return _StreamOutcome(response.status_code, dict(response.headers))

        async def _attempt(
            active_client: httpx.AsyncClient,
        ) -> tuple[_StreamOutcome, AuthTransport]:
            return await attempt_with_auth(
                policy,
                token,
                file.url,
                base_headers,
                send=lambda u, h: _send(active_client, u, h),
                status_of=lambda r: r.status_code,
            )

        outcome, transport = await _attempt(client)

        if outcome.status_code in AUTH_RETRY_STATUSES:
            final_url, _ = (
                apply_auth(policy, transport, file.url, base_headers, token)
                if policy is not None
                else (file.url, base_headers)
            )
            fallback_hint = (
                " Enable ACS_CIVITAI_ALLOW_QUERY_TOKEN_FALLBACK=true if this host "
                "rejects header authentication."
                if policy is not None and policy.name == "civitai" and policy.fallback is None
                else ""
            )
            raise DownloadError(
                f"{file.filename}: authentication failed ({outcome.status_code}) "
                f"for {redact_url(final_url, secrets=self._secret_values)}.{fallback_hint}"
            )

        if file.sha256 and hasher:
            actual_hash = hasher.hexdigest()
            if actual_hash != file.sha256:
                part_path.unlink(missing_ok=True)
                raise _ChecksumMismatchError(file.filename, file.sha256, actual_hash)

        server_filename = parse_content_disposition(outcome.headers.get("content-disposition"))
        if server_filename is None:
            log.debug("download.filename.absent", expected=file.filename)
        elif Path(server_filename).suffix.lower() != Path(file.filename).suffix.lower():
            log.warning(
                "download.filename.mismatch", expected=file.filename, server=server_filename
            )
        else:
            log.debug("download.filename.renamed", expected=file.filename, server=server_filename)

        part_path.replace(path)

    async def _verify_checksum(self, path: Path, expected_sha256: str) -> bool:
        """Verify file checksum."""
        hasher = hashlib.sha256()

        async with aiofiles.open(path, "rb") as f:
            while chunk := await f.read(self.CHUNK_SIZE):
                hasher.update(chunk)

        return hasher.hexdigest() == expected_sha256
