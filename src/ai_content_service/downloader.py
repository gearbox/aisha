"""Model downloader for AI Content Service."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import os
import shutil
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
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
from .cache_keys import cache_key_for_sha256
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
from .download_transport import (
    TransportFetchError,
    TransportRequest,
    TransportUnavailableError,
    select_transport,
)
from .hf_xet_transport import HfXetTransport
from .http_utils import parse_content_length, parse_content_range_total, parse_retry_after
from .r2_transfer import read_creds_from_settings

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from .config import ModelConfig, ModelFileConfig, Settings
    from .download_auth import BoundCredential, HostAuthPolicy
    from .download_transport import DownloadTransport
    from .r2_transfer import R2ReadCreds

console = Console()
log = structlog.get_logger()

_RETRYABLE_STATUS_FLOOR = 500
_CREDENTIAL_BOUND_EXTENSION = "aisha.credential_bound"


class DownloadError(Exception):
    """Raised when download fails."""


@dataclass(frozen=True, slots=True)
class DownloadProgress:
    """Progress classified by whether bytes were reused or materialized."""

    bytes_done: int
    bytes_total: int
    files_done: int
    files_total: int
    materialized_bytes_done: int
    reused_bytes_done: int
    expected_materialized_bytes: int | None
    reclassified_materialized_bytes: int = 0


ProgressSink = Callable[[DownloadProgress], Awaitable[None]]


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

    def __init__(self, filename: str, written: int, expected: int, part_preserved: bool) -> None:
        self.filename = filename
        self.written = written
        self.expected = expected
        self.part_preserved = part_preserved
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
                    now=datetime.now(UTC),
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
    on_progress: ProgressSink
    predicted_reused: set[str]
    declared_by_key: Mapping[str, int | None]
    _per_file: dict[str, int] = field(default_factory=dict)
    _files_done: int = 0
    _reused_keys: set[str] = field(default_factory=set)
    _reclassified_materialized_bytes: int = 0

    def __post_init__(self) -> None:
        self._reused_keys = set(self.predicted_reused)

    @property
    def bytes_done(self) -> int:
        return sum(self._per_file.values())

    @property
    def expected_materialized_bytes(self) -> int | None:
        """Declared bytes for files now known to require materialization."""
        pending = [
            self.declared_by_key.get(key)
            for key in self.declared_by_key
            if key not in self._reused_keys
        ]
        if any(declared is None for declared in pending):
            return None
        return sum(declared or 0 for declared in pending)

    async def set_file_bytes(self, key: str, absolute: int) -> None:
        self._per_file[key] = absolute
        await self._emit()

    async def on_file_classified(self, *, key: str, reused: bool) -> None:
        """Record the source truth immediately after the skip check.

        The stat-only prediction is useful for early planning but checksum
        verification decides whether a file is really reusable.  Resolving it
        before a transfer emits bytes keeps both numerator and denominator
        honest for the entire download.
        """
        was_reused = key in self._reused_keys
        if reused:
            self._reused_keys.add(key)
        else:
            self._reused_keys.discard(key)
            if was_reused:
                self._reclassified_materialized_bytes += self._per_file.get(key, 0)

    async def on_file_done(self, *, key: str, source: str) -> None:
        reused = source == "skip"
        was_reused = key in self._reused_keys
        await self.on_file_classified(key=key, reused=reused)
        if not reused and was_reused:
            log.debug("download.progress.reuse_misprediction", key=key, source=source)
        self._files_done += 1
        await self._emit()

    async def _emit(self) -> None:
        bytes_done = self.bytes_done
        materialized_bytes_done = sum(
            value for key, value in self._per_file.items() if key not in self._reused_keys
        )
        await self.on_progress(
            DownloadProgress(
                bytes_done=bytes_done,
                bytes_total=self.bytes_total,
                files_done=self._files_done,
                files_total=self.files_total,
                materialized_bytes_done=materialized_bytes_done,
                reused_bytes_done=bytes_done - materialized_bytes_done,
                expected_materialized_bytes=self.expected_materialized_bytes,
                reclassified_materialized_bytes=self._reclassified_materialized_bytes,
            )
        )
        self._reclassified_materialized_bytes = 0


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
    """Aggregate outcome of a ``download_all`` call.

    ``materialized_bytes`` is the final destination size of files acquired by
    non-``skip`` sources in this invocation.  It is deliberately *not* wire
    traffic: resumed/retried attempts can transfer more (or fewer, via a
    cache) bytes than the final materialized files.  ``reused_bytes`` is the
    final size of verified pre-existing files.  Unknown declared sizes remain
    visible through ``unknown_size_files`` rather than being treated as zero.
    """

    succeeded: int
    failed: tuple[FileFailure, ...]
    sources: Mapping[str, int] = field(default_factory=dict)
    """Per-file counts by winning source: "skip", a transport's name (e.g.
    "hf_xet"), "r2", or "httpx"."""
    declared_bytes: int = 0
    reused_bytes: int = 0
    materialized_bytes: int = 0
    unknown_size_files: int = 0

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


def _existing_file_size(path: Path) -> int | None:
    """Return an existing file's size, or ``None`` when it cannot be statted."""
    try:
        return path.stat().st_size
    except OSError:
        return None


def _will_skip_download(file: ModelFileConfig, path: Path, *, skip_existing: bool) -> bool:
    """Whether a file is expected to be skipped without transferring bytes.

    This is deliberately stat-only: the exact checksum check remains in
    ``_download_file``. It is shared by the early missing-URL guard and the
    disk-space estimate so those checks cannot disagree about what is pending.
    """
    if not skip_existing or not file.sha256:
        return False
    try:
        on_disk = path.stat().st_size
    except OSError:
        return False
    return file.size_bytes is None or on_disk == file.size_bytes


def _can_obtain_without_url(
    file: ModelFileConfig,
    path: Path,
    *,
    skip_existing: bool,
    r2_configured: bool,
) -> bool:
    """Whether a file can be obtained even with an empty ``url``.

    Two routes exist: it is already on disk and will be skipped, or it is in the
    R2 cache, which is content-addressed by the file digest and never consults
    the URL. Snapshot-authored bundles (``url: ''``) deploy to fresh nodes entirely
    through the second route, so the missing-URL guard must account for it.

    Stat-only and optimistic: no network call is made here. ``_download_file``
    raises if neither route actually delivers.
    """
    if _will_skip_download(file, path, skip_existing=skip_existing):
        return True
    return r2_configured and bool(file.sha256)


def build_transports(settings: Settings) -> tuple[DownloadTransport, ...]:
    """The single site that constructs `DownloadTransport` instances (C4).

    `ModelDownloader` receives transports; it never builds one itself, so
    every composition root (CLI deploy, `acs models fetch`, ...) must funnel
    through here rather than instantiating `HfXetTransport` directly.
    """
    transports: list[DownloadTransport] = []
    if settings.hf_xet_enabled:
        transports.append(HfXetTransport(settings))
    return tuple(transports)


class ModelDownloader:
    """Async model downloader with progress tracking and verification.

    Supports downloading from:
    - Hugging Face (with optional token for private/gated models)
    - Civitai (with API token; domains configurable via `Settings.civitai_domains`)
    - Direct URLs

    Sources are tried fastest-first: a registered `DownloadTransport` (e.g.
    `hf_xet`), then the R2 cache, then plain httpx (L3). The cache is a
    fallback, not the primary source -- see `_download_file`.
    """

    CHUNK_SIZE = 1024 * 1024  # 1MB chunks

    def __init__(
        self,
        settings: Settings,
        transports: Sequence[DownloadTransport] | None = None,
    ) -> None:
        self._transports: Sequence[DownloadTransport] = transports or ()
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
        on_progress: ProgressSink | None = None,
    ) -> DownloadReport:
        """Download all models with concurrent limit.

        Args:
            models: Model groups to download.
            models_base_path: Root directory for model files.
            on_progress: Optional callback invoked with classified progress after
                each chunk and file completion. Caller is responsible for
                throttling if needed.

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

        r2_configured = self._r2_creds is not None
        missing_urls = [
            f.filename
            for _m, f, p in tasks
            if not f.url
            and not _can_obtain_without_url(
                f,
                p,
                skip_existing=self._skip_existing,
                r2_configured=r2_configured,
            )
        ]
        if missing_urls:
            cache_note = (
                "Files already present with a matching checksum, or available in the R2 "
                "cache by sha256, are exempt."
                if r2_configured
                else "No R2 cache is configured, so a source URL is the only route."
            )
            raise DownloadError(
                f"{len(missing_urls)} model file(s) have no source URL and cannot be "
                f"downloaded: {', '.join(missing_urls)}. {cache_note} Snapshot writes "
                f"`url: ''` as a "
                f"placeholder — fill these in, then re-run `acs models check`."
            )

        skip_by_key = {
            str(path): _will_skip_download(file, path, skip_existing=self._skip_existing)
            for _model, file, path in tasks
        }
        predicted_reused = {key for key, will_skip in skip_by_key.items() if will_skip}
        declared_by_key = {str(path): file.size_bytes for _model, file, path in tasks}

        files_total = len(tasks)
        bytes_total_all = sum(f.size_bytes or 0 for _, f, _ in tasks)
        unknown_size_files = sum(f.size_bytes is None for _, f, _ in tasks)
        if bytes_total_all > 0:
            space_path = models_base_path
            while not space_path.exists() and space_path != space_path.parent:
                space_path = space_path.parent
            if space_path.exists():
                free = shutil.disk_usage(space_path).free
                required = 0
                skipped = 0
                n_pending = 0
                pending_existing = 0
                for _model, file, path in tasks:
                    declared = file.size_bytes or 0
                    will_skip = skip_by_key[str(path)]
                    if declared <= 0:
                        continue

                    if will_skip:
                        skipped += declared
                        continue

                    on_disk = _existing_file_size(path)
                    n_pending += 1
                    # Expose the destination bytes that remain allocated while
                    # this pending file is streamed into its sibling .part.
                    pending_existing += on_disk or 0
                    # A replacement streams into a sibling .part, while the
                    # existing destination remains allocated until the rename.
                    required += max(declared - _part_size(_part_path(path)), 0)

                log.debug(
                    "download.space.check",
                    declared=bytes_total_all,
                    required=required,
                    skipped=skipped,
                    pending_existing=pending_existing,
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
            _ProgressTracker(
                bytes_total_all,
                files_total,
                on_progress,
                predicted_reused,
                declared_by_key,
            )
            if on_progress is not None
            else None
        )

        failures: list[FileFailure] = []
        sources: dict[str, int] = {}
        reused_bytes = 0
        materialized_bytes = 0

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
                    nonlocal materialized_bytes, reused_bytes
                    async with self._semaphore:
                        task_id = progress.add_task(
                            f"[cyan]{file.filename}",
                            total=file.size_bytes or 0,
                        )
                        key = str(path)

                        async def _on_bytes(absolute: int) -> None:
                            if tracker is not None:
                                await tracker.set_file_bytes(key, absolute)

                        async def _on_reuse_resolved(reused: bool) -> None:
                            if tracker is not None:
                                await tracker.on_file_classified(key=key, reused=reused)

                        try:
                            source = await self._download_file(
                                file,
                                path,
                                progress,
                                task_id,
                                on_bytes=_on_bytes if tracker is not None else None,
                                on_reuse_resolved=_on_reuse_resolved
                                if tracker is not None
                                else None,
                                client=client,
                            )
                            if tracker is not None:
                                await tracker.on_file_done(key=key, source=source)
                            sources[source] = sources.get(source, 0) + 1
                            final_size = _existing_file_size(path)
                            if final_size is None:
                                log.warning(
                                    "download.result.destination_missing",
                                    filename=file.filename,
                                    source=source,
                                )
                                final_size = 0
                            if source == "skip":
                                reused_bytes += final_size
                            else:
                                materialized_bytes += final_size
                            log.info(
                                "download.source",
                                filename=file.filename,
                                source=source,
                                bytes=final_size,
                            )
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
                return DownloadReport(
                    succeeded=sum(results),
                    failed=tuple(failures),
                    sources=sources,
                    declared_bytes=bytes_total_all,
                    reused_bytes=reused_bytes,
                    materialized_bytes=materialized_bytes,
                    unknown_size_files=unknown_size_files,
                )

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
        assert_no_credential_egress(
            str(request.url),
            request.headers,
            self._credentials,
            credential_carried=bool(request.extensions.get(_CREDENTIAL_BOUND_EXTENSION)),
        )

    async def _download_file(
        self,
        file: ModelFileConfig,
        path: Path,
        progress: Progress,
        task_id: TaskID,
        client: httpx.AsyncClient,
        on_bytes: Callable[[int], Awaitable[None]] | None = None,
        on_reuse_resolved: Callable[[bool], Awaitable[None]] | None = None,
    ) -> str:
        """Download a single file, trying sources fastest-first (C4/L3).

        Candidates, in order: already-on-disk, a registered transport (e.g.
        `hf_xet`), the R2 cache, plain httpx. Every candidate but the last
        degrades to the next on failure (L6); only the last raises. Returns
        the name of the source that produced the file.
        """
        if (
            self._skip_existing
            and path.exists()
            and file.sha256
            and await self._verify_checksum(path, file.sha256)
        ):
            if on_reuse_resolved is not None:
                await on_reuse_resolved(True)
            file_size = path.stat().st_size
            progress.update(task_id, completed=file_size)
            if on_bytes is not None:
                await on_bytes(file_size)
            return "skip"

        if on_reuse_resolved is not None:
            await on_reuse_resolved(False)

        # A URL-less file (snapshot placeholder) can only ever be served from
        # the cache -- there is no URL to hand a transport or probe_digest.
        transport = select_transport(self._transports, file.url) if file.url else None

        upstream_digest: str | None = None
        drifted = False
        if transport is not None and file.sha256:
            upstream_digest = await transport.probe_digest(file.url)
            if upstream_digest is not None and upstream_digest != file.sha256:
                drifted = True
                log.warning(
                    "download.upstream_drift",
                    filename=file.filename,
                    declared=file.sha256,
                    upstream=upstream_digest,
                    url=redact_url(file.url, secrets=self._secret_values),
                )

        # Candidate 1: the registered transport (drift skips it -- L5).
        if transport is not None and not drifted:
            source = await self._try_transport(transport, file, path, progress, task_id, on_bytes)
            if source is not None:
                return source

        # Candidate 2: the R2 cache. Content-addressed, so it cannot serve the
        # wrong weight (L3/L4) -- the only source drift does not disqualify.
        if self._r2_creds is not None and file.sha256:
            source = await self._try_r2_cache(file, file.sha256, path, progress, task_id, on_bytes)
            if source is not None:
                return source
        elif self._r2_creds is not None:
            log.debug("cache.skip.no_sha256", filename=file.filename)

        if drifted:
            cache_note = (
                "and the R2 cache did not have a copy of the declared weight"
                if self._r2_creds is not None
                else "and no R2 cache is configured"
            )
            raise DownloadError(
                f"{file.filename}: upstream now advertises sha256={upstream_digest}, which "
                f"contradicts the bundle's declared sha256={file.sha256}, {cache_note}. The "
                f"bundle pins a weight that no longer exists at its source URL."
            )

        # Candidate 3: plain httpx. R2 was attempted above; a hit has already
        # returned. Only now is the URL genuinely required.
        if not file.url:
            raise DownloadError(
                f"{file.filename}: no source URL. The file on disk did not match its "
                f"recorded checksum and the R2 cache did not have it. Fill in the URL "
                "in bundle.yaml."
            )

        await self._download_http(file, path, progress, task_id, on_bytes, client)
        return "httpx"

    async def _try_transport(
        self,
        transport: DownloadTransport,
        file: ModelFileConfig,
        path: Path,
        progress: Progress,
        task_id: TaskID,
        on_bytes: Callable[[int], Awaitable[None]] | None,
    ) -> str | None:
        """Attempt candidate 1. A failure or a checksum mismatch degrades to
        the next candidate (L6) rather than failing the file."""
        request = TransportRequest(
            url=file.url,
            destination=path,
            expected_sha256=file.sha256,
            expected_size=file.size_bytes,
        )

        async def _on_progress(bytes_so_far: int, _expected_size: int) -> None:
            progress.update(task_id, completed=bytes_so_far)
            if on_bytes is not None:
                await on_bytes(bytes_so_far)

        try:
            result = await transport.fetch(request, _on_progress)
        except (TransportUnavailableError, TransportFetchError) as exc:
            log.warning(
                "download.transport.fallback",
                filename=file.filename,
                transport=transport.name,
                reason=str(exc),
            )
            return None

        if file.sha256 and not await self._verify_checksum(path, file.sha256):
            log.warning(
                "download.transport.checksum_mismatch",
                filename=file.filename,
                transport=transport.name,
            )
            with contextlib.suppress(FileNotFoundError):
                path.unlink()
            return None

        progress.update(
            task_id, completed=result.bytes_written, total=file.size_bytes or result.bytes_written
        )
        if on_bytes is not None:
            await on_bytes(result.bytes_written)
        console.print(f"  [green]{transport.name}[/green]  {file.filename}")
        return transport.name

    async def _try_r2_cache(
        self,
        file: ModelFileConfig,
        sha256: str,
        path: Path,
        progress: Progress,
        task_id: TaskID,
        on_bytes: Callable[[int], Awaitable[None]] | None,
    ) -> str | None:
        """Attempt candidate 2. Any failure -- miss, corrupt, rclone error --
        degrades to the next candidate."""
        r2_creds = self._r2_creds
        if r2_creds is None:
            return None
        tmp_path = path.with_name(f"{path.name}.r2tmp")
        try:
            await asyncio.to_thread(
                r2_transfer.pull,
                key=cache_key_for_sha256(sha256),
                dest_path=tmp_path,
                creds=r2_creds,
                bucket=self._r2_bucket,
                endpoint=self._r2_endpoint,
                rclone_path=self._rclone_path,
                multi_thread_streams=self._rclone_multi_thread_streams,
                size_bytes=file.size_bytes,
                max_timeout_s=self._rclone_max_transfer_seconds,
            )
            if await self._verify_checksum(tmp_path, sha256):
                tmp_path.replace(path)
                log.info("cache.pull.hit", filename=file.filename)
                console.print(f"  [green]cache hit[/green]  {file.filename}")
                file_size = path.stat().st_size
                progress.update(task_id, completed=file_size)
                if on_bytes is not None:
                    await on_bytes(file_size)
                return "r2"
            log.warning("cache.pull.corrupt", filename=file.filename)
            console.print(f"  [yellow]cache corrupt[/yellow] {file.filename} — fetching upstream")
        except Exception as exc:
            log.warning("cache.pull.fallback", filename=file.filename, error=str(exc))
            console.print(f"  [yellow]cache miss[/yellow] {file.filename} — fetching upstream")
        finally:
            with contextlib.suppress(FileNotFoundError):
                tmp_path.unlink()
        return None

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
            retry_mode = "resuming preserved partial files" if exc.part_preserved else "restarting"
            raise DownloadError(
                f"{exc.filename}: transfer truncated on every one of "
                f"{self._max_attempts} attempts (last: {exc.written} of {exc.expected} bytes; "
                f"{retry_mode})"
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
        hash_gate = bool(file.sha256) and self._verify_checksums
        hasher = hashlib.sha256() if hash_gate else None
        expected_total: int | None = None

        base_headers = {"User-Agent": self._user_agent, "Accept-Encoding": "identity"}
        if offset > 0:
            base_headers = {**base_headers, "Range": f"bytes={offset}-"}

        async def _send(
            active_client: httpx.AsyncClient, url: str, headers: dict[str, str]
        ) -> _StreamOutcome:
            nonlocal expected_total
            async with active_client.stream(
                "GET",
                url,
                headers=headers,
                extensions={_CREDENTIAL_BOUND_EXTENSION: token is not None},
            ) as response:
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
                        # A partial is safe to preserve across truncation retries
                        # only because its existing prefix is re-hashed here.
                        async with aiofiles.open(part_path, "rb") as existing:
                            while chunk := await existing.read(self.CHUNK_SIZE):
                                hasher.update(chunk)
                    progress.update(task_id, completed=effective_offset)

                content_length = parse_content_length(response.headers)
                content_range_total = parse_content_range_total(response.headers)
                if content_range_total is not None:
                    expected_total = content_range_total
                elif content_length is not None:
                    expected_total = content_length + effective_offset
                else:
                    expected_total = None

                if (
                    expected_total is not None
                    and file.size_bytes is not None
                    and expected_total != file.size_bytes
                ):
                    log.warning(
                        "download.size.declared_mismatch",
                        filename=file.filename,
                        declared=file.size_bytes,
                        server=expected_total,
                    )

                progress_total = expected_total if expected_total is not None else file.size_bytes
                if progress_total is not None:
                    progress.update(task_id, total=progress_total)

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
                    made_progress = written > effective_offset
                    if not made_progress:
                        # A preserved part is safe because resumed attempts
                        # re-hash its prefix above. Drop only stuck boundaries.
                        part_path.unlink(missing_ok=True)
                    log.warning(
                        "download.truncated",
                        filename=file.filename,
                        written=written,
                        expected=expected_total,
                        part_preserved=made_progress,
                    )
                    raise _TruncatedTransferError(
                        file.filename, written, expected_total, made_progress
                    )

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

        if not hash_gate and expected_total is None:
            log.warning(
                "download.unverified",
                filename=file.filename,
                reason="checksum verification disabled" if file.sha256 else "no sha256",
                host=urlparse(file.url).netloc,
            )

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
