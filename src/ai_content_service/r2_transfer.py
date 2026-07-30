"""rclone-based R2 transfer wrapper for model weight cache."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from pathlib import Path

    from .config import Settings

log = structlog.get_logger()

_MIN_TRANSFER_TIMEOUT_S = 600
_FLOOR_THROUGHPUT_BYTES_S = 10 * 1024 * 1024  # 10 MiB/s


def compute_transfer_timeout(size_bytes: int | None, max_timeout_s: int) -> int:
    """Wall-clock timeout for a single rclone transfer.

    Derived as size_bytes at a 10 MiB/s throughput floor, with a 600 s
    minimum for known sizes; unknown/invalid sizes get max_timeout_s.
    max_timeout_s is an operator ceiling and always takes precedence —
    a value below 600 s deliberately lowers the effective minimum.
    """
    if size_bytes is None or size_bytes <= 0:
        return max_timeout_s
    return min(max(_MIN_TRANSFER_TIMEOUT_S, size_bytes // _FLOOR_THROUGHPUT_BYTES_S), max_timeout_s)


class R2TransferError(Exception):
    """Base for rclone-backed R2 transfer failures."""


class CachePullError(R2TransferError):
    """Raised when rclone pull fails (cache miss, network error, etc.)."""


class CachePushError(R2TransferError):
    """Raised when rclone push fails."""


@dataclass
class R2ReadCreds:
    """Read-only R2 credentials baked into the Vast.ai template env."""

    access_key_id: str
    secret_access_key: str


@dataclass
class R2WriteCreds:
    """Short-TTL write credentials minted by Apex per-push."""

    access_key_id: str
    secret_access_key: str
    session_token: str | None = None


def read_creds_from_settings(settings: Settings) -> R2ReadCreds | None:
    """Typed read-only creds if fully configured, else None (cache disabled)."""
    if (
        settings.r2_s3_endpoint
        and settings.r2_readonly_access_key_id
        and settings.r2_readonly_secret_access_key
    ):
        return R2ReadCreds(
            access_key_id=settings.r2_readonly_access_key_id,
            secret_access_key=settings.r2_readonly_secret_access_key.get_secret_value(),
        )
    return None


def _require_rclone(rclone_path: str) -> str:
    """Return resolved rclone path or raise with a clear message."""
    resolved = shutil.which(rclone_path)
    if resolved is None:
        raise R2TransferError(
            f"rclone not found at {rclone_path!r}. "
            "Install rclone and ensure it is on PATH, or set ACS_RCLONE_PATH."
        )
    return resolved


def pull(
    *,
    key: str,
    dest_path: Path,
    creds: R2ReadCreds,
    bucket: str,
    endpoint: str,
    rclone_path: str = "rclone",
    multi_thread_streams: int = 4,
    size_bytes: int | None = None,
    max_timeout_s: int = 3600,
) -> None:
    """Pull a model from R2 to dest_path via rclone copyto.

    Uses copyto (not copy) so the object lands at exactly dest_path regardless
    of key shape.  Raises CachePullError on any non-zero rclone exit or if the
    wall-clock timeout (derived from size_bytes, capped at max_timeout_s) expires.
    """
    rclone = _require_rclone(rclone_path)
    remote = f":s3:{bucket}/{key}"

    cmd = [
        rclone,
        "copyto",
        "--s3-provider",
        "Cloudflare",
        f"--s3-endpoint={endpoint}",
        "--s3-no-check-bucket",
        f"--multi-thread-streams={multi_thread_streams}",
        "--multi-thread-cutoff=250M",
        "--contimeout=30s",
        "--timeout=300s",
        "--retries=3",
        "--low-level-retries=10",
        "--",
        remote,
        str(dest_path),
    ]

    env = os.environ.copy()
    env["RCLONE_S3_ACCESS_KEY_ID"] = creds.access_key_id
    env["RCLONE_S3_SECRET_ACCESS_KEY"] = creds.secret_access_key

    timeout = compute_transfer_timeout(size_bytes, max_timeout_s)
    try:
        # S603: shell=False; rclone path is shutil.which-resolved. remote and
        # dest_path are the only dynamic elements, and they are placed after `--`
        # (with all flags before it), so a leading-dash dest_path is taken as a
        # literal positional argument rather than parsed as a flag. Verified
        # against the pinned rclone version — a trailing `--` alone does not work,
        # since cobra/pflag has already consumed any earlier dash-prefixed args
        # as flags by the time it reaches it.
        result = subprocess.run(cmd, capture_output=True, env=env, timeout=timeout)  # noqa: S603
    except subprocess.TimeoutExpired as exc:
        raise CachePullError(f"rclone pull timed out after {timeout}s for key {key!r}") from exc
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace")
        log.debug("rclone.pull.failed", exit_code=result.returncode, key=key, stderr=stderr)
        raise CachePullError(
            f"rclone pull failed (exit {result.returncode}) for key {key!r}: {stderr}"
        )


def push(
    *,
    src_path: Path,
    key: str,
    creds: R2WriteCreds,
    bucket: str,
    endpoint: str,
    rclone_path: str = "rclone",
    upload_concurrency: int = 8,
    chunk_size_mb: int = 128,
    size_bytes: int | None = None,
    max_timeout_s: int = 3600,
) -> None:
    """Push src_path to R2 via rclone copyto with write credentials.

    Raises CachePushError on any non-zero rclone exit or if the wall-clock
    timeout (derived from size_bytes, capped at max_timeout_s) expires.
    """
    rclone = _require_rclone(rclone_path)
    remote = f":s3:{bucket}/{key}"

    cmd = [
        rclone,
        "copyto",
        "--s3-provider",
        "Cloudflare",
        f"--s3-endpoint={endpoint}",
        "--s3-no-check-bucket",
        f"--s3-upload-concurrency={upload_concurrency}",
        f"--s3-chunk-size={chunk_size_mb}M",
        "--contimeout=30s",
        "--timeout=300s",
        "--retries=3",
        "--low-level-retries=10",
        "--progress",
        "--",
        str(src_path),
        remote,
    ]

    env = os.environ.copy()
    env["RCLONE_S3_ACCESS_KEY_ID"] = creds.access_key_id
    env["RCLONE_S3_SECRET_ACCESS_KEY"] = creds.secret_access_key
    if creds.session_token:
        env["RCLONE_S3_SESSION_TOKEN"] = creds.session_token

    timeout = compute_transfer_timeout(size_bytes, max_timeout_s)
    try:
        # S603: shell=False; rclone path is shutil.which-resolved. src_path and
        # remote are the only dynamic elements, and they are placed after `--`
        # (with all flags before it), so a leading-dash src_path is taken as a
        # literal positional argument rather than parsed as a flag. Verified
        # against the pinned rclone version — a trailing `--` alone does not work,
        # since cobra/pflag has already consumed any earlier dash-prefixed args
        # as flags by the time it reaches it.
        result = subprocess.run(cmd, env=env, timeout=timeout)  # noqa: S603
    except subprocess.TimeoutExpired as exc:
        raise CachePushError(f"rclone push timed out after {timeout}s for key {key!r}") from exc
    if result.returncode != 0:
        raise CachePushError(f"rclone push failed (exit {result.returncode}) for key {key!r}")
