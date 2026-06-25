"""rclone-based R2 transfer wrapper for model weight cache."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

log = logging.getLogger(__name__)


class CachePullError(Exception):
    """Raised when rclone pull fails (cache miss, network error, etc.)."""


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


def _require_rclone(rclone_path: str) -> str:
    """Return resolved rclone path or raise with a clear message."""
    resolved = shutil.which(rclone_path)
    if resolved is None:
        raise RuntimeError(
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
) -> None:
    """Pull a model from R2 to dest_path via rclone copyto.

    Uses copyto (not copy) so the object lands at exactly dest_path regardless
    of key shape.  Raises CachePullError on any non-zero rclone exit.
    """
    rclone = _require_rclone(rclone_path)
    remote = f":s3:{bucket}/{key}"

    cmd = [
        rclone,
        "copyto",
        remote,
        str(dest_path),
        "--s3-provider",
        "Cloudflare",
        f"--s3-endpoint={endpoint}",
        f"--multi-thread-streams={multi_thread_streams}",
        "--multi-thread-cutoff=250M",
        "--contimeout=30s",
        "--timeout=300s",
        "--retries=3",
        "--low-level-retries=10",
    ]

    env = os.environ.copy()
    env["RCLONE_S3_ACCESS_KEY_ID"] = creds.access_key_id
    env["RCLONE_S3_SECRET_ACCESS_KEY"] = creds.secret_access_key

    try:
        result = subprocess.run(cmd, capture_output=True, env=env, timeout=360)
    except subprocess.TimeoutExpired as exc:
        raise CachePullError(f"rclone pull timed out for key {key!r}") from exc
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace")
        log.debug("rclone pull exit=%d key=%s stderr=%s", result.returncode, key, stderr)
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
) -> None:
    """Push src_path to R2 via rclone copyto with write credentials.

    Raises RuntimeError on any non-zero rclone exit.
    """
    rclone = _require_rclone(rclone_path)
    remote = f":s3:{bucket}/{key}"

    cmd = [
        rclone,
        "copyto",
        str(src_path),
        remote,
        "--s3-provider",
        "Cloudflare",
        f"--s3-endpoint={endpoint}",
        f"--s3-upload-concurrency={upload_concurrency}",
        f"--s3-chunk-size={chunk_size_mb}M",
        "--contimeout=30s",
        "--timeout=300s",
        "--retries=3",
        "--low-level-retries=10",
        "--progress",
    ]

    env = os.environ.copy()
    env["RCLONE_S3_ACCESS_KEY_ID"] = creds.access_key_id
    env["RCLONE_S3_SECRET_ACCESS_KEY"] = creds.secret_access_key
    if creds.session_token:
        env["RCLONE_S3_SESSION_TOKEN"] = creds.session_token

    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"rclone push failed (exit {result.returncode}) for key {key!r}")
