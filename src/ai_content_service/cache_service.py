"""Model-cache push orchestration (Typer-free core).

Mirrors the registry_service.py pattern: no Typer types cross this boundary,
console is injected, and everything here is testable without invoking the CLI.
"""

from __future__ import annotations

import contextlib
import hashlib
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import parse_qs, urlencode, urlparse

from .cache_credentials import CacheCredentialError, CacheCredentialProvider
from .cache_keys import cache_key_for_sha256
from .r2_transfer import R2TransferError, read_creds_from_settings, stat
from .r2_transfer import pull as r2_pull
from .r2_transfer import push as r2_push

if TYPE_CHECKING:
    from rich.console import Console

    from .config import BundleConfig, ModelConfig, ModelFileConfig, Settings


_LOWERCASE_HEX_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _compute_sha256(path: Path) -> str:
    """Compute SHA-256 of a file in chunks."""
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def _strip_token_from_url(url: str) -> str:
    """Remove the Civitai 'token' query param before sending to Apex."""
    parsed = urlparse(url)
    query = parse_qs(parsed.query, keep_blank_values=True)
    query.pop("token", None)
    return parsed._replace(query=urlencode(query, doseq=True)).geturl()


@dataclass
class PushTarget:
    """A single model file resolved to its on-disk location."""

    model: ModelConfig
    file: ModelFileConfig
    disk_path: Path


@dataclass
class PushFileResult:
    """Outcome of pushing a single model file."""

    filename: str
    ok: bool
    detail: str = ""


@dataclass
class PushReport:
    """Aggregate outcome of a `cache push` invocation."""

    results: list[PushFileResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(r.ok for r in self.results)


@dataclass(frozen=True, slots=True)
class VerifyFileResult:
    """Outcome of verifying one model-cache object."""

    filename: str
    key: str
    ok: bool
    status: str
    detail: str = ""


@dataclass
class VerifyReport:
    """Aggregate outcome of a `cache verify` invocation."""

    results: list[VerifyFileResult] = field(default_factory=list)
    configuration_error: str | None = None

    @property
    def ok(self) -> bool:
        return self.configuration_error is None and all(result.ok for result in self.results)


def collect_targets(
    bundle_config: BundleConfig, models_base: Path, only_filename: str | None
) -> list[PushTarget]:
    """Collect push targets from a bundle, optionally filtered to one filename."""
    targets: list[PushTarget] = []
    for mc in bundle_config.models:
        model_dir = models_base / mc.target_subpath
        targets.extend(
            PushTarget(model=mc, file=fc, disk_path=model_dir / fc.filename)
            for fc in mc.files
            if not only_filename or fc.filename == only_filename
        )
    return targets


def push_models(
    settings: Settings,
    targets: list[PushTarget],
    console: Console,
    *,
    provider: CacheCredentialProvider,
) -> PushReport:
    """Push each target to the R2 model cache: mint credentials -> rclone push -> finalize.

    Per-file failures are recorded as a failed PushFileResult and processing
    continues with the next target (no fail-fast). Credential policy is
    injected by the command composition root, so direct authoring never needs
    Apex while the default mode preserves its mint/finalize contract.
    """
    models_base = settings.comfyui_path / "models"
    results: list[PushFileResult] = []

    for target in targets:
        mc, fc, file_path = target.model, target.file, target.disk_path
        console.print(f"\n[bold]Pushing {fc.filename}[/bold]")

        resolved = file_path.resolve()
        if models_base.resolve() not in resolved.parents:
            detail = f"Refusing path outside models dir: {fc.filename!r}"
            console.print(f"  [red]✗[/red] {detail}")
            results.append(PushFileResult(fc.filename, ok=False, detail=detail))
            continue

        if not file_path.exists():
            detail = f"File not found on disk: {file_path}"
            console.print(f"  [red]✗[/red] {detail}")
            results.append(PushFileResult(fc.filename, ok=False, detail=detail))
            continue

        # ModelFileConfig.sha256 is the normalisation authority (config.py's
        # field_validator); _compute_sha256 always returns a lowercase hexdigest.
        # A mismatch here means one of those two invariants regressed.
        sha256 = fc.sha256 or _compute_sha256(file_path)
        if not _LOWERCASE_HEX_SHA256_RE.fullmatch(sha256):
            detail = f"sha256 for {fc.filename!r} is not lowercase-hex: {sha256!r}"
            console.print(f"  [red]✗[/red] {detail}")
            results.append(PushFileResult(fc.filename, ok=False, detail=detail))
            continue
        size_bytes = file_path.stat().st_size
        source_url = _strip_token_from_url(fc.url)

        console.print(f"  Minting write credentials ({provider.name})…")
        try:
            minted = provider.mint(
                sha256=sha256,
                filename=fc.filename,
                model_type=mc.model_type,
                source_url=source_url,
            )
        except CacheCredentialError as exc:
            detail = str(exc)
            console.print(f"  [red]✗[/red] {detail}")
            results.append(PushFileResult(fc.filename, ok=False, detail=detail))
            continue

        console.print(f"  Uploading to R2 ({size_bytes / 1024 / 1024:.1f} MiB)…")
        try:
            r2_push(
                src_path=file_path,
                key=minted.r2_key,
                creds=minted.creds,
                bucket=settings.r2_model_cache_bucket,
                endpoint=settings.r2_s3_endpoint or "",
                rclone_path=settings.rclone_path,
                upload_concurrency=settings.rclone_upload_concurrency,
                chunk_size_mb=settings.rclone_chunk_size_mb,
                size_bytes=size_bytes,
                max_timeout_s=settings.rclone_max_transfer_seconds,
            )
        except R2TransferError as exc:
            detail = f"rclone push failed: {exc}"
            console.print(f"  [red]✗[/red] {detail}")
            results.append(PushFileResult(fc.filename, ok=False, detail=detail))
            continue

        console.print("  Finalizing cache entry…")
        try:
            provider.finalize(sha256=sha256, size_bytes=size_bytes)
        except CacheCredentialError as exc:
            detail = str(exc)
            console.print(f"  [red]✗[/red] {detail}")
            results.append(PushFileResult(fc.filename, ok=False, detail=detail))
            continue

        console.print(f"  [green]✓[/green] cache.push.done — {fc.filename}")
        results.append(PushFileResult(fc.filename, ok=True))

    return PushReport(results=results)


def verify_models(
    settings: Settings,
    targets: list[PushTarget],
    console: Console,
    *,
    deep: bool,
) -> VerifyReport:
    """Verify target objects through the read-only R2 credential path.

    The default check is a cheap object stat. ``deep`` pulls each object to a
    temporary file under the local models directory and re-hashes it, matching
    the filesystem characteristics of a real deployment without retaining the
    downloaded bytes.
    """
    del console
    read_creds = read_creds_from_settings(settings)
    if read_creds is None:
        return VerifyReport(
            configuration_error=(
                "ACS_R2_S3_ENDPOINT, ACS_R2_READONLY_ACCESS_KEY_ID, and "
                "ACS_R2_READONLY_SECRET_ACCESS_KEY must be set for cache verify"
            )
        )

    models_base = settings.comfyui_path / "models"
    results: list[VerifyFileResult] = []
    for target in targets:
        fc = target.file
        if fc.sha256 is None:
            results.append(
                VerifyFileResult(
                    filename=fc.filename,
                    key="",
                    ok=False,
                    status="NO SHA256",
                    detail="A content-addressed cache lookup requires sha256.",
                )
            )
            continue

        key = ""
        try:
            key = cache_key_for_sha256(fc.sha256)
            object_stat = stat(
                key=key,
                creds=read_creds,
                bucket=settings.r2_model_cache_bucket,
                endpoint=settings.r2_s3_endpoint or "",
                rclone_path=settings.rclone_path,
            )
        except (R2TransferError, ValueError) as exc:
            results.append(
                VerifyFileResult(
                    filename=fc.filename,
                    key=key,
                    ok=False,
                    status="STAT ERROR",
                    detail=str(exc),
                )
            )
            continue

        if object_stat is None:
            results.append(
                VerifyFileResult(filename=fc.filename, key=key, ok=False, status="MISSING")
            )
            continue
        if fc.size_bytes is not None and object_stat.size_bytes != fc.size_bytes:
            results.append(
                VerifyFileResult(
                    filename=fc.filename,
                    key=key,
                    ok=False,
                    status="SIZE MISMATCH",
                    detail=f"R2 reports {object_stat.size_bytes} bytes; bundle declares {fc.size_bytes}.",
                )
            )
            continue
        if not deep:
            results.append(
                VerifyFileResult(
                    filename=fc.filename,
                    key=key,
                    ok=True,
                    status="PRESENT",
                    detail=f"{object_stat.size_bytes} bytes",
                )
            )
            continue

        try:
            models_base.mkdir(parents=True, exist_ok=True)
            available = shutil.disk_usage(models_base).free
        except OSError as exc:
            results.append(
                VerifyFileResult(
                    filename=fc.filename,
                    key=key,
                    ok=False,
                    status="DEEP VERIFY ERROR",
                    detail=f"Could not inspect models filesystem: {exc}",
                )
            )
            continue
        if available < object_stat.size_bytes:
            results.append(
                VerifyFileResult(
                    filename=fc.filename,
                    key=key,
                    ok=False,
                    status="INSUFFICIENT SPACE",
                    detail=(
                        f"Need {object_stat.size_bytes} free bytes for a temporary pull; "
                        f"only {available} are available."
                    ),
                )
            )
            continue

        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix=f".{fc.filename}.", suffix=".cache-verify", dir=models_base, delete=False
            ) as temp_file:
                temp_path = Path(temp_file.name)
            r2_pull(
                key=key,
                dest_path=temp_path,
                creds=read_creds,
                bucket=settings.r2_model_cache_bucket,
                endpoint=settings.r2_s3_endpoint or "",
                rclone_path=settings.rclone_path,
                multi_thread_streams=settings.rclone_multi_thread_streams,
                size_bytes=object_stat.size_bytes,
                max_timeout_s=settings.rclone_max_transfer_seconds,
            )
            actual = _compute_sha256(temp_path)
        except Exception as exc:  # Never let a reporter leave a stale deep-verify file behind.
            results.append(
                VerifyFileResult(
                    filename=fc.filename,
                    key=key,
                    ok=False,
                    status="DEEP VERIFY ERROR",
                    detail=str(exc),
                )
            )
        else:
            if actual != fc.sha256:
                results.append(
                    VerifyFileResult(
                        filename=fc.filename,
                        key=key,
                        ok=False,
                        status="CHECKSUM MISMATCH",
                        detail=f"R2 content hashes to {actual}; bundle declares {fc.sha256}.",
                    )
                )
            else:
                results.append(
                    VerifyFileResult(
                        filename=fc.filename,
                        key=key,
                        ok=True,
                        status="CHECKSUM OK",
                        detail=f"{object_stat.size_bytes} bytes",
                    )
                )
        finally:
            if temp_path is not None:
                # The verification outcome is already recorded above. Do not
                # let an unlink race turn a reporter into an exception path.
                with contextlib.suppress(OSError):
                    temp_path.unlink(missing_ok=True)

    return VerifyReport(results=results)
