"""Model-cache push orchestration (Typer-free core).

Mirrors the registry_service.py pattern: no Typer types cross this boundary,
console is injected, and everything here is testable without invoking the CLI.
"""

from __future__ import annotations

import contextlib
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .cache_credentials import CacheCredentialError, CacheCredentialProvider
from .cache_keys import cache_key_for_sha256
from .file_hashes import compute_file_sha256
from .r2_transfer import R2ObjectStat, R2ReadCreds, R2TransferError, read_creds_from_settings, stat
from .r2_transfer import pull as r2_pull
from .r2_transfer import push as r2_push
from .url_sanitizer import sanitize_civitai_url_for_output

if TYPE_CHECKING:
    from rich.console import Console

    from .config import BundleConfig, ModelConfig, ModelFileConfig, Settings


_LOWERCASE_HEX_SHA256_RE = re.compile(r"[0-9a-f]{64}")


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


@dataclass(frozen=True, slots=True)
class _PreparedPushTarget:
    """The local bytes and metadata safe to hand to a credential provider."""

    path: Path
    sha256: str
    size_bytes: int
    source_url: str


class _PushTargetPreparationError(Exception):
    """An expected, per-file local filesystem preparation failure."""


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


def _push_failure(
    results: list[PushFileResult], console: Console, filename: str, detail: str
) -> None:
    """Record and render one isolated cache-push failure."""
    console.print(f"  [red]✗[/red] {detail}")
    results.append(PushFileResult(filename, ok=False, detail=detail))


def _prepare_push_target(models_base: Path, target: PushTarget) -> _PreparedPushTarget:
    """Validate and inspect one local target without widening batch failure scope."""
    file_path = target.disk_path
    try:
        base_resolved = models_base.resolve()
        resolved = file_path.resolve()
        if base_resolved not in resolved.parents:
            raise _PushTargetPreparationError(
                f"Refusing path outside models dir: {target.file.filename!r}"
            )
        if not resolved.is_file():
            raise _PushTargetPreparationError(f"File not found on disk: {file_path}")
        sha256 = target.file.sha256 or compute_file_sha256(resolved)
        if not _LOWERCASE_HEX_SHA256_RE.fullmatch(sha256):
            raise _PushTargetPreparationError(
                f"sha256 for {target.file.filename!r} is not lowercase-hex: {sha256!r}"
            )
        size_bytes = resolved.stat().st_size
    except OSError as exc:
        raise _PushTargetPreparationError(
            f"Cannot prepare local file {target.file.filename!r}: {exc}"
        ) from exc
    return _PreparedPushTarget(
        path=resolved,
        sha256=sha256,
        size_bytes=size_bytes,
        source_url=sanitize_civitai_url_for_output(target.file.url),
    )


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
        mc, fc = target.model, target.file
        console.print(f"\n[bold]Pushing {fc.filename}[/bold]")

        try:
            prepared = _prepare_push_target(models_base, target)
        except _PushTargetPreparationError as exc:
            _push_failure(results, console, fc.filename, str(exc))
            continue

        console.print(f"  Minting write credentials ({provider.name})…")
        try:
            minted = provider.mint(
                sha256=prepared.sha256,
                filename=fc.filename,
                model_type=mc.model_type,
                source_url=prepared.source_url,
            )
        except CacheCredentialError as exc:
            _push_failure(results, console, fc.filename, str(exc))
            continue

        console.print(f"  Uploading to R2 ({prepared.size_bytes / 1024 / 1024:.1f} MiB)…")
        try:
            r2_push(
                src_path=prepared.path,
                key=minted.r2_key,
                creds=minted.creds,
                bucket=settings.r2_model_cache_bucket,
                endpoint=settings.r2_s3_endpoint or "",
                rclone_path=settings.rclone_path,
                upload_concurrency=settings.rclone_upload_concurrency,
                chunk_size_mb=settings.rclone_chunk_size_mb,
                size_bytes=prepared.size_bytes,
                max_timeout_s=settings.rclone_max_transfer_seconds,
            )
        except R2TransferError as exc:
            _push_failure(results, console, fc.filename, f"rclone push failed: {exc}")
            continue

        console.print("  Finalizing cache entry…")
        try:
            provider.finalize(sha256=prepared.sha256, size_bytes=prepared.size_bytes)
        except CacheCredentialError as exc:
            _push_failure(results, console, fc.filename, str(exc))
            continue

        console.print(f"  [green]✓[/green] cache.push.done — {fc.filename}")
        results.append(PushFileResult(fc.filename, ok=True))

    return PushReport(results=results)


def _verify_result(
    target: PushTarget, key: str, ok: bool, status: str, detail: str = ""
) -> VerifyFileResult:
    """Build one verification result with the target's stable display name."""
    return VerifyFileResult(
        filename=target.file.filename,
        key=key,
        ok=ok,
        status=status,
        detail=detail,
    )


def _verify_shallow(target: PushTarget, key: str, object_stat: R2ObjectStat) -> VerifyFileResult:
    """Perform the size-only portion of a successful object stat."""
    expected_size = target.file.size_bytes
    if expected_size is not None and object_stat.size_bytes != expected_size:
        return _verify_result(
            target,
            key,
            False,
            "SIZE MISMATCH",
            f"R2 reports {object_stat.size_bytes} bytes; bundle declares {expected_size}.",
        )
    return _verify_result(target, key, True, "PRESENT", f"{object_stat.size_bytes} bytes")


def _verify_deep(
    settings: Settings,
    target: PushTarget,
    key: str,
    object_stat: R2ObjectStat,
    read_creds: R2ReadCreds,
    models_base: Path,
) -> VerifyFileResult:
    """Pull and hash one object, cleaning its local temporary file on every path."""
    try:
        models_base.mkdir(parents=True, exist_ok=True)
        available = shutil.disk_usage(models_base).free
    except OSError as exc:
        return _verify_result(
            target,
            key,
            False,
            "DEEP VERIFY ERROR",
            f"Could not inspect models filesystem: {exc}",
        )
    if available < object_stat.size_bytes:
        return _verify_result(
            target,
            key,
            False,
            "INSUFFICIENT SPACE",
            (
                f"Need {object_stat.size_bytes} free bytes for a temporary pull; "
                f"only {available} are available."
            ),
        )

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{target.file.filename}.",
            suffix=".cache-verify",
            dir=models_base,
            delete=False,
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
        actual = compute_file_sha256(temp_path)
    except (OSError, R2TransferError, ValueError) as exc:
        return _verify_result(target, key, False, "DEEP VERIFY ERROR", str(exc))
    finally:
        if temp_path is not None:
            with contextlib.suppress(OSError):
                temp_path.unlink(missing_ok=True)

    if actual != target.file.sha256:
        return _verify_result(
            target,
            key,
            False,
            "CHECKSUM MISMATCH",
            f"R2 content hashes to {actual}; bundle declares {target.file.sha256}.",
        )
    return _verify_result(target, key, True, "CHECKSUM OK", f"{object_stat.size_bytes} bytes")


def verify_models(
    settings: Settings,
    targets: list[PushTarget],
    *,
    deep: bool,
) -> VerifyReport:
    """Verify target objects through the read-only R2 credential path.

    The default check is a cheap object stat. ``deep`` pulls each object to a
    temporary file under the local models directory and re-hashes it, matching
    the filesystem characteristics of a real deployment without retaining the
    downloaded bytes.
    """
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
                _verify_result(
                    target,
                    "",
                    False,
                    "NO SHA256",
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
            results.append(_verify_result(target, key, False, "STAT ERROR", str(exc)))
            continue

        if object_stat is None:
            results.append(_verify_result(target, key, False, "MISSING"))
            continue
        shallow_result = _verify_shallow(target, key, object_stat)
        if not shallow_result.ok:
            results.append(shallow_result)
            continue
        if not deep:
            results.append(shallow_result)
            continue
        results.append(_verify_deep(settings, target, key, object_stat, read_creds, models_base))

    return VerifyReport(results=results)
