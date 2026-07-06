"""Model-cache push orchestration (Typer-free core).

Mirrors the registry_service.py pattern: no Typer types cross this boundary,
console is injected, and everything here is testable without invoking the CLI.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

from .r2_transfer import R2TransferError, R2WriteCreds
from .r2_transfer import push as r2_push

if TYPE_CHECKING:
    from pathlib import Path

    from rich.console import Console

    from .config import BundleConfig, ModelConfig, ModelFileConfig, Settings


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


def push_models(settings: Settings, targets: list[PushTarget], console: Console) -> PushReport:
    """Push each target to the R2 model cache: mint credentials -> rclone push -> finalize.

    Owns a single httpx.Client for the whole batch. Per-file failures are
    recorded as a failed PushFileResult and processing continues with the
    next target (no fail-fast).
    """
    models_base = settings.comfyui_path / "models"
    admin_token = settings.apex_admin_token.get_secret_value()  # type: ignore[union-attr]
    apex_base = settings.apex_base_url.rstrip("/")  # type: ignore[union-attr]
    credentials_url = f"{apex_base}/v1/admin/model-cache/credentials"
    finalize_url = f"{apex_base}/v1/admin/model-cache/finalize"

    results: list[PushFileResult] = []

    with httpx.Client(timeout=30.0) as client:
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

            sha256 = fc.sha256 or _compute_sha256(file_path)
            size_bytes = file_path.stat().st_size
            source_url = _strip_token_from_url(fc.url)

            console.print("  Minting write credentials from Apex…")
            try:
                resp = client.post(
                    credentials_url,
                    json={
                        "sha256": sha256,
                        "filename": fc.filename,
                        "model_type": mc.model_type,
                        "source_url": source_url,
                    },
                    headers={"Authorization": f"Bearer {admin_token}"},
                )
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                detail = f"Apex credentials request failed: {e}"
                console.print(f"  [red]✗[/red] {detail}")
                results.append(PushFileResult(fc.filename, ok=False, detail=detail))
                continue
            except httpx.HTTPError as e:
                detail = f"Apex credentials request error: {e}"
                console.print(f"  [red]✗[/red] {detail}")
                results.append(PushFileResult(fc.filename, ok=False, detail=detail))
                continue

            cred_data = resp.json()
            r2_key: str = cred_data["r2_key"]
            raw_creds: dict[str, str] = cred_data["credentials"]
            write_creds = R2WriteCreds(
                access_key_id=raw_creds["access_key_id"],
                secret_access_key=raw_creds["secret_access_key"],
                session_token=raw_creds.get("session_token"),
            )

            console.print(f"  Uploading to R2 ({size_bytes / 1024 / 1024:.1f} MiB)…")
            try:
                r2_push(
                    src_path=file_path,
                    key=r2_key,
                    creds=write_creds,
                    bucket=settings.r2_model_cache_bucket,
                    endpoint=settings.r2_s3_endpoint,  # type: ignore[arg-type]
                    rclone_path=settings.rclone_path,
                    upload_concurrency=settings.rclone_upload_concurrency,
                    chunk_size_mb=settings.rclone_chunk_size_mb,
                    size_bytes=size_bytes,
                    max_timeout_s=settings.rclone_max_transfer_seconds,
                )
            except R2TransferError as e:
                detail = f"rclone push failed: {e}"
                console.print(f"  [red]✗[/red] {detail}")
                results.append(PushFileResult(fc.filename, ok=False, detail=detail))
                continue

            console.print("  Finalizing with Apex…")
            try:
                fin_resp = client.post(
                    finalize_url,
                    json={"sha256": sha256, "size_bytes": size_bytes},
                    headers={"Authorization": f"Bearer {admin_token}"},
                )
                if fin_resp.status_code in (409, 422):
                    detail = f"Apex finalize rejected ({fin_resp.status_code}): {fin_resp.text}"
                    console.print(f"  [red]✗[/red] {detail}")
                    results.append(PushFileResult(fc.filename, ok=False, detail=detail))
                    continue
                fin_resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                detail = f"Apex finalize failed: {e}"
                console.print(f"  [red]✗[/red] {detail}")
                results.append(PushFileResult(fc.filename, ok=False, detail=detail))
                continue
            except httpx.HTTPError as e:
                detail = f"Apex finalize error: {e}"
                console.print(f"  [red]✗[/red] {detail}")
                results.append(PushFileResult(fc.filename, ok=False, detail=detail))
                continue

            console.print(f"  [green]✓[/green] cache.push.done — {fc.filename}")
            results.append(PushFileResult(fc.filename, ok=True))

    return PushReport(results=results)
