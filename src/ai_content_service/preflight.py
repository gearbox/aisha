"""Bundle model-file preflight checks (Typer-free core).

Mirrors the registry_service.py / cache_service.py pattern: no Typer types
cross this boundary, console is injected. Probes every model file with a
1-byte Range GET — reports what a real download would hit without writing
anything to disk, so a broken bundle surfaces in seconds instead of after a
multi-dollar GPU provision.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from http import HTTPStatus
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import httpx
import yaml
from pydantic import ValidationError
from rich.table import Table

from .config import BundleConfig, unwrap_secret
from .content_disposition_utils import parse_content_disposition
from .download_auth import (
    AUTH_RETRY_STATUSES,
    assert_no_credential_egress,
    attempt_with_auth,
    build_credentials,
    build_registry,
    redact_url,
    resolve_policy,
)
from .http_utils import parse_content_length, parse_content_range_total

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping, Sequence
    from pathlib import Path

    from rich.console import Console

    from .config import ModelConfig, ModelFileConfig, Settings
    from .download_auth import BoundCredential, HostAuthPolicy

_PROBE_RANGE = "bytes=0-0"
_MISSING_URL_FLAG = "Missing source URL — replace the snapshot TODO before deployment"


@dataclass(frozen=True, slots=True)
class _ProbeResult:
    """Headers-only outcome of a single Range probe. The body is never read."""

    status_code: int
    headers: dict[str, str]


@dataclass(frozen=True, slots=True)
class FileCheckResult:
    """Outcome of probing a single model file."""

    model_name: str
    filename: str
    url: str  # already redacted
    status: str
    content_type: str
    server_filename: str | None
    content_length: int | None
    ok: bool
    range_supported: bool
    flag: str | None = None


@dataclass(frozen=True, slots=True)
class PreflightReport:
    """Aggregate outcome of `check_bundle`."""

    results: tuple[FileCheckResult, ...]

    @property
    def ok(self) -> bool:
        return all(r.ok for r in self.results)


@dataclass(frozen=True, slots=True)
class BundleCheckResult:
    """Outcome of checking one bundle in a `--all` run.

    A bundle that failed to parse carries ``parse_error`` and no file results
    (C4b) -- it is a reported row, never a raised exception.
    """

    bundle_name: str
    parse_error: str | None = None
    file_results: tuple[FileCheckResult, ...] = ()

    @property
    def ok(self) -> bool:
        return self.parse_error is None and all(r.ok for r in self.file_results)


@dataclass(frozen=True, slots=True)
class MultiBundleReport:
    """Aggregate outcome of `check_all_bundles`."""

    results: tuple[BundleCheckResult, ...]

    @property
    def ok(self) -> bool:
        return bool(self.results) and all(r.ok for r in self.results)


def _make_egress_guard(
    credentials: tuple[BoundCredential, ...],
) -> Callable[[httpx.Request], Awaitable[None]]:
    async def _guard(request: httpx.Request) -> None:
        assert_no_credential_egress(str(request.url), request.headers, credentials)

    return _guard


async def check_bundle(
    bundle: BundleConfig,
    settings: Settings,
    *,
    offline: bool = False,
    semaphore: asyncio.Semaphore | None = None,
) -> PreflightReport:
    """Range-probe every model file in *bundle*, bounded by max_concurrent_downloads.

    Writes nothing to disk and never reads a response body. With ``offline=True``
    (C4c), no HTTP client is ever constructed -- every file with a URL is
    reported "not checked (offline)" rather than a green row that would imply
    it was actually probed (pitfall #7); a missing URL is still flagged, since
    that requires no network access to detect.
    """
    registry = build_registry(settings)
    tokens: dict[str, str | None] = {
        "huggingface": unwrap_secret(settings.hf_token),
        "civitai": unwrap_secret(settings.civitai_api_token),
    }
    credentials = build_credentials(registry, tokens)
    secret_values = tuple(c.token for c in credentials)

    if offline:
        offline_results = tuple(
            _offline_result(model, file, secret_values)
            if file.url
            else _missing_url_result(model, file)
            for model in bundle.models
            for file in model.files
        )
        return PreflightReport(results=offline_results)

    sem = (
        semaphore if semaphore is not None else asyncio.Semaphore(settings.max_concurrent_downloads)
    )

    async def _bounded(
        client: httpx.AsyncClient, model: ModelConfig, file: ModelFileConfig
    ) -> FileCheckResult:
        async with sem:
            return await _check_file(
                client, model, file, registry, tokens, settings.download_user_agent, secret_values
            )

    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=30.0,
        event_hooks={"request": [_make_egress_guard(credentials)]},
    ) as client:
        results = await asyncio.gather(
            *[_bounded(client, model, file) for model in bundle.models for file in model.files]
        )

    return PreflightReport(results=tuple(results))


def _load_bundle_config(bundle_path: Path) -> BundleConfig:
    """Load and parse bundle.yaml at *bundle_path*. Raises on any problem; the
    caller (`check_bundle_path`) is where that becomes a reported row (C4b).
    """
    bundle_yaml = bundle_path / "bundle.yaml"
    raw = yaml.safe_load(bundle_yaml.read_text())
    if not isinstance(raw, dict):
        msg = f"invalid bundle config at {bundle_yaml}: expected a mapping"
        raise ValueError(msg)
    return BundleConfig.model_validate(raw)


async def check_bundle_path(
    bundle_name: str,
    bundle_path: Path,
    settings: Settings,
    *,
    offline: bool = False,
    semaphore: asyncio.Semaphore | None = None,
) -> BundleCheckResult:
    """Load, parse, and probe one bundle by path.

    A parse failure (missing file, bad YAML, or a schema violation such as
    `extra="forbid"` rejecting an unknown key) becomes ``parse_error`` on the
    result -- this function never raises (C4b), so a `--all` run over many
    bundles can't be aborted by the first broken one.
    """
    try:
        bundle_config = _load_bundle_config(bundle_path)
    except (OSError, yaml.YAMLError, ValidationError, ValueError) as e:
        return BundleCheckResult(bundle_name=bundle_name, parse_error=str(e))

    report = await check_bundle(bundle_config, settings, offline=offline, semaphore=semaphore)
    return BundleCheckResult(bundle_name=bundle_name, file_results=report.results)


async def check_all_bundles(
    entries: Sequence[tuple[str, Path | None]] | Sequence[str],
    settings: Settings,
    *,
    offline: bool = False,
    resolve_bundle_path: Callable[[str], Awaitable[Path]] | None = None,
) -> MultiBundleReport:
    """Check every ``(bundle_name, bundle_path)`` pair. Never raises (C4b) --
    each bundle is independent, so one broken bundle never stops the run.

    ``entries`` may contain already-resolved paths, as used by direct callers,
    or bundle names with ``resolve_bundle_path`` supplied by a registry. The
    latter keeps per-bundle registry failures in the report instead of making
    the CLI reconstruct a second multi-bundle code path.
    """
    semaphore = asyncio.Semaphore(settings.max_concurrent_downloads)

    async def _check_entry(entry: tuple[str, Path | None] | str) -> BundleCheckResult:
        if isinstance(entry, str):
            name = entry
            path: Path | None = None
        else:
            name, path = entry

        if path is None:
            if resolve_bundle_path is None:
                return BundleCheckResult(
                    bundle_name=name,
                    parse_error="no bundle path or registry resolver provided",
                )
            try:
                path = await resolve_bundle_path(name)
            except Exception as e:
                return BundleCheckResult(bundle_name=name, parse_error=str(e))

        return await check_bundle_path(
            name,
            path,
            settings,
            offline=offline,
            semaphore=semaphore,
        )

    results = await asyncio.gather(*(_check_entry(entry) for entry in entries))
    return MultiBundleReport(results=tuple(results))


async def _probe(
    client: httpx.AsyncClient, url: str, headers: dict[str, str]
) -> _ProbeResult | Exception:
    """Headers-only probe. The body is never iterated, so a host that
    ignores Range costs one round trip, not one model file.
    """
    try:
        async with client.stream("GET", url, headers=headers) as response:
            return _ProbeResult(
                status_code=response.status_code,
                headers=dict(response.headers),
            )
    except Exception as e:  # M1a — never abort the run
        return e


def _resolve_size(status: int, headers: Mapping[str, str]) -> int | None:
    """Size is the `Content-Range` total, or `Content-Length` on a `200`, or None.

    Never the length of a probe slice — that is the R2 bug this replaces.
    """
    content_range_total = parse_content_range_total(headers)
    if content_range_total is not None:
        return content_range_total
    return parse_content_length(headers) if status == HTTPStatus.OK else None


def _missing_url_result(model: ModelConfig, file: ModelFileConfig) -> FileCheckResult:
    """A file with no source URL — the snapshot placeholder. Requires no network access."""
    return FileCheckResult(
        model_name=model.name,
        filename=file.filename,
        url="",
        status="MISSING URL",
        content_type="",
        server_filename=None,
        content_length=None,
        ok=False,
        range_supported=False,
        flag=_MISSING_URL_FLAG,
    )


def _offline_result(
    model: ModelConfig, file: ModelFileConfig, secrets: tuple[str, ...]
) -> FileCheckResult:
    """A file with a URL, not probed because `--offline` forbids any request (C4c)."""
    return FileCheckResult(
        model_name=model.name,
        filename=file.filename,
        url=redact_url(file.url, secrets),
        status="OFFLINE",
        content_type="",
        server_filename=None,
        content_length=None,
        ok=True,
        range_supported=False,
        flag="not checked (offline)",
    )


async def _check_file(
    client: httpx.AsyncClient,
    model: ModelConfig,
    file: ModelFileConfig,
    registry: tuple[HostAuthPolicy, ...],
    tokens: dict[str, str | None],
    user_agent: str,
    secrets: tuple[str, ...] = (),
) -> FileCheckResult:
    """Always returns a row. Never raises — a preflight tool that dies on the
    malformed input it exists to detect is useless (E2). ``redact_url`` cannot
    raise, so it is safe above the ``try``; nothing else is placed there.
    """
    if not file.url:
        return _missing_url_result(model, file)
    try:
        return await _check_file_inner(client, model, file, registry, tokens, user_agent, secrets)
    except Exception as e:
        return FileCheckResult(
            model_name=model.name,
            filename=file.filename,
            url=redact_url(file.url, secrets),
            status=f"ERROR: {redact_url(str(e), secrets)}",
            content_type="",
            server_filename=None,
            content_length=None,
            ok=False,
            range_supported=False,
            flag="invalid URL or unresolvable host",
        )


async def _check_file_inner(
    client: httpx.AsyncClient,
    model: ModelConfig,
    file: ModelFileConfig,
    registry: tuple[HostAuthPolicy, ...],
    tokens: dict[str, str | None],
    user_agent: str,
    secrets: tuple[str, ...],
) -> FileCheckResult:
    policy = resolve_policy(registry, file.url)
    token = tokens.get(policy.name) if policy is not None else None
    base_headers = {"User-Agent": user_agent, "Range": _PROBE_RANGE}

    async def _send(url: str, headers: dict[str, str]) -> _ProbeResult | Exception:
        return await _probe(client, url, headers)

    def _status_of(outcome: _ProbeResult | Exception) -> int:
        return outcome.status_code if isinstance(outcome, _ProbeResult) else -1

    outcome, _transport = await attempt_with_auth(
        policy, token, file.url, base_headers, send=_send, status_of=_status_of
    )

    redacted_url = redact_url(file.url, secrets)

    if isinstance(outcome, Exception):
        return FileCheckResult(
            model_name=model.name,
            filename=file.filename,
            url=redacted_url,
            status=f"ERROR: {redact_url(str(outcome), secrets)}",
            content_type="",
            server_filename=None,
            content_length=None,
            ok=False,
            range_supported=False,
            flag="request failed",
        )

    status = outcome.status_code
    headers = outcome.headers
    content_type = headers.get("content-type", "")
    server_filename = parse_content_disposition(headers.get("content-disposition"))
    content_length = _resolve_size(status, headers)

    is_html = content_type.split(";", 1)[0].strip().lower() == "text/html"
    range_supported = status == HTTPStatus.PARTIAL_CONTENT
    ok = status in (HTTPStatus.OK, HTTPStatus.PARTIAL_CONTENT) and not is_html

    flag: str | None = None
    if is_html:
        flag = "HTML response — auth/domain problem"
    elif status in AUTH_RETRY_STATUSES:
        flag = "Unauthorized — check API token"
    elif status == HTTPStatus.NOT_FOUND:
        host = urlparse(file.url).netloc.lower()
        flag = (
            "404 on civitai.com — model may be NSFW; try civitai.red"
            if "civitai.com" in host
            else "404 Not Found"
        )
    elif status == HTTPStatus.OK:
        flag = "Range ignored — no resume support"

    return FileCheckResult(
        model_name=model.name,
        filename=file.filename,
        url=redacted_url,
        status=str(status),
        content_type=content_type,
        server_filename=server_filename,
        content_length=content_length,
        ok=ok,
        range_supported=range_supported,
        flag=flag,
    )


def render_report(report: PreflightReport, console: Console) -> None:
    """Render *report* as a Rich table."""
    table = Table(title="Model Preflight Check")
    table.add_column("Model", style="cyan")
    table.add_column("Filename", style="green")
    table.add_column("Status")
    table.add_column("Content-Type")
    table.add_column("Server Filename")
    table.add_column("Size")
    table.add_column("Resume")
    table.add_column("Flag", style="yellow")

    for r in report.results:
        style = _row_style(r)
        resume_style = "green" if r.range_supported else "yellow"
        table.add_row(
            r.model_name,
            r.filename,
            f"[{style}]{r.status}[/{style}]",
            r.content_type or "-",
            r.server_filename or "-",
            str(r.content_length) if r.content_length is not None else "-",
            f"[{resume_style}]{'yes' if r.range_supported else 'no'}[/{resume_style}]",
            r.flag or "",
        )

    console.print(table)


def _row_style(result: FileCheckResult) -> str:
    """Return the status colour for a file row."""
    if result.status in ("OFFLINE", "MISSING URL"):
        return "yellow"
    return "green" if result.ok else "red"


def render_multi_report(report: MultiBundleReport, console: Console) -> None:
    """Render *report* as one section per bundle -- a parse failure is a red
    summary line (name + error) rather than a table, since there are no file
    rows to show (C4b)."""
    if not report.results:
        console.print("[yellow]no bundles found[/yellow]")
        return

    for bundle_result in report.results:
        console.print(f"\n[bold cyan]{bundle_result.bundle_name}[/bold cyan]")
        if bundle_result.parse_error is not None:
            console.print(f"[red]✗ failed to parse:[/red] {bundle_result.parse_error}")
            continue
        render_report(PreflightReport(results=bundle_result.file_results), console)


def _file_result_to_dict(r: FileCheckResult) -> dict[str, object]:
    return {
        "model": r.model_name,
        "filename": r.filename,
        "url": r.url,
        "status": r.status,
        "content_type": r.content_type,
        "server_filename": r.server_filename,
        "content_length": r.content_length,
        "ok": r.ok,
        "range_supported": r.range_supported,
        "flag": r.flag,
    }


def report_to_dict(report: PreflightReport) -> dict[str, object]:
    """Machine-readable form of a single-bundle report, for `--json`."""
    return {"ok": report.ok, "files": [_file_result_to_dict(r) for r in report.results]}


def multi_report_to_dict(report: MultiBundleReport) -> dict[str, object]:
    """Machine-readable form of a `--all` report, for `--json` (C4d)."""
    result: dict[str, object] = {
        "ok": report.ok,
        "bundles": [
            {
                "bundle": b.bundle_name,
                "ok": b.ok,
                "parse_error": b.parse_error,
                "files": [_file_result_to_dict(r) for r in b.file_results],
            }
            for b in report.results
        ],
    }
    if not report.results:
        result["message"] = "no bundles found"
    return result
