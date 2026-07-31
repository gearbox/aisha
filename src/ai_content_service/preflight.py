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
from rich.table import Table

from .config import unwrap_secret
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
    from collections.abc import Awaitable, Callable, Mapping

    from rich.console import Console

    from .config import BundleConfig, ModelConfig, ModelFileConfig, Settings
    from .download_auth import BoundCredential, HostAuthPolicy

_PROBE_RANGE = "bytes=0-0"


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


def _make_egress_guard(
    credentials: tuple[BoundCredential, ...],
) -> Callable[[httpx.Request], Awaitable[None]]:
    async def _guard(request: httpx.Request) -> None:
        assert_no_credential_egress(str(request.url), request.headers, credentials)

    return _guard


async def check_bundle(bundle: BundleConfig, settings: Settings) -> PreflightReport:
    """Range-probe every model file in *bundle*, bounded by max_concurrent_downloads.

    Writes nothing to disk and never reads a response body.
    """
    registry = build_registry(settings)
    tokens: dict[str, str | None] = {
        "huggingface": unwrap_secret(settings.hf_token),
        "civitai": unwrap_secret(settings.civitai_api_token),
    }
    credentials = build_credentials(registry, tokens)
    secret_values = tuple(c.token for c in credentials)

    sem = asyncio.Semaphore(settings.max_concurrent_downloads)

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
    if status == HTTPStatus.OK:
        return parse_content_length(headers)
    return None


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
        style = "green" if r.ok else "red"
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
