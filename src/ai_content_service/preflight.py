"""Bundle model-file preflight checks (Typer-free core).

Mirrors the registry_service.py / cache_service.py pattern: no Typer types
cross this boundary, console is injected. Probes every model file with a
1-byte Range GET — reports what a real download would hit without writing
anything to disk, so a broken bundle surfaces in seconds instead of after a
multi-dollar GPU provision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import httpx
from rich.table import Table

from .config import unwrap_secret
from .content_disposition_utils import parse_content_disposition
from .download_auth import AuthTransport, apply_auth, build_registry, redact_url, resolve_policy

if TYPE_CHECKING:
    from rich.console import Console

    from .config import BundleConfig, ModelConfig, ModelFileConfig, Settings
    from .download_auth import HostAuthPolicy

_PROBE_RANGE = "bytes=0-0"
_AUTH_RETRY_STATUSES = (401, 403)


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
    flag: str | None = None


@dataclass(frozen=True, slots=True)
class PreflightReport:
    """Aggregate outcome of `check_bundle`."""

    results: tuple[FileCheckResult, ...]

    @property
    def ok(self) -> bool:
        return all(r.ok for r in self.results)


async def check_bundle(bundle: BundleConfig, settings: Settings) -> PreflightReport:
    """Range-probe every model file in *bundle*. Writes nothing to disk."""
    registry = build_registry(settings)
    tokens: dict[str, str | None] = {
        "huggingface": unwrap_secret(settings.hf_token),
        "civitai": unwrap_secret(settings.civitai_api_token),
    }

    results: list[FileCheckResult] = []
    async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
        for model in bundle.models:
            for file in model.files:
                results.append(
                    await _check_file(
                        client, model, file, registry, tokens, settings.download_user_agent
                    )
                )

    return PreflightReport(results=tuple(results))


async def _probe(
    client: httpx.AsyncClient, url: str, headers: dict[str, str]
) -> httpx.Response | Exception:
    try:
        return await client.get(url, headers=headers)
    except httpx.HTTPError as e:
        return e


async def _check_file(
    client: httpx.AsyncClient,
    model: ModelConfig,
    file: ModelFileConfig,
    registry: tuple[HostAuthPolicy, ...],
    tokens: dict[str, str | None],
    user_agent: str,
) -> FileCheckResult:
    policy = resolve_policy(registry, file.url)
    token = tokens.get(policy.name) if policy is not None else None
    transport = policy.primary if policy is not None else AuthTransport.NONE

    def _build(transport: AuthTransport) -> tuple[str, dict[str, str]]:
        url, headers = (
            apply_auth(policy, transport, file.url, {}, token)
            if policy is not None
            else (file.url, {})
        )
        return url, {**headers, "User-Agent": user_agent, "Range": _PROBE_RANGE}

    url, headers = _build(transport)
    outcome = await _probe(client, url, headers)

    if (
        isinstance(outcome, httpx.Response)
        and outcome.status_code in _AUTH_RETRY_STATUSES
        and policy is not None
        and policy.fallback is not None
    ):
        url, headers = _build(policy.fallback)
        outcome = await _probe(client, url, headers)

    redacted_url = redact_url(file.url)

    if isinstance(outcome, Exception):
        return FileCheckResult(
            model_name=model.name,
            filename=file.filename,
            url=redacted_url,
            status=f"ERROR: {redact_url(str(outcome))}",
            content_type="",
            server_filename=None,
            content_length=None,
            ok=False,
            flag="request failed",
        )

    response = outcome
    content_type = response.headers.get("content-type", "")
    server_filename = parse_content_disposition(response.headers.get("content-disposition"))
    content_length_header = response.headers.get("content-length")
    content_length = int(content_length_header) if content_length_header else None

    is_html = content_type.split(";", 1)[0].strip().lower() == "text/html"
    ok = response.status_code in (200, 206) and not is_html

    flag: str | None = None
    if is_html:
        flag = "HTML response — auth/domain problem"
    elif response.status_code in _AUTH_RETRY_STATUSES:
        flag = "Unauthorized — check API token"
    elif response.status_code == 404:
        host = urlparse(file.url).netloc.lower()
        flag = (
            "404 on civitai.com — model may be NSFW; try civitai.red"
            if "civitai.com" in host
            else "404 Not Found"
        )

    return FileCheckResult(
        model_name=model.name,
        filename=file.filename,
        url=redacted_url,
        status=str(response.status_code),
        content_type=content_type,
        server_filename=server_filename,
        content_length=content_length,
        ok=ok,
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
    table.add_column("Flag", style="yellow")

    for r in report.results:
        style = "green" if r.ok else "red"
        table.add_row(
            r.model_name,
            r.filename,
            f"[{style}]{r.status}[/{style}]",
            r.content_type or "-",
            r.server_filename or "-",
            str(r.content_length) if r.content_length is not None else "-",
            r.flag or "",
        )

    console.print(table)
