"""Per-host download authentication policies.

Registry-driven so that adding or moving a provider domain is a data change,
not a code change. No call site branches on a host string.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from http import HTTPStatus
from typing import TYPE_CHECKING, Final, TypeVar
from urllib.parse import parse_qs, urlencode, urlparse

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

    from .config import Settings

AUTH_RETRY_STATUSES: Final = (HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN)

_R = TypeVar("_R")


class AuthTransport(str, Enum):
    """How a credential is attached to an outbound download request."""

    BEARER_HEADER = "bearer_header"
    # enum member name, not a credential
    QUERY_TOKEN = "query_token"  # noqa: S105
    NONE = "none"


@dataclass(frozen=True, slots=True)
class HostAuthPolicy:
    """Auth rules for one provider, keyed by a tuple of eligible domains."""

    name: str
    domains: tuple[str, ...]
    primary: AuthTransport
    fallback: AuthTransport | None

    def matches(self, netloc: str) -> bool:
        """Exact-or-subdomain host match; strips userinfo and port."""
        host = netloc.lower().rsplit("@", 1)[-1].split(":", 1)[0]
        return any(host == d or host.endswith(f".{d}") for d in self.domains)


def build_registry(settings: Settings) -> tuple[HostAuthPolicy, ...]:
    """Construct the HF and Civitai auth policies from *settings*."""
    return (
        HostAuthPolicy(
            name="huggingface",
            domains=("huggingface.co", "hf.co"),
            primary=AuthTransport.BEARER_HEADER,
            fallback=None,
        ),
        HostAuthPolicy(
            name="civitai",
            domains=settings.civitai_domains,
            primary=AuthTransport.BEARER_HEADER,
            fallback=(
                AuthTransport.QUERY_TOKEN if settings.civitai_allow_query_token_fallback else None
            ),
        ),
    )


def resolve_policy(registry: tuple[HostAuthPolicy, ...], url: str) -> HostAuthPolicy | None:
    """Return the first policy in *registry* whose domains match *url*'s host."""
    netloc = urlparse(url).netloc
    return next((policy for policy in registry if policy.matches(netloc)), None)


def apply_auth(
    policy: HostAuthPolicy,
    transport: AuthTransport,
    url: str,
    headers: dict[str, str],
    token: str | None,
) -> tuple[str, dict[str, str]]:
    """Attach *token* to *url*/*headers* per *transport*. The only place a credential is attached.

    *transport* must be either ``policy.primary`` or ``policy.fallback`` —
    this is a deliberate guard so a caller cannot accidentally invent a
    transport the policy never declared.
    """
    if transport not in (policy.primary, policy.fallback):
        msg = f"{policy.name}: transport {transport!r} is not configured for this policy"
        raise ValueError(msg)

    if token is None or transport is AuthTransport.NONE:
        return url, headers

    if transport is AuthTransport.BEARER_HEADER:
        return url, {**headers, "Authorization": f"Bearer {token}"}

    # AuthTransport.QUERY_TOKEN
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    query["token"] = [token]
    new_query = urlencode(query, doseq=True)
    return parsed._replace(query=new_query).geturl(), headers


async def attempt_with_auth(
    policy: HostAuthPolicy | None,
    token: str | None,
    url: str,
    base_headers: dict[str, str],
    send: Callable[[str, dict[str, str]], Awaitable[_R]],
    status_of: Callable[[_R], int],
    *,
    allow_fallback: bool = True,
) -> tuple[_R, AuthTransport]:
    """Send with the policy's primary transport; on 401/403 retry once with
    the fallback. Returns the response and the transport that produced it.
    Never issues more than two attempts.
    """
    transport = policy.primary if policy is not None else AuthTransport.NONE
    url_, headers = (
        apply_auth(policy, transport, url, base_headers, token)
        if policy is not None
        else (url, base_headers)
    )
    result = await send(url_, headers)

    if (
        allow_fallback
        and status_of(result) in AUTH_RETRY_STATUSES
        and policy is not None
        and policy.fallback is not None
        and transport != policy.fallback
    ):
        transport = policy.fallback
        url_, headers = apply_auth(policy, transport, url, base_headers, token)
        result = await send(url_, headers)

    return result, transport


class CredentialEgressError(Exception):
    """A credential is bound for a request to a host its policy does not cover."""


_SENSITIVE_QUERY_KEYS = ("token", "api_key", "access_token")
_REDACT_RE = re.compile(r"(?i)\b(" + "|".join(_SENSITIVE_QUERY_KEYS) + r")=[^&\s'\"]+")


def redact_url(url: str) -> str:
    """Mask sensitive query-param values in *url* (or free text containing one).

    Regex-based rather than a strict URL parse: this also runs on exception
    messages (e.g. httpx's ``HTTPStatusError`` text) that embed a URL inside
    a larger string. Must never raise — it runs exclusively in error paths,
    where an exception here would mask the original failure.
    """
    try:
        return _REDACT_RE.sub(lambda m: f"{m.group(1)}=***", url)
    except Exception:
        return "<redact_url error>"


def assert_no_credential_egress(
    policy: HostAuthPolicy | None, url: str, headers: Mapping[str, str]
) -> None:
    """Raise CredentialEgressError if a credential is bound for a host the
    policy does not cover.

    D4 relies on httpx stripping Authorization across cross-origin redirects,
    and on the original query string not being carried onto the redirect
    target. Both hold in httpx 0.28.1 (verified), but neither is part of
    httpx's public API contract. This turns that assumption into an enforced,
    fail-loud invariant.
    """
    try:
        parsed = urlparse(url)
        has_auth_header = any(k.lower() == "authorization" for k in headers)
        query_keys = {k.lower() for k in parse_qs(parsed.query)}
        has_sensitive_query = bool(query_keys & set(_SENSITIVE_QUERY_KEYS))
        netloc = parsed.netloc
    except Exception:
        return

    if not (has_auth_header or has_sensitive_query):
        return

    if policy is not None and policy.matches(netloc):
        return

    policy_name = policy.name if policy is not None else "any known"
    msg = f"credential bound for host {netloc!r}, which is outside the {policy_name} policy domains"
    raise CredentialEgressError(msg)
