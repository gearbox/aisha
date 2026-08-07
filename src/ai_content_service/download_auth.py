"""Per-host download authentication policies.

Registry-driven so that adding or moving a provider domain is a data change,
not a code change. No call site branches on a host string.
"""

from __future__ import annotations

import hmac
import re
from dataclasses import dataclass
from enum import Enum
from http import HTTPStatus
from typing import TYPE_CHECKING, Final, TypeVar
from urllib.parse import parse_qs, urlencode, urlparse

import structlog

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable, Mapping

    from .config import Settings

log = structlog.get_logger()

AUTH_RETRY_STATUSES: Final = (HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN)

# Below this, the substring scan in assert_no_credential_egress and the value
# pass in redact_url are skipped -- a short token makes both misfire (MY-7).
# Exact bearer/query-value comparisons are unaffected at any length.
_MIN_SCANNABLE_SECRET_LEN: Final = 8

_R = TypeVar("_R")


class AuthTransport(str, Enum):
    """How a credential is attached to an outbound download request."""

    BEARER_HEADER = "bearer_header"
    # enum member name, not a credential
    QUERY_TOKEN = "query_token"  # noqa: S105
    NONE = "none"


def domain_matches(netloc: str, domains: Iterable[str]) -> bool:
    """Exact-or-subdomain host match; strips userinfo and port.

    Standalone so other host-eligibility checks (e.g. `HfXetTransport.can_handle`)
    can reuse the exact same rule instead of a second, possibly-diverging copy.
    """
    host = netloc.lower().rsplit("@", 1)[-1].split(":", 1)[0]
    return any(host == d or host.endswith(f".{d}") for d in domains)


@dataclass(frozen=True, slots=True)
class HostAuthPolicy:
    """Auth rules for one provider, keyed by a tuple of eligible domains."""

    name: str
    domains: tuple[str, ...]
    primary: AuthTransport
    fallback: AuthTransport | None

    def matches(self, netloc: str) -> bool:
        """Exact-or-subdomain host match; strips userinfo and port."""
        return domain_matches(netloc, self.domains)


def build_huggingface_policy(settings: Settings) -> HostAuthPolicy:
    """Construct the HF auth policy from *settings*.

    Split out of `build_registry` so `HfXetTransport` can build the exact
    same policy it uses for `can_handle` -- domains come from
    `settings.hf_domains`, not a hardcoded tuple, so a configured mirror is
    eligible for the HF token in both the httpx and hf_xet paths, not just
    routing.
    """
    return HostAuthPolicy(
        name="huggingface",
        domains=settings.hf_domains,
        primary=AuthTransport.BEARER_HEADER,
        fallback=None,
    )


def build_registry(settings: Settings) -> tuple[HostAuthPolicy, ...]:
    """Construct the HF and Civitai auth policies from *settings*."""
    huggingface_policy = build_huggingface_policy(settings)
    civitai_policy = HostAuthPolicy(
        name="civitai",
        domains=settings.civitai_domains,
        primary=AuthTransport.BEARER_HEADER,
        fallback=(
            AuthTransport.QUERY_TOKEN if settings.civitai_allow_query_token_fallback else None
        ),
    )
    log.info("auth.registry.civitai_domains", domains=list(civitai_policy.domains))
    return huggingface_policy, civitai_policy


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

    if not token or transport is AuthTransport.NONE:
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

    No *token* means no credential (E3) -- retrying an unauthenticated 401
    with a different transport for the same absent credential would just
    burn a second request without changing anything, so the fallback never
    fires when *token* is falsy.
    """
    transport = policy.primary if policy is not None else AuthTransport.NONE
    url_, headers = (
        apply_auth(policy, transport, url, base_headers, token)
        if policy is not None
        else (url, base_headers)
    )
    result = await send(url_, headers)
    status = status_of(result)

    if (
        allow_fallback
        and status in AUTH_RETRY_STATUSES
        and policy is not None
        and policy.fallback is not None
        and transport != policy.fallback
        and token
    ):
        transport = policy.fallback
        try:
            host = urlparse(url).netloc
        except ValueError:
            host = ""
        log.warning(
            "auth.query_fallback",
            policy=policy.name,
            host=host,
            status=status,
        )
        url_, headers = apply_auth(policy, transport, url, base_headers, token)
        result = await send(url_, headers)

    return result, transport


class CredentialEgressError(Exception):
    """A credential is bound for a request to a host its policy does not cover."""


@dataclass(frozen=True, slots=True)
class BoundCredential:
    """A token together with the policy that issued it.

    The pairing is the point: passing a flat tuple of token values made it
    impossible to tell whether a matched credential belonged to the provider
    it was being sent to (E4) — our Civitai token has no business on
    huggingface.co even though huggingface.co is a host we know.
    """

    policy: HostAuthPolicy
    token: str


def build_credentials(
    registry: tuple[HostAuthPolicy, ...],
    tokens: Mapping[str, str | None],
) -> tuple[BoundCredential, ...]:
    """Pair each policy with its configured token. Blank tokens are dropped."""
    credentials: list[BoundCredential] = []
    for policy in registry:
        token = tokens.get(policy.name)
        if not token:
            continue
        if len(token) < _MIN_SCANNABLE_SECRET_LEN:
            log.warning("auth.token.too_short", policy=policy.name, length=len(token))
        credentials.append(BoundCredential(policy, token))
    return tuple(credentials)


_SENSITIVE_QUERY_KEYS = ("token", "api_key", "access_token")
_REDACT_RE = re.compile(r"(?i)\b(" + "|".join(_SENSITIVE_QUERY_KEYS) + r")=[^&\s'\"]+")


def redact_url(url: str, secrets: Iterable[str] = ()) -> str:
    """Mask sensitive query-param values in *url* (or free text containing one).

    *secrets*, when given, are masked first by literal value — this catches a
    credential we hold regardless of what parameter name (or path segment) it
    appears under. The name-based regex pass then runs as a backstop for
    credentials we do *not* hold, e.g. a bundle's own embedded ``?token=``.
    Secrets shorter than `_MIN_SCANNABLE_SECRET_LEN` are skipped (MY-7a) — a
    short value is too likely to occur incidentally in unrelated text.

    Regex-based rather than a strict URL parse: this also runs on exception
    messages (e.g. httpx's ``HTTPStatusError`` text) that embed a URL inside
    a larger string. Must never raise — it runs exclusively in error paths,
    where an exception here would mask the original failure.
    """
    try:
        for secret in secrets:
            if secret and len(secret) >= _MIN_SCANNABLE_SECRET_LEN:
                url = url.replace(secret, "***")
        return _REDACT_RE.sub(lambda m: f"{m.group(1)}=***", url)
    except Exception:
        return "<redact_url error>"


def _constant_time_eq(a: str, b: str) -> bool:
    """``hmac.compare_digest`` requires matching str/bytes types and ASCII-only
    str operands; fall back to a plain compare rather than let a non-ASCII
    token turn into an unhandled TypeError (pitfall #2)."""
    try:
        return hmac.compare_digest(a, b)
    except TypeError:
        return a == b


def _header_value(headers: Mapping[str, str], name: str) -> str:
    return next((v for k, v in headers.items() if k.lower() == name.lower()), "")


def assert_no_credential_egress(
    url: str,
    headers: Mapping[str, str],
    credentials: Iterable[BoundCredential] = (),
) -> None:
    """Raise CredentialEgressError if one of our credentials is bound for a
    host **its own issuing policy** does not cover.

    Each credential is checked against its owner, not against whichever
    policy happens to match the destination: our Civitai token has no
    business on huggingface.co even though huggingface.co is a host we know
    (E4). Keys on credential *values*, not parameter names — a bundle's own
    embedded ``?token=``, and a provider's presigned redirect that happens to
    use the same parameter name, are not our credentials and are none of our
    business (E1/MY-4).

    D4 relies on httpx stripping Authorization across cross-origin redirects
    and on the original query string not being carried onto the redirect
    target. Both hold in httpx 0.28.1 (verified) but neither is a public API
    contract. This turns that assumption into an enforced, fail-loud invariant.
    """
    creds = [c for c in credentials if c.token]
    if not creds:
        return

    # `bearer` is header-derived, not url-derived, so it survives a urlparse
    # failure below -- a credential-bearing request must still be checked
    # (and can still raise) even when the URL itself fails to parse (MY-9a).
    bearer = _header_value(headers, "Authorization").removeprefix("Bearer ")
    try:
        parsed = urlparse(url)
        netloc = parsed.netloc
        query_values = [v for values in parse_qs(parsed.query).values() for v in values]
    except Exception:
        netloc = ""
        query_values = []

    def _matched(cred: BoundCredential) -> bool:
        return (
            _constant_time_eq(bearer, cred.token)
            or any(_constant_time_eq(v, cred.token) for v in query_values)
            or (len(cred.token) >= _MIN_SCANNABLE_SECRET_LEN and cred.token in url)
        )

    offenders = [c for c in creds if _matched(c) and not c.policy.matches(netloc)]
    if not offenders:
        return

    names = ", ".join(sorted({c.policy.name for c in offenders}))
    msg = f"credential bound for host {netloc!r}, outside its issuing policy domains ({names})"
    raise CredentialEgressError(msg)
