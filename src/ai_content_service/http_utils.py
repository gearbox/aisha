"""Small helpers for parsing HTTP response headers safely."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping


def safe_int(value: str | None) -> int | None:
    """Parse *value* as an integer, returning ``None`` when it is invalid."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _header(headers: Mapping[str, str], name: str) -> str | None:
    """Get a header from either an ``httpx.Headers`` or a plain mapping."""
    return next(
        (value for key, value in headers.items() if key.lower() == name.lower()),
        None,
    )


def parse_content_length(headers: Mapping[str, str]) -> int | None:
    """Return ``Content-Length`` as an integer, or ``None`` if it is invalid."""
    return safe_int(_header(headers, "content-length"))


_CONTENT_RANGE_RE = re.compile(r"^bytes\s+\d+-\d+/(\d+|\*)$", re.IGNORECASE)


def parse_content_range_total(headers: Mapping[str, str]) -> int | None:
    """Return the total from ``Content-Range``, or ``None`` when unavailable."""
    value = _header(headers, "content-range")
    if value is None:
        return None
    match = _CONTENT_RANGE_RE.fullmatch(value.strip())
    if match is None or match.group(1) == "*":
        return None
    return safe_int(match.group(1))


def parse_retry_after(value: str | None, *, now: datetime, max_seconds: float) -> float | None:
    """Parse ``Retry-After`` delta-seconds or an HTTP-date.

    Returned delays are clamped to ``[0, max_seconds]``.  Invalid and absent
    values return ``None`` so callers can use their normal backoff strategy.
    """
    if value is None:
        return None

    value = value.strip()
    seconds = safe_int(value)
    if seconds is not None:
        return None if seconds < 0 else min(float(seconds), max_seconds)
    try:
        retry_at = parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError):
        return None

    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    delay = max(0.0, (retry_at - now).total_seconds())
    return min(delay, max_seconds)
