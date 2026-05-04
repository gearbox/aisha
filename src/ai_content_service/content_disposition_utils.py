"""Utilities for parsing HTTP Content-Disposition headers."""

from __future__ import annotations

from urllib.parse import unquote


def parse_content_disposition(header: str | None) -> str | None:
    """Parse filename from Content-Disposition header.

    Prefers filename*= (RFC 5987 UTF-8 encoded) over filename=.
    Intended for future use in download filename resolution.
    """
    if not header:
        return None

    for part in header.split(";"):
        part = part.strip()
        if part.lower().startswith("filename*="):
            value = part[len("filename*=") :]
            if "''" in value:
                _, _, encoded = value.partition("''")
                return unquote(encoded)

    for part in header.split(";"):
        part = part.strip()
        if part.lower().startswith("filename="):
            value = part[len("filename=") :].strip()
            if value.startswith('"') and value.endswith('"'):
                return value[1:-1]
            return value or None

    return None
