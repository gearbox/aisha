"""Credential-safe source URL helpers."""

from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


def strip_credential_query_params(url: str) -> str:
    """Ensure credential-bearing query parameters never leave this process.

    This applies to both network payloads and rendered artefacts. ``parse_qsl``
    retains the order and multiplicity of unrelated parameters, unlike a
    dict-based query parser.
    """
    parsed = urlsplit(url)
    retained = [
        (name, value)
        for name, value in parse_qsl(parsed.query, keep_blank_values=True)
        if name.lower() != "token"
    ]
    return urlunsplit(parsed._replace(query=urlencode(retained, doseq=True)))
