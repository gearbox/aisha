"""Safe presentation helpers for source URLs."""

from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

_CIVITAI_HOSTS = ("civitai.com", "civitai.red", "civitai.green")


def sanitize_civitai_url_for_output(url: str) -> str:
    """Remove Civitai's query-token credential while preserving other query data.

    A signed non-Civitai URL may legitimately use a ``token`` query parameter,
    so it is left untouched.  ``parse_qsl`` retains the order and multiplicity
    of unrelated parameters, unlike a dict-based query parser.
    """
    parsed = urlsplit(url)
    host = (parsed.hostname or "").lower()
    if not any(host == domain or host.endswith(f".{domain}") for domain in _CIVITAI_HOSTS):
        return url

    retained = [
        (name, value)
        for name, value in parse_qsl(parsed.query, keep_blank_values=True)
        if name.lower() != "token"
    ]
    return urlunsplit(parsed._replace(query=urlencode(retained, doseq=True)))
