"""Pluggable download transport seam.

`ModelDownloader` never names a host; it asks `select_transport` for the
first registered transport whose `can_handle` matches a file's URL, then
falls through to the R2 cache and finally plain httpx. Registration order
is the composition root's business — see `build_transports` in
`downloader.py` — not this module's or the downloader's.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence
    from pathlib import Path

    ProgressCallback = Callable[[int, int], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class TransportRequest:
    """One file to fetch, addressed by its final on-disk destination."""

    url: str
    destination: Path  # final path, models/<target_subpath>/<filename>
    expected_sha256: str | None
    expected_size: int | None


@dataclass(frozen=True, slots=True)
class TransportResult:
    """What a transport actually delivered."""

    bytes_written: int
    transport: str


class DownloadTransport(Protocol):
    """A pluggable, host-specific way to fetch a file faster than plain httpx."""

    @property
    def name(self) -> str: ...

    def can_handle(self, url: str) -> bool: ...

    async def probe_digest(self, url: str) -> str | None:
        """Upstream's advertised sha256, or None when unavailable.

        Cheap metadata call only. None means "cannot tell" -- never a guess,
        because a wrong answer here silently reroutes or wastes a transfer.
        """
        ...

    async def fetch(
        self, request: TransportRequest, on_progress: ProgressCallback | None
    ) -> TransportResult: ...


class TransportUnavailableError(Exception):
    """The transport cannot run here -- caller should fall back (L6)."""


class TransportFetchError(Exception):
    """The transport ran and failed. Caller should fall back (L6)."""


def select_transport(transports: Sequence[DownloadTransport], url: str) -> DownloadTransport | None:
    """Return the first transport in *transports* whose `can_handle` matches *url*."""
    return next((t for t in transports if t.can_handle(url)), None)
