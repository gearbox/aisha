"""Provisioning progress reporter — emits phase/download callbacks to apex."""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone

import httpx

logger = logging.getLogger(__name__)

_THROTTLE_INTERVAL_S = 3.0
_THROTTLE_PERCENT = 5.0


class ProvisioningReporter:
    """Emits provisioning progress to apex over HTTP.

    All methods are async and best-effort: any failure is logged and swallowed —
    it never propagates to the caller.

    Construct via ``from_env()`` or ``disabled()``. When disabled every method
    is a no-op with zero overhead and zero network activity.
    """

    def __init__(
        self,
        *,
        session_id: str,
        callback_url: str,
        callback_token: str,
        enabled: bool = True,
        start_ts: float | None = None,
    ) -> None:
        self._enabled = enabled
        self._session_id = session_id
        self._callback_url = callback_url.rstrip("/")
        self._callback_token = callback_token
        self._start_ts: float = start_ts if start_ts is not None else time.monotonic()
        self._last_progress_ts: float = 0.0
        self._last_progress_pct: float = -1.0

    @classmethod
    def from_env(cls) -> ProvisioningReporter:
        """Construct from ACS_APEX_* env vars; disabled if any is missing/empty."""
        session_id = os.environ.get("ACS_APEX_SESSION_ID", "")
        callback_url = os.environ.get("ACS_APEX_CALLBACK_URL", "")
        callback_token = os.environ.get("ACS_APEX_CALLBACK_TOKEN", "")
        enabled = bool(session_id and callback_url and callback_token)
        return cls(
            session_id=session_id,
            callback_url=callback_url,
            callback_token=callback_token,
            enabled=enabled,
        )

    @classmethod
    def disabled(cls) -> ProvisioningReporter:
        """Return a no-op reporter (default for local/dev runs)."""
        return cls(session_id="", callback_url="", callback_token="", enabled=False)

    def _elapsed(self) -> int:
        return int(time.monotonic() - self._start_ts)

    def _build_payload(
        self,
        phase: str,
        message: str = "",
        download: dict[str, int] | None = None,
        error: str | None = None,
    ) -> dict[str, object]:
        return {
            "session_id": self._session_id,
            "phase": phase,
            "message": message,
            "download": download,
            "elapsed_seconds": self._elapsed(),
            "error": error,
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

    async def _post(self, payload: dict[str, object]) -> None:
        url = f"{self._callback_url}/v1/internal/gpu-sessions/{self._session_id}/provisioning"
        headers = {
            "Authorization": f"Bearer {self._callback_token}",
            "Content-Type": "application/json",
        }
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.post(url, json=payload, headers=headers)
        except Exception:
            logger.debug("Provisioning callback to %s failed", url, exc_info=True)

    async def phase(self, name: str, message: str = "") -> None:
        """Emit a phase-transition event."""
        if not self._enabled:
            return
        await self._post(self._build_payload(name, message))

    async def download_progress(
        self,
        bytes_done: int,
        bytes_total: int,
        files_done: int,
        files_total: int,
    ) -> None:
        """Emit a downloading progress event, throttled to ~3 s or ~5% change."""
        if not self._enabled:
            return

        now = time.monotonic()
        pct = (bytes_done / bytes_total * 100.0) if bytes_total > 0 else 0.0
        is_final = bytes_total > 0 and bytes_done >= bytes_total
        time_ok = (now - self._last_progress_ts) >= _THROTTLE_INTERVAL_S
        pct_ok = (pct - self._last_progress_pct) >= _THROTTLE_PERCENT

        if not (is_final or time_ok or pct_ok):
            return

        self._last_progress_ts = now
        self._last_progress_pct = pct
        download: dict[str, int] = {
            "bytes_done": bytes_done,
            "bytes_total": bytes_total,
            "files_done": files_done,
            "files_total": files_total,
        }
        await self._post(self._build_payload("downloading", download=download))

    async def ready(self) -> None:
        """Emit the terminal ready event."""
        if not self._enabled:
            return
        await self._post(self._build_payload("ready"))

    async def failed(self, error: str) -> None:
        """Emit the terminal failed event."""
        if not self._enabled:
            return
        await self._post(self._build_payload("failed", error=error))
