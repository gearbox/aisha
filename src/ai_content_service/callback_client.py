"""Best-effort HTTP transport for Apex operation events."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import httpx
import structlog
from tenacity import AsyncRetrying, RetryError, retry_if_exception_type, stop_after_attempt
from tenacity.wait import wait_exponential

from .config import unwrap_secret

if TYPE_CHECKING:
    from .config import Settings

log = structlog.get_logger()


class _RetryableResponseError(Exception):
    """An unsuccessful response that is safe to retry."""


class CallbackClient:
    """Transport-only callback client with lazy HTTP client construction."""

    def __init__(self, *, base_url: str, token: str, enabled: bool, timeout: float = 5.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._token = token
        self._enabled = enabled
        self._timeout = timeout
        self._callback_ok = True
        self._logged_ok = False
        self._client: httpx.AsyncClient | None = None
        self._closed = False

    @property
    def enabled(self) -> bool:
        """Whether requests are configured for delivery."""
        return self._enabled

    async def __aenter__(self) -> CallbackClient:
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close the lazily-created HTTP client, if any."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        self._closed = True

    @classmethod
    def from_settings(cls, settings: Settings) -> CallbackClient:
        """Build a configured transport only when all callback fields exist."""
        token = unwrap_secret(settings.apex_callback_token)
        enabled = bool(settings.apex_session_id and settings.apex_callback_url and token)
        return cls(
            base_url=settings.apex_callback_url,
            token=token or "",
            enabled=enabled,
            timeout=settings.agent_claim_timeout_seconds,
        )

    @classmethod
    def disabled(cls) -> CallbackClient:
        """Build a local-only transport that never creates an HTTP client."""
        return cls(base_url="", token="", enabled=False)

    async def post_best_effort(self, path: str, payload: Mapping[str, object]) -> None:
        """Perform one POST and swallow every delivery failure."""
        if not self._enabled:
            return
        try:
            await self._post(path, payload)
        except Exception:
            # _post already recorded a safe diagnosis; event delivery must not
            # influence provisioning work.
            return

    async def post_retried(self, path: str, payload: Mapping[str, object]) -> bool:
        """POST with bounded retry for transport failures and 5xx responses."""
        if not self._enabled:
            return True
        try:
            async for attempt in self._retrying():
                with attempt:
                    return await self._post(path, payload)
        except RetryError:
            return False
        return False

    def _retrying(self) -> AsyncRetrying:
        """Return the shared bounded retry policy for callback POSTs."""
        return AsyncRetrying(
            retry=retry_if_exception_type((httpx.TransportError, _RetryableResponseError)),
            stop=stop_after_attempt(5),
            wait=wait_exponential(multiplier=1, max=30),
            reraise=False,
        )

    async def post_json(
        self, path: str, payload: Mapping[str, object]
    ) -> tuple[int, Mapping[str, object] | None]:
        """POST and return the status plus its decoded JSON object, if any.

        Claims need their response body, unlike event delivery.  They retain
        the exact bounded retry policy used for terminal operation callbacks,
        while making transport and retry exhaustion non-exceptional to the
        polling loop.
        """
        if not self._enabled:
            return 204, None
        try:
            async for attempt in self._retrying():
                with attempt:
                    return await self._post_json(path, payload)
        except RetryError:
            return 0, None
        return 0, None

    async def claim_command(
        self, session_id: str, agent_id: str
    ) -> tuple[int, Mapping[str, object] | None]:
        """Claim one queued command for the agent's Apex GPU session."""
        return await self.post_json(
            f"/v1/internal/gpu-sessions/{session_id}/commands/claim",
            {"agent_id": agent_id, "schema_version": 2},
        )

    async def _post(self, path: str, payload: Mapping[str, object]) -> bool:
        """Send one request; false denotes a non-retryable delivery failure."""
        url = f"{self._base_url}{path}"
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }
        if self._client is None:
            if self._closed:
                log.warning("provisioning.client.recreated_after_close")
            self._client = httpx.AsyncClient(timeout=self._timeout)
            self._closed = False
        try:
            response = await self._client.post(url, json=payload, headers=headers)
        except httpx.TransportError:
            self._record_failure(url, "request error", exc_info=True)
            raise
        except Exception:
            self._record_failure(url, "request error", exc_info=True)
            return False

        content_type = response.headers.get("content-type", "").lower()
        if response.is_success and content_type.startswith("application/json"):
            self._record_success(url)
            return True

        detail = self._diagnose(response, content_type)
        retryable = 500 <= response.status_code < 600
        self._record_failure(url, detail, error=not retryable)
        if retryable:
            raise _RetryableResponseError(detail)
        return False

    async def _post_json(
        self, path: str, payload: Mapping[str, object]
    ) -> tuple[int, Mapping[str, object] | None]:
        """Send one claim request and retain a JSON response object when present."""
        url = f"{self._base_url}{path}"
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }
        if self._client is None:
            if self._closed:
                log.warning("provisioning.client.recreated_after_close")
            self._client = httpx.AsyncClient(timeout=self._timeout)
            self._closed = False
        try:
            response = await self._client.post(url, json=payload, headers=headers)
        except httpx.TransportError:
            self._record_failure(url, "request error", exc_info=True)
            raise
        except Exception:
            self._record_failure(url, "request error", exc_info=True)
            return 0, None

        if response.status_code == 204:
            self._record_success(url)
            return 204, None

        content_type = response.headers.get("content-type", "").lower()
        retryable = 500 <= response.status_code < 600
        if retryable:
            detail = self._diagnose(response, content_type)
            self._record_failure(url, detail)
            raise _RetryableResponseError(detail)
        if not response.is_success:
            self._record_failure(url, self._diagnose(response, content_type), error=True)
            return response.status_code, None
        if not content_type.startswith("application/json"):
            self._record_failure(url, self._diagnose(response, content_type), error=True)
            return response.status_code, None
        try:
            decoded = response.json()
        except ValueError:
            self._record_failure(url, self._diagnose(response, content_type), error=True)
            return response.status_code, None
        if not isinstance(decoded, Mapping):
            self._record_failure(url, "HTTP 200 JSON response is not an object", error=True)
            return response.status_code, None
        self._record_success(url)
        return response.status_code, decoded

    @staticmethod
    def _diagnose(response: httpx.Response, content_type: str) -> str:
        """Return the established safe diagnosis for a callback response."""
        code = response.status_code
        if response.is_success:
            return (
                f"HTTP {code} with content-type={content_type or 'unknown'!r} (not JSON) — "
                "APEX_CALLBACK_URL may not point at the Apex API "
                "(non-API response; e.g. a frontend/static host or proxy)"
            )
        if code == 405:
            return (
                "HTTP 405 Method Not Allowed — URL may point at a static host (e.g. Cloudflare "
                "Pages/frontend); set APEX_CALLBACK_URL to the Apex API origin"
            )
        location = response.headers.get("location", "")
        if (response.is_redirect or code in (401, 403)) and (
            "cloudflareaccess.com" in location or "text/html" in content_type
        ):
            return (
                f"HTTP {code} — endpoint appears to be behind an auth proxy (e.g. Cloudflare "
                "Access); bypass it for /v1/internal/* or send service-token headers"
            )
        return f"HTTP {code}"

    def _record_success(self, url: str) -> None:
        if not self._logged_ok:
            log.info("provisioning.callback.reachable", url=url)
            self._logged_ok = True
        self._callback_ok = True

    def _record_failure(
        self,
        url: str,
        detail: str,
        *,
        exc_info: bool = False,
        error: bool = False,
    ) -> None:
        if error:
            log.error("provisioning.callback.failed", url=url, detail=detail, exc_info=exc_info)
            self._callback_ok = False
            return
        if self._callback_ok:
            log.warning(
                "provisioning.callback.failed",
                url=url,
                detail=detail,
                exc_info=exc_info,
            )
            self._callback_ok = False
        else:
            log.debug("provisioning.callback.failed", url=url, detail=detail, exc_info=exc_info)
