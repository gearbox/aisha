"""Credential providers for writes to the R2 model cache.

This module contains policy only. Upload orchestration stays in
``cache_service`` and command selection stays at the CLI composition root.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import httpx
import structlog

from .cache_keys import cache_key_for_sha256
from .r2_transfer import R2WriteCreds

log = structlog.get_logger()


@dataclass(frozen=True, slots=True)
class MintedCredentials:
    """The object key and write credentials authorized for one upload."""

    r2_key: str
    creds: R2WriteCreds


class CacheCredentialProvider(Protocol):
    """How a push obtains a write credential and records the result."""

    @property
    def name(self) -> str: ...

    def mint(
        self, *, sha256: str, filename: str, model_type: str, source_url: str
    ) -> MintedCredentials: ...

    def finalize(self, *, sha256: str, size_bytes: int) -> None: ...

    def close(self) -> None: ...


class CacheCredentialError(Exception):
    """Raised when a provider cannot mint or finalize credentials."""


class ApexCacheCredentialProvider:
    """Credential policy backed by Apex's admin model-cache endpoints."""

    def __init__(
        self,
        *,
        base_url: str,
        admin_token: str,
        client: httpx.Client | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._admin_token = admin_token
        self._client = client if client is not None else httpx.Client(timeout=30.0)
        self._owns_client = client is None

    @property
    def name(self) -> str:
        """Name shown to the operator before a batch begins."""
        return "apex"

    @property
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._admin_token}"}

    def _post_raw(self, path: str, payload: Mapping[str, object]) -> httpx.Response:
        """Issue an Apex request while leaving status policy to its operation."""
        return self._client.post(
            f"{self._base_url}{path}",
            json=payload,
            headers=self._headers,
        )

    def mint(
        self, *, sha256: str, filename: str, model_type: str, source_url: str
    ) -> MintedCredentials:
        """Ask Apex to mint the short-lived write credential for one file."""
        try:
            expected_key = cache_key_for_sha256(sha256)
            response = self._post_raw(
                "/v1/admin/model-cache/credentials",
                {
                    "sha256": sha256,
                    "filename": filename,
                    "model_type": model_type,
                    "source_url": source_url,
                },
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPStatusError as exc:
            raise CacheCredentialError(f"Apex credentials request failed: {exc}") from exc
        except httpx.HTTPError as exc:
            raise CacheCredentialError(f"Apex credentials request error: {exc}") from exc
        except ValueError as exc:
            raise CacheCredentialError(f"Apex credentials response invalid: {exc}") from exc

        try:
            if not isinstance(payload, Mapping):
                raise TypeError("response must be a mapping")
            r2_key = payload.get("r2_key")
            if not isinstance(r2_key, str):
                raise TypeError("r2_key must be a string")
            if r2_key != expected_key:
                raise ValueError(f"r2_key {r2_key!r} does not match expected cache key")

            raw_creds = payload.get("credentials")
            if not isinstance(raw_creds, Mapping):
                raise TypeError("credentials must be a mapping")
            access_key_id = raw_creds.get("access_key_id")
            secret_access_key = raw_creds.get("secret_access_key")
            session_token = raw_creds.get("session_token")
            if not isinstance(access_key_id, str) or not access_key_id:
                raise TypeError("credentials.access_key_id must be a non-empty string")
            if not isinstance(secret_access_key, str) or not secret_access_key:
                raise TypeError("credentials.secret_access_key must be a non-empty string")
            if session_token is not None and not isinstance(session_token, str):
                raise TypeError("credentials.session_token must be a string when present")
            return MintedCredentials(
                r2_key=expected_key,
                creds=R2WriteCreds(
                    access_key_id=access_key_id,
                    secret_access_key=secret_access_key,
                    session_token=session_token,
                ),
            )
        except (TypeError, ValueError) as exc:
            raise CacheCredentialError(f"Apex credentials response invalid: {exc}") from exc

    def finalize(self, *, sha256: str, size_bytes: int) -> None:
        """Tell Apex that the upload completed successfully."""
        try:
            response = self._post_raw(
                "/v1/admin/model-cache/finalize",
                {"sha256": sha256, "size_bytes": size_bytes},
            )
            if response.status_code in (409, 422):
                raise CacheCredentialError(
                    f"Apex finalize rejected ({response.status_code}): {response.text}"
                )
            response.raise_for_status()
        except CacheCredentialError:
            raise
        except httpx.HTTPStatusError as exc:
            raise CacheCredentialError(f"Apex finalize failed: {exc}") from exc
        except httpx.HTTPError as exc:
            raise CacheCredentialError(f"Apex finalize error: {exc}") from exc

    def close(self) -> None:
        """Close the HTTP client owned by this provider."""
        if self._owns_client:
            self._client.close()


class StaticCacheCredentialProvider:
    """Direct operator-supplied R2 write credentials for offline authoring."""

    def __init__(self, creds: R2WriteCreds) -> None:
        self._creds = creds

    @property
    def name(self) -> str:
        """Name shown to the operator before a batch begins."""
        return "direct"

    def mint(
        self, *, sha256: str, filename: str, model_type: str, source_url: str
    ) -> MintedCredentials:
        """Return the deterministic cache key and configured credentials."""
        del filename, model_type, source_url
        return MintedCredentials(r2_key=cache_key_for_sha256(sha256), creds=self._creds)

    def finalize(self, *, sha256: str, size_bytes: int) -> None:
        """Direct mode has no Apex catalog entry to finalize."""
        log.info("cache.push.finalize_skipped", sha256=sha256, size_bytes=size_bytes)

    def close(self) -> None:
        """Static credentials have no external resource to close."""
