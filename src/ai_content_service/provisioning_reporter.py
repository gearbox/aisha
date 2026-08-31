"""Factory for operation-scoped provisioning telemetry streams."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .callback_client import CallbackClient
from .operation_telemetry import BatchRef, OperationTarget, OperationTelemetry
from .rate_estimator import ThroughputEstimator
from .telemetry_contract import OperationKind, new_id

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from .config import Settings


@dataclass(frozen=True, slots=True)
class _OperationDefaults:
    """Settings copied into each new operation without sharing mutable state."""

    progress_interval_seconds: float = 3.0
    progress_percent: float = 5.0
    eta_warmup_seconds: float = 5.0
    eta_warmup_bytes: int = 268_435_456
    ewma_alpha: float = 0.3
    secrets: tuple[str, ...] = ()


class ProvisioningReporter:
    """Own callback transport and create isolated operation telemetry objects."""

    def __init__(
        self,
        client: CallbackClient,
        *,
        session_id: str,
        settings_defaults: _OperationDefaults | None = None,
    ) -> None:
        self._client = client
        self._session_id = session_id
        self._settings_defaults = settings_defaults or _OperationDefaults()

    async def __aenter__(self) -> ProvisioningReporter:
        await self._client.__aenter__()
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.aclose()

    @classmethod
    def from_settings(cls, settings: Settings) -> ProvisioningReporter:
        """Build an operation factory from parsed settings."""
        from .config import unwrap_secret

        secrets = tuple(
            value
            for secret in (
                settings.hf_token,
                settings.civitai_api_token,
                settings.cf_tunnel_token,
                settings.apex_callback_token,
                settings.r2_readonly_secret_access_key,
                settings.r2_write_secret_access_key,
                settings.apex_admin_token,
                settings.github_token,
            )
            if (value := unwrap_secret(secret))
        )
        return cls(
            CallbackClient.from_settings(settings),
            session_id=settings.apex_session_id,
            settings_defaults=_OperationDefaults(
                progress_interval_seconds=settings.telemetry_progress_interval_seconds,
                progress_percent=settings.telemetry_progress_percent,
                eta_warmup_seconds=settings.telemetry_eta_warmup_seconds,
                eta_warmup_bytes=settings.telemetry_eta_warmup_bytes,
                ewma_alpha=settings.telemetry_ewma_alpha,
                secrets=secrets,
            ),
        )

    @classmethod
    def disabled(cls) -> ProvisioningReporter:
        """Build a local-only reporter that still tracks all operation state."""
        return cls(CallbackClient.disabled(), session_id="")

    @asynccontextmanager
    async def operation(
        self,
        *,
        operation_id: str | None = None,
        kind: OperationKind,
        target: OperationTarget | None = None,
        batch: BatchRef | None = None,
    ) -> AsyncIterator[OperationTelemetry]:
        """Create one operation with fresh clocks, sequence, and estimator."""
        defaults = self._settings_defaults
        operation = OperationTelemetry(
            self._client,
            session_id=self._session_id,
            operation_id=operation_id or new_id(),
            kind=kind,
            target=target,
            batch=batch,
            estimator=ThroughputEstimator(
                alpha=defaults.ewma_alpha,
                warmup_seconds=defaults.eta_warmup_seconds,
                warmup_bytes=defaults.eta_warmup_bytes,
            ),
            progress_interval_seconds=defaults.progress_interval_seconds,
            progress_percent=defaults.progress_percent,
            secrets=defaults.secrets,
        )
        async with operation:
            yield operation

    async def aclose(self) -> None:
        """Close the factory's shared callback transport."""
        await self._client.aclose()
