"""Factory for operation-scoped provisioning telemetry streams."""

from __future__ import annotations

from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .callback_client import CallbackClient
from .config import unwrap_secret
from .operation_telemetry import BatchRef, OperationTarget, OperationTelemetry
from .rate_estimator import ThroughputEstimator
from .telemetry_contract import OperationKind, new_id

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from .config import Settings


OPERATION_STATE_CAP = 32


@dataclass(frozen=True, slots=True)
class OperationDefaults:
    """Settings copied into each new operation without sharing mutable state."""

    progress_interval_seconds: float = 3.0
    progress_percent: float = 5.0
    eta_warmup_seconds: float = 5.0
    eta_warmup_bytes: int = 268_435_456
    ewma_alpha: float = 0.3
    secrets: tuple[str, ...] = ()


@dataclass(slots=True)
class OperationState:
    """Small shared state needed to resume a command's telemetry stream."""

    terminated: bool = False
    next_sequence: int = 0


class ProvisioningReporter:
    """Own callback transport and create isolated operation telemetry objects."""

    def __init__(
        self,
        client: CallbackClient,
        *,
        session_id: str,
        settings_defaults: OperationDefaults | None = None,
    ) -> None:
        self._client = client
        self._session_id = session_id
        self._settings_defaults = settings_defaults or OperationDefaults()
        self._operation_states: OrderedDict[str, OperationState] = OrderedDict()

    async def __aenter__(self) -> ProvisioningReporter:
        await self._client.__aenter__()
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.aclose()

    @classmethod
    def from_settings(cls, settings: Settings) -> ProvisioningReporter:
        """Build an operation factory from parsed settings."""
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
            settings_defaults=OperationDefaults(
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
        """Create an operation, resuming an existing command stream when needed."""
        defaults = self._settings_defaults
        resolved_operation_id = operation_id or new_id()
        state = self._state_for(resolved_operation_id)
        operation = OperationTelemetry(
            self._client,
            session_id=self._session_id,
            operation_id=resolved_operation_id,
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
            initial_sequence=state.next_sequence,
            state_observer=self,
        )
        async with operation:
            yield operation

    def is_terminated(self, operation_id: str) -> bool:
        """Return whether this retained operation already emitted a terminal event."""
        state = self._operation_states.get(operation_id)
        return state.terminated if state is not None else False

    def sequence_consumed(self, operation_id: str, next_sequence: int) -> None:
        """Record a sequence allocation made by an active telemetry object."""
        self._state_for(operation_id).next_sequence = next_sequence

    def terminal_emitted(self, operation_id: str) -> None:
        """Record the terminal event that protects against executor fallback duplicates."""
        self._state_for(operation_id).terminated = True

    def _state_for(self, operation_id: str) -> OperationState:
        state = self._operation_states.get(operation_id)
        if state is not None:
            return state
        state = OperationState()
        self._operation_states[operation_id] = state
        while len(self._operation_states) > OPERATION_STATE_CAP:
            self._operation_states.popitem(last=False)
        return state

    async def aclose(self) -> None:
        """Close the factory's shared callback transport."""
        await self._client.aclose()
