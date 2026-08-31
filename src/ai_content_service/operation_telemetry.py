"""Per-operation state and event-envelope construction for provisioning telemetry."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import structlog

from .provisioning_timing import sanitize_error
from .rate_estimator import ThroughputEstimator
from .telemetry_contract import (
    SCHEMA_VERSION,
    EtaBasis,
    OperationKind,
    OperationStatus,
    ProvisioningPhase,
    WorkUnit,
    new_id,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from .callback_client import CallbackClient
    from .downloader import DownloadProgress

log = structlog.get_logger()


@dataclass(frozen=True, slots=True)
class OperationTarget:
    """The bundle deployment targeted by an operation."""

    bundle: str
    bundle_version: str
    mode: str

    def as_dict(self) -> dict[str, str]:
        """Return the JSON-compatible target shape."""
        return {
            "bundle": self.bundle,
            "bundle_version": self.bundle_version,
            "mode": self.mode,
        }


@dataclass(frozen=True, slots=True)
class BatchRef:
    """Optional batch position attached by future operation producers."""

    batch_id: str
    index: int
    total: int

    def as_dict(self) -> dict[str, str | int]:
        """Return the JSON-compatible batch shape."""
        return {"batch_id": self.batch_id, "index": self.index, "total": self.total}


class OperationTelemetry:
    """Build and emit one isolated operation's v2 event stream."""

    def __init__(
        self,
        client: CallbackClient,
        *,
        session_id: str,
        operation_id: str,
        kind: OperationKind,
        target: OperationTarget | None = None,
        batch: BatchRef | None = None,
        estimator: ThroughputEstimator | None = None,
        progress_interval_seconds: float = 3.0,
        progress_percent: float = 5.0,
        secrets: Iterable[str] = (),
    ) -> None:
        self._client = client
        self._session_id = session_id
        self._operation_id = operation_id
        self._kind = kind
        self._target = target
        self._batch = batch
        self._estimator = estimator or ThroughputEstimator()
        self._progress_interval_seconds = progress_interval_seconds
        self._progress_percent = progress_percent
        self._secrets = tuple(secrets)
        self._started_monotonic = time.monotonic()
        self._started_at = _timestamp()
        self._phase_started_monotonic: float | None = None
        self._phase: ProvisioningPhase | None = None
        self._sequence = 0
        self._last_progress_monotonic: float | None = None
        self._last_progress_percent = -1.0
        self._terminal_emitted = False

    async def __aenter__(self) -> OperationTelemetry:
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        return None

    @property
    def events_emitted(self) -> int:
        """Number of event envelopes emitted by this operation."""
        return self._sequence

    async def started(self, plan: Mapping[str, object] | None, message: str = "") -> None:
        """Emit the non-terminal operation start event with its full plan."""
        await self._emit(
            status=OperationStatus.RUNNING,
            phase=None,
            plan=plan,
            summary=None,
            message=message,
            error=None,
            progress=None,
            retried=True,
        )

    async def begin_phase(self, phase: ProvisioningPhase, message: str = "") -> None:
        """Transition to a phase and ensure its first progress event is visible."""
        now = time.monotonic()
        self._phase = phase
        self._phase_started_monotonic = now
        self._last_progress_monotonic = None
        self._last_progress_percent = -1.0
        await self._emit(
            status=OperationStatus.RUNNING,
            phase=phase,
            plan=None,
            summary=None,
            message=message,
            error=None,
            progress=None,
            retried=False,
        )

    async def progress(self, download: DownloadProgress) -> None:
        """Emit throttled generic progress for the active download phase."""
        now = time.monotonic()
        self._estimator.observe(
            materialized_bytes=download.materialized_bytes_done,
            now_monotonic=now,
        )
        percent = (
            download.bytes_done / download.bytes_total * 100.0 if download.bytes_total > 0 else 0.0
        )
        final = download.bytes_total > 0 and download.bytes_done >= download.bytes_total
        time_ok = (
            self._last_progress_monotonic is None
            or now - self._last_progress_monotonic >= self._progress_interval_seconds
        )
        percent_ok = percent - self._last_progress_percent >= self._progress_percent
        if not (final or time_ok or percent_ok):
            return

        self._last_progress_monotonic = now
        self._last_progress_percent = percent
        rate = self._estimator.rate_bytes_per_second()
        remaining = (
            max(download.expected_materialized_bytes - download.materialized_bytes_done, 0)
            if download.expected_materialized_bytes is not None
            else None
        )
        eta = self._estimator.eta_seconds(remaining_materialized_bytes=remaining)
        await self._emit(
            status=OperationStatus.RUNNING,
            phase=self._phase,
            plan=None,
            summary=None,
            message="",
            error=None,
            progress={
                "work": {
                    "completed": download.bytes_done,
                    "total": download.bytes_total,
                    "unit": WorkUnit.BYTES.value,
                },
                "items": {
                    "completed": download.files_done,
                    "total": download.files_total,
                    "unit": WorkUnit.FILES.value,
                },
                "rate": (
                    {"value": round(rate, 3), "unit": "bytes_per_second"}
                    if rate is not None
                    else None
                ),
                "eta_seconds": round(eta, 3) if eta is not None else None,
                "eta_basis": EtaBasis.LIVE_THROUGHPUT.value if eta is not None else None,
            },
            retried=False,
        )

    async def succeeded(
        self,
        summary: Mapping[str, object] | None,
        message: str = "",
    ) -> None:
        """Emit a terminal success event once."""
        await self._terminal(
            status=OperationStatus.SUCCEEDED,
            summary=summary,
            message=message,
            error=None,
        )

    async def failed(
        self,
        error: str,
        summary: Mapping[str, object] | None = None,
    ) -> None:
        """Emit a terminal failure event once."""
        await self._terminal(
            status=OperationStatus.FAILED,
            summary=summary,
            message="",
            error=error,
        )

    async def _terminal(
        self,
        *,
        status: OperationStatus,
        summary: Mapping[str, object] | None,
        message: str,
        error: str | None,
    ) -> None:
        if self._terminal_emitted:
            log.warning(
                "operation.terminal.duplicate_suppressed",
                operation_id=self._operation_id,
                status=status.value,
            )
            return
        self._terminal_emitted = True
        delivered = await self._emit(
            status=status,
            phase=None,
            plan=None,
            summary=summary,
            message=message,
            error=error,
            progress=None,
            retried=True,
        )
        if not delivered:
            log.error(
                "operation.terminal.lost",
                operation_id=self._operation_id,
                status=status.value,
            )

    async def _emit(
        self,
        *,
        status: OperationStatus,
        phase: ProvisioningPhase | None,
        plan: Mapping[str, object] | None,
        summary: Mapping[str, object] | None,
        message: str,
        error: str | None,
        progress: Mapping[str, object] | None,
        retried: bool,
    ) -> bool:
        sequence = self._sequence
        self._sequence += 1
        now = time.monotonic()
        payload: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "event_id": new_id(),
            "session_id": self._session_id,
            "operation_id": self._operation_id,
            "operation_kind": self._kind.value,
            "batch": self._batch.as_dict() if self._batch is not None else None,
            "sequence": sequence,
            "target": self._target.as_dict() if self._target is not None else None,
            "status": status.value,
            "phase": phase.value if phase is not None else None,
            "started_at": self._started_at,
            "ts": _timestamp(),
            "elapsed_seconds": round(max(now - self._started_monotonic, 0.0), 3),
            "phase_elapsed_seconds": (
                round(max(now - self._phase_started_monotonic, 0.0), 3)
                if phase is not None and self._phase_started_monotonic is not None
                else None
            ),
            "progress": progress,
            "plan": dict(plan) if plan is not None else None,
            "summary": dict(summary) if summary is not None else None,
            "message": sanitize_error(message, secrets=self._secrets),
            "error": sanitize_error(error, secrets=self._secrets) if error is not None else None,
        }
        log.debug("operation.event", payload=payload)
        path = (
            f"/v1/internal/gpu-sessions/{self._session_id}/operations/{self._operation_id}/events"
        )
        if retried:
            return await self._client.post_retried(path, payload)
        await self._client.post_best_effort(path, payload)
        return True


def _timestamp() -> str:
    """Return a UTC timestamp in the envelope's compact RFC 3339 form."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
