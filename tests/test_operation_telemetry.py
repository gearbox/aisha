"""Tests for operation-scoped event state and envelope invariants."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ai_content_service.callback_client import CallbackClient
from ai_content_service.downloader import DownloadProgress
from ai_content_service.operation_telemetry import OperationTelemetry
from ai_content_service.rate_estimator import ThroughputEstimator
from ai_content_service.telemetry_contract import OperationKind, ProvisioningPhase

if TYPE_CHECKING:
    from collections.abc import Mapping


class _CapturingClient:
    def __init__(self, *, retried_result: bool = True) -> None:
        self.enabled = True
        self.retried_result = retried_result
        self.best_effort: list[dict[str, object]] = []
        self.retried: list[dict[str, object]] = []

    async def post_best_effort(self, _path: str, payload: Mapping[str, object]) -> None:
        self.best_effort.append(dict(payload))

    async def post_retried(self, _path: str, payload: Mapping[str, object]) -> bool:
        self.retried.append(dict(payload))
        return self.retried_result


def _progress(*, done: int, materialized: int, expected: int | None = 100) -> DownloadProgress:
    return DownloadProgress(
        bytes_done=done,
        bytes_total=100,
        files_done=1 if done == 100 else 0,
        files_total=1,
        materialized_bytes_done=materialized,
        reused_bytes_done=done - materialized,
        expected_materialized_bytes=expected,
    )


async def test_sequence_starts_at_zero_and_increments_per_emitted_event() -> None:
    client = _CapturingClient()
    operation = OperationTelemetry(
        client,  # type: ignore[arg-type]
        session_id="session",
        operation_id="operation",
        kind=OperationKind.BUNDLE_PROVISION,
    )

    await operation.started(plan=None)
    await operation.begin_phase(ProvisioningPhase.MODELS)
    await operation.succeeded(summary=None)

    events = [*client.retried, *client.best_effort]
    sequences: list[int] = []
    for event in events:
        sequence = event["sequence"]
        assert isinstance(sequence, int)
        sequences.append(sequence)
    assert sorted(sequences) == [0, 1, 2]


async def test_throttled_progress_does_not_consume_sequence() -> None:
    client = _CapturingClient()
    operation = OperationTelemetry(
        client,  # type: ignore[arg-type]
        session_id="session",
        operation_id="operation",
        kind=OperationKind.BUNDLE_PROVISION,
        progress_interval_seconds=60,
    )

    await operation.begin_phase(ProvisioningPhase.MODELS)
    await operation.progress(_progress(done=1, materialized=1))
    await operation.progress(_progress(done=2, materialized=2))

    assert operation.events_emitted == 2


async def test_terminal_retry_preserves_sequence_and_event_id() -> None:
    client = _CapturingClient()
    operation = OperationTelemetry(
        client,  # type: ignore[arg-type]
        session_id="session",
        operation_id="operation",
        kind=OperationKind.BUNDLE_PROVISION,
    )
    await operation.started(plan=None)
    await operation.failed("failure")

    terminal = client.retried[-1]
    assert terminal["sequence"] == 1
    assert terminal["event_id"]


async def test_second_terminal_call_is_suppressed() -> None:
    client = _CapturingClient()
    operation = OperationTelemetry(
        client,  # type: ignore[arg-type]
        session_id="session",
        operation_id="operation",
        kind=OperationKind.BUNDLE_PROVISION,
    )
    await operation.succeeded(summary=None)
    await operation.failed("too late")

    assert len(client.retried) == 1


async def test_begin_phase_resets_phase_clock_and_throttle() -> None:
    client = _CapturingClient()
    operation = OperationTelemetry(
        client,  # type: ignore[arg-type]
        session_id="session",
        operation_id="operation",
        kind=OperationKind.BUNDLE_PROVISION,
        progress_interval_seconds=60,
    )
    await operation.begin_phase(ProvisioningPhase.MODELS)
    await operation.progress(_progress(done=1, materialized=1))
    await operation.begin_phase(ProvisioningPhase.WORKFLOW)
    await operation.progress(_progress(done=2, materialized=2))

    assert operation.events_emitted == 4


async def test_two_operations_share_no_state() -> None:
    client = _CapturingClient()
    one = OperationTelemetry(
        client,  # type: ignore[arg-type]
        session_id="session",
        operation_id="one",
        kind=OperationKind.BUNDLE_PROVISION,
    )
    two = OperationTelemetry(
        client,  # type: ignore[arg-type]
        session_id="session",
        operation_id="two",
        kind=OperationKind.BUNDLE_PROVISION,
    )
    await one.started(plan=None)
    await two.started(plan=None)

    assert [event["sequence"] for event in client.retried] == [0, 0]


async def test_disabled_client_still_tracks_locally() -> None:
    operation = OperationTelemetry(
        CallbackClient.disabled(),
        session_id="",
        operation_id="operation",
        kind=OperationKind.BUNDLE_PROVISION,
    )

    await operation.started(plan=None)
    await operation.begin_phase(ProvisioningPhase.MODELS)
    await operation.progress(_progress(done=10, materialized=10))

    assert operation.events_emitted == 3


async def test_error_and_message_are_sanitized() -> None:
    client = _CapturingClient()
    operation = OperationTelemetry(
        client,  # type: ignore[arg-type]
        session_id="session",
        operation_id="operation",
        kind=OperationKind.BUNDLE_PROVISION,
        secrets=("secret-value",),
    )
    await operation.started(plan=None, message="secret-value")
    await operation.failed("https://example.test/?token=secret-value")

    assert "secret-value" not in str(client.retried)


async def test_eta_fields_are_both_null_or_both_set() -> None:
    client = _CapturingClient()
    operation = OperationTelemetry(
        client,  # type: ignore[arg-type]
        session_id="session",
        operation_id="operation",
        kind=OperationKind.BUNDLE_PROVISION,
        estimator=ThroughputEstimator(warmup_seconds=0, warmup_bytes=0),
    )
    await operation.begin_phase(ProvisioningPhase.MODELS)
    await operation.progress(_progress(done=1, materialized=1, expected=None))

    progress = client.best_effort[-1]["progress"]
    assert isinstance(progress, dict)
    assert progress["eta_seconds"] is None
    assert progress["eta_basis"] is None
