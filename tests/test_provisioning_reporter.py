"""Tests for the operation factory boundary."""

from __future__ import annotations

import uuid

from ai_content_service.callback_client import CallbackClient
from ai_content_service.provisioning_reporter import OPERATION_STATE_CAP, ProvisioningReporter
from ai_content_service.telemetry_contract import OperationKind


async def test_operation_generates_uuid7_when_id_absent() -> None:
    reporter = ProvisioningReporter(CallbackClient.disabled(), session_id="session-1")

    async with reporter.operation(kind=OperationKind.BUNDLE_PROVISION) as operation:
        assert uuid.UUID(operation._operation_id).version == 7


async def test_operation_uses_supplied_id() -> None:
    reporter = ProvisioningReporter(CallbackClient.disabled(), session_id="session-1")

    async with reporter.operation(
        operation_id="operation-from-apex",
        kind=OperationKind.BUNDLE_PROVISION,
    ) as operation:
        assert operation._operation_id == "operation-from-apex"


async def test_reporter_operation_state_is_capped() -> None:
    reporter = ProvisioningReporter(CallbackClient.disabled(), session_id="session-1")

    for number in range(OPERATION_STATE_CAP + 1):
        async with reporter.operation(
            operation_id=f"operation-{number}", kind=OperationKind.BUNDLE_PROVISION
        ) as operation:
            await operation.started(plan=None)

    assert len(reporter._operation_states) == OPERATION_STATE_CAP
    assert "operation-0" not in reporter._operation_states
    assert not {"_sequence", "_phase", "_last_progress_monotonic"} & reporter.__dict__.keys()


async def test_reporter_operation_resumes_the_next_sequence_after_a_prior_event() -> None:
    reporter = ProvisioningReporter(CallbackClient.disabled(), session_id="session-1")

    async with reporter.operation(
        operation_id="operation", kind=OperationKind.BUNDLE_PROVISION
    ) as first:
        await first.started(plan=None)
    async with reporter.operation(
        operation_id="operation", kind=OperationKind.BUNDLE_PROVISION
    ) as second:
        await second.failed("later failure")

    assert first._sequence == 1
    assert second._sequence == 2
    assert reporter.is_terminated("operation")
