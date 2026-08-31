"""Tests for the operation factory boundary."""

from __future__ import annotations

import uuid

from ai_content_service.callback_client import CallbackClient
from ai_content_service.provisioning_reporter import ProvisioningReporter
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


async def test_reporter_holds_no_per_operation_state() -> None:
    reporter = ProvisioningReporter(CallbackClient.disabled(), session_id="session-1")

    async with reporter.operation(kind=OperationKind.BUNDLE_PROVISION) as first:
        await first.started(plan=None)
    async with reporter.operation(kind=OperationKind.BUNDLE_PROVISION) as second:
        await second.started(plan=None)

    assert first.events_emitted == second.events_emitted == 1
    assert not {"_sequence", "_phase", "_last_progress_monotonic"} & reporter.__dict__.keys()
