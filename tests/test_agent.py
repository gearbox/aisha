"""Tests for provisioning-agent polling, backoff, and stop behavior."""

from __future__ import annotations

import asyncio
import signal
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

from ai_content_service.agent import ProvisioningAgent
from ai_content_service.callback_client import CallbackClient
from ai_content_service.command_executor import CommandExecutor
from ai_content_service.config import Settings
from ai_content_service.provisioning_reporter import ProvisioningReporter
from ai_content_service.telemetry_contract import OperationKind

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import pytest

    from ai_content_service.agent_contract import Command


def _body() -> dict[str, object]:
    return {
        "command_id": "command-1",
        "operation_id": "operation-1",
        "kind": "comfyui_restart",
        "batch": None,
        "payload": {"node_class": None},
    }


class _Client(CallbackClient):
    def __init__(self, responses: Sequence[tuple[int, dict[str, object] | None]]) -> None:
        super().__init__(base_url="", token="", enabled=False)
        self.claim_command_mock = AsyncMock(side_effect=responses)

    async def claim_command(
        self, session_id: str, agent_id: str
    ) -> tuple[int, Mapping[str, object] | None]:
        return await self.claim_command_mock(session_id, agent_id)


class _Executor(CommandExecutor):
    def __init__(self, on_execute: object | None = None) -> None:
        self.execute_mock = AsyncMock(side_effect=on_execute)
        self.report_unparseable_mock = AsyncMock()

    async def execute(self, command: Command) -> bool:
        await self.execute_mock(command)
        return True

    async def report_unparseable(
        self, *, operation_id: str, kind: OperationKind, detail: str
    ) -> None:
        await self.report_unparseable_mock(
            operation_id=operation_id,
            kind=kind,
            detail=detail,
        )


class _EventClient:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    async def post_best_effort(self, _path: str, payload: object) -> None:
        self.events.append(dict(payload))  # type: ignore[arg-type]

    async def post_retried(self, _path: str, payload: object) -> bool:
        self.events.append(dict(payload))  # type: ignore[arg-type]
        return True


async def test_claims_again_immediately_after_a_command() -> None:
    client = _Client([(200, _body()), (204, None)])
    reporter = ProvisioningReporter(CallbackClient.disabled(), session_id="session")
    executor = _Executor()
    agent = ProvisioningAgent(
        Settings(apex_session_id="session"), client=client, reporter=reporter, executor=executor
    )

    async def sleep(interval: float) -> None:
        del interval
        agent.request_stop()

    agent._sleep_with_jitter = sleep
    await agent.run()

    assert executor.execute_mock.await_count == 1
    assert client.claim_command_mock.await_count == 2


async def test_idle_claim_sleeps_with_jitter() -> None:
    client = _Client([(204, None)])
    reporter = ProvisioningReporter(CallbackClient.disabled(), session_id="session")
    agent = ProvisioningAgent(Settings(apex_session_id="session"), client=client, reporter=reporter)
    delays: list[float] = []

    async def sleep(interval: float) -> None:
        delays.append(interval)
        agent.request_stop()

    agent._sleep_with_jitter = sleep
    await agent.run()

    assert 2.5 <= delays[0] <= 5.0


async def test_backoff_grows_then_resets_on_204() -> None:
    client = _Client([(0, None), (0, None), (204, None)])
    reporter = ProvisioningReporter(CallbackClient.disabled(), session_id="session")
    agent = ProvisioningAgent(Settings(apex_session_id="session"), client=client, reporter=reporter)
    intervals: list[float] = []

    async def capture(interval: float) -> None:
        intervals.append(interval)
        if len(intervals) == 3:
            agent.request_stop()

    agent._sleep_with_jitter = capture
    await agent.run()

    assert intervals == [5.0, 10.0, 5.0]


async def test_restart_parse_error_is_reported_as_comfyui_restart() -> None:
    event_client = _EventClient()
    reporter = ProvisioningReporter(event_client, session_id="session")  # type: ignore[arg-type]
    agent = ProvisioningAgent(Settings(apex_session_id="session"), reporter=reporter)

    await agent._execute_claim(
        {
            "command_id": "command-1",
            "operation_id": "operation-1",
            "kind": "comfyui_restart",
            "batch": None,
            "payload": {"node_class": ""},
        }
    )

    assert [event["operation_kind"] for event in event_client.events] == [
        "comfyui_restart",
        "comfyui_restart",
    ]


async def test_removal_parse_error_is_reported_as_bundle_removal() -> None:
    event_client = _EventClient()
    reporter = ProvisioningReporter(event_client, session_id="session")  # type: ignore[arg-type]
    agent = ProvisioningAgent(Settings(apex_session_id="session"), reporter=reporter)

    await agent._execute_claim(
        {
            "command_id": "command-1",
            "operation_id": "operation-1",
            "kind": "bundle_removal",
            "batch": None,
            "payload": {},
        }
    )

    assert [event["operation_kind"] for event in event_client.events] == [
        "bundle_removal",
        "bundle_removal",
    ]


async def test_agent_delegates_unparseable_reporting_to_the_executor() -> None:
    executor = _Executor()
    reporter = ProvisioningReporter(CallbackClient.disabled(), session_id="session")
    agent = ProvisioningAgent(Settings(), reporter=reporter, executor=executor)

    await agent._execute_claim(
        {
            "command_id": "command-1",
            "operation_id": "operation-1",
            "kind": "comfyui_restart",
            "batch": None,
            "payload": {"node_class": ""},
        }
    )

    executor.report_unparseable_mock.assert_awaited_once_with(
        operation_id="operation-1",
        kind=OperationKind.COMFYUI_RESTART,
        detail="payload field 'node_class' must be a non-empty string or null",
    )
    executor.execute_mock.assert_not_awaited()


async def test_unknown_kind_is_logged_without_a_fabricated_operation_kind() -> None:
    event_client = _EventClient()
    reporter = ProvisioningReporter(event_client, session_id="session")  # type: ignore[arg-type]
    agent = ProvisioningAgent(Settings(apex_session_id="session"), reporter=reporter)

    with patch("ai_content_service.agent.log.error") as error:
        await agent._execute_claim({"operation_id": "operation-1", "kind": "nope"})

    error.assert_called_once()
    assert event_client.events == []


def test_agent_id_defaults_to_session_and_hostname(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ai_content_service.agent.socket.gethostname", lambda: "gpu-1")

    agent = ProvisioningAgent(Settings(apex_session_id="session"))

    assert agent.agent_id == "session:gpu-1"


async def test_command_claimed_during_stop_is_still_executed() -> None:
    client = _Client([])
    reporter = ProvisioningReporter(CallbackClient.disabled(), session_id="session")
    executor = _Executor()
    agent = ProvisioningAgent(
        Settings(apex_session_id="session"), client=client, reporter=reporter, executor=executor
    )

    async def claim_then_stop(
        _session_id: str, _agent_id: str
    ) -> tuple[int, Mapping[str, object] | None]:
        agent.request_stop()
        return 200, _body()

    client.claim_command_mock.side_effect = claim_then_stop
    await agent.run()

    assert executor.execute_mock.await_count == 1
    assert client.claim_command_mock.await_count == 1


async def test_request_stop_finishes_the_in_flight_command() -> None:
    client = _Client([(200, _body())])
    reporter = ProvisioningReporter(CallbackClient.disabled(), session_id="session")
    executor = _Executor()
    agent = ProvisioningAgent(
        Settings(apex_session_id="session"), client=client, reporter=reporter, executor=executor
    )

    async def stop_after_execute(_command: Command) -> None:
        agent.request_stop()

    executor.execute_mock.side_effect = stop_after_execute
    await agent.run()

    assert executor.execute_mock.await_count == 1
    assert client.claim_command_mock.await_count == 1


async def test_stop_before_a_claim_returns_without_claiming() -> None:
    client = _Client([(200, _body())])
    reporter = ProvisioningReporter(CallbackClient.disabled(), session_id="session")
    agent = ProvisioningAgent(Settings(apex_session_id="session"), client=client, reporter=reporter)
    agent.request_stop()

    await agent.run()

    client.claim_command_mock.assert_not_awaited()


async def test_stop_during_an_idle_claim_returns_without_sleeping() -> None:
    client = _Client([])
    reporter = ProvisioningReporter(CallbackClient.disabled(), session_id="session")
    agent = ProvisioningAgent(Settings(apex_session_id="session"), client=client, reporter=reporter)

    async def claim_then_stop(
        _session_id: str, _agent_id: str
    ) -> tuple[int, Mapping[str, object] | None]:
        agent.request_stop()
        return 204, None

    client.claim_command_mock.side_effect = claim_then_stop
    sleep = AsyncMock()
    agent._sleep_with_jitter = sleep

    await agent.run()

    sleep.assert_not_awaited()


async def test_4xx_logs_rejected_and_backs_off() -> None:
    client = _Client([(401, None)])
    reporter = ProvisioningReporter(CallbackClient.disabled(), session_id="session")
    agent = ProvisioningAgent(Settings(apex_session_id="session"), client=client, reporter=reporter)
    delays: list[float] = []

    async def sleep(interval: float) -> None:
        delays.append(interval)
        agent.request_stop()

    agent._sleep_with_jitter = sleep
    with patch("ai_content_service.agent.log.error") as error:
        await agent.run()

    error.assert_called_once_with("agent.claim.rejected", status=401)
    assert delays == [5.0]


async def test_unexpected_claim_status_logs_a_warning() -> None:
    client = _Client([(503, None)])
    reporter = ProvisioningReporter(CallbackClient.disabled(), session_id="session")
    agent = ProvisioningAgent(Settings(apex_session_id="session"), client=client, reporter=reporter)

    async def sleep(interval: float) -> None:  # noqa: ARG001
        agent.request_stop()

    agent._sleep_with_jitter = sleep
    with patch("ai_content_service.agent.log.warning") as warning:
        await agent.run()

    warning.assert_called_once_with("agent.claim.unexpected_status", status=503)


async def test_sleep_with_jitter_returns_when_stop_event_is_set() -> None:
    agent = ProvisioningAgent(Settings())
    agent._stop_event = asyncio.Event()
    agent.request_stop()

    await agent._sleep_with_jitter(5.0)


async def test_sleep_with_jitter_uses_asyncio_sleep_without_a_stop_event() -> None:
    agent = ProvisioningAgent(Settings())

    with patch("ai_content_service.agent.asyncio.sleep", new=AsyncMock()) as sleep:
        await agent._sleep_with_jitter(5.0)

    sleep.assert_awaited_once()


async def test_sleep_with_jitter_returns_after_timeout() -> None:
    agent = ProvisioningAgent(Settings())
    agent._stop_event = asyncio.Event()

    await agent._sleep_with_jitter(0.0)


def test_install_signal_handlers_registers_handlers() -> None:
    agent = ProvisioningAgent(Settings())
    loop = MagicMock()

    with patch("ai_content_service.agent.asyncio.get_running_loop", return_value=loop):
        agent.install_signal_handlers()

    assert loop.add_signal_handler.call_count == 2
    assert loop.add_signal_handler.call_args_list[0].args == (signal.SIGTERM, agent.request_stop)
    assert loop.add_signal_handler.call_args_list[1].args == (signal.SIGINT, agent.request_stop)


def test_install_signal_handlers_logs_when_unavailable() -> None:
    agent = ProvisioningAgent(Settings())

    with (
        patch(
            "ai_content_service.agent.asyncio.get_running_loop",
            side_effect=RuntimeError("no running loop"),
        ),
        patch("ai_content_service.agent.log.debug") as debug,
    ):
        agent.install_signal_handlers()

    debug.assert_called_once_with("agent.signal_handlers.unavailable")


async def test_unparseable_command_without_an_operation_id_is_logged() -> None:
    reporter = ProvisioningReporter(CallbackClient.disabled(), session_id="session")
    agent = ProvisioningAgent(Settings(), reporter=reporter)

    with patch("ai_content_service.agent.log.error") as error:
        await agent._execute_claim({"kind": "nope"})

    error.assert_called_once()


async def test_owned_reporter_is_closed_after_run() -> None:
    client = _Client([])
    agent = ProvisioningAgent(Settings(apex_session_id="session"), client=client)
    agent.request_stop()

    await agent.run()

    assert client._closed is True


async def test_injected_reporter_is_not_closed_by_agent() -> None:
    client = _Client([])
    reporter_client = CallbackClient.disabled()
    reporter = ProvisioningReporter(reporter_client, session_id="session")
    agent = ProvisioningAgent(Settings(apex_session_id="session"), client=client, reporter=reporter)
    agent.request_stop()

    await agent.run()

    assert reporter_client._closed is False
