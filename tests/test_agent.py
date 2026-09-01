"""Tests for provisioning-agent polling, backoff, and stop behavior."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

from ai_content_service.agent import ProvisioningAgent
from ai_content_service.callback_client import CallbackClient
from ai_content_service.command_executor import CommandExecutor
from ai_content_service.config import Settings
from ai_content_service.provisioning_reporter import ProvisioningReporter

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

    async def execute(self, command: Command) -> bool:
        await self.execute_mock(command)
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


async def test_unparseable_command_with_operation_id_emits_terminal() -> None:
    client = _Client([(200, {"operation_id": "operation-1", "kind": "nope"})])
    reporter = ProvisioningReporter(CallbackClient.disabled(), session_id="session")
    agent = ProvisioningAgent(Settings(apex_session_id="session"), client=client, reporter=reporter)

    async def stop_after_error(body: Mapping[str, object]) -> None:
        del body
        agent.request_stop()

    agent._execute_claim = stop_after_error
    await agent.run()

    assert client.claim_command_mock.await_count == 1


def test_agent_id_defaults_to_session_and_hostname(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ai_content_service.agent.socket.gethostname", lambda: "gpu-1")

    agent = ProvisioningAgent(Settings(apex_session_id="session"))

    assert agent.agent_id == "session:gpu-1"
