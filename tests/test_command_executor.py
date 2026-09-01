"""Tests for serial agent command execution and batch refusal behavior."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

from ai_content_service.agent_contract import Command, RestartPayload, parse_command
from ai_content_service.batch_guard import HeadroomVerdict
from ai_content_service.command_executor import ABANDONED_BATCH_CAP, CommandExecutor
from ai_content_service.config import Settings
from ai_content_service.provisioning_reporter import ProvisioningReporter
from ai_content_service.telemetry_contract import OperationKind

if TYPE_CHECKING:
    from pathlib import Path


def _command(
    *,
    kind: str = "bundle_provision",
    payload: dict[str, object] | None = None,
    batch: object = None,
):
    return parse_command(
        {
            "command_id": "command-1",
            "operation_id": "operation-1",
            "kind": kind,
            "batch": batch,
            "payload": payload or {"bundle": "wan", "mode": "additive"},
        }
    )


def _executor(tmp_path: Path) -> CommandExecutor:
    settings = Settings(comfyui_path=tmp_path / "ComfyUI", cache_path=tmp_path / "cache")
    return CommandExecutor(settings, reporter=ProvisioningReporter.disabled())


class _CapturingClient:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    async def post_best_effort(self, _path: str, payload: object) -> None:
        self.events.append(dict(payload))  # type: ignore[arg-type]

    async def post_retried(self, _path: str, payload: object) -> bool:
        self.events.append(dict(payload))  # type: ignore[arg-type]
        return True


def _capturing_executor(tmp_path: Path) -> tuple[CommandExecutor, _CapturingClient]:
    client = _CapturingClient()
    reporter = ProvisioningReporter(client, session_id="session")  # type: ignore[arg-type]
    settings = Settings(comfyui_path=tmp_path / "ComfyUI", cache_path=tmp_path / "cache")
    return CommandExecutor(settings, reporter=reporter), client


def _invalid_provision_command() -> Command:
    return Command(
        command_id="command-1",
        operation_id="operation-1",
        kind=OperationKind.BUNDLE_PROVISION,
        batch=None,
        payload=RestartPayload(),
    )


async def test_provision_dispatches_with_force_false(tmp_path: Path) -> None:
    executor = _executor(tmp_path)
    deploy = AsyncMock(return_value=SimpleNamespace(success=True))

    with patch("ai_content_service.command_executor.run_deploy", deploy):
        assert await executor.execute(_command()) is True

    assert deploy.await_args is not None
    assert deploy.await_args.kwargs["force"] is False


async def test_batch_reference_reaches_the_operation(tmp_path: Path) -> None:
    executor = _executor(tmp_path)
    command = _command(
        batch={"batch_id": "batch-1", "index": 1, "total": 2},
    )
    deploy = AsyncMock(return_value=SimpleNamespace(success=True))

    with patch("ai_content_service.command_executor.run_deploy", deploy):
        assert await executor.execute(command) is True

    assert deploy.await_args is not None
    assert deploy.await_args.kwargs["batch"] == command.batch


async def test_removal_and_restart_dispatch_to_their_roots(tmp_path: Path) -> None:
    executor = _executor(tmp_path)
    removal = AsyncMock()
    restart = AsyncMock()

    with (
        patch("ai_content_service.command_executor.run_removal", removal),
        patch("ai_content_service.command_executor.run_comfyui_restart", restart),
    ):
        assert await executor.execute(
            _command(kind="bundle_removal", payload={"bundle": "wan", "retain_bundles": ["qwen"]})
        )
        assert await executor.execute(
            _command(kind="comfyui_restart", payload={"node_class": "KSampler"})
        )

    removal.assert_awaited_once()
    restart.assert_awaited_once()


async def test_headroom_refusal_emits_started_and_failed(tmp_path: Path) -> None:
    executor = _executor(tmp_path)
    command = _command(
        payload={"bundle": "wan", "mode": "full", "batch_declared_bytes": 10},
        batch={"batch_id": "batch-1", "index": 0, "total": 2},
    )
    refusal = HeadroomVerdict(False, 1, 10, "not enough free bytes")

    with patch("ai_content_service.command_executor.check_batch_headroom", return_value=refusal):
        assert await executor.execute(command) is False


async def test_abandoned_batch_fast_fails_without_resolving_a_bundle(tmp_path: Path) -> None:
    executor = _executor(tmp_path)
    first = _command(
        payload={"bundle": "wan", "mode": "full", "batch_declared_bytes": 10},
        batch={"batch_id": "batch-1", "index": 0, "total": 2},
    )
    later = _command(
        payload={"bundle": "qwen", "mode": "full"},
        batch={"batch_id": "batch-1", "index": 1, "total": 2},
    )
    refusal = HeadroomVerdict(False, 1, 10, "not enough free bytes")
    deploy = AsyncMock(return_value=SimpleNamespace(success=True))

    with (
        patch("ai_content_service.command_executor.check_batch_headroom", return_value=refusal),
        patch("ai_content_service.command_executor.run_deploy", deploy),
    ):
        assert await executor.execute(first) is False
        assert await executor.execute(later) is False

    deploy.assert_not_awaited()


def test_abandoned_batches_are_capped(tmp_path: Path) -> None:
    executor = _executor(tmp_path)

    for number in range(ABANDONED_BATCH_CAP + 1):
        executor._abandon(f"batch-{number}")

    assert len(executor._abandoned_batches) == ABANDONED_BATCH_CAP
    assert "batch-0" not in executor._abandoned_batches


async def test_execute_never_propagates(tmp_path: Path) -> None:
    executor = _executor(tmp_path)
    deploy = AsyncMock(side_effect=RuntimeError("boom"))

    with patch("ai_content_service.command_executor.run_deploy", deploy):
        assert await executor.execute(_command()) is False


async def test_unresolvable_bundle_still_emits_terminal_failed(tmp_path: Path) -> None:
    executor, client = _capturing_executor(tmp_path)

    assert await executor.execute(_command()) is False

    assert [event["status"] for event in client.events] == ["running", "failed"]


async def test_malformed_bundle_reference_emits_terminal_failed(tmp_path: Path) -> None:
    executor, client = _capturing_executor(tmp_path)

    with patch(
        "ai_content_service.command_executor.BundleReference.parse",
        side_effect=ValueError("bad reference"),
    ):
        assert await executor.execute(_command()) is False

    assert [event["status"] for event in client.events] == ["running", "failed"]
    target = client.events[0]["target"]
    assert isinstance(target, dict)
    assert target["bundle_version"] is None


async def test_invalid_provision_payload_emits_terminal_failed(tmp_path: Path) -> None:
    executor, client = _capturing_executor(tmp_path)

    assert await executor.execute(_invalid_provision_command()) is False

    assert [event["status"] for event in client.events] == ["running", "failed"]


async def test_headroom_stat_failure_emits_terminal_failed(tmp_path: Path) -> None:
    executor, client = _capturing_executor(tmp_path)
    command = _command(
        payload={"bundle": "wan", "mode": "full", "batch_declared_bytes": 10},
        batch={"batch_id": "batch-1", "index": 0, "total": 1},
    )

    with patch(
        "ai_content_service.command_executor.check_batch_headroom",
        side_effect=FileNotFoundError("models directory missing"),
    ):
        assert await executor.execute(command) is False

    assert [event["status"] for event in client.events] == ["running", "failed"]


async def test_ensure_terminal_is_silent_when_the_root_already_reported(tmp_path: Path) -> None:
    executor, client = _capturing_executor(tmp_path)
    reporter = executor._reporter
    async with reporter.operation(
        operation_id="operation-1", kind=OperationKind.BUNDLE_PROVISION
    ) as operation:
        await operation.started(plan=None)
        await operation.failed("root failure")

    with patch("ai_content_service.command_executor.run_deploy", side_effect=RuntimeError("boom")):
        assert await executor.execute(_command()) is False

    assert [event["status"] for event in client.events] == ["running", "failed"]


async def test_fallback_terminal_continues_the_sequence_after_started(tmp_path: Path) -> None:
    executor, client = _capturing_executor(tmp_path)
    reporter = executor._reporter
    async with reporter.operation(
        operation_id="operation-1", kind=OperationKind.BUNDLE_PROVISION
    ) as operation:
        await operation.started(plan=None)

    assert await executor.execute(_invalid_provision_command()) is False

    assert [event["sequence"] for event in client.events] == [0, 1, 2]
