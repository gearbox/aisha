"""Tests for serial agent command execution and batch refusal behavior."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

from ai_content_service.agent_contract import parse_command
from ai_content_service.batch_guard import HeadroomVerdict
from ai_content_service.command_executor import ABANDONED_BATCH_CAP, CommandExecutor
from ai_content_service.config import Settings
from ai_content_service.provisioning_reporter import ProvisioningReporter

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


async def test_provision_dispatches_with_force_false(tmp_path: Path) -> None:
    executor = _executor(tmp_path)
    deploy = AsyncMock(return_value=SimpleNamespace(success=True))

    with patch("ai_content_service.command_executor.run_deploy", deploy):
        assert await executor.execute(_command()) is True

    assert deploy.await_args is not None
    assert deploy.await_args.kwargs["force"] is False


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
