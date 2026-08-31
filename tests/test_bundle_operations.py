"""Tests for telemetry envelopes around removal and restart operations."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import pytest

from ai_content_service.bundle_operations import run_comfyui_restart, run_removal
from ai_content_service.config import Settings
from ai_content_service.remover import RemovalError, RemovalResult
from ai_content_service.residency import ResidencyStore, ResidentBundle
from ai_content_service.telemetry_contract import OperationKind, ProvisioningPhase

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path


class _Operation:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    async def started(self, plan: object, message: str = "") -> None:
        _ = message
        self.calls.append(("started", plan))

    async def begin_phase(self, phase: ProvisioningPhase, message: str = "") -> None:
        _ = message
        self.calls.append(("phase", phase))

    async def succeeded(self, summary: object, message: str = "") -> None:
        _ = message
        self.calls.append(("succeeded", summary))

    async def failed(self, error: str, summary: object = None) -> None:
        _ = summary
        self.calls.append(("failed", error))


class _Reporter:
    def __init__(self) -> None:
        self.operation_instance = _Operation()
        self.kind: OperationKind | None = None

    async def __aenter__(self) -> _Reporter:
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        return None

    @asynccontextmanager
    async def operation(
        self, *, kind: OperationKind, **_kwargs: object
    ) -> AsyncIterator[_Operation]:
        self.kind = kind
        yield self.operation_instance


def _settings(tmp_path: Path) -> Settings:
    return Settings(comfyui_path=tmp_path / "ComfyUI", cache_path=tmp_path / "cache")


async def test_removal_emits_exactly_two_operation_events_and_no_timing_jsonl(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    reporter = _Reporter()
    result = RemovalResult("bundle", (), (), 0, False, ())

    with patch(
        "ai_content_service.bundle_operations.BundleRemover.remove",
        new=AsyncMock(return_value=result),
    ):
        returned = await run_removal(settings, "bundle", reporter=reporter)  # type: ignore[arg-type]

    assert returned is result
    assert reporter.kind is OperationKind.BUNDLE_REMOVAL
    assert reporter.operation_instance.calls == [("started", None), ("succeeded", None)]
    assert not (settings.cache_path / "provisioning-timings.jsonl").exists()


async def test_removal_failure_still_emits_terminal_event(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    reporter = _Reporter()

    with (
        patch(
            "ai_content_service.bundle_operations.BundleRemover.remove",
            new=AsyncMock(side_effect=RemovalError("unknown")),
        ),
        pytest.raises(RemovalError, match="unknown"),
    ):
        await run_removal(settings, "bundle", reporter=reporter)  # type: ignore[arg-type]

    assert [name for name, _value in reporter.operation_instance.calls] == ["started", "failed"]


async def test_restart_emits_restart_phase_and_clears_pending_flags(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    ResidencyStore(settings.residency_path).record(
        ResidentBundle(
            name="bundle",
            version="260901-01",
            registry=None,
            mode="additive",
            deployed_at="2026-09-01T00:00:00+00:00",
            model_files=(),
            custom_nodes=(),
            workflow_filename=None,
            readiness_node_class="Ready",
            pending_restart=True,
        )
    )
    reporter = _Reporter()

    with patch(
        "ai_content_service.bundle_operations.ComfyUIManager.restart_and_wait", new=AsyncMock()
    ):
        await run_comfyui_restart(settings, node_class="Ready", reporter=reporter)  # type: ignore[arg-type]

    assert reporter.kind is OperationKind.COMFYUI_RESTART
    assert [name for name, _value in reporter.operation_instance.calls] == [
        "started",
        "phase",
        "succeeded",
    ]
    assert ResidencyStore(settings.residency_path).load()["bundle"].pending_restart is False
