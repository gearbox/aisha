"""Telemetry-scoped bundle removal and ComfyUI restart operations."""

from __future__ import annotations

import shlex
from typing import TYPE_CHECKING

from .comfyui import ComfyUIManager
from .provisioning_reporter import ProvisioningReporter
from .remover import BundleRemover, RemovalResult
from .residency import ResidencyStore
from .telemetry_contract import OperationKind, ProvisioningPhase
from .workflows import WorkflowManager

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .config import Settings


async def run_removal(
    settings: Settings,
    bundle_name: str,
    *,
    retain_bundles: Sequence[str] | None = None,
    dry_run: bool = False,
    operation_id: str | None = None,
    reporter: ProvisioningReporter | None = None,
) -> RemovalResult:
    """Remove a bundle with only start and terminal telemetry events."""
    active_reporter = reporter or ProvisioningReporter.from_settings(settings)
    remover = BundleRemover(
        settings,
        residency=ResidencyStore(settings.residency_path),
        workflow_manager=WorkflowManager(settings.comfyui_path),
    )
    async with (
        active_reporter,
        active_reporter.operation(
            operation_id=operation_id or settings.apex_operation_id or None,
            kind=OperationKind.BUNDLE_REMOVAL,
            target=None,
        ) as operation,
    ):
        await operation.started(plan=None, message="Removing bundle")
        try:
            result = await remover.remove(
                bundle_name,
                retain_bundles=retain_bundles,
                dry_run=dry_run,
            )
        except Exception as exc:
            await operation.failed(str(exc), summary=None)
            raise
        await operation.succeeded(summary=None)
        return result


async def run_comfyui_restart(
    settings: Settings,
    *,
    node_class: str | None,
    timeout_s: float | None = None,
    operation_id: str | None = None,
    reporter: ProvisioningReporter | None = None,
) -> None:
    """Restart ComfyUI and clear all pending restart markers on success."""
    active_reporter = reporter or ProvisioningReporter.from_settings(settings)
    manager = ComfyUIManager(
        settings.comfyui_path,
        python_executable=settings.comfyui_python,
        registry_archive_dir=settings.cache_path,
    )
    residency = ResidencyStore(settings.residency_path)
    command = tuple(shlex.split(settings.comfyui_restart_command))
    effective_timeout = timeout_s or settings.comfyui_restart_timeout_seconds
    async with (
        active_reporter,
        active_reporter.operation(
            operation_id=operation_id or settings.apex_operation_id or None,
            kind=OperationKind.COMFYUI_RESTART,
            target=None,
        ) as operation,
    ):
        await operation.started(
            plan={"phases": [{"phase": ProvisioningPhase.RESTART.value, "will_run": True}]},
            message="Restarting ComfyUI",
        )
        await operation.begin_phase(ProvisioningPhase.RESTART, "Restarting ComfyUI")
        try:
            await manager.restart_and_wait(
                node_class=node_class,
                restart_command=command,
                timeout_s=effective_timeout,
                poll_interval_s=settings.comfyui_restart_poll_interval_seconds,
            )
            residency.mark_all_restarted()
        except Exception as exc:
            await operation.failed(str(exc), summary=None)
            raise
        await operation.succeeded(summary=None)
