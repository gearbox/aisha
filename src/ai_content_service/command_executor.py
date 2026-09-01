"""Single-command execution for the long-lived provisioning agent."""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

import structlog
from rich.console import Console

from .agent_contract import Command, ProvisionPayload, RemovalPayload, RestartPayload
from .batch_guard import check_batch_headroom
from .bundle_operations import run_comfyui_restart, run_removal
from .bundle_registry import BundleReference
from .operation_telemetry import OperationTarget
from .registry_service import run_deploy
from .telemetry_contract import OperationKind

if TYPE_CHECKING:
    from .config import Settings
    from .provisioning_reporter import ProvisioningReporter

log = structlog.get_logger()
ABANDONED_BATCH_CAP = 32


class CommandExecutor:
    """Dispatch one validated command while preserving terminal telemetry."""

    def __init__(
        self,
        settings: Settings,
        *,
        reporter: ProvisioningReporter,
        console: Console | None = None,
    ) -> None:
        self._settings = settings
        self._reporter = reporter
        self._console = console or Console()
        self._abandoned_batches: OrderedDict[str, None] = OrderedDict()

    async def execute(self, command: Command) -> bool:
        """Run one command to a terminal event without leaking exceptions."""
        try:
            if command.kind is OperationKind.BUNDLE_PROVISION:
                provision = _provision_payload(command)
                if command.batch is not None and command.batch.batch_id in self._abandoned_batches:
                    await self._emit_failed(
                        command,
                        "batch was abandoned after an earlier disk headroom refusal",
                    )
                    return False
                if provision.batch_declared_bytes is not None:
                    verdict = check_batch_headroom(
                        models_path=self._settings.models_path,
                        declared_bytes=provision.batch_declared_bytes,
                        margin=self._settings.batch_disk_margin,
                    )
                    if not verdict.ok:
                        if command.batch is not None:
                            self._abandon(command.batch.batch_id)
                        await self._emit_failed(command, verdict.detail)
                        return False
                result = await run_deploy(
                    self._settings,
                    BundleReference.parse(provision.bundle),
                    provision.mode,
                    provision.verify,
                    False,
                    sync=None,
                    console=self._console,
                    # The agent is never allowed to override additive preflight safeguards.
                    force=False,
                    operation_id=command.operation_id,
                    operation_kind=command.kind,
                    batch=command.batch,
                    reporter=self._reporter,
                )
                return result.success
            if command.kind is OperationKind.BUNDLE_REMOVAL:
                removal = _removal_payload(command)
                await run_removal(
                    self._settings,
                    removal.bundle,
                    retain_bundles=removal.retain_bundles,
                    operation_id=command.operation_id,
                    batch=command.batch,
                    reporter=self._reporter,
                )
                return True
            if command.kind is OperationKind.COMFYUI_RESTART:
                restart = _restart_payload(command)
                await run_comfyui_restart(
                    self._settings,
                    node_class=restart.node_class,
                    operation_id=command.operation_id,
                    batch=command.batch,
                    reporter=self._reporter,
                )
                return True
            await self._emit_failed(command, f"unsupported command kind {command.kind.value!r}")
            return False
        except Exception:
            log.error(
                "agent.command.failed",
                command_id=command.command_id,
                operation_id=command.operation_id,
                exc_info=True,
            )
            return False

    async def _emit_failed(self, command: Command, detail: str) -> None:
        """Use the standard operation stream for pre-dispatch refusals."""
        async with self._reporter.operation(
            operation_id=command.operation_id,
            kind=command.kind,
            target=_target(command),
            batch=command.batch,
        ) as operation:
            await operation.started(plan=None, message="Starting command")
            await operation.failed(detail, summary=None)

    def _abandon(self, batch_id: str) -> None:
        """Remember a refused batch with a fixed FIFO memory bound."""
        if batch_id in self._abandoned_batches:
            return
        self._abandoned_batches[batch_id] = None
        while len(self._abandoned_batches) > ABANDONED_BATCH_CAP:
            self._abandoned_batches.popitem(last=False)


def _target(command: Command) -> OperationTarget | None:
    if isinstance(command.payload, ProvisionPayload):
        return OperationTarget(
            bundle=command.payload.bundle,
            bundle_version="",
            mode=command.payload.mode.value,
        )
    return None


def _provision_payload(command: Command) -> ProvisionPayload:
    if not isinstance(command.payload, ProvisionPayload):
        raise TypeError("bundle_provision command has an invalid payload")
    return command.payload


def _removal_payload(command: Command) -> RemovalPayload:
    if not isinstance(command.payload, RemovalPayload):
        raise TypeError("bundle_removal command has an invalid payload")
    return command.payload


def _restart_payload(command: Command) -> RestartPayload:
    if not isinstance(command.payload, RestartPayload):
        raise TypeError("comfyui_restart command has an invalid payload")
    return command.payload
