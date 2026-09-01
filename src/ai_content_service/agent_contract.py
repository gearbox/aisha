"""Pure parsing for the Apex provisioning-agent command envelope."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .config import DeployMode
from .operation_telemetry import BatchRef
from .telemetry_contract import OperationKind

AGENT_SCHEMA_VERSION = 2


@dataclass(frozen=True, slots=True)
class ProvisionPayload:
    """Validated payload for one bundle provisioning operation."""

    bundle: str
    mode: DeployMode
    verify: bool = True
    batch_declared_bytes: int | None = None


@dataclass(frozen=True, slots=True)
class RemovalPayload:
    """Validated payload for one bundle removal operation."""

    bundle: str
    retain_bundles: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RestartPayload:
    """Validated payload for one ComfyUI restart operation."""

    node_class: str | None = None


@dataclass(frozen=True, slots=True)
class Command:
    """A command Apex assigned to this node's single provisioning agent."""

    command_id: str
    operation_id: str
    kind: OperationKind
    batch: BatchRef | None
    payload: ProvisionPayload | RemovalPayload | RestartPayload


class CommandParseError(Exception):
    """A malformed command envelope, optionally attributable to an operation."""

    def __init__(
        self,
        message: str,
        *,
        operation_id: str | None,
        kind: OperationKind | None = None,
    ) -> None:
        super().__init__(message)
        self.operation_id = operation_id
        self.kind = kind


@dataclass(frozen=True, slots=True)
class _ErrorContext:
    """Fields Apex needs to accept a terminal event for a rejected command."""

    operation_id: str | None
    kind: OperationKind | None = None

    def error(self, message: str) -> CommandParseError:
        """Build an error without dropping any envelope fields parsed so far."""
        return CommandParseError(message, operation_id=self.operation_id, kind=self.kind)


def parse_command(body: Mapping[str, object]) -> Command:
    """Parse one v2 command envelope without performing I/O.

    The operation id and kind are recovered first so malformed commands still
    produce a terminal event whenever Apex supplied an addressable operation.
    """
    operation_id = _optional_nonempty_string(body.get("operation_id"))
    context = _ErrorContext(operation_id=operation_id)
    kind = _parse_kind(body.get("kind"), context)
    context = _ErrorContext(operation_id=operation_id, kind=kind)
    if kind is OperationKind.SESSION_BOOTSTRAP:
        raise context.error("session_bootstrap is not a command Apex may enqueue")
    command_id = _required_string(body, "command_id", context)
    resolved_operation_id = _required_string(body, "operation_id", context)
    batch = _parse_batch(body.get("batch"), context)
    payload = _mapping(body.get("payload"), "payload", context)
    if "force" in payload:
        raise context.error("payload field 'force' is not allowed for agent commands")

    parsed_payload: ProvisionPayload | RemovalPayload | RestartPayload
    match kind:
        case OperationKind.BUNDLE_PROVISION:
            parsed_payload = _parse_provision(payload, batch, context)
        case OperationKind.BUNDLE_REMOVAL:
            parsed_payload = _parse_removal(payload, context)
        case OperationKind.COMFYUI_RESTART:
            parsed_payload = _parse_restart(payload, context)
        case _:
            # Defensive backstop for future enum additions.
            raise context.error(f"unsupported command kind {kind.value!r}")
    return Command(
        command_id=command_id,
        operation_id=resolved_operation_id,
        kind=kind,
        batch=batch,
        payload=parsed_payload,
    )


def _parse_kind(value: object, context: _ErrorContext) -> OperationKind:
    if not isinstance(value, str):
        raise context.error("kind must be a string")
    try:
        return OperationKind(value)
    except ValueError as exc:
        raise context.error(f"unknown command kind {value!r}") from exc


def _parse_batch(value: object, context: _ErrorContext) -> BatchRef | None:
    if value is None:
        return None
    raw = _mapping(value, "batch", context)
    batch_id = _required_string(raw, "batch_id", context)
    index = _required_integer(raw, "index", context)
    total = _required_integer(raw, "total", context)
    if index < 0 or total <= 0 or index >= total:
        raise context.error("batch index must be non-negative and smaller than total")
    return BatchRef(batch_id=batch_id, index=index, total=total)


def _parse_provision(
    payload: Mapping[str, object], batch: BatchRef | None, context: _ErrorContext
) -> ProvisionPayload:
    bundle = _required_string(payload, "bundle", context)
    mode_value = _required_string(payload, "mode", context)
    try:
        mode = DeployMode(mode_value)
    except ValueError as exc:
        raise context.error(f"unknown deployment mode {mode_value!r}") from exc
    verify_value = payload.get("verify", True)
    if not isinstance(verify_value, bool):
        raise context.error("payload field 'verify' must be a boolean")
    declared = payload.get("batch_declared_bytes")
    if declared is not None:
        if isinstance(declared, bool) or not isinstance(declared, int) or declared < 0:
            raise context.error(
                "payload field 'batch_declared_bytes' must be a non-negative integer"
            )
        if batch is None or batch.index != 0:
            raise context.error("batch_declared_bytes is permitted only on batch index 0")
    return ProvisionPayload(
        bundle=bundle,
        mode=mode,
        verify=verify_value,
        batch_declared_bytes=declared,
    )


def _parse_removal(payload: Mapping[str, object], context: _ErrorContext) -> RemovalPayload:
    bundle = _required_string(payload, "bundle", context)
    raw_retain = payload.get("retain_bundles", ())
    if not isinstance(raw_retain, (list, tuple)) or any(
        not isinstance(item, str) or not item for item in raw_retain
    ):
        raise context.error("payload field 'retain_bundles' must be a list of non-empty strings")
    return RemovalPayload(bundle=bundle, retain_bundles=tuple(raw_retain))


def _parse_restart(payload: Mapping[str, object], context: _ErrorContext) -> RestartPayload:
    node_class = payload.get("node_class")
    if node_class is not None and (not isinstance(node_class, str) or not node_class):
        raise context.error("payload field 'node_class' must be a non-empty string or null")
    return RestartPayload(node_class=node_class)


def _mapping(value: object, field: str, context: _ErrorContext) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise context.error(f"{field} must be an object")
    return value


def _required_string(values: Mapping[str, object], field: str, context: _ErrorContext) -> str:
    value = _optional_nonempty_string(values.get(field))
    if value is None:
        raise context.error(f"{field} must be a non-empty string")
    return value


def _optional_nonempty_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _required_integer(values: Mapping[str, object], field: str, context: _ErrorContext) -> int:
    value = values.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise context.error(f"{field} must be an integer")
    return value
