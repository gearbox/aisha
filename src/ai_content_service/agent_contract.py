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

    def __init__(self, message: str, *, operation_id: str | None) -> None:
        super().__init__(message)
        self.operation_id = operation_id


def parse_command(body: Mapping[str, object]) -> Command:
    """Parse one v2 command envelope without performing I/O.

    The operation id is recovered first so malformed commands still produce a
    terminal event whenever Apex supplied an addressable operation.
    """
    operation_id = _optional_nonempty_string(body.get("operation_id"))
    command_id = _required_string(body, "command_id", operation_id)
    resolved_operation_id = _required_string(body, "operation_id", operation_id)
    kind = _parse_kind(body.get("kind"), operation_id)
    batch = _parse_batch(body.get("batch"), operation_id)
    payload = _mapping(body.get("payload"), "payload", operation_id)
    if "force" in payload:
        raise CommandParseError(
            "payload field 'force' is not allowed for agent commands", operation_id=operation_id
        )

    parsed_payload: ProvisionPayload | RemovalPayload | RestartPayload
    match kind:
        case OperationKind.BUNDLE_PROVISION:
            parsed_payload = _parse_provision(payload, batch, operation_id)
        case OperationKind.BUNDLE_REMOVAL:
            parsed_payload = _parse_removal(payload, operation_id)
        case OperationKind.COMFYUI_RESTART:
            parsed_payload = _parse_restart(payload, operation_id)
        case _:
            # Defensive backstop for future enum additions.
            raise CommandParseError(
                f"unsupported command kind {kind.value!r}", operation_id=operation_id
            )
    return Command(
        command_id=command_id,
        operation_id=resolved_operation_id,
        kind=kind,
        batch=batch,
        payload=parsed_payload,
    )


def _parse_kind(value: object, operation_id: str | None) -> OperationKind:
    if not isinstance(value, str):
        raise CommandParseError("kind must be a string", operation_id=operation_id)
    if value == OperationKind.SESSION_BOOTSTRAP.value:
        raise CommandParseError(
            "session_bootstrap is not a command Apex may enqueue", operation_id=operation_id
        )
    try:
        return OperationKind(value)
    except ValueError as exc:
        raise CommandParseError(
            f"unknown command kind {value!r}", operation_id=operation_id
        ) from exc


def _parse_batch(value: object, operation_id: str | None) -> BatchRef | None:
    if value is None:
        return None
    raw = _mapping(value, "batch", operation_id)
    batch_id = _required_string(raw, "batch_id", operation_id)
    index = _required_integer(raw, "index", operation_id)
    total = _required_integer(raw, "total", operation_id)
    if index < 0 or total <= 0 or index >= total:
        raise CommandParseError(
            "batch index must be non-negative and smaller than total", operation_id=operation_id
        )
    return BatchRef(batch_id=batch_id, index=index, total=total)


def _parse_provision(
    payload: Mapping[str, object], batch: BatchRef | None, operation_id: str | None
) -> ProvisionPayload:
    bundle = _required_string(payload, "bundle", operation_id)
    mode_value = _required_string(payload, "mode", operation_id)
    try:
        mode = DeployMode(mode_value)
    except ValueError as exc:
        raise CommandParseError(
            f"unknown deployment mode {mode_value!r}", operation_id=operation_id
        ) from exc
    verify_value = payload.get("verify", True)
    if not isinstance(verify_value, bool):
        raise CommandParseError(
            "payload field 'verify' must be a boolean", operation_id=operation_id
        )
    declared = payload.get("batch_declared_bytes")
    if declared is not None:
        if isinstance(declared, bool) or not isinstance(declared, int) or declared < 0:
            raise CommandParseError(
                "payload field 'batch_declared_bytes' must be a non-negative integer",
                operation_id=operation_id,
            )
        if batch is None or batch.index != 0:
            raise CommandParseError(
                "batch_declared_bytes is permitted only on batch index 0", operation_id=operation_id
            )
    return ProvisionPayload(
        bundle=bundle,
        mode=mode,
        verify=verify_value,
        batch_declared_bytes=declared,
    )


def _parse_removal(payload: Mapping[str, object], operation_id: str | None) -> RemovalPayload:
    bundle = _required_string(payload, "bundle", operation_id)
    raw_retain = payload.get("retain_bundles", ())
    if not isinstance(raw_retain, (list, tuple)) or any(
        not isinstance(item, str) or not item for item in raw_retain
    ):
        raise CommandParseError(
            "payload field 'retain_bundles' must be a list of non-empty strings",
            operation_id=operation_id,
        )
    return RemovalPayload(bundle=bundle, retain_bundles=tuple(raw_retain))


def _parse_restart(payload: Mapping[str, object], operation_id: str | None) -> RestartPayload:
    node_class = payload.get("node_class")
    if node_class is not None and (not isinstance(node_class, str) or not node_class):
        raise CommandParseError(
            "payload field 'node_class' must be a non-empty string or null",
            operation_id=operation_id,
        )
    return RestartPayload(node_class=node_class)


def _mapping(value: object, field: str, operation_id: str | None) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise CommandParseError(f"{field} must be an object", operation_id=operation_id)
    return value


def _required_string(values: Mapping[str, object], field: str, operation_id: str | None) -> str:
    value = _optional_nonempty_string(values.get(field))
    if value is None:
        raise CommandParseError(f"{field} must be a non-empty string", operation_id=operation_id)
    return value


def _optional_nonempty_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _required_integer(values: Mapping[str, object], field: str, operation_id: str | None) -> int:
    value = values.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise CommandParseError(f"{field} must be an integer", operation_id=operation_id)
    return value
