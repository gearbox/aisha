"""Tests for the pure provisioning-agent command contract."""

from __future__ import annotations

import pytest

from ai_content_service.agent_contract import (
    CommandParseError,
    ProvisionPayload,
    RemovalPayload,
    RestartPayload,
    parse_command,
)
from ai_content_service.config import DeployMode
from ai_content_service.telemetry_contract import OperationKind


def _envelope(
    *, kind: str = "bundle_provision", payload: object | None = None
) -> dict[str, object]:
    return {
        "command_id": "cmd-1",
        "operation_id": "operation-1",
        "kind": kind,
        "batch": {"batch_id": "batch-1", "index": 0, "total": 2},
        "payload": payload
        if payload is not None
        else {"bundle": "remote/wan:260101-01", "mode": "additive", "verify": False},
    }


def test_parse_provision_command() -> None:
    command = parse_command(_envelope(payload={"bundle": "wan", "mode": "additive"}))

    assert command.kind is OperationKind.BUNDLE_PROVISION
    assert command.batch is not None and command.batch.batch_id == "batch-1"
    assert command.payload == ProvisionPayload("wan", DeployMode.ADDITIVE)


def test_parse_removal_and_restart_commands() -> None:
    removal = parse_command(
        _envelope(kind="bundle_removal", payload={"bundle": "wan", "retain_bundles": ["qwen"]})
    )
    restart = parse_command(_envelope(kind="comfyui_restart", payload={"node_class": "KSampler"}))

    assert removal.payload == RemovalPayload("wan", ("qwen",))
    assert restart.payload == RestartPayload("KSampler")


def test_force_in_payload_is_a_parse_error() -> None:
    with pytest.raises(CommandParseError, match="force"):
        parse_command(_envelope(payload={"bundle": "wan", "mode": "full", "force": False}))


def test_session_bootstrap_kind_is_rejected() -> None:
    with pytest.raises(CommandParseError, match="session_bootstrap"):
        parse_command(_envelope(kind="session_bootstrap"))


def test_unknown_mode_is_a_parse_error() -> None:
    with pytest.raises(CommandParseError, match="unknown deployment mode"):
        parse_command(_envelope(payload={"bundle": "wan", "mode": "unsafe"}))


def test_batch_declared_bytes_outside_index_zero_is_a_parse_error() -> None:
    body = _envelope(payload={"bundle": "wan", "mode": "full", "batch_declared_bytes": 12})
    body["batch"] = {"batch_id": "batch-1", "index": 1, "total": 2}

    with pytest.raises(CommandParseError, match="index 0"):
        parse_command(body)


def test_parse_error_carries_operation_id_when_present() -> None:
    with pytest.raises(CommandParseError) as raised:
        parse_command(_envelope(payload={"bundle": "wan", "mode": "unknown"}))

    assert raised.value.operation_id == "operation-1"
