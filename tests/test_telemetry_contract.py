"""Tests for the stable operation telemetry vocabulary."""

from __future__ import annotations

import uuid

from ai_content_service.telemetry_contract import ProvisioningPhase, new_id


def test_new_id_is_uuid7_and_stdlib_uuid_parseable() -> None:
    identifier = uuid.UUID(new_id())

    assert identifier.version == 7


def test_phase_values_match_shipped_jsonl_schema() -> None:
    assert [
        phase.value
        for phase in ProvisioningPhase
        if phase not in {ProvisioningPhase.PREFLIGHT, ProvisioningPhase.RESTART}
    ] == [
        "comfyui",
        "requirements_base",
        "requirements_locked",
        "custom_nodes",
        "models",
        "workflow",
        "verifying",
    ]
