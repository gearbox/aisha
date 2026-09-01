"""CLI contract tests for provisioning-agent commands."""

from __future__ import annotations

from typer.testing import CliRunner

from ai_content_service.cli import app


def test_agent_run_exits_two_with_named_missing_apex_setting() -> None:
    result = CliRunner().invoke(app, ["agent", "run"])

    assert result.exit_code == 2
    assert "ACS_APEX_CALLBACK_URL" in result.output
