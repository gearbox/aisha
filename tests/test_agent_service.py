"""Tests for the provisioning-agent supervisor service renderer."""

from __future__ import annotations

import stat
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

from ai_content_service.agent_service import (
    ENV_DENYLIST,
    install_agent_service,
    render_startup_script,
    render_supervisor_conf,
    shell_escape,
)
from ai_content_service.config import Settings

if TYPE_CHECKING:
    import pytest


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        agent_script_path=tmp_path / "scripts" / "agent.sh",
        agent_supervisor_conf_path=tmp_path / "conf" / "agent.conf",
    )


def test_script_is_written_with_mode_700(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ACS_APEX_CALLBACK_TOKEN", "secret")
    settings = _settings(tmp_path)

    script_path, _conf_path = install_agent_service(settings)

    assert stat.S_IMODE(script_path.stat().st_mode) == 0o700


def test_conf_contains_only_proc_name_environment() -> None:
    conf = render_supervisor_conf(script_path=Path("/opt/supervisor-scripts/aisha-agent.sh"))

    assert 'environment=PROC_NAME="%(program_name)s"' in conf
    assert "ACS_" not in conf


def test_shell_escape_handles_quote_dollar_backtick_backslash() -> None:
    assert shell_escape('a\\b"$`') == 'a\\\\b\\"\\$\\`'


def test_token_value_appears_only_in_the_script_not_the_conf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    token = 'secret"$`'
    monkeypatch.setenv("ACS_APEX_CALLBACK_TOKEN", token)
    settings = _settings(tmp_path)

    script_path, conf_path = install_agent_service(settings)

    assert shell_escape(token) in script_path.read_text()
    assert token not in conf_path.read_text()


def test_denylisted_one_shot_variables_are_omitted() -> None:
    environment = dict.fromkeys(ENV_DENYLIST, "value")
    environment["ACS_APEX_SESSION_ID"] = "session"

    script = render_startup_script(
        acs_bin=Path("/workspace/aisha-venv/bin/acs"),
        workdir=Path("/workspace"),
        environment=environment,
    )

    assert "ACS_APEX_SESSION_ID" in script
    assert all(key not in script for key in ENV_DENYLIST)


def test_script_contains_provisioning_wait_block() -> None:
    script = render_startup_script(acs_bin=Path("/acs"), workdir=Path("/workspace"), environment={})

    assert 'while [ -f "/.provisioning" ]' in script


def test_utils_sourcing_is_guarded_for_environments_without_them() -> None:
    script = render_startup_script(acs_bin=Path("/acs"), workdir=Path("/workspace"), environment={})

    assert '[ -d "${utils}" ] &&' in script


def test_dry_run_prints_without_writing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    settings = _settings(tmp_path)

    install_agent_service(settings, dry_run=True)

    output = capsys.readouterr().out
    assert "[program:aisha-agent]" in output
    assert not settings.agent_script_path.exists()
    assert not settings.agent_supervisor_conf_path.exists()


def test_dry_run_redacts_secret_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("ACS_APEX_CALLBACK_TOKEN", "apex_TOKEN_ABCDEF")
    monkeypatch.setenv("ACS_GITHUB_TOKEN", "ghp_SUPERSECRET123")

    install_agent_service(_settings(tmp_path), dry_run=True)

    output = capsys.readouterr().out
    assert "apex_TOKEN_ABCDEF" not in output
    assert "ghp_SUPERSECRET123" not in output
    assert output.count("***redacted***") == 2


def test_dry_run_show_secrets_prints_real_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("ACS_APEX_CALLBACK_TOKEN", "apex_TOKEN_ABCDEF")

    install_agent_service(_settings(tmp_path), dry_run=True, show_secrets=True)

    assert "apex_TOKEN_ABCDEF" in capsys.readouterr().out


def test_written_script_always_contains_real_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ACS_APEX_CALLBACK_TOKEN", "apex_TOKEN_ABCDEF")

    script_path, _ = install_agent_service(_settings(tmp_path))

    assert "apex_TOKEN_ABCDEF" in script_path.read_text()


def test_token_never_logged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    token = "apex_TOKEN_ABCDEF"
    monkeypatch.setenv("ACS_APEX_CALLBACK_TOKEN", token)

    with patch("ai_content_service.agent_service.log.info") as info:
        install_agent_service(_settings(tmp_path), dry_run=True)

    assert token not in str(info.call_args_list)


def test_script_cd_target_comes_from_settings_not_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workdir = tmp_path / "chosen-workdir"
    monkeypatch.chdir(tmp_path)
    settings = _settings(tmp_path).model_copy(update={"agent_workdir": workdir})

    install_agent_service(settings)

    assert f'cd "{workdir}"' in settings.agent_script_path.read_text()


def test_acs_bin_is_overridable(tmp_path: Path) -> None:
    acs_bin = tmp_path / "venv" / "bin" / "acs"
    settings = _settings(tmp_path).model_copy(update={"agent_acs_bin": acs_bin})

    install_agent_service(settings)

    assert f'exec "{acs_bin}" agent run' in settings.agent_script_path.read_text()


def test_supervisor_conf_uses_the_explicit_script_path() -> None:
    script_path = Path("/custom/aisha-agent.sh")

    conf = render_supervisor_conf(script_path=script_path)

    assert f"command={script_path}" in conf
