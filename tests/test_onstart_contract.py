"""Tests for the onstart.sh bash script contract.

Bash tests use subprocess.run to spawn the script under a fake environment
where apt-get, dpkg, git, uv, acs, cloudflared, pkill, supervisord, and
supervisorctl are all no-op stubs on $PATH.  This lets the full script run
without touching the real system.

Tests 1-3 (validation-gate tests) run the script with the heavy ops still
stubbed but fail fast due to early-exit validation logic.
Tests 4-6 (conf-generation tests) run the full script to completion and
inspect the generated supervisord config file.
Tests 7-8 (Python Settings tests) instantiate the Settings class directly.
"""

from __future__ import annotations

import base64
import os
import stat
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from ai_content_service.config import Settings, reset_settings
from tests.helpers import make_path_stubs

if TYPE_CHECKING:
    from collections.abc import Iterator

ONSTART_SH = Path(__file__).parent.parent / "scripts" / "onstart.sh"

# All binaries the script may exec; stubs shadow each one with "exit 0".
_STUB_NAMES = [
    "apt-get",
    "dpkg",
    "cloudflared",
    "pkill",
    "pgrep",
    "sleep",
    "supervisorctl",
    "supervisord",
    "curl",
    "git",
    "uv",
    "acs",
]


def _run(env: dict[str, str], *, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(ONSTART_SH)],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _source_and_call(
    script_path: Path,
    function_name: str,
    env: dict[str, str],
    *,
    timeout: int = 10,
) -> subprocess.CompletedProcess[str]:
    """Source the script without running main(), then call the named function."""
    cmd = f". {script_path}; {function_name}"
    return subprocess.run(
        ["bash", "-c", cmd],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _base_env(tmp_path: Path) -> dict[str, str]:
    """Minimal env suitable for function-level tests via _source_and_call."""
    bin_dir = make_path_stubs(tmp_path, _STUB_NAMES)
    return {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "HOME": str(tmp_path),
    }


def _full_env(tmp_path: Path, extra: dict[str, str] | None = None) -> dict[str, str]:
    """Build a complete environment for conf-generation tests.

    Creates the required workspace directories and stubs all heavy binaries so
    the script can run end-to-end without apt, git, or supervisord.
    """
    bin_dir = make_path_stubs(tmp_path, _STUB_NAMES)

    comfyui_src = tmp_path / "opt" / "workspace-internal" / "ComfyUI"
    comfyui_src.mkdir(parents=True)

    comfyui_link = tmp_path / "workspace" / "ComfyUI"

    aisha_dir = tmp_path / "aisha"
    aisha_dir.mkdir()
    (aisha_dir / ".git").mkdir()  # makes clone_or_update_repo take the "update" path

    bundles_dir = tmp_path / "bundles"
    bundles_dir.mkdir()
    (bundles_dir / ".git").mkdir()

    conf_path = tmp_path / "aisha.conf"
    log_dir = tmp_path / "logs"

    env: dict[str, str] = {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "ACS_GITHUB_TOKEN": "fake_token",
        "ACS_BUNDLE": "test_bundle",
        "ACS_COMFYUI_SRC": str(comfyui_src),
        "ACS_COMFYUI_PATH": str(comfyui_link),
        "ACS_AISHA_PATH": str(aisha_dir),
        "ACS_BUNDLES_PATH": str(bundles_dir),
        "ACS_SUPERVISOR_CONF_PATH": str(conf_path),
        "ACS_SUPERVISORCTL_BIN": str(bin_dir / "supervisorctl"),
        "ACS_SUPERVISOR_LOG_DIR": str(log_dir),
    }
    if extra:
        env |= extra
    return env


# ---------------------------------------------------------------------------
# 1. Branch defaults
# ---------------------------------------------------------------------------


def test_branch_defaults_are_master() -> None:
    """The script must default to 'master', not 'main', for both repos.

    Tested by grepping the source: the default assignment lines are the single
    authoritative place — no need to spawn the script for a constant check.
    """
    content = ONSTART_SH.read_text()
    assert 'AISHA_BRANCH="${ACS_AISHA_BRANCH:-master}"' in content
    assert 'BUNDLES_BRANCH="${ACS_BUNDLES_BRANCH:-master}"' in content


# ---------------------------------------------------------------------------
# 2 & 3. Early-exit validation
# ---------------------------------------------------------------------------


def test_no_github_auth_exits_2(tmp_path: Path) -> None:
    """Script must exit 2 with a clear error when no GitHub auth is configured."""
    bin_dir = make_path_stubs(tmp_path, _STUB_NAMES)

    env = {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "ACS_BUNDLE": "foo",
        # deliberately omit ACS_GITHUB_TOKEN, ACS_SSH_KEY_PATH, ACS_SSH_KEY_CONTENT
    }

    result = _run(env)

    assert result.returncode == 2
    assert "No GitHub auth configured" in result.stderr


def test_no_bundle_exits_2(tmp_path: Path) -> None:
    """Script must exit 2 with a clear error when ACS_BUNDLE is not set."""
    bin_dir = make_path_stubs(tmp_path, _STUB_NAMES)

    env = {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "ACS_GITHUB_TOKEN": "fake_token",
        # deliberately omit ACS_BUNDLE
    }

    result = _run(env)

    assert result.returncode == 2
    assert "ACS_BUNDLE not set" in result.stderr


def test_ssh_github_auth_is_accepted(tmp_path: Path) -> None:
    """Script must not fail with 'No GitHub auth configured' when SSH key path is provided."""
    bin_dir = make_path_stubs(tmp_path, _STUB_NAMES)

    ssh_key_path = tmp_path / "id_rsa"
    ssh_key_path.write_text("dummy ssh key")

    env = {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "ACS_BUNDLE": "foo",
        "ACS_SSH_KEY_PATH": str(ssh_key_path),
        # deliberately omit ACS_GITHUB_TOKEN and ACS_SSH_KEY_CONTENT
    }

    result = _run(env)

    # Auth gate must have been passed: the specific error must not appear.
    assert "No GitHub auth configured" not in result.stderr
    # exit-2 validation failures must not occur
    assert result.returncode != 2 or (
        "No GitHub auth configured" not in result.stderr
        and "ACS_BUNDLE not set" not in result.stderr
    )


def test_ssh_github_auth_via_content_is_accepted(tmp_path: Path) -> None:
    """Script must not fail with 'No GitHub auth configured' when SSH key content is provided."""
    bin_dir = make_path_stubs(tmp_path, _STUB_NAMES)

    dummy_key_b64 = base64.b64encode(b"dummy ssh key content").decode()

    env = {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "ACS_BUNDLE": "foo",
        "ACS_SSH_KEY_CONTENT": dummy_key_b64,
        # deliberately omit ACS_GITHUB_TOKEN and ACS_SSH_KEY_PATH
    }

    result = _run(env)

    assert "No GitHub auth configured" not in result.stderr
    assert result.returncode != 2 or (
        "No GitHub auth configured" not in result.stderr
        and "ACS_BUNDLE not set" not in result.stderr
    )


# ---------------------------------------------------------------------------
# 4-6. Supervisor config generation
# ---------------------------------------------------------------------------


def test_generate_supervisor_conf_with_tunnel_token(tmp_path: Path) -> None:
    """aisha.conf must contain [program:cloudflared] but NOT [program:comfyui]."""
    env = _full_env(tmp_path, {"ACS_CF_TUNNEL_TOKEN": "eyJfake_tunnel_token"})
    conf_path = Path(env["ACS_SUPERVISOR_CONF_PATH"])

    result = _run(env)

    assert result.returncode == 0, result.stderr
    conf = conf_path.read_text()
    assert "[program:cloudflared]" in conf
    assert "[program:comfyui]" not in conf
    # Token must not appear in command= (security: token goes in environment= only).
    cloudflared_block = conf.split("[program:cloudflared]")[1]
    command_line = next(
        line for line in cloudflared_block.splitlines() if line.startswith("command=")
    )
    assert "eyJfake_tunnel_token" not in command_line
    # Token must appear in the environment= stanza as TUNNEL_TOKEN.
    assert 'TUNNEL_TOKEN="eyJfake_tunnel_token"' in conf
    # Conf file must be mode 0600 (no world-readable secrets).
    mode = stat.S_IMODE(conf_path.stat().st_mode)
    assert mode == 0o600, f"expected 0o600, got {oct(mode)}"


def test_generate_supervisor_conf_without_tunnel_token(tmp_path: Path) -> None:
    """Without CF token, aisha.conf must be removed (no stale conf) and WARN logged."""
    env = _full_env(tmp_path)  # no ACS_CF_TUNNEL_TOKEN
    conf_path = Path(env["ACS_SUPERVISOR_CONF_PATH"])
    # Pre-create a stale conf to simulate a leftover from a previous boot.
    conf_path.write_text("[program:cloudflared]\ncommand=old\n")

    result = _run(env)

    assert result.returncode == 0, result.stderr
    assert not conf_path.exists(), "stale aisha.conf should have been removed"
    assert "[WARN]" in result.stderr


def test_generate_supervisor_conf_no_comfyui_block_regardless_of_port(tmp_path: Path) -> None:
    """aisha.conf must never contain [program:comfyui], even when port is explicitly set."""
    env = _full_env(
        tmp_path,
        {
            "ACS_CF_TUNNEL_TOKEN": "eyJtoken",
            "ACS_COMFYUI_PORT": "18188",
        },
    )
    conf_path = Path(env["ACS_SUPERVISOR_CONF_PATH"])

    result = _run(env)

    assert result.returncode == 0, result.stderr
    conf = conf_path.read_text()
    assert "[program:comfyui]" not in conf


# ---------------------------------------------------------------------------
# 7 & 8. Python Settings — new env var fields
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset() -> Iterator[None]:
    reset_settings()
    yield  # type: ignore[misc]
    reset_settings()


def test_settings_reads_new_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings must pick up the new env vars with correct types."""
    monkeypatch.setenv("ACS_COMFYUI_PORT", "18188")
    monkeypatch.setenv("ACS_COMFYUI_HOST", "127.0.0.1")
    monkeypatch.setenv("ACS_COMFYUI_EXTRA_ARGS", "--preview-method auto")
    monkeypatch.setenv("ACS_CF_TUNNEL_TOKEN", "my_tunnel_secret")
    monkeypatch.setenv("ACS_APEX_SESSION_ID", "sess-abc123")
    monkeypatch.setenv("ACS_APEX_CALLBACK_URL", "https://apex.example.com/cb")
    monkeypatch.setenv("ACS_APEX_CALLBACK_TOKEN", "cb_secret")
    monkeypatch.setenv("ACS_SUPERVISOR_LOG_DIR", "/tmp/test_logs")

    s = Settings()

    assert s.comfyui_port == 18188
    assert s.comfyui_host == "127.0.0.1"
    assert s.comfyui_extra_args == "--preview-method auto"
    assert s.cf_tunnel_token is not None
    assert s.cf_tunnel_token.get_secret_value() == "my_tunnel_secret"
    assert s.apex_session_id == "sess-abc123"
    assert s.apex_callback_url == "https://apex.example.com/cb"
    assert s.apex_callback_token is not None
    assert s.apex_callback_token.get_secret_value() == "cb_secret"
    assert s.supervisor_log_dir == Path("/tmp/test_logs")


def test_settings_defaults_when_env_unset() -> None:
    """New Settings fields must carry the documented defaults when env vars are absent."""
    s = Settings()

    assert s.comfyui_port == 18188
    assert s.comfyui_host == "0.0.0.0"
    assert s.comfyui_extra_args == ""
    assert s.cf_tunnel_token is None
    assert s.apex_session_id == ""
    assert s.apex_callback_url == ""
    assert s.apex_callback_token is None
    assert s.supervisor_log_dir == Path("/var/log/aisha")


# ---------------------------------------------------------------------------
# 9. Port validation
# ---------------------------------------------------------------------------


def test_settings_invalid_port_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings must reject non-integer ACS_COMFYUI_PORT values."""
    monkeypatch.setenv("ACS_COMFYUI_PORT", "not-a-number")

    with pytest.raises(ValidationError):
        Settings()


def test_settings_negative_port_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings must reject negative port values (Field ge=1 constraint)."""
    monkeypatch.setenv("ACS_COMFYUI_PORT", "-1")

    with pytest.raises(ValidationError):
        Settings()


# ---------------------------------------------------------------------------
# 10. link_comfyui_workspace behaviour
# ---------------------------------------------------------------------------


def test_link_comfyui_workspace_creates_symlink(tmp_path: Path) -> None:
    """link_comfyui_workspace symlinks COMFYUI_SRC to COMFYUI_PATH."""
    src = tmp_path / "opt" / "workspace-internal" / "ComfyUI"
    src.mkdir(parents=True)
    (src / "main.py").write_text("# fake comfyui\n")

    target = tmp_path / "workspace" / "ComfyUI"

    env = _base_env(tmp_path)
    env["ACS_COMFYUI_SRC"] = str(src)
    env["ACS_COMFYUI_PATH"] = str(target)

    result = _source_and_call(ONSTART_SH, "link_comfyui_workspace", env)

    assert result.returncode == 0, result.stderr
    assert target.is_symlink()
    assert target.resolve() == src.resolve()


def test_link_comfyui_workspace_is_idempotent(tmp_path: Path) -> None:
    """Running link_comfyui_workspace twice succeeds both times without error."""
    src = tmp_path / "opt" / "workspace-internal" / "ComfyUI"
    src.mkdir(parents=True)
    target = tmp_path / "workspace" / "ComfyUI"

    env = _base_env(tmp_path)
    env["ACS_COMFYUI_SRC"] = str(src)
    env["ACS_COMFYUI_PATH"] = str(target)

    result1 = _source_and_call(ONSTART_SH, "link_comfyui_workspace", env)
    result2 = _source_and_call(ONSTART_SH, "link_comfyui_workspace", env)

    assert result1.returncode == 0, result1.stderr
    assert result2.returncode == 0, result2.stderr
    assert target.is_symlink()


def test_link_comfyui_workspace_fails_when_src_missing(tmp_path: Path) -> None:
    """link_comfyui_workspace exits non-zero when COMFYUI_SRC does not exist."""
    missing_src = tmp_path / "nonexistent" / "ComfyUI"
    target = tmp_path / "workspace" / "ComfyUI"

    env = _base_env(tmp_path)
    env["ACS_COMFYUI_SRC"] = str(missing_src)
    env["ACS_COMFYUI_PATH"] = str(target)

    result = _source_and_call(ONSTART_SH, "link_comfyui_workspace", env)

    assert result.returncode != 0
    assert str(missing_src) in result.stderr


def test_link_comfyui_workspace_leaves_real_directory_alone(tmp_path: Path) -> None:
    """link_comfyui_workspace does not replace a real directory at COMFYUI_PATH."""
    src = tmp_path / "opt" / "workspace-internal" / "ComfyUI"
    src.mkdir(parents=True)

    target = tmp_path / "workspace" / "ComfyUI"
    target.mkdir(parents=True)
    (target / "existing_file.txt").write_text("important\n")

    env = _base_env(tmp_path)
    env["ACS_COMFYUI_SRC"] = str(src)
    env["ACS_COMFYUI_PATH"] = str(target)

    result = _source_and_call(ONSTART_SH, "link_comfyui_workspace", env)

    assert result.returncode == 0, result.stderr
    assert target.is_dir() and not target.is_symlink(), "real directory must be left untouched"
    assert (target / "existing_file.txt").exists()


# ---------------------------------------------------------------------------
# 11. start_supervisord behaviour
# ---------------------------------------------------------------------------


def test_start_supervisord_invokes_supervisord_with_image_config(tmp_path: Path) -> None:
    """start_supervisord runs supervisord -c <config> when not already running."""
    env = _base_env(tmp_path)
    bin_dir = Path(env["PATH"].split(":")[0])

    sentinel = tmp_path / "supervisord_started"
    args_log = tmp_path / "supervisord_args.log"

    # supervisord: log its argv and create a sentinel so supervisorctl knows it's up
    (bin_dir / "supervisord").write_text(f'#!/bin/sh\necho "$*" >> {args_log}\ntouch {sentinel}\n')

    # supervisorctl: fail until supervisord has run (sentinel present)
    (bin_dir / "supervisorctl").write_text(f"#!/bin/sh\n[ -f {sentinel} ] && exit 0 || exit 1\n")

    config_path = tmp_path / "supervisord.conf"
    config_path.write_text("[supervisord]\n")

    env["ACS_SUPERVISORCTL_BIN"] = str(bin_dir / "supervisorctl")
    env["ACS_SUPERVISORD_CONFIG_PATH"] = str(config_path)

    result = _source_and_call(ONSTART_SH, "start_supervisord", env)

    assert result.returncode == 0, result.stderr
    assert args_log.exists(), "supervisord was not invoked"
    assert f"-c {config_path}" in args_log.read_text()


def test_start_supervisord_is_idempotent(tmp_path: Path) -> None:
    """start_supervisord returns 0 without invoking supervisord when already running."""
    env = _base_env(tmp_path)
    bin_dir = Path(env["PATH"].split(":")[0])

    invocation_log = tmp_path / "supervisord_calls.log"
    (bin_dir / "supervisord").write_text(f"#!/bin/sh\necho called >> {invocation_log}\n")
    # supervisorctl exits 0 immediately — daemon "already running"
    # (default stub from make_path_stubs already does this)

    env["ACS_SUPERVISORCTL_BIN"] = str(bin_dir / "supervisorctl")

    result = _source_and_call(ONSTART_SH, "start_supervisord", env)

    assert result.returncode == 0, result.stderr
    assert not invocation_log.exists(), "supervisord must not be launched when already running"


def test_start_supervisord_times_out_when_socket_never_appears(tmp_path: Path) -> None:
    """start_supervisord exits non-zero when supervisord never becomes reachable."""
    env = _base_env(tmp_path)
    bin_dir = Path(env["PATH"].split(":")[0])

    # supervisorctl always fails — daemon never becomes reachable
    (bin_dir / "supervisorctl").write_text("#!/bin/sh\nexit 1\n")

    config_path = tmp_path / "supervisord.conf"
    config_path.write_text("[supervisord]\n")

    env["ACS_SUPERVISORCTL_BIN"] = str(bin_dir / "supervisorctl")
    env["ACS_SUPERVISORD_CONFIG_PATH"] = str(config_path)
    env["ACS_SUPERVISORD_START_TIMEOUT"] = "2"

    result = _source_and_call(ONSTART_SH, "start_supervisord", env, timeout=15)

    assert result.returncode != 0
    assert "did not become reachable" in result.stderr


# ---------------------------------------------------------------------------
# 12. pkill regression — image's ComfyUI must not be killed
# ---------------------------------------------------------------------------


def test_main_does_not_kill_image_comfyui(tmp_path: Path) -> None:
    """main() must never call pkill against ComfyUI's main.py (stop_base_image_comfyui removed)."""
    env = _full_env(tmp_path)

    # Replace the pkill stub with one that logs its arguments so we can inspect calls.
    bin_dir = Path(env["PATH"].split(":")[0])
    pkill_log = tmp_path / "pkill_calls.log"
    pkill_stub = bin_dir / "pkill"
    pkill_stub.write_text(f'#!/bin/sh\necho "$*" >> {pkill_log}\n')

    result = _run(env)

    assert result.returncode == 0, result.stderr
    if pkill_log.exists():
        calls = pkill_log.read_text()
        assert "main.py" not in calls, f"pkill was called with main.py: {calls}"
