"""Tests for scripts/aisha-provision-comfyui.sh.

Bash tests use subprocess.run to spawn the script (or source individual
functions) under a fake environment where all heavy binaries are no-op stubs
on $PATH.  This lets the full script run without touching the real system.

Setting __SOURCED__=1 in the subprocess environment suppresses the main()
call at the bottom of the script so individual functions can be tested in
isolation via _source_and_call().
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

from tests.helpers import make_path_stubs

PROVISION_SH = Path(__file__).parent.parent.parent / "scripts" / "aisha-provision-comfyui.sh"

_STUB_NAMES = [
    "apt-get",
    "dpkg",
    "cloudflared",
    "curl",
    "git",
    "uv",
    "acs",
]


# ---------------------------------------------------------------------------
# Harness helpers
# ---------------------------------------------------------------------------


def _run(env: dict[str, str], *, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(PROVISION_SH)],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _source_and_call(
    function_name: str,
    env: dict[str, str],
    *,
    timeout: int = 10,
) -> subprocess.CompletedProcess[str]:
    """Source the script with __SOURCED__=1 to suppress main(), then call the named function."""
    cmd = f". {PROVISION_SH}; {function_name}"
    return subprocess.run(
        ["bash", "-c", cmd],
        env={**env, "__SOURCED__": "1"},
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _base_env(tmp_path: Path) -> dict[str, str]:
    """Minimal env for function-level tests."""
    bin_dir = make_path_stubs(tmp_path, _STUB_NAMES)
    return {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "HOME": str(tmp_path),
    }


def _full_env(tmp_path: Path, extra: dict[str, str] | None = None) -> dict[str, str]:
    """Complete environment for end-to-end tests with all heavy ops stubbed."""
    bin_dir = make_path_stubs(tmp_path, _STUB_NAMES)

    # Pre-create venv structure so install_aisha's existence check passes
    aisha_venv = tmp_path / "aisha-venv"
    venv_bin = aisha_venv / "bin"
    venv_bin.mkdir(parents=True)
    for name in ("python", "acs"):
        stub = venv_bin / name
        stub.write_text("#!/bin/sh\nexit 0\n")
        stub.chmod(stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    aisha_dir = tmp_path / "aisha"
    aisha_dir.mkdir()
    (aisha_dir / ".git").mkdir()

    bundles_dir = tmp_path / "bundles"
    bundles_dir.mkdir()
    (bundles_dir / ".git").mkdir()

    conf_path = tmp_path / "aisha-cloudflared.conf"
    log_dir = tmp_path / "logs"

    env: dict[str, str] = {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "ACS_GITHUB_TOKEN": "fake_token",
        "ACS_BUNDLE": "test_bundle",
        "ACS_AISHA_PATH": str(aisha_dir),
        "ACS_BUNDLES_PATH": str(bundles_dir),
        "ACS_AISHA_VENV": str(aisha_venv),
        "ACS_SUPERVISOR_CONF_PATH": str(conf_path),
        "ACS_SUPERVISOR_LOG_DIR": str(log_dir),
    }
    if extra:
        env |= extra
    return env


# ---------------------------------------------------------------------------
# escape_for_supervisord_env
# ---------------------------------------------------------------------------


def test_escape_for_supervisord_env_handles_special_chars(tmp_path: Path) -> None:
    env = _base_env(tmp_path)

    plain = _source_and_call("escape_for_supervisord_env 'hello'", env)
    assert plain.returncode == 0
    assert plain.stdout.strip() == "hello"

    comma = _source_and_call("escape_for_supervisord_env 'a,b'", env)
    assert comma.returncode == 0
    assert comma.stdout.strip() == r"a\,b"

    dquote = _source_and_call(r"""escape_for_supervisord_env 'a"b'""", env)
    assert dquote.returncode == 0
    assert dquote.stdout.strip() == r'a\"b'

    # Use a regular Python string so 'a\\b' in bash single-quotes = one backslash
    backslash = _source_and_call("escape_for_supervisord_env 'a\\b'", env)
    assert backslash.returncode == 0
    assert backslash.stdout.strip() == r"a\\b"


# ---------------------------------------------------------------------------
# get_authenticated_url
# ---------------------------------------------------------------------------


def test_get_authenticated_url_injects_github_token(tmp_path: Path) -> None:
    env = _base_env(tmp_path)

    # Token injected into github.com URL
    env_with_token = {**env, "ACS_GITHUB_TOKEN": "ghp_xxx"}
    result = _source_and_call(
        "get_authenticated_url https://github.com/gearbox/aisha.git",
        env_with_token,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == "https://ghp_xxx@github.com/gearbox/aisha.git"

    # No token — output equals input
    env_no_token = {k: v for k, v in env.items() if k != "ACS_GITHUB_TOKEN"}
    result_no_token = _source_and_call(
        "get_authenticated_url https://github.com/gearbox/aisha.git",
        env_no_token,
    )
    assert result_no_token.returncode == 0
    assert result_no_token.stdout.strip() == "https://github.com/gearbox/aisha.git"

    # Non-github URL — token not injected
    result_non_gh = _source_and_call(
        "get_authenticated_url https://gitlab.com/org/repo.git",
        env_with_token,
    )
    assert result_non_gh.returncode == 0
    assert result_non_gh.stdout.strip() == "https://gitlab.com/org/repo.git"


# ---------------------------------------------------------------------------
# install_aisha — venv creation
# ---------------------------------------------------------------------------


@pytest.fixture()
def _fake_aisha_repo(tmp_path: Path) -> Path:
    """Return path to a minimal installable aisha-like package."""
    pkg_dir = tmp_path / "fake_aisha"
    pkg_dir.mkdir()
    (pkg_dir / "pyproject.toml").write_text(
        "[project]\n"
        'name = "fake-aisha"\n'
        'version = "0.1.0"\n'
        'requires-python = ">=3.10"\n'
        "\n"
        "[project.scripts]\n"
        'acs = "fake_acs_mod:main"\n'
        "\n"
        "[build-system]\n"
        'requires = ["hatchling"]\n'
        'build-backend = "hatchling.build"\n'
        "\n"
        "[tool.hatch.build.targets.wheel]\n"
        'packages = ["src/fake_acs_mod"]\n'
    )
    src = pkg_dir / "src" / "fake_acs_mod"
    src.mkdir(parents=True)
    (src / "__init__.py").write_text("def main(): pass\n")
    return pkg_dir


@pytest.mark.skipif(not shutil.which("uv"), reason="uv not installed")
def test_install_aisha_creates_venv_when_missing(
    tmp_path: Path, _fake_aisha_repo: Path
) -> None:
    venv_path = tmp_path / "test-venv"
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(tmp_path),
        "ACS_AISHA_PATH": str(_fake_aisha_repo),
        "ACS_AISHA_VENV": str(venv_path),
    }

    result = _source_and_call("install_aisha", env, timeout=60)

    assert result.returncode == 0, result.stderr
    python_bin = venv_path / "bin" / "python"
    acs_bin = venv_path / "bin" / "acs"
    assert python_bin.exists(), "venv/bin/python not created"
    assert os.access(python_bin, os.X_OK), "venv/bin/python not executable"
    assert acs_bin.exists(), "venv/bin/acs not created"
    assert os.access(acs_bin, os.X_OK), "venv/bin/acs not executable"


@pytest.mark.skipif(not shutil.which("uv"), reason="uv not installed")
def test_install_aisha_reuses_existing_venv(
    tmp_path: Path, _fake_aisha_repo: Path
) -> None:
    venv_path = tmp_path / "test-venv"
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(tmp_path),
        "ACS_AISHA_PATH": str(_fake_aisha_repo),
        "ACS_AISHA_VENV": str(venv_path),
    }

    # First run creates the venv
    result1 = _source_and_call("install_aisha", env, timeout=60)
    assert result1.returncode == 0, result1.stderr

    # Wrap uv with a logging proxy
    real_uv = shutil.which("uv")
    assert real_uv is not None
    bin_wrap = tmp_path / "bin_wrap"
    bin_wrap.mkdir()
    uv_log = tmp_path / "uv_calls.log"
    uv_wrapper = bin_wrap / "uv"
    uv_wrapper.write_text(f"#!/bin/sh\necho \"$*\" >> {uv_log}\nexec {real_uv} \"$@\"\n")
    uv_wrapper.chmod(0o755)

    env2 = {**env, "PATH": f"{bin_wrap}:{os.environ['PATH']}"}
    result2 = _source_and_call("install_aisha", env2, timeout=60)
    assert result2.returncode == 0, result2.stderr

    assert uv_log.exists(), "uv was not called at all on second run"
    calls = uv_log.read_text()
    # Each logged line starts with the uv sub-command (e.g. "venv /path" or "pip ...").
    assert not any(
        line.startswith("venv") for line in calls.splitlines()
    ), f"uv venv was called on second run:\n{calls}"


# ---------------------------------------------------------------------------
# write_cloudflared_dropin
# ---------------------------------------------------------------------------


def test_write_cloudflared_dropin_with_token(tmp_path: Path) -> None:
    bin_dir = make_path_stubs(tmp_path, ["cloudflared"])
    fake_cf = str(bin_dir / "cloudflared")
    conf_path = tmp_path / "aisha-cloudflared.conf"
    log_dir = tmp_path / "log"

    env = {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "ACS_CF_TUNNEL_TOKEN": "eyJtZXN0Igo=",
        "ACS_SUPERVISOR_CONF_PATH": str(conf_path),
        "ACS_SUPERVISOR_LOG_DIR": str(log_dir),
    }

    result = _source_and_call(
        f"CLOUDFLARED_BIN={shlex.quote(fake_cf)}; write_cloudflared_dropin",
        env,
    )

    assert result.returncode == 0, result.stderr
    assert conf_path.exists(), "conf file was not written"
    conf = conf_path.read_text()
    assert "[program:cloudflared]" in conf
    assert "[program:comfyui]" not in conf, "regression: conf must not contain [program:comfyui]"
    assert 'TUNNEL_TOKEN="eyJtZXN0Igo="' in conf
    mode = stat.S_IMODE(conf_path.stat().st_mode)
    assert mode == 0o600, f"expected mode 0600, got {oct(mode)}"


def test_write_cloudflared_dropin_without_token_clears_stale_file(tmp_path: Path) -> None:
    bin_dir = make_path_stubs(tmp_path, _STUB_NAMES)
    conf_path = tmp_path / "aisha-cloudflared.conf"
    # Pre-create a stale drop-in simulating a previous boot
    conf_path.write_text("[program:cloudflared]\ncommand=old\n")

    env = {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "ACS_SUPERVISOR_CONF_PATH": str(conf_path),
        "ACS_SUPERVISOR_LOG_DIR": str(tmp_path / "log"),
        # ACS_CF_TUNNEL_TOKEN deliberately absent
    }

    result = _source_and_call("write_cloudflared_dropin", env)

    assert result.returncode == 0, result.stderr
    assert not conf_path.exists(), "stale conf must be removed, not left in place"


def test_write_cloudflared_dropin_token_with_special_chars_is_escaped(tmp_path: Path) -> None:
    bin_dir = make_path_stubs(tmp_path, ["cloudflared"])
    fake_cf = str(bin_dir / "cloudflared")
    conf_path = tmp_path / "aisha-cloudflared.conf"

    # Token containing comma, backslash, and double-quote
    token = r'tok,en\val"ue'
    env = {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "ACS_CF_TUNNEL_TOKEN": token,
        "ACS_SUPERVISOR_CONF_PATH": str(conf_path),
        "ACS_SUPERVISOR_LOG_DIR": str(tmp_path / "log"),
    }

    result = _source_and_call(
        f"CLOUDFLARED_BIN={shlex.quote(fake_cf)}; write_cloudflared_dropin",
        env,
    )

    assert result.returncode == 0, result.stderr
    conf = conf_path.read_text()
    env_line = next(line for line in conf.splitlines() if line.startswith("environment="))
    # comma escaped
    assert r"\," in env_line
    # backslash escaped
    assert r"\\" in env_line
    # double-quote escaped
    assert r'\"' in env_line


# ---------------------------------------------------------------------------
# main() — validation exits
# ---------------------------------------------------------------------------


def test_main_validation_exits_2_when_github_token_missing(tmp_path: Path) -> None:
    bin_dir = make_path_stubs(tmp_path, _STUB_NAMES)
    env = {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "ACS_BUNDLE": "anything",
        # ACS_GITHUB_TOKEN deliberately absent
    }

    result = _run(env)

    assert result.returncode == 2
    assert "ACS_GITHUB_TOKEN" in result.stderr


def test_main_validation_exits_2_when_bundle_missing(tmp_path: Path) -> None:
    bin_dir = make_path_stubs(tmp_path, _STUB_NAMES)
    env = {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "ACS_GITHUB_TOKEN": "anything",
        # ACS_BUNDLE deliberately absent
    }

    result = _run(env)

    assert result.returncode == 2
    assert "ACS_BUNDLE" in result.stderr


# ---------------------------------------------------------------------------
# Ready line format
# ---------------------------------------------------------------------------


def test_ready_line_format_is_grep_friendly(tmp_path: Path) -> None:
    """The structured ready line must match the regex apex greps for."""
    env = _full_env(tmp_path, {"ACS_CF_TUNNEL_TOKEN": "eyJtoken", "ACS_APEX_SESSION_ID": "sess-1"})

    result = _run(env, timeout=30)

    assert result.returncode == 0, result.stderr
    pattern = re.compile(
        r"^acs\.provision\.ready session_id=\S* elapsed=\d+s bundle=\S+ cloudflared=(on|off)$",
        re.MULTILINE,
    )
    assert pattern.search(result.stdout), (
        f"ready line not found in stdout:\n{result.stdout}"
    )
