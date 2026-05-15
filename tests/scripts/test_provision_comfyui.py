"""Tests for scripts/aisha-provision-comfyui.sh.

Bash tests use subprocess.run to spawn the script (or source individual
functions) under a fake environment where heavy binaries are no-op stubs on
$PATH.  This lets the full script run without touching the real system.

Stubbing strategy:
- `_HEAVY_STUBS` are ALWAYS stubbed: apt-get, dpkg, cloudflared, curl, uv, acs.
  These either touch the network, install packages, or are expensive to run.
- `git` is NOT in the always-stub list. clone_or_update_repo now does a real
  HEAD-resolution check after fetch/reset, so a no-op `git` stub would fail
  that check. Tests that don't need real git behavior pass `stub_git=True`
  to _base_env / _full_env, which appends `git` to the stub set AND prepares
  the on-disk repo state the script expects to find.

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

# Always-stubbed binaries — network/install/expensive operations we never want
# to perform during tests. `git` is deliberately NOT in this list; see the
# module docstring.
_HEAVY_STUBS = [
    "apt-get",
    "dpkg",
    "cloudflared",
    "curl",
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


def _base_env(tmp_path: Path, *, stub_git: bool = False) -> dict[str, str]:
    """Minimal env for function-level tests.

    By default, `git` is NOT stubbed — helper functions that touch git
    (sanitize_remote_url, clone_or_update_repo) need the real binary to
    take effect. Pass `stub_git=True` for tests where git invocations are
    unwanted (e.g., main() validation tests that exit before any clone).
    """
    stubs = [*_HEAVY_STUBS, "git"] if stub_git else _HEAVY_STUBS
    bin_dir = make_path_stubs(tmp_path, stubs)
    return {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "HOME": str(tmp_path),
    }


def _init_seed_repo(upstream: Path, branch: str = "master") -> None:
    """Initialize a bare upstream repo with one commit on the named branch.

    Used by _full_env and any test that needs a realistic clone source.
    """
    subprocess.run(["git", "init", "-q", "--bare", str(upstream)], check=True)
    seed = upstream.parent / f"{upstream.stem}-seed"
    subprocess.run(["git", "clone", "-q", str(upstream), str(seed)], check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=seed, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=seed, check=True)
    (seed / "README").write_text("seed\n")
    subprocess.run(["git", "add", "."], cwd=seed, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=seed, check=True)
    subprocess.run(["git", "branch", "-M", branch], cwd=seed, check=True)
    subprocess.run(["git", "push", "-q", "origin", branch], cwd=seed, check=True)


def _full_env(tmp_path: Path, extra: dict[str, str] | None = None) -> dict[str, str]:
    """Complete environment for end-to-end tests with heavy ops stubbed.

    Sets up *real* local bare-repo upstreams for aisha and ai-bundles, and
    points the script at them via file:// URLs. This lets the script's real
    git invocations (fetch, reset, rev-parse HEAD) work end-to-end without
    network access, while everything else (cloudflared install, uv venv,
    acs deploy) remains stubbed.

    A pre-existing aisha-venv with stub python/acs is created so
    install_aisha's existence check passes without invoking real uv.
    """
    # Heavy stubs only — real git is needed so clone_or_update_repo's
    # HEAD-resolution check passes against a realistic upstream.
    bin_dir = make_path_stubs(tmp_path, _HEAVY_STUBS)

    # Pre-create the aisha-venv structure so install_aisha's existence
    # check passes. The actual `uv pip install` inside install_aisha is a
    # no-op against the stub uv.
    aisha_venv = tmp_path / "aisha-venv"
    venv_bin = aisha_venv / "bin"
    venv_bin.mkdir(parents=True)
    for name in ("python", "acs"):
        stub = venv_bin / name
        stub.write_text("#!/bin/sh\nexit 0\n")
        stub.chmod(stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    # Set up real local upstream repos. The script will clone from these
    # via file:// URLs, then later updates use real git too.
    upstreams = tmp_path / "upstreams"
    upstreams.mkdir()
    aisha_upstream = upstreams / "aisha.git"
    bundles_upstream = upstreams / "ai-bundles.git"
    _init_seed_repo(aisha_upstream)
    _init_seed_repo(bundles_upstream)

    # Target paths where the script will land the clones. Must NOT pre-exist
    # — the script's "already cloned?" check is `[[ -d "$path/.git" ]]`, and
    # we want it to take the fresh-clone path.
    aisha_dir = tmp_path / "aisha"
    bundles_dir = tmp_path / "bundles"

    conf_path = tmp_path / "aisha-cloudflared.conf"
    log_dir = tmp_path / "logs"

    env: dict[str, str] = {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "ACS_GITHUB_TOKEN": "fake_token",
        "ACS_BUNDLE": "test_bundle",
        "ACS_AISHA_REPO": f"file://{aisha_upstream}",
        "ACS_BUNDLES_REPO": f"file://{bundles_upstream}",
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
    bin_dir = make_path_stubs(tmp_path, _HEAVY_STUBS)
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
    # Stub git too — main() exits before any git call, so stubbing avoids any
    # accidental network behavior if validation logic ever moves.
    bin_dir = make_path_stubs(tmp_path, [*_HEAVY_STUBS, "git"])
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
    bin_dir = make_path_stubs(tmp_path, [*_HEAVY_STUBS, "git"])
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
        r"^acs\.provision\.ready session_id=\S* elapsed=\d+s bundle=\S+ cloudflared=on$",
        re.MULTILINE,
    )
    assert pattern.search(result.stdout), (
        f"ready line not found in stdout:\n{result.stdout}"
    )


def test_ready_line_emits_cloudflared_off_when_token_missing(tmp_path: Path) -> None:
    """Without ACS_CF_TUNNEL_TOKEN, the ready line must still be emitted with cloudflared=off."""
    # No ACS_CF_TUNNEL_TOKEN — exercises the other branch of the conditional.
    env = _full_env(tmp_path, {"ACS_APEX_SESSION_ID": "sess-noflared"})

    result = _run(env, timeout=30)

    assert result.returncode == 0, result.stderr
    pattern = re.compile(
        r"^acs\.provision\.ready session_id=\S* elapsed=\d+s bundle=\S+ cloudflared=off$",
        re.MULTILINE,
    )
    assert pattern.search(result.stdout), (
        f"ready line with cloudflared=off not found in stdout:\n{result.stdout}"
    )


# ---------------------------------------------------------------------------
# github_auth_header_arg + sanitize_remote_url
# ---------------------------------------------------------------------------


def test_github_auth_header_arg_emits_bearer_when_token_set(tmp_path: Path) -> None:
    env = {**_base_env(tmp_path), "ACS_GITHUB_TOKEN": "ghp_xxx"}
    result = _source_and_call("github_auth_header_arg", env)

    assert result.returncode == 0, result.stderr
    assert (
        result.stdout.strip()
        == "http.https://github.com/.extraheader=Authorization: Bearer ghp_xxx"
    )


def test_github_auth_header_arg_is_empty_when_no_token(tmp_path: Path) -> None:
    """Empty output lets the caller splice it in only when present (a no-op `git -c ''` would error)."""
    env = _base_env(tmp_path)  # no ACS_GITHUB_TOKEN
    result = _source_and_call("github_auth_header_arg", env)

    assert result.returncode == 0, result.stderr
    assert result.stdout == ""


def test_sanitize_remote_url_strips_token_from_git_config(tmp_path: Path) -> None:
    """After clone, .git/config must not contain the embedded token."""
    # Build a real local bare repo + a local clone, then plant a tokenized URL
    upstream = tmp_path / "upstream.git"
    work = tmp_path / "work"
    subprocess.run(["git", "init", "-q", "--bare", str(upstream)], check=True)
    subprocess.run(["git", "clone", "-q", str(upstream), str(work)], check=True)
    subprocess.run(
        [
            "git",
            "remote",
            "set-url",
            "origin",
            "https://ghp_secret_token@github.com/gearbox/aisha.git",
        ],
        cwd=str(work),
        check=True,
    )
    # Sanity: token is in config before sanitize
    pre_url = subprocess.run(
        ["git", "config", "--get", "remote.origin.url"],
        cwd=str(work),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert "ghp_secret_token" in pre_url

    env = _base_env(tmp_path)
    result = _source_and_call(
        f"sanitize_remote_url {shlex.quote(str(work))} "
        f"https://github.com/gearbox/aisha.git",
        env,
    )
    assert result.returncode == 0, result.stderr

    post_url = subprocess.run(
        ["git", "config", "--get", "remote.origin.url"],
        cwd=str(work),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert post_url == "https://github.com/gearbox/aisha.git"
    assert "ghp_secret_token" not in post_url

    # Grep the entire .git directory for residual leakage
    grep = subprocess.run(
        ["grep", "-r", "ghp_secret_token", str(work / ".git")],
        capture_output=True,
        text=True,
    )
    assert grep.returncode != 0, (
        f"token leaked into .git tree:\n{grep.stdout}"
    )


# ---------------------------------------------------------------------------
# clone_or_update_repo — fail-loud behavior
# ---------------------------------------------------------------------------


def test_clone_or_update_repo_fails_on_missing_branch(tmp_path: Path) -> None:
    """A nonexistent branch must abort provisioning, not fall back to a stale ref.

    Regression guard for the previous behavior that silently continued via
    `git pull --ff-only` when `git fetch` failed.
    """
    upstream = tmp_path / "upstream.git"
    work = tmp_path / "work"
    subprocess.run(["git", "init", "-q", "--bare", str(upstream)], check=True)

    # Seed upstream with one commit on master
    seed = tmp_path / "seed"
    subprocess.run(["git", "clone", "-q", str(upstream), str(seed)], check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=seed, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=seed, check=True)
    (seed / "f").write_text("hi")
    subprocess.run(["git", "add", "."], cwd=seed, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=seed, check=True)
    subprocess.run(["git", "branch", "-M", "master"], cwd=seed, check=True)
    subprocess.run(["git", "push", "-q", "origin", "master"], cwd=seed, check=True)

    # First, do a successful clone on `master` so we land in the "update" path
    env = _base_env(tmp_path)
    bootstrap = _source_and_call(
        f"clone_or_update_repo test_repo file://{shlex.quote(str(upstream))} "
        f"{shlex.quote(str(work))} master",
        env,
    )
    assert bootstrap.returncode == 0, bootstrap.stderr

    # Now attempt update against a nonexistent branch — must FAIL
    result = _source_and_call(
        f"clone_or_update_repo test_repo file://{shlex.quote(str(upstream))} "
        f"{shlex.quote(str(work))} totally-not-a-branch",
        env,
        timeout=15,
    )
    assert result.returncode != 0, (
        f"clone_or_update_repo silently succeeded for a missing branch.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


# ---------------------------------------------------------------------------
# check_uv
# ---------------------------------------------------------------------------


def test_check_uv_succeeds_when_uv_on_path(tmp_path: Path) -> None:
    """check_uv should resolve uv from PATH stubs."""
    env = _base_env(tmp_path)  # _HEAVY_STUBS includes "uv"
    result = _source_and_call("check_uv", env)
    assert result.returncode == 0, result.stderr


def test_check_uv_fails_when_uv_missing(tmp_path: Path) -> None:
    """check_uv must exit non-zero with an actionable error when uv is absent.

    This is the boundary we accept: we deliberately do NOT `curl | sh` from
    astral.sh at runtime, so we surface 'no uv' as a hard failure with an
    actionable hint pointing at the base image.
    """
    # PATH without our stubs and without anywhere uv could plausibly live
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(tmp_path),
    }
    result = _source_and_call("check_uv", env)
    assert result.returncode != 0
    assert "uv is not on PATH" in result.stderr


# ---------------------------------------------------------------------------
# run_deployment — flag / env-var contract
# ---------------------------------------------------------------------------


def test_run_deployment_does_not_pass_bundles_path_flag(tmp_path: Path) -> None:
    """run_deployment must NOT pass --bundles-path to acs deploy.

    The wired `acs` CLI (ai_content_service.cli:app) does not define that flag;
    bundles_path is resolved by Settings from the ACS_BUNDLES_PATH env var.
    Regression guard for the v0.6.1 → v0.6.2 fix.
    """
    args_log = tmp_path / "acs_args.log"
    fake_aisha_venv = tmp_path / "venv"
    bin_dir = fake_aisha_venv / "bin"
    bin_dir.mkdir(parents=True)
    acs_stub = bin_dir / "acs"
    acs_stub.write_text(
        f"#!/bin/sh\nprintf '%s\\n' \"$@\" > {args_log}\nexit 0\n"
    )
    acs_stub.chmod(0o755)

    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(tmp_path),
        "ACS_AISHA_VENV": str(fake_aisha_venv),
        "ACS_BUNDLE": "qwen_rapid_aio",
        "ACS_BUNDLES_PATH": str(tmp_path / "ai-bundles"),
        "ACS_COMFYUI_PATH": str(tmp_path / "ComfyUI"),
    }

    result = _source_and_call("run_deployment", env)

    assert result.returncode == 0, result.stderr
    args = args_log.read_text().splitlines()
    assert "--bundle" in args
    assert "qwen_rapid_aio" in args
    assert "--comfyui" in args
    assert "--bundles-path" not in args, (
        "regression: --bundles-path leaked back into the deploy invocation"
    )


def test_run_deployment_exports_acs_bundles_path_with_bundles_suffix(tmp_path: Path) -> None:
    """Exported ACS_BUNDLES_PATH must point at <repo>/bundles, not <repo>.

    The clone lands at $BUNDLES_PATH (e.g. /workspace/ai-bundles); bundles live
    under $BUNDLES_PATH/bundles/. Aisha's Settings.bundles_path is the latter.
    """
    env_log = tmp_path / "acs_env.log"
    fake_aisha_venv = tmp_path / "venv"
    bin_dir = fake_aisha_venv / "bin"
    bin_dir.mkdir(parents=True)
    acs_stub = bin_dir / "acs"
    acs_stub.write_text(
        f'#!/bin/sh\nenv | grep -E "^ACS_BUNDLES_PATH=" > {env_log}\nexit 0\n'
    )
    acs_stub.chmod(0o755)

    repo_root = tmp_path / "ai-bundles"
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(tmp_path),
        "ACS_AISHA_VENV": str(fake_aisha_venv),
        "ACS_BUNDLE": "qwen_rapid_aio",
        "ACS_BUNDLES_PATH": str(repo_root),
        "ACS_COMFYUI_PATH": str(tmp_path / "ComfyUI"),
    }

    result = _source_and_call("run_deployment", env)

    assert result.returncode == 0, result.stderr
    logged = env_log.read_text().strip()
    expected = f"ACS_BUNDLES_PATH={repo_root}/bundles"
    assert logged == expected, f"expected {expected!r}, got {logged!r}"


def test_run_deployment_forwards_optional_flags(tmp_path: Path) -> None:
    """ACS_BUNDLE_VERSION, ACS_MODELS_ONLY, ACS_NO_VERIFY must still propagate."""
    args_log = tmp_path / "acs_args.log"
    fake_aisha_venv = tmp_path / "venv"
    bin_dir = fake_aisha_venv / "bin"
    bin_dir.mkdir(parents=True)
    acs_stub = bin_dir / "acs"
    acs_stub.write_text(
        f"#!/bin/sh\nprintf '%s\\n' \"$@\" > {args_log}\nexit 0\n"
    )
    acs_stub.chmod(0o755)

    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(tmp_path),
        "ACS_AISHA_VENV": str(fake_aisha_venv),
        "ACS_BUNDLE": "qwen_rapid_aio",
        "ACS_BUNDLE_VERSION": "260515-01",
        "ACS_MODELS_ONLY": "true",
        "ACS_NO_VERIFY": "true",
        "ACS_BUNDLES_PATH": str(tmp_path / "ai-bundles"),
        "ACS_COMFYUI_PATH": str(tmp_path / "ComfyUI"),
    }

    result = _source_and_call("run_deployment", env)

    assert result.returncode == 0, result.stderr
    args = args_log.read_text().splitlines()
    assert "--version" in args
    assert "260515-01" in args
    assert "--models-only" in args
    assert "--no-verify" in args


def test_run_deployment_propagates_acs_failure(tmp_path: Path) -> None:
    """If acs deploy exits non-zero, run_deployment must propagate it.

    Guards against accidentally adding `|| true` or `&` during the refactor.
    """
    fake_aisha_venv = tmp_path / "venv"
    bin_dir = fake_aisha_venv / "bin"
    bin_dir.mkdir(parents=True)
    acs_stub = bin_dir / "acs"
    acs_stub.write_text("#!/bin/sh\nexit 17\n")
    acs_stub.chmod(0o755)

    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(tmp_path),
        "ACS_AISHA_VENV": str(fake_aisha_venv),
        "ACS_BUNDLE": "qwen_rapid_aio",
        "ACS_BUNDLES_PATH": str(tmp_path / "ai-bundles"),
        "ACS_COMFYUI_PATH": str(tmp_path / "ComfyUI"),
    }

    result = _source_and_call("run_deployment", env)

    assert result.returncode != 0
