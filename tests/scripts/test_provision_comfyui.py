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
import sys
import uuid
from pathlib import Path

import pytest

from tests.helpers import make_path_stubs

PROVISION_SH = Path(__file__).parent.parent.parent / "scripts" / "aisha-provision-comfyui.sh"
BASH = shutil.which("bash") or "/bin/bash"

# Always-stubbed binaries — network/install/expensive operations we never want
# to perform during tests. `git` is deliberately NOT in this list; see the
# module docstring.
_HEAVY_STUBS = [
    "uv",
    "acs",
    "rclone",
]


# ---------------------------------------------------------------------------
# Harness helpers
# ---------------------------------------------------------------------------


def _run(env: dict[str, str], *, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [BASH, str(PROVISION_SH)],
        env={**env, "ACS_WORKSPACE": env.get("ACS_WORKSPACE", env["HOME"])},
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
        [BASH, "-c", cmd],
        env={
            **env,
            "ACS_WORKSPACE": env.get("ACS_WORKSPACE", env["HOME"]),
            "__SOURCED__": "1",
        },
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
    _write_rclone_version_stub(bin_dir)
    return {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "HOME": str(tmp_path),
    }


def _write_rclone_version_stub(bin_dir: Path, version: str = "v1.71.0") -> None:
    """Make the generic rclone stub satisfy the pinned-version no-op contract."""
    rclone = bin_dir / "rclone"
    rclone.write_text(
        f"#!/bin/sh\nif [ \"${{1:-}}\" = version ]; then\n  echo 'rclone {version}'\nfi\n"
    )
    rclone.chmod(0o755)


def _write_executable(path: Path, contents: str) -> None:
    path.write_text(contents)
    path.chmod(0o755)


def _init_seed_repo(
    upstream: Path, branch: str = "master", *, include_manifest_script: bool = False
) -> None:
    """Initialize a bare upstream repo with one commit on the named branch.

    Used by _full_env and any test that needs a realistic clone source.
    """
    subprocess.run(["git", "init", "-q", "--bare", str(upstream)], check=True)
    seed = upstream.parent / f"{upstream.stem}-seed"
    subprocess.run(["git", "clone", "-q", str(upstream), str(seed)], check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=seed, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=seed, check=True)
    (seed / "README").write_text("seed\n")
    if include_manifest_script:
        script = seed / "scripts" / "capture-env-manifest.sh"
        script.parent.mkdir()
        script.write_text(
            '#!/bin/sh\nprintf \'{"captured_before_install": true, "comfyui_version": "v-test", "packages": {}}\\n\'\n'
        )
        script.chmod(0o755)
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
    _write_rclone_version_stub(bin_dir)

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

    # Stub python for ACS_COMFYUI_PYTHON so run_deployment's executable check passes.
    comfyui_python = tmp_path / "venv-main" / "bin" / "python"
    comfyui_python.parent.mkdir(parents=True)
    comfyui_python.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n')
    comfyui_python.chmod(comfyui_python.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    comfyui_path = tmp_path / "ComfyUI"
    comfyui_path.mkdir()
    (comfyui_path / "comfyui_version.py").write_text('__version__ = "v-test"\n')

    # Set up real local upstream repos. The script will clone from these
    # via file:// URLs, then later updates use real git too.
    upstreams = tmp_path / "upstreams"
    upstreams.mkdir()
    aisha_upstream = upstreams / "aisha.git"
    bundles_upstream = upstreams / "ai-bundles.git"
    _init_seed_repo(aisha_upstream, include_manifest_script=True)
    _init_seed_repo(bundles_upstream)

    # Target paths where the script will land the clones. Must NOT pre-exist
    # — the script's "already cloned?" check is `[[ -d "$path/.git" ]]`, and
    # we want it to take the fresh-clone path.
    aisha_dir = tmp_path / "aisha"
    bundles_dir = tmp_path / "bundles"

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
        "ACS_COMFYUI_PATH": str(comfyui_path),
        "ACS_COMFYUI_PYTHON": str(comfyui_python),
        "ACS_CACHE_PATH": str(tmp_path / "cache"),
    }
    if extra:
        env |= extra
    return env


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


def _write_uv_install_stub(bin_dir: Path, log_path: Path | None = None) -> None:
    """Stub only the installer calls while preserving their argument contract."""
    bin_dir.mkdir(exist_ok=True)
    log = f'printf "%s\\n" "$*" >> {shlex.quote(str(log_path))}\n' if log_path else ""
    (bin_dir / "uv").write_text(
        "#!/bin/sh\n"
        f"{log}"
        'if [ "${1:-}" = "venv" ]; then\n'
        '  mkdir -p "$2/bin"\n'
        '  printf "#!/bin/sh\\nexit 0\\n" > "$2/bin/python"\n'
        '  chmod 755 "$2/bin/python"\n'
        "  exit 0\n"
        "fi\n"
        'if [ "${1:-}" = "pip" ]; then\n'
        '  while [ "$#" -gt 0 ]; do\n'
        '    if [ "$1" = "--python" ]; then\n'
        '      acs="$(dirname "$2")/acs"\n'
        '      printf "#!/bin/sh\\nexit 0\\n" > "$acs"\n'
        '      chmod 755 "$acs"\n'
        "      exit 0\n"
        "    fi\n"
        "    shift\n"
        "  done\n"
        "fi\n"
        "exit 1\n"
    )
    (bin_dir / "uv").chmod(0o755)


@pytest.mark.skipif(not shutil.which("uv"), reason="uv not installed")
def test_install_aisha_creates_venv_when_missing(tmp_path: Path, _fake_aisha_repo: Path) -> None:
    venv_path = tmp_path / "test-venv"
    bin_dir = tmp_path / "bin"
    _write_uv_install_stub(bin_dir)
    env = {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "ACS_AISHA_PATH": str(_fake_aisha_repo),
        "ACS_AISHA_VENV": str(venv_path),
        "ACS_COMFYUI_PYTHON": sys.executable,
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
def test_install_aisha_reuses_existing_venv(tmp_path: Path, _fake_aisha_repo: Path) -> None:
    venv_path = tmp_path / "test-venv"
    bin_dir = tmp_path / "bin"
    _write_uv_install_stub(bin_dir)
    env = {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "ACS_AISHA_PATH": str(_fake_aisha_repo),
        "ACS_AISHA_VENV": str(venv_path),
        "ACS_COMFYUI_PYTHON": sys.executable,
    }

    # First run creates the venv
    result1 = _source_and_call("install_aisha", env, timeout=60)
    assert result1.returncode == 0, result1.stderr

    # Replace the stub with a logging equivalent for the reuse assertion.
    uv_log = tmp_path / "uv_calls.log"
    _write_uv_install_stub(bin_dir, uv_log)

    env2 = env
    result2 = _source_and_call("install_aisha", env2, timeout=60)
    assert result2.returncode == 0, result2.stderr

    assert uv_log.exists(), "uv was not called at all on second run"
    calls = uv_log.read_text()
    # Each logged line starts with the uv sub-command (e.g. "venv /path" or "pip ...").
    assert not any(line.startswith("venv") for line in calls.splitlines()), (
        f"uv venv was called on second run:\n{calls}"
    )


def test_uv_venv_pins_comfyui_python(tmp_path: Path, _fake_aisha_repo: Path) -> None:
    venv_path = tmp_path / "test-venv"
    bin_dir = tmp_path / "bin"
    uv_log = tmp_path / "uv_calls.log"
    _write_uv_install_stub(bin_dir, uv_log)
    env = {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "ACS_AISHA_PATH": str(_fake_aisha_repo),
        "ACS_AISHA_VENV": str(venv_path),
        "ACS_COMFYUI_PYTHON": sys.executable,
    }

    result = _source_and_call("install_aisha", env)

    assert result.returncode == 0, result.stderr
    assert f"venv {venv_path} --python {sys.executable} --quiet" in uv_log.read_text()


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


def test_ready_line_format_no_cloudflared_field(tmp_path: Path) -> None:
    """The ready line must match the new format and must not contain a cloudflared= field."""
    env = _full_env(tmp_path, {"ACS_APEX_SESSION_ID": "sess-1"})

    result = _run(env, timeout=30)

    assert result.returncode == 0, result.stderr
    pattern = re.compile(
        r"^acs\.provision\.ready session_id=\S* elapsed=\d+s bundle=\S+$",
        re.MULTILINE,
    )
    assert pattern.search(result.stdout), f"ready line not found in stdout:\n{result.stdout}"
    assert "cloudflared=" not in result.stdout, "ready line must not contain cloudflared= field"


def test_main_captures_and_preserves_pristine_base_manifest(tmp_path: Path) -> None:
    env = _full_env(tmp_path)
    manifest = Path(env["ACS_CACHE_PATH"]) / "base-manifest.json"

    first = _run(env, timeout=30)
    assert first.returncode == 0, first.stderr
    assert (
        manifest.read_text()
        == '{"captured_before_install": true, "comfyui_version": "v-test", "packages": {}}\n'
    )

    manifest.write_text(
        '{"captured_before_install": true, "comfyui_version": "v-test", "packages": {"preserved": "1"}}\n'
    )
    second = _run(env, timeout=30)

    assert second.returncode == 0, second.stderr
    assert manifest.read_text() == (
        '{"captured_before_install": true, "comfyui_version": "v-test", '
        '"packages": {"preserved": "1"}}\n'
    )
    assert "Reusing pristine base manifest" in second.stdout


def test_capture_base_manifest_invokes_script_via_bash(tmp_path: Path) -> None:
    """capture_base_manifest must not depend on the script's own executable bit.

    A fresh git clone checks out capture-env-manifest.sh at mode 644 (git
    tracks it non-executable). If the call site ever regresses to invoking
    the script directly instead of through `bash`, this non-executable stub
    reproduces that failure with "Permission denied" even though other tests
    in this file mark their seed copy of the script executable and would not
    catch the regression.
    """
    aisha_path = tmp_path / "aisha"
    script = aisha_path / "scripts" / "capture-env-manifest.sh"
    script.parent.mkdir(parents=True)
    script.write_text("#!/bin/sh\nprintf '{\"packages\": {}}\\n'\n")
    script.chmod(0o644)

    cache_path = tmp_path / "cache"
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(tmp_path),
        "ACS_AISHA_PATH": str(aisha_path),
        "ACS_CACHE_PATH": str(cache_path),
        "ACS_COMFYUI_PATH": str(tmp_path / "ComfyUI"),
    }

    result = _source_and_call("capture_base_manifest", env)

    assert result.returncode == 0, result.stderr
    assert (cache_path / "base-manifest.json").read_text() == '{"packages": {}}\n'


def test_capture_base_manifest_failure_removes_temporary_file(tmp_path: Path) -> None:
    """A failed capture must leave neither a partial manifest nor its temporary file."""
    aisha_path = tmp_path / "aisha"
    script = aisha_path / "scripts" / "capture-env-manifest.sh"
    script.parent.mkdir(parents=True)
    script.write_text("#!/bin/sh\nprintf '{\"partial\": true}'\nexit 7\n")

    cache_path = tmp_path / "cache"
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(tmp_path),
        "ACS_AISHA_PATH": str(aisha_path),
        "ACS_CACHE_PATH": str(cache_path),
        "ACS_COMFYUI_PATH": str(tmp_path / "ComfyUI"),
        "ACS_COMFYUI_PYTHON": sys.executable,
    }

    result = _source_and_call("capture_base_manifest", env)

    assert result.returncode != 0
    assert not (cache_path / "base-manifest.json").exists()
    assert not list(cache_path.glob(".base-manifest.*"))
    assert "overlays cannot be generated on this node" in result.stderr


def _manifest_capture_env(
    tmp_path: Path, previous_manifest: str, base_image: str | None
) -> dict[str, str]:
    aisha_path = tmp_path / "aisha"
    script = aisha_path / "scripts" / "capture-env-manifest.sh"
    script.parent.mkdir(parents=True)
    script.write_text(
        """#!/bin/sh
printf '{"base_image": "new-image", "captured_before_install": %s, "packages": {}}\n' "${ACS_MANIFEST_CAPTURED_BEFORE_INSTALL:-true}"
"""
    )

    cache_path = tmp_path / "cache"
    cache_path.mkdir()
    manifest = cache_path / "base-manifest.json"
    manifest.write_text(previous_manifest)
    env: dict[str, str] = {
        "PATH": os.environ["PATH"],
        "HOME": str(tmp_path),
        "ACS_AISHA_PATH": str(aisha_path),
        "ACS_CACHE_PATH": str(cache_path),
        "ACS_COMFYUI_PYTHON": sys.executable,
    }
    if base_image is not None:
        env["ACS_BASE_IMAGE"] = base_image
    return env


def test_capture_base_manifest_reuses_matching_image(tmp_path: Path) -> None:
    previous_manifest = (
        '{"base_image": "same-image", "captured_before_install": true, '
        '"packages": {"preserved": "1"}}\n'
    )
    env = _manifest_capture_env(tmp_path, previous_manifest, "same-image")

    result = _source_and_call("capture_base_manifest", env)

    assert result.returncode == 0, result.stderr
    assert (Path(env["ACS_CACHE_PATH"]) / "base-manifest.json").read_text() == previous_manifest
    assert "Reusing pristine base manifest" in result.stdout


def test_capture_base_manifest_recaptures_when_image_differs(tmp_path: Path) -> None:
    previous_manifest = (
        '{"base_image": "old-image", "captured_before_install": true, "packages": {}}\n'
    )
    env = _manifest_capture_env(tmp_path, previous_manifest, "new-image")

    result = _source_and_call("capture_base_manifest", env)

    assert result.returncode == 0, result.stderr
    cache_path = Path(env["ACS_CACHE_PATH"])
    assert (cache_path / "base-manifest.json").read_text() == (
        '{"base_image": "new-image", "captured_before_install": false, "packages": {}}\n'
    )
    assert (cache_path / "base-manifest.superseded.json").read_text() == previous_manifest
    assert "base_manifest_stale" in result.stderr
    assert "manifest_base_image=old-image live_base_image=new-image" in result.stderr


def test_capture_base_manifest_reuses_when_image_is_unset_despite_comfyui_change(
    tmp_path: Path,
) -> None:
    previous_manifest = (
        '{"base_image": "old-image", "captured_before_install": true, '
        '"comfyui_version": "v-previous", "packages": {}}\n'
    )
    env = _manifest_capture_env(tmp_path, previous_manifest, None)

    result = _source_and_call("capture_base_manifest", env)

    assert result.returncode == 0, result.stderr
    assert (Path(env["ACS_CACHE_PATH"]) / "base-manifest.json").read_text() == previous_manifest
    assert "Reusing pristine base manifest" in result.stdout


def test_capture_base_manifest_reuses_non_pristine_manifest(tmp_path: Path) -> None:
    previous_manifest = (
        '{"base_image": "same-image", "captured_before_install": false, "packages": {}}\n'
    )
    env = _manifest_capture_env(tmp_path, previous_manifest, "same-image")

    result = _source_and_call("capture_base_manifest", env)

    assert result.returncode == 0, result.stderr
    cache_path = Path(env["ACS_CACHE_PATH"])
    assert (cache_path / "base-manifest.json").read_text() == previous_manifest
    assert not (cache_path / "base-manifest.superseded.json").exists()
    assert "base_manifest_not_pristine" in result.stderr
    assert "base-manifest.superseded.json" in result.stderr


def test_capture_base_manifest_preserves_first_superseded_inventory(tmp_path: Path) -> None:
    previous_manifest = (
        '{"base_image": "same-image", "captured_before_install": false, "packages": {}}\n'
    )
    original_inventory = (
        '{"base_image": "first-image", "captured_before_install": true, "packages": {}}\n'
    )
    env = _manifest_capture_env(tmp_path, previous_manifest, "same-image")
    superseded_manifest = Path(env["ACS_CACHE_PATH"]) / "base-manifest.superseded.json"
    superseded_manifest.write_text(original_inventory)

    result = _source_and_call("capture_base_manifest", env)

    assert result.returncode == 0, result.stderr
    assert superseded_manifest.read_text() == original_inventory
    assert "base_manifest_not_pristine" in result.stderr


def test_cf_tunnel_token_optional(tmp_path: Path) -> None:
    """Script must succeed whether or not ACS_CF_TUNNEL_TOKEN is present."""
    without_token_dir = tmp_path / "no_token"
    without_token_dir.mkdir()
    result_no_token = _run(_full_env(without_token_dir), timeout=30)
    assert result_no_token.returncode == 0, f"failed without token:\n{result_no_token.stderr}"

    with_token_dir = tmp_path / "with_token"
    with_token_dir.mkdir()
    result_with_token = _run(
        _full_env(with_token_dir, {"ACS_CF_TUNNEL_TOKEN": "eyJtoken"}),
        timeout=30,
    )
    assert result_with_token.returncode == 0, f"failed with token:\n{result_with_token.stderr}"


def test_no_cloudflared_supervisor_conf_written(tmp_path: Path) -> None:
    """The script must not write any cloudflared supervisor conf."""
    conf_path = tmp_path / "cloudflared.conf"
    env = _full_env(tmp_path, {"ACS_SUPERVISOR_CONF_PATH": str(conf_path)})
    result = _run(env, timeout=30)
    assert result.returncode == 0, result.stderr
    assert not conf_path.exists(), "script must not write a cloudflared supervisor conf"


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
        f"sanitize_remote_url {shlex.quote(str(work))} https://github.com/gearbox/aisha.git",
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
    assert grep.returncode != 0, f"token leaked into .git tree:\n{grep.stdout}"


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
# check_rclone
# ---------------------------------------------------------------------------


def test_check_rclone_is_noop_when_rclone_on_path(tmp_path: Path) -> None:
    """An image-provided rclone must be used without invoking the installer."""
    env = _base_env(tmp_path)
    result = _source_and_call("check_rclone", env)
    assert result.returncode == 0, result.stderr
    assert "rclone present: rclone v1.71.0" in result.stdout


def _rclone_install_env(
    tmp_path: Path,
    *,
    preinstalled_version: str | None = None,
    architecture: str = "x86_64",
    checksum_ok: bool = True,
    installed_version: str = "v1.71.0",
    install_fails: bool = False,
) -> tuple[dict[str, str], Path, Path]:
    """Build explicit executable stubs for deterministic rclone installer tests."""
    bin_dir = tmp_path / "rclone-bin"
    bin_dir.mkdir()
    curl_log = tmp_path / "curl.log"
    install_log = tmp_path / "install.log"
    temp_root = tmp_path / "tmp"
    temp_root.mkdir()
    archive = "rclone-v1.71.0-linux-amd64.zip"

    (bin_dir / "curl").write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' \"$@\" >> {shlex.quote(str(curl_log))}\n"
        f"printf '%s\\n' --CALL-- >> {shlex.quote(str(curl_log))}\n"
        "output=''\n"
        "previous=''\n"
        "url=''\n"
        'for arg in "$@"; do\n'
        '  if [ "$previous" = --output ]; then output="$arg"; fi\n'
        '  previous="$arg"\n'
        '  url="$arg"\n'
        "done\n"
        'case "$url" in\n'
        f"  *SHA256SUMS) printf '%064d  {archive}\\n' 0 > \"$output\" ;;\n"
        '  *) printf archive > "$output" ;;\n'
        "esac\n"
    )
    (bin_dir / "sha256sum").write_text(
        f"#!/bin/sh\ncat >/dev/null\nexit {0 if checksum_ok else 1}\n"
    )
    (bin_dir / "uname").write_text(f"#!/bin/sh\necho {shlex.quote(architecture)}\n")
    (bin_dir / "unzip").write_text(
        "#!/bin/sh\n"
        "destination=''\n"
        "previous=''\n"
        'for arg in "$@"; do\n'
        '  if [ "$previous" = -d ]; then destination="$arg"; fi\n'
        '  previous="$arg"\n'
        "done\n"
        f'mkdir -p "$destination/rclone-v1.71.0-linux-amd64"\n'
        f"printf '#!/bin/sh\\necho rclone {installed_version}\\n' > \"$destination/rclone-v1.71.0-linux-amd64/rclone\"\n"
        f'chmod 755 "$destination/rclone-v1.71.0-linux-amd64/rclone"\n'
    )
    (bin_dir / "install").write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' \"$@\" >> {shlex.quote(str(install_log))}\n"
        f"if [ {1 if install_fails else 0} -eq 1 ]; then exit 1; fi\n"
        'cp "$3" "$4"\n'
    )
    if preinstalled_version is not None:
        _write_rclone_version_stub(bin_dir, preinstalled_version)
    for path in bin_dir.iterdir():
        path.chmod(0o755)
    return (
        {
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "HOME": str(tmp_path),
            "TMPDIR": str(temp_root),
            "ACS_AISHA_VENV": str(tmp_path / "aisha-venv"),
            "ACS_AISHA_BIN": str(tmp_path / "aisha-bin"),
            "ACS_RCLONE_VERSION": "v1.71.0",
        },
        curl_log,
        install_log,
    )


def test_check_rclone_installs_exact_pinned_archive_with_https_only_curl(tmp_path: Path) -> None:
    env, curl_log, install_log = _rclone_install_env(tmp_path)

    result = _source_and_call("check_rclone", env)

    assert result.returncode == 0, result.stderr
    assert install_log.exists()
    calls = [call.splitlines() for call in curl_log.read_text().split("--CALL--\n") if call]
    assert len(calls) == 2
    for args in calls:
        assert {"--fail", "--show-error", "--silent", "--location"} <= set(args)
        assert args[args.index("--proto") + 1] == "=https"
        assert args[args.index("--proto-redir") + 1] == "=https"
    assert any("/v1.71.0/rclone-v1.71.0-linux-amd64.zip" in arg for arg in calls[0])
    assert any("/v1.71.0/SHA256SUMS" in arg for arg in calls[1])
    assert "0755" in install_log.read_text()
    assert (tmp_path / "aisha-bin" / "rclone").is_file()
    assert not (tmp_path / "aisha-venv").exists()
    assert not list((tmp_path / "tmp").glob("aisha-rclone.*"))


def test_check_rclone_replaces_wrong_installed_version(tmp_path: Path) -> None:
    env, _curl_log, install_log = _rclone_install_env(tmp_path, preinstalled_version="v1.0.0")

    result = _source_and_call("check_rclone", env)

    assert result.returncode == 0, result.stderr
    assert "replacing rclone rclone v1.0.0" in result.stdout
    assert install_log.exists()


def test_check_rclone_before_install_aisha_keeps_fresh_venv_target_empty(tmp_path: Path) -> None:
    """rclone installation must not make uv's first-boot venv target non-empty."""
    env, _curl_log, _install_log = _rclone_install_env(tmp_path)
    fake_aisha_repo = tmp_path / "fake-aisha"
    fake_aisha_repo.mkdir()
    env["ACS_AISHA_PATH"] = str(fake_aisha_repo)
    env["ACS_COMFYUI_PYTHON"] = sys.executable

    bin_dir = Path(env["PATH"].split(":", maxsplit=1)[0])
    venv_path = Path(env["ACS_AISHA_VENV"])
    (bin_dir / "uv").write_text(
        "#!/bin/sh\n"
        'if [ "${1:-}" = "--version" ]; then echo "uv test"; exit 0; fi\n'
        'if [ "${1:-}" = "venv" ]; then\n'
        '  target="$2"\n'
        '  if [ -d "$target" ] && [ -n "$(find "$target" -mindepth 1 -print -quit)" ]; then\n'
        '    echo "refusing non-empty venv target: $target" >&2\n'
        "    exit 99\n"
        "  fi\n"
        '  mkdir -p "$target/bin"\n'
        '  : > "$target/pyvenv.cfg"\n'
        '  printf "#!/bin/sh\\nexit 0\\n" > "$target/bin/python"\n'
        '  printf "#!/bin/sh\\nexit 0\\n" > "$target/bin/acs"\n'
        '  chmod 755 "$target/bin/python" "$target/bin/acs"\n'
        "  exit 0\n"
        "fi\n"
        'if [ "${1:-}" = "pip" ]; then exit 0; fi\n'
        "exit 1\n"
    )
    (bin_dir / "uv").chmod(0o755)

    result = _source_and_call("check_rclone; install_aisha", env)

    assert result.returncode == 0, result.stderr
    assert (Path(env["ACS_AISHA_BIN"]) / "rclone").is_file()
    assert (venv_path / "pyvenv.cfg").is_file()


def test_check_rclone_reports_missing_unzip_before_attempting_install(tmp_path: Path) -> None:
    bin_dir = make_path_stubs(tmp_path, ["curl", "rclone"])
    _write_rclone_version_stub(bin_dir, "v1.0.0")
    env = {
        # Do not include a system directory: on Ubuntu, /bin is merged with
        # /usr/bin and therefore exposes the runner's real `unzip`.
        "PATH": str(bin_dir),
        "HOME": str(tmp_path),
        "ACS_AISHA_VENV": str(tmp_path / "aisha-venv"),
    }

    result = _source_and_call("check_rclone", env)

    assert result.returncode != 0
    assert "unzip is not on PATH" in result.stderr


@pytest.mark.parametrize("version", ["stable", "1.71.0", "v1.71", " v1.71.0 "])
def test_check_rclone_requires_a_pinned_release_tag(tmp_path: Path, version: str) -> None:
    env, curl_log, install_log = _rclone_install_env(tmp_path)
    env["ACS_RCLONE_VERSION"] = version

    result = _source_and_call("check_rclone", env)

    if version.strip() == "v1.71.0":
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode != 0
        assert "ACS_RCLONE_VERSION must be a release tag" in result.stderr
        assert not curl_log.exists()
        assert not install_log.exists()


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"checksum_ok": False}, "checksum verification failed"),
        ({"architecture": "ppc64le"}, "unsupported rclone architecture"),
        ({"install_fails": True}, "rclone install failed"),
        ({"installed_version": "v0.0.0"}, "installed rclone version mismatch"),
    ],
)
def test_check_rclone_propagates_install_failures(
    tmp_path: Path, kwargs: dict[str, object], expected: str
) -> None:
    env, _curl_log, install_log = _rclone_install_env(tmp_path, **kwargs)  # type: ignore[arg-type]

    result = _source_and_call("check_rclone", env)

    assert result.returncode != 0
    assert expected in result.stderr
    assert not list((tmp_path / "tmp").glob("aisha-rclone.*"))
    if kwargs.get("checksum_ok") is False:
        assert not install_log.exists()


# ---------------------------------------------------------------------------
# run_deployment — flag / env-var contract
# ---------------------------------------------------------------------------


def test_deploy_omits_operation_id_when_unset(tmp_path: Path) -> None:
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
    acs_stub.write_text(f"#!/bin/sh\nprintf '%s\\n' \"$@\" > {args_log}\nexit 0\n")
    acs_stub.chmod(0o755)

    stub_python = tmp_path / "python"
    stub_python.write_text("#!/bin/sh\nexit 0\n")
    stub_python.chmod(0o755)

    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(tmp_path),
        "ACS_AISHA_VENV": str(fake_aisha_venv),
        "ACS_BUNDLE": "qwen_rapid_aio",
        "ACS_BUNDLES_PATH": str(tmp_path / "ai-bundles"),
        "ACS_COMFYUI_PATH": str(tmp_path / "ComfyUI"),
        "ACS_COMFYUI_PYTHON": str(stub_python),
    }

    result = _source_and_call("run_deployment", env)

    assert result.returncode == 0, result.stderr
    args = args_log.read_text().splitlines()
    assert "--bundle" in args
    assert "qwen_rapid_aio" in args
    assert "--comfyui" in args
    assert "--bootstrap" in args
    assert "--operation-id" not in args
    assert "--bundles-path" not in args, (
        "regression: --bundles-path leaked back into the deploy invocation"
    )


def test_run_deployment_exports_acs_bundles_path_as_repo_root(tmp_path: Path) -> None:
    """Exported ACS_BUNDLES_PATH must remain the ai-bundles repository root."""
    env_log = tmp_path / "acs_env.log"
    fake_aisha_venv = tmp_path / "venv"
    bin_dir = fake_aisha_venv / "bin"
    bin_dir.mkdir(parents=True)
    acs_stub = bin_dir / "acs"
    acs_stub.write_text(f'#!/bin/sh\nenv | grep -E "^ACS_BUNDLES_PATH=" > {env_log}\nexit 0\n')
    acs_stub.chmod(0o755)

    stub_python = tmp_path / "python"
    stub_python.write_text("#!/bin/sh\nexit 0\n")
    stub_python.chmod(0o755)

    repo_root = tmp_path / "ai-bundles"
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(tmp_path),
        "ACS_AISHA_VENV": str(fake_aisha_venv),
        "ACS_BUNDLE": "qwen_rapid_aio",
        "ACS_BUNDLES_PATH": str(repo_root),
        "ACS_COMFYUI_PATH": str(tmp_path / "ComfyUI"),
        "ACS_COMFYUI_PYTHON": str(stub_python),
    }

    result = _source_and_call("run_deployment", env)

    assert result.returncode == 0, result.stderr
    logged = env_log.read_text().strip()
    expected = f"ACS_BUNDLES_PATH={repo_root}"
    assert logged == expected, f"expected {expected!r}, got {logged!r}"


def test_run_deployment_forwards_optional_flags(tmp_path: Path) -> None:
    """ACS_BUNDLE_VERSION, ACS_MODELS_ONLY, ACS_NO_VERIFY must still propagate."""
    args_log = tmp_path / "acs_args.log"
    fake_aisha_venv = tmp_path / "venv"
    bin_dir = fake_aisha_venv / "bin"
    bin_dir.mkdir(parents=True)
    acs_stub = bin_dir / "acs"
    acs_stub.write_text(f"#!/bin/sh\nprintf '%s\\n' \"$@\" > {args_log}\nexit 0\n")
    acs_stub.chmod(0o755)

    stub_python = tmp_path / "python"
    stub_python.write_text("#!/bin/sh\nexit 0\n")
    stub_python.chmod(0o755)

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
        "ACS_COMFYUI_PYTHON": str(stub_python),
    }

    result = _source_and_call("run_deployment", env)

    assert result.returncode == 0, result.stderr
    args = args_log.read_text().splitlines()
    assert "--bundle-version" in args, "ACS_BUNDLE_VERSION must produce --bundle-version flag"
    assert "--version" not in args, "legacy --version flag must never be passed"
    assert "260515-01" in args
    assert "--models-only" in args
    assert "--no-verify" in args


def test_deploy_passes_operation_id_when_set(tmp_path: Path) -> None:
    args_log = tmp_path / "acs_args.log"
    fake_aisha_venv = tmp_path / "venv"
    bin_dir = fake_aisha_venv / "bin"
    bin_dir.mkdir(parents=True)
    acs_stub = bin_dir / "acs"
    acs_stub.write_text(f"#!/bin/sh\nprintf '%s\\n' \"$@\" > {args_log}\nexit 0\n")
    acs_stub.chmod(0o755)
    python = tmp_path / "python"
    python.write_text("#!/bin/sh\nexit 0\n")
    python.chmod(0o755)
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(tmp_path),
        "ACS_AISHA_VENV": str(fake_aisha_venv),
        "ACS_BUNDLE": "qwen_rapid_aio",
        "ACS_BUNDLES_PATH": str(tmp_path / "ai-bundles"),
        "ACS_COMFYUI_PATH": str(tmp_path / "ComfyUI"),
        "ACS_COMFYUI_PYTHON": str(python),
        "ACS_APEX_OPERATION_ID": "apex-operation",
    }

    result = _source_and_call("run_deployment", env)

    assert result.returncode == 0, result.stderr
    args = args_log.read_text().splitlines()
    assert args[args.index("--operation-id") + 1] == "apex-operation"


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

    stub_python = tmp_path / "python"
    stub_python.write_text("#!/bin/sh\nexit 0\n")
    stub_python.chmod(0o755)

    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(tmp_path),
        "ACS_AISHA_VENV": str(fake_aisha_venv),
        "ACS_BUNDLE": "qwen_rapid_aio",
        "ACS_BUNDLES_PATH": str(tmp_path / "ai-bundles"),
        "ACS_COMFYUI_PATH": str(tmp_path / "ComfyUI"),
        "ACS_COMFYUI_PYTHON": str(stub_python),
    }

    result = _source_and_call("run_deployment", env)

    assert result.returncode != 0


# ---------------------------------------------------------------------------
# run_deployment — ACS_COMFYUI_PYTHON contract
# ---------------------------------------------------------------------------


def test_run_deployment_exports_acs_comfyui_python_default(tmp_path: Path) -> None:
    """When ACS_COMFYUI_PYTHON is set to a valid path, run_deployment exports it."""
    env_log = tmp_path / "acs_env.log"

    stub_python = tmp_path / "venv-main" / "bin" / "python"
    stub_python.parent.mkdir(parents=True)
    stub_python.write_text("#!/bin/sh\nexit 0\n")
    stub_python.chmod(0o755)

    fake_aisha_venv = tmp_path / "venv"
    bin_dir = fake_aisha_venv / "bin"
    bin_dir.mkdir(parents=True)
    acs_stub = bin_dir / "acs"
    acs_stub.write_text(f'#!/bin/sh\nenv | grep -E "^ACS_COMFYUI_PYTHON=" > {env_log}\nexit 0\n')
    acs_stub.chmod(0o755)

    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(tmp_path),
        "ACS_AISHA_VENV": str(fake_aisha_venv),
        "ACS_BUNDLE": "qwen_rapid_aio",
        "ACS_BUNDLES_PATH": str(tmp_path / "ai-bundles"),
        "ACS_COMFYUI_PATH": str(tmp_path / "ComfyUI"),
        "ACS_COMFYUI_PYTHON": str(stub_python),
    }

    result = _source_and_call("run_deployment", env)

    assert result.returncode == 0, result.stderr
    logged = env_log.read_text().strip()
    assert logged == f"ACS_COMFYUI_PYTHON={stub_python}"


def test_comfyui_port_defaults_to_18188(tmp_path: Path) -> None:
    """When ACS_COMFYUI_PORT is not set, run_deployment must export it as 18188.

    18188 is the port used by the vastai/comfy base image's comfyui.sh wrapper.
    """
    env_log = tmp_path / "acs_env.log"
    fake_aisha_venv = tmp_path / "venv"
    bin_dir = fake_aisha_venv / "bin"
    bin_dir.mkdir(parents=True)
    acs_stub = bin_dir / "acs"
    acs_stub.write_text(f'#!/bin/sh\nenv | grep -E "^ACS_COMFYUI_PORT=" > {env_log}\nexit 0\n')
    acs_stub.chmod(0o755)

    stub_python = tmp_path / "python"
    stub_python.write_text("#!/bin/sh\nexit 0\n")
    stub_python.chmod(0o755)

    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(tmp_path),
        "ACS_AISHA_VENV": str(fake_aisha_venv),
        "ACS_BUNDLE": "qwen_rapid_aio",
        "ACS_BUNDLES_PATH": str(tmp_path / "ai-bundles"),
        "ACS_COMFYUI_PATH": str(tmp_path / "ComfyUI"),
        "ACS_COMFYUI_PYTHON": str(stub_python),
        # ACS_COMFYUI_PORT deliberately absent — must default to 18188
    }

    result = _source_and_call("run_deployment", env)

    assert result.returncode == 0, result.stderr
    logged = env_log.read_text().strip()
    assert logged == "ACS_COMFYUI_PORT=18188", f"expected ACS_COMFYUI_PORT=18188, got {logged!r}"


def test_main_fails_when_comfyui_python_missing_before_clone_or_capture(tmp_path: Path) -> None:
    """The ComfyUI interpreter is validated before expensive provisioning work."""
    env = _base_env(tmp_path, stub_git=True) | {
        "ACS_GITHUB_TOKEN": "fake-token",
        "ACS_BUNDLE": "qwen_rapid_aio",
        "ACS_AISHA_PATH": str(tmp_path / "aisha"),
        "ACS_BUNDLES_PATH": str(tmp_path / "ai-bundles"),
        "ACS_CACHE_PATH": str(tmp_path / "cache"),
        "ACS_COMFYUI_PYTHON": str(tmp_path / "does-not-exist"),
    }

    result = _run(env)

    assert result.returncode != 0
    assert "ACS_COMFYUI_PYTHON not executable" in result.stderr
    assert not (tmp_path / "aisha").exists()
    assert not (tmp_path / "cache" / "base-manifest.json").exists()


# ---------------------------------------------------------------------------
# terminal-failure redaction
# ---------------------------------------------------------------------------


def _make_curl_recorder(tmp_path: Path) -> tuple[Path, Path]:
    """Create a bin dir with all heavy stubs + a curl recorder.

    Returns (bin_dir, curl_log_path).  Each curl invocation appends its argv
    (one arg per line) to curl_log_path so tests can inspect what was sent.
    """
    bin_dir = make_path_stubs(tmp_path, _HEAVY_STUBS)
    curl_log = tmp_path / "curl_args.log"
    curl_stub = bin_dir / "curl"
    curl_stub.write_text(f"#!/bin/sh\nprintf '%s\\n' \"$@\" >> {curl_log}\nexit 0\n")
    curl_stub.chmod(0o755)
    return bin_dir, curl_log


def test_report_failed_disabled_when_env_unset(tmp_path: Path) -> None:
    bin_dir, curl_log = _make_curl_recorder(tmp_path)

    result = _source_and_call(
        "report_failed 'test error'",
        {"PATH": f"{bin_dir}:{os.environ['PATH']}", "HOME": str(tmp_path)},
    )

    assert result.returncode == 0
    assert not curl_log.exists()


def test_report_failed_sends_v2_operation_envelope(tmp_path: Path) -> None:
    import json as jsonlib

    bin_dir, curl_log = _make_curl_recorder(tmp_path)
    session_id = "sess-test-12345"
    env = {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "ACS_APEX_SESSION_ID": session_id,
        "ACS_APEX_CALLBACK_URL": "https://apex.example.com/",
        "ACS_APEX_CALLBACK_TOKEN": "bearer-token-secret",
    }

    result = _source_and_call("report_failed 'test error message'", env)

    assert result.returncode == 0
    args = curl_log.read_text().splitlines()
    url = next(arg for arg in args if "/v1/internal/gpu-sessions/" in arg)
    assert f"/gpu-sessions/{session_id}/operations/" in url
    assert url.endswith("/events")
    body = jsonlib.loads(args[args.index("-d") + 1])
    assert body["schema_version"] == 2
    assert body["operation_kind"] == "session_bootstrap"
    assert body["status"] == "failed"
    assert body["phase"] is None
    assert body["sequence"] == 0
    assert body["target"] is None
    assert body["progress"] is None
    assert body["plan"] is None
    assert body["summary"] is None
    assert body["error"] == "test error message"
    assert body["operation_id"]
    assert body["event_id"]


def test_report_failed_fallback_without_jq_emits_valid_v2_json(tmp_path: Path) -> None:
    import json as jsonlib

    bin_dir, curl_log = _make_curl_recorder(tmp_path)
    bash_dir = str(Path(BASH).parent)
    system_dirs = [
        directory
        for directory in os.environ["PATH"].split(os.pathsep)
        if directory and not (Path(directory) / "jq").exists()
    ]
    if bash_dir not in system_dirs:
        system_dirs.insert(0, bash_dir)
    env = {
        "PATH": os.pathsep.join([str(bin_dir), *system_dirs]),
        "HOME": str(tmp_path),
        "ACS_APEX_SESSION_ID": "sess-nojq",
        "ACS_APEX_CALLBACK_URL": "https://apex.example.com",
        "ACS_APEX_CALLBACK_TOKEN": "tok",
    }

    result = _source_and_call("report_failed 'err with \"quote\"'", env)

    assert result.returncode == 0
    args = curl_log.read_text().splitlines()
    body = jsonlib.loads(args[args.index("-d") + 1])
    assert body["schema_version"] == 2
    assert body["error"] == 'err with "quote"'


def test_trap_fires_backstop_when_marker_is_absent(tmp_path: Path) -> None:
    bin_dir, curl_log = _make_curl_recorder(tmp_path)
    env = {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "ACS_APEX_SESSION_ID": "sess-trap-test",
        "ACS_APEX_CALLBACK_URL": "https://apex.example.com",
        "ACS_APEX_CALLBACK_TOKEN": "secret-token",
    }

    result = _source_and_call("false", env)

    assert result.returncode != 0
    assert curl_log.exists()


def test_backstop_skipped_when_acs_marker_exists(tmp_path: Path) -> None:
    bin_dir, curl_log = _make_curl_recorder(tmp_path)
    (tmp_path / ".aisha-acs-started").touch()
    env = {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "ACS_APEX_SESSION_ID": "sess-marker-test",
        "ACS_APEX_CALLBACK_URL": "https://apex.example.com",
        "ACS_APEX_CALLBACK_TOKEN": "secret-token",
    }

    result = _source_and_call("false", env)

    assert result.returncode != 0
    assert not curl_log.exists()


def test_no_secret_leak_on_failure(tmp_path: Path) -> None:
    """The callback token must never appear in stdout or stderr during a failing run."""
    bin_dir, _ = _make_curl_recorder(tmp_path)
    secret_token = "super-secret-token-xyz789"
    env = {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "ACS_APEX_SESSION_ID": "sess-leak-test",
        "ACS_APEX_CALLBACK_URL": "https://apex.example.com",
        "ACS_APEX_CALLBACK_TOKEN": secret_token,
    }

    result = _source_and_call("false", env)

    assert result.returncode != 0
    assert secret_token not in result.stdout, "token must not appear in stdout"
    assert secret_token not in result.stderr, "token must not appear in stderr"


# ---------------------------------------------------------------------------
# marker lifecycle (N1) — the marker means "this run" handed off to acs, not
# "acs has ever run on this node"
# ---------------------------------------------------------------------------


def test_stale_marker_is_cleared_at_start_of_run(tmp_path: Path) -> None:
    """main() must remove a pre-existing marker before any real work runs.

    ACS_GITHUB_TOKEN is deliberately absent so main() exits (via its own
    validation, not the ERR trap) right after the marker-cleanup line --
    enough to prove the clear happens without needing a full provisioning
    run.
    """
    bin_dir = make_path_stubs(tmp_path, [*_HEAVY_STUBS, "git"])
    marker = tmp_path / ".aisha-acs-started"
    marker.touch()
    env = {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "HOME": str(tmp_path),
    }

    result = _run(env)

    assert result.returncode == 2
    assert not marker.exists(), "stale marker from a previous run must be cleared at run start"


def test_backstop_fires_on_second_run_after_marker_cleanup(tmp_path: Path) -> None:
    """Regression guard: a second run on a node whose /workspace already has
    the marker (left by a prior run) must still report a pre-acs failure.

    Before the N1 fix, `report_failed`'s early `[[ -f .aisha-acs-started ]]`
    check treated any marker as "acs already took over," silencing every
    later run's pre-acs failures once one run had ever reached acs on this
    node. install_aisha's `uv pip install` is broken here so the run fails
    genuinely (uncaught, via the ERR trap) before run_deployment would touch
    the marker itself.
    """
    env = _full_env(
        tmp_path,
        {
            "ACS_APEX_SESSION_ID": "sess-second-run",
            "ACS_APEX_CALLBACK_URL": "https://apex.example.com",
            "ACS_APEX_CALLBACK_TOKEN": "tok",
        },
    )
    bin_dir = tmp_path / "bin"
    curl_log = tmp_path / "curl_calls.log"
    (bin_dir / "curl").write_text(f"#!/bin/sh\nprintf '%s\\n' \"$@\" >> {curl_log}\nexit 0\n")
    (bin_dir / "curl").chmod(0o755)
    # Break install_aisha's unguarded `uv pip install` step so the run fails
    # for real before run_deployment is ever reached.
    (bin_dir / "uv").write_text(
        "#!/bin/sh\n"
        'if [ "${1:-}" = "pip" ]; then exit 1; fi\n'
        'if [ "${1:-}" = "venv" ]; then exit 0; fi\n'
        "exit 0\n"
    )
    (bin_dir / "uv").chmod(0o755)

    # Simulate a marker left by a previously completed run on this node.
    (tmp_path / ".aisha-acs-started").touch()

    result = _run(env, timeout=30)

    assert result.returncode != 0, "the broken uv pip install must fail the run"
    assert curl_log.exists(), "a marker left by a previous run silenced this run's pre-acs backstop"


# ---------------------------------------------------------------------------
# new_uuid (N2) — every emitted id must be a parseable UUID, or empty
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("source", ["kernel", "uuidgen", "last_resort"])
def test_new_uuid_is_parseable_from_every_available_source(tmp_path: Path, source: str) -> None:
    """Every branch of new_uuid must emit a parseable UUID, or (last-resort
    only) emit nothing and warn -- never a malformed id.

    __KERNEL_UUID_PATH__ overrides the kernel source so each branch can be
    forced deterministically regardless of what the test host provides (CI
    runs on Linux, where the real kernel source always exists).
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    missing_kernel_path = tmp_path / "no-such-kernel-uuid-file"

    if source == "kernel":
        fake_kernel_uuid = tmp_path / "fake-kernel-uuid"
        fake_kernel_uuid.write_text("11111111-2222-4333-8444-555555555555\n")
        env = {
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "HOME": str(tmp_path),
            "__KERNEL_UUID_PATH__": str(fake_kernel_uuid),
        }
    elif source == "uuidgen":
        uuidgen_stub = bin_dir / "uuidgen"
        uuidgen_stub.write_text("#!/bin/sh\necho 22222222-3333-4444-8555-666666666666\n")
        uuidgen_stub.chmod(0o755)
        env = {
            "PATH": str(bin_dir),
            "HOME": str(tmp_path),
            "__KERNEL_UUID_PATH__": str(missing_kernel_path),
        }
    else:
        env = {
            # bin_dir only: no real uuidgen is reachable from here.
            "PATH": str(bin_dir),
            "HOME": str(tmp_path),
            "__KERNEL_UUID_PATH__": str(missing_kernel_path),
        }

    result = _source_and_call("new_uuid", env)

    assert result.returncode == 0, result.stderr
    if source == "last_resort":
        assert result.stdout == ""
        assert "no UUID source available" in result.stderr
    else:
        uuid.UUID(result.stdout.strip())  # raises ValueError if malformed


def test_report_failed_skipped_when_operation_id_is_empty(tmp_path: Path) -> None:
    """When no UUID source is available, APEX_OPERATION_ID resolves to
    empty; report_failed must decline rather than build a double-slash
    .../operations//events URL apex would reject."""
    bin_dir, curl_log = _make_curl_recorder(tmp_path)
    env = {
        # bin_dir only (uv/acs/rclone/curl stubs, no uuidgen): no real
        # uuidgen is reachable from here.
        "PATH": str(bin_dir),
        "HOME": str(tmp_path),
        "ACS_APEX_SESSION_ID": "sess-empty-oid",
        "ACS_APEX_CALLBACK_URL": "https://apex.example.com",
        "ACS_APEX_CALLBACK_TOKEN": "tok",
        "__KERNEL_UUID_PATH__": str(tmp_path / "no-such-kernel-uuid-file"),
    }

    result = _source_and_call("report_failed 'test error'", env)

    assert result.returncode == 0, result.stderr
    assert not curl_log.exists()


def test_agent_service_skipped_without_apex_settings(tmp_path: Path) -> None:
    env = _base_env(tmp_path)
    calls = tmp_path / "calls"
    acs = tmp_path / "aisha-venv" / "bin" / "acs"
    acs.parent.mkdir(parents=True)
    _write_executable(acs, f"#!/bin/sh\nprintf '%s\\n' \"$*\" >> {shlex.quote(str(calls))}\n")

    result = _source_and_call("install_agent_service", env)

    assert result.returncode == 0, result.stderr
    assert "agent service skipped" in result.stdout
    assert not calls.exists()


def test_agent_service_installs_after_bootstrap_when_apex_is_configured(tmp_path: Path) -> None:
    env = _base_env(tmp_path)
    calls = tmp_path / "calls"
    bin_dir = Path(env["PATH"].split(":", 1)[0])
    acs = tmp_path / "aisha-venv" / "bin" / "acs"
    acs.parent.mkdir(parents=True)
    _write_executable(acs, f"#!/bin/sh\nprintf '%s\\n' \"$*\" >> {shlex.quote(str(calls))}\n")
    _write_executable(bin_dir / "supervisorctl", "#!/bin/sh\nexit 0\n")
    env |= {
        "ACS_APEX_CALLBACK_URL": "https://apex.test",
        "ACS_APEX_CALLBACK_TOKEN": "token",
        "ACS_APEX_SESSION_ID": "session",
    }

    result = _source_and_call("install_agent_service", env)

    assert result.returncode == 0, result.stderr
    assert calls.read_text().strip() == "agent install-service"
