"""Regression coverage for the benchmark and environment-capture scripts."""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
PROVISION_BENCH = ROOT / "scripts" / "bench" / "bench-provision.sh"
CAPTURE_MANIFEST = ROOT / "scripts" / "capture-env-manifest.sh"
BASH = shutil.which("bash") or "/bin/bash"


def _write_acs_stub(tmp_path: Path) -> Path:
    """Create the small `acs` subset needed by the downloads-only benchmark."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    acs = bin_dir / "acs"
    acs.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ $1 == timings && $2 == show ]]; then exit 0; fi\n"
        "if [[ $1 == models && $2 == fetch ]]; then\n"
        "  shift 2\n"
        "  while [[ $# -gt 0 ]]; do\n"
        "    if [[ $1 == --filename ]]; then name=$2; shift 2; continue; fi\n"
        "    shift\n"
        "  done\n"
        "  target=$ACS_COMFYUI_PATH/models/checkpoints/$name\n"
        '  mkdir -p "$(dirname "$target")"\n'
        '  head -c 1048576 /dev/zero > "$target"\n'
        "fi\n"
    )
    acs.chmod(acs.stat().st_mode | stat.S_IXUSR)

    date = bin_dir / "date"
    date.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ $1 == +%s.%N ]]; then python3 -c 'import time; print(time.time())'; "
        'else /bin/date "$@"; fi\n'
    )
    date.chmod(date.stat().st_mode | stat.S_IXUSR)

    # bench-provision.sh uses GNU-style `stat -c%s`, which real GNU stat (Linux
    # CI runners) already understands natively. Only macOS's BSD stat needs the
    # `-f %z` translation; applying it unconditionally on Linux makes stat run
    # in filesystem-status mode instead, whose "  File: ..." banner then blows
    # up the caller's `[[ "$bytes" -lt ... ]]` arithmetic under `set -u`.
    stat_command = bin_dir / "stat"
    stat_command.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ $1 == -c%s && "$(uname)" == Darwin ]]; then /usr/bin/stat -f %z "$2"; '
        'else /usr/bin/stat "$@"; fi\n'
    )
    stat_command.chmod(stat_command.stat().st_mode | stat.S_IXUSR)

    sha256sum = bin_dir / "sha256sum"
    sha256sum.write_text('#!/usr/bin/env bash\n/usr/bin/shasum -a 256 "$@"\n')
    sha256sum.chmod(sha256sum.stat().st_mode | stat.S_IXUSR)
    return bin_dir


def test_provision_benchmark_reports_net_and_gross_and_cleans_download(tmp_path: Path) -> None:
    bin_dir = _write_acs_stub(tmp_path)
    work = tmp_path / "work"
    comfyui = tmp_path / "ComfyUI"
    (comfyui / "custom_nodes").mkdir(parents=True)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "WORK": str(work),
        "MIN_FREE_GB": "1",
        "ACS_COMFYUI_PATH": str(comfyui),
        "ACS_BUNDLES_PATH": str(tmp_path / "bundles"),
        "CKPT_URL": "https://huggingface.co/example/repo/resolve/main/tiny.safetensors",
        "CKPT_NAME": "tiny.safetensors",
        "SKIP_SLOW_BASELINE": "1",
    }

    result = subprocess.run(
        [BASH, str(PROVISION_BENCH), "--downloads-only"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "net " in result.stdout
    assert "gross " in result.stdout, result.stdout
    assert "warm-cache hash" in result.stdout
    assert not (work / "dl").exists()


def test_capture_manifest_handles_shell_sensitive_metadata(tmp_path: Path) -> None:
    comfyui = tmp_path / "ComfyUI ' quoted"
    node = comfyui / "custom_nodes" / "node ' quoted"
    node.mkdir(parents=True)

    result = subprocess.run(
        [BASH, str(CAPTURE_MANIFEST)],
        env={
            **os.environ,
            "ACS_COMFYUI_PATH": str(comfyui),
            "ACS_COMFYUI_PYTHON": sys.executable,
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads(result.stdout)
    assert manifest["comfyui_path"] == str(comfyui)
    assert manifest["baked_custom_nodes"] == [node.name]
