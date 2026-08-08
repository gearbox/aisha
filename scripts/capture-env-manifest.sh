#!/usr/bin/env bash
# capture-env-manifest.sh — inventory the base image BEFORE anything installs.
#
# Run this as the very first command on a fresh node. Once `acs deploy` has run,
# the image's pristine package set is gone and this session can no longer answer
# "is requirements.lock mostly redundant against the base image?".
#
#   bash capture-env-manifest.sh > /workspace/base-manifest.json
#
# Reads: ACS_COMFYUI_PATH, ACS_COMFYUI_PYTHON (both defaulted for vastai/comfy).

set -uo pipefail

COMFYUI_PATH="${ACS_COMFYUI_PATH:-/opt/workspace-internal/ComfyUI}"
PY="${ACS_COMFYUI_PYTHON:-/venv/main/bin/python}"

[[ -x "$PY" ]] || { echo "interpreter not found: $PY" >&2; exit 1; }

comfyui_commit=""
comfyui_tag=""
if [[ -d "$COMFYUI_PATH/.git" ]]; then
  comfyui_commit=$(git -C "$COMFYUI_PATH" rev-parse HEAD 2>/dev/null || echo "")
  comfyui_tag=$(git -C "$COMFYUI_PATH" describe --tags --always 2>/dev/null || echo "")
fi

# ComfyUI ships its version in comfyui_version.py on recent releases; fall back
# to the git description when it is absent.
comfyui_version=$(
  "$PY" - "$COMFYUI_PATH" <<'PY' 2>/dev/null || echo ""
import pathlib, re, sys
p = pathlib.Path(sys.argv[1]) / "comfyui_version.py"
if p.exists():
    m = re.search(r'__version__\s*=\s*["\']([^"\']+)', p.read_text())
    print(m.group(1) if m else "")
PY
)
[[ -n "$comfyui_version" ]] || comfyui_version="$comfyui_tag"

baked_nodes=$(ls -1 "$COMFYUI_PATH/custom_nodes" 2>/dev/null | grep -v '^__' || true)

"$PY" - <<PY
import json, os, platform, subprocess, sys

def pip_list():
    try:
        out = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--format=json", "--disable-pip-version-check"],
            capture_output=True, text=True, timeout=180, check=True,
        ).stdout
        return {p["name"].lower(): p["version"] for p in json.loads(out)}
    except Exception as exc:          # noqa: BLE001 -- inventory must never abort the session
        print(f"pip list failed: {exc}", file=sys.stderr)
        return {}

packages = pip_list()

def ver(name):
    return packages.get(name)

torch_cuda = None
try:
    import torch
    torch_cuda = torch.version.cuda
except Exception:
    pass

manifest = {
    "schema": 1,
    "captured_before_install": True,
    "base_image": os.environ.get("ACS_BASE_IMAGE") or os.environ.get("VAST_CONTAINERLABEL") or None,
    "instance": os.environ.get("VAST_CONTAINERLABEL") or None,
    "python": platform.python_version(),
    "interpreter": sys.executable,
    "comfyui_path": ${COMFYUI_PATH@Q},
    "comfyui_version": ${comfyui_version@Q} or None,
    "comfyui_commit": ${comfyui_commit@Q} or None,
    "baked_custom_nodes": [n for n in ${baked_nodes@Q}.splitlines() if n],
    "torch": ver("torch"),
    "torch_cuda": torch_cuda,
    "torchvision": ver("torchvision"),
    "xformers": ver("xformers"),
    "package_count": len(packages),
    "packages": packages,
}
json.dump(manifest, sys.stdout, indent=2, sort_keys=True)
print()
PY
