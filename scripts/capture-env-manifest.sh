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
    m = re.search(r"__version__\s*=\s*[\"\x27]([^\"\x27]+)", p.read_text())
    print(m.group(1) if m else "")
PY
)
[[ -n "$comfyui_version" ]] || comfyui_version="$comfyui_tag"

baked_nodes=""
for node_path in "$COMFYUI_PATH/custom_nodes"/*; do
  [[ -e "$node_path" ]] || continue
  node_name=${node_path##*/}
  [[ "$node_name" == __* ]] || baked_nodes+="${baked_nodes:+$'\n'}$node_name"
done

BASE_IMAGE="${ACS_BASE_IMAGE:-}"
COMFYUI_PATH="$COMFYUI_PATH" COMFYUI_VERSION="$comfyui_version" \
COMFYUI_COMMIT="$comfyui_commit" BAKED_NODES="$baked_nodes" \
ACS_BASE_IMAGE="$BASE_IMAGE" "$PY" - <<'PY'
import json, os, platform, subprocess, sys

comfyui_path = os.environ["COMFYUI_PATH"]
comfyui_version = os.environ["COMFYUI_VERSION"]
comfyui_commit = os.environ["COMFYUI_COMMIT"]
baked_nodes = os.environ["BAKED_NODES"]

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
    "base_image": os.environ.get("ACS_BASE_IMAGE") or None,
    "instance": os.environ.get("VAST_CONTAINERLABEL") or None,
    "python": platform.python_version(),
    "interpreter": sys.executable,
    "comfyui_path": comfyui_path,
    "comfyui_version": comfyui_version or None,
    "comfyui_commit": comfyui_commit or None,
    "baked_custom_nodes": [n for n in baked_nodes.splitlines() if n],
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
