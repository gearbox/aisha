#!/usr/bin/env bash
# bench-provision.sh — where do the provisioning minutes go, and does hf_xet
# hold up on a 28 GB checkpoint?
#
# Runs four bundle variants that differ only in which install phases they
# declare, from a clean ComfyUI state each time, and records per-phase timings
# through aisha's ProvisioningTimer. Then optionally A/B tests the flagship
# checkpoint download with hf_xet on and off.
#
#   export ACS_BUNDLES_PATH=/workspace/bench-bundles      # repo ROOT
#   export ACS_COMFYUI_PATH=/opt/workspace-internal/ComfyUI
#   export ACS_COMFYUI_PYTHON=/venv/main/bin/python
#   export ACS_HF_TOKEN=... ACS_CIVITAI_API_TOKEN=...
#   bash bench-provision.sh 2>&1 | tee /workspace/provision-bench.log
#
# Flags:
#   --downloads-only   skip the matrix, run only the checkpoint A/B
#   --matrix-only      skip the A/B
#   --variants "a b"   override the variant list
#
# Requires: the phase2b-lite branch (ProvisioningTimer + `acs timings show`).

set -uo pipefail

WORK="${WORK:-/workspace/bench}"
VARIANTS="${VARIANTS:-v-full v-noco v-nolock v-thin}"
TIMINGS="${ACS_PROVISIONING_TIMING_PATH:-$WORK/provisioning-timings.jsonl}"
KEEP_MODELS="${KEEP_MODELS:-1}"   # models are identical across variants; re-download only if 0

RUN_MATRIX=1
RUN_DOWNLOADS=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --downloads-only) RUN_MATRIX=0; shift ;;
    --matrix-only)    RUN_DOWNLOADS=0; shift ;;
    --variants)       VARIANTS="$2"; shift 2 ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done

say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
warn() { printf '\033[33m   ! %s\033[0m\n' "$*"; }
die()  { printf '\033[31m   x %s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- preflight
say "0. Preflight"
command -v acs >/dev/null || die "acs not on PATH — activate the aisha venv"
: "${ACS_COMFYUI_PATH:?set ACS_COMFYUI_PATH to the image's ComfyUI install}"
: "${ACS_BUNDLES_PATH:?set ACS_BUNDLES_PATH to the bench registry ROOT}"

acs timings show --help >/dev/null 2>&1 \
  || die "this aisha build has no 'acs timings' — the phase2b-lite branch is required"

mkdir -p "$WORK"
export ACS_PROVISIONING_TIMING_PATH="$TIMINGS"

MODELS_DIR="$ACS_COMFYUI_PATH/models"
NODES_DIR="$ACS_COMFYUI_PATH/custom_nodes"
STASH="$WORK/models-stash"

echo "comfyui:  $ACS_COMFYUI_PATH"
echo "bundles:  $ACS_BUNDLES_PATH"
echo "timings:  $TIMINGS"
echo "free:     $(df -BG --output=avail "$WORK" | tail -1 | tr -dc '0-9')G"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | sed 's/^/gpu:      /'

# Snapshot the pristine custom_nodes set so each variant starts clean without
# destroying anything the base image shipped.
BASELINE_NODES="$WORK/.baseline-nodes.txt"
[[ -f "$BASELINE_NODES" ]] || ls -1 "$NODES_DIR" 2>/dev/null | sort > "$BASELINE_NODES"

reset_env() {
  # Custom nodes installed by a previous variant must go, or the next variant
  # measures an install that has already happened.
  if [[ -d "$NODES_DIR" ]]; then
    comm -13 "$BASELINE_NODES" <(ls -1 "$NODES_DIR" | sort) | while read -r n; do
      [[ -n "$n" ]] && rm -rf "${NODES_DIR:?}/$n"
    done
  fi
  # Models are identical across variants and cost ~7 min each time. Stash them
  # so the models phase is measured once, honestly, and skipped thereafter.
  if [[ "$KEEP_MODELS" == "1" && -d "$MODELS_DIR" ]]; then
    mkdir -p "$STASH"
    cp -al "$MODELS_DIR"/. "$STASH"/ 2>/dev/null || true
  fi
}

# ------------------------------------------------------------- 1. the matrix
if [[ "$RUN_MATRIX" == "1" ]]; then
  say "1. Provisioning matrix"
  warn "each variant deploys from a clean custom_nodes state; expect 60-75 min total"

  first=1
  for v in $VARIANTS; do
    say "1.$v"
    reset_env
    if [[ "$first" == "0" && "$KEEP_MODELS" == "1" ]]; then
      echo "  (models retained from the first variant — its 'models' phase is the honest one)"
    fi
    t0=$(date +%s)
    if acs deploy -b "$v" --no-verify >"$WORK/deploy-$v.out" 2>&1; then
      echo "  $v: $(( $(date +%s) - t0 ))s  (see $WORK/deploy-$v.out)"
    else
      warn "$v FAILED after $(( $(date +%s) - t0 ))s"
      tail -15 "$WORK/deploy-$v.out" | sed 's/^/      /'
    fi
    first=0
  done

  say "1.results"
  acs timings show --last "$(wc -w <<<"$VARIANTS")"
fi

# ------------------------------------------------- 2. checkpoint download A/B
if [[ "$RUN_DOWNLOADS" == "1" ]]; then
  say "2. Flagship checkpoint: hf_xet on vs off"

  : "${CKPT_URL:?set CKPT_URL to the 28 GB checkpoint's huggingface.co resolve URL}"
  CKPT_NAME="${CKPT_NAME:-bench_ckpt.safetensors}"

  measure() {   # $1=label $2=xet-enabled $3=outfile
    local label="$1" xet="$2" out="$3" t0 t1 secs bytes
    rm -rf "$WORK/dl"; mkdir -p "$WORK/dl"
    sync; t0=$(date +%s.%N)
    if ! env ACS_COMFYUI_PATH="$WORK/dl" ACS_HF_XET_ENABLED="$xet" \
         acs models fetch --url "$CKPT_URL" --model-type checkpoints \
           --filename "$CKPT_NAME" >"$WORK/$label.out" 2>&1
    then
      warn "$label FAILED"; tail -5 "$WORK/$label.out" | sed 's/^/      /'; return 0
    fi
    t1=$(date +%s.%N); secs=$(echo "$t1 - $t0" | bc)
    bytes=$(stat -c%s "$out" 2>/dev/null || echo 0)
    if [[ "$bytes" -lt 1048576 ]]; then
      warn "$label produced ${bytes}B — NO-OP, not a result"; return 0
    fi
    printf '  %-22s %8.1fs %9.1f MB/s  (%s bytes)\n' \
      "$label" "$secs" "$(echo "scale=1; $bytes/1048576/$secs" | bc)" "$bytes"
    rm -rf "$WORK/dl"
  }

  measure "xet-on"  true  "$WORK/dl/models/checkpoints/$CKPT_NAME"
  if [[ "${SKIP_SLOW_BASELINE:-0}" == "1" ]]; then
    echo "  xet-off: skipped (SKIP_SLOW_BASELINE=1) — compare against 16.2 MB/s from the 2a bench"
  else
    warn "xet-off on 28 GB at ~16 MB/s is roughly 30 minutes; set SKIP_SLOW_BASELINE=1 to skip"
    measure "xet-off" false "$WORK/dl/models/checkpoints/$CKPT_NAME"
  fi
fi

# ---------------------------------------------------------------- 3. summary
say "3. Summary"
cat <<'EOF'
Read the matrix like this:
  v-full minus v-noco     -> what the ComfyUI clone+checkout actually costs
  v-full minus v-nolock   -> what requirements.lock actually costs
  v-thin                  -> the floor: models + custom nodes only

Decide from it:
  lock < 60s              -> leave requirements.lock alone, no delta generator
  lock > 3min AND mostly  -> build the delta (cross-check base-manifest.json)
     already in the base
  comfyui < 30s           -> keep/drop comfyui: on correctness grounds, not speed
  comfyui > 2min          -> drop it; the base image's pin is better anyway
  v-thin < 4min           -> no image baking; the original Phase 2b stays withdrawn
  v-thin > 8min           -> revisit baking, but measure the image pull first
EOF
echo
echo "Raw timings: $TIMINGS   deploy logs: $WORK/deploy-<variant>.out"
echo "Copy off before terminating:  scp root@<host>:$TIMINGS ."
