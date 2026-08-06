#!/usr/bin/env bash
# bench-download.sh — where do the 20 minutes go, and can a faster client fix it?
#
# Run on the Phase 1 authoring node, in one sitting, before touching Phase 2.
# Vast.ai host bandwidth varies by an order of magnitude between offers, so
# results are only comparable within a single run on a single instance.
#
#   scp bench-download.sh root@<host>:/workspace/ && ssh root@<host>
#   export HF_FILE_URL=... CIVITAI_URL=... [ACS_R2_* ...]
#   bash /workspace/bench-download.sh 2>&1 | tee /workspace/bench.log
#
# Reads: HF_FILE_URL, CIVITAI_URL, ACS_CIVITAI_API_TOKEN, ACS_HF_TOKEN (optional),
#        ACS_R2_* (optional — skips the R2 variant if unset)

set -uo pipefail

WORK="${WORK:-/workspace/bench}"
CEILING_BYTES="${CEILING_BYTES:-2147483648}"   # 2 GiB sample for the line-rate probe
CEILING_CONNS="${CEILING_CONNS:-8}"

: "${HF_FILE_URL:?set HF_FILE_URL to the .../resolve/main/<file> URL of the text encoder}"
: "${CIVITAI_URL:?set CIVITAI_URL to the https://civitai.red/api/download/models/<versionId> URL}"

mkdir -p "$WORK"; cd "$WORK"
RESULTS="$WORK/results.tsv"
: > "$RESULTS"

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
record() { printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" >> "$RESULTS"; }

# $1=label $2=bytes $3=command...
timed() {
  local label="$1" expect="$2"; shift 2
  sync; local t0 t1 secs got mbs
  t0=$(date +%s.%N)
  if ! "$@" >"$WORK/${label}.out" 2>&1; then
    echo "FAILED: $label (see $WORK/${label}.out)"; record "$label" "FAIL" "-" "-"; return 0
  fi
  t1=$(date +%s.%N)
  secs=$(echo "$t1 - $t0" | bc)
  got="${expect}"
  mbs=$(echo "scale=1; $got / 1048576 / $secs" | bc)
  printf '  %-28s %8.1fs  %8s MB/s\n' "$label" "$secs" "$mbs"
  record "$label" "$(printf '%.1f' "$secs")" "$mbs" "$got"
}

# ---------------------------------------------------------------- node facts
say "0. Node"
echo "instance: ${VAST_CONTAINERLABEL:-unknown}   $(nproc) cores"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1
df -h "$WORK" | tail -1
echo "disk write:"; dd if=/dev/zero of="$WORK/.ddtest" bs=1M count=2048 conv=fsync 2>&1 | tail -1
rm -f "$WORK/.ddtest"

# ------------------------------------------------------- 1. line-rate ceiling
# N parallel range GETs against the HF CDN. This is the number every client
# below is competing against — if they all land near it, the link is the
# constraint and no client change will help.
say "1. Line-rate ceiling (${CEILING_CONNS} parallel range GETs, $((CEILING_BYTES/1024/1024)) MiB)"
chunk=$(( CEILING_BYTES / CEILING_CONNS ))
ceiling_dl() {
  local pids=()
  for i in $(seq 0 $((CEILING_CONNS-1))); do
    local start=$(( i * chunk )) end=$(( (i+1) * chunk - 1 ))
    curl -sL ${ACS_HF_TOKEN:+-H "Authorization: Bearer $ACS_HF_TOKEN"} \
         -r "${start}-${end}" -o "$WORK/.ceil.$i" "$HF_FILE_URL" &
    pids+=($!)
  done
  wait "${pids[@]}"
}
timed "ceiling-${CEILING_CONNS}conn" "$CEILING_BYTES" ceiling_dl
rm -f "$WORK"/.ceil.*

# --------------------------------------------------------- 2. HF text encoder
say "2. HF text encoder (~8 GB)"
HF_REPO="$(sed -E 's#https://huggingface.co/([^/]+/[^/]+)/resolve/.*#\1#' <<<"$HF_FILE_URL")"
HF_PATH="$(sed -E 's#.*/resolve/[^/]+/##' <<<"$HF_FILE_URL")"
HF_BYTES="$(curl -sIL ${ACS_HF_TOKEN:+-H "Authorization: Bearer $ACS_HF_TOKEN"} "$HF_FILE_URL" \
            | awk 'BEGIN{IGNORECASE=1}/^content-length:/{v=$2}END{gsub(/\r/,"",v);print v}')"
echo "repo=$HF_REPO path=$HF_PATH bytes=${HF_BYTES:-unknown}"

# A — what aisha does today: one httpx connection, streamed
rm -rf "$WORK/a"; mkdir -p "$WORK/a"
timed "A-aisha-httpx" "$HF_BYTES" \
  env ACS_COMFYUI_PATH="$WORK/a" acs models fetch \
      --url "$HF_FILE_URL" --model-type clip --filename bench_clip.safetensors

pip install -q "huggingface_hub[hf_xet]" 2>/dev/null || true

# B — hf_xet defaults
rm -rf "$WORK/b" ~/.cache/huggingface; mkdir -p "$WORK/b"
timed "B-hf_xet-default" "$HF_BYTES" \
  hf download "$HF_REPO" "$HF_PATH" --local-dir "$WORK/b"

# C — hf_xet, concurrency turned up
rm -rf "$WORK/c" ~/.cache/huggingface; mkdir -p "$WORK/c"
timed "C-hf_xet-tuned" "$HF_BYTES" \
  env HF_XET_HIGH_PERFORMANCE=1 HF_XET_NUM_CONCURRENT_RANGE_GETS=32 \
  hf download "$HF_REPO" "$HF_PATH" --local-dir "$WORK/c"

# ------------------------------------------------------- 3. Civitai checkpoint
say "3. Civitai checkpoint (~8 GB)"
CIV_BYTES="$(curl -sIL -H "Authorization: Bearer ${ACS_CIVITAI_API_TOKEN:-}" "$CIVITAI_URL" \
             | awk 'BEGIN{IGNORECASE=1}/^content-length:/{v=$2}END{gsub(/\r/,"",v);print v}')"
echo "bytes=${CIV_BYTES:-unknown}"

# D — aisha today, direct from Civitai
rm -rf "$WORK/d"; mkdir -p "$WORK/d"
timed "D-civitai-httpx" "$CIV_BYTES" \
  env ACS_COMFYUI_PATH="$WORK/d" acs models fetch \
      --url "$CIVITAI_URL" --model-type checkpoints --filename bench_ckpt.safetensors

# E — from R2, the Phase 1 path (needs the weight already pushed)
if [[ -n "${ACS_R2_READONLY_ACCESS_KEY_ID:-}" && -n "${BENCH_SHA256:-}" ]]; then
  rm -rf "$WORK/e"; mkdir -p "$WORK/e"
  timed "E-r2-rclone-${ACS_RCLONE_MULTI_THREAD_STREAMS:-4}x" "$CIV_BYTES" \
    env RCLONE_S3_ACCESS_KEY_ID="$ACS_R2_READONLY_ACCESS_KEY_ID" \
        RCLONE_S3_SECRET_ACCESS_KEY="$ACS_R2_READONLY_SECRET_ACCESS_KEY" \
    rclone copyto --s3-provider Cloudflare --s3-endpoint "$ACS_R2_S3_ENDPOINT" \
        --s3-no-check-bucket \
        --multi-thread-streams "${ACS_RCLONE_MULTI_THREAD_STREAMS:-4}" \
        -- ":s3:${ACS_R2_MODEL_CACHE_BUCKET}/models/by-sha256/${BENCH_SHA256}" \
        "$WORK/e/bench_ckpt.safetensors"
else
  echo "  skipped E — set ACS_R2_* and BENCH_SHA256 (the checkpoint's digest) to include it"
fi

# --------------------------------------------------- 4. combined, as provisioned
say "4. Combined — both files, the way deploy actually does it"
# Cross-file concurrency already exists (asyncio.gather + semaphore, default 3),
# so this measures the real starting point, not a hypothetical.
if [[ -n "${BENCH_BUNDLE:-}" ]]; then
  rm -rf "$WORK/f"; mkdir -p "$WORK/f"
  TOTAL=$(( ${HF_BYTES:-0} + ${CIV_BYTES:-0} ))
  timed "F-acs-deploy-models" "$TOTAL" \
    env ACS_COMFYUI_PATH="$WORK/f" acs deploy -b "$BENCH_BUNDLE" --models-only --no-verify
else
  echo "  skipped F — set BENCH_BUNDLE to the bundle name to include it"
fi

# ----------------------------------------------------------------- 5. results
say "5. Results"
printf '%-30s %10s %12s\n' VARIANT SECONDS MB/s
awk -F'\t' '{printf "%-30s %10s %12s\n", $1, $2, $3}' "$RESULTS"
cat <<'EOF'

Read it like this:
  * Every variant close to the ceiling  -> the node's link is the constraint.
    Raise hardware.min_network_download_mbps and re-measure. No code change helps.
  * C >> A  -> intra-file concurrency is the win. Adopt hf_xet for HF URLs.
  * E >> D  -> the R2 cache is earning its keep for Civitai. Keep it there.
  * E ~= D  -> the cache is not buying speed; keep it only for the reasons that
    survive (immutability, Early Access gating, version deletion).
  * F ~= max(individual times), not the sum -> cross-file parallelism is already
    working and there is nothing to gain from "downloading both at once".
EOF
