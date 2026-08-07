#!/usr/bin/env bash
# bench-download-v2.sh — where do the provisioning minutes go?
#
# Fixes over v1:
#   * throughput is computed from the bytes actually written, never from a
#     declared size or a HEAD probe (Civitai presigned URLs reject HEAD)
#   * a variant that produces < 1 MiB is reported as NO-OP, not as a fast result
#   * every output is deleted immediately after measuring (v1 needed ~30 GB free)
#   * the R2 variant seeds the object first, so it measures a transfer
#   * the line-rate probe runs last, warm, to /dev/null, and is advisory only
#   * preflight checks fail early with a named cause instead of mid-run
#
# Usage:
#   scp bench-download-v2.sh root@<host>:/workspace/
#   ssh root@<host>
#   export HF_FILE_URL=... CIVITAI_URL=... ACS_CIVITAI_API_TOKEN=...
#   export ACS_R2_S3_ENDPOINT=... ACS_R2_MODEL_CACHE_BUCKET=...
#   export ACS_R2_WRITE_ACCESS_KEY_ID=... ACS_R2_WRITE_SECRET_ACCESS_KEY=...
#   export ACS_R2_READONLY_ACCESS_KEY_ID=... ACS_R2_READONLY_SECRET_ACCESS_KEY=...
#   bash /workspace/bench-download-v2.sh 2>&1 | tee /workspace/bench2.log
#
# Optional: ACS_HF_TOKEN, BENCH_BUNDLE (+ ACS_BUNDLES_PATH or ACS_BUNDLES_REPO),
#           KEEP=1 to retain downloads, XET_STREAMS (default 32)

set -uo pipefail

WORK="${WORK:-/workspace/bench}"
KEEP="${KEEP:-0}"
XET_STREAMS="${XET_STREAMS:-32}"
CEILING_CONNS="${CEILING_CONNS:-8}"
CEILING_BYTES="${CEILING_BYTES:-2147483648}"
MIN_FREE_GB="${MIN_FREE_GB:-25}"

RESULTS="$WORK/results.tsv"

say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
warn() { printf '\033[33m   ! %s\033[0m\n' "$*"; }
die()  { printf '\033[31m   x %s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- preflight
say "0. Preflight"

: "${HF_FILE_URL:?set HF_FILE_URL to a https://huggingface.co/<repo>/resolve/<rev>/<file> URL}"
: "${CIVITAI_URL:?set CIVITAI_URL to a https://civitai.red/api/download/models/<versionId> URL}"

for tool in curl bc awk stat sha256sum; do
  command -v "$tool" >/dev/null || die "missing required tool: $tool"
done
command -v acs >/dev/null || die "acs not on PATH — activate the aisha venv first"

HF_CLI=""
if command -v hf >/dev/null; then HF_CLI="hf"
elif command -v huggingface-cli >/dev/null; then HF_CLI="huggingface-cli"
fi

R2_READY=0
if [[ -n "${ACS_R2_S3_ENDPOINT:-}" && -n "${ACS_R2_MODEL_CACHE_BUCKET:-}" \
   && -n "${ACS_R2_WRITE_ACCESS_KEY_ID:-}" && -n "${ACS_R2_READONLY_ACCESS_KEY_ID:-}" ]]; then
  if command -v rclone >/dev/null; then R2_READY=1
  else warn "rclone not installed — R2 variants will be skipped"; fi
else
  warn "ACS_R2_* incomplete — R2 variants will be skipped (need endpoint, bucket, write AND readonly keys)"
fi

if [[ -n "${BENCH_BUNDLE:-}" && -z "${ACS_BUNDLES_PATH:-}" && -z "${ACS_BUNDLES_REPO:-}" ]]; then
  warn "BENCH_BUNDLE set but neither ACS_BUNDLES_PATH nor ACS_BUNDLES_REPO is —"
  warn "section 4 will fail with 'No default registry configured'. Unsetting BENCH_BUNDLE."
  BENCH_BUNDLE=""
fi

mkdir -p "$WORK" || die "cannot create $WORK"
cd "$WORK"
: > "$RESULTS"

# Keep every HF cache inside WORK so it can be wiped and accounted for.
export HF_HOME="$WORK/.hf"

FREE_GB=$(df -BG --output=avail "$WORK" | tail -1 | tr -dc '0-9')
echo "instance:   ${VAST_CONTAINERLABEL:-unknown}   cores=$(nproc)"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | sed 's/^/gpu:        /'
echo "free disk:  ${FREE_GB}G at $WORK"
[[ "$FREE_GB" -lt "$MIN_FREE_GB" ]] && \
  warn "less than ${MIN_FREE_GB}G free — outputs are deleted between variants, but a large bundle may still not fit"
echo -n "disk write: "
dd if=/dev/zero of="$WORK/.ddtest" bs=1M count=2048 conv=fsync 2>&1 | tail -1
rm -f "$WORK/.ddtest"

# ------------------------------------------------------------------ helpers

# measure <label> <expected-output-path> <cmd...>
# Times the command, derives throughput from the bytes actually produced, and
# removes the output unless KEEP=1.
measure() {
  local label="$1" out="$2"; shift 2
  local t0 t1 secs bytes mbs rc

  sync
  t0=$(date +%s.%N)
  "$@" >"$WORK/${label}.out" 2>&1; rc=$?
  t1=$(date +%s.%N)
  secs=$(echo "$t1 - $t0" | bc)

  bytes=$(stat -c%s "$out" 2>/dev/null || echo 0)

  if [[ $rc -ne 0 ]]; then
    printf '  %-24s \033[31mFAILED (exit %d)\033[0m  see %s\n' "$label" "$rc" "$WORK/${label}.out"
    tail -3 "$WORK/${label}.out" | sed 's/^/      /'
    printf '%s\tFAIL\t-\t%s\n' "$label" "$bytes" >> "$RESULTS"
    return 0
  fi
  if [[ "$bytes" -lt 1048576 ]]; then
    printf '  %-24s \033[33mNO-OP\033[0m produced %s bytes in %.1fs — nothing was transferred\n' \
      "$label" "$bytes" "$secs"
    printf '%s\tNOOP\t-\t%s\n' "$label" "$bytes" >> "$RESULTS"
    return 0
  fi

  mbs=$(echo "scale=1; $bytes / 1048576 / $secs" | bc)
  printf '  %-24s %8.1fs %10s MB/s  (%s bytes)\n' "$label" "$secs" "$mbs" "$bytes"
  printf '%s\t%.1f\t%s\t%s\n' "$label" "$secs" "$mbs" "$bytes" >> "$RESULTS"
}

drop() { [[ "$KEEP" == "1" ]] || rm -rf "$@"; }

# ------------------------------------------------------- 1. HF text encoder
say "1. HF weight — single connection vs Xet"

HF_REPO="$(sed -E 's#https://huggingface.co/([^/]+/[^/]+)/resolve/.*#\1#' <<<"$HF_FILE_URL")"
HF_PATH="$(sed -E 's#.*/resolve/[^/]+/##' <<<"$HF_FILE_URL")"
HF_NAME="$(basename "$HF_PATH")"
echo "repo=$HF_REPO  path=$HF_PATH"

# Warm the CDN edge so variant order does not decide the winner.
echo -n "warming edge... "
curl -sL ${ACS_HF_TOKEN:+-H "Authorization: Bearer $ACS_HF_TOKEN"} \
     -r 0-1048575 -o /dev/null "$HF_FILE_URL" && echo ok || echo "failed (continuing)"

# A — what ModelDownloader does today: one httpx connection, streamed
rm -rf "$WORK/a"; mkdir -p "$WORK/a"
measure A-aisha-httpx "$WORK/a/models/clip/$HF_NAME" \
  env ACS_COMFYUI_PATH="$WORK/a" acs models fetch \
      --url "$HF_FILE_URL" --model-type clip --filename "$HF_NAME"
drop "$WORK/a"

if [[ -n "$HF_CLI" ]]; then
  pip install -q -U "huggingface_hub[hf_xet]" 2>/dev/null || warn "hf_xet install failed; B/C may fall back to plain HTTP"

  # B — Xet defaults
  rm -rf "$WORK/b" "$HF_HOME"; mkdir -p "$WORK/b"
  measure B-hf_xet-default "$WORK/b/$HF_PATH" \
    "$HF_CLI" download "$HF_REPO" "$HF_PATH" --local-dir "$WORK/b"
  drop "$WORK/b" "$HF_HOME"

  # C — Xet with concurrency raised
  rm -rf "$WORK/c" "$HF_HOME"; mkdir -p "$WORK/c"
  measure "C-hf_xet-${XET_STREAMS}x" "$WORK/c/$HF_PATH" \
    env HF_XET_HIGH_PERFORMANCE=1 HF_XET_NUM_CONCURRENT_RANGE_GETS="$XET_STREAMS" \
    "$HF_CLI" download "$HF_REPO" "$HF_PATH" --local-dir "$WORK/c"
  drop "$WORK/c" "$HF_HOME"
else
  warn "no hf/huggingface-cli on PATH — skipping B and C, which are the point of this run"
fi

# --------------------------------------------------- 2. Civitai + R2 round-trip
say "2. Civitai weight — direct vs R2"

CIV_NAME="bench_ckpt.safetensors"
CIV_FILE="$WORK/d/models/checkpoints/$CIV_NAME"

# D — Civitai direct, single httpx connection (what deploy does today)
rm -rf "$WORK/d"; mkdir -p "$WORK/d"
measure D-civitai-httpx "$CIV_FILE" \
  env ACS_COMFYUI_PATH="$WORK/d" acs models fetch \
      --url "$CIVITAI_URL" --model-type checkpoints --filename "$CIV_NAME"

if [[ -s "$CIV_FILE" && "$R2_READY" == "1" ]]; then
  echo -n "  hashing for the cache key... "
  BENCH_SHA256=$(sha256sum "$CIV_FILE" | cut -d' ' -f1); echo "$BENCH_SHA256"
  R2_KEY="models/by-sha256/${BENCH_SHA256}"

  # Seed the object with WRITE credentials. Raw rclone rather than `acs cache
  # push` so this measures transport only and needs no bundle to exist.
  echo -n "  seeding $R2_KEY ... "
  if env RCLONE_S3_ACCESS_KEY_ID="$ACS_R2_WRITE_ACCESS_KEY_ID" \
         RCLONE_S3_SECRET_ACCESS_KEY="$ACS_R2_WRITE_SECRET_ACCESS_KEY" \
       rclone copyto --s3-provider Cloudflare --s3-endpoint "$ACS_R2_S3_ENDPOINT" \
         --s3-no-check-bucket --s3-chunk-size "${ACS_RCLONE_CHUNK_SIZE_MB:-128}M" \
         -- "$CIV_FILE" ":s3:${ACS_R2_MODEL_CACHE_BUCKET}/${R2_KEY}" \
         >"$WORK/seed.out" 2>&1
  then echo "ok"; else echo "FAILED"; tail -3 "$WORK/seed.out" | sed 's/^/      /'; R2_READY=0; fi

  drop "$WORK/d"   # free the 8 GB before pulling it back

  if [[ "$R2_READY" == "1" ]]; then
    # E — same bytes from R2 with READ-ONLY credentials, 4 streams (aisha default)
    rm -rf "$WORK/e"; mkdir -p "$WORK/e"
    measure E-r2-rclone-4x "$WORK/e/$CIV_NAME" \
      env RCLONE_S3_ACCESS_KEY_ID="$ACS_R2_READONLY_ACCESS_KEY_ID" \
          RCLONE_S3_SECRET_ACCESS_KEY="$ACS_R2_READONLY_SECRET_ACCESS_KEY" \
      rclone copyto --s3-provider Cloudflare --s3-endpoint "$ACS_R2_S3_ENDPOINT" \
          --s3-no-check-bucket --multi-thread-streams 4 \
          -- ":s3:${ACS_R2_MODEL_CACHE_BUCKET}/${R2_KEY}" "$WORK/e/$CIV_NAME"
    drop "$WORK/e"

    # E8 — same, 8 streams, to see whether rclone concurrency is the lever
    rm -rf "$WORK/e8"; mkdir -p "$WORK/e8"
    measure E-r2-rclone-8x "$WORK/e8/$CIV_NAME" \
      env RCLONE_S3_ACCESS_KEY_ID="$ACS_R2_READONLY_ACCESS_KEY_ID" \
          RCLONE_S3_SECRET_ACCESS_KEY="$ACS_R2_READONLY_SECRET_ACCESS_KEY" \
      rclone copyto --s3-provider Cloudflare --s3-endpoint "$ACS_R2_S3_ENDPOINT" \
          --s3-no-check-bucket --multi-thread-streams 8 \
          -- ":s3:${ACS_R2_MODEL_CACHE_BUCKET}/${R2_KEY}" "$WORK/e8/$CIV_NAME"
    drop "$WORK/e8"
  fi
else
  [[ -s "$CIV_FILE" ]] || warn "Civitai download produced nothing — check $WORK/D-civitai-httpx.out"
  drop "$WORK/d"
fi

# ------------------------------------------------ 3. combined, as provisioned
say "3. Combined — the whole bundle, the way deploy does it"
if [[ -n "${BENCH_BUNDLE:-}" ]]; then
  rm -rf "$WORK/f"; mkdir -p "$WORK/f"
  T0=$(date +%s.%N)
  if env ACS_COMFYUI_PATH="$WORK/f" acs deploy -b "$BENCH_BUNDLE" --models-only --no-verify \
       >"$WORK/F-deploy.out" 2>&1
  then
    T1=$(date +%s.%N)
    SECS=$(echo "$T1 - $T0" | bc)
    TOT=$(find "$WORK/f/models" -type f -printf '%s\n' 2>/dev/null | awk '{s+=$1}END{print s+0}')
    printf '  %-24s %8.1fs %10s MB/s  (%s bytes total)\n' "F-acs-deploy-models" "$SECS" \
      "$(echo "scale=1; $TOT/1048576/$SECS" | bc)" "$TOT"
    printf 'F-acs-deploy-models\t%.1f\t%s\t%s\n' "$SECS" \
      "$(echo "scale=1; $TOT/1048576/$SECS" | bc)" "$TOT" >> "$RESULTS"
  else
    printf '  %-24s \033[31mFAILED\033[0m see %s\n' "F-acs-deploy-models" "$WORK/F-deploy.out"
    tail -5 "$WORK/F-deploy.out" | sed 's/^/      /'
  fi
  drop "$WORK/f"
else
  echo "  skipped — set BENCH_BUNDLE (and ACS_BUNDLES_PATH or ACS_BUNDLES_REPO)"
fi

# ------------------------------------------------------- 4. advisory ceiling
# Runs last, against a warm edge, to /dev/null. Naive byte-range parallelism is
# NOT the Xet path, so treat this as a lower bound on the link, never a ceiling.
say "4. Link probe (advisory — ${CEILING_CONNS} parallel range GETs, warm edge, /dev/null)"
chunk=$(( CEILING_BYTES / CEILING_CONNS ))
t0=$(date +%s.%N); pids=()
for i in $(seq 0 $((CEILING_CONNS-1))); do
  curl -sL ${ACS_HF_TOKEN:+-H "Authorization: Bearer $ACS_HF_TOKEN"} \
       -r "$(( i*chunk ))-$(( (i+1)*chunk - 1 ))" -o /dev/null "$HF_FILE_URL" &
  pids+=($!)
done
wait "${pids[@]}" 2>/dev/null
t1=$(date +%s.%N); secs=$(echo "$t1 - $t0" | bc)
printf '  %-24s %8.1fs %10s MB/s\n' "link-${CEILING_CONNS}conn" "$secs" \
  "$(echo "scale=1; $CEILING_BYTES/1048576/$secs" | bc)"

# ----------------------------------------------------------------- 5. results
say "5. Results"
printf '%-24s %10s %12s %18s\n' VARIANT SECONDS MB/s BYTES
awk -F'\t' '{printf "%-24s %10s %12s %18s\n", $1, $2, $3, $4}' "$RESULTS"

cat <<'EOF'

Interpretation:
  C >> A            intra-file concurrency is the win. Adopt hf_xet for HF URLs.
  E ~= D            the R2 cache buys no speed for Civitai; keep it only for
                    immutability, Early Access gating and version deletion.
  E >> D            the cache earns its keep on speed too; keep it general.
  E-8x >> E-4x      raise ACS_RCLONE_MULTI_THREAD_STREAMS from its default of 4.
  D >> A            single-stream throughput is source-dependent: Civitai/R2 is
                    fine on one connection, HF's Xet endpoint is not.
  NO-OP             the variant produced nothing. Not a fast result — a bug.
EOF
echo
echo "Raw TSV: $RESULTS   per-variant logs: $WORK/<label>.out"
