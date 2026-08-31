#!/bin/bash
# ==============================================================================
# AISHA — Vast.ai PROVISIONING_SCRIPT for vastai/comfy-based templates
# ==============================================================================
# Runs inside the image's blessed environment while /.provisioning is held:
#   - /venv/main is activated for us
#   - supervisord has NOT started yet (it will, after we exit cleanly)
#   - ComfyUI lives at /opt/workspace-internal/ComfyUI (image-baked)
#   - The image's provisioning chain has reconciled /workspace; $WORKSPACE/ComfyUI
#     and /opt/workspace-internal/ComfyUI are the same logical location.
#
# Our job: deploy the requested bundle. The image's supervisord starts ComfyUI
# after we exit. The Vast.ai Instance Portal owns the named cloudflared tunnel
# (it reads CF_TUNNEL_TOKEN from the instance env at boot).
#
# Required env (set on the Vast.ai instance, not in the template):
#   ACS_BUNDLE           — bundle name (e.g. qwen_rapid_aio)
#   ACS_GITHUB_TOKEN     — to clone gearbox/aisha + gearbox/ai-bundles
#
# Optional env:
#   ACS_CF_TUNNEL_TOKEN      — accepted but ignored; the Instance Portal reads it directly
#   ACS_BUNDLE_VERSION       — pin a bundle version
#   ACS_HF_TOKEN             — for HuggingFace model downloads during deploy
#   ACS_HF_CACHE_PATH        — HF_HOME for the hf_xet chunk cache; default $WORKSPACE/.aisha-cache/hf
#   ACS_HF_XET_ENABLED       — "false" forces the httpx path instead of hf_xet (debugging only)
#   ACS_HF_XET_CONCURRENT_RANGE_GETS — hf_xet intra-file range-GET concurrency; default 32
#   ACS_APEX_SESSION_ID      — apex session UUID, echoed in the ready line
#   ACS_APEX_OPERATION_ID    — optional Apex operation UUID for bootstrap telemetry
#   ACS_AISHA_BRANCH         — defaults to "master"
#   ACS_BUNDLES_BRANCH       — defaults to "master"
#   ACS_MODELS_ONLY          — "true" to skip non-model deploy steps
#   ACS_NO_VERIFY            — "true" to skip checksum verification
#   ACS_COMFYUI_PYTHON       — Python interpreter owning ComfyUI's venv; default /venv/main/bin/python
#   ACS_COMFYUI_PORT         — port ComfyUI binds to; default 18188
#   ACS_BASE_IMAGE           — image tag set by the Vast.ai template; enables manifest staleness checks
#   ACS_R2_MODEL_CACHE_BUCKET / ACS_R2_S3_ENDPOINT — R2 cache location, supplied by the Vast.ai template (not Apex)
#   ACS_R2_READONLY_ACCESS_KEY_ID / ACS_R2_READONLY_SECRET_ACCESS_KEY — read-only cache credentials, supplied by the Vast.ai template (not Apex)
#   ACS_RCLONE_PATH / ACS_RCLONE_* — rclone executable and transfer tuning, supplied by the Vast.ai template (not Apex)
#   ACS_RCLONE_VERSION        — pinned rclone version installed when rclone is absent; default v1.71.0
#   ACS_AISHA_BIN             — aisha-owned executable directory; default $WORKSPACE/aisha-bin
# ==============================================================================

set -Eeuo pipefail
# -E (errtrace): without it, the ERR trap below is not inherited by shell
# functions -- and virtually all real work here (clone_or_update_repo,
# install_aisha, run_deployment, main itself) runs inside a function. Omitting
# it would mean report_failed's trap silently never fires for a real failure.

# ==============================================================================
# Logging
# ==============================================================================
# Defined before new_uuid() (and the rest of Configuration) on purpose: the
# top-level operation-id auto-generation below calls new_uuid(), which can
# call log_warn() on its last-resort path, before main() ever runs.
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info()    { echo -e "${CYAN}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $1" >&2; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1" >&2; }
log_step()    { echo -e "${BLUE}[STEP]${NC} $1"; }

new_uuid() {
    # Linux nodes expose a kernel UUID at /proc/sys/kernel/random/uuid; keep
    # the script sourceable in minimal shells too by falling back to uuidgen.
    # Neither is expected to be missing on a Linux GPU node. When both are,
    # warn and produce nothing rather than a hand-rolled approximation:
    # printf can't reliably respect hex-group widths from $RANDOM's
    # 0-32767 range, so a fabricated id is worse than no id. Callers must
    # treat an empty result as "no id available" (see report_failed's
    # operation-id guard).
    #
    # __KERNEL_UUID_PATH__ lets the test suite force this chain
    # deterministically on a system where the kernel source is always
    # present; it is never set in production.
    local kernel_uuid_path="${__KERNEL_UUID_PATH__:-/proc/sys/kernel/random/uuid}"
    if [[ -r "$kernel_uuid_path" ]]; then
        cat "$kernel_uuid_path"
    elif command -v uuidgen >/dev/null 2>&1; then
        uuidgen
    else
        log_warn "no UUID source available (no ${kernel_uuid_path}, no uuidgen); leaving id empty"
    fi
}

# ==============================================================================
# Configuration (override via env)
# ==============================================================================
AISHA_REPO="${ACS_AISHA_REPO:-https://github.com/gearbox/aisha.git}"
BUNDLES_REPO="${ACS_BUNDLES_REPO:-https://github.com/gearbox/ai-bundles.git}"
AISHA_BRANCH="${ACS_AISHA_BRANCH:-master}"
BUNDLES_BRANCH="${ACS_BUNDLES_BRANCH:-master}"

WORKSPACE="${ACS_WORKSPACE:-/workspace}"
AISHA_PATH="${ACS_AISHA_PATH:-$WORKSPACE/aisha}"
# ACS_BUNDLES_PATH is the ai-bundles repository root. Keep the clone target
# separate from the exported setting so it cannot accidentally become /bundles.
BUNDLES_REPO_PATH="${ACS_BUNDLES_PATH:-$WORKSPACE/ai-bundles}"
COMFYUI_PATH="${ACS_COMFYUI_PATH:-$WORKSPACE/ComfyUI}"
CACHE_PATH="${ACS_CACHE_PATH:-$WORKSPACE/.aisha-cache}"
BASE_MANIFEST="${CACHE_PATH}/base-manifest.json"
SUPERSEDED_BASE_MANIFEST="${CACHE_PATH}/base-manifest.superseded.json"
# Keep `acs snapshot` pointed at the same location as the provisioning capture,
# including when ACS_WORKSPACE changes the default workspace root.
export ACS_CACHE_PATH="$CACHE_PATH"
RCLONE_VERSION="${ACS_RCLONE_VERSION:-v1.71.0}"

# Dedicated venv for aisha. Placed under /workspace so it survives pause/resume
# and matches where the rest of aisha-owned state lives (repo + bundles).
# Idempotent reuse: created on first boot, reused thereafter. If you ever need
# to force-recreate, delete the directory and re-run.
AISHA_VENV="${ACS_AISHA_VENV:-$WORKSPACE/aisha-venv}"
ACS_BIN="${AISHA_VENV}/bin/acs"
# Keep non-Python tools out of the venv directory: check_rclone runs before
# install_aisha on first boot, and uv requires an empty/nonexistent target when
# creating a venv. This directory also takes precedence over image or distro
# managed binaries and remains discoverable by Aisha's Python rclone wrapper.
AISHA_BIN_DIR="${ACS_AISHA_BIN:-$WORKSPACE/aisha-bin}"
export PATH="${AISHA_BIN_DIR}:${AISHA_VENV}/bin:${PATH}"

# Auth
GITHUB_TOKEN="${ACS_GITHUB_TOKEN:-}"

# Deployment
BUNDLE="${ACS_BUNDLE:-}"
BUNDLE_VERSION="${ACS_BUNDLE_VERSION:-}"
MODELS_ONLY="${ACS_MODELS_ONLY:-false}"
NO_VERIFY="${ACS_NO_VERIFY:-false}"

# HuggingFace (consumed by acs deploy)
HF_TOKEN="${ACS_HF_TOKEN:-}"
export HF_TOKEN

# hf_xet chunk cache. Off the root filesystem on purpose -- on a node already
# near capacity this lands *in addition to* the weight itself (see
# Settings.hf_home). Mirrors the aisha-internal default of cache_path/"hf" so
# `acs` and this script agree even when neither ACS_HF_CACHE_PATH nor
# ACS_CACHE_PATH is set.
HF_HOME="${ACS_HF_CACHE_PATH:-$WORKSPACE/.aisha-cache/hf}"
export HF_HOME

# Apex operation callbacks (consumed by acs deploy and the pre-acs backstop)
APEX_SESSION_ID="${ACS_APEX_SESSION_ID:-}"
APEX_CALLBACK_URL="${ACS_APEX_CALLBACK_URL:-}"
APEX_CALLBACK_TOKEN="${ACS_APEX_CALLBACK_TOKEN:-}"
# Keep the bash backstop and acs on one operation when Apex provisioned a
# session but did not supply an id. Local/no-Apex deploys retain the CLI's
# generated-id behaviour and do not receive a meaningless bootstrap flag.
APEX_OPERATION_ID="${ACS_APEX_OPERATION_ID:-}"
if [[ -z "$APEX_OPERATION_ID" && -n "$APEX_SESSION_ID" ]]; then
    APEX_OPERATION_ID="$(new_uuid)"
fi

# ==============================================================================
# Apex terminal-failure callback — best-effort backstop before acs takes over
# ==============================================================================

report_failed() {
    trap - ERR
    local error_msg="${1:-provisioning failed}"
    [[ -f "${WORKSPACE}/.aisha-acs-started" ]] && return 0
    [[ -z "${APEX_CALLBACK_URL:-}" || -z "${APEX_SESSION_ID:-}" || -z "${APEX_CALLBACK_TOKEN:-}" \
        || -z "${APEX_OPERATION_ID:-}" ]] && return 0

    local elapsed
    elapsed=$(( $(date +%s) - ${start_time:-$(date +%s)} ))
    local ts event_id
    ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    event_id="$(new_uuid)"
    local url="${APEX_CALLBACK_URL%/}/v1/internal/gpu-sessions/${APEX_SESSION_ID}/operations/${APEX_OPERATION_ID}/events"

    # Prefer jq for proper escaping. The literal fallback remains deliberately
    # narrow: trap text is script-controlled and only needs JSON quoting.
    local payload
    if command -v jq >/dev/null 2>&1; then
        payload="$(jq -nc \
            --arg sid "$APEX_SESSION_ID" --arg oid "$APEX_OPERATION_ID" \
            --arg eid "$event_id" --arg ts "$ts" --arg err "$error_msg" \
            --argjson elapsed "$elapsed" \
            '{schema_version:2, event_id:$eid, session_id:$sid, operation_id:$oid,
              operation_kind:"session_bootstrap", batch:null, sequence:0, target:null,
              status:"failed", phase:null, started_at:$ts, ts:$ts,
              elapsed_seconds:$elapsed, phase_elapsed_seconds:null, progress:null,
              plan:null, summary:null, message:"provisioning script aborted", error:$err}')"
    else
        local safe_err="${error_msg//\\/\\\\}"
        safe_err="${safe_err//\"/\\\"}"
        payload="{\"schema_version\":2,\"event_id\":\"${event_id}\",\"session_id\":\"${APEX_SESSION_ID}\",\"operation_id\":\"${APEX_OPERATION_ID}\",\"operation_kind\":\"session_bootstrap\",\"batch\":null,\"sequence\":0,\"target\":null,\"status\":\"failed\",\"phase\":null,\"started_at\":\"${ts}\",\"ts\":\"${ts}\",\"elapsed_seconds\":${elapsed},\"phase_elapsed_seconds\":null,\"progress\":null,\"plan\":null,\"summary\":null,\"message\":\"provisioning script aborted\",\"error\":\"${safe_err}\"}"
    fi

    curl --silent --show-error --max-time 5 \
        -X POST "$url" \
        -H "Authorization: Bearer ${APEX_CALLBACK_TOKEN}" \
        -H "Content-Type: application/json" \
        -d "$payload" >/dev/null || true
}
# shellcheck disable=SC2154  # rc is assigned by rc=$? at the start of the trap body
trap 'rc=$?; echo "[FATAL] aisha-provision-comfyui failed at line $LINENO with exit $rc" >&2; report_failed "aborted at line $LINENO (exit $rc)"' ERR

# ==============================================================================
# Helpers
# ==============================================================================

# Build an authenticated clone URL by embedding the token. NOTE: this token
# ends up in the freshly-cloned repo's .git/config as a remembered remote URL,
# which is a credential-leak surface. We sanitize it immediately after the
# clone completes (see sanitize_remote_url) so the on-disk state has only
# token-less URLs. Subsequent fetches re-inject auth at request time via
# an http.extraheader override.
get_authenticated_url() {
    local url="$1"
    if [[ -n "$GITHUB_TOKEN" && "$url" == https://github.com/* ]]; then
        url="${url/https:\/\/github.com/https://${GITHUB_TOKEN}@github.com}"
    fi
    echo "$url"
}

# Strip any embedded token from `origin`'s URL after the initial clone,
# so .git/config persists the canonical (token-less) URL on /workspace.
# Idempotent: a no-op if the URL is already token-free.
sanitize_remote_url() {
    local path="$1"
    local clean_url="$2"
    (
        cd "$path"
        git remote set-url origin "$clean_url"
    )
}

# Echo the per-invocation HTTP Authorization header used for token-protected
# fetches against github.com. Empty output if no token is set, so callers can
# safely splice it into `git -c http.extraheader=...` only when present.
github_auth_header_arg() {
    if [[ -n "$GITHUB_TOKEN" ]]; then
        # `Bearer` works for both classic PATs and fine-grained tokens.
        printf 'http.https://github.com/.extraheader=Authorization: Bearer %s' "$GITHUB_TOKEN"
    fi
}

clone_or_update_repo() {
    local name="$1"
    local url="$2"
    local path="$3"
    local branch="${4:-master}"

    local auth_url
    auth_url=$(get_authenticated_url "$url")

    # Pre-compute the auth header (empty string if no token); used on
    # updates so we don't have to re-embed the token into the remote URL.
    local auth_header
    auth_header=$(github_auth_header_arg)

    if [[ -d "$path/.git" ]]; then
        log_info "Updating $name..."
        (
            cd "$path"
            # Inject auth via a one-shot config entry. -c is per-invocation,
            # so the header is never persisted to .git/config.
            if [[ -n "$auth_header" ]]; then
                git -c "$auth_header" fetch origin "$branch" --depth=1
            else
                git fetch origin "$branch" --depth=1
            fi
            # Hard-reset to the fetched ref. No silent fallback — if this
            # fails, provisioning aborts rather than running against a
            # stale or unintended revision.
            git reset --hard "origin/$branch"
        )
    else
        log_info "Cloning $name..."
        git clone --branch "$branch" --depth 1 "$auth_url" "$path"
        # Replace the token-bearing remote URL with the canonical one.
        # Subsequent fetches will use the auth header instead.
        sanitize_remote_url "$path" "$url"
    fi

    # Sanity check: after either path, HEAD must be reachable on the
    # requested branch. This catches the case where someone hand-edited
    # the on-disk repo between boots.
    local head_branch
    head_branch=$(cd "$path" && git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")
    if [[ -z "$head_branch" ]]; then
        log_error "$name at $path has no resolvable HEAD after sync"
        exit 1
    fi

    log_success "$name ready at $path (HEAD=$head_branch)"
}

check_uv() {
    # uv must be pre-installed in the base image. We deliberately do NOT
    # `curl | sh` from astral.sh at runtime — that would execute whatever
    # the upstream serves at provisioning time, which is a supply-chain risk
    # we can avoid because vastai/comfy-derived images already ship uv at
    # /opt/instance-tools/bin/uv (verified in Phase 3 inspection).
    if ! command -v uv &>/dev/null; then
        log_error "uv is not on PATH. Expected from base image (e.g. /opt/instance-tools/bin/uv)."
        log_error "If the base image no longer ships uv, install it deterministically at image-build time."
        exit 1
    fi
    log_info "uv resolved to: $(command -v uv) ($(uv --version 2>&1))"
    log_success "check_uv"
}

_rclone_install_dir() {
    local install_dir="$AISHA_BIN_DIR"
    mkdir -p "$install_dir" || return 1
    printf '%s\n' "$install_dir"
}

_install_pinned_rclone() (
    set -euo pipefail

    local version="$1"
    local machine arch archive base_url checksum_file temp_dir install_dir staged_binary
    local expected_checksum installed_line

    if [[ ! "$version" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        log_error "ACS_RCLONE_VERSION must be a release tag such as v1.71.0; got ${version@Q}"
        return 1
    fi

    machine="$(uname -m)"
    case "$machine" in
        x86_64|amd64) arch="amd64" ;;
        aarch64|arm64) arch="arm64" ;;
        *)
            log_error "unsupported rclone architecture: $machine (supported: x86_64/amd64, aarch64/arm64)"
            return 1
            ;;
    esac

    install_dir="$(_rclone_install_dir)" || {
        log_error "could not create aisha-owned rclone directory at ${AISHA_BIN_DIR}"
        return 1
    }
    log_info "rclone install directory: ${install_dir}"
    temp_dir="$(mktemp -d "${TMPDIR:-/tmp}/aisha-rclone.XXXXXX")" || {
        log_error "could not create temporary directory for rclone installation"
        return 1
    }
    trap 'rm -rf "$temp_dir"' EXIT

    archive="rclone-${version}-linux-${arch}.zip"
    base_url="https://downloads.rclone.org/${version}"
    checksum_file="${temp_dir}/SHA256SUMS"

    curl --fail --show-error --silent --location --proto '=https' --proto-redir '=https' \
        --output "${temp_dir}/${archive}" "${base_url}/${archive}"
    curl --fail --show-error --silent --location --proto '=https' --proto-redir '=https' \
        --output "$checksum_file" "${base_url}/SHA256SUMS"

    expected_checksum="$(awk -v archive="$archive" '$2 == archive || $2 == "*" archive {print $1; exit}' "$checksum_file")"
    if [[ ! "$expected_checksum" =~ ^[0-9a-fA-F]{64}$ ]]; then
        log_error "official rclone checksum manifest has no valid entry for ${archive}"
        return 1
    fi
    printf '%s  %s\n' "$expected_checksum" "$archive" | (
        cd "$temp_dir"
        sha256sum --check --status -
    ) || {
        log_error "rclone archive checksum verification failed for ${archive}"
        return 1
    }

    unzip -q "${temp_dir}/${archive}" -d "$temp_dir"
    if [[ ! -f "${temp_dir}/rclone-${version}-linux-${arch}/rclone" ]]; then
        log_error "rclone archive did not contain the expected binary"
        return 1
    fi

    staged_binary="${install_dir}/.rclone.${version}.$$"
    install -m 0755 "${temp_dir}/rclone-${version}-linux-${arch}/rclone" "$staged_binary"
    mv -f "$staged_binary" "${install_dir}/rclone"

    installed_line="$(rclone version 2>/dev/null | sed -n '1p')"
    if [[ "$installed_line" != "rclone ${version}" ]]; then
        log_error "installed rclone version mismatch: expected rclone ${version}, got ${installed_line:-unavailable}"
        return 1
    fi
)

check_rclone() {
    local installed_line version
    if ! command -v curl &>/dev/null; then
        log_error "curl is not on PATH. Install the curl package in the base image."
        return 1
    fi
    if ! command -v unzip &>/dev/null; then
        log_error "unzip is not on PATH. Install the unzip package in the base image."
        return 1
    fi
    version="$RCLONE_VERSION"
    version="${version#"${version%%[![:space:]]*}"}"
    version="${version%"${version##*[![:space:]]}"}"
    if [[ ! "$version" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        log_error "ACS_RCLONE_VERSION must be a release tag such as v1.71.0; got ${RCLONE_VERSION@Q}"
        return 1
    fi
    installed_line="$(rclone version 2>/dev/null | sed -n '1p' || true)"
    if [[ "$installed_line" == "rclone ${version}" ]]; then
        log_info "rclone present: ${installed_line}"
        return 0
    fi

    if command -v rclone >/dev/null 2>&1; then
        log_info "replacing rclone ${installed_line:-with unavailable version} with pinned ${version}"
    else
        log_step "Installing rclone ${version}"
    fi
    _install_pinned_rclone "$version" || {
        log_error "rclone install failed -- the R2 model cache will be unavailable"
        return 1
    }
}

install_aisha() {
    log_step "starting install_aisha"

    if [[ ! -x "${AISHA_VENV}/bin/python" ]]; then
        log_info "Creating aisha venv at ${AISHA_VENV}"
        # Pin the template interpreter: this avoids a managed-Python download
        # and version-matches the ComfyUI template by construction.
        uv venv "${AISHA_VENV}" --python "${ACS_COMFYUI_PYTHON}" --quiet
    else
        log_info "Reusing existing aisha venv at ${AISHA_VENV}"
    fi

    cd "$AISHA_PATH"
    uv pip install -e . --python "${AISHA_VENV}/bin/python" --quiet

    if [[ ! -x "${ACS_BIN}" ]]; then
        log_error "acs entrypoint not found at ${ACS_BIN} after install"
        exit 1
    fi
    log_success "install_aisha (venv=${AISHA_VENV})"
}

get_manifest_base_image() {
    local python="${ACS_COMFYUI_PYTHON:-/venv/main/bin/python}"

    [[ -x "$python" ]] || return 0
    "$python" - "$BASE_MANIFEST" <<'PY' 2>/dev/null || true
import json
import pathlib
import sys

try:
    manifest = json.loads(pathlib.Path(sys.argv[1]).read_text())
except (OSError, ValueError):
    manifest = {}
base_image = manifest.get("base_image") if isinstance(manifest, dict) else None
print(base_image if isinstance(base_image, str) else "")
PY
}

get_manifest_captured_before_install() {
    local python="${ACS_COMFYUI_PYTHON:-/venv/main/bin/python}"

    [[ -x "$python" ]] || return 0
    "$python" - "$BASE_MANIFEST" <<'PY' 2>/dev/null || true
import json
import pathlib
import sys

try:
    manifest = json.loads(pathlib.Path(sys.argv[1]).read_text())
except (OSError, ValueError):
    manifest = {}
captured = manifest.get("captured_before_install") if isinstance(manifest, dict) else None
print("true" if captured is True else "false")
PY
}

capture_base_manifest() {
    # This must happen before rclone, aisha, or bundle requirements can install
    # anything. On a resumed node preserve the first-boot inventory: replacing
    # it after a bundle has installed would make future overlays too small.
    local recapturing=false
    local captured_before_install=true
    if [[ -z "${ACS_BASE_IMAGE:-}" ]]; then
        log_info "base_manifest_staleness_detection_disabled reason=ACS_BASE_IMAGE_unset"
    fi
    if [[ -f "$BASE_MANIFEST" ]]; then
        local manifest_image
        captured_before_install="$(get_manifest_captured_before_install)"
        if [[ "$captured_before_install" != "true" ]]; then
            log_warn "base_manifest_not_pristine captured_before_install=${captured_before_install:-missing}; reusing existing manifest and preserving $SUPERSEDED_BASE_MANIFEST"
            return 0
        else
            manifest_image="$(get_manifest_base_image)"
            if [[ -z "${ACS_BASE_IMAGE:-}" || -z "$manifest_image" ]]; then
                log_info "Reusing pristine base manifest at $BASE_MANIFEST"
                return 0
            fi
            if [[ "$manifest_image" == "$ACS_BASE_IMAGE" ]]; then
                log_info "Reusing pristine base manifest at $BASE_MANIFEST"
                return 0
            fi
            log_warn "base_manifest_stale manifest_base_image=${manifest_image} live_base_image=${ACS_BASE_IMAGE}; recapturing"
            recapturing=true
        fi
    fi

    if [[ "$recapturing" == true ]]; then
        captured_before_install=false
    fi

    log_step "Capturing pristine base environment manifest"
    mkdir -p "$CACHE_PATH"
    local tmp_manifest
    tmp_manifest="$(mktemp "${CACHE_PATH}/.base-manifest.XXXXXX")" || {
        log_error "Could not create a temporary base manifest in $CACHE_PATH"
        exit 1
    }

    if ACS_COMFYUI_PATH="$COMFYUI_PATH" \
        ACS_COMFYUI_PYTHON="${ACS_COMFYUI_PYTHON:-/venv/main/bin/python}" \
        ACS_BASE_IMAGE="${ACS_BASE_IMAGE:-}" \
        ACS_MANIFEST_CAPTURED_BEFORE_INSTALL="$captured_before_install" \
        bash "$AISHA_PATH/scripts/capture-env-manifest.sh" > "$tmp_manifest"
    then
        if [[ "$recapturing" == true && ! -f "$SUPERSEDED_BASE_MANIFEST" ]]; then
            if ! mv "$BASE_MANIFEST" "$SUPERSEDED_BASE_MANIFEST"; then
                rm -f "$tmp_manifest"
                log_error "Could not preserve superseded base manifest at $SUPERSEDED_BASE_MANIFEST"
                exit 1
            fi
        elif [[ "$recapturing" == true ]]; then
            log_warn "base_manifest_superseded_exists preserving $SUPERSEDED_BASE_MANIFEST"
        fi
        if ! mv "$tmp_manifest" "$BASE_MANIFEST"; then
            rm -f "$tmp_manifest"
            log_error "Could not save base manifest at $BASE_MANIFEST"
            exit 1
        fi
    else
        rm -f "$tmp_manifest"
        log_error "capture_base_manifest failed; overlays cannot be generated on this node"
        exit 1
    fi
    log_success "capture_base_manifest ($BASE_MANIFEST)"
}

run_deployment() {
    log_step "starting run_deployment: $BUNDLE"

    # ACS_BUNDLES_PATH is the ai-bundles repository root, where
    # bundle-index.yaml is stored. Settings uses that index to resolve bundles.
    export ACS_BUNDLES_PATH="$BUNDLES_REPO_PATH"

    export ACS_COMFYUI_PORT="${ACS_COMFYUI_PORT:-18188}"
    log_info "ACS_COMFYUI_PORT=${ACS_COMFYUI_PORT}"

    local cmd=("${ACS_BIN}" deploy
        --bundle "$BUNDLE"
        --comfyui "$COMFYUI_PATH"
        --bootstrap)

    [[ -n "$APEX_OPERATION_ID" ]] && cmd+=(--operation-id "$APEX_OPERATION_ID")
    [[ -n "$BUNDLE_VERSION" ]] && cmd+=(--bundle-version "$BUNDLE_VERSION")
    [[ "$MODELS_ONLY" == "true" ]] && cmd+=(--models-only)
    [[ "$NO_VERIFY" == "true" ]] && cmd+=(--no-verify)

    # Marks that *this run* has handed terminal reporting to acs; report_failed
    # treats its presence as "acs may already be reporting" and stays silent.
    # Cleared at the top of every run in main(), so a marker left by a
    # previous run on this node's persistent /workspace can never silence
    # this run's pre-acs backstop.
    touch "${WORKSPACE}/.aisha-acs-started"
    "${cmd[@]}"

    log_success "run_deployment"
}

# ==============================================================================
# Main
# ==============================================================================

main() {
    echo "=================================================="
    echo "  AISHA - Vast.ai Provisioning (ComfyUI variant)"
    echo "=================================================="
    echo ""

    local start_time
    start_time=$(date +%s)

    # Clear any marker left by a previous run on this node's persistent
    # /workspace before anything that could trip the ERR trap runs. A stale
    # marker from a prior boot would otherwise silence this run's own
    # pre-acs backstop (see run_deployment's touch site).
    rm -f "${WORKSPACE}/.aisha-acs-started"

    # Validate required env
    if [[ -z "$GITHUB_TOKEN" ]]; then
        log_error "ACS_GITHUB_TOKEN is required to clone gearbox/aisha + gearbox/ai-bundles"
        exit 2
    fi
    if [[ -z "$BUNDLE" ]]; then
        log_error "ACS_BUNDLE is required"
        exit 2
    fi

    log_info "session_id=${APEX_SESSION_ID} bundle=${BUNDLE}"

    # uv ships in the base image; check_uv only validates its presence before
    # the base inventory is captured.
    check_uv

    # Point Aisha's ComfyUIManager at the image's blessed ComfyUI venv.  Fail
    # before cloning or capturing anything so an invalid interpreter cannot
    # produce a misleading/stale base manifest.
    export ACS_COMFYUI_PYTHON="${ACS_COMFYUI_PYTHON:-/venv/main/bin/python}"
    if [[ ! -x "$ACS_COMFYUI_PYTHON" ]]; then
        log_error "ACS_COMFYUI_PYTHON not executable: $ACS_COMFYUI_PYTHON"
        exit 1
    fi
    log_info "ACS_COMFYUI_PYTHON=${ACS_COMFYUI_PYTHON}"

    # Repos in parallel.
    # `wait` doesn't always trip `set -e`, so check the exit status explicitly
    # and abort provisioning on any sync failure — otherwise install_aisha
    # would run against an empty/stale checkout and the error would surface
    # much later as a confusing import or deploy failure.
    log_step "Syncing repositories..."
    clone_or_update_repo "aisha" "$AISHA_REPO" "$AISHA_PATH" "$AISHA_BRANCH" &
    local pid_aisha=$!
    clone_or_update_repo "ai-bundles" "$BUNDLES_REPO" "$BUNDLES_REPO_PATH" "$BUNDLES_BRANCH" &
    local pid_bundles=$!

    local sync_failed=0
    if ! wait "$pid_aisha"; then
        log_error "aisha repo sync failed"
        sync_failed=1
    fi
    if ! wait "$pid_bundles"; then
        log_error "ai-bundles repo sync failed"
        sync_failed=1
    fi
    if (( sync_failed )); then
        exit 1
    fi

    # This is intentionally before check_rclone: the latter may install a
    # pinned binary, while this manifest must describe the untouched image.
    capture_base_manifest
    check_rclone

    # Aisha CLI + bundle deploy
    install_aisha
    run_deployment

    # Final structured ready line (grepped by apex and humans)
    local elapsed
    elapsed=$(($(date +%s) - start_time))
    echo "acs.provision.ready session_id=${APEX_SESSION_ID} elapsed=${elapsed}s bundle=${BUNDLE}"
}

# Run main unless the script is being sourced (e.g., by the test harness).
# In production (direct exec or curl|bash), __SOURCED__ is never set so main runs.
[[ "${__SOURCED__:-}" == "1" ]] || main "$@"
