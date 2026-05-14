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
# Our job: deploy the requested bundle and configure cloudflared. The image's
# supervisord will start ComfyUI and cloudflared after we exit.
#
# Required env (set on the Vast.ai instance, not in the template):
#   ACS_BUNDLE           — bundle name (e.g. qwen_rapid_aio)
#   ACS_GITHUB_TOKEN     — to clone gearbox/aisha + gearbox/ai-bundles
#   ACS_CF_TUNNEL_TOKEN  — cloudflared tunnel token
#
# Optional env:
#   ACS_BUNDLE_VERSION       — pin a bundle version
#   ACS_HF_TOKEN             — for HuggingFace model downloads during deploy
#   ACS_APEX_SESSION_ID      — apex session UUID, echoed in the ready line
#   ACS_AISHA_BRANCH         — defaults to "master"
#   ACS_BUNDLES_BRANCH       — defaults to "master"
#   ACS_MODELS_ONLY          — "true" to skip non-model deploy steps
#   ACS_NO_VERIFY            — "true" to skip checksum verification
# ==============================================================================

set -euo pipefail
trap 'echo "[FATAL] aisha-provision-comfyui failed at line $LINENO with exit $?" >&2' ERR

# ==============================================================================
# Version pins
# ==============================================================================
# Verify cloudflared still reads TUNNEL_TOKEN from env when bumping.
CLOUDFLARED_VERSION="2024.12.2"

# ==============================================================================
# Configuration (override via env)
# ==============================================================================
AISHA_REPO="${ACS_AISHA_REPO:-https://github.com/gearbox/aisha.git}"
BUNDLES_REPO="${ACS_BUNDLES_REPO:-https://github.com/gearbox/ai-bundles.git}"
AISHA_BRANCH="${ACS_AISHA_BRANCH:-master}"
BUNDLES_BRANCH="${ACS_BUNDLES_BRANCH:-master}"

WORKSPACE="${ACS_WORKSPACE:-/workspace}"
AISHA_PATH="${ACS_AISHA_PATH:-$WORKSPACE/aisha}"
BUNDLES_PATH="${ACS_BUNDLES_PATH:-$WORKSPACE/ai-bundles}"
COMFYUI_PATH="${ACS_COMFYUI_PATH:-$WORKSPACE/ComfyUI}"

# Dedicated venv for aisha. Placed under /workspace so it survives pause/resume
# and matches where the rest of aisha-owned state lives (repo + bundles).
# Idempotent reuse: created on first boot, reused thereafter. If you ever need
# to force-recreate, delete the directory and re-run.
AISHA_VENV="${ACS_AISHA_VENV:-$WORKSPACE/aisha-venv}"
ACS_BIN="${AISHA_VENV}/bin/acs"

# Auth
GITHUB_TOKEN="${ACS_GITHUB_TOKEN:-}"

# Deployment
BUNDLE="${ACS_BUNDLE:-}"
BUNDLE_VERSION="${ACS_BUNDLE_VERSION:-}"
MODELS_ONLY="${ACS_MODELS_ONLY:-false}"
NO_VERIFY="${ACS_NO_VERIFY:-false}"

# Cloudflare tunnel
CF_TUNNEL_TOKEN="${ACS_CF_TUNNEL_TOKEN:-}"
CLOUDFLARED_BIN=""

# HuggingFace (consumed by acs deploy)
HF_TOKEN="${ACS_HF_TOKEN:-}"
export HF_TOKEN

# Apex context (informational only — echoed in the ready line)
APEX_SESSION_ID="${ACS_APEX_SESSION_ID:-}"

# Supervisord drop-in
SUPERVISOR_LOG_DIR="${ACS_SUPERVISOR_LOG_DIR:-/var/log/aisha}"
SUPERVISOR_CONF_PATH="${ACS_SUPERVISOR_CONF_PATH:-/etc/supervisor/conf.d/aisha-cloudflared.conf}"

# ==============================================================================
# Logging
# ==============================================================================
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

# ==============================================================================
# Helpers
# ==============================================================================

# Escape a value for supervisord's environment="K=v,..." stanza.
# Supervisord's parser is comma-separated with double-quoted values;
# backslash, double-quote, and comma must all be backslash-escaped.
escape_for_supervisord_env() {
    local value=${1-}
    value=${value//\\/\\\\}
    value=${value//\"/\\\"}
    value=${value//,/\\,}
    printf '%s' "$value"
}

get_authenticated_url() {
    local url="$1"
    if [[ -n "$GITHUB_TOKEN" && "$url" == https://github.com/* ]]; then
        url="${url/https:\/\/github.com/https://${GITHUB_TOKEN}@github.com}"
    fi
    echo "$url"
}

clone_or_update_repo() {
    local name="$1"
    local url="$2"
    local path="$3"
    local branch="${4:-master}"

    local auth_url
    auth_url=$(get_authenticated_url "$url")

    if [[ -d "$path/.git" ]]; then
        log_info "Updating $name..."
        (
            cd "$path"
            git fetch origin "$branch" --depth=1 2>/dev/null || true
            git reset --hard "origin/$branch" 2>/dev/null || git pull --ff-only
        )
    else
        log_info "Cloning $name..."
        git clone --branch "$branch" --depth 1 "$auth_url" "$path"
    fi
    log_success "$name ready at $path"
}

install_cloudflared() {
    log_step "starting install_cloudflared"

    if command -v cloudflared &>/dev/null; then
        log_success "cloudflared already installed: $(cloudflared --version 2>&1)"
    else
        log_info "Installing cloudflared ${CLOUDFLARED_VERSION}..."
        if command -v dpkg &>/dev/null; then
            local deb_url="https://github.com/cloudflare/cloudflared/releases/download/${CLOUDFLARED_VERSION}/cloudflared-linux-amd64.deb"
            curl -fsSL "$deb_url" -o /tmp/cloudflared.deb
            dpkg -i /tmp/cloudflared.deb
            rm -f /tmp/cloudflared.deb
        else
            local bin_url="https://github.com/cloudflare/cloudflared/releases/download/${CLOUDFLARED_VERSION}/cloudflared-linux-amd64"
            curl -fsSL "$bin_url" -o /usr/local/bin/cloudflared
            chmod +x /usr/local/bin/cloudflared
        fi
        log_success "cloudflared installed: $(cloudflared --version 2>&1)"
    fi

    CLOUDFLARED_BIN="$(command -v cloudflared)"
    if [[ -z "$CLOUDFLARED_BIN" ]]; then
        log_error "cloudflared not on PATH after install"
        exit 1
    fi
    log_info "cloudflared resolved to: $CLOUDFLARED_BIN"
    log_success "install_cloudflared"
}

install_uv() {
    if command -v uv &>/dev/null; then
        log_success "uv already installed"
        return 0
    fi
    log_info "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    log_success "uv installed"
}

install_aisha() {
    log_step "starting install_aisha"

    if [[ ! -x "${AISHA_VENV}/bin/python" ]]; then
        log_info "Creating aisha venv at ${AISHA_VENV}"
        uv venv "${AISHA_VENV}" --quiet
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

run_deployment() {
    log_step "starting run_deployment: $BUNDLE"

    local cmd=("${ACS_BIN}" deploy
        --bundle "$BUNDLE"
        --bundles-path "$BUNDLES_PATH/bundles"
        --comfyui "$COMFYUI_PATH")

    [[ -n "$BUNDLE_VERSION" ]] && cmd+=(--version "$BUNDLE_VERSION")
    [[ "$MODELS_ONLY" == "true" ]] && cmd+=(--models-only)
    [[ "$NO_VERIFY" == "true" ]] && cmd+=(--no-verify)

    "${cmd[@]}"

    log_success "run_deployment"
}

write_cloudflared_dropin() {
    log_step "starting write_cloudflared_dropin"

    if [[ -z "$CF_TUNNEL_TOKEN" ]]; then
        log_warn "ACS_CF_TUNNEL_TOKEN not set; cloudflared will not be configured — apex will be unable to reach this node"
        # Clear any stale drop-in from a prior boot under a different config.
        rm -f "$SUPERVISOR_CONF_PATH"
        log_success "write_cloudflared_dropin (no-op, no tunnel token)"
        return 0
    fi

    if [[ -z "$CLOUDFLARED_BIN" ]]; then
        log_error "CF_TUNNEL_TOKEN is set but CLOUDFLARED_BIN is empty (install_cloudflared did not run?)"
        exit 1
    fi

    mkdir -p "$(dirname "$SUPERVISOR_CONF_PATH")"
    mkdir -p "$SUPERVISOR_LOG_DIR"

    cat > "$SUPERVISOR_CONF_PATH" << EOF
[program:cloudflared]
command=${CLOUDFLARED_BIN} tunnel --no-autoupdate run
autostart=true
autorestart=true
startsecs=5
startretries=5
stopwaitsecs=15
priority=200
stdout_logfile=${SUPERVISOR_LOG_DIR}/cloudflared.stdout.log
stderr_logfile=${SUPERVISOR_LOG_DIR}/cloudflared.stderr.log
stdout_logfile_maxbytes=10MB
stdout_logfile_backups=3
stderr_logfile_maxbytes=10MB
stderr_logfile_backups=3
environment=TUNNEL_TOKEN="$(escape_for_supervisord_env "${CF_TUNNEL_TOKEN}")"
EOF

    chmod 600 "$SUPERVISOR_CONF_PATH"
    log_info "supervisor drop-in written to $SUPERVISOR_CONF_PATH (mode 600)"
    log_success "write_cloudflared_dropin"
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

    # Validate required env
    if [[ -z "$GITHUB_TOKEN" ]]; then
        log_error "ACS_GITHUB_TOKEN is required to clone gearbox/aisha + gearbox/ai-bundles"
        exit 2
    fi
    if [[ -z "$BUNDLE" ]]; then
        log_error "ACS_BUNDLE is required"
        exit 2
    fi
    if [[ -z "$CF_TUNNEL_TOKEN" ]]; then
        log_warn "ACS_CF_TUNNEL_TOKEN not set; node will be unreachable by apex"
    fi

    log_info "session_id=${APEX_SESSION_ID} bundle=${BUNDLE} cf_tunnel_token_set=$([[ -n "$CF_TUNNEL_TOKEN" ]] && echo true || echo false)"

    # System dependencies (idempotent on the image)
    install_cloudflared
    install_uv

    # Repos in parallel
    log_step "Syncing repositories..."
    clone_or_update_repo "aisha" "$AISHA_REPO" "$AISHA_PATH" "$AISHA_BRANCH" &
    local pid_aisha=$!
    clone_or_update_repo "ai-bundles" "$BUNDLES_REPO" "$BUNDLES_PATH" "$BUNDLES_BRANCH" &
    local pid_bundles=$!
    wait "$pid_aisha"
    wait "$pid_bundles"

    # Aisha CLI + bundle deploy
    install_aisha
    run_deployment

    # Cloudflared drop-in — picked up by image's supervisord after we exit
    write_cloudflared_dropin

    # Final structured ready line (grepped by apex and humans)
    local elapsed
    elapsed=$(($(date +%s) - start_time))
    local cf_status="off"
    [[ -n "$CF_TUNNEL_TOKEN" ]] && cf_status="on"
    echo "acs.provision.ready session_id=${APEX_SESSION_ID} elapsed=${elapsed}s bundle=${BUNDLE} cloudflared=${cf_status}"
}

# Run main unless the script is being sourced (e.g., by the test harness).
# In production (direct exec or curl|bash), __SOURCED__ is never set so main runs.
[[ "${__SOURCED__:-}" == "1" ]] || main "$@"
