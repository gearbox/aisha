#!/bin/bash
# ==============================================================================
# AISHA - Automated Deployment + Runtime Supervision
# ==============================================================================
# Clones aisha and ai-bundles, deploys the requested bundle, then hands off
# to supervisord which owns ComfyUI and cloudflared as long-lived processes.
#
# Usage (set as Vast.ai onstart, or run manually):
#   export ACS_BUNDLE=wan_2.2_i2v
#   export ACS_GITHUB_TOKEN=ghp_xxxxx
#   export ACS_CF_TUNNEL_TOKEN=eyJhIjoiX...
#   ./onstart.sh
#
# ==============================================================================

set -euo pipefail

trap 'echo "[FATAL] failed at line $LINENO with exit $?" >&2' ERR

# Escape a value for use inside supervisord's environment="KEY=value,..." stanza.
# Supervisord's parser is comma-separated with double-quoted values; backslash,
# double-quote, and comma must all be backslash-escaped.
escape_for_supervisord_env() {
    local value=${1-}
    value=${value//\\/\\\\}
    value=${value//\"/\\\"}
    value=${value//,/\\,}
    printf '%s' "$value"
}

# ==============================================================================
# Version pins
# ==============================================================================
# This script assumes cloudflared reads TUNNEL_TOKEN from env; verify when bumping.
CLOUDFLARED_VERSION="2024.12.2"

# ==============================================================================
# Configuration (override via environment)
# ==============================================================================
AISHA_REPO="${ACS_AISHA_REPO:-https://github.com/gearbox/aisha.git}"
BUNDLES_REPO="${ACS_BUNDLES_REPO:-https://github.com/gearbox/ai-bundles.git}"
AISHA_BRANCH="${ACS_AISHA_BRANCH:-master}"
BUNDLES_BRANCH="${ACS_BUNDLES_BRANCH:-master}"

WORKSPACE="${ACS_WORKSPACE:-/workspace}"
AISHA_PATH="${ACS_AISHA_PATH:-$WORKSPACE/aisha}"
BUNDLES_PATH="${ACS_BUNDLES_PATH:-$WORKSPACE/ai-bundles}"
COMFYUI_PATH="${ACS_COMFYUI_PATH:-$WORKSPACE/ComfyUI}"

# Authentication (use one of these)
GITHUB_TOKEN="${ACS_GITHUB_TOKEN:-}"
SSH_KEY_PATH="${ACS_SSH_KEY_PATH:-}"
SSH_KEY_CONTENT="${ACS_SSH_KEY_CONTENT:-}"  # Base64-encoded SSH key

# Deployment options
BUNDLE="${ACS_BUNDLE:-}"
BUNDLE_VERSION="${ACS_BUNDLE_VERSION:-}"
MODELS_ONLY="${ACS_MODELS_ONLY:-false}"
NO_VERIFY="${ACS_NO_VERIFY:-false}"

# ComfyUI runtime
COMFYUI_PORT="${ACS_COMFYUI_PORT:-8188}"
COMFYUI_HOST="${ACS_COMFYUI_HOST:-0.0.0.0}"
COMFYUI_EXTRA_ARGS="${ACS_COMFYUI_EXTRA_ARGS:-}"

# Cloudflare tunnel
CF_TUNNEL_TOKEN="${ACS_CF_TUNNEL_TOKEN:-}"
CLOUDFLARED_BIN=""  # Resolved in install_cloudflared; used in generate_supervisor_conf

# HuggingFace (forwarded into ComfyUI environment via supervisord)
HF_TOKEN="${ACS_HF_TOKEN:-}"

# Apex context
APEX_SESSION_ID="${ACS_APEX_SESSION_ID:-}"
# ACS_APEX_CALLBACK_URL and ACS_APEX_CALLBACK_TOKEN are phase-2; read but unused here

# Supervisor
SUPERVISOR_LOG_DIR="${ACS_SUPERVISOR_LOG_DIR:-/var/log/aisha}"
# ACS_SUPERVISOR_CONF_PATH and ACS_SUPERVISORCTL_BIN are test-override hooks
SUPERVISOR_CONF_PATH="${ACS_SUPERVISOR_CONF_PATH:-/etc/supervisor/conf.d/aisha.conf}"
SUPERVISORCTL_BIN="${ACS_SUPERVISORCTL_BIN:-supervisorctl}"

# Timeouts
COMFYUI_WAIT_TIMEOUT="${ACS_COMFYUI_WAIT_TIMEOUT:-300}"

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
# Helper Functions
# ==============================================================================

setup_ssh_key() {
    if [[ -z "$SSH_KEY_PATH" && -z "$SSH_KEY_CONTENT" ]]; then
        return 0
    fi

    log_step "starting setup_ssh_key"
    mkdir -p ~/.ssh
    chmod 700 ~/.ssh

    if [[ -n "$SSH_KEY_CONTENT" ]]; then
        SSH_KEY_PATH=~/.ssh/deploy_key
        echo "$SSH_KEY_CONTENT" | base64 -d > "$SSH_KEY_PATH"
    fi

    chmod 600 "$SSH_KEY_PATH"

    cat >> ~/.ssh/config << EOF
Host github.com
    IdentityFile $SSH_KEY_PATH
    StrictHostKeyChecking accept-new
EOF

    log_success "setup_ssh_key"
}

get_authenticated_url() {
    local url="$1"

    if [[ -n "$SSH_KEY_PATH" || -n "$SSH_KEY_CONTENT" ]]; then
        if [[ "$url" == https://github.com/* ]]; then
            url="${url/https:\/\/github.com\//git@github.com:}"
        fi
    elif [[ -n "$GITHUB_TOKEN" && "$url" == https://github.com/* ]]; then
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

    if command -v cloudflared &> /dev/null; then
        log_success "cloudflared already installed: $(cloudflared --version 2>&1)"
    else
        log_info "Installing cloudflared ${CLOUDFLARED_VERSION}..."

        if command -v dpkg &> /dev/null; then
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

install_supervisord() {
    log_step "starting install_supervisord"

    if command -v supervisord &> /dev/null; then
        log_success "supervisord already installed: $(supervisord --version 2>&1)"
    elif command -v apt-get &> /dev/null; then
        log_info "Installing supervisord via apt..."
        apt-get update -qq
        apt-get install -y supervisor
        log_success "supervisord installed: $(supervisord --version 2>&1)"
    else
        log_error "supervisord not installed and apt-get not available; install supervisord manually"
        exit 1
    fi

    mkdir -p "$SUPERVISOR_LOG_DIR"
    log_success "install_supervisord"
}

install_uv() {
    if command -v uv &> /dev/null; then
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
    cd "$AISHA_PATH"
    uv pip install -e . --system --quiet
    log_success "install_aisha"
}

# Waits for the ComfyUI *directory* to appear (base image extraction pre-flight).
# Renamed from wait_for_comfyui to distinguish from the HTTP probe below.
wait_for_comfyui_dir() {
    log_step "starting wait_for_comfyui_dir"

    local waited=0
    while [[ ! -d "$COMFYUI_PATH" ]] && (( waited < COMFYUI_WAIT_TIMEOUT )); do
        sleep 5
        waited=$((waited + 5))
        log_info "Waiting for ComfyUI dir... ($waited/${COMFYUI_WAIT_TIMEOUT}s)"
    done

    if [[ ! -d "$COMFYUI_PATH" ]]; then
        log_error "ComfyUI not found at $COMFYUI_PATH after ${COMFYUI_WAIT_TIMEOUT}s"
        exit 1
    fi

    log_success "wait_for_comfyui_dir"
}

run_deployment() {
    log_step "starting run_deployment: $BUNDLE"

    local cmd=(acs deploy --bundle "$BUNDLE" --bundles-path "$BUNDLES_PATH/bundles" --comfyui "$COMFYUI_PATH")

    [[ -n "$BUNDLE_VERSION" ]] && cmd+=(--version "$BUNDLE_VERSION")
    [[ "$MODELS_ONLY" == "true" ]] && cmd+=(--models-only)
    [[ "$NO_VERIFY" == "true" ]] && cmd+=(--no-verify)

    "${cmd[@]}"

    log_success "run_deployment"
}

generate_supervisor_conf() {
    log_step "starting generate_supervisor_conf"

    mkdir -p "$(dirname "$SUPERVISOR_CONF_PATH")"

    # Write resolved values directly — simpler and more debuggable than relying
    # on %(ENV_FOO)s substitution, which requires supervisord to inherit the env.
    cat > "$SUPERVISOR_CONF_PATH" << EOF
[program:comfyui]
command=/usr/bin/python3 ${COMFYUI_PATH}/main.py --listen ${COMFYUI_HOST} --port ${COMFYUI_PORT} ${COMFYUI_EXTRA_ARGS}
directory=${COMFYUI_PATH}
autostart=true
autorestart=true
startsecs=10
startretries=3
stopwaitsecs=30
stopasgroup=true
killasgroup=true
priority=100
stdout_logfile=${SUPERVISOR_LOG_DIR}/comfyui.stdout.log
stderr_logfile=${SUPERVISOR_LOG_DIR}/comfyui.stderr.log
stdout_logfile_maxbytes=10MB
stdout_logfile_backups=3
stderr_logfile_maxbytes=10MB
stderr_logfile_backups=3
environment=ACS_APEX_SESSION_ID="$(escape_for_supervisord_env "${APEX_SESSION_ID}")",HF_TOKEN="$(escape_for_supervisord_env "${HF_TOKEN}")"
EOF

    if [[ -n "$CF_TUNNEL_TOKEN" ]]; then
        if [[ -z "$CLOUDFLARED_BIN" ]]; then
            log_error "CF_TUNNEL_TOKEN is set but CLOUDFLARED_BIN is empty (install_cloudflared did not run?)"
            exit 1
        fi
        cat >> "$SUPERVISOR_CONF_PATH" << EOF

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
        log_success "supervisord config written: comfyui + cloudflared"
    else
        log_warn "cloudflared program omitted — ACS_CF_TUNNEL_TOKEN not set"
        log_success "supervisord config written: comfyui only"
    fi

    chmod 600 "$SUPERVISOR_CONF_PATH"
    log_info "supervisor conf permissions set to 600"
    log_success "generate_supervisor_conf"
}

stop_base_image_comfyui() {
    log_step "starting stop_base_image_comfyui"

    # Match ComfyUI specifically by its absolute main.py path, not any python+main.py.
    local pattern="${COMFYUI_PATH}/main.py"

    # Warn if ComfyUI is PID 1 — pkill cannot terminate PID 1; use a different base image.
    if [[ -r /proc/1/cmdline ]] && tr '\0' ' ' < /proc/1/cmdline | grep -q "${COMFYUI_PATH}/main.py"; then
        log_warn "ComfyUI is running as PID 1; cannot stop it from onstart. supervisord may fail to bind ${COMFYUI_PORT}"
    fi

    # Audit: log matching processes before killing.
    if pgrep -af "$pattern" > /tmp/comfyui_to_kill.log 2>&1; then
        log_info "Stopping pre-existing ComfyUI processes:"
        while IFS= read -r line; do
            log_info "  $line"
        done < /tmp/comfyui_to_kill.log
        pkill -f "$pattern" || true
        # Give it a moment to release the port before supervisord starts a new one.
        sleep 2
    else
        log_info "No pre-existing ComfyUI process found"
    fi
    rm -f /tmp/comfyui_to_kill.log

    log_success "stop_base_image_comfyui"
}

# HTTP probe: waits up to 60 s for ComfyUI to respond on /system_stats.
# Exit 1 on timeout so apex gets a clear failure signal at provisioning-probe time.
probe_comfyui_http() {
    log_step "starting probe_comfyui_http"

    local timeout=60
    local waited=0
    local url="http://127.0.0.1:${COMFYUI_PORT}/system_stats"

    while (( waited < timeout )); do
        if curl -sf "$url" > /dev/null 2>&1; then
            log_success "ComfyUI responding at $url"
            log_success "probe_comfyui_http"
            return 0
        fi
        sleep 5
        waited=$((waited + 5))
        log_info "Waiting for ComfyUI HTTP... ($waited/${timeout}s)"
    done

    log_error "ComfyUI did not respond within ${timeout}s — check ${SUPERVISOR_LOG_DIR}/comfyui.stderr.log"
    exit 1
}

# ==============================================================================
# Main
# ==============================================================================

main() {
    echo "=============================================="
    echo "  AISHA - Automated Deployment + Supervision"
    echo "=============================================="
    echo ""

    local start_time
    start_time=$(date +%s)

    # --- Validation ---
    if [[ -z "$GITHUB_TOKEN" && -z "$SSH_KEY_PATH" && -z "$SSH_KEY_CONTENT" ]]; then
        log_error "No GitHub auth configured; ai-bundles is private and clone will fail"
        exit 2
    fi

    if [[ -z "$BUNDLE" ]]; then
        log_error "ACS_BUNDLE not set"
        exit 2
    fi

    if [[ -z "$CF_TUNNEL_TOKEN" ]]; then
        log_warn "ACS_CF_TUNNEL_TOKEN not set; cloudflared will not be started — apex will be unable to reach this node"
    fi

    log_info "session_id=${APEX_SESSION_ID} bundle=${BUNDLE} cf_tunnel_token_set=$([ -n "$CF_TUNNEL_TOKEN" ] && echo true || echo false)"

    # --- System dependencies ---
    setup_ssh_key
    install_cloudflared
    install_supervisord
    install_uv

    # --- Wait for the base image to finish extracting ComfyUI ---
    wait_for_comfyui_dir

    # --- Clone/update repositories (in parallel) ---
    log_step "Syncing repositories..."
    clone_or_update_repo "aisha" "$AISHA_REPO" "$AISHA_PATH" "$AISHA_BRANCH" &
    local pid_aisha=$!

    clone_or_update_repo "ai-bundles" "$BUNDLES_REPO" "$BUNDLES_PATH" "$BUNDLES_BRANCH" &
    local pid_bundles=$!

    wait $pid_aisha
    wait $pid_bundles

    # --- Install CLI and deploy bundle ---
    install_aisha
    run_deployment

    # --- Hand off to supervisord ---
    generate_supervisor_conf
    stop_base_image_comfyui

    log_step "Reloading supervisord..."
    "$SUPERVISORCTL_BIN" reread
    "$SUPERVISORCTL_BIN" update
    "$SUPERVISORCTL_BIN" start comfyui
    if [[ -n "$CF_TUNNEL_TOKEN" ]]; then
        "$SUPERVISORCTL_BIN" start cloudflared
    fi

    # --- Confirm ComfyUI is up ---
    probe_comfyui_http

    # --- Final structured ready line (grepped by apex and humans) ---
    local elapsed
    elapsed=$(($(date +%s) - start_time))
    local cf_status="off"
    [[ -n "$CF_TUNNEL_TOKEN" ]] && cf_status="on"

    echo "acs.onstart.ready session_id=${APEX_SESSION_ID} elapsed=${elapsed}s comfyui_port=${COMFYUI_PORT} cloudflared=${cf_status}"
}

main "$@"
