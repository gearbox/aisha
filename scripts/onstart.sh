#!/bin/bash
# ==============================================================================
# AISHA - Automated Deployment + Runtime Supervision
# ==============================================================================
# Clones aisha and ai-bundles, deploys the requested bundle, then coordinates
# with the vastai/comfy image's supervisord to restart ComfyUI and start
# cloudflared.
#
# Designed for vastai/comfy base images where supervisord and ComfyUI are
# pre-installed. Aisha owns cloudflared only; the image owns ComfyUI.
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
# Image-baked ComfyUI source path
# ==============================================================================
# The vastai/comfy:v0.15.1-cuda-12.9-py312 image bakes ComfyUI at this path.
# The image's own /opt/supervisor-scripts/comfyui.sh wrapper computes
# COMFYUI_DIR=${WORKSPACE}/ComfyUI; we symlink so both paths resolve to the
# same place.
#
# If a future image variant uses a different path, override via env var.
COMFYUI_SRC="${ACS_COMFYUI_SRC:-/opt/workspace-internal/ComfyUI}"

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

# ComfyUI runtime — port must match the image's comfyui.sh wrapper (--port 18188)
COMFYUI_PORT="${ACS_COMFYUI_PORT:-18188}"
COMFYUI_HOST="${ACS_COMFYUI_HOST:-0.0.0.0}"
COMFYUI_EXTRA_ARGS="${ACS_COMFYUI_EXTRA_ARGS:-}"

# Cloudflare tunnel
CF_TUNNEL_TOKEN="${ACS_CF_TUNNEL_TOKEN:-}"
CLOUDFLARED_BIN=""  # Resolved in install_cloudflared; used in generate_supervisor_conf

# HuggingFace
HF_TOKEN="${ACS_HF_TOKEN:-}"

# Apex context
APEX_SESSION_ID="${ACS_APEX_SESSION_ID:-}"
# ACS_APEX_CALLBACK_URL and ACS_APEX_CALLBACK_TOKEN are phase-2; read but unused here

# Supervisor
SUPERVISOR_LOG_DIR="${ACS_SUPERVISOR_LOG_DIR:-/var/log/aisha}"
# ACS_SUPERVISOR_CONF_PATH, ACS_SUPERVISORCTL_BIN, and ACS_SUPERVISORD_CONFIG_PATH are test-override hooks
SUPERVISOR_CONF_PATH="${ACS_SUPERVISOR_CONF_PATH:-/etc/supervisor/conf.d/aisha.conf}"
SUPERVISORCTL_BIN="${ACS_SUPERVISORCTL_BIN:-supervisorctl}"
SUPERVISORD_CONFIG_PATH="${ACS_SUPERVISORD_CONFIG_PATH:-/etc/supervisor/supervisord.conf}"

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

# Symlinks the image-baked ComfyUI at $COMFYUI_SRC to $COMFYUI_PATH so that
# both the image's wrapper (cd ${WORKSPACE}/ComfyUI) and aisha's deployment
# logic (which expects $COMFYUI_PATH to be writable for bundle materialization)
# resolve to the same directory.
#
# Replaces previous polling — the image's "provisioning"
# step has already completed at /.launch time, so nothing materializes /workspace
# asynchronously. If $COMFYUI_SRC is missing at this point, the image is broken
# or unfamiliar; fail fast instead of polling for something that won't appear.
link_comfyui_workspace() {
    log_step "starting link_comfyui_workspace"

    if [[ ! -d "$COMFYUI_SRC" ]]; then
        log_error "image-baked ComfyUI not found at $COMFYUI_SRC"
        log_error "this is unexpected for vastai/comfy-derived images; check ACS_COMFYUI_SRC override"
        exit 1
    fi

    # Idempotent: if $COMFYUI_PATH already resolves to $COMFYUI_SRC, do nothing.
    if [[ -L "$COMFYUI_PATH" ]]; then
        local existing
        existing="$(readlink -f "$COMFYUI_PATH")"
        if [[ "$existing" == "$(readlink -f "$COMFYUI_SRC")" ]]; then
            log_info "$COMFYUI_PATH already symlinked to $COMFYUI_SRC"
            log_success "link_comfyui_workspace"
            return 0
        fi
        log_warn "$COMFYUI_PATH is a symlink to $existing (expected $COMFYUI_SRC); replacing"
        rm -f "$COMFYUI_PATH"
    elif [[ -d "$COMFYUI_PATH" ]]; then
        # Real directory at $COMFYUI_PATH would mean a future image variant
        # actually populates /workspace itself. We don't expect this on
        # vastai/comfy:v0.15.1, but if it ever happens, leave it alone.
        log_info "$COMFYUI_PATH is a real directory; leaving it untouched"
        log_success "link_comfyui_workspace"
        return 0
    elif [[ -e "$COMFYUI_PATH" ]]; then
        log_error "$COMFYUI_PATH exists but is neither a symlink nor a directory"
        exit 1
    fi

    # Make sure the parent dir exists. On vastai/comfy it does, but defensive.
    mkdir -p "$(dirname "$COMFYUI_PATH")"

    ln -s "$COMFYUI_SRC" "$COMFYUI_PATH"
    log_info "symlinked $COMFYUI_PATH -> $COMFYUI_SRC"
    log_success "link_comfyui_workspace"
}

# Starts the image's own supervisord using the image's own config file
# ($SUPERVISORD_CONFIG_PATH) so all of /etc/supervisor/conf.d/*.conf get loaded —
# comfyui, caddy, tunnel_manager, api-wrapper, etc. — plus our own aisha.conf
# for cloudflared.
#
# Why we have to do this: Vast.ai's /.launch (PID 1) replaces the image's
# default entrypoint chain. The image's own startup mechanism that would
# normally have started supervisord never runs. We are the only chance.
#
# Idempotent: if supervisord is already running (manual SSH retries, etc.),
# this is a no-op.
start_supervisord() {
    log_step "starting start_supervisord"

    if "$SUPERVISORCTL_BIN" status &>/dev/null; then
        log_info "supervisord already running"
        log_success "start_supervisord"
        return 0
    fi

    if [[ ! -f "$SUPERVISORD_CONFIG_PATH" ]]; then
        log_error "$SUPERVISORD_CONFIG_PATH missing — base image does not look like vastai/comfy"
        exit 1
    fi

    # Start as a daemon. The image's config sets nodaemon=false by default,
    # but pass -c explicitly so we don't depend on supervisord's search path.
    supervisord -c "$SUPERVISORD_CONFIG_PATH"

    # Wait briefly for the socket to appear. The daemon fork is fast (<1s
    # typical) but not instant; without this small wait, the very next
    # supervisorctl call can race.
    local waited=0
    local timeout="${ACS_SUPERVISORD_START_TIMEOUT:-30}"
    while (( waited < timeout )); do
        if "$SUPERVISORCTL_BIN" status &>/dev/null; then
            log_success "supervisord up (waited ${waited}s)"
            log_success "start_supervisord"
            return 0
        fi
        sleep 1
        waited=$((waited + 1))
    done

    log_error "supervisord did not become reachable within ${timeout}s after launch"
    exit 1
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

    if [[ -z "$CF_TUNNEL_TOKEN" ]]; then
        log_warn "ACS_CF_TUNNEL_TOKEN not set; cloudflared will not be configured — apex will be unable to reach this node"
        # Ensure no stale aisha.conf is left from a prior boot with a different config.
        rm -f "$SUPERVISOR_CONF_PATH"
        log_success "generate_supervisor_conf (no-op, no tunnel token)"
        return 0
    fi

    if [[ -z "$CLOUDFLARED_BIN" ]]; then
        log_error "CF_TUNNEL_TOKEN is set but CLOUDFLARED_BIN is empty (install_cloudflared did not run?)"
        exit 1
    fi

    mkdir -p "$(dirname "$SUPERVISOR_CONF_PATH")"
    mkdir -p "$SUPERVISOR_LOG_DIR"

    # Aisha owns ONLY cloudflared. ComfyUI is owned by the image's supervisord
    # via /opt/supervisor-scripts/comfyui.sh. Do not add a comfyui program block
    # here — it would compete with the image's program and cause a port-bind race.
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
    log_info "supervisor conf permissions set to 600"
    log_success "generate_supervisor_conf (cloudflared only)"
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

    log_info "session_id=${APEX_SESSION_ID} bundle=${BUNDLE} cf_tunnel_token_set=$([[ -n "$CF_TUNNEL_TOKEN" ]] && echo true || echo false)"

    # --- System dependencies ---
    setup_ssh_key
    install_cloudflared
    install_uv

    # --- Ensure the image-baked ComfyUI is reachable at $COMFYUI_PATH ---
    link_comfyui_workspace

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

    # --- Write our cloudflared drop-in (if a tunnel token was provided) ---
    generate_supervisor_conf

    # --- Start the image's supervisord ourselves — Vast.ai's /.launch doesn't ---
    start_supervisord

    # --- Have supervisord pick up our cloudflared drop-in ---
    log_step "Reloading supervisord config..."
    "$SUPERVISORCTL_BIN" reread
    "$SUPERVISORCTL_BIN" update

    # ComfyUI starts automatically from the image's comfyui.conf when supervisord
    # boots (autostart=true is supervisord's default). But we MUST restart it
    # after acs deploy so it picks up new custom_nodes/ — only imported at startup.
    log_step "Restarting image's comfyui program to pick up new custom_nodes..."
    "$SUPERVISORCTL_BIN" restart comfyui

    if [[ -n "$CF_TUNNEL_TOKEN" ]]; then
        log_step "Starting cloudflared..."
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

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
