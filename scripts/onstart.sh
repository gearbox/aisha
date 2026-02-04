#!/bin/bash
# ==============================================================================
# AISHA - Fast Automated Deployment Script
# ==============================================================================
# This script provides fast, automated deployment of aisha and ai-bundles
# to Vast.ai GPU nodes. Designed for use as an onstart script or manual execution.
#
# Features:
# - Clones both aisha (public) and ai-bundles (private) repositories
# - Supports GitHub PAT and SSH key authentication
# - Caches repositories for faster subsequent deployments
# - Parallel cloning for speed
# - Automatic bundle deployment
#
# Usage:
#   # As onstart script (set these as environment variables in Vast.ai):
#   export ACS_BUNDLE=wan_2.2_i2v
#   export ACS_GITHUB_TOKEN=ghp_xxxxx
#   ./onstart.sh
#
#   # Manual execution:
#   ACS_BUNDLE=wan_2.2_i2v ACS_GITHUB_TOKEN=ghp_xxxxx ./onstart.sh
#
# ==============================================================================

set -euo pipefail

# ==============================================================================
# Configuration (override via environment)
# ==============================================================================
AISHA_REPO="${ACS_AISHA_REPO:-https://github.com/gearbox/aisha.git}"
BUNDLES_REPO="${ACS_BUNDLES_REPO:-https://github.com/gearbox/ai-bundles.git}"
AISHA_BRANCH="${ACS_AISHA_BRANCH:-main}"
BUNDLES_BRANCH="${ACS_BUNDLES_BRANCH:-main}"

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

# Timeouts
COMFYUI_WAIT_TIMEOUT="${ACS_COMFYUI_WAIT_TIMEOUT:-300}"  # 5 minutes

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
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; }
log_step()    { echo -e "${BLUE}[STEP]${NC} $1"; }

# ==============================================================================
# Helper Functions
# ==============================================================================

setup_ssh_key() {
    # Skip if no SSH key configured
    if [[ -z "$SSH_KEY_PATH" && -z "$SSH_KEY_CONTENT" ]]; then
        return 0
    fi

    log_info "Setting up SSH authentication..."
    
    mkdir -p ~/.ssh
    chmod 700 ~/.ssh
    
    # If SSH key content is provided (base64), decode it
    if [[ -n "$SSH_KEY_CONTENT" ]]; then
        SSH_KEY_PATH=~/.ssh/deploy_key
        echo "$SSH_KEY_CONTENT" | base64 -d > "$SSH_KEY_PATH"
    fi
    
    chmod 600 "$SSH_KEY_PATH"
    
    # Configure SSH to use the key for GitHub
    cat >> ~/.ssh/config << EOF
Host github.com
    IdentityFile $SSH_KEY_PATH
    StrictHostKeyChecking accept-new
EOF
    
    log_success "SSH key configured"
}

get_authenticated_url() {
    local url="$1"
    
    # If using SSH key, convert to SSH URL
    if [[ -n "$SSH_KEY_PATH" || -n "$SSH_KEY_CONTENT" ]]; then
        # https://github.com/user/repo.git -> git@github.com:user/repo.git
        if [[ "$url" == https://github.com/* ]]; then
            url="${url/https:\/\/github.com\//git@github.com:}"
        fi
    # If using token, inject into HTTPS URL
    elif [[ -n "$GITHUB_TOKEN" && "$url" == https://github.com/* ]]; then
        url="${url/https:\/\/github.com/https://${GITHUB_TOKEN}@github.com}"
    fi
    
    echo "$url"
}

clone_or_update_repo() {
    local name="$1"
    local url="$2"
    local path="$3"
    local branch="${4:-main}"
    
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

wait_for_comfyui() {
    log_info "Waiting for ComfyUI..."
    
    local waited=0
    while [[ ! -d "$COMFYUI_PATH" ]] && (( waited < COMFYUI_WAIT_TIMEOUT )); do
        sleep 5
        waited=$((waited + 5))
        log_info "Waiting... ($waited/${COMFYUI_WAIT_TIMEOUT}s)"
    done
    
    if [[ ! -d "$COMFYUI_PATH" ]]; then
        log_error "ComfyUI not found at $COMFYUI_PATH after ${COMFYUI_WAIT_TIMEOUT}s"
        exit 1
    fi
    
    log_success "ComfyUI found"
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
    log_info "Installing aisha..."
    cd "$AISHA_PATH"
    uv pip install -e . --system --quiet
    log_success "aisha installed"
}

run_deployment() {
    if [[ -z "$BUNDLE" ]]; then
        log_warn "No bundle specified (ACS_BUNDLE), skipping deployment"
        log_info "Available bundles:"
        acs list --bundles-path "$BUNDLES_PATH/bundles" 2>/dev/null || ls -1 "$BUNDLES_PATH/bundles"
        return 0
    fi
    
    log_step "Deploying bundle: $BUNDLE"
    
    local cmd=(acs deploy --bundle "$BUNDLE" --bundles-path "$BUNDLES_PATH/bundles" --comfyui "$COMFYUI_PATH")
    
    [[ -n "$BUNDLE_VERSION" ]] && cmd+=(--version "$BUNDLE_VERSION")
    [[ "$MODELS_ONLY" == "true" ]] && cmd+=(--models-only)
    [[ "$NO_VERIFY" == "true" ]] && cmd+=(--no-verify)
    
    "${cmd[@]}"
    
    log_success "Deployment complete!"
}

# ==============================================================================
# Main
# ==============================================================================

main() {
    echo "=============================================="
    echo "  AISHA - Fast Automated Deployment"
    echo "=============================================="
    echo ""
    
    local start_time
    start_time=$(date +%s)
    
    # Setup
    setup_ssh_key
    wait_for_comfyui
    install_uv
    
    # Clone/update repositories (in parallel)
    log_step "Syncing repositories..."
    clone_or_update_repo "aisha" "$AISHA_REPO" "$AISHA_PATH" "$AISHA_BRANCH" &
    local pid_aisha=$!
    
    clone_or_update_repo "ai-bundles" "$BUNDLES_REPO" "$BUNDLES_PATH" "$BUNDLES_BRANCH" &
    local pid_bundles=$!
    
    wait $pid_aisha
    wait $pid_bundles
    
    # Install and deploy
    install_aisha
    run_deployment
    
    # Summary
    local elapsed
    elapsed=$(($(date +%s) - start_time))
    echo ""
    echo "=============================================="
    log_success "Completed in ${elapsed}s"
    echo "=============================================="
}

# Run main function
main "$@"
