#!/bin/bash
# AI Content Service - Deployment Script
# This script can be used as an onstart script or run manually after SSH

set -euo pipefail

# Configuration
REPO_URL="${ACS_REPO_URL:-https://github.com/gearbox/aisha.git}"
INSTALL_DIR="${ACS_INSTALL_DIR:-/workspace/ai-content-service}"
COMFYUI_PATH="${ACS_COMFYUI_PATH:-/workspace/ComfyUI}"
CONFIG_FILE="${ACS_CONFIG_FILE:-config/models.yaml}"
WORKFLOWS_DIR="${ACS_WORKFLOWS_DIR:-workflows}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${CYAN}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Wait for ComfyUI to be ready
wait_for_comfyui() {
    log_info "Waiting for ComfyUI to be available..."
    local max_wait=120
    local waited=0
    
    while [ ! -d "$COMFYUI_PATH" ] && [ $waited -lt $max_wait ]; do
        sleep 5
        waited=$((waited + 5))
        log_info "Waiting... ($waited/$max_wait seconds)"
    done
    
    if [ ! -d "$COMFYUI_PATH" ]; then
        log_error "ComfyUI not found at $COMFYUI_PATH after $max_wait seconds"
        exit 1
    fi
    
    log_success "ComfyUI found at $COMFYUI_PATH"
}

# Install uv if not present
install_uv() {
    if ! command -v uv &> /dev/null; then
        log_info "Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.cargo/bin:$PATH"
    fi
    log_success "uv is available"
}

# Clone or update the repository
setup_repository() {
    if [ -d "$INSTALL_DIR" ]; then
        log_info "Updating existing installation..."
        cd "$INSTALL_DIR"
        git pull --ff-only || true
    else
        log_info "Cloning repository..."
        git clone "$REPO_URL" "$INSTALL_DIR"
        cd "$INSTALL_DIR"
    fi
    log_success "Repository ready at $INSTALL_DIR"
}

# Install the package
install_package() {
    log_info "Installing ai-content-service..."
    cd "$INSTALL_DIR"
    uv pip install -e . --system
    log_success "Package installed"
}

# Run deployment
run_deployment() {
    log_info "Running deployment..."
    cd "$INSTALL_DIR"
    
    # Check if config file exists
    if [ ! -f "$CONFIG_FILE" ]; then
        log_info "No config file found, using built-in WAN 2.2 deployment"
        acs deploy-wan --comfyui "$COMFYUI_PATH"
    else
        log_info "Deploying from config: $CONFIG_FILE"
        if [ -d "$WORKFLOWS_DIR" ]; then
            acs deploy --config "$CONFIG_FILE" --workflows "$WORKFLOWS_DIR" --comfyui "$COMFYUI_PATH"
        else
            acs deploy --config "$CONFIG_FILE" --comfyui "$COMFYUI_PATH"
        fi
    fi
    
    log_success "Deployment complete!"
}

# Show status
show_status() {
    log_info "Current deployment status:"
    acs status --comfyui "$COMFYUI_PATH"
}

# Main execution
main() {
    echo "========================================"
    echo "AI Content Service - Deployment"
    echo "========================================"
    
    wait_for_comfyui
    install_uv
    setup_repository
    install_package
    run_deployment
    show_status
    
    echo ""
    log_success "All done! ComfyUI is ready with WAN 2.2 models."
    echo ""
    echo "To access ComfyUI, use the Vast.ai proxy URL for port 8188"
}

# Run main function
main "$@"
