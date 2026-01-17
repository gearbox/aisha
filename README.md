# AISHA - AI Content Service

Bundle-based deployment automation for ComfyUI.

## Features

- 📦 **Bundle System** - Reproducible deployments with version pinning
- 🔄 **Snapshot Capture** - Freeze working setups into reusable bundles
- 🚀 **Automated Deployment** - One command to deploy complete environments
- ⚡ **Models-Only Mode** - Lightweight deployment for existing setups
- ✅ **Verification** - Automatic validation via ComfyUI `/object_info`
- 📊 **Progress Tracking** - Rich CLI with download progress and status reports
- ⚡ **Async Downloads** - Concurrent model downloads with configurable limits

## Bundle System

Bundles provide reproducible ComfyUI deployments by capturing:

- ComfyUI commit SHA
- Custom nodes with pinned commits
- Python dependencies (`pip freeze`)
- Workflow JSON file
- Optional `extra_model_paths.yaml`

### Bundle Structure

```
config/bundles/
├── wan_2.2_i2v/
│   ├── current -> 260101-02/       # Symlink to active version
│   ├── 260101-01/
│   │   ├── bundle.yaml             # Main configuration
│   │   ├── requirements.lock       # Pip freeze output
│   │   ├── workflow.json           # ComfyUI workflow
│   │   └── extra_model_paths.yaml  # Optional
│   └── 260101-02/
│       └── ...
└── ltx_i2v/
    └── ...
```

## Quick Start

### 1. Install

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/gearbox/aisha.git
cd aisha
uv pip install -e .
```

### 2. Create a Bundle (from working setup)

After manually setting up ComfyUI with ComfyUI-Manager:

```bash
# Capture current state as a bundle
acs snapshot \
    --name wan_2.2_i2v \
    --workflow /path/to/your/workflow.json \
    --description "WAN 2.2 Image-to-Video setup"
```

This creates:
- `config/bundles/wan_2.2_i2v/260103-01/bundle.yaml`
- `config/bundles/wan_2.2_i2v/260103-01/requirements.lock`
- `config/bundles/wan_2.2_i2v/260103-01/workflow.json`

### 3. Add Models to Bundle

Edit `bundle.yaml` to add model definitions:

```yaml
models:
  - name: dasiwaWAN22I2V14B-GGUF-Q8
    model_type: diffusion_models
    files:
      - name: WAN 2.2 High Noise Q8
        url: https://huggingface.co/Bedovyy/dasiwaWAN22I2V14B-GGUF/resolve/main/HighNoise/dasiwaWAN22I2V14B_midnightflirtHigh-Q8_0.gguf
        filename: dasiwaWAN22I2V14B_midnightflirtHigh-Q8_0.gguf
        sha256: 0ab7f1fc4aa0f17de33877d1d87fef1c538b844c4a3a9decbcc88a741a3af7cd
```

### 4. Deploy Bundle

```bash
# Full deployment
acs deploy --bundle wan_2.2_i2v

# Models-only deployment (skip ComfyUI setup)
acs deploy --bundle wan_2.2_i2v --models-only
```

## Deployment Modes

AISHA supports two deployment modes to fit different use cases:

### Full Deployment (default)

Deploys the complete environment:

1. **ComfyUI checkout** - Updates to pinned commit
2. **Base requirements** - Installs ComfyUI's requirements.txt
3. **Locked requirements** - Applies pip freeze overlay
4. **Custom nodes** - Clones/updates nodes to pinned commits
5. **Models** - Downloads from HuggingFace/Civitai with checksum verification
6. **Workflow** - Copies to ComfyUI user workflows
7. **Verification** - Validates via `/object_info` endpoint

```bash
acs deploy --bundle wan_2.2_i2v
acs deploy --bundle wan_2.2_i2v --version 260101-01
```

### Models-Only Deployment (`--models-only` / `-m`)

Lightweight deployment that **only** downloads models and installs the workflow. Use this when:

- You already have a working ComfyUI installation
- You want to add a new workflow without modifying your existing setup
- You're testing a new model configuration
- You need faster deployments on pre-configured nodes

```bash
# Models-only deployment
acs deploy --bundle wan_2.2_i2v --models-only

# Short form
acs deploy -b wan_2.2_i2v -m

# Preview what will be deployed
acs deploy --bundle wan_2.2_i2v --models-only --dry-run
```

**What's skipped in models-only mode:**
- ComfyUI git checkout
- Base requirements installation
- Locked requirements installation  
- Custom nodes installation

**What's still executed:**
- Model downloads (with checksum verification)
- Workflow installation
- Verification (optional, use `--no-verify` to skip)

### Deployment Plan Preview

Use `--dry-run` to see what will be deployed without making changes:

```bash
$ acs deploy --bundle wan_2.2_i2v --models-only --dry-run

        Deployment Plan: wan_2.2_i2v (260101-01)
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃ Step                ┃ Action                        ┃ Status ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Mode                │ Models Only                   │        │
│ ComfyUI             │ Checkout to pinned commit     │ ○      │
│ Base Requirements   │ Install ComfyUI requirements  │ ○      │
│ Locked Requirements │ Install pip freeze overlay    │ ○      │
│ Custom Nodes        │ Install 2 nodes               │ ○      │
│ Models              │ Download 3 files              │ ●      │
│ Workflow            │ Install workflow.json         │ ●      │
│ Verify              │ Check ComfyUI /object_info    │ ●      │
└─────────────────────┴───────────────────────────────┴────────┘

Dry run - no changes made
```

## CLI Commands

### Deploy

```bash
# Deploy bundle (uses ACS_BUNDLE env or --bundle flag)
acs deploy
acs deploy --bundle wan_2.2_i2v
acs deploy --bundle wan_2.2_i2v --version 260101-01

# Models-only deployment
acs deploy --bundle wan_2.2_i2v --models-only
acs deploy -b wan_2.2_i2v -m

# Additional options
acs deploy --bundle wan_2.2_i2v --no-verify      # Skip verification
acs deploy --bundle wan_2.2_i2v --dry-run        # Preview only
acs deploy --bundle wan_2.2_i2v -m -n            # Models-only dry run
```

### Snapshot

```bash
# Capture snapshot from working ComfyUI setup
acs snapshot --name wan_2.2_i2v --workflow workflow.json
acs snapshot -n wan_2.2_i2v -w workflow.json -d "Initial setup"
acs snapshot -n wan_2.2_i2v -w workflow.json --extra-model-paths extra_model_paths.yaml
```

### Bundle Management

```bash
# List all bundles
acs bundle list

# List versions of a specific bundle
acs bundle list wan_2.2_i2v

# Show bundle details
acs bundle show wan_2.2_i2v
acs bundle show wan_2.2_i2v --version 260101-01

# Set current version
acs bundle set-current wan_2.2_i2v 260101-02

# Delete a version
acs bundle delete wan_2.2_i2v 260101-01
```

### Status

```bash
# Show deployment status
acs status --comfyui /workspace/ComfyUI
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ACS_COMFYUI_PATH` | `/workspace/ComfyUI` | ComfyUI installation path |
| `ACS_BUNDLES_PATH` | `config/bundles` | Bundles directory |
| `ACS_BUNDLE` | - | Bundle name to deploy |
| `ACS_BUNDLE_VERSION` | - | Specific version (default: current) |
| `ACS_HF_TOKEN` | - | Hugging Face API token |
| `ACS_CIVITAI_API_TOKEN` | - | Civitai API token for model downloads |
| `ACS_MAX_CONCURRENT_DOWNLOADS` | `3` | Max parallel downloads |
| `ACS_NO_VERIFY` | `false` | Skip ComfyUI verification |

## Model Downloads

The service supports downloading models from multiple sources:

### Hugging Face

Standard HuggingFace URLs work out of the box. For private/gated models, set `ACS_HF_TOKEN`:

```yaml
files:
  - name: Model File
    url: https://huggingface.co/org/model/resolve/main/model.safetensors
    filename: model.safetensors
```

### Civitai

Civitai downloads require an API token. Get yours from [Civitai Settings](https://civitai.com/user/account).

```bash
export ACS_CIVITAI_API_TOKEN=your_token_here
```

```yaml
files:
  - name: SDXL Model
    url: https://civitai.com/api/download/models/128713
    filename: sdxl_model.safetensors
    sha256: abc123...
```

## Bundle Configuration

### bundle.yaml

```yaml
metadata:
  name: wan_2.2_i2v
  version: "260101-01"
  description: WAN 2.2 Image-to-Video with GGUF Q8 models
  created_at: "2026-01-01T10:30:00Z"
  tested: true

comfyui:
  repo: https://github.com/comfyanonymous/ComfyUI
  commit: abc123def456789...

custom_nodes:
  - name: ComfyUI-GGUF
    git_url: https://github.com/city96/ComfyUI-GGUF
    commit_sha: def456789...
    
  - name: ComfyUI-VideoHelperSuite
    git_url: https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite
    commit_sha: 789abcdef...

models:
  - name: dasiwaWAN22I2V14B-GGUF-Q8
    model_type: diffusion_models
    files:
      - name: WAN 2.2 High Noise Q8
        url: https://huggingface.co/Bedovyy/dasiwaWAN22I2V14B-GGUF/resolve/main/HighNoise/dasiwaWAN22I2V14B_midnightflirtHigh-Q8_0.gguf
        filename: dasiwaWAN22I2V14B_midnightflirtHigh-Q8_0.gguf
        sha256: 0ab7f1fc4aa0f17de33877d1d87fef1c538b844c4a3a9decbcc88a741a3af7cd

# Files in bundle directory
requirements_lock_file: requirements.lock
workflow_file: workflow.json
extra_model_paths_file: extra_model_paths.yaml
```

## Project Structure

```
aisha/
├── pyproject.toml
├── README.md
├── config/
│   └── bundles/           # Bundle storage
│       └── wan_2.2_i2v/
│           ├── current -> 260101-01/
│           └── 260101-01/
│               ├── bundle.yaml
│               ├── requirements.lock
│               └── workflow.json
├── src/
│   └── ai_content_service/
│       ├── __init__.py
│       ├── cli.py          # Typer CLI
│       ├── config.py       # Settings, DeployMode, DeploymentPlan
│       ├── bundle.py       # Bundle management
│       ├── deployer.py     # Deployment orchestration
│       ├── comfyui.py      # ComfyUI setup & verification
│       ├── downloader.py   # Async model downloader
│       ├── workflows.py    # Workflow management
│       └── snapshot.py     # Snapshot capture
└── tests/
    └── test_deploy_mode.py
```

## Development

```bash
# Clone and install with dev dependencies
git clone https://github.com/gearbox/aisha.git
cd aisha
uv pip install -e ".[dev]"

# Run tests
pytest

# Lint and format
ruff check src/
ruff format src/

# Type checking
mypy src/
```

## License

MIT
