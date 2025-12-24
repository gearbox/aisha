# AISHA - AI Content Service

Automated model deployment and content generation service for cloud + ComfyUI.

## Features

- 🚀 **Automated Model Deployment** - Download and install AI models with checksum verification
- 🔧 **Custom Node Management** - Automatically install required ComfyUI custom nodes
- 📁 **Workflow Management** - Deploy and manage ComfyUI workflow files
- 📊 **Progress Tracking** - Rich CLI with download progress and status reports
- ⚡ **Async Downloads** - Concurrent downloads with configurable limits
- 🔄 **Resumable** - Skip already downloaded files, resume interrupted deployments

## Quick Start

### Option 1: Standalone Script (No Installation Required)

For the fastest deployment on a cloud node:

```bash
# SSH into your cloud instance, then:
curl -fsSL https://raw.githubusercontent.com/gearbox/aisha/master/scripts/quick_deploy.py | python3 -
```

Or download and run:

```bash
wget https://raw.githubusercontent.com/gearbox/aisha/master/scripts/quick_deploy.py
python3 quick_deploy.py --comfyui /workspace/ComfyUI
```

### Option 2: Full Installation

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/gearbox/aisha.git
cd ai-content-service

# Install the package
uv pip install -e .

# Deploy WAN 2.2 models
acs deploy-wan --comfyui /workspace/ComfyUI
```

### Option 3: Cloud Onstart Script

Use this as your onstart script when renting a cloud instance:

```bash
curl -fsSL https://raw.githubusercontent.com/gearbox/aisha/master/scripts/deploy.sh | bash
```

## CLI Commands

### Deploy WAN 2.2 Models

```bash
# Quick deployment with defaults
acs deploy-wan

# Specify ComfyUI path
acs deploy-wan --comfyui /path/to/ComfyUI

# Force re-download existing files
acs deploy-wan --force
```

### Deploy from Configuration

```bash
# Deploy from YAML config
acs deploy --config config/models.yaml

# Include custom workflows
acs deploy --config config/models.yaml --workflows ./my-workflows/
```

### Check Status

```bash
# Show installed models, nodes, and workflows
acs status --comfyui /workspace/ComfyUI
```

### Install Single Custom Node

```bash
# Install a custom node
acs install-node https://github.com/city96/ComfyUI-GGUF

# Install with specific commit
acs install-node https://github.com/city96/ComfyUI-GGUF --commit abc123
```

### Install Workflow

```bash
# Install a workflow file
acs install-workflow my_workflow.json
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ACS_COMFYUI_PATH` | `/workspace/ComfyUI` | ComfyUI installation path |
| `ACS_CONFIG_PATH` | `/workspace/config` | Configuration files path |
| `ACS_WORKFLOWS_PATH` | `/workspace/workflows` | Workflows directory |
| `ACS_HF_TOKEN` | - | Hugging Face API token |
| `ACS_MAX_CONCURRENT_DOWNLOADS` | `3` | Max parallel downloads |
| `ACS_SKIP_EXISTING` | `true` | Skip already downloaded files |
| `ACS_VERIFY_CHECKSUMS` | `true` | Verify SHA256 after download |

### Configuration File (models.yaml)

```yaml
# Custom nodes to install
custom_nodes:
  - name: ComfyUI-GGUF
    git_url: https://github.com/city96/ComfyUI-GGUF
    commit_sha: abc123  # Optional: pin to specific commit

# Models to download
models:
  - name: my-model
    description: Model description
    model_type: diffusion_models  # diffusion_models, unet, clip, vae, loras, etc.
    files:
      - name: Model File
        url: https://example.com/model.gguf
        filename: model.gguf
        sha256: abc123...  # Optional: for verification
        size_bytes: 12345  # Optional: for progress tracking

# Workflows to install
workflows:
  - name: my-workflow
    filename: workflow.json
    description: Workflow description
```

## Project Structure

```
ai-content-service/
├── pyproject.toml          # Package configuration
├── README.md
├── config/
│   └── models.yaml         # Default deployment config
├── scripts/
│   ├── deploy.sh           # Cloud deployment script
│   └── quick_deploy.py     # Standalone deployment script
├── workflows/              # Custom workflow JSON files
└── src/
    └── ai_content_service/
        ├── __init__.py
        ├── cli.py           # Typer CLI
        ├── config.py        # Pydantic settings
        ├── deployer.py      # Deployment orchestration
        ├── downloader.py    # Async model downloader
        ├── comfyui.py       # ComfyUI setup
        └── workflows.py     # Workflow management
```

## Adding Custom Workflows

1. Place your workflow JSON files in the `workflows/` directory
2. Deploy with the workflows flag:

```bash
acs deploy --config config/models.yaml --workflows ./workflows/
```

Or install a single workflow:

```bash
acs install-workflow path/to/my_workflow.json
```

## WAN 2.2 Model Usage in ComfyUI

After deployment, use the WAN 2.2 models in ComfyUI:

1. Add a **UnetLoaderGGUF** node (from ComfyUI-GGUF)
2. Select one of:
   - `dasiwaWAN22I2V14B_midnightflirtHigh-Q8_0.gguf` (High Noise)
   - `dasiwaWAN22I2V14B_midnightflirtLow-Q8_0.gguf` (Low Noise)
3. Connect to your video generation workflow

**High Noise** vs **Low Noise**:
- **High Noise**: Better for stylized outputs, more creative freedom
- **Low Noise**: Better for realistic outputs, closer to reference

## Development

```bash
# Clone and install with dev dependencies
git clone https://github.com/your-org/ai-content-service.git
cd ai-content-service
uv pip install -e ".[dev]"

# Run tests
pytest

# Lint and format
ruff check src/
ruff format src/

# Type checking
mypy src/
```

## Architecture

This service follows SOLID principles with a clean separation of concerns:

- **Config** (`config.py`): Pydantic models for type-safe configuration
- **Downloader** (`downloader.py`): Async file downloads with retry logic
- **ComfyUI** (`comfyui.py`): ComfyUI environment management
- **Workflows** (`workflows.py`): Workflow file management
- **Deployer** (`deployer.py`): Orchestrates the complete deployment
- **CLI** (`cli.py`): User interface via Typer

## License

MIT
