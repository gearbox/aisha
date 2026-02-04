# AISHA Deployment Architecture

## Overview

This document describes the improved deployment architecture that separates the deployment tool (`aisha`) from bundle configurations (`ai-bundles`).

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              GitHub                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────┐      ┌──────────────────────────┐             │
│  │  gearbox/aisha (public)  │      │ gearbox/ai-bundles       │             │
│  │                          │      │ (private)                │             │
│  │  - Deployment CLI        │      │                          │             │
│  │  - Bundle registry       │      │  - Bundle configs        │             │
│  │  - Model downloader      │      │  - Workflow JSONs        │             │
│  │  - ComfyUI manager       │      │  - Requirements locks    │             │
│  │  - Onstart scripts       │      │  - Model definitions     │             │
│  └──────────────────────────┘      └──────────────────────────┘             │
│              │                                  │                            │
└──────────────┼──────────────────────────────────┼────────────────────────────┘
               │                                  │
               │  git clone (public)              │  git clone (PAT/SSH)
               │                                  │
               ▼                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Vast.ai GPU Node                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  /workspace/                                                                 │
│  ├── aisha/                    # Deployment tool                            │
│  │   ├── src/ai_content_service/                                            │
│  │   └── scripts/onstart.sh                                                 │
│  │                                                                           │
│  ├── ai-bundles/               # Bundle configurations                      │
│  │   ├── bundle-index.yaml                                                  │
│  │   └── bundles/                                                           │
│  │       ├── wan_2.2_i2v/                                                   │
│  │       └── ...                                                            │
│  │                                                                           │
│  └── ComfyUI/                  # ComfyUI installation                       │
│      ├── models/                                                            │
│      ├── custom_nodes/                                                      │
│      └── user/workflows/                                                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Deployment Flow

### 1. Onstart Script Execution

When a Vast.ai instance starts, the onstart script:

```bash
# 1. Setup SSH key (if configured)
setup_ssh_key

# 2. Wait for ComfyUI base image
wait_for_comfyui

# 3. Install uv package manager
install_uv

# 4. Clone/update repositories (parallel)
clone_or_update_repo "aisha" "$AISHA_REPO" "$AISHA_PATH" &
clone_or_update_repo "ai-bundles" "$BUNDLES_REPO" "$BUNDLES_PATH" &
wait

# 5. Install aisha
install_aisha

# 6. Deploy specified bundle
run_deployment
```

### 2. Bundle Resolution

The registry system resolves bundle references:

```
"wan_2.2_i2v"           → Default registry, latest version
"wan_2.2_i2v:260103-01" → Specific version
"remote/wan_2.2_i2v"    → Explicit registry
```

### 3. Deployment Execution

```
┌─────────────┐
│ Load Bundle │
└──────┬──────┘
       ▼
┌─────────────────────┐
│ Create Deploy Plan  │──────────────┐
└──────┬──────────────┘              │
       ▼                              │
┌─────────────────────┐              │
│ FULL mode?          │              │
└──────┬──────────────┘              │
       │                              │
       ├── Yes ──▶ ┌─────────────────────────────┐
       │           │ 1. Update ComfyUI           │
       │           │ 2. Install base requirements│
       │           │ 3. Install locked deps      │
       │           │ 4. Install custom nodes     │
       │           └─────────────┬───────────────┘
       │                         │
       ├── No (MODELS_ONLY) ─────┤
       │                         │
       ▼                         ▼
┌─────────────────────────────────────┐
│ 5. Download models (async, parallel)│
│ 6. Install workflow                 │
│ 7. Verify deployment                │
└─────────────────────────────────────┘
```

## Authentication Options

### Option 1: GitHub Personal Access Token (Recommended for simplicity)

```bash
# In Vast.ai environment variables:
ACS_GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx
```

Pros:
- Simple setup
- Works with HTTPS URLs
- Easy to rotate

Cons:
- Token visible in environment
- Must be kept secret

### Option 2: SSH Deploy Key (Recommended for security)

```bash
# Generate key pair
ssh-keygen -t ed25519 -C "aisha-deploy" -f deploy_key

# Add public key to repo Settings → Deploy keys
# Use private key in deployment:
ACS_SSH_KEY_PATH=/path/to/deploy_key
```

Pros:
- More secure (key never in URL)
- Can be read-only
- Standard practice for CI/CD

Cons:
- More setup required
- Key file management

### Option 3: Base64-encoded SSH Key (For containers)

```bash
# Encode key
base64 -w0 < deploy_key > deploy_key.b64

# Use in environment
ACS_SSH_KEY_CONTENT=$(cat deploy_key.b64)
```

Pros:
- No file needed on disk
- Works well with container secrets

## Configuration Reference

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ACS_BUNDLE` | Bundle to deploy | (required) |
| `ACS_BUNDLE_VERSION` | Specific version | `current` |
| `ACS_BUNDLES_REPO` | Git URL for bundles | - |
| `ACS_BUNDLES_BRANCH` | Git branch | `main` |
| `ACS_GITHUB_TOKEN` | GitHub PAT | - |
| `ACS_SSH_KEY_PATH` | SSH key path | - |
| `ACS_COMFYUI_PATH` | ComfyUI directory | `/workspace/ComfyUI` |
| `ACS_MODELS_ONLY` | Skip ComfyUI setup | `false` |
| `ACS_NO_VERIFY` | Skip verification | `false` |
| `ACS_HF_TOKEN` | Hugging Face token | - |
| `ACS_CIVITAI_API_TOKEN` | Civitai token | - |
| `ACS_MAX_CONCURRENT_DOWNLOADS` | Parallel downloads | `3` |

### Vast.ai Template Configuration

```json
{
  "env": {
    "ACS_BUNDLE": "wan_2.2_i2v",
    "ACS_GITHUB_TOKEN": "{{secrets.GITHUB_TOKEN}}",
    "ACS_HF_TOKEN": "{{secrets.HF_TOKEN}}"
  },
  "onstart": "curl -sL https://raw.githubusercontent.com/gearbox/aisha/main/scripts/onstart.sh | bash"
}
```

## Bundle Registry System

### Registry Types

1. **LocalBundleRegistry**: Reads from local filesystem
2. **GitBundleRegistry**: Clones/pulls from Git repository

### Registry Priority

```python
manager = BundleRegistryManager()
manager.register(local_registry)           # Fallback
manager.register(git_registry, default=True)  # Primary
```

### Bundle Resolution Order

1. Parse reference (e.g., `remote/wan_2.2_i2v:260103-01`)
2. Select registry (or default)
3. Lookup in `bundle-index.yaml`
4. Resolve version (specific → default → current symlink)
5. Return bundle path

## Migration Guide

### From Embedded Bundles to External Repository

1. **Create the ai-bundles repository**
   ```bash
   mkdir ai-bundles && cd ai-bundles
   git init
   ```

2. **Copy existing bundles**
   ```bash
   cp -r ../aisha/config/bundles/* bundles/
   ```

3. **Create bundle-index.yaml**
   ```yaml
   version: "1"
   bundles:
     - name: wan_2.2_i2v
       path: bundles/wan_2.2_i2v
       description: "WAN 2.2 I2V"
   ```

4. **Update aisha configuration**
   ```bash
   # .env
   ACS_BUNDLES_REPO=https://github.com/gearbox/ai-bundles
   ACS_GITHUB_TOKEN=ghp_xxx
   ```

5. **Test deployment**
   ```bash
   acs registry sync
   acs registry list
   acs deploy -b wan_2.2_i2v --dry-run
   ```

## Performance Considerations

### Parallel Operations

- Repository cloning happens in parallel
- Model downloads are concurrent (configurable limit)
- Custom node installation is sequential (dependency order)

### Caching

- Git repositories use shallow clones (`--depth 1`)
- Models with matching checksums are skipped
- Registry index is cached in memory

### Typical Deployment Times

| Operation | Time (estimate) |
|-----------|-----------------|
| Clone aisha | 5-10s |
| Clone ai-bundles | 3-5s |
| Install aisha | 10-15s |
| ComfyUI update | 30-60s |
| Custom nodes | 60-120s |
| Model downloads (WAN) | 5-15 min |
| **Total (full)** | **7-18 min** |
| **Total (models-only)** | **5-16 min** |

## Troubleshooting

### Authentication Failures

```bash
# Check token permissions
curl -H "Authorization: token $ACS_GITHUB_TOKEN" \
  https://api.github.com/repos/gearbox/ai-bundles

# Test SSH key
ssh -T git@github.com -i $ACS_SSH_KEY_PATH
```

### Bundle Not Found

```bash
# List available bundles
acs registry list

# Check bundle index
cat /workspace/ai-bundles/bundle-index.yaml
```

### Model Download Failures

```bash
# Check HF token
curl -H "Authorization: Bearer $ACS_HF_TOKEN" \
  https://huggingface.co/api/whoami

# Retry specific bundle
acs deploy -b wan_2.2_i2v --models-only
```
