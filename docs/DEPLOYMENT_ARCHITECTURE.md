# AISHA Deployment Architecture

> **Phase migration in progress.** Two provisioning paths coexist:
> 1. *Legacy*: apex submits `onstart` command (`curl | bash` of
>    `scripts/onstart.sh`). Documented below. Will be removed in Phase 3.
> 2. *New*: Vast.ai template with `PROVISIONING_SCRIPT` env var pointing at
>    `scripts/aisha-provision-comfyui.sh`. See [docs/PROVISIONING.md](./PROVISIONING.md).
>    Becomes the default in Phase 2 (when apex switches over).

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

When a Vast.ai instance starts, `onstart.sh` runs and then **exits** — it does
not block.  Long-lived processes (ComfyUI, cloudflared) are owned by
`supervisord` after onstart completes.

```bash
# 1. Validate env (ACS_GITHUB_TOKEN / ACS_BUNDLE required; exit 2 if absent)
# 2. Setup SSH key (if configured)
setup_ssh_key

# 3. Install cloudflared (pinned version; idempotent if image provides it)
install_cloudflared

# 4. Install uv (idempotent if image provides it)
install_uv

# 5. Symlink /opt/workspace-internal/ComfyUI -> /workspace/ComfyUI
#    (replaces "wait for ComfyUI directory" — the image bakes it at
#    /opt/workspace-internal/ComfyUI; symlink makes apex's existing
#    /workspace/ComfyUI assumption work)
link_comfyui_workspace

# 6. Clone/update repositories (aisha + ai-bundles, in parallel)
clone_or_update_repo "aisha" "$AISHA_REPO" "$AISHA_PATH" &
clone_or_update_repo "ai-bundles" "$BUNDLES_REPO" "$BUNDLES_PATH" &
wait

# 7. Install aisha
install_aisha

# 8. Deploy specified bundle (acs deploy ...)
run_deployment

# 9. Write aisha.conf drop-in (cloudflared only, if CF_TUNNEL_TOKEN set)
generate_supervisor_conf

# 10. Start the image's supervisord ourselves (Vast.ai's /.launch doesn't)
#     This boots all of /etc/supervisor/conf.d/*.conf — comfyui, caddy,
#     tunnel_manager, api-wrapper, our cloudflared, etc.
start_supervisord

# 11. supervisorctl reread + update (defensive — handles SSH retry case)
supervisorctl reread && supervisorctl update

# 12. supervisorctl restart comfyui (pick up new custom_nodes/)
supervisorctl restart comfyui

# 13. supervisorctl start cloudflared (if token set)
[[ -n "$CF_TUNNEL_TOKEN" ]] && supervisorctl start cloudflared

# 14. HTTP-probe ComfyUI on /system_stats (60s timeout, port 18188)
probe_comfyui_http

# 15. Print structured ready line — apex and humans grep for this:
# acs.onstart.ready session_id=... elapsed=...s comfyui_port=... cloudflared=on|off
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

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ACS_BUNDLE` | yes | — | Bundle to deploy |
| `ACS_BUNDLE_VERSION` | no | `current` | Specific version |
| `ACS_BUNDLES_REPO` | no | `https://github.com/gearbox/ai-bundles.git` | Git URL for bundles |
| `ACS_BUNDLES_BRANCH` | no | `master` | Git branch for bundles |
| `ACS_AISHA_REPO` | no | `https://github.com/gearbox/aisha.git` | Git URL for aisha |
| `ACS_AISHA_BRANCH` | no | `master` | Git branch for aisha |
| `ACS_GITHUB_TOKEN` | yes* | — | GitHub PAT (*required unless SSH key set) |
| `ACS_SSH_KEY_PATH` | no | — | SSH key path (alternative auth) |
| `ACS_SSH_KEY_CONTENT` | no | — | Base64-encoded SSH key (alternative auth) |
| `ACS_COMFYUI_PATH` | no | `/workspace/ComfyUI` | ComfyUI directory |
| `ACS_COMFYUI_PORT` | no | `18188` | ComfyUI listen port (must match image's `comfyui.sh`) |
| `ACS_COMFYUI_HOST` | no | `0.0.0.0` | ComfyUI listen interface |
| `ACS_COMFYUI_EXTRA_ARGS` | no | — | Extra args for `python main.py` |
| `ACS_CF_TUNNEL_TOKEN` | yes* | — | Cloudflare tunnel token (*required for apex to reach the node) |
| `ACS_APEX_SESSION_ID` | no | — | Log enrichment; not used in control flow |
| `ACS_APEX_CALLBACK_URL` | no | — | Phase-2; read but unused |
| `ACS_APEX_CALLBACK_TOKEN` | no | — | Phase-2; read but unused |
| `ACS_WORKSPACE` | no | `/workspace` | Parent dir for all clones |
| `ACS_MODELS_ONLY` | no | `false` | Skip ComfyUI setup |
| `ACS_NO_VERIFY` | no | `false` | Skip verification |
| `ACS_HF_TOKEN` | no | — | Hugging Face token |
| `ACS_CIVITAI_API_TOKEN` | no | — | Civitai token |
| `ACS_MAX_CONCURRENT_DOWNLOADS` | no | `3` | Parallel downloads |
| `ACS_SUPERVISOR_LOG_DIR` | no | `/var/log/aisha` | supervisord + child log dir |
| `ACS_COMFYUI_SRC` | no | `/opt/workspace-internal/ComfyUI` | Image-baked ComfyUI path to symlink from |
| `ACS_SUPERVISORD_START_TIMEOUT` | no | `30` | Seconds to wait for supervisord to become reachable after launch |

### Vast.ai Template Configuration

```json
{
  "env": {
    "ACS_BUNDLE": "wan_2.2_i2v",
    "ACS_GITHUB_TOKEN": "{{secrets.GITHUB_TOKEN}}",
    "ACS_HF_TOKEN": "{{secrets.HF_TOKEN}}"
  },
  "onstart": "curl -sL https://raw.githubusercontent.com/gearbox/aisha/master/scripts/onstart.sh | bash"
}
```

## Runtime Supervision

After `onstart.sh` completes, ComfyUI and cloudflared run as supervised
processes under `supervisord`.  `onstart.sh` itself exits — it no longer blocks.

### Process ownership

```
onstart.sh ──► symlink /opt/workspace-internal/ComfyUI -> /workspace/ComfyUI
           ──► write /etc/supervisor/conf.d/aisha.conf (cloudflared only)
           ──► start supervisord (aisha owns the process lifecycle on Vast.ai;
           │   Vast.ai's /.launch doesn't start it — we are the only chance)
           ──► supervisorctl restart comfyui ──► supervisorctl start cloudflared ──► exit

supervisord (long-lived; started by aisha on Vast.ai, config owned by vastai/comfy image)
  ├── [program:comfyui]      owned by image (/opt/supervisor-scripts/comfyui.sh)
  └── [program:cloudflared]  owned by aisha's drop-in conf
```

### Log locations

| Log | Path |
|-----|------|
| ComfyUI stdout | `/var/log/aisha/comfyui.stdout.log` |
| ComfyUI stderr | `/var/log/aisha/comfyui.stderr.log` |
| cloudflared stdout | `/var/log/aisha/cloudflared.stdout.log` |
| cloudflared stderr | `/var/log/aisha/cloudflared.stderr.log` |

`ACS_SUPERVISOR_LOG_DIR` overrides the base log directory (default `/var/log/aisha`).

### Common supervisorctl commands

`comfyui` is owned by the image's supervisord (`/opt/supervisor-scripts/comfyui.sh`).
`cloudflared` is owned by aisha's drop-in (`/etc/supervisor/conf.d/aisha.conf`).

```bash
# Restart ComfyUI (image-owned program)
supervisorctl restart comfyui

# Restart cloudflared (aisha-owned program)
supervisorctl restart cloudflared

# Stream ComfyUI logs
supervisorctl tail -f comfyui
tail -f /var/log/aisha/comfyui.stderr.log

# Check process status
supervisorctl status
```

### cloudflared is conditional

If `ACS_CF_TUNNEL_TOKEN` is empty, `onstart.sh` emits a `[WARN]`, removes any
stale `aisha.conf`, and skips writing the drop-in.  ComfyUI still starts (via
the image's supervisord).  This allows standalone ComfyUI debugging without a
tunnel.  Apex will be unable to reach the node in this state.

### Non-Debian base images

All current Vast.ai base images are `vastai/comfy`-derived; supervisord is
provided by the base image.  Aisha does not install supervisord.

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
