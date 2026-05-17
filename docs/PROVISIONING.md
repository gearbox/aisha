# AISHA Provisioning via Vast.ai `PROVISIONING_SCRIPT`

## Overview

Vast.ai's `PROVISIONING_SCRIPT` mechanism lets a template declare an HTTPS URL
to a bash script.  On instance first boot, after the image's own bootstrap
completes, the image's entrypoint chain downloads and executes the script
*before* starting supervisord.  The script runs inside the image's blessed
environment with `/.provisioning` held as a semaphore: every supervisord wrapper
in `/opt/supervisor-scripts/` gates on that file, so nothing races with our
script for ports, file locks, or GPU compute.

This replaces the old `curl | bash` onstart model where apex injected a command
into the instance start-up sequence.  The key differences are: (1) the new
script runs *inside* the `/.provisioning` window — ComfyUI has not started yet,
eliminating the need for HTTP probing or restart logic; (2) the image's own
provisioning chain has already reconciled `/workspace ↔ /opt/workspace-internal`
before our script runs, so we never touch ComfyUI's install or path; (3) when
our script exits 0 the image removes `/.provisioning` and supervisord starts
automatically, picking up any drop-ins we wrote under `/etc/supervisor/conf.d/`.
Our script's only jobs are: deploy the requested bundle into ComfyUI's
directories, and write the cloudflared supervisord drop-in.

## Template families

| Template variant | Runtime image | Provisioning script |
|---|---|---|
| `comfyui` | `vastai/comfy` | `scripts/aisha-provision-comfyui.sh` |
| `base` *(anticipated, not yet implemented)* | TBD | `scripts/aisha-provision-base.sh` |

The `-comfyui` suffix leaves room for a future `-base` sibling for non-ComfyUI
bundles (e.g., synthara coder models) without requiring a rename.

## Env vars consumed

All variables are read from the instance environment (set at instance-creation
time in the Vast.ai console, **not** baked into the template).

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `ACS_BUNDLE` | **yes** | — | Bundle name to deploy (e.g. `qwen_rapid_aio`) |
| `ACS_GITHUB_TOKEN` | **yes** | — | PAT to clone `gearbox/aisha` and `gearbox/ai-bundles` |
| `ACS_CF_TUNNEL_TOKEN` | recommended | — | Cloudflare tunnel token; node is unreachable by apex without it |
| `ACS_BUNDLE_VERSION` | no | latest | Pin a specific bundle version |
| `ACS_HF_TOKEN` | no | — | Hugging Face token for gated model downloads |
| `ACS_APEX_SESSION_ID` | no | `""` | Apex session UUID; echoed in the ready line for correlation |
| `ACS_AISHA_BRANCH` | no | `master` | Branch of `gearbox/aisha` to clone |
| `ACS_BUNDLES_BRANCH` | no | `master` | Branch of `gearbox/ai-bundles` to clone |
| `ACS_MODELS_ONLY` | no | `false` | `"true"` to skip non-model deploy steps |
| `ACS_NO_VERIFY` | no | `false` | `"true"` to skip checksum verification |
| `ACS_WORKSPACE` | no | `/workspace` | Parent directory for all clones and the aisha venv |
| `ACS_AISHA_PATH` | no | `$WORKSPACE/aisha` | Override clone path for aisha repo |
| `ACS_BUNDLES_PATH` | no | `$WORKSPACE/ai-bundles` | Parent dir for the cloned `ai-bundles` repo root; bundles reside at `$ACS_BUNDLES_PATH/bundles/`, which is the path `acs deploy` reads via `ACS_BUNDLES_PATH` in Aisha's `Settings` |
| `ACS_COMFYUI_PATH` | no | `$WORKSPACE/ComfyUI` | ComfyUI directory (must match image's path) |
| `ACS_COMFYUI_PYTHON` | no | `/venv/main/bin/python` | Python interpreter that owns ComfyUI's venv. `pip` operations for base requirements, locked overlay, and custom-node deps target this interpreter's site-packages. Override only if the base image relocates ComfyUI's venv. |
| `ACS_AISHA_VENV` | no | `$WORKSPACE/aisha-venv` | Path for the dedicated aisha Python venv |
| `ACS_AISHA_REPO` | no | `https://github.com/gearbox/aisha.git` | Override aisha repo URL |
| `ACS_BUNDLES_REPO` | no | `https://github.com/gearbox/ai-bundles.git` | Override bundles repo URL |
| `ACS_SUPERVISOR_CONF_PATH` | no | `/etc/supervisor/conf.d/aisha-cloudflared.conf` | Drop-in conf path |
| `ACS_SUPERVISOR_LOG_DIR` | no | `/var/log/aisha` | Log directory for cloudflared |

## Manual template creation steps

Create a **private** template in the Vast.ai console with these settings:

1. **Image path**: `vastai/comfy:v0.15.1-cuda-12.9-py312` (or latest blessed tag)
2. **Launch mode**: SSH (not entrypoint mode — supervisord must be the PID 1 chain)
3. **On-start script**: leave **empty** (the image handles launching supervisord)
4. **Environment variables** (template-level, set once):
   - `PROVISIONING_SCRIPT` = `https://raw.githubusercontent.com/gearbox/aisha/<tag>/scripts/aisha-provision-comfyui.sh`
     (replace `<tag>` with a release tag — see [Tag pinning](#tag-pinning))
5. **Ports**: `18188` (ComfyUI, hardcoded in the image's wrapper), plus any
   ports cloudflared requires
6. **Disk**: allocate enough for models (bundle-specific; WAN bundles need ~60 GB)
7. **Visibility**: **private**

### Base-image requirements

The provisioning script assumes the base image ships these tools and
**fails fast** if any are missing — it deliberately does not bootstrap
them at runtime (no `curl | sh` of upstream installers, which would be a
supply-chain risk in a provisioning context):

- `uv` — Python package manager. Present on `vastai/comfy` at
  `/opt/instance-tools/bin/uv`.
- `git` — for cloning `gearbox/aisha` and `gearbox/ai-bundles`.
- `curl` — for fetching cloudflared release artifacts.
- `dpkg` or write access to `/usr/local/bin/` — for installing cloudflared.

`cloudflared` itself may or may not be on the base image; the script
installs it conditionally and verifies architecture (`x86_64` only).

If a future base image stops shipping `uv`, install it deterministically
at image-build time (e.g., bake a pinned version into a derived image)
rather than re-introducing a runtime `curl | sh`.

The provisioning script tells Aisha which Python interpreter owns ComfyUI's venv (`ACS_COMFYUI_PYTHON`, default `/venv/main/bin/python` on `vastai/comfy`-based images). Aisha targets that interpreter for all ComfyUI-side pip operations, regardless of which venv `acs` itself runs from. If the base image relocates ComfyUI's venv in a future release, update this env var rather than relying on PATH activation order.

Per-instance env vars (`ACS_BUNDLE`, `ACS_GITHUB_TOKEN`, `ACS_CF_TUNNEL_TOKEN`,
etc.) are set at **instance creation time**, not in the template, so the same
template can serve multiple sessions with different bundles.

## Tag pinning

`PROVISIONING_SCRIPT` should always point at a **tagged URL**, not `master`:

```
https://raw.githubusercontent.com/gearbox/aisha/v0.X.Y/scripts/aisha-provision-comfyui.sh
```

Using `master` means a code push can silently change the script that runs on
the next instance boot, making deployed instances non-reproducible.

**Release process:**

1. Merge changes to `master`.
2. Cut a release tag: `git tag v0.X.Y && git push origin v0.X.Y`.
3. Verify the tagged URL resolves:
   ```bash
   curl -fsSI https://raw.githubusercontent.com/gearbox/aisha/v0.X.Y/scripts/aisha-provision-comfyui.sh
   ```
4. Update the template's `PROVISIONING_SCRIPT` value in the Vast.ai console to
   point at the new tag.

## Coexistence with `scripts/onstart.sh`

`scripts/onstart.sh` is the **legacy provisioning path** still in production.
Apex submits it via `curl | bash` as part of the instance `onstart` command.

The two paths are mutually exclusive for a given instance:

- Instances created from the **old** apex flow use `onstart.sh`.
- Instances created from the **new** template use `aisha-provision-comfyui.sh`
  via `PROVISIONING_SCRIPT`.

**Do not set both** `onstart` and `PROVISIONING_SCRIPT` on the same instance —
they would race and potentially corrupt each other's state.

`scripts/onstart.sh` will remain in the repository until Phase 3 removes it
(after apex switches over in Phase 2 and the old path is confirmed dead).
