# Additive bundle deployment

Additive deployment installs a bundle beside bundles already resident on a running ComfyUI node. It is for shared nodes: it never checks out ComfyUI, installs base requirements, restarts ComfyUI, or changes an existing custom node's pin. Before it moves any bytes, Aisha compares the requested bundle with the node's residency manifest.

```bash
# Add a compatible bundle to this node.
acs deploy --bundle wan_2.2_i2v --additive

# Inspect the declarations currently recorded for the node.
acs residency show
```

`--additive` cannot be combined with `--models-only`. A successful additive deployment records its models, custom-node pins, workflow, readiness class, and a `pending_restart` flag in `$ACS_CACHE_PATH/residency.json` (by default, `/workspace/.aisha-cache/residency.json`). The manifest is a local record, not a distributed lock; operate one deployment at a time per node.

## Preflight decisions

The additive preflight is read-only and reports every finding before it downloads a model or invokes pip. Blocking findings stop the deployment:

| Code | Meaning | Remedy |
| --- | --- | --- |
| `requirements_full_lock` | The bundle declares a full requirements lock rather than an overlay. | Recreate the bundle's overlay against the node/template base manifest. |
| `comfyui_revision_mismatch` | The bundle pins a different ComfyUI revision from the running checkout. | Use a matching node/template or deploy it as a full, isolated environment. |
| `custom_node_pin_conflict` | A resident bundle pins the same custom-node directory to a different source or pin. | Align the pin/source, choose a distinct node directory, or use another node. |
| `model_sha_collision` | Two bundles use the same model destination with different SHA256 values. | Give the model a different destination/name, or use the identical verified artifact. |

`model_path_unverifiable` is advisory: the model destination is shared but a SHA256 is missing. An undetermined ComfyUI checkout also reports an advisory `comfyui_revision_mismatch`; verify the node manually before continuing.

`acs deploy --force --additive` is an interactive-only escape hatch. It logs each blocking finding and continues, but it does not make a collision safe. Automation and future Agent APIs deliberately have no equivalent override.

## Restarting after an additive deploy

An additive deploy marks the new bundle `pending_restart`; it does **not** restart a shared ComfyUI process. Restart it in a deliberate maintenance window and wait for the bundle's readiness node class:

```bash
acs comfyui restart --bundle wan_2.2_i2v
# Or, when validating a specific class directly:
acs comfyui restart --node-class "MyCustomNode"
```

The default command is `supervisorctl restart comfyui`, matching the repository's Vast/supervisord template program name. It remains unverified against a live Vast.ai node; confirm it with `supervisorctl status` before relying on the default in a new template. Override it with `ACS_COMFYUI_RESTART_COMMAND` when a deployment uses a different supervisor program. `ACS_COMFYUI_RESTART_TIMEOUT_SECONDS` and `ACS_COMFYUI_RESTART_POLL_INTERVAL_SECONDS` control the readiness wait. After a successful restart Aisha clears `pending_restart` for every resident bundle.

## Removing a bundle

Removal is manifest-driven so models shared by another resident bundle remain on disk. The command shows a plan and requires confirmation unless `--yes` is supplied:

```bash
acs remove wan_2.2_i2v
acs remove wan_2.2_i2v --retain ltx_i2v --yes
```

`--retain` adds bundles whose model declarations must be protected even if they are otherwise not in the removal set. Aisha removes the recorded workflow and only deletes model files that no retained resident bundle declares. It never removes custom-node directories or Python packages: those installations can be shared and are intentionally left for explicit operator maintenance.

## Corrupt or missing residency state

Missing residency state means the node has no recorded bundles. A corrupt, wrong-version, or unsafe manifest is rejected loudly and is never repaired automatically. Fix the manifest from a known-good backup when possible. If you delete it, you explicitly accept that the collision checks and shared-file reference counts for the previous deployments are no longer available; rebuild the resident declarations before using additive deployment or removal.
