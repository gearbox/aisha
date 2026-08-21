# Changelog

## 0.14.0

- Snapshotting now aborts without writing an artifact when ComfyUI provider coverage cannot be correlated. Start ComfyUI and re-run, or use `--force` to create a deliberately invalid, non-deployable inspection artifact.
- `--force` is now the only custom-node escape hatch; `--allow-unverified-custom-nodes` has been removed.
