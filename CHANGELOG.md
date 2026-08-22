# Changelog

## 0.14.1

- Snapshotting no longer aborts for an archive-installed custom node that the workflow does not use. Registry lookup failures remain attached to a skipped provider when it is relevant, while carried seed pins and known git providers continue to certify normally.

## 0.14.0

- Snapshotting now aborts without writing an artifact when ComfyUI provider coverage cannot be correlated. Start ComfyUI and re-run, or use `--force` to create a deliberately invalid, non-deployable inspection artifact.
- `--force` is now the only custom-node escape hatch; `--allow-unverified-custom-nodes` has been removed.
