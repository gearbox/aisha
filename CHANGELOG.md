# Changelog

## 0.17.0

- Added workflow contract v2 with explicit graph media, semantic media-input slots, video
  parameters, and API-link validation for every declared media target.

## 0.16.0

- Snapshots from template-pinned seeds now leave ComfyUI to the template, avoiding a redundant
  checkout and base-requirements reinstall during full deployment.
- Snapshotting reports when the live ComfyUI revision has drifted from the pristine base image.

## 0.14.1

- Snapshotting no longer aborts for an archive-installed custom node that the workflow does not use. Registry lookup failures remain attached to a skipped provider when it is relevant, while carried seed pins and known git providers continue to certify normally.

## 0.14.0

- Snapshotting now aborts without writing an artifact when ComfyUI provider coverage cannot be correlated. Start ComfyUI and re-run, or use `--force` to create a deliberately invalid, non-deployable inspection artifact.
- `--force` is now the only custom-node escape hatch; `--allow-unverified-custom-nodes` has been removed.
