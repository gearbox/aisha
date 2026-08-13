# Qwen upgrade runbook — template-owned environment and provisioning verification

**Goal:** move `qwen_rapid_aio` to a tested Vast.ai template, verify one
generation, and retain a small, valid measurement of the only remaining
environment-pinning cost.

The old four-variant experiment is complete. Do not recreate it for this
upgrade.

---

## Decisions already supported by the measurement

The original measurement ran on a warm RTX 4090 with
`vastai/comfy:v0.32.0-cuda-13.2-py312`:

| variant | outcome | total |
|---|---|---:|
| `v-full` | failed in `requirements_locked` after 1.9 s | — |
| `v-noco` | failed in `requirements_locked` after 0.2 s | — |
| `v-nolock` | ready | 24.5 s |
| `v-thin` | ready | 23.1 s |

The failed rows are not results and must not be used in a timing delta.
`v-nolock` minus `v-thin` was only **1.4 s** for the ComfyUI checkout and its
base requirements. Custom-node installation was about 1.5 s where applicable;
`qwen_rapid_aio` itself declares no custom nodes.

More importantly, the pristine-image manifest and the bundle lock differed as
follows:

```text
lock packages:      159
base packages:      161
identical version:  159
version differs:      0
absent from base:     0
```

The lock is therefore a byte-for-byte subset of the base image package set.
The image already ships the bundle's `comfyui.commit`, too. Reinstalling the
lock with `--ignore-installed` was the failure: it attempted to reinstall torch,
CUDA, and conda-provided packages from build-only paths.

The resulting policy is:

- The Vast.ai template owns ComfyUI, CUDA, Python, and the base package set.
- `hardware.template_hash_id` is the environment pin. On the Qwen upgrade
  branch, set it together with `hardware.base_image`.
- Remove Qwen's redundant `comfyui:` block and `requirements_lock_file`. They
  remain supported only as exceptional overlays for a dependency unavailable
  from a template.
- Do not regenerate the lock or chase a ComfyUI commit for a template-only
  upgrade. The withdrawn image-baking design remains withdrawn.

The current deployer measures a retained lock against the live environment
before it invokes pip. A matching lock records
`requirements.lock.delta` and marks `requirements_locked` as **skipped**; a
missing or conflicting package is visible before any install begins.

---

## 0 — Rent and set up

Create a Vast.ai template from `vastai/comfy:v0.32.0-cuda-13.2-py312`, then
record its template hash ID. Use an RTX 4090 or better, `runtype: ssh`, and
enough disk for the Qwen model plus the temporary benchmark download. A
high-bandwidth node (`inet_down ≥ 1000 Mbps`) is required if download throughput
is part of this session.

```bash
ssh -p <port> root@<host>
export WORKSPACE=/workspace
export AISHA_PATH=$WORKSPACE/aisha
export BUNDLES_PATH=$WORKSPACE/ai-bundles
export COMFYUI_PATH=/opt/workspace-internal/ComfyUI
export ACS_COMFYUI_PATH=$COMFYUI_PATH
export ACS_COMFYUI_PYTHON=/venv/main/bin/python
export ACS_BUNDLES_PATH=$BUNDLES_PATH       # registry root, not /bundles
export ACS_COMFYUI_PORT=18188
export ACS_BASE_IMAGE=vastai/comfy:v0.32.0-cuda-13.2-py312
export TEMPLATE_HASH_ID=<Vast.ai template hash ID>
export ACS_GITHUB_TOKEN=... ACS_HF_TOKEN=... ACS_CIVITAI_API_TOKEN=...
export PATH=/opt/instance-tools/bin:$PATH

# Confirm the actual image layout before relying on these defaults.
ls -d $COMFYUI_PATH
$ACS_COMFYUI_PYTHON -c 'import torch; print(torch.__version__, torch.version.cuda)'
command -v uv
```

If the image relocates ComfyUI or its interpreter, correct the corresponding
`ACS_COMFYUI_*` variable here. Do not rely on the venv that runs `acs` to target
the ComfyUI environment.

```bash
git clone --branch master https://x-access-token:$ACS_GITHUB_TOKEN@github.com/gearbox/aisha.git $AISHA_PATH
git clone --branch master https://x-access-token:$ACS_GITHUB_TOKEN@github.com/gearbox/ai-bundles.git $BUNDLES_PATH
uv venv $WORKSPACE/aisha-venv
cd $AISHA_PATH
uv pip install -e . --python $WORKSPACE/aisha-venv/bin/python
export PATH=$WORKSPACE/aisha-venv/bin:$PATH
acs --version
```

---

## 1 — Capture the pristine base-image manifest

Do this before any `acs deploy`. It establishes the package set against which a
retained lock is evaluated. `base_image` is deliberately `null` when
`ACS_BASE_IMAGE` is unset; an instance ID is not image provenance.

```bash
bash $AISHA_PATH/scripts/capture-env-manifest.sh > $WORKSPACE/base-manifest.json
jq '{base_image, instance, package_count, comfyui_version, comfyui_commit, torch, torch_cuda}' \
  $WORKSPACE/base-manifest.json
```

Expected: `base_image` is the value exported above and `instance` is recorded
separately. Preserve this file even if the upgrade does not proceed.

---

## 2 — Prepare only the two informative benchmark variants

Copy the current Qwen bundle to a scratch registry; leave the real registry
untouched while measuring.

```bash
mkdir -p $WORKSPACE/bench-bundles/bundles
cp -r $BUNDLES_PATH/bundles/qwen_rapid_aio $WORKSPACE/bench-bundles/bundles/
cd $WORKSPACE/bench-bundles/bundles/qwen_rapid_aio
VER=$(readlink current)
for v in v-full v-thin; do cp -r $VER $v; done
```

For **both** variants, set `metadata.version` to the directory name and add:

```yaml
hardware:
  base_image: "vastai/comfy:v0.32.0-cuda-13.2-py312"
  template_hash_id: "<Vast.ai template hash ID>"
```

Then edit only the environment-pinning fields:

| variant | `comfyui:` | `requirements_lock_file:` | purpose |
|---|---|---|---|
| `v-full` | retain | retain | verify the legacy overlay is now delta-aware and safely skipped |
| `v-thin` | remove | remove | the target template-only configuration |

Add both directories to `$WORKSPACE/bench-bundles/bundle-index.yaml` with
separate names (`v-full` and `v-thin`). The `v-full` warnings are intentional:
it has both a template pin and bundle-level pins. The `v-thin` configuration is
the one to ship.

```bash
export ACS_BUNDLES_PATH=$WORKSPACE/bench-bundles
acs bundle validate --all
```

Expect no errors. `v-full` should warn with `environment.dual_pinning` and
`requirements_lock.redundant`; `v-thin` should have no environment-pinning
warning. An omitted `hardware.comfyui_port` is valid and means Apex's default,
18188.

Do not create `v-noco` or `v-nolock`. The 159/159 lock match already answers
the lock question, and `v-full` versus `v-thin` isolates the remaining ComfyUI
checkout plus base-requirements cost.

---

## 3 — Run the two-variant acceptance check

The harness runs the provisioning matrix and, by default, the 28 GB checkpoint
download A/B. Set `CKPT_URL` to the Qwen checkpoint's Hugging Face
`.../resolve/...` URL and `CKPT_NAME` to its filename before running it.

To measure actual model transfer, use an isolated benchmark node and first
remove its existing models. This deletes the contents of the node's ComfyUI
model directory; do not use this command on a node whose models must be kept.

```bash
export CKPT_URL=<Qwen 28 GB Hugging Face resolve URL>
export CKPT_NAME=<checkpoint filename>
set -o pipefail
rm -rf $ACS_COMFYUI_PATH/models/*
VARIANTS="v-full v-thin" MIN_FREE_GB=40 \
  bash $AISHA_PATH/scripts/bench/bench-provision.sh \
  2>&1 | tee $WORKSPACE/provision-bench.log

acs timings show --last 2
acs timings show --last 2 --json > $WORKSPACE/timings.json
```

The harness prints the model state during preflight. If files are already
present, the models phase measures skip-existing verification rather than
transfer; that is valid, but is not a download-speed result.

A failed variant produces a red `FAILED VARIANTS` block and makes the harness
exit non-zero. Because `pipefail` is enabled above, the `tee` pipeline also
fails. Stop there: any timing delta involving a failed variant is meaningless.

If the slow xet-off comparison is not worth another roughly 30 minutes, retain
the matrix but skip only that baseline:

```bash
SKIP_SLOW_BASELINE=1 VARIANTS="v-full v-thin" MIN_FREE_GB=40 \
  bash $AISHA_PATH/scripts/bench/bench-provision.sh
```

---

## 4 — Read the result

Inspect the timing records, including the lock delta recorded under `metrics`:

```bash
acs timings show --last 2 --json | jq -c \
  '{bundle, bundle_version, outcome, total_s,
    requirements_locked: .metrics.requirements_locked,
    models: .metrics.models}'
```

For `v-full`, expect:

- `requirements_locked` has status `skipped`;
- `metrics.requirements_locked` reports `missing: 0`, `conflicting: 0`, and
  `outcome: "skipped"`;
- the deploy log includes `requirements.lock.delta`.

For `v-thin`, `requirements_locked` is skipped because there is no lock and no
lock-delta metric is expected. Compare `v-full` and `v-thin` only when both
deployment outcomes are ready. The difference is the remaining cost of the
legacy ComfyUI checkout and base requirements; it should be small enough that
there is no case for image baking.

For the models phase, use `metrics.models.materialized_bytes` and
`metrics.models.effective_mib_per_s`. A result with
`materialized_bytes: 0` measures verification, not transfer. The download
benchmark prints net MB/s separately from the post-download digest estimate.

---

## 5 — Update the real bundle

On a bundle branch, make the real `qwen_rapid_aio` version template-only:

```yaml
hardware:
  base_image: "vastai/comfy:v0.32.0-cuda-13.2-py312"
  template_hash_id: "<Vast.ai template hash ID>"
```

Remove the `comfyui:` block and the `requirements_lock_file` entry from that
version. Do not run `acs snapshot` just to update this environment: a snapshot
captures the node's ComfyUI commit and full pip freeze, which would reintroduce
the redundant pins. Preserve any existing model hashes, URLs, workflow,
generation settings, and readiness marker.

Mark `metadata.tested: false`, repoint `current`, and validate:

```bash
export ACS_BUNDLES_PATH=$BUNDLES_PATH
acs bundle validate qwen_rapid_aio
acs deploy -b qwen_rapid_aio --sync
# Open ComfyUI, run one Qwen generation, and confirm the output.
```

After the generation succeeds, set `metadata.tested: true` and commit the
two-field template update plus the removed legacy pins. A lock or `comfyui:`
override should return only when the bundle has a demonstrated dependency that
the chosen template cannot provide.

---

## 6 — Preserve evidence and tear down

Copy the provenance, timing, and benchmark logs off the node before destroying
it:

```bash
scp -P <port> root@<host>:/workspace/{timings.json,base-manifest.json,provision-bench.log} .
```

## Exit criteria

- [ ] `base-manifest.json` was captured before the first deployment and has an
  image tag (or an intentional `null`), never an instance ID, in `base_image`.
- [ ] Both `v-full` and `v-thin` completed, or a failed matrix was treated as
  failed rather than compared.
- [ ] The `v-full` lock delta is a measured skipped zero:
  `missing=0` and `conflicting=0`.
- [ ] The Qwen bundle has `hardware.base_image` and
  `hardware.template_hash_id`, with no redundant `comfyui:` or
  `requirements_lock_file`.
- [ ] One generation succeeded on the template-only bundle and
  `metadata.tested` is true.
- [ ] Timing records and the manifest were copied off the node; the instance
  was destroyed.
