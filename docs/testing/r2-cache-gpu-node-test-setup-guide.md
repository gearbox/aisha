# Setup Guide — Unblock the R2 Cache GPU-Node Test

**Goal:** get a Vast.ai GPU node to serve a model from the R2 cache (a `cache hit`) during a bundle deploy.
**Grounded in `gearbox/aisha@master`.** The read path activates only when **all three** of `ACS_R2_S3_ENDPOINT`, `ACS_R2_READONLY_ACCESS_KEY_ID`, `ACS_R2_READONLY_SECRET_ACCESS_KEY` are set **and** `rclone` is on `PATH`. Objects are addressed at the key `models/by-sha256/{sha256}`. Missing any of these → cache silently stays off and every model downloads from upstream (safe, but no hit).

> **Scope note — write path is blocked.** `acs cache push` calls Apex endpoints (`/v1/admin/model-cache/credentials`, `/finalize`) that aren't built yet. So you cannot populate the cache via the CLI today. This guide populates R2 **manually** (Part 2) so the read path can be tested now; the CLI push gets tested after the Apex side ships.

---

## Part 1 — Cloudflare R2

1. **Create the bucket.** R2 → *Create bucket* → name `apex-model-cache`.
   - You're in NL: if you pick the **EU jurisdiction**, the S3 endpoint becomes `https://<ACCOUNT_ID>.eu.r2.cloudflarestorage.com` (note the `.eu.`). Default jurisdiction → `https://<ACCOUNT_ID>.r2.cloudflarestorage.com`. Use whichever matches what you create — this is the value for `ACS_R2_S3_ENDPOINT`.

2. **Note your Account ID.** R2 overview page (or any bucket's *Settings* → *S3 API*). It's the `<ACCOUNT_ID>` in the endpoint above.

3. **Create the read-only token (for the node).** R2 → *Manage R2 API Tokens* → *Create API token*:
   - Permission: **Object Read only**.
   - Scope: **Apply to specific buckets → `apex-model-cache`**.
   - After creation, copy the **Access Key ID** and **Secret Access Key** (the S3 credentials — *not* the "Token value" bearer string). These become `ACS_R2_READONLY_ACCESS_KEY_ID` / `ACS_R2_READONLY_SECRET_ACCESS_KEY`.

4. **Create a read-write token (temporary, for seeding).** Same flow, permission **Object Read & Write**, same bucket scope. Used only from your laptop/node in Part 2. Later this role is what Apex's parent token fills; you can delete it after seeding if you prefer.

5. **(Recommended, not required now)** Bucket → *Settings* → lifecycle: add an **Abort incomplete multipart uploads** rule at ~7 days. Irrelevant until real pushes happen, but set it once and forget it.

---

## Part 2 — Seed a test object (manual stand-in for `acs cache push`)

You need one real model file whose **sha256 matches the value declared in the bundle** you'll deploy (the read path verifies the pulled bytes against `file.sha256`). Pick a small-ish model from a test bundle in `ai-bundles` to keep the loop fast.

1. Get the file and its sha256 (the bundle's `sha256`, or compute it):
   ```bash
   sha256sum model.safetensors   # must equal the bundle's declared sha256
   ```

2. Upload it to the content-addressed key with rclone + the **read-write** token. Note the credential mechanism mirrors the code (`RCLONE_S3_*`, not `AWS_*`):
   ```bash
   export RCLONE_S3_ACCESS_KEY_ID=<rw-access-key-id>
   export RCLONE_S3_SECRET_ACCESS_KEY=<rw-secret>
   SHA=<sha256-of-the-file>

   rclone copyto ./model.safetensors \
     ":s3:apex-model-cache/models/by-sha256/${SHA}" \
     --s3-provider Cloudflare \
     --s3-endpoint=https://<ACCOUNT_ID>.r2.cloudflarestorage.com
   ```
   The object key is `models/by-sha256/<sha256>` with **no filename or extension** — that's intentional (content addressing; the node restores the filename on pull).

3. Confirm it's there:
   ```bash
   rclone lsl ":s3:apex-model-cache/models/by-sha256/" \
     --s3-provider Cloudflare \
     --s3-endpoint=https://<ACCOUNT_ID>.r2.cloudflarestorage.com
   ```

---

## Part 3 — Vast.ai template changes

### 3a. Put `rclone` on the node

The provisioning scripts don't install it, and the `vastai/comfy` image isn't guaranteed to ship it. Two options:

- **For the template (recommended, durable):** bake a pinned rclone static binary into your image at build time and ensure it's on `PATH` (e.g. `/usr/local/bin/rclone`). Pin a specific current stable release and verify its checksum — consistent with the repo's deliberate "no `curl | sh` at runtime" stance. Any modern rclone works (session-token + `RCLONE_S3_*` support is long-standing; multi-thread download needs ≥ 1.56).
- **For a quick one-off test:** install it on the node before the deploy step:
  ```bash
  cd /tmp && V=$(curl -s https://downloads.rclone.org/version.txt | awk '{print $2}')
  curl -sLO "https://downloads.rclone.org/${V}/rclone-${V}-linux-amd64.zip"
  unzip -q "rclone-${V}-linux-amd64.zip" && install -m 755 rclone-*/rclone /usr/local/bin/rclone
  rclone version
  ```
  If you'd rather not touch `PATH`, set `ACS_RCLONE_PATH=/full/path/to/rclone`.

### 3b. Add the R2 env vars to the instance environment

These are read from the instance env at deploy time (same mechanism as every other `ACS_*` var). Add to your Vast.ai template's environment (alongside `ACS_BUNDLE`, `ACS_GITHUB_TOKEN`, etc.):

| Env var | Value | Required for read path |
|---------|-------|------------------------|
| `ACS_R2_S3_ENDPOINT` | `https://<ACCOUNT_ID>.r2.cloudflarestorage.com` (or `.eu.` variant) | **yes** |
| `ACS_R2_READONLY_ACCESS_KEY_ID` | read-only Access Key ID (Part 1.3) | **yes** |
| `ACS_R2_READONLY_SECRET_ACCESS_KEY` | read-only Secret Access Key (Part 1.3) | **yes** |
| `ACS_R2_MODEL_CACHE_BUCKET` | `apex-model-cache` | only if you renamed the bucket (default already matches) |

All three "yes" vars must be present or `_r2_enabled` stays false and the cache is skipped. The read-only token is acceptable to bake onto a third-party host: it can't write/delete the shared cache, and the node already holds the plaintext weights on disk during provisioning, so it exposes nothing new.

Leave `ACS_APEX_BASE_URL` / `ACS_APEX_ADMIN_TOKEN` **unset** for now — they're only for the (still-unbuilt) push path.

---

## Part 4 — Run the test and what to look for

1. Launch an instance from the template, with the env above plus your usual deploy vars, pointing `ACS_BUNDLE` at the bundle whose model you seeded in Part 2.
2. During the deploy/model-download phase, watch the console:
   - Seeded model → **`cache hit  <filename>`** (green), no upstream fetch.
   - Other models → **`cache miss <filename> — fetching upstream`** (yellow).
3. **Negative test (fallback is safe):** delete the R2 object, redeploy → expect `cache miss` + clean upstream download + **successful deploy**.
4. **rclone-missing test (graceful degrade):** set `ACS_RCLONE_PATH=/nonexistent`, redeploy → deploy still succeeds entirely via upstream.

### Quick auth sanity check on the node (isolates Cloudflare/token problems from aisha)
```bash
export RCLONE_S3_ACCESS_KEY_ID=$ACS_R2_READONLY_ACCESS_KEY_ID
export RCLONE_S3_SECRET_ACCESS_KEY=$ACS_R2_READONLY_SECRET_ACCESS_KEY
rclone lsl ":s3:apex-model-cache/models/by-sha256/" \
  --s3-provider Cloudflare --s3-endpoint="$ACS_R2_S3_ENDPOINT"
```
Lists the seeded object → creds/endpoint are good. `403`/`AccessDenied` → token permission or endpoint (check the `.eu.` jurisdiction) is wrong.

---

## Troubleshooting

| Symptom | Likely cause |
|---------|--------------|
| Every model shows `cache miss`, even the seeded one | One of the three required env vars missing → cache disabled; or rclone not on `PATH`; or the seeded key doesn't match `models/by-sha256/<sha256>` exactly. |
| `403` / `AccessDenied` on the sanity check | Wrong token (used the bearer "Token value" instead of S3 Access Key/Secret), token not scoped to this bucket, or wrong endpoint (EU jurisdiction needs `.eu.`). |
| `cache miss` with an exception logged, model still downloads | Expected graceful fallback — read it as "cache didn't serve this one," not a deploy failure. Check the logged `exc=` to see why (often a key mismatch or transient R2 error). |
| `cache corrupt` in logs, then upstream | The seeded object's bytes don't hash to the sha256 in its key / the bundle's declared sha256. Re-seed the correct file. |
| Pull seems slow on a big model | Multi-GB pulls aren't wall-clock-capped (that was removed); rclone's `--timeout` is idle-only. A slow link is just slow — it still completes and verifies. |

---

## When Apex ships (later)

To test `acs cache push` end-to-end: set `ACS_APEX_BASE_URL` and `ACS_APEX_ADMIN_TOKEN`, give Apex the **read-write** parent token + a standing R2 credential that can HEAD `apex-model-cache` (for `finalize`), and run `acs cache push <bundle> --model <file>` on a node that already deployed the bundle. Until then, Part 2's manual rclone seed is the way to populate the cache.

---

### One-line summary of changes
- **Cloudflare:** bucket `apex-model-cache` + a read-only S3 token (node) + a temporary read-write token (seeding); note the account endpoint (`.eu.` if EU).
- **Template:** install/bake `rclone`; set `ACS_R2_S3_ENDPOINT`, `ACS_R2_READONLY_ACCESS_KEY_ID`, `ACS_R2_READONLY_SECRET_ACCESS_KEY` (+ bucket var only if renamed).
- **Seed once** at `models/by-sha256/<sha256>`, then deploy and watch for the green `cache hit`.
