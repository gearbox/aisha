# Provisioning agent and batch commands

After a successful bootstrap deployment, the Vast provisioning script installs
`aisha-agent` under supervisord when `ACS_APEX_CALLBACK_URL`,
`ACS_APEX_CALLBACK_TOKEN`, and `ACS_APEX_SESSION_ID` are present. The agent
claims work from:

```
POST /v1/internal/gpu-sessions/{session_id}/commands/claim
{"agent_id":"{session}:{hostname}","schema_version":2}
```

The claim response is either `204` (no work) or a command envelope. Apex owns
the `operation_id`; Aisha uses it unchanged for the v2 operation-event stream.

```json
{
  "command_id": "cmd_01...",
  "operation_id": "01...",
  "kind": "bundle_provision",
  "batch": {"batch_id": "01...", "index": 0, "total": 3},
  "payload": {
    "bundle": "gearbox/wan:260801-01",
    "mode": "additive",
    "verify": true,
    "batch_declared_bytes": 94489280512
  }
}
```

Supported payloads are `bundle_provision` (`bundle`, `mode`, optional
`verify`), `bundle_removal` (`bundle`, optional `retain_bundles`), and
`comfyui_restart` (optional `node_class`). `session_bootstrap`, unknown modes,
and any `force` field are rejected. The agent never enables force: additive
preflight remains fail-closed.

## Execution and stop semantics

There is exactly one in-flight command. The agent claims again immediately
after a command reaches its terminal event; it sleeps with jitter only after a
`204`. Transport failures and rejected claims back off exponentially up to
`ACS_AGENT_MAX_BACKOFF_SECONDS`; every successful claim, including `204`,
resets the backoff. SIGTERM and SIGINT stop future claims but allow the active
download/deployment to complete and emit its terminal event.

The generated startup script also waits for `/.provisioning` to disappear.
That cross-process guard prevents a freshly installed supervisor program from
claiming work while the bootstrap deploy is still mutating the node.

## Batches and disk headroom

A batch is independent commands sharing a `batch_id`, not one aggregate
operation. Each command carries its own `operation_id` and terminal status.
On batch index zero, `batch_declared_bytes` is checked against free space at
the models directory plus `ACS_BATCH_DISK_MARGIN` (default five percent).

The declared figure is gross: Aisha does not subtract files that an earlier
bundle might reuse because it does not resolve every batch member in advance.
It can therefore conservatively refuse a batch that would fit with reuse. If
the guard refuses, later commands with the same batch id fail immediately and
emit their own terminal event; no bundle is resolved or downloaded. Aisha
retains at most 32 abandoned batch ids in FIFO order.

## Service installation

Use `acs agent install-service` after provisioning, or inspect it safely with
`acs agent install-service --dry-run`. It writes:

- `/opt/supervisor-scripts/aisha-agent.sh` (mode `0700`), containing the
  `ACS_*` environment exports and the provisioning-marker wait;
- `/etc/supervisor/conf.d/aisha-agent.conf` (mode `0644`), containing only
  `PROC_NAME` in `environment=`.

The split is deliberate. Supervisord's `environment=` parser cannot safely
represent arbitrary values containing quotes, so callback and GitHub tokens
must remain in the private shell script rather than the public conf.
