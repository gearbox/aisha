# Provisioning telemetry v2

Every provisioning-like action emits an operation-scoped event stream to:

```
POST /v1/internal/gpu-sessions/{session_id}/operations/{operation_id}/events
```

An event has `schema_version: 2`, a fresh UUIDv7 `event_id`, a stable UUIDv7
`operation_id`, and an operation-local `sequence`. The start event has sequence
zero and carries the complete execution `plan`. Phase and progress events are
best-effort. Terminal success or failure events carry a timing `summary` and
are retried up to five times for transport errors and 5xx responses.

`sequence` advances only for emitted events. Progress events suppressed by the
throttle consume no sequence value. Retried terminal requests reuse both their
event ID and sequence, allowing Apex to de-duplicate delivery safely.

`status` (`running`, `succeeded`, or `failed`) and `phase` are independent:
terminal events have no phase, and phase names are never statuses. Progress is
generic (`work`, `items`, rate, ETA) so later operation kinds can use the same
envelope. ETAs use EWMA throughput over newly materialized bytes only; verified
reused files count toward completion but never inflate throughput.

`message` is diagnostics only, not a UI or machine contract. Consumers render
from `status`, `phase`, `progress`, `target`, and `plan`. Aisha sanitizes error
and message values before they leave the process.

The local `provisioning-timings.jsonl` record remains schema 2 with its shipped
`*_s` fields. `ProvisioningTimer.snapshot()` translates that data into the wire
summary's `*_seconds` fields and deliberately excludes environment provenance.
