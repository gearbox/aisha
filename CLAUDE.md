# CLAUDE.md

Guidance for working in this repository.

## Development environment

Use `uv sync --extra dev` as the only supported development install. It uses
the repository's `uv.lock` to provide the checked lint and type-tool versions.
`pip install -e ".[dev]"` is not equivalent and may resolve different tool or
stub versions.

## Style Rules

- **Async filesystem access**: wrap bulk-data or cross-device operations (`shutil.copy*`, `Path.replace` across devices, subprocess transfers) in `asyncio.to_thread`. Metadata syscalls (`exists`, `stat`, `mkdir`, `unlink`) run inline — thread dispatch costs more than the syscall. `ASYNC240` is disabled because its coverage is receiver-type-dependent.
- **Telemetry v2**: provisioning events are operation-scoped. Keep `status` and `phase` separate, do not add v1 callback paths, and sanitize diagnostics before emission.
