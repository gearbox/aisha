# CLAUDE.md

Guidance for working in this repository.

## Style Rules

- **Async filesystem access**: wrap bulk-data or cross-device operations (`shutil.copy*`, `Path.replace` across devices, subprocess transfers) in `asyncio.to_thread`. Metadata syscalls (`exists`, `stat`, `mkdir`, `unlink`) run inline — thread dispatch costs more than the syscall. `ASYNC240` is disabled because its coverage is receiver-type-dependent.
