# CLAUDE.md

Guidance for working in this repository.

## Development environment

Use `uv sync --extra dev` as the only supported development install. It uses
the repository's `uv.lock` to provide the checked lint and type-tool versions.
`pip install -e ".[dev]"` is not equivalent and may resolve different tool or
stub versions.

Coverage on new code comes from `make diff-cover`; it checks changed lines
against `origin/master` at the same 85% threshold as CI. Duplication detection
was deliberately dropped rather than replaced.

## Standing lint debt

Per-file Ruff entries are grandfathering: a new module is expected to satisfy
every enabled rule without one. The 31 functions above `max-complexity = 12`
(`C901`, ranging from 13 to 28) are concentrated in `snapshot.py`,
`bundle_contract.py`, `cli.py`, and `downloader.py`; the tail above 20 is where
the refactoring value is. `PLR0913` entries are permanent because composition
roots deliberately expose their dependencies.

`TRY301` and the remaining `PLR2004` entries are per-file and should be
revisited at their individual call sites. `TRY003` and `TRY004` are permanent
decisions, with their rationale recorded beside the global ignores in
`pyproject.toml`.

## Private-repository review

Unauthenticated codeload fetches do not work for this private repository. To
share a reviewable branch without placing a credential in a chat transcript,
create an uploadable, clonable bundle that preserves the history needed to
diff against the pinned `master` base:

```bash
git bundle create aisha-<phase>.bundle master..feat/<branch> --branch=master
```

A short-lived repository-scoped PAT is also supported for review access.

## Style Rules

- **Async filesystem access**: wrap bulk-data or cross-device operations (`shutil.copy*`, `Path.replace` across devices, subprocess transfers) in `asyncio.to_thread`. Metadata syscalls (`exists`, `stat`, `mkdir`, `unlink`) run inline — thread dispatch costs more than the syscall. `ASYNC240` is disabled because its coverage is receiver-type-dependent.
- **Telemetry v2**: provisioning events are operation-scoped. Keep `status` and `phase` separate, do not add v1 callback paths, and sanitize diagnostics before emission.
- **Additive deployment**: preserve the residency manifest's strict fail-closed behavior. Additive preflight must finish before any deployment mutation; shared models are reference-counted on removal, while custom nodes and Python packages are never removed automatically.
- **ComfyUI restart**: additive deploys do not restart a shared process. Use the explicit restart-and-readiness path (default `supervisorctl restart comfyui`) and clear pending-restart flags only after it succeeds.
- **Provisioning agent**: claim and execute exactly one command at a time. Reuse the supplied long-lived reporter without closing it in composition roots, never pass `force`, and preserve a terminal v2 event even for a rejected batch command.
- **Agent service secrets**: keep `ACS_*` values in the mode-`0700` startup script. Do not place them in supervisord's `environment=` declaration; its parser cannot safely escape arbitrary tokens.
