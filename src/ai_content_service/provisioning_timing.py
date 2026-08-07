"""Per-deployment provisioning phase timing telemetry (Phase 2b-lite).

Always-on JSONL sink for provisioning phase durations. Deliberately
independent of Apex's callback machinery, which is only configured for
managed sessions -- exactly when this offline record is needed most: manual
nodes, benchmarks, and local runs. This module is pure: no console output,
no CLI framework, and no knowledge of any callback mechanism.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from .config import Settings

log = structlog.get_logger()

_GPU_QUERY_TIMEOUT_S = 3.0


class PhaseId(str, Enum):
    """Stable machine identifiers for provisioning phases (B-L3).

    Distinct from `ProvisioningReporter`'s display messages, and distinct
    from each other -- `deployer.py` previously emitted the same display
    name for both requirements phases, which made them indistinguishable in
    a timing series.
    """

    COMFYUI = "comfyui"
    REQUIREMENTS_BASE = "requirements_base"
    REQUIREMENTS_LOCKED = "requirements_locked"
    CUSTOM_NODES = "custom_nodes"
    MODELS = "models"
    WORKFLOW = "workflow"
    VERIFYING = "verifying"


@dataclass(frozen=True, slots=True)
class PhaseTiming:
    """One phase's outcome within a single deployment's timing record."""

    phase: PhaseId
    started_at: float
    """Epoch seconds -- wall clock, for correlating with other logs."""
    duration_s: float
    """Elapsed time from `time.monotonic()`, immune to clock steps."""
    skipped: bool
    """True when the phase did not apply to this bundle/mode (B contract #4:
    distinct from a real phase that happened to take 0.0s)."""


class ProvisioningTimer:
    """Records phase durations for one deployment and writes a JSONL record.

    Instrumentation only (B-L5): `write` never raises. A read-only disk, or
    any other failure while assembling the record, must not cost a
    deployment -- it is logged at WARNING and swallowed.
    """

    def __init__(self) -> None:
        self._start_monotonic = time.monotonic()
        self._phases: list[PhaseTiming] = []
        self._context: dict[str, object] = {}

    @contextlib.contextmanager
    def start(self, phase: PhaseId) -> Iterator[None]:
        """Time *phase*. A raising body still has its duration recorded."""
        started_at = time.time()
        started_mono = time.monotonic()
        try:
            yield
        finally:
            duration_s = max(0.0, time.monotonic() - started_mono)
            self._phases.append(
                PhaseTiming(
                    phase=phase, started_at=started_at, duration_s=duration_s, skipped=False
                )
            )

    def mark_skipped(self, phase: PhaseId) -> None:
        """Record *phase* as not applicable to this bundle/mode."""
        self._phases.append(
            PhaseTiming(phase=phase, started_at=time.time(), duration_s=0.0, skipped=True)
        )

    def record(self, key: str, value: object) -> None:
        """Attach free-form context (bundle, bundle_version, mode, models, env, ...)."""
        self._context[key] = value

    def write(self, path: Path, *, outcome: str, error: str | None = None) -> None:
        """Append one JSONL record to *path*. Never raises."""
        total_s = max(0.0, time.monotonic() - self._start_monotonic)
        payload: dict[str, object] = {
            "schema": 1,
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "outcome": outcome,
        }
        if error is not None:
            payload["error"] = error
        payload["total_s"] = round(total_s, 1)
        for key in ("bundle", "bundle_version", "mode"):
            if key in self._context:
                payload[key] = self._context[key]
        payload["phases"] = [
            {
                "phase": pt.phase.value,
                "duration_s": round(pt.duration_s, 1),
                "skipped": pt.skipped,
            }
            for pt in self._phases
        ]
        for key, value in self._context.items():
            if key not in ("bundle", "bundle_version", "mode"):
                payload[key] = value

        line = json.dumps(payload) + "\n"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # A single write() call of one newline-terminated line, opened in
            # append mode -- concurrent deployments on one node must not
            # interleave partial lines.
            with path.open("a", encoding="utf-8") as fh:
                fh.write(line)
        except OSError as exc:
            log.warning("provisioning_timing.write_failed", path=str(path), error=str(exc))


def detect_gpu_name() -> str | None:
    """Best-effort GPU name via `nvidia-smi`. `None` off-GPU or on failure."""
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return None
    try:
        result = subprocess.run(  # noqa: S603
            [nvidia_smi, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=_GPU_QUERY_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    first_line = result.stdout.strip().splitlines()[0].strip() if result.stdout.strip() else ""
    return first_line or None


def detect_instance_label() -> str | None:
    """Vast.ai's instance label (e.g. ``C.46979259``), or `None` off Vast.ai."""
    return os.environ.get("VAST_CONTAINERLABEL") or None


def build_env_context(
    settings: Settings,
    *,
    base_image: str | None,
    comfyui_source: str,
) -> dict[str, object]:
    """Environment provenance for a timing record (B-L4).

    Unknown provenance is `None` (JSON `null`), never a guess -- *base_image*
    is the caller's resolved `bundle.hardware.base_image`, since no verified
    Vast.ai environment variable carries the running container's image
    reference (`VAST_CONTAINERLABEL` is an instance label, not an image tag).
    """
    from . import __version__

    return {
        "base_image": base_image,
        "gpu": detect_gpu_name(),
        "cpu_count": os.cpu_count(),
        "aisha_version": __version__,
        "comfyui_source": comfyui_source,
        "hf_xet_enabled": settings.hf_xet_enabled,
        "instance": detect_instance_label(),
    }


def read_records(path: Path) -> list[dict[str, Any]]:
    """Parse a provisioning-timings JSONL file. Malformed lines are skipped."""
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records
