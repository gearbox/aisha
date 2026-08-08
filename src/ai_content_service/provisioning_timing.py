"""Versioned, best-effort provisioning timing telemetry.

The JSONL sink is intentionally independent of Apex callbacks.  Records use
schema 2: deployment facts are fixed root fields, extensible observations live
under ``metrics``, and advisory bundle provenance is kept separate from runtime
observations.  Writes are atomic for concurrent *local* writers via one
``O_APPEND``/``os.write`` call.  Shared/network filesystem atomicity is a
property of that filesystem and is not promised here.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import subprocess
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any

import structlog

from .download_auth import redact_url

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from pathlib import Path

    from .config import Settings

log = structlog.get_logger()

_GPU_QUERY_TIMEOUT_S = 3.0
_ERROR_LIMIT = 4_096
_AUTHORIZATION_RE = re.compile(r"(?i)(authorization\s*[:=]\s*)(?:bearer\s+)?[^\s,;]+")
_IDENTITY_KEYS = frozenset({"bundle", "bundle_version", "mode"})


class PhaseId(str, Enum):
    """Stable machine identifiers for provisioning phases."""

    COMFYUI = "comfyui"
    REQUIREMENTS_BASE = "requirements_base"
    REQUIREMENTS_LOCKED = "requirements_locked"
    CUSTOM_NODES = "custom_nodes"
    MODELS = "models"
    WORKFLOW = "workflow"
    VERIFYING = "verifying"


class PhaseStatus(str, Enum):
    """Explicit phase outcome, independent of the deployment's outcome."""

    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class PhaseTiming:
    """One phase's chronology, duration, and outcome."""

    phase: PhaseId
    started_at: float
    """UTC epoch seconds, used only for chronology/correlation."""
    duration_s: float
    """Elapsed monotonic seconds; immune to wall-clock adjustments."""
    skipped: bool
    """Legacy-compatible convenience field; schema 2 serializes ``status``."""
    status: PhaseStatus = PhaseStatus.COMPLETED


def _utc_timestamp(epoch_s: float) -> str:
    return datetime.fromtimestamp(epoch_s, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sanitize_error(error: str, *, secrets: Iterable[str] = ()) -> str:
    """Redact and bound an error before it reaches durable telemetry.

    The same URL/query redactor used by download failures handles known token
    values and sensitive query keys.  Authorization text gets an additional
    syntactic redaction because it may contain a credential unknown to this
    process (for example, a proxy-generated header).
    """
    sanitized = redact_url(error, secrets=secrets)
    sanitized = _AUTHORIZATION_RE.sub(r"\1***", sanitized)
    if len(sanitized) > _ERROR_LIMIT:
        return f"{sanitized[:_ERROR_LIMIT]}… [truncated]"
    return sanitized


class ProvisioningTimer:
    """Record one deployment's timing telemetry without affecting deployment.

    ``finish()`` freezes the deployment duration.  Deployers must call it at
    the deployment boundary, before terminal callbacks and environment probes.
    ``write()`` retains a compatibility fallback for standalone callers that
    did not explicitly finish, but never changes an already-frozen duration.
    """

    def __init__(self) -> None:
        self._started_at = time.time()
        self._start_monotonic = time.monotonic()
        self._finished_at: float | None = None
        self._total_s: float | None = None
        self._phases: list[PhaseTiming] = []
        self._identity: dict[str, object] = {}
        self._metrics: dict[str, object] = {}
        self._env: dict[str, object] | None = None

    @contextlib.contextmanager
    def start(self, phase: PhaseId) -> Iterator[None]:
        """Time *phase*, recording failed/cancelled bodies before re-raising."""
        started_at = time.time()
        started_mono = time.monotonic()
        status = PhaseStatus.COMPLETED
        try:
            yield
        except BaseException:
            status = PhaseStatus.FAILED
            raise
        finally:
            duration_s = max(0.0, time.monotonic() - started_mono)
            self._phases.append(
                PhaseTiming(
                    phase=phase,
                    started_at=started_at,
                    duration_s=duration_s,
                    skipped=False,
                    status=status,
                )
            )

    def mark_skipped(self, phase: PhaseId) -> None:
        """Record *phase* as not applicable to this bundle/mode."""
        self._phases.append(
            PhaseTiming(
                phase=phase,
                started_at=time.time(),
                duration_s=0.0,
                skipped=True,
                status=PhaseStatus.SKIPPED,
            )
        )

    def finish(self) -> None:
        """Freeze the total duration exactly once."""
        if self._total_s is not None:
            return
        self._finished_at = time.time()
        self._total_s = max(0.0, time.monotonic() - self._start_monotonic)

    def record(self, key: str, value: object) -> None:
        """Attach identity or an extensible metric without root-key collisions.

        Only the three typed identity keys stay at the root.  Every other key,
        including a would-be core key such as ``total_s`` or ``schema``, lives
        below ``metrics`` and can never replace an authoritative timer field.
        """
        if key in _IDENTITY_KEYS:
            self._identity[key] = value
        else:
            self._metrics[key] = value

    def record_metric(self, key: str, value: object) -> None:
        """Attach an explicitly named metric below the schema's metrics object."""
        self._metrics[key] = value

    def record_env(self, value: dict[str, object]) -> None:
        """Attach the structured environment/provenance section."""
        self._env = value

    def _payload(
        self, *, outcome: str, error: str | None, secrets: Iterable[str]
    ) -> dict[str, object]:
        self.finish()
        finished_at = self._finished_at
        total_s = self._total_s
        if finished_at is None or total_s is None:
            msg = "timer did not finalize"
            raise RuntimeError(msg)
        phase_sum_s = sum(pt.duration_s for pt in self._phases if not pt.skipped)
        payload: dict[str, object] = {
            "schema": 2,
            "started_at": _utc_timestamp(self._started_at),
            "finished_at": _utc_timestamp(finished_at),
            "outcome": outcome,
            "error": sanitize_error(error, secrets=secrets) if error is not None else None,
            "total_s": round(total_s, 3),
            "phase_sum_s": round(phase_sum_s, 3),
            "overhead_s": round(max(total_s - phase_sum_s, 0.0), 3),
            "phases": [
                {
                    "phase": pt.phase.value,
                    "started_at": _utc_timestamp(pt.started_at),
                    "duration_s": round(pt.duration_s, 3),
                    "status": pt.status.value,
                }
                for pt in self._phases
            ],
            "metrics": self._metrics,
        }
        payload |= self._identity
        if self._env is not None:
            payload["env"] = self._env
        return payload

    def write(
        self,
        path: Path,
        *,
        outcome: str,
        error: str | None = None,
        secrets: Iterable[str] = (),
    ) -> None:
        """Append one JSONL record to *path*.  Instrumentation never raises.

        A local filesystem writer uses ``O_APPEND`` plus exactly one
        ``os.write`` call for the newline-terminated record.  That prevents
        local concurrent writers from interleaving a record; it deliberately
        makes no stronger claim for distributed/network filesystems.
        """
        try:
            line = (
                json.dumps(self._payload(outcome=outcome, error=error, secrets=secrets)) + "\n"
            ).encode("utf-8")
            path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
            try:
                os.write(fd, line)
            finally:
                os.close(fd)
        except Exception as exc:
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
    bundle_base_image: str | None,
    comfyui_source: str,
) -> dict[str, object]:
    """Return observed environment facts and separately labelled provenance.

    ``bundle_base_image`` is advisory metadata from bundle.yaml.  Aisha does
    not currently have an authoritative runtime-image source, so
    ``runtime_base_image`` is intentionally null rather than guessed.
    """
    from . import __version__

    return {
        "bundle_base_image": bundle_base_image,
        "runtime_base_image": None,
        "gpu": detect_gpu_name(),
        "cpu_count": os.cpu_count(),
        "aisha_version": __version__,
        "comfyui_source": comfyui_source,
        "hf_xet_enabled": settings.hf_xet_enabled,
        "instance": detect_instance_label(),
    }


def iter_records(path: Path) -> Iterator[dict[str, Any]]:
    """Yield valid JSON-object records one line at a time.

    Missing/unreadable histories produce a structured warning and no records;
    a bad individual line never hides later records.
    """
    try:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    yield record
    except FileNotFoundError:
        return
    except OSError as exc:
        log.warning("provisioning_timing.read_failed", path=str(path), error=str(exc))


def read_records(
    path: Path,
    *,
    bundle: str | None = None,
    last: int | None = None,
) -> list[dict[str, Any]]:
    """Read selected JSONL history with streaming filtering and bounded tails."""
    if last is not None and last < 1:
        msg = "last must be a positive integer"
        raise ValueError(msg)
    selected = (
        record for record in iter_records(path) if bundle is None or record.get("bundle") == bundle
    )
    return list(selected) if last is None else list(deque(selected, maxlen=last))
