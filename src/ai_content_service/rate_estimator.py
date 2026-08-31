"""Pure EWMA throughput and ETA calculations for operation telemetry."""

from __future__ import annotations

from dataclasses import dataclass, field

_MIN_RATE_BYTES_PER_SECOND = 1024.0
_MAX_ETA_SECONDS = 86_400.0


@dataclass
class ThroughputEstimator:
    """Estimate materialization rate from absolute byte observations."""

    alpha: float = 0.3
    warmup_seconds: float = 5.0
    warmup_bytes: int = 268_435_456
    _first_at: float | None = field(default=None, init=False)
    _last_at: float | None = field(default=None, init=False)
    _last_bytes: int = field(default=0, init=False)
    _materialized_bytes: int = field(default=0, init=False)
    _rate: float | None = field(default=None, init=False)

    def observe(self, *, materialized_bytes: int, now_monotonic: float) -> None:
        """Add an absolute materialized-byte observation to the EWMA."""
        if self._first_at is None:
            self._first_at = now_monotonic
            self._last_at = now_monotonic
            self._last_bytes = materialized_bytes
            self._materialized_bytes = materialized_bytes
            return
        previous_at = self._last_at
        if previous_at is None:
            return
        elapsed = now_monotonic - previous_at
        self._materialized_bytes = materialized_bytes
        if elapsed <= 0:
            return
        sample = max(materialized_bytes - self._last_bytes, 0) / elapsed
        self._rate = (
            sample if self._rate is None else self.alpha * sample + (1 - self.alpha) * self._rate
        )
        self._last_at = now_monotonic
        self._last_bytes = materialized_bytes

    def rebase(self, *, materialized_bytes: int, now_monotonic: float) -> None:
        """Reset the absolute-observation baseline without sampling a jump.

        A file reclassified from reused to materialized can add its accumulated
        bytes to the materialized total at once.  That is accounting, not
        transfer throughput, so it must not feed the EWMA.
        """
        if self._first_at is None:
            self._first_at = now_monotonic
        self._last_at = now_monotonic
        self._last_bytes = materialized_bytes
        self._materialized_bytes = materialized_bytes

    def rate_bytes_per_second(self) -> float | None:
        """Return a rate once both the time and byte warm-up gates are open."""
        if self._first_at is None or self._last_at is None or self._rate is None:
            return None
        if self._last_at - self._first_at < self.warmup_seconds:
            return None
        if self._materialized_bytes < self.warmup_bytes:
            return None
        return self._rate if self._rate >= _MIN_RATE_BYTES_PER_SECOND else None

    def eta_seconds(self, *, remaining_materialized_bytes: int | None) -> float | None:
        """Return a non-negative ETA, or ``None`` when it cannot be honest."""
        rate = self.rate_bytes_per_second()
        if rate is None or remaining_materialized_bytes is None:
            return None
        eta = max(remaining_materialized_bytes, 0) / rate
        return eta if eta <= _MAX_ETA_SECONDS else None
