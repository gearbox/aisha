"""Tests for pure throughput estimation."""

from __future__ import annotations

import pytest

from ai_content_service.rate_estimator import ThroughputEstimator


def test_no_rate_before_warmup_seconds_and_bytes() -> None:
    estimator = ThroughputEstimator(alpha=0.5, warmup_seconds=5, warmup_bytes=100)
    estimator.observe(materialized_bytes=0, now_monotonic=0)
    estimator.observe(materialized_bytes=10_000, now_monotonic=4)

    assert estimator.rate_bytes_per_second() is None

    estimator.observe(materialized_bytes=10_000, now_monotonic=5)
    assert estimator.rate_bytes_per_second() is not None


def test_ewma_converges_to_steady_rate() -> None:
    estimator = ThroughputEstimator(alpha=0.5, warmup_seconds=0, warmup_bytes=0)
    estimator.observe(materialized_bytes=0, now_monotonic=0)
    estimator.observe(materialized_bytes=2_048, now_monotonic=1)
    estimator.observe(materialized_bytes=4_096, now_monotonic=2)

    assert estimator.rate_bytes_per_second() == pytest.approx(2_048)


def test_zero_delta_sample_decays_rate() -> None:
    estimator = ThroughputEstimator(alpha=0.5, warmup_seconds=0, warmup_bytes=0)
    estimator.observe(materialized_bytes=0, now_monotonic=0)
    estimator.observe(materialized_bytes=2_048, now_monotonic=1)
    initial = estimator.rate_bytes_per_second()
    estimator.observe(materialized_bytes=2_048, now_monotonic=2)

    assert initial is not None
    assert estimator.rate_bytes_per_second() == pytest.approx(initial / 2)


def test_eta_none_when_remaining_unknown() -> None:
    estimator = ThroughputEstimator(warmup_seconds=0, warmup_bytes=0)
    estimator.observe(materialized_bytes=0, now_monotonic=0)
    estimator.observe(materialized_bytes=2_048, now_monotonic=1)

    assert estimator.eta_seconds(remaining_materialized_bytes=None) is None


def test_eta_clamped_at_zero() -> None:
    estimator = ThroughputEstimator(warmup_seconds=0, warmup_bytes=0)
    estimator.observe(materialized_bytes=0, now_monotonic=0)
    estimator.observe(materialized_bytes=2_048, now_monotonic=1)

    assert estimator.eta_seconds(remaining_materialized_bytes=-1) == 0


def test_exact_zero_rate_never_divides_when_warmup_is_open() -> None:
    """A baseline followed by no materialized bytes must be telemetry-only."""
    estimator = ThroughputEstimator(warmup_seconds=1.0, warmup_bytes=100)
    estimator.observe(materialized_bytes=500, now_monotonic=0.0)
    estimator.observe(materialized_bytes=500, now_monotonic=2.0)

    assert estimator.rate_bytes_per_second() is None
    assert estimator.eta_seconds(remaining_materialized_bytes=1_000_000) is None


def test_stalled_transfer_has_no_misleading_eta() -> None:
    estimator = ThroughputEstimator(warmup_seconds=1.0, warmup_bytes=100)
    estimator.observe(materialized_bytes=0, now_monotonic=0.0)
    estimator.observe(materialized_bytes=10_000, now_monotonic=0.5)
    for sample in range(2, 200):
        estimator.observe(materialized_bytes=10_000, now_monotonic=0.5 + sample * 0.5)

    assert estimator.rate_bytes_per_second() is None
    assert estimator.eta_seconds(remaining_materialized_bytes=10_000_000_000) is None


def test_eta_above_one_day_is_not_honest() -> None:
    estimator = ThroughputEstimator(warmup_seconds=0, warmup_bytes=0)
    estimator.observe(materialized_bytes=0, now_monotonic=0)
    estimator.observe(materialized_bytes=1_024, now_monotonic=1)

    assert estimator.eta_seconds(remaining_materialized_bytes=1_024 * 86_401) is None


def test_rebase_does_not_invent_a_rate_sample() -> None:
    estimator = ThroughputEstimator(warmup_seconds=0, warmup_bytes=0)
    estimator.observe(materialized_bytes=0, now_monotonic=0)
    estimator.observe(materialized_bytes=2_048, now_monotonic=1)
    rate = estimator.rate_bytes_per_second()
    estimator.rebase(materialized_bytes=1_000_000, now_monotonic=2)

    assert estimator.rate_bytes_per_second() == rate
