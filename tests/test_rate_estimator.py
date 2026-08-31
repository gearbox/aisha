"""Tests for pure throughput estimation."""

from __future__ import annotations

import pytest

from ai_content_service.rate_estimator import ThroughputEstimator


def test_no_rate_before_warmup_seconds_and_bytes() -> None:
    estimator = ThroughputEstimator(alpha=0.5, warmup_seconds=5, warmup_bytes=100)
    estimator.observe(materialized_bytes=0, now_monotonic=0)
    estimator.observe(materialized_bytes=100, now_monotonic=4)

    assert estimator.rate_bytes_per_second() is None

    estimator.observe(materialized_bytes=100, now_monotonic=5)
    assert estimator.rate_bytes_per_second() is not None


def test_ewma_converges_to_steady_rate() -> None:
    estimator = ThroughputEstimator(alpha=0.5, warmup_seconds=0, warmup_bytes=0)
    estimator.observe(materialized_bytes=0, now_monotonic=0)
    estimator.observe(materialized_bytes=100, now_monotonic=1)
    estimator.observe(materialized_bytes=200, now_monotonic=2)

    assert estimator.rate_bytes_per_second() == pytest.approx(100)


def test_zero_delta_sample_decays_rate() -> None:
    estimator = ThroughputEstimator(alpha=0.5, warmup_seconds=0, warmup_bytes=0)
    estimator.observe(materialized_bytes=0, now_monotonic=0)
    estimator.observe(materialized_bytes=100, now_monotonic=1)
    initial = estimator.rate_bytes_per_second()
    estimator.observe(materialized_bytes=100, now_monotonic=2)

    assert initial is not None
    assert estimator.rate_bytes_per_second() == pytest.approx(initial / 2)


def test_eta_none_when_remaining_unknown() -> None:
    estimator = ThroughputEstimator(warmup_seconds=0, warmup_bytes=0)
    estimator.observe(materialized_bytes=0, now_monotonic=0)
    estimator.observe(materialized_bytes=100, now_monotonic=1)

    assert estimator.eta_seconds(remaining_materialized_bytes=None) is None


def test_eta_clamped_at_zero() -> None:
    estimator = ThroughputEstimator(warmup_seconds=0, warmup_bytes=0)
    estimator.observe(materialized_bytes=0, now_monotonic=0)
    estimator.observe(materialized_bytes=100, now_monotonic=1)

    assert estimator.eta_seconds(remaining_materialized_bytes=-1) == 0
