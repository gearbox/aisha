"""Tests for the versioned provisioning timing telemetry contract."""

from __future__ import annotations

import json
import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ai_content_service.provisioning_timing import (
    PhaseStatus,
    PhaseTiming,
    ProvisioningTimer,
    build_env_context,
    detect_gpu_name,
    detect_instance_label,
    read_records,
)
from ai_content_service.telemetry_contract import ProvisioningPhase


class _Clock:
    def __init__(self) -> None:
        self.wall = 1_700_000_000.0
        self.mono = 100.0

    def time(self) -> float:
        return self.wall

    def monotonic(self) -> float:
        return self.mono

    def advance(self, seconds: float) -> None:
        self.wall += seconds
        self.mono += seconds


class TestProvisioningPhaseStability:
    def test_every_phase_id_is_a_distinct_string(self) -> None:
        values = [phase.value for phase in ProvisioningPhase]
        assert len(values) == len(set(values))


class TestProvisioningTimer:
    def test_completed_phase_has_chronology_and_explicit_status(self, tmp_path: Path) -> None:
        timer = ProvisioningTimer()
        with timer.start(ProvisioningPhase.WORKFLOW):
            pass
        timer.finish()
        timer.write(tmp_path / "timings.jsonl", outcome="ready")

        phase = read_records(tmp_path / "timings.jsonl")[0]["phases"][0]
        assert phase["phase"] == "workflow"
        assert phase["status"] == "completed"
        assert phase["started_at"].endswith("Z")
        assert phase["duration_s"] >= 0.0

    def test_snapshot_uses_seconds_keys_and_jsonl_keeps_s_keys(self, tmp_path: Path) -> None:
        timer = ProvisioningTimer()
        with timer.start(ProvisioningPhase.MODELS):
            pass
        timer.record_metric("models", {"materialized_bytes": 100})

        snapshot = timer.snapshot()
        timer.write(tmp_path / "timings.jsonl", outcome="ready")
        record = read_records(tmp_path / "timings.jsonl")[0]

        assert "total_seconds" in snapshot
        phases = snapshot["phases"]
        assert isinstance(phases, list)
        assert isinstance(phases[0], dict)
        assert "duration_seconds" in phases[0]
        assert "total_s" in record
        assert "duration_s" in record["phases"][0]

    def test_snapshot_excludes_env_and_unlisted_metrics(self) -> None:
        timer = ProvisioningTimer()
        timer.record_env({"instance": "private"})
        timer.record_metric("models", {"sources": {"skip": 1}})
        timer.record_metric("unlisted", {"secret": "not wire-safe"})

        snapshot = timer.snapshot()

        assert "env" not in snapshot
        metrics = snapshot["metrics"]
        assert isinstance(metrics, dict)
        assert set(metrics) == {"models"}

    def test_snapshot_sanitizes_every_allowlisted_metric_string_leaf(self) -> None:
        timer = ProvisioningTimer()
        secret = "metric-secret"
        timer.record_metric(
            "models",
            {
                "source": f"https://example.test/?token={secret}",
                "nested": [
                    secret,
                    {"message": f"Authorization: Bearer {secret}"},
                ],
            },
        )

        snapshot = timer.snapshot(secrets=(secret,))

        assert secret not in str(snapshot["metrics"])

    def test_snapshot_leaves_metric_dict_keys_unsanitized(self) -> None:
        """Keys come from bundle config, not process output -- see N4:
        sanitizing them risked a truncation collision silently dropping a
        metric, for no real redaction benefit."""
        timer = ProvisioningTimer()
        timer.record_metric("custom_node_requirements", {"some/custom-node": ["pkg==1.0"]})

        snapshot = timer.snapshot()

        metrics = snapshot["metrics"]
        assert isinstance(metrics, dict)
        assert "some/custom-node" in metrics["custom_node_requirements"]

    def test_snapshot_is_idempotent_and_finishes_timer(self) -> None:
        timer = ProvisioningTimer()

        first = timer.snapshot()
        second = timer.snapshot()

        assert first == second

    def test_jsonl_record_unchanged_for_equivalent_deployment(self, tmp_path: Path) -> None:
        """Schema-2 JSONL remains independent from the event-envelope snapshot."""
        clock = _Clock()
        with (
            patch("ai_content_service.provisioning_timing.time.time", clock.time),
            patch("ai_content_service.provisioning_timing.time.monotonic", clock.monotonic),
        ):
            timer = ProvisioningTimer()
            timer.record("bundle", "qwen")
            timer.record("bundle_version", "1")
            timer.record("mode", "full")
            with timer.start(ProvisioningPhase.MODELS):
                clock.advance(2.0)
            timer.record_metric("models", {"materialized_bytes": 2_048})
            timer.finish()
            timer.write(tmp_path / "timings.jsonl", outcome="ready")

        record = read_records(tmp_path / "timings.jsonl")[0]
        assert record["schema"] == 2
        assert record["total_s"] == 2.0
        assert record["phases"] == [
            {
                "phase": "models",
                "started_at": "2023-11-14T22:13:20Z",
                "duration_s": 2.0,
                "status": "completed",
            }
        ]

    def test_raising_phase_is_explicitly_failed_and_reraises(self, tmp_path: Path) -> None:
        timer = ProvisioningTimer()

        def fail_phase() -> None:
            with timer.start(ProvisioningPhase.MODELS):
                raise RuntimeError("download exploded")

        with pytest.raises(RuntimeError, match="download exploded"):
            fail_phase()
        timer.finish()
        timer.write(tmp_path / "timings.jsonl", outcome="failed", error="download exploded")

        record = read_records(tmp_path / "timings.jsonl")[0]
        assert record["phases"][0]["status"] == "failed"
        assert record["outcome"] == "failed"
        assert record["error"] == "download exploded"

    def test_skipped_and_completed_zero_duration_are_distinct(self, tmp_path: Path) -> None:
        timer = ProvisioningTimer()
        with timer.start(ProvisioningPhase.WORKFLOW):
            pass
        timer.mark_skipped(ProvisioningPhase.CUSTOM_NODES)
        timer.finish()
        timer.write(tmp_path / "timings.jsonl", outcome="ready")

        phases = {p["phase"]: p for p in read_records(tmp_path / "timings.jsonl")[0]["phases"]}
        assert phases["workflow"]["status"] == "completed"
        assert phases["custom_nodes"]["status"] == "skipped"

    def test_finish_freezes_duration_before_later_write(self, tmp_path: Path) -> None:
        clock = _Clock()
        with (
            patch("ai_content_service.provisioning_timing.time.time", clock.time),
            patch("ai_content_service.provisioning_timing.time.monotonic", clock.monotonic),
        ):
            timer = ProvisioningTimer()
            clock.advance(2.0)
            timer.finish()
            clock.advance(99.0)
            timer.write(tmp_path / "timings.jsonl", outcome="ready")

        record = read_records(tmp_path / "timings.jsonl")[0]
        assert record["total_s"] == 2.0
        assert record["finished_at"] == "2023-11-14T22:13:22Z"

    def test_phase_duration_uses_monotonic_time(self, tmp_path: Path) -> None:
        clock = _Clock()
        with (
            patch("ai_content_service.provisioning_timing.time.time", clock.time),
            patch("ai_content_service.provisioning_timing.time.monotonic", clock.monotonic),
        ):
            timer = ProvisioningTimer()
            with timer.start(ProvisioningPhase.COMFYUI):
                clock.mono += 3.0
                clock.wall -= 100.0
            timer.finish()
            timer.write(tmp_path / "timings.jsonl", outcome="ready")

        record = read_records(tmp_path / "timings.jsonl")[0]
        assert record["phases"][0]["duration_s"] == 3.0
        assert record["total_s"] == 3.0

    def test_context_cannot_overwrite_core_fields(self, tmp_path: Path) -> None:
        timer = ProvisioningTimer()
        timer.record("bundle", "qwen")
        timer.record("schema", 999)
        timer.record("total_s", "not a duration")
        timer.record_metric("models", {"materialized_bytes": 10})
        timer.finish()
        timer.write(tmp_path / "timings.jsonl", outcome="ready")

        record = read_records(tmp_path / "timings.jsonl")[0]
        assert record["schema"] == 2
        assert isinstance(record["total_s"], float)
        assert record["total_s"] != "not a duration"
        assert record["metrics"]["schema"] == 999
        assert record["metrics"]["total_s"] == "not a duration"
        assert record["metrics"]["models"] == {"materialized_bytes": 10}

    @pytest.mark.parametrize(
        ("value", "expected_str"),
        [
            (Path("/x"), str(Path("/x"))),
            ({"a", "b"}, None),
        ],
    )
    def test_serialize_degraded_metric_falls_back_to_str_rather_than_discarding(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
        value: object,
        expected_str: str | None,
    ) -> None:
        """R5: a `Path`/`Enum`/`set`/`Decimal` metric must not silently
        discard the whole deployment's timing record -- it degrades to
        `str()` and a warning names the offending key."""
        path = tmp_path / "timings.jsonl"
        timer = ProvisioningTimer()
        with timer.start(ProvisioningPhase.WORKFLOW):
            pass
        timer.record_metric("p", value)
        timer.finish()

        with caplog.at_level(logging.WARNING, logger="ai_content_service.provisioning_timing"):
            timer.write(path, outcome="ready")

        record = read_records(path)[0]
        assert record["metrics"]["p"] == (expected_str if expected_str is not None else str(value))
        assert isinstance(record["total_s"], float)
        assert len(record["phases"]) == 1
        events = [
            r.msg
            for r in caplog.records
            if isinstance(r.msg, dict)
            and r.msg.get("event") == "provisioning_timing.serialize_degraded"
        ]
        assert len(events) == 1
        assert events[0]["key"] == "p"

    def test_error_is_redacted_and_bounded(self, tmp_path: Path) -> None:
        token = "super-secret-token-value"
        timer = ProvisioningTimer()
        timer.finish()
        timer.write(
            tmp_path / "timings.jsonl",
            outcome="failed",
            error=(
                f"request https://example.test/x?token={token} Authorization: Bearer {token} "
                + "x" * 5_000
            ),
            secrets=(token,),
        )

        error = read_records(tmp_path / "timings.jsonl")[0]["error"]
        assert token not in error
        assert "token=***" in error
        assert "Authorization: ***" in error
        assert error.endswith("… [truncated]")

    def test_local_concurrent_writes_are_each_complete_json_line(self, tmp_path: Path) -> None:
        path = tmp_path / "timings.jsonl"

        def write_one(index: int) -> None:
            timer = ProvisioningTimer()
            timer.record("bundle", f"bundle-{index}")
            timer.finish()
            timer.write(path, outcome="ready")

        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(write_one, range(64)))

        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 64
        assert all(isinstance(json.loads(line), dict) for line in lines)


class TestReadRecords:
    def test_schema_1_records_remain_readable(self, tmp_path: Path) -> None:
        path = tmp_path / "timings.jsonl"
        path.write_text('{"schema": 1, "ts": "2026-08-07T14:03:11Z", "outcome": "ready"}\n')

        assert read_records(path)[0]["schema"] == 1

    def test_streams_filter_and_bounded_tail_without_read_text(self, tmp_path: Path) -> None:
        path = tmp_path / "timings.jsonl"
        path.write_text(
            "".join(
                json.dumps({"bundle": "wanted" if i % 2 else "other", "n": i}) + "\n"
                for i in range(500)
            )
            + "not json\n"
        )

        with patch.object(Path, "read_text", side_effect=AssertionError("must stream")):
            records = read_records(path, bundle="wanted", last=3)

        assert [record["n"] for record in records] == [495, 497, 499]

    def test_rejects_non_positive_tail(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="positive"):
            read_records(tmp_path / "timings.jsonl", last=0)


class TestDetectGpuName:
    def test_returns_none_when_nvidia_smi_missing(self) -> None:
        with patch("ai_content_service.provisioning_timing.shutil.which", return_value=None):
            assert detect_gpu_name() is None

    def test_returns_first_line_on_success(self) -> None:
        result = MagicMock(returncode=0, stdout="NVIDIA GeForce RTX 4090\n")
        with (
            patch(
                "ai_content_service.provisioning_timing.shutil.which",
                return_value="/usr/bin/nvidia-smi",
            ),
            patch("ai_content_service.provisioning_timing.subprocess.run", return_value=result),
        ):
            assert detect_gpu_name() == "NVIDIA GeForce RTX 4090"

    def test_returns_none_on_timeout(self) -> None:
        with (
            patch("ai_content_service.provisioning_timing.shutil.which", return_value="nvidia-smi"),
            patch(
                "ai_content_service.provisioning_timing.subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=3.0),
            ),
        ):
            assert detect_gpu_name() is None


class TestEnvironmentContext:
    def test_advisory_and_runtime_base_images_are_separate(self) -> None:
        from ai_content_service.config import Settings

        context = build_env_context(
            Settings(),
            bundle_base_image="vastai/comfy:v0.30.0-cuda-13.2-py312",
            comfyui_source="bundle_checkout",
        )

        assert context["bundle_base_image"] == "vastai/comfy:v0.30.0-cuda-13.2-py312"
        assert context["runtime_base_image"] is None
        assert context["comfyui_source"] == "bundle_checkout"

    def test_instance_label_is_optional(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("VAST_CONTAINERLABEL", raising=False)
        assert detect_instance_label() is None


class TestPhaseTimingDataclass:
    def test_is_frozen(self) -> None:
        timing = PhaseTiming(phase=ProvisioningPhase.MODELS, started_at=0.0, duration_s=1.0)
        with pytest.raises(AttributeError):
            timing.duration_s = 2.0  # type: ignore[misc]
        assert timing.status is PhaseStatus.COMPLETED

    def test_skipped_is_derived_from_status(self) -> None:
        """R6: `skipped` is a property over `status`, not a second stored
        field -- there is exactly one way to represent "this phase didn't
        run", so a SKIPPED/FAILED timing can never disagree with itself."""
        skipped = PhaseTiming(
            phase=ProvisioningPhase.CUSTOM_NODES,
            started_at=0.0,
            duration_s=0.0,
            status=PhaseStatus.SKIPPED,
        )
        failed = PhaseTiming(
            phase=ProvisioningPhase.MODELS,
            started_at=0.0,
            duration_s=1.0,
            status=PhaseStatus.FAILED,
        )
        assert skipped.skipped is True
        assert failed.skipped is False
