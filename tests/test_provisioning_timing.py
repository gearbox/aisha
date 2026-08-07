"""Tests for provisioning_timing (Phase 2b-lite, Part B)."""

from __future__ import annotations

import json
import subprocess
import time
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from ai_content_service.provisioning_timing import (
    PhaseId,
    PhaseTiming,
    ProvisioningTimer,
    build_env_context,
    detect_gpu_name,
    detect_instance_label,
    read_records,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestPhaseIdStability:
    def test_every_phase_id_is_a_distinct_string(self) -> None:
        values = [phase.value for phase in PhaseId]
        assert len(values) == len(set(values))

    def test_expected_phase_ids(self) -> None:
        assert {phase.value for phase in PhaseId} == {
            "comfyui",
            "requirements_base",
            "requirements_locked",
            "custom_nodes",
            "models",
            "workflow",
            "verifying",
        }


class TestProvisioningTimerStart:
    def test_successful_phase_is_recorded_not_skipped(self, tmp_path: Path) -> None:
        timer = ProvisioningTimer()
        with timer.start(PhaseId.WORKFLOW):
            pass
        timer.write(tmp_path / "timings.jsonl", outcome="ready")

        record = read_records(tmp_path / "timings.jsonl")[0]
        phases = {p["phase"]: p for p in record["phases"]}
        assert phases["workflow"]["skipped"] is False
        assert phases["workflow"]["duration_s"] >= 0.0

    def test_raising_phase_still_records_duration(self, tmp_path: Path) -> None:
        """A phase that raises must still contribute a duration -- a failure's
        timing is often the most interesting one."""
        timer = ProvisioningTimer()
        with pytest.raises(RuntimeError), timer.start(PhaseId.MODELS):
            time.sleep(0.15)
            raise RuntimeError("download exploded")
        timer.write(tmp_path / "timings.jsonl", outcome="failed", error="download exploded")

        record = read_records(tmp_path / "timings.jsonl")[0]
        phases = {p["phase"]: p for p in record["phases"]}
        assert phases["models"]["skipped"] is False
        assert phases["models"]["duration_s"] > 0.0
        assert record["outcome"] == "failed"
        assert record["error"] == "download exploded"

    def test_monotonic_used_for_duration_survives_wall_clock_step(self, tmp_path: Path) -> None:
        """A node whose wall clock steps backward during a phase must not
        produce a negative duration -- only `time.monotonic()` may back it."""
        stepping_time = iter([1_700_000_000.0, 1_000_000.0, 1_000_000.0])

        with patch("ai_content_service.provisioning_timing.time.time", lambda: next(stepping_time)):
            timer = ProvisioningTimer()
            with timer.start(PhaseId.COMFYUI):
                pass
            timer.write(tmp_path / "timings.jsonl", outcome="ready")

        record = read_records(tmp_path / "timings.jsonl")[0]
        phases = {p["phase"]: p for p in record["phases"]}
        assert phases["comfyui"]["duration_s"] >= 0.0
        assert record["total_s"] >= 0.0


class TestProvisioningTimerSkipped:
    def test_mark_skipped_is_true_not_zero_duration_masquerade(self, tmp_path: Path) -> None:
        timer = ProvisioningTimer()
        timer.mark_skipped(PhaseId.CUSTOM_NODES)
        timer.write(tmp_path / "timings.jsonl", outcome="ready")

        record = read_records(tmp_path / "timings.jsonl")[0]
        phases = {p["phase"]: p for p in record["phases"]}
        assert phases["custom_nodes"] == {
            "phase": "custom_nodes",
            "duration_s": 0.0,
            "skipped": True,
        }

    def test_skipped_and_real_zero_duration_are_distinguishable_by_flag(
        self, tmp_path: Path
    ) -> None:
        timer = ProvisioningTimer()
        with timer.start(PhaseId.WORKFLOW):
            pass  # a genuinely instantaneous phase
        timer.mark_skipped(PhaseId.CUSTOM_NODES)
        timer.write(tmp_path / "timings.jsonl", outcome="ready")

        phases = {p["phase"]: p for p in read_records(tmp_path / "timings.jsonl")[0]["phases"]}
        assert phases["workflow"]["skipped"] is False
        assert phases["custom_nodes"]["skipped"] is True


class TestProvisioningTimerRecord:
    def test_record_attaches_free_form_context(self, tmp_path: Path) -> None:
        timer = ProvisioningTimer()
        timer.record("bundle", "qwen_rapid_aio")
        timer.record("bundle_version", "260805-01")
        timer.record("mode", "full")
        timer.record("models", {"sources": {"hf_xet": 3}, "bytes_total": 100, "mbps": 12.3})
        timer.write(tmp_path / "timings.jsonl", outcome="ready")

        record = read_records(tmp_path / "timings.jsonl")[0]
        assert record["bundle"] == "qwen_rapid_aio"
        assert record["bundle_version"] == "260805-01"
        assert record["mode"] == "full"
        assert record["models"] == {"sources": {"hf_xet": 3}, "bytes_total": 100, "mbps": 12.3}


class TestProvisioningTimerWrite:
    def test_record_is_one_newline_terminated_json_line(self, tmp_path: Path) -> None:
        path = tmp_path / "timings.jsonl"
        timer = ProvisioningTimer()
        timer.write(path, outcome="ready")

        content = path.read_text(encoding="utf-8")
        lines = content.splitlines(keepends=True)
        assert len(lines) == 1
        assert content.endswith("\n")
        parsed = json.loads(lines[0])
        assert parsed["schema"] == 1
        assert parsed["outcome"] == "ready"

    def test_appends_rather_than_overwrites(self, tmp_path: Path) -> None:
        path = tmp_path / "timings.jsonl"
        ProvisioningTimer().write(path, outcome="ready")
        ProvisioningTimer().write(path, outcome="failed", error="boom")

        records = read_records(path)
        assert len(records) == 2
        assert records[0]["outcome"] == "ready"
        assert records[1]["outcome"] == "failed"

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "dir" / "timings.jsonl"
        ProvisioningTimer().write(path, outcome="ready")
        assert path.exists()

    def test_unwritable_path_logs_warning_and_does_not_raise(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        path = tmp_path / "timings.jsonl"
        timer = ProvisioningTimer()

        with (
            patch("pathlib.Path.mkdir", side_effect=OSError("read-only file system")),
            caplog.at_level(logging.WARNING, logger="ai_content_service.provisioning_timing"),
        ):
            timer.write(path, outcome="ready")  # must not raise

        assert not path.exists()
        assert any(
            isinstance(record.msg, dict)
            and record.msg.get("event") == "provisioning_timing.write_failed"
            for record in caplog.records
        )

    def test_no_error_key_when_error_is_none(self, tmp_path: Path) -> None:
        path = tmp_path / "timings.jsonl"
        ProvisioningTimer().write(path, outcome="ready", error=None)

        record = read_records(path)[0]
        assert "error" not in record

    def test_total_s_reflects_elapsed_time(self, tmp_path: Path) -> None:
        path = tmp_path / "timings.jsonl"
        timer = ProvisioningTimer()
        time.sleep(0.15)
        timer.write(path, outcome="ready")

        record = read_records(path)[0]
        assert record["total_s"] > 0.0


class TestReadRecords:
    def test_missing_file_returns_empty_list(self, tmp_path: Path) -> None:
        assert read_records(tmp_path / "does-not-exist.jsonl") == []

    def test_blank_lines_are_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "timings.jsonl"
        path.write_text('{"a": 1}\n\n{"a": 2}\n')
        assert read_records(path) == [{"a": 1}, {"a": 2}]

    def test_malformed_line_is_skipped_not_raised(self, tmp_path: Path) -> None:
        path = tmp_path / "timings.jsonl"
        path.write_text('{"a": 1}\nnot json\n{"a": 2}\n')
        assert read_records(path) == [{"a": 1}, {"a": 2}]

    def test_non_object_json_line_is_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "timings.jsonl"
        path.write_text('[1, 2, 3]\n{"a": 1}\n')
        assert read_records(path) == [{"a": 1}]


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

    def test_returns_none_on_nonzero_exit(self) -> None:
        result = MagicMock(returncode=1, stdout="")
        with (
            patch(
                "ai_content_service.provisioning_timing.shutil.which",
                return_value="/usr/bin/nvidia-smi",
            ),
            patch("ai_content_service.provisioning_timing.subprocess.run", return_value=result),
        ):
            assert detect_gpu_name() is None

    def test_returns_none_on_timeout(self) -> None:
        with (
            patch(
                "ai_content_service.provisioning_timing.shutil.which",
                return_value="/usr/bin/nvidia-smi",
            ),
            patch(
                "ai_content_service.provisioning_timing.subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=3.0),
            ),
        ):
            assert detect_gpu_name() is None


class TestDetectInstanceLabel:
    def test_reads_vast_containerlabel(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VAST_CONTAINERLABEL", "C.46979259")
        assert detect_instance_label() == "C.46979259"

    def test_returns_none_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("VAST_CONTAINERLABEL", raising=False)
        assert detect_instance_label() is None


class TestBuildEnvContext:
    def test_unknown_provenance_is_null_not_a_guess(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from ai_content_service.config import Settings

        monkeypatch.delenv("VAST_CONTAINERLABEL", raising=False)
        with patch("ai_content_service.provisioning_timing.detect_gpu_name", return_value=None):
            context = build_env_context(Settings(), base_image=None, comfyui_source="image")

        assert context["base_image"] is None
        assert context["instance"] is None
        assert context["gpu"] is None
        assert context["comfyui_source"] == "image"
        assert isinstance(context["cpu_count"], int) or context["cpu_count"] is None

    def test_base_image_passthrough(self) -> None:
        from ai_content_service.config import Settings

        context = build_env_context(
            Settings(),
            base_image="vastai/comfy:v0.30.0-cuda-13.2-py312",
            comfyui_source="bundle",
        )
        assert context["base_image"] == "vastai/comfy:v0.30.0-cuda-13.2-py312"
        assert context["comfyui_source"] == "bundle"
        assert context["hf_xet_enabled"] == Settings().hf_xet_enabled


class TestPhaseTimingDataclass:
    def test_is_frozen(self) -> None:
        timing = PhaseTiming(phase=PhaseId.MODELS, started_at=0.0, duration_s=1.0, skipped=False)
        with pytest.raises(AttributeError):
            timing.duration_s = 2.0  # type: ignore[misc]
