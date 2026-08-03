"""Tests for read-only model-cache verification."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

from ai_content_service import cache_service
from ai_content_service.config import ModelConfig, ModelFileConfig, Settings
from ai_content_service.r2_transfer import R2ObjectStat

if TYPE_CHECKING:
    from pathlib import Path


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    base = {
        "comfyui_path": tmp_path / "ComfyUI",
        "r2_s3_endpoint": "https://account.r2.cloudflarestorage.com",
        "r2_readonly_access_key_id": "READ",
        "r2_readonly_secret_access_key": "SECRET",
    } | overrides
    return Settings.model_validate(base)


def _target(
    tmp_path: Path,
    content: bytes,
    *,
    sha256: str | None = None,
    size_bytes: int | None = None,
) -> cache_service.PushTarget:
    digest = sha256 or hashlib.sha256(content).hexdigest()
    file = ModelFileConfig(
        name="model.safetensors",
        url="https://example.com/model.safetensors",
        filename="model.safetensors",
        sha256=digest,
        size_bytes=size_bytes,
    )
    model = ModelConfig(name="model", model_type="checkpoints", files=[file])
    return cache_service.PushTarget(
        model=model,
        file=file,
        disk_path=tmp_path / "ComfyUI" / "models" / "checkpoints" / file.filename,
    )


def test_verify_reports_missing_object(tmp_path: Path) -> None:
    target = _target(tmp_path, b"weights")
    with patch("ai_content_service.cache_service.stat", return_value=None):
        report = cache_service.verify_models(_settings(tmp_path), [target], MagicMock(), deep=False)

    assert report.ok is False
    assert report.results[0].status == "MISSING"


def test_verify_reports_size_mismatch(tmp_path: Path) -> None:
    target = _target(tmp_path, b"weights", size_bytes=99)
    with patch(
        "ai_content_service.cache_service.stat",
        return_value=R2ObjectStat(key="models/by-sha256/a", size_bytes=7),
    ):
        report = cache_service.verify_models(_settings(tmp_path), [target], MagicMock(), deep=False)

    assert report.ok is False
    assert report.results[0].status == "SIZE MISMATCH"


def test_deep_verify_reports_checksum_mismatch_and_removes_temp_file(tmp_path: Path) -> None:
    target = _target(tmp_path, b"expected")

    def _pull(*, dest_path: Path, **_kwargs: object) -> None:
        dest_path.write_bytes(b"other")

    with (
        patch(
            "ai_content_service.cache_service.stat",
            return_value=R2ObjectStat(key="models/by-sha256/a", size_bytes=5),
        ),
        patch("ai_content_service.cache_service.r2_pull", side_effect=_pull),
    ):
        report = cache_service.verify_models(_settings(tmp_path), [target], MagicMock(), deep=True)

    assert report.ok is False
    assert report.results[0].status == "CHECKSUM MISMATCH"
    assert not list((tmp_path / "ComfyUI" / "models").glob("*.cache-verify"))


def test_verify_fails_before_any_stat_when_read_credentials_are_absent(tmp_path: Path) -> None:
    target = _target(tmp_path, b"weights")
    with patch("ai_content_service.cache_service.stat") as stat:
        report = cache_service.verify_models(
            _settings(
                tmp_path,
                r2_readonly_access_key_id=None,
                r2_readonly_secret_access_key=None,
            ),
            [target],
            MagicMock(),
            deep=False,
        )

    assert report.ok is False
    assert report.configuration_error is not None
    assert "ACS_R2_READONLY_ACCESS_KEY_ID" in report.configuration_error
    stat.assert_not_called()
