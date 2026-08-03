"""Tests for read-only model-cache verification."""

from __future__ import annotations

import contextlib
import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ai_content_service import cache_service
from ai_content_service.config import ModelConfig, ModelFileConfig, Settings
from ai_content_service.r2_transfer import R2ObjectStat


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
        report = cache_service.verify_models(_settings(tmp_path), [target], deep=False)

    assert report.ok is False
    assert report.results[0].status == "MISSING"


def test_verify_reports_present_for_a_shallow_match(tmp_path: Path) -> None:
    target = _target(tmp_path, b"weights", size_bytes=7)
    with patch(
        "ai_content_service.cache_service.stat",
        return_value=R2ObjectStat(key="models/by-sha256/a", size_bytes=7),
    ):
        report = cache_service.verify_models(_settings(tmp_path), [target], deep=False)

    assert report.ok is True
    assert report.results[0].status == "PRESENT"


def test_verify_reports_no_sha256_without_stat(tmp_path: Path) -> None:
    target = _target(tmp_path, b"weights", sha256="a" * 64)
    target.file.sha256 = None
    with patch("ai_content_service.cache_service.stat") as stat:
        report = cache_service.verify_models(_settings(tmp_path), [target], deep=False)

    assert report.results[0].status == "NO SHA256"
    stat.assert_not_called()


def test_verify_reports_size_mismatch(tmp_path: Path) -> None:
    target = _target(tmp_path, b"weights", size_bytes=99)
    with patch(
        "ai_content_service.cache_service.stat",
        return_value=R2ObjectStat(key="models/by-sha256/a", size_bytes=7),
    ):
        report = cache_service.verify_models(_settings(tmp_path), [target], deep=False)

    assert report.ok is False
    assert report.results[0].status == "SIZE MISMATCH"


def test_verify_reports_stat_error_and_uses_read_credentials_only(tmp_path: Path) -> None:
    target = _target(tmp_path, b"weights")
    settings = _settings(
        tmp_path,
        r2_write_access_key_id="WRITE",
        r2_write_secret_access_key="write-secret",
    )
    with patch(
        "ai_content_service.cache_service.stat", side_effect=cache_service.R2TransferError("denied")
    ) as stat:
        report = cache_service.verify_models(settings, [target], deep=False)

    assert report.results[0].status == "STAT ERROR"
    assert stat.call_args.kwargs["creds"].access_key_id == "READ"


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
        report = cache_service.verify_models(_settings(tmp_path), [target], deep=True)

    assert report.ok is False
    assert report.results[0].status == "CHECKSUM MISMATCH"
    assert not list((tmp_path / "ComfyUI" / "models").glob("*.cache-verify"))


def test_deep_verify_succeeds_and_never_requires_the_local_model_file(tmp_path: Path) -> None:
    content = b"expected"
    target = _target(tmp_path, content)

    def _pull(*, dest_path: Path, **_kwargs: object) -> None:
        dest_path.write_bytes(content)

    with (
        patch(
            "ai_content_service.cache_service.stat",
            return_value=R2ObjectStat(key="models/by-sha256/a", size_bytes=len(content)),
        ),
        patch("ai_content_service.cache_service.r2_pull", side_effect=_pull) as r2_pull,
    ):
        report = cache_service.verify_models(_settings(tmp_path), [target], deep=True)

    assert report.ok is True
    assert report.results[0].status == "CHECKSUM OK"
    assert not target.disk_path.exists()
    assert r2_pull.call_args.kwargs["progress"] is True


def test_deep_verify_reports_insufficient_space_before_pull(tmp_path: Path) -> None:
    target = _target(tmp_path, b"weights")
    disk_usage = MagicMock(free=1)
    with (
        patch(
            "ai_content_service.cache_service.stat",
            return_value=R2ObjectStat(key="models/by-sha256/a", size_bytes=5),
        ),
        patch("ai_content_service.cache_service.shutil.disk_usage", return_value=disk_usage),
        patch("ai_content_service.cache_service.r2_pull") as pull,
    ):
        report = cache_service.verify_models(_settings(tmp_path), [target], deep=True)

    assert report.results[0].status == "INSUFFICIENT SPACE"
    pull.assert_not_called()


@pytest.mark.parametrize("failure", ["mkdir", "pull", "hash"])
def test_deep_verify_reports_operational_failures_and_cleans_temporary_file(
    tmp_path: Path, failure: str
) -> None:
    target = _target(tmp_path, b"weights")
    stat_result = R2ObjectStat(key="models/by-sha256/a", size_bytes=5)
    if failure == "mkdir":
        patches = [
            patch.object(Path, "mkdir", side_effect=PermissionError("denied")),
        ]
    elif failure == "pull":
        patches = [
            patch(
                "ai_content_service.cache_service.r2_pull",
                side_effect=cache_service.R2TransferError("bad pull"),
            ),
        ]
    else:
        patches = [
            patch(
                "ai_content_service.cache_service.compute_file_sha256",
                side_effect=PermissionError("denied"),
            ),
        ]

    def _pull_for_hash(*, dest_path: Path, **_kwargs: object) -> None:
        dest_path.write_bytes(b"weights")

    transfer_patch = (
        patch("ai_content_service.cache_service.r2_pull", side_effect=_pull_for_hash)
        if failure == "hash"
        else contextlib.nullcontext()
    )

    with (
        patch("ai_content_service.cache_service.stat", return_value=stat_result),
        patches[0],
        transfer_patch,
    ):
        report = cache_service.verify_models(_settings(tmp_path), [target], deep=True)

    assert report.results[0].status == "DEEP VERIFY ERROR"
    models_dir = tmp_path / "ComfyUI" / "models"
    if models_dir.exists():
        assert not list(models_dir.glob("*.cache-verify"))


def test_verify_continues_after_a_failed_target(tmp_path: Path) -> None:
    first = _target(tmp_path, b"one")
    second = _target(tmp_path, b"two")
    second.file.filename = "second.safetensors"
    with patch(
        "ai_content_service.cache_service.stat",
        side_effect=[None, R2ObjectStat(key="models/by-sha256/b", size_bytes=3)],
    ):
        report = cache_service.verify_models(_settings(tmp_path), [first, second], deep=False)

    assert [result.status for result in report.results] == ["MISSING", "PRESENT"]


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
            deep=False,
        )

    assert report.ok is False
    assert report.configuration_error is not None
    assert "ACS_R2_READONLY_ACCESS_KEY_ID" in report.configuration_error
    stat.assert_not_called()
