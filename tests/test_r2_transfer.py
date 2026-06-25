"""Tests for r2_transfer — rclone wrapper for R2 model cache."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from ai_content_service.r2_transfer import (
    CachePullError,
    R2ReadCreds,
    R2WriteCreds,
    pull,
    push,
)

if TYPE_CHECKING:
    from pathlib import Path

_READ_CREDS = R2ReadCreds(
    access_key_id="READKEYID",
    secret_access_key="readsecret",
)
_WRITE_CREDS = R2WriteCreds(
    access_key_id="WRITEKEYID",
    secret_access_key="writesecret",
    session_token="sessiontok",
)
_BUCKET = "apex-model-cache"
_ENDPOINT = "https://account123.r2.cloudflarestorage.com"
_KEY = "models/by-sha256/abc123"


def _mock_rclone(returncode: int = 0) -> MagicMock:
    result = MagicMock(spec=subprocess.CompletedProcess)
    result.returncode = returncode
    result.stderr = b""
    return result


# ---------------------------------------------------------------------------
# _require_rclone
# ---------------------------------------------------------------------------


class TestRequireRclone:
    def test_raises_when_rclone_missing(self, tmp_path: Path) -> None:
        with (
            patch("ai_content_service.r2_transfer.shutil.which", return_value=None),
            pytest.raises(RuntimeError, match="rclone not found"),
        ):
            pull(
                key=_KEY,
                dest_path=tmp_path / "model.safetensors",
                creds=_READ_CREDS,
                bucket=_BUCKET,
                endpoint=_ENDPOINT,
                rclone_path="rclone",
            )

    def test_raises_for_custom_path_when_missing(self, tmp_path: Path) -> None:
        with (
            patch("ai_content_service.r2_transfer.shutil.which", return_value=None),
            pytest.raises(RuntimeError, match="/opt/bin/rclone"),
        ):
            pull(
                key=_KEY,
                dest_path=tmp_path / "model.safetensors",
                creds=_READ_CREDS,
                bucket=_BUCKET,
                endpoint=_ENDPOINT,
                rclone_path="/opt/bin/rclone",
            )


# ---------------------------------------------------------------------------
# pull
# ---------------------------------------------------------------------------


class TestPull:
    def test_uses_copyto_subcommand(self, tmp_path: Path) -> None:
        dest = tmp_path / "model.safetensors"
        with (
            patch("ai_content_service.r2_transfer.shutil.which", return_value="/usr/bin/rclone"),
            patch(
                "ai_content_service.r2_transfer.subprocess.run", return_value=_mock_rclone()
            ) as mock_run,
        ):
            pull(key=_KEY, dest_path=dest, creds=_READ_CREDS, bucket=_BUCKET, endpoint=_ENDPOINT)

        cmd = mock_run.call_args[0][0]
        assert cmd[1] == "copyto"

    def test_source_is_s3_remote_with_bucket_and_key(self, tmp_path: Path) -> None:
        dest = tmp_path / "model.safetensors"
        with (
            patch("ai_content_service.r2_transfer.shutil.which", return_value="/usr/bin/rclone"),
            patch(
                "ai_content_service.r2_transfer.subprocess.run", return_value=_mock_rclone()
            ) as mock_run,
        ):
            pull(key=_KEY, dest_path=dest, creds=_READ_CREDS, bucket=_BUCKET, endpoint=_ENDPOINT)

        cmd = mock_run.call_args[0][0]
        src = cmd[2]
        assert src == f":s3:{_BUCKET}/{_KEY}"

    def test_dest_is_exact_path(self, tmp_path: Path) -> None:
        dest = tmp_path / "subdir" / "model.safetensors"
        with (
            patch("ai_content_service.r2_transfer.shutil.which", return_value="/usr/bin/rclone"),
            patch(
                "ai_content_service.r2_transfer.subprocess.run", return_value=_mock_rclone()
            ) as mock_run,
        ):
            pull(key=_KEY, dest_path=dest, creds=_READ_CREDS, bucket=_BUCKET, endpoint=_ENDPOINT)

        cmd = mock_run.call_args[0][0]
        assert cmd[3] == str(dest)

    def test_includes_cloudflare_provider_flag(self, tmp_path: Path) -> None:
        dest = tmp_path / "model.safetensors"
        with (
            patch("ai_content_service.r2_transfer.shutil.which", return_value="/usr/bin/rclone"),
            patch(
                "ai_content_service.r2_transfer.subprocess.run", return_value=_mock_rclone()
            ) as mock_run,
        ):
            pull(key=_KEY, dest_path=dest, creds=_READ_CREDS, bucket=_BUCKET, endpoint=_ENDPOINT)

        cmd = mock_run.call_args[0][0]
        assert "--s3-provider" in cmd
        assert "Cloudflare" in cmd

    def test_includes_endpoint_flag(self, tmp_path: Path) -> None:
        dest = tmp_path / "model.safetensors"
        with (
            patch("ai_content_service.r2_transfer.shutil.which", return_value="/usr/bin/rclone"),
            patch(
                "ai_content_service.r2_transfer.subprocess.run", return_value=_mock_rclone()
            ) as mock_run,
        ):
            pull(key=_KEY, dest_path=dest, creds=_READ_CREDS, bucket=_BUCKET, endpoint=_ENDPOINT)

        cmd = mock_run.call_args[0][0]
        assert any(_ENDPOINT in arg for arg in cmd)

    def test_includes_multi_thread_streams(self, tmp_path: Path) -> None:
        dest = tmp_path / "model.safetensors"
        with (
            patch("ai_content_service.r2_transfer.shutil.which", return_value="/usr/bin/rclone"),
            patch(
                "ai_content_service.r2_transfer.subprocess.run", return_value=_mock_rclone()
            ) as mock_run,
        ):
            pull(
                key=_KEY,
                dest_path=dest,
                creds=_READ_CREDS,
                bucket=_BUCKET,
                endpoint=_ENDPOINT,
                multi_thread_streams=8,
            )

        cmd = mock_run.call_args[0][0]
        assert any("multi-thread-streams=8" in arg for arg in cmd)

    def test_passes_credentials_via_env(self, tmp_path: Path) -> None:
        dest = tmp_path / "model.safetensors"
        with (
            patch("ai_content_service.r2_transfer.shutil.which", return_value="/usr/bin/rclone"),
            patch(
                "ai_content_service.r2_transfer.subprocess.run", return_value=_mock_rclone()
            ) as mock_run,
        ):
            pull(key=_KEY, dest_path=dest, creds=_READ_CREDS, bucket=_BUCKET, endpoint=_ENDPOINT)

        env = mock_run.call_args[1]["env"]
        assert env["RCLONE_S3_ACCESS_KEY_ID"] == _READ_CREDS.access_key_id
        assert env["RCLONE_S3_SECRET_ACCESS_KEY"] == _READ_CREDS.secret_access_key
        assert "RCLONE_S3_SESSION_TOKEN" not in env
        assert "AWS_ACCESS_KEY_ID" not in env
        assert "AWS_SECRET_ACCESS_KEY" not in env

    def test_nonzero_exit_raises_cache_pull_error(self, tmp_path: Path) -> None:
        dest = tmp_path / "model.safetensors"
        with (
            patch("ai_content_service.r2_transfer.shutil.which", return_value="/usr/bin/rclone"),
            patch(
                "ai_content_service.r2_transfer.subprocess.run",
                return_value=_mock_rclone(returncode=1),
            ),
            pytest.raises(CachePullError),
        ):
            pull(key=_KEY, dest_path=dest, creds=_READ_CREDS, bucket=_BUCKET, endpoint=_ENDPOINT)

    def test_no_session_token_in_env_for_read_path(self, tmp_path: Path) -> None:
        dest = tmp_path / "model.safetensors"
        with (
            patch("ai_content_service.r2_transfer.shutil.which", return_value="/usr/bin/rclone"),
            patch(
                "ai_content_service.r2_transfer.subprocess.run", return_value=_mock_rclone()
            ) as mock_run,
        ):
            pull(key=_KEY, dest_path=dest, creds=_READ_CREDS, bucket=_BUCKET, endpoint=_ENDPOINT)

        env = mock_run.call_args[1]["env"]
        assert "RCLONE_S3_SESSION_TOKEN" not in env
        assert "AWS_SESSION_TOKEN" not in env


# ---------------------------------------------------------------------------
# push
# ---------------------------------------------------------------------------


class TestPush:
    def test_uses_copyto_subcommand(self, tmp_path: Path) -> None:
        src = tmp_path / "model.safetensors"
        src.touch()
        with (
            patch("ai_content_service.r2_transfer.shutil.which", return_value="/usr/bin/rclone"),
            patch(
                "ai_content_service.r2_transfer.subprocess.run", return_value=_mock_rclone()
            ) as mock_run,
        ):
            push(src_path=src, key=_KEY, creds=_WRITE_CREDS, bucket=_BUCKET, endpoint=_ENDPOINT)

        cmd = mock_run.call_args[0][0]
        assert cmd[1] == "copyto"

    def test_src_is_file_path_and_dest_is_r2_remote(self, tmp_path: Path) -> None:
        src = tmp_path / "model.safetensors"
        src.touch()
        with (
            patch("ai_content_service.r2_transfer.shutil.which", return_value="/usr/bin/rclone"),
            patch(
                "ai_content_service.r2_transfer.subprocess.run", return_value=_mock_rclone()
            ) as mock_run,
        ):
            push(src_path=src, key=_KEY, creds=_WRITE_CREDS, bucket=_BUCKET, endpoint=_ENDPOINT)

        cmd = mock_run.call_args[0][0]
        assert cmd[2] == str(src)
        assert cmd[3] == f":s3:{_BUCKET}/{_KEY}"

    def test_includes_cloudflare_provider_flag(self, tmp_path: Path) -> None:
        src = tmp_path / "model.safetensors"
        src.touch()
        with (
            patch("ai_content_service.r2_transfer.shutil.which", return_value="/usr/bin/rclone"),
            patch(
                "ai_content_service.r2_transfer.subprocess.run", return_value=_mock_rclone()
            ) as mock_run,
        ):
            push(src_path=src, key=_KEY, creds=_WRITE_CREDS, bucket=_BUCKET, endpoint=_ENDPOINT)

        cmd = mock_run.call_args[0][0]
        assert "--s3-provider" in cmd
        assert "Cloudflare" in cmd

    def test_includes_upload_concurrency_and_chunk_size(self, tmp_path: Path) -> None:
        src = tmp_path / "model.safetensors"
        src.touch()
        with (
            patch("ai_content_service.r2_transfer.shutil.which", return_value="/usr/bin/rclone"),
            patch(
                "ai_content_service.r2_transfer.subprocess.run", return_value=_mock_rclone()
            ) as mock_run,
        ):
            push(
                src_path=src,
                key=_KEY,
                creds=_WRITE_CREDS,
                bucket=_BUCKET,
                endpoint=_ENDPOINT,
                upload_concurrency=16,
                chunk_size_mb=256,
            )

        cmd = mock_run.call_args[0][0]
        assert any("upload-concurrency=16" in arg for arg in cmd)
        assert any("chunk-size=256M" in arg for arg in cmd)

    def test_passes_all_write_credentials_via_env(self, tmp_path: Path) -> None:
        src = tmp_path / "model.safetensors"
        src.touch()
        with (
            patch("ai_content_service.r2_transfer.shutil.which", return_value="/usr/bin/rclone"),
            patch(
                "ai_content_service.r2_transfer.subprocess.run", return_value=_mock_rclone()
            ) as mock_run,
        ):
            push(src_path=src, key=_KEY, creds=_WRITE_CREDS, bucket=_BUCKET, endpoint=_ENDPOINT)

        env = mock_run.call_args[1]["env"]
        assert env["RCLONE_S3_ACCESS_KEY_ID"] == _WRITE_CREDS.access_key_id
        assert env["RCLONE_S3_SECRET_ACCESS_KEY"] == _WRITE_CREDS.secret_access_key
        assert env["RCLONE_S3_SESSION_TOKEN"] == _WRITE_CREDS.session_token
        assert "AWS_ACCESS_KEY_ID" not in env
        assert "AWS_SECRET_ACCESS_KEY" not in env
        assert "AWS_SESSION_TOKEN" not in env

    def test_no_session_token_when_none(self, tmp_path: Path) -> None:
        src = tmp_path / "model.safetensors"
        src.touch()
        creds = R2WriteCreds(access_key_id="KEY", secret_access_key="SECRET", session_token=None)
        with (
            patch("ai_content_service.r2_transfer.shutil.which", return_value="/usr/bin/rclone"),
            patch(
                "ai_content_service.r2_transfer.subprocess.run", return_value=_mock_rclone()
            ) as mock_run,
        ):
            push(src_path=src, key=_KEY, creds=creds, bucket=_BUCKET, endpoint=_ENDPOINT)

        env = mock_run.call_args[1]["env"]
        assert "RCLONE_S3_SESSION_TOKEN" not in env
        assert "AWS_SESSION_TOKEN" not in env

    def test_nonzero_exit_raises_runtime_error(self, tmp_path: Path) -> None:
        src = tmp_path / "model.safetensors"
        src.touch()
        with (
            patch("ai_content_service.r2_transfer.shutil.which", return_value="/usr/bin/rclone"),
            patch(
                "ai_content_service.r2_transfer.subprocess.run",
                return_value=_mock_rclone(returncode=1),
            ),
            pytest.raises(RuntimeError),
        ):
            push(src_path=src, key=_KEY, creds=_WRITE_CREDS, bucket=_BUCKET, endpoint=_ENDPOINT)

    def test_missing_rclone_raises_loudly(self, tmp_path: Path) -> None:
        src = tmp_path / "model.safetensors"
        src.touch()
        with (
            patch("ai_content_service.r2_transfer.shutil.which", return_value=None),
            pytest.raises(RuntimeError, match="rclone not found"),
        ):
            push(src_path=src, key=_KEY, creds=_WRITE_CREDS, bucket=_BUCKET, endpoint=_ENDPOINT)
