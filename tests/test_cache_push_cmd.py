"""Tests for acs cache push CLI command."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
import yaml
from pydantic import SecretStr
from typer.testing import CliRunner

from ai_content_service.cli import app
from ai_content_service.config import Settings, reset_settings

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

runner = CliRunner()


@pytest.fixture(autouse=True)
def reset_singleton() -> Iterator[None]:
    reset_settings()
    yield
    reset_settings()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_bundle_tree(tmp_path: Path) -> tuple[Settings, Path, str]:
    """Create a minimal bundle + model file structure in tmp_path.

    Returns (settings, model_file_path, sha256).
    """
    bundles_path = tmp_path / "bundles"
    comfyui_path = tmp_path / "ComfyUI"

    bundle_version_dir = bundles_path / "test_bundle" / "260101-01"
    bundle_version_dir.mkdir(parents=True)

    model_dir = comfyui_path / "models" / "diffusion_models"
    model_dir.mkdir(parents=True)

    content = b"fake model weights for testing"
    sha256 = hashlib.sha256(content).hexdigest()
    model_file = model_dir / "model.safetensors"
    model_file.write_bytes(content)

    bundle_yaml = bundle_version_dir / "bundle.yaml"
    bundle_yaml.write_text(
        yaml.dump(
            {
                "metadata": {
                    "name": "test_bundle",
                    "version": "260101-01",
                    "created_at": "2026-01-01T00:00:00+00:00",
                },
                "models": [
                    {
                        "name": "Test Model",
                        "model_type": "diffusion_models",
                        "files": [
                            {
                                "name": "model.safetensors",
                                "url": "https://huggingface.co/test/model.safetensors",
                                "filename": "model.safetensors",
                                "sha256": sha256,
                            }
                        ],
                    }
                ],
            }
        )
    )

    settings = Settings(
        comfyui_path=comfyui_path,
        bundles_path=bundles_path,
        apex_base_url="https://api.example.com",
        apex_admin_token="test-apex-admin-key",  # type: ignore[arg-type]
        r2_s3_endpoint="https://account.r2.cloudflarestorage.com",
        r2_model_cache_bucket="apex-model-cache",
    )
    return settings, model_file, sha256


def _mock_http_client(cred_response: MagicMock, finalize_response: MagicMock) -> MagicMock:
    """Build a mock HTTP client with preset POST responses."""
    mock_client = MagicMock()
    mock_client.post.side_effect = [cred_response, finalize_response]
    return mock_client


def _cred_response(sha256: str) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "r2_key": f"models/by-sha256/{sha256}",
        "credentials": {
            "access_key_id": "TEMPKEY123",
            "secret_access_key": "TEMPSECRET123",
            "session_token": "TEMPSESSION123",
        },
    }
    return resp


def _ok_finalize_response() -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    return resp


def _error_finalize_response(status_code: int) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = f"object not found ({status_code})"
    resp.raise_for_status = MagicMock()
    return resp


# ---------------------------------------------------------------------------
# Guards: missing required config
# ---------------------------------------------------------------------------


class TestMissingConfig:
    def test_missing_admin_token_refuses_before_transfer(self, tmp_path: Path) -> None:
        settings = Settings(
            comfyui_path=tmp_path / "ComfyUI",
            bundles_path=tmp_path / "bundles",
            apex_base_url="https://api.example.com",
            # apex_admin_token intentionally absent
        )
        with (
            patch("ai_content_service.cli.get_settings", return_value=settings),
            patch("ai_content_service.cache_service.r2_push") as mock_push,
        ):
            result = runner.invoke(app, ["cache", "push", "test_bundle", "--all"])

        assert result.exit_code != 0
        assert "ACS_APEX_ADMIN_TOKEN" in result.output
        mock_push.assert_not_called()

    def test_missing_apex_base_url_exits(self, tmp_path: Path) -> None:
        settings = Settings(
            comfyui_path=tmp_path / "ComfyUI",
            bundles_path=tmp_path / "bundles",
            apex_admin_token="token",  # type: ignore[arg-type]
            # apex_base_url absent
        )
        with patch("ai_content_service.cli.get_settings", return_value=settings):
            result = runner.invoke(app, ["cache", "push", "test_bundle", "--all"])

        assert result.exit_code != 0
        assert "ACS_APEX_BASE_URL" in result.output

    def test_missing_r2_endpoint_exits(self, tmp_path: Path) -> None:
        settings = Settings(
            comfyui_path=tmp_path / "ComfyUI",
            bundles_path=tmp_path / "bundles",
            apex_admin_token="token",  # type: ignore[arg-type]
            apex_base_url="https://api.example.com",
            # r2_s3_endpoint absent
        )
        with patch("ai_content_service.cli.get_settings", return_value=settings):
            result = runner.invoke(app, ["cache", "push", "test_bundle", "--all"])

        assert result.exit_code != 0
        assert "ACS_R2_S3_ENDPOINT" in result.output

    def test_no_model_selector_exits(self, tmp_path: Path) -> None:
        settings = Settings(
            comfyui_path=tmp_path / "ComfyUI",
            bundles_path=tmp_path / "bundles",
            apex_admin_token="token",  # type: ignore[arg-type]
            apex_base_url="https://api.example.com",
            r2_s3_endpoint="https://endpoint",
        )
        with patch("ai_content_service.cli.get_settings", return_value=settings):
            result = runner.invoke(app, ["cache", "push", "test_bundle"])
        assert result.exit_code != 0

    def test_both_model_and_all_exits(self, tmp_path: Path) -> None:
        settings = Settings(
            comfyui_path=tmp_path / "ComfyUI",
            bundles_path=tmp_path / "bundles",
            apex_admin_token="token",  # type: ignore[arg-type]
            apex_base_url="https://api.example.com",
            r2_s3_endpoint="https://endpoint",
        )
        with patch("ai_content_service.cli.get_settings", return_value=settings):
            result = runner.invoke(
                app, ["cache", "push", "test_bundle", "--model", "a.safetensors", "--all"]
            )
        assert result.exit_code != 0

    def test_direct_mode_requires_write_credentials_before_resolving_bundle(
        self, tmp_path: Path
    ) -> None:
        settings = Settings(
            comfyui_path=tmp_path / "ComfyUI",
            bundles_path=tmp_path / "bundles",
            r2_s3_endpoint="https://endpoint",
        )
        with patch("ai_content_service.cli.get_settings", return_value=settings):
            result = runner.invoke(app, ["cache", "push", "test_bundle", "--all", "--direct"])

        assert result.exit_code != 0
        assert "ACS_R2_WRITE_ACCESS_KEY_ID" in result.output


# ---------------------------------------------------------------------------
# Happy path — mint → push → finalize
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_push_single_model_succeeds(self, tmp_path: Path) -> None:
        settings, _model_file, sha256 = _make_bundle_tree(tmp_path)
        http_ctx = _mock_http_client(_cred_response(sha256), _ok_finalize_response())

        with (
            patch("ai_content_service.cli.get_settings", return_value=settings),
            patch("ai_content_service.cache_credentials.httpx.Client", return_value=http_ctx),
            patch("ai_content_service.cache_service.r2_push") as mock_push,
            patch("ai_content_service.r2_transfer.shutil.which", return_value="/usr/bin/rclone"),
        ):
            result = runner.invoke(
                app,
                ["cache", "push", "test_bundle:260101-01", "--model", "model.safetensors"],
            )

        assert result.exit_code == 0, result.output
        assert "cache.push.done" in result.output
        mock_push.assert_called_once()

    def test_push_posts_sha256_to_credentials_endpoint(self, tmp_path: Path) -> None:
        settings, _model_file, sha256 = _make_bundle_tree(tmp_path)
        http_ctx = _mock_http_client(_cred_response(sha256), _ok_finalize_response())

        with (
            patch("ai_content_service.cli.get_settings", return_value=settings),
            patch("ai_content_service.cache_credentials.httpx.Client", return_value=http_ctx),
            patch("ai_content_service.cache_service.r2_push"),
            patch("ai_content_service.r2_transfer.shutil.which", return_value="/usr/bin/rclone"),
        ):
            runner.invoke(
                app,
                ["cache", "push", "test_bundle:260101-01", "--model", "model.safetensors"],
            )

        mock_client = http_ctx
        cred_call = mock_client.post.call_args_list[0]
        posted_json = cred_call.kwargs["json"]
        assert posted_json["sha256"] == sha256
        assert posted_json["filename"] == "model.safetensors"
        assert posted_json["model_type"] == "diffusion_models"

    def test_push_posts_sha256_and_size_to_finalize(self, tmp_path: Path) -> None:
        settings, model_file, sha256 = _make_bundle_tree(tmp_path)
        http_ctx = _mock_http_client(_cred_response(sha256), _ok_finalize_response())

        with (
            patch("ai_content_service.cli.get_settings", return_value=settings),
            patch("ai_content_service.cache_credentials.httpx.Client", return_value=http_ctx),
            patch("ai_content_service.cache_service.r2_push"),
            patch("ai_content_service.r2_transfer.shutil.which", return_value="/usr/bin/rclone"),
        ):
            runner.invoke(
                app,
                ["cache", "push", "test_bundle:260101-01", "--model", "model.safetensors"],
            )

        mock_client = http_ctx
        fin_call = mock_client.post.call_args_list[1]
        fin_json = fin_call.kwargs["json"]
        assert fin_json["sha256"] == sha256
        assert fin_json["size_bytes"] == model_file.stat().st_size

    def test_push_sends_admin_token_in_auth_header(self, tmp_path: Path) -> None:
        settings, _model_file, sha256 = _make_bundle_tree(tmp_path)
        http_ctx = _mock_http_client(_cred_response(sha256), _ok_finalize_response())

        with (
            patch("ai_content_service.cli.get_settings", return_value=settings),
            patch("ai_content_service.cache_credentials.httpx.Client", return_value=http_ctx),
            patch("ai_content_service.cache_service.r2_push"),
            patch("ai_content_service.r2_transfer.shutil.which", return_value="/usr/bin/rclone"),
        ):
            runner.invoke(
                app,
                ["cache", "push", "test_bundle:260101-01", "--model", "model.safetensors"],
            )

        mock_client = http_ctx
        for call in mock_client.post.call_args_list:
            assert call.kwargs["headers"]["Authorization"] == "Bearer test-apex-admin-key"

    def test_push_strips_civitai_token_from_source_url(self, tmp_path: Path) -> None:
        """Civitai token must be removed from source_url sent to Apex."""
        bundles_path = tmp_path / "bundles"
        comfyui_path = tmp_path / "ComfyUI"
        bundle_dir = bundles_path / "civitai_bundle" / "260101-01"
        bundle_dir.mkdir(parents=True)
        model_dir = comfyui_path / "models" / "checkpoints"
        model_dir.mkdir(parents=True)
        content = b"civitai weights"
        sha256 = hashlib.sha256(content).hexdigest()
        (model_dir / "civitai.safetensors").write_bytes(content)

        (bundle_dir / "bundle.yaml").write_text(
            yaml.dump(
                {
                    "metadata": {
                        "name": "civitai_bundle",
                        "version": "260101-01",
                        "created_at": "2026-01-01T00:00:00+00:00",
                    },
                    "models": [
                        {
                            "name": "Civitai Model",
                            "model_type": "checkpoints",
                            "files": [
                                {
                                    "name": "civitai.safetensors",
                                    "url": "https://civitai.com/api/download/models/123?token=SECRET_TOKEN",
                                    "filename": "civitai.safetensors",
                                    "sha256": sha256,
                                }
                            ],
                        }
                    ],
                }
            )
        )

        settings = Settings(
            comfyui_path=comfyui_path,
            bundles_path=bundles_path,
            apex_base_url="https://api.example.com",
            apex_admin_token="test-apex-admin-key",  # type: ignore[arg-type]
            r2_s3_endpoint="https://account.r2.cloudflarestorage.com",
        )
        http_ctx = _mock_http_client(_cred_response(sha256), _ok_finalize_response())

        with (
            patch("ai_content_service.cli.get_settings", return_value=settings),
            patch("ai_content_service.cache_credentials.httpx.Client", return_value=http_ctx),
            patch("ai_content_service.cache_service.r2_push"),
            patch("ai_content_service.r2_transfer.shutil.which", return_value="/usr/bin/rclone"),
        ):
            runner.invoke(
                app,
                ["cache", "push", "civitai_bundle:260101-01", "--model", "civitai.safetensors"],
            )

        mock_client = http_ctx
        cred_call = mock_client.post.call_args_list[0]
        source_url = cred_call.kwargs["json"]["source_url"]
        assert "SECRET_TOKEN" not in source_url
        assert "token=" not in source_url

    def test_push_computes_sha256_when_not_declared(self, tmp_path: Path) -> None:
        """If ModelFileConfig.sha256 is None, the command computes it from disk."""
        bundles_path = tmp_path / "bundles"
        comfyui_path = tmp_path / "ComfyUI"
        bundle_dir = bundles_path / "no_sha_bundle" / "260101-01"
        bundle_dir.mkdir(parents=True)
        model_dir = comfyui_path / "models" / "vae"
        model_dir.mkdir(parents=True)
        content = b"vae weights data"
        expected_sha256 = hashlib.sha256(content).hexdigest()
        (model_dir / "vae.safetensors").write_bytes(content)

        (bundle_dir / "bundle.yaml").write_text(
            yaml.dump(
                {
                    "metadata": {
                        "name": "no_sha_bundle",
                        "version": "260101-01",
                        "created_at": "2026-01-01T00:00:00+00:00",
                    },
                    "models": [
                        {
                            "name": "VAE",
                            "model_type": "vae",
                            "files": [
                                {
                                    "name": "vae.safetensors",
                                    "url": "https://huggingface.co/test/vae.safetensors",
                                    "filename": "vae.safetensors",
                                    # sha256 intentionally absent
                                }
                            ],
                        }
                    ],
                }
            )
        )

        settings = Settings(
            comfyui_path=comfyui_path,
            bundles_path=bundles_path,
            apex_base_url="https://api.example.com",
            apex_admin_token="test-apex-admin-key",  # type: ignore[arg-type]
            r2_s3_endpoint="https://account.r2.cloudflarestorage.com",
        )
        http_ctx = _mock_http_client(_cred_response(expected_sha256), _ok_finalize_response())

        with (
            patch("ai_content_service.cli.get_settings", return_value=settings),
            patch("ai_content_service.cache_credentials.httpx.Client", return_value=http_ctx),
            patch("ai_content_service.cache_service.r2_push"),
            patch("ai_content_service.r2_transfer.shutil.which", return_value="/usr/bin/rclone"),
        ):
            result = runner.invoke(
                app, ["cache", "push", "no_sha_bundle:260101-01", "--model", "vae.safetensors"]
            )

        assert result.exit_code == 0, result.output
        mock_client = http_ctx
        cred_call = mock_client.post.call_args_list[0]
        assert cred_call.kwargs["json"]["sha256"] == expected_sha256

    def test_direct_push_uses_static_write_credentials_without_apex(self, tmp_path: Path) -> None:
        settings, _model_file, _sha256 = _make_bundle_tree(tmp_path)
        settings = settings.model_copy(
            update={
                "apex_base_url": None,
                "apex_admin_token": None,
                "r2_write_access_key_id": "DIRECT_KEY",
                "r2_write_secret_access_key": SecretStr("DIRECT_SECRET"),
            }
        )
        with (
            patch("ai_content_service.cli.get_settings", return_value=settings),
            patch("ai_content_service.cache_service.r2_push") as mock_push,
        ):
            result = runner.invoke(
                app,
                ["cache", "push", "test_bundle:260101-01", "--all", "--direct"],
            )

        assert result.exit_code == 0, result.output
        assert "credential mode: direct" in result.output
        assert mock_push.call_args.kwargs["creds"].access_key_id == "DIRECT_KEY"


# ---------------------------------------------------------------------------
# Finalize rejection (409 / 422) — non-zero exit
# ---------------------------------------------------------------------------


class TestFinalizeRejection:
    @pytest.mark.parametrize("status_code", [409, 422])
    def test_finalize_rejection_exits_nonzero(self, tmp_path: Path, status_code: int) -> None:
        settings, _model_file, sha256 = _make_bundle_tree(tmp_path)
        http_ctx = _mock_http_client(_cred_response(sha256), _error_finalize_response(status_code))

        with (
            patch("ai_content_service.cli.get_settings", return_value=settings),
            patch("ai_content_service.cache_credentials.httpx.Client", return_value=http_ctx),
            patch("ai_content_service.cache_service.r2_push"),
            patch("ai_content_service.r2_transfer.shutil.which", return_value="/usr/bin/rclone"),
        ):
            result = runner.invoke(
                app,
                ["cache", "push", "test_bundle:260101-01", "--model", "model.safetensors"],
            )

        assert result.exit_code != 0
        assert str(status_code) in result.output
