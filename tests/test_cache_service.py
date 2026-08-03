"""Tests for cache_service (Typer-free `cache push` orchestration core)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from ai_content_service import cache_service
from ai_content_service.cache_credentials import ApexCacheCredentialProvider
from ai_content_service.config import (
    BundleConfig,
    BundleMetadata,
    ModelConfig,
    ModelFileConfig,
    Settings,
)
from ai_content_service.r2_transfer import CachePushError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    base = {
        "comfyui_path": tmp_path,
        "apex_base_url": "https://api.example.com",
        "apex_admin_token": "admin-token",
        "r2_s3_endpoint": "https://account.r2.cloudflarestorage.com",
    } | overrides
    return Settings(**base)  # type: ignore[arg-type]


def _model_file(filename: str, sha256: str | None = None) -> ModelFileConfig:
    return ModelFileConfig(
        name=filename,
        url=f"https://huggingface.co/test/{filename}",
        filename=filename,
        sha256=sha256,
    )


def _target(
    tmp_path: Path, filename: str, content: bytes, sha256: str | None = None
) -> cache_service.PushTarget:
    """Build a PushTarget whose disk_path sits under `tmp_path/models/checkpoints/`,
    matching the models_base that push_models derives from settings.comfyui_path."""
    model = ModelConfig(name="m", model_type="checkpoints", files=[_model_file(filename, sha256)])
    disk_path = tmp_path / "models" / "checkpoints" / filename
    disk_path.parent.mkdir(parents=True, exist_ok=True)
    disk_path.write_bytes(content)
    return cache_service.PushTarget(model=model, file=model.files[0], disk_path=disk_path)


def _provider(*responses: MagicMock) -> ApexCacheCredentialProvider:
    client = MagicMock()
    client.post.side_effect = list(responses)
    return ApexCacheCredentialProvider(
        base_url="https://api.example.com",
        admin_token="admin-token",
        client=client,
    )


def _cred_response(sha256: str) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "r2_key": f"models/by-sha256/{sha256}",
        "credentials": {
            "access_key_id": "KEY",
            "secret_access_key": "SECRET",
            "session_token": "TOKEN",
        },
    }
    return resp


def _error_response(status_code: int) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.raise_for_status = MagicMock(
        side_effect=httpx.HTTPStatusError(f"HTTP {status_code}", request=MagicMock(), response=resp)
    )
    return resp


def _ok_finalize_response() -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    return resp


def _rejected_finalize_response(status_code: int) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = f"rejected ({status_code})"
    resp.raise_for_status = MagicMock()
    return resp


# ---------------------------------------------------------------------------
# push_models
# ---------------------------------------------------------------------------


class TestPushModelsHappyPath:
    def test_all_green_reports_ok(self, tmp_path: Path) -> None:
        content = b"weights"
        sha256 = hashlib.sha256(content).hexdigest()
        target = _target(tmp_path, "model.safetensors", content, sha256=sha256)
        provider = _provider(_cred_response(sha256), _ok_finalize_response())

        with patch("ai_content_service.cache_service.r2_push") as mock_push:
            report = cache_service.push_models(
                _settings(tmp_path), [target], console=MagicMock(), provider=provider
            )

        assert report.ok is True
        assert len(report.results) == 1
        assert report.results[0].filename == "model.safetensors"
        assert report.results[0].ok is True
        mock_push.assert_called_once()


class TestPushModelsFailures:
    def test_credentials_4xx_yields_failed_result(self, tmp_path: Path) -> None:
        content = b"weights"
        sha256 = hashlib.sha256(content).hexdigest()
        target = _target(tmp_path, "model.safetensors", content, sha256=sha256)
        provider = _provider(_error_response(401))

        with patch("ai_content_service.cache_service.r2_push") as mock_push:
            report = cache_service.push_models(
                _settings(tmp_path), [target], console=MagicMock(), provider=provider
            )

        assert report.ok is False
        assert len(report.results) == 1
        assert report.results[0].ok is False
        mock_push.assert_not_called()

    def test_mismatched_apex_key_never_starts_r2_upload(self, tmp_path: Path) -> None:
        content = b"weights"
        sha256 = hashlib.sha256(content).hexdigest()
        target = _target(tmp_path, "model.safetensors", content, sha256=sha256)
        response = _cred_response(sha256)
        response.json.return_value["r2_key"] = f"models/by-sha256/{'0' * 64}"
        provider = _provider(response)

        with patch("ai_content_service.cache_service.r2_push") as mock_push:
            report = cache_service.push_models(
                _settings(tmp_path), [target], console=MagicMock(), provider=provider
            )

        assert report.ok is False
        assert "response invalid" in report.results[0].detail
        mock_push.assert_not_called()

    @pytest.mark.parametrize("status_code", [409, 422])
    def test_finalize_rejection_yields_failed_result(
        self, tmp_path: Path, status_code: int
    ) -> None:
        content = b"weights"
        sha256 = hashlib.sha256(content).hexdigest()
        target = _target(tmp_path, "model.safetensors", content, sha256=sha256)
        provider = _provider(_cred_response(sha256), _rejected_finalize_response(status_code))

        with patch("ai_content_service.cache_service.r2_push"):
            report = cache_service.push_models(
                _settings(tmp_path), [target], console=MagicMock(), provider=provider
            )

        assert report.ok is False
        assert report.results[0].ok is False
        assert str(status_code) in report.results[0].detail

    def test_rclone_push_error_yields_failed_result(self, tmp_path: Path) -> None:
        content = b"weights"
        sha256 = hashlib.sha256(content).hexdigest()
        target = _target(tmp_path, "model.safetensors", content, sha256=sha256)
        provider = _provider(_cred_response(sha256))

        with patch(
            "ai_content_service.cache_service.r2_push",
            side_effect=CachePushError("rclone exploded"),
        ):
            report = cache_service.push_models(
                _settings(tmp_path), [target], console=MagicMock(), provider=provider
            )

        assert report.ok is False
        assert "rclone" in report.results[0].detail.lower()

    def test_missing_file_yields_failed_result_without_http(self, tmp_path: Path) -> None:
        model = ModelConfig(
            name="m", model_type="checkpoints", files=[_model_file("missing.safetensors")]
        )
        target = cache_service.PushTarget(
            model=model,
            file=model.files[0],
            disk_path=tmp_path / "models" / "checkpoints" / "missing.safetensors",
        )

        provider = MagicMock()
        report = cache_service.push_models(
            _settings(tmp_path), [target], console=MagicMock(), provider=provider
        )

        assert report.ok is False
        assert "not found" in report.results[0].detail.lower()
        provider.mint.assert_not_called()

    def test_multiple_targets_continue_after_failure(self, tmp_path: Path) -> None:
        """A failed target must not stop processing of subsequent targets."""
        good_content = b"good weights"
        good_sha256 = hashlib.sha256(good_content).hexdigest()
        bad_target = _target(tmp_path, "bad.safetensors", b"x", sha256="0" * 64)
        good_target = _target(tmp_path, "good.safetensors", good_content, sha256=good_sha256)

        provider = _provider(
            _error_response(500), _cred_response(good_sha256), _ok_finalize_response()
        )

        with patch("ai_content_service.cache_service.r2_push"):
            report = cache_service.push_models(
                _settings(tmp_path),
                [bad_target, good_target],
                console=MagicMock(),
                provider=provider,
            )

        assert report.ok is False
        assert [r.ok for r in report.results] == [False, True]

    @pytest.mark.parametrize("failure", ["hash", "stat"])
    def test_local_preparation_failure_does_not_abort_the_next_target(
        self, tmp_path: Path, failure: str
    ) -> None:
        bad_target = _target(tmp_path, "bad.safetensors", b"bad")
        good_content = b"good"
        good_sha256 = hashlib.sha256(good_content).hexdigest()
        good_target = _target(tmp_path, "good.safetensors", good_content, sha256=good_sha256)
        provider = MagicMock()
        provider.name = "direct"
        provider.mint.return_value = MagicMock(
            r2_key=f"models/by-sha256/{good_sha256}", creds=MagicMock()
        )

        if failure == "hash":
            with (
                patch(
                    "ai_content_service.cache_service.compute_file_sha256",
                    side_effect=PermissionError("denied"),
                ),
                patch("ai_content_service.cache_service.r2_push"),
            ):
                report = cache_service.push_models(
                    _settings(tmp_path),
                    [bad_target, good_target],
                    console=MagicMock(),
                    provider=provider,
                )
        else:
            original_stat = Path.stat

            def side_effect(path: Path, *args: object, **kwargs: object) -> object:
                if path == bad_target.disk_path:
                    raise PermissionError("denied")
                return original_stat(path, *args, **kwargs)

            with (
                patch.object(Path, "stat", autospec=True, side_effect=side_effect),
                patch("ai_content_service.cache_service.r2_push"),
            ):
                report = cache_service.push_models(
                    _settings(tmp_path),
                    [bad_target, good_target],
                    console=MagicMock(),
                    provider=provider,
                )

        assert [result.ok for result in report.results] == [False, True]

    def test_disappearing_file_does_not_abort_the_next_target(self, tmp_path: Path) -> None:
        bad_target = _target(tmp_path, "bad.safetensors", b"bad")
        good_content = b"good"
        good_sha256 = hashlib.sha256(good_content).hexdigest()
        good_target = _target(tmp_path, "good.safetensors", good_content, sha256=good_sha256)
        provider = MagicMock()
        provider.name = "direct"
        provider.mint.return_value = MagicMock(
            r2_key=f"models/by-sha256/{good_sha256}", creds=MagicMock()
        )
        original_hash = cache_service.compute_file_sha256

        def _remove_before_hash(path: Path) -> str:
            if path == bad_target.disk_path:
                path.unlink()
            return original_hash(path)

        with (
            patch(
                "ai_content_service.cache_service.compute_file_sha256",
                side_effect=_remove_before_hash,
            ),
            patch("ai_content_service.cache_service.r2_push"),
        ):
            report = cache_service.push_models(
                _settings(tmp_path),
                [bad_target, good_target],
                console=MagicMock(),
                provider=provider,
            )

        assert [result.ok for result in report.results] == [False, True]

    def test_civitai_token_is_sanitized_before_credential_mint(self, tmp_path: Path) -> None:
        content = b"weights"
        digest = hashlib.sha256(content).hexdigest()
        target = _target(tmp_path, "model.safetensors", content, sha256=digest)
        target.file.url = "https://civitai.com/api/download/models/1?a=1&TOKEN=secret&a=2"
        provider = MagicMock()
        provider.name = "direct"
        provider.mint.return_value = MagicMock(
            r2_key=f"models/by-sha256/{digest}", creds=MagicMock()
        )

        with patch("ai_content_service.cache_service.r2_push"):
            report = cache_service.push_models(
                _settings(tmp_path), [target], console=MagicMock(), provider=provider
            )

        assert report.ok is True
        assert provider.mint.call_args.kwargs["source_url"] == (
            "https://civitai.com/api/download/models/1?a=1&a=2"
        )


class TestPushModelsSha256Guard:
    """D5 normalizes sha256 at the ModelFileConfig boundary; this guards a regression there."""

    def test_non_lowercase_hex_sha256_is_rejected_without_any_http_call(
        self, tmp_path: Path
    ) -> None:
        """Simulates validation having been bypassed (e.g. model_construct)."""
        content = b"weights"
        model = ModelConfig(
            name="m",
            model_type="checkpoints",
            files=[
                ModelFileConfig.model_construct(
                    name="model.safetensors",
                    url="https://huggingface.co/test/model.safetensors",
                    filename="model.safetensors",
                    sha256="NOT-LOWERCASE-HEX",
                    size_bytes=None,
                )
            ],
        )
        disk_path = tmp_path / "models" / "checkpoints" / "model.safetensors"
        disk_path.parent.mkdir(parents=True, exist_ok=True)
        disk_path.write_bytes(content)
        target = cache_service.PushTarget(model=model, file=model.files[0], disk_path=disk_path)

        provider = MagicMock()
        report = cache_service.push_models(
            _settings(tmp_path), [target], console=MagicMock(), provider=provider
        )

        assert report.ok is False
        assert "lowercase-hex" in report.results[0].detail
        provider.mint.assert_not_called()


# ---------------------------------------------------------------------------
# collect_targets
# ---------------------------------------------------------------------------


class TestCollectTargets:
    def _bundle(self) -> BundleConfig:
        return BundleConfig(
            metadata=BundleMetadata(name="b", version="260101-01"),
            models=[
                ModelConfig(
                    name="m",
                    model_type="checkpoints",
                    files=[_model_file("a.safetensors"), _model_file("b.safetensors")],
                )
            ],
        )

    def test_collect_all_when_no_filter(self, tmp_path: Path) -> None:
        targets = cache_service.collect_targets(self._bundle(), tmp_path, None)
        assert {t.file.filename for t in targets} == {"a.safetensors", "b.safetensors"}

    def test_collect_filters_by_filename(self, tmp_path: Path) -> None:
        targets = cache_service.collect_targets(self._bundle(), tmp_path, "b.safetensors")
        assert [t.file.filename for t in targets] == ["b.safetensors"]

    def test_collect_uses_target_subpath(self, tmp_path: Path) -> None:
        bundle = BundleConfig(
            metadata=BundleMetadata(name="b", version="260101-01"),
            models=[
                ModelConfig(
                    name="m",
                    model_type="loras",
                    subdirectory="anime",
                    files=[_model_file("a.safetensors")],
                )
            ],
        )
        targets = cache_service.collect_targets(bundle, tmp_path, None)
        assert targets[0].disk_path == tmp_path / "loras" / "anime" / "a.safetensors"
