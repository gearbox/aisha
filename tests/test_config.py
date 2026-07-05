"""Tests for configuration models."""

from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from ai_content_service.config import (
    BundleConfig,
    BundleMetadata,
    BundleVersion,
    ComfyUIConfig,
    CustomNode,
    ModelConfig,
    ModelFileConfig,
    Settings,
)


class TestCustomNode:
    """Tests for CustomNode model."""

    def test_valid_custom_node(self) -> None:
        """Test creating a valid custom node."""
        node = CustomNode(
            name="ComfyUI-GGUF",
            git_url="https://github.com/city96/ComfyUI-GGUF",
        )
        assert node.name == "ComfyUI-GGUF"
        assert node.commit_sha is None

    def test_custom_node_with_commit(self) -> None:
        """Test custom node with pinned commit."""
        node = CustomNode(
            name="ComfyUI-GGUF",
            git_url="https://github.com/city96/ComfyUI-GGUF",
            commit_sha="abc123def",
        )
        assert node.commit_sha == "abc123def"


class TestBundleVersion:
    """Tests for BundleVersion model."""

    def test_valid_version(self) -> None:
        """Test creating a valid bundle version."""
        version = BundleVersion(version="260101-01")
        assert version.version == "260101-01"
        assert str(version) == "260101-01"

    def test_invalid_version_format(self) -> None:
        """Test that invalid version formats are rejected."""
        with pytest.raises(ValidationError):
            BundleVersion(version="2025-01-01")

        with pytest.raises(ValidationError):
            BundleVersion(version="260101")

        with pytest.raises(ValidationError):
            BundleVersion(version="260101-1")

    def test_create_new_first_of_day(self) -> None:
        """Test creating first version of the day."""
        version = BundleVersion.create_new([])
        # Should end with -01
        assert version.version.endswith("-01")

    def test_create_new_increment(self) -> None:
        """Test creating incremented version."""
        # Get today's date prefix
        today = datetime.now(tz=timezone.utc).strftime("%y%m%d")
        existing = [f"{today}-01", f"{today}-02"]

        version = BundleVersion.create_new(existing)
        assert version.version == f"{today}-03"

    def test_create_new_ignores_other_dates(self) -> None:
        """Test that versions from other dates are ignored."""
        today = datetime.now(tz=timezone.utc).strftime("%y%m%d")
        existing = ["240101-01", "240101-02", f"{today}-01"]

        version = BundleVersion.create_new(existing)
        assert version.version == f"{today}-02"


class TestBundleMetadata:
    """Tests for BundleMetadata model."""

    def test_valid_metadata(self) -> None:
        """Test creating valid metadata."""
        metadata = BundleMetadata(
            name="wan_2.2_i2v",
            version="260101-01",
            description="WAN 2.2 Image to Video",
        )
        assert metadata.name == "wan_2.2_i2v"
        assert metadata.version == "260101-01"
        assert metadata.tested is False

    def test_created_at_default(self) -> None:
        """Test that created_at has a default value."""
        metadata = BundleMetadata(
            name="test",
            version="260101-01",
        )
        assert metadata.created_at is not None
        assert isinstance(metadata.created_at, datetime)


class TestComfyUIConfig:
    """Tests for ComfyUIConfig model."""

    def test_valid_config(self) -> None:
        """Test creating valid ComfyUI config."""
        config = ComfyUIConfig(commit="abc123def456")
        assert config.commit == "abc123def456"
        assert config.repo == "https://github.com/comfyanonymous/ComfyUI"

    def test_custom_repo(self) -> None:
        """Test with custom repo URL."""
        config = ComfyUIConfig(
            repo="https://github.com/fork/ComfyUI",
            commit="abc123",
        )
        assert config.repo == "https://github.com/fork/ComfyUI"


class TestBundleConfig:
    """Tests for BundleConfig model."""

    def test_valid_bundle_config(self) -> None:
        """Test creating a valid bundle config."""
        config = BundleConfig(
            metadata=BundleMetadata(
                name="test",
                version="260101-01",
            ),
            comfyui=ComfyUIConfig(commit="abc123"),
            custom_nodes=[
                CustomNode(
                    name="TestNode",
                    git_url="https://github.com/test/node",
                    commit_sha="def456",
                ),
            ],
        )
        assert config.metadata.name == "test"
        assert config.comfyui is not None
        assert config.comfyui.commit == "abc123"
        assert len(config.custom_nodes) == 1

    def test_custom_nodes_require_commit_sha(self) -> None:
        """Test that custom nodes in bundles must have commit_sha."""
        with pytest.raises(ValidationError) as exc_info:
            BundleConfig(
                metadata=BundleMetadata(
                    name="test",
                    version="260101-01",
                ),
                comfyui=ComfyUIConfig(commit="abc123"),
                custom_nodes=[
                    CustomNode(
                        name="TestNode",
                        git_url="https://github.com/test/node",
                        # Missing commit_sha
                    ),
                ],
            )
        assert "commit_sha" in str(exc_info.value)


class TestSettings:
    """Tests for Settings model."""

    def test_default_settings(self) -> None:
        """Test default settings values."""
        settings = Settings()
        assert settings.comfyui_path == Path("/workspace/ComfyUI")
        assert settings.max_concurrent_downloads == 3
        assert settings.verify_checksums is True
        assert settings.skip_existing is True
        assert settings.no_verify is False

    def test_models_path_property(self) -> None:
        """Test models_path derived property."""
        settings = Settings(comfyui_path=Path("/test/ComfyUI"))
        assert settings.models_path == Path("/test/ComfyUI/models")

    def test_custom_nodes_path_property(self) -> None:
        """Test custom_nodes_path derived property."""
        settings = Settings(comfyui_path=Path("/test/ComfyUI"))
        assert settings.custom_nodes_path == Path("/test/ComfyUI/custom_nodes")

    def test_bundles_path_default(self) -> None:
        """Test default bundles path."""
        settings = Settings()
        assert settings.bundles_path == Path("config/bundles")

    def test_settings_rejects_nonexistent_comfyui_python(self, tmp_path: Path) -> None:
        """Settings must fail-fast if comfyui_python points at a missing file."""
        bogus = tmp_path / "definitely-not-a-real-python"
        with pytest.raises(ValueError, match="comfyui_python"):
            Settings(comfyui_python=bogus)

    def test_settings_accepts_real_python(self, tmp_path: Path) -> None:
        """Settings must accept a comfyui_python path that exists and is a file."""
        real_python = tmp_path / "python"
        real_python.write_text("")
        real_python.chmod(0o755)
        settings = Settings(comfyui_python=real_python)
        assert settings.comfyui_python == real_python

    def test_rclone_max_transfer_seconds_default(self) -> None:
        settings = Settings()
        assert settings.rclone_max_transfer_seconds == 3600

    def test_settings_rclone_max_transfer_seconds_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ACS_RCLONE_MAX_TRANSFER_SECONDS", "120")
        settings = Settings()
        assert settings.rclone_max_transfer_seconds == 120


class TestModelFileConfigPathTraversal:
    """Tests for path-traversal validators on the live ModelFileConfig class."""

    @pytest.mark.parametrize("bad_filename", ["../x", "a/b", "a\\b", "..", ".", ""])
    def test_model_file_config_filename_path_traversal_rejected(self, bad_filename: str) -> None:
        with pytest.raises(ValidationError):
            ModelFileConfig(
                name="Test",
                url="https://example.com/model.gguf",
                filename=bad_filename,
            )

    def test_model_file_config_valid_filename_accepted(self) -> None:
        cfg = ModelFileConfig(
            name="Test",
            url="https://example.com/model.gguf",
            filename="model.gguf",
        )
        assert cfg.filename == "model.gguf"


class TestModelConfigPathTraversal:
    """Tests for path-traversal validators on the live ModelConfig class."""

    @pytest.mark.parametrize("bad_subdirectory", ["../loras", "/abs", "a/../b"])
    def test_model_config_subdirectory_traversal_rejected(self, bad_subdirectory: str) -> None:
        with pytest.raises(ValidationError):
            ModelConfig(
                name="m",
                model_type="loras",
                files=[
                    ModelFileConfig(name="f", url="https://example.com/f.gguf", filename="f.gguf")
                ],
                subdirectory=bad_subdirectory,
            )

    @pytest.mark.parametrize("bad_model_type", ["../etc", "/abs", "a/../b", ""])
    def test_model_config_model_type_traversal_rejected(self, bad_model_type: str) -> None:
        with pytest.raises(ValidationError):
            ModelConfig(
                name="m",
                model_type=bad_model_type,
                files=[
                    ModelFileConfig(name="f", url="https://example.com/f.gguf", filename="f.gguf")
                ],
            )

    def test_model_config_valid_values_accepted(self) -> None:
        config = ModelConfig(
            name="m",
            model_type="loras",
            files=[ModelFileConfig(name="f", url="https://example.com/f.gguf", filename="f.gguf")],
            subdirectory="sdxl",
        )
        assert config.model_type == "loras"
        assert config.subdirectory == "sdxl"

    def test_model_config_no_subdirectory_accepted(self) -> None:
        config = ModelConfig(
            name="m",
            model_type="vae",
            files=[ModelFileConfig(name="f", url="https://example.com/f.gguf", filename="f.gguf")],
        )
        assert config.subdirectory is None

    def test_target_subpath_without_subdirectory(self) -> None:
        config = ModelConfig(
            name="m",
            model_type="diffusion_models",
            files=[ModelFileConfig(name="f", url="https://example.com/f.gguf", filename="f.gguf")],
        )
        assert config.target_subpath == "diffusion_models"

    def test_target_subpath_with_subdirectory(self) -> None:
        config = ModelConfig(
            name="m",
            model_type="loras",
            files=[ModelFileConfig(name="f", url="https://example.com/f.gguf", filename="f.gguf")],
            subdirectory="anime",
        )
        assert config.target_subpath == "loras/anime"
