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
    ModelDefinition,
    ModelFile,
    ModelType,
    Settings,
)


class TestModelFile:
    """Tests for ModelFile model."""

    def test_valid_model_file(self) -> None:
        """Test creating a valid model file."""
        model_file = ModelFile(
            name="Test Model",
            url="https://example.com/model.gguf",
            filename="model.gguf",
        )
        assert model_file.name == "Test Model"
        assert str(model_file.url) == "https://example.com/model.gguf"
        assert model_file.filename == "model.gguf"
        assert model_file.sha256 is None

    def test_model_file_with_checksum(self) -> None:
        """Test model file with SHA256 checksum."""
        model_file = ModelFile(
            name="Test Model",
            url="https://example.com/model.gguf",
            filename="model.gguf",
            sha256="abc123",
            size_bytes=1024,
        )
        assert model_file.sha256 == "abc123"
        assert model_file.size_bytes == 1024

    def test_invalid_filename_with_path(self) -> None:
        """Test that filenames with path separators are rejected."""
        with pytest.raises(ValidationError):
            ModelFile(
                name="Test",
                url="https://example.com/model.gguf",
                filename="../evil.gguf",
            )

    def test_invalid_filename_with_backslash(self) -> None:
        """Test that filenames with backslashes are rejected."""
        with pytest.raises(ValidationError):
            ModelFile(
                name="Test",
                url="https://example.com/model.gguf",
                filename="path\\model.gguf",
            )


class TestModelDefinition:
    """Tests for ModelDefinition model."""

    def test_valid_model_definition(self) -> None:
        """Test creating a valid model definition."""
        model = ModelDefinition(
            name="test-model",
            description="A test model",
            model_type=ModelType.DIFFUSION,
            files=[
                ModelFile(
                    name="File 1",
                    url="https://example.com/file1.gguf",
                    filename="file1.gguf",
                ),
            ],
        )
        assert model.name == "test-model"
        assert model.model_type == ModelType.DIFFUSION
        assert len(model.files) == 1

    def test_target_subpath_without_subfolder(self) -> None:
        """Test target_subpath property without subfolder."""
        model = ModelDefinition(
            name="test",
            model_type=ModelType.DIFFUSION,
            files=[
                ModelFile(
                    name="F1",
                    url="https://example.com/f.gguf",
                    filename="f.gguf",
                ),
            ],
        )
        assert model.target_subpath == "diffusion_models"

    def test_target_subpath_with_subfolder(self) -> None:
        """Test target_subpath property with subfolder."""
        model = ModelDefinition(
            name="test",
            model_type=ModelType.LORA,
            subfolder="anime",
            files=[
                ModelFile(
                    name="F1",
                    url="https://example.com/f.safetensors",
                    filename="f.safetensors",
                ),
            ],
        )
        assert model.target_subpath == "loras/anime"

    def test_empty_files_rejected(self) -> None:
        """Test that models with no files are rejected."""
        with pytest.raises(ValidationError):
            ModelDefinition(
                name="test",
                model_type=ModelType.DIFFUSION,
                files=[],
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
