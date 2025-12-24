"""Tests for configuration models."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from ai_content_service.config import (
    CustomNode,
    DeploymentConfig,
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


class TestDeploymentConfig:
    """Tests for DeploymentConfig model."""

    def test_empty_config(self) -> None:
        """Test creating an empty deployment config."""
        config = DeploymentConfig()
        assert config.models == []
        assert config.custom_nodes == []
        assert config.workflows == []

    def test_full_config(self) -> None:
        """Test creating a full deployment config."""
        config = DeploymentConfig(
            models=[
                ModelDefinition(
                    name="test-model",
                    model_type=ModelType.DIFFUSION,
                    files=[
                        ModelFile(
                            name="F1",
                            url="https://example.com/f.gguf",
                            filename="f.gguf",
                        ),
                    ],
                ),
            ],
            custom_nodes=[
                CustomNode(
                    name="test-node",
                    git_url="https://github.com/test/node",
                ),
            ],
        )
        assert len(config.models) == 1
        assert len(config.custom_nodes) == 1


class TestSettings:
    """Tests for Settings model."""

    def test_default_settings(self) -> None:
        """Test default settings values."""
        settings = Settings()
        assert settings.comfyui_path == Path("/workspace/ComfyUI")
        assert settings.max_concurrent_downloads == 3
        assert settings.verify_checksums is True
        assert settings.skip_existing is True

    def test_models_path_property(self) -> None:
        """Test models_path derived property."""
        settings = Settings(comfyui_path=Path("/test/ComfyUI"))
        assert settings.models_path == Path("/test/ComfyUI/models")

    def test_custom_nodes_path_property(self) -> None:
        """Test custom_nodes_path derived property."""
        settings = Settings(comfyui_path=Path("/test/ComfyUI"))
        assert settings.custom_nodes_path == Path("/test/ComfyUI/custom_nodes")
