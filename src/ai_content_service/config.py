"""Configuration models and settings for the AI content service."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field, HttpUrl, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelType(str, Enum):
    """Supported model types for ComfyUI."""

    DIFFUSION = "diffusion_models"
    UNET = "unet"
    CLIP = "clip"
    VAE = "vae"
    LORA = "loras"
    CONTROLNET = "controlnet"
    CHECKPOINT = "checkpoints"


class ModelFile(BaseModel):
    """Individual model file definition."""

    name: str = Field(..., description="Display name for the model file")
    url: Annotated[str, HttpUrl] = Field(..., description="Direct download URL")
    filename: str = Field(..., description="Target filename")
    sha256: str | None = Field(None, description="SHA256 hash for verification")
    size_bytes: int | None = Field(None, description="Expected file size in bytes")

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, v: str) -> str:
        """Ensure filename is safe and doesn't contain path traversal."""
        if ".." in v or "/" in v or "\\" in v:
            raise ValueError("Filename cannot contain path separators or '..'")
        return v


class ModelDefinition(BaseModel):
    """Complete model definition with all files and metadata."""

    name: str = Field(..., description="Model identifier")
    description: str = Field("", description="Human-readable description")
    model_type: ModelType = Field(..., description="Type determines target directory")
    files: list[ModelFile] = Field(..., min_length=1)
    custom_node_required: str | None = Field(
        None,
        description="Git URL of required custom node",
    )
    subfolder: str | None = Field(
        None,
        description="Optional subfolder within model type directory",
    )

    @property
    def target_subpath(self) -> str:
        """Get the relative path within ComfyUI models directory."""
        if self.subfolder:
            return f"{self.model_type.value}/{self.subfolder}"
        return self.model_type.value


class CustomNode(BaseModel):
    """Custom node definition for ComfyUI."""

    name: str = Field(..., description="Node identifier")
    git_url: Annotated[str, HttpUrl] = Field(..., description="Git repository URL")
    commit_sha: str | None = Field(None, description="Specific commit to checkout")
    pip_requirements: list[str] = Field(
        default_factory=list,
        description="Additional pip packages required",
    )


class WorkflowDefinition(BaseModel):
    """ComfyUI workflow definition."""

    name: str = Field(..., description="Workflow identifier")
    description: str = Field("", description="Human-readable description")
    filename: str = Field(..., description="Workflow JSON filename")
    required_models: list[str] = Field(
        default_factory=list,
        description="List of model names this workflow requires",
    )


class DeploymentConfig(BaseModel):
    """Complete deployment configuration."""

    models: list[ModelDefinition] = Field(default_factory=list)
    custom_nodes: list[CustomNode] = Field(default_factory=list)
    workflows: list[WorkflowDefinition] = Field(default_factory=list)


class Settings(BaseSettings):
    """Application settings loaded from environment."""

    model_config = SettingsConfigDict(
        env_prefix="ACS_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Paths
    comfyui_path: Path = Field(
        default=Path("/workspace/ComfyUI"),
        description="Path to ComfyUI installation",
    )
    config_path: Path = Field(
        default=Path("/workspace/config"),
        description="Path to configuration files",
    )
    workflows_path: Path = Field(
        default=Path("/workspace/workflows"),
        description="Path to workflow files",
    )

    # Download settings
    download_chunk_size: Annotated[int, Field(gt=0)] = 8 * 1024 * 1024  # 8MB chunks
    download_timeout: Annotated[int, Field(gt=0)] = 3600  # 1 hour timeout
    max_concurrent_downloads: Annotated[int, Field(gt=0, le=10)] = 3
    retry_attempts: Annotated[int, Field(ge=0, le=10)] = 3
    retry_delay: Annotated[float, Field(ge=0)] = 5.0

    # Hugging Face settings (optional, for private repos)
    hf_token: str | None = Field(None, description="Hugging Face API token")

    # Verification
    verify_checksums: bool = Field(True, description="Verify SHA256 after download")
    skip_existing: bool = Field(True, description="Skip files that already exist")

    @property
    def models_path(self) -> Path:
        """Get the ComfyUI models directory path."""
        return self.comfyui_path / "models"

    @property
    def custom_nodes_path(self) -> Path:
        """Get the ComfyUI custom_nodes directory path."""
        return self.comfyui_path / "custom_nodes"

    @property
    def user_workflows_path(self) -> Path:
        """Get the ComfyUI user workflows path."""
        return self.comfyui_path / "user"


def get_settings() -> Settings:
    """Factory function for dependency injection of settings."""
    return Settings() # type: ignore
