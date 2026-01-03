"""Configuration models and settings for the AI content service."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Self

from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator
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


# =============================================================================
# Bundle Configuration Models
# =============================================================================


class BundleVersion(BaseModel):
    """Bundle version identifier in YYMMDD-nn format."""

    version: str = Field(..., pattern=r"^\d{6}-\d{2}$")

    @classmethod
    def create_new(cls, existing_versions: list[str] | None = None) -> Self:
        """Create a new version based on current date."""
        today = datetime.now(tz=timezone.utc).strftime("%y%m%d")
        existing = existing_versions or []

        # Find existing versions for today
        today_pattern = re.compile(rf"^{today}-(\d{{2}})$")
        today_numbers = []
        for v in existing:
            if match := today_pattern.match(v):
                today_numbers.append(int(match[1]))

        # Get next number
        next_num = max(today_numbers, default=0) + 1
        if next_num > 99:
            raise ValueError(f"Maximum versions per day (99) reached for {today}")

        return cls(version=f"{today}-{next_num:02d}")

    def __str__(self) -> str:
        return self.version


class ComfyUIConfig(BaseModel):
    """ComfyUI installation configuration."""

    repo: str = Field(
        default="https://github.com/comfyanonymous/ComfyUI",
        description="ComfyUI git repository URL",
    )
    commit: str = Field(..., description="Specific commit SHA to checkout")


class BundleMetadata(BaseModel):
    """Metadata for a bundle version."""

    name: str = Field(..., description="Bundle name (e.g., wan_2.2_i2v)")
    version: str = Field(..., description="Version in YYMMDD-nn format")
    description: str = Field("", description="Human-readable description")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc),
        description="Creation timestamp",
    )
    tested: bool = Field(False, description="Whether this bundle has been tested")
    author: str | None = None
    notes: str = ""


class BundleConfig(BaseModel):
    """Complete bundle configuration."""

    metadata: BundleMetadata = Field(..., description="Bundle metadata")
    comfyui: ComfyUIConfig = Field(..., description="ComfyUI configuration")
    custom_nodes: list[CustomNode] = Field(
        default_factory=list,
        description="Custom nodes to install",
    )
    models: list[ModelDefinition] = Field(
        default_factory=list,
        description="Models to download",
    )

    # These are stored as separate files but referenced here
    requirements_lock_file: str = Field(
        default="requirements.lock",
        description="Filename for pip requirements lock file",
    )
    workflow_file: str = Field(
        default="workflow.json",
        description="Filename for ComfyUI workflow",
    )
    extra_model_paths_file: str | None = Field(
        None,
        description="Optional filename for extra_model_paths.yaml",
    )

    @model_validator(mode="after")
    def validate_node_commits(self) -> Self:
        """Ensure all custom nodes have commit SHAs for reproducibility."""
        for node in self.custom_nodes:
            if not node.commit_sha:
                raise ValueError(
                    f"Custom node '{node.name}' must have commit_sha for reproducible bundles"
                )
        return self


class BundleFiles(BaseModel):
    """Represents all files in a bundle version directory."""

    bundle_config: BundleConfig
    requirements_lock: str = Field(..., description="Contents of requirements.lock")
    workflow_json: dict = Field(..., description="Parsed workflow JSON")
    extra_model_paths: str | None = Field(None, description="Contents of extra_model_paths.yaml")

    @property
    def expected_node_types(self) -> set[str]:
        """Extract expected node types from workflow."""
        node_types: set[str] = set()
        nodes = self.workflow_json.get("nodes", [])

        # Handle both array format and dict format
        if isinstance(nodes, list):
            for node in nodes:
                if isinstance(node, dict) and "type" in node:
                    node_types.add(node["type"])
        elif isinstance(nodes, dict):
            for node in nodes.values():
                if isinstance(node, dict) and "type" in node:
                    node_types.add(node["type"])

        # Also check for numbered keys (ComfyUI API format)
        for key, value in self.workflow_json.items():
            if key.isdigit() and isinstance(value, dict) and "class_type" in value:
                node_types.add(value["class_type"])

        return node_types


# =============================================================================
# Application Settings
# =============================================================================


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
    bundles_path: Path = Field(
        default=Path("config/bundles"),
        description="Path to bundles directory",
    )

    # Bundle selection
    bundle: str | None = Field(
        None,
        description="Bundle name to deploy (e.g., wan_2.2_i2v)",
    )
    bundle_version: str | None = Field(
        None,
        description="Specific bundle version (default: current symlink)",
    )

    # Download settings
    download_chunk_size: Annotated[int, Field(gt=0)] = 8 * 1024 * 1024  # 8MB chunks
    download_timeout: Annotated[int, Field(gt=0)] = 3600  # 1 hour timeout
    max_concurrent_downloads: Annotated[int, Field(gt=0, le=10)] = 3
    retry_attempts: Annotated[int, Field(ge=0, le=10)] = 3
    retry_delay: Annotated[float, Field(ge=0)] = 5.0

    # Hugging Face settings (optional, for private repos)
    hf_token: str | None = Field(None, description="Hugging Face API token")

    # Civitai settings (optional, for downloading from civitai.com)
    civitai_api_token: str | None = Field(
        None,
        description="Civitai API token for downloading models",
    )

    # Verification
    verify_checksums: bool = Field(True, description="Verify SHA256 after download")
    skip_existing: bool = Field(True, description="Skip files that already exist")
    no_verify: bool = Field(
        False,
        description="Skip ComfyUI verification after deployment",
    )

    # ComfyUI settings for verification
    comfyui_host: str = Field("127.0.0.1", description="ComfyUI host for verification")
    comfyui_port: int = Field(8188, description="ComfyUI port for verification")
    comfyui_startup_timeout: int = Field(120, description="Timeout for ComfyUI startup in seconds")

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

    @property
    def comfyui_requirements_path(self) -> Path:
        """Get the ComfyUI requirements.txt path."""
        return self.comfyui_path / "requirements.txt"


@lru_cache
def get_settings() -> Settings:
    """Factory function for dependency injection of settings."""
    return Settings()  # type: ignore[call-arg]


def get_fresh_settings() -> Settings:
    """Get settings without caching (useful for testing)."""
    return Settings()  # type: ignore[call-arg]
