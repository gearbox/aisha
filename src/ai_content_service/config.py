"""Configuration models and settings for AI Content Service."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

if TYPE_CHECKING:
    from datetime import datetime


class DeployMode(str, Enum):
    """Deployment mode controlling which components are installed."""

    FULL = "full"
    """Full deployment: ComfyUI, custom nodes, requirements, models, workflow."""

    MODELS_ONLY = "models_only"
    """Models-only deployment: Only downloads models and installs workflow.

    Use this when you already have a working ComfyUI setup and just want to
    add a new workflow with its required models.
    """


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="ACS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
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
        default=None,
        description="Bundle name to deploy",
    )
    bundle_version: str | None = Field(
        default=None,
        description="Specific bundle version (default: current)",
    )

    # Authentication tokens
    hf_token: str | None = Field(
        default=None,
        description="Hugging Face API token for private/gated models",
    )
    civitai_api_token: str | None = Field(
        default=None,
        description="Civitai API token for model downloads",
    )

    # Download settings
    max_concurrent_downloads: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum number of concurrent model downloads",
    )

    # Deployment options
    no_verify: bool = Field(
        default=False,
        description="Skip ComfyUI verification after deployment",
    )
    deploy_mode: DeployMode = Field(
        default=DeployMode.FULL,
        description="Deployment mode (full or models_only)",
    )


# Bundle configuration models


class BundleMetadata(BaseModel):
    """Bundle metadata."""

    name: str
    version: str
    description: str = ""
    created_at: datetime
    tested: bool = False


class ComfyUIConfig(BaseModel):
    """ComfyUI repository configuration."""

    repo: str = "https://github.com/comfyanonymous/ComfyUI"
    commit: str


class CustomNodeConfig(BaseModel):
    """Custom node configuration."""

    name: str
    git_url: str
    commit_sha: str
    pip_requirements: list[str] = Field(default_factory=list)


class ModelFileConfig(BaseModel):
    """Individual model file configuration."""

    name: str
    url: str
    filename: str
    sha256: str | None = None
    size_bytes: int | None = None


class ModelConfig(BaseModel):
    """Model group configuration."""

    name: str
    model_type: str = Field(
        description="ComfyUI model subdirectory (e.g., 'diffusion_models', 'clip', 'vae')"
    )
    files: list[ModelFileConfig]
    subdirectory: str | None = Field(
        default=None,
        description="Optional subdirectory within model_type folder",
    )


class BundleConfig(BaseModel):
    """Complete bundle configuration."""

    metadata: BundleMetadata
    comfyui: ComfyUIConfig | None = None
    custom_nodes: list[CustomNodeConfig] = Field(default_factory=list)
    models: list[ModelConfig] = Field(default_factory=list)

    # Bundle files
    requirements_lock_file: str | None = None
    workflow_file: str | None = None
    extra_model_paths_file: str | None = None

    def get_all_model_files(self) -> list[tuple[ModelConfig, ModelFileConfig]]:
        """Get flat list of all model files with their parent config."""
        result: list[tuple[ModelConfig, ModelFileConfig]] = []
        for model in self.models:
            result.extend((model, file) for file in model.files)
        return result

    def requires_comfyui_setup(self) -> bool:
        """Check if this bundle requires ComfyUI setup (commit checkout)."""
        return self.comfyui is not None

    def requires_custom_nodes(self) -> bool:
        """Check if this bundle has custom nodes to install."""
        return len(self.custom_nodes) > 0

    def requires_models(self) -> bool:
        """Check if this bundle has models to download."""
        return len(self.models) > 0


class DeploymentPlan(BaseModel):
    """Deployment plan showing what will be installed."""

    mode: DeployMode
    bundle_name: str
    bundle_version: str

    # What will be done
    will_update_comfyui: bool = False
    will_install_base_requirements: bool = False
    will_install_locked_requirements: bool = False
    will_install_custom_nodes: bool = False
    will_download_models: bool = False
    will_install_workflow: bool = False
    will_verify: bool = False

    # Counts
    custom_nodes_count: int = 0
    models_count: int = 0
    model_files_count: int = 0

    @classmethod
    def from_bundle(
        cls,
        bundle: BundleConfig,
        mode: DeployMode,
        verify: bool = True,
    ) -> DeploymentPlan:
        """Create deployment plan from bundle config and mode."""
        model_files = bundle.get_all_model_files()

        if mode == DeployMode.FULL:
            return cls(
                mode=mode,
                bundle_name=bundle.metadata.name,
                bundle_version=bundle.metadata.version,
                will_update_comfyui=bundle.requires_comfyui_setup(),
                will_install_base_requirements=bundle.requires_comfyui_setup(),
                will_install_locked_requirements=bundle.requirements_lock_file is not None,
                will_install_custom_nodes=bundle.requires_custom_nodes(),
                will_download_models=bundle.requires_models(),
                will_install_workflow=bundle.workflow_file is not None,
                will_verify=verify,
                custom_nodes_count=len(bundle.custom_nodes),
                models_count=len(bundle.models),
                model_files_count=len(model_files),
            )
        else:  # MODELS_ONLY
            return cls(
                mode=mode,
                bundle_name=bundle.metadata.name,
                bundle_version=bundle.metadata.version,
                will_update_comfyui=False,
                will_install_base_requirements=False,
                will_install_locked_requirements=False,
                will_install_custom_nodes=False,
                will_download_models=bundle.requires_models(),
                will_install_workflow=bundle.workflow_file is not None,
                will_verify=verify,
                custom_nodes_count=0,
                models_count=len(bundle.models),
                model_files_count=len(model_files),
            )


# Singleton settings instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get application settings (singleton)."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Reset settings singleton (useful for testing)."""
    global _settings
    _settings = None
