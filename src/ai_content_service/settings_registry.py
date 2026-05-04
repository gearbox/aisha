"""Extended settings for bundle registry support."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import SettingsConfigDict

from .config import Settings


class RegistryConfig(BaseModel):
    """Configuration for a bundle registry."""

    name: str = Field(description="Unique name for this registry")
    type: str = Field(default="git", description="Registry type: 'git' or 'local'")

    # Git registry settings
    repo_url: str | None = Field(default=None, description="Git repository URL")
    branch: str = Field(default="main", description="Git branch to use")
    local_path: Path | None = Field(
        default=None, description="Local path for cloned repo or local registry"
    )

    # Authentication
    auth_token: str | None = Field(default=None, description="GitHub PAT or other auth token")
    ssh_key_path: Path | None = Field(default=None, description="Path to SSH private key")

    # Behavior
    auto_sync: bool = Field(default=True, description="Auto-sync on startup")


class ExtendedSettings(Settings):
    """Extended settings with bundle registry support."""

    model_config = SettingsConfigDict(
        env_prefix="ACS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_nested_delimiter="__",
    )

    # ============================================================
    # Paths
    # ============================================================
    cache_path: Path = Field(
        default=Path("/workspace/.aisha-cache"),
        description="Cache directory for cloned registries",
    )

    # ============================================================
    # Bundle Registry Configuration
    # ============================================================
    bundles_repo: str | None = Field(
        default=None,
        description="Git URL for bundles repository (e.g., https://github.com/gearbox/ai-bundles)",
    )
    bundles_branch: str = Field(
        default="main",
        description="Branch to use for bundles repository",
    )

    # ============================================================
    # Authentication
    # ============================================================
    github_token: str | None = Field(
        default=None,
        description="GitHub Personal Access Token for private repos",
    )
    github_ssh_key: Path | None = Field(
        default=None,
        description="Path to GitHub SSH private key",
    )

    # ============================================================
    # Deployment Options
    # ============================================================
    auto_sync_registries: bool = Field(
        default=True,
        description="Automatically sync registries on deploy",
    )

    def get_bundles_cache_path(self) -> Path:
        """Get the path where bundles repo will be cloned."""
        return self.cache_path / "ai-bundles"

    def has_remote_bundles(self) -> bool:
        """Check if remote bundles repository is configured."""
        return self.bundles_repo is not None


# Example .env configuration:
"""
# AI Content Service Configuration

# ComfyUI Installation
ACS_COMFYUI_PATH=/workspace/ComfyUI

# Bundle Repository (private)
ACS_BUNDLES_REPO=https://github.com/gearbox/ai-bundles
ACS_BUNDLES_BRANCH=main

# Authentication
# Option 1: GitHub PAT (for HTTPS)
ACS_GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Option 2: SSH Key (for SSH URLs)
# ACS_GITHUB_SSH_KEY=/root/.ssh/id_ed25519

# Model Download Tokens
ACS_HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
ACS_CIVITAI_API_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Performance
ACS_MAX_CONCURRENT_DOWNLOADS=5
"""
