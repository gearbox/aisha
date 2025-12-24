"""ComfyUI setup and configuration management."""

from __future__ import annotations

import asyncio
import shutil
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rich.console import Console

if TYPE_CHECKING:
    from pathlib import Path

    from ai_content_service.config import CustomNode, Settings


class ComfyUIError(Exception):
    """Base exception for ComfyUI setup errors."""


class CustomNodeInstallError(ComfyUIError):
    """Raised when custom node installation fails."""


@dataclass
class SetupResult:
    """Result of a setup operation."""

    name: str
    success: bool
    message: str


class ComfyUISetup:
    """Handles ComfyUI environment setup and configuration."""

    def __init__(
        self,
        settings: Settings,
        console: Console | None = None,
    ) -> None:
        self._settings = settings
        self._console = console or Console()

    def verify_installation(self) -> bool:
        """Verify ComfyUI is properly installed."""
        required_paths = [
            self._settings.comfyui_path,
            self._settings.comfyui_path / "main.py",
            self._settings.models_path,
            self._settings.custom_nodes_path,
        ]
        for path in required_paths:
            if not path.exists():
                self._console.print(f"[red]Missing required path: {path}[/red]")
                return False
        return True

    def ensure_model_directories(self) -> list[Path]:
        """Ensure all model type directories exist."""
        from ai_content_service.config import ModelType

        created_dirs: list[Path] = []
        for model_type in ModelType:
            dir_path = self._settings.models_path / model_type.value
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
                created_dirs.append(dir_path)
                self._console.print(f"[green]Created directory: {dir_path}[/green]")
        return created_dirs

    async def install_custom_node(self, node: CustomNode) -> SetupResult:
        """Install a custom node from git repository."""
        node_path = self._settings.custom_nodes_path / node.name

        # Check if already installed
        if node_path.exists():
            if node.commit_sha:
                # Verify correct commit
                try:
                    result = await asyncio.to_thread(
                        subprocess.run,
                        ["git", "rev-parse", "HEAD"],
                        cwd=node_path,
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    current_sha = result.stdout.strip()
                    if current_sha.startswith(node.commit_sha) or node.commit_sha.startswith(
                        current_sha
                    ):
                        return SetupResult(
                            name=node.name,
                            success=True,
                            message=f"Already installed at commit {current_sha[:8]}",
                        )
                    # Wrong commit, need to checkout
                    await asyncio.to_thread(
                        subprocess.run,
                        ["git", "fetch", "--all"],
                        cwd=node_path,
                        check=True,
                    )
                    await asyncio.to_thread(
                        subprocess.run,
                        ["git", "checkout", node.commit_sha],
                        cwd=node_path,
                        check=True,
                    )
                except subprocess.CalledProcessError as e:
                    return SetupResult(
                        name=node.name,
                        success=False,
                        message=f"Failed to verify/update commit: {e}",
                    )
            else:
                return SetupResult(
                    name=node.name,
                    success=True,
                    message="Already installed",
                )

        try:
            # Clone repository
            self._console.print(f"[cyan]Cloning {node.name}...[/cyan]")
            await asyncio.to_thread(
                subprocess.run,
                ["git", "clone", str(node.git_url), str(node_path)],
                check=True,
                capture_output=True,
            )

            # Checkout specific commit if specified
            if node.commit_sha:
                await asyncio.to_thread(
                    subprocess.run,
                    ["git", "checkout", node.commit_sha],
                    cwd=node_path,
                    check=True,
                    capture_output=True,
                )

            # Install pip requirements if present
            requirements_file = node_path / "requirements.txt"
            if requirements_file.exists():
                self._console.print(f"[cyan]Installing requirements for {node.name}...[/cyan]")
                await asyncio.to_thread(
                    subprocess.run,
                    ["pip", "install", "-r", str(requirements_file)],
                    check=True,
                    capture_output=True,
                )

            # Install additional pip requirements if specified
            if node.pip_requirements:
                await asyncio.to_thread(
                    subprocess.run,
                    ["pip", "install", *node.pip_requirements],
                    check=True,
                    capture_output=True,
                )

            return SetupResult(
                name=node.name,
                success=True,
                message="Installed successfully",
            )

        except subprocess.CalledProcessError as e:
            # Cleanup on failure
            if node_path.exists():
                shutil.rmtree(node_path)
            return SetupResult(
                name=node.name,
                success=False,
                message=f"Installation failed: {e.stderr.decode() if e.stderr else str(e)}",
            )

    async def install_custom_nodes(
        self,
        nodes: list[CustomNode],
    ) -> list[SetupResult]:
        """Install multiple custom nodes."""
        results = []
        for node in nodes:
            result = await self.install_custom_node(node)
            results.append(result)
            if result.success:
                self._console.print(f"[green]✓ {node.name}: {result.message}[/green]")
            else:
                self._console.print(f"[red]✗ {node.name}: {result.message}[/red]")
        return results

    def get_model_target_path(self, model_type: str, subfolder: str | None = None) -> Path:
        """Get the target path for a model type."""
        base_path = self._settings.models_path / model_type
        if subfolder:
            return base_path / subfolder
        return base_path

    def list_installed_models(self) -> dict[str, list[str]]:
        """List all installed models by type."""
        from ai_content_service.config import ModelType

        installed: dict[str, list[str]] = {}
        for model_type in ModelType:
            type_path = self._settings.models_path / model_type.value
            if type_path.exists():
                files = [
                    f.name
                    for f in type_path.rglob("*")
                    if f.is_file() and not f.name.startswith(".")
                ]
                if files:
                    installed[model_type.value] = sorted(files)
        return installed

    def list_installed_custom_nodes(self) -> list[str]:
        """List all installed custom nodes."""
        if not self._settings.custom_nodes_path.exists():
            return []
        return sorted(
            d.name
            for d in self._settings.custom_nodes_path.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )
