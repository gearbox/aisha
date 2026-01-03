"""ComfyUI setup, configuration management, and verification."""

from __future__ import annotations

import asyncio
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import httpx
from rich.console import Console

if TYPE_CHECKING:
    from pathlib import Path

    from ai_content_service.config import CustomNode, Settings


class ComfyUIError(Exception):
    """Base exception for ComfyUI setup errors."""


class CustomNodeInstallError(ComfyUIError):
    """Raised when custom node installation fails."""


class ComfyUIVerificationError(ComfyUIError):
    """Raised when ComfyUI verification fails."""


@dataclass
class SetupResult:
    """Result of a setup operation."""

    name: str
    success: bool
    message: str


@dataclass
class VerificationResult:
    """Result of ComfyUI verification."""

    success: bool
    available_nodes: set[str]
    missing_nodes: set[str]
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

    def get_current_commit(self) -> str | None:
        """Get current ComfyUI git commit SHA."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self._settings.comfyui_path,
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return None

    async def update_to_commit(self, target_commit: str) -> SetupResult:
        """Update ComfyUI to a specific commit.

        Args:
            target_commit: Git commit SHA to checkout

        Returns:
            SetupResult indicating success or failure
        """
        comfyui_path = self._settings.comfyui_path
        current_commit = self.get_current_commit()

        if current_commit and current_commit.startswith(target_commit):
            return SetupResult(
                name="ComfyUI",
                success=True,
                message=f"Already at commit {target_commit[:8]}",
            )

        self._console.print(f"[cyan]Updating ComfyUI to commit {target_commit[:8]}...[/cyan]")

        try:
            # Fetch all commits
            await asyncio.to_thread(
                subprocess.run,
                ["git", "fetch", "--all"],
                cwd=comfyui_path,
                capture_output=True,
                check=True,
            )

            # Checkout specific commit
            await asyncio.to_thread(
                subprocess.run,
                ["git", "checkout", target_commit],
                cwd=comfyui_path,
                capture_output=True,
                check=True,
            )

            return SetupResult(
                name="ComfyUI",
                success=True,
                message=f"Updated to commit {target_commit[:8]}",
            )

        except subprocess.CalledProcessError as e:
            return SetupResult(
                name="ComfyUI",
                success=False,
                message=f"Failed to update: {e.stderr.decode() if e.stderr else str(e)}",
            )

    async def install_base_requirements(self) -> SetupResult:
        """Install ComfyUI base requirements.txt."""
        requirements_path = self._settings.comfyui_requirements_path

        if not requirements_path.exists():
            return SetupResult(
                name="ComfyUI Requirements",
                success=False,
                message=f"requirements.txt not found at {requirements_path}",
            )

        self._console.print("[cyan]Installing ComfyUI base requirements...[/cyan]")

        try:
            await asyncio.to_thread(
                subprocess.run,
                [sys.executable, "-m", "pip", "install", "-r", str(requirements_path)],
                capture_output=True,
                check=True,
            )

            return SetupResult(
                name="ComfyUI Requirements",
                success=True,
                message="Base requirements installed",
            )

        except subprocess.CalledProcessError as e:
            return SetupResult(
                name="ComfyUI Requirements",
                success=False,
                message=f"Failed to install: {e.stderr.decode() if e.stderr else str(e)}",
            )

    async def install_requirements_lock(self, requirements_content: str) -> SetupResult:
        """Install requirements from lock file content.

        Args:
            requirements_content: Contents of requirements.lock file

        Returns:
            SetupResult indicating success or failure
        """
        self._console.print("[cyan]Installing locked requirements...[/cyan]")

        # Write to temporary file
        temp_requirements = self._settings.comfyui_path / "requirements.lock.tmp"

        try:
            temp_requirements.write_text(requirements_content)

            await asyncio.to_thread(
                subprocess.run,
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "-r",
                    str(temp_requirements),
                    "--no-deps",  # Don't install dependencies, lock file is complete
                ],
                capture_output=True,
                check=True,
            )

            return SetupResult(
                name="Requirements Lock",
                success=True,
                message="Locked requirements installed",
            )

        except subprocess.CalledProcessError as e:
            return SetupResult(
                name="Requirements Lock",
                success=False,
                message=f"Failed to install: {e.stderr.decode() if e.stderr else str(e)}",
            )

        finally:
            temp_requirements.unlink(missing_ok=True)

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

        # Check if already installed at correct commit
        if node_path.exists():
            if node.commit_sha:
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
                    self._console.print(
                        f"[cyan]Updating {node.name} to commit {node.commit_sha[:8]}...[/cyan]"
                    )

                    await asyncio.to_thread(
                        subprocess.run,
                        ["git", "fetch", "--all"],
                        cwd=node_path,
                        check=True,
                        capture_output=True,
                    )

                    await asyncio.to_thread(
                        subprocess.run,
                        ["git", "checkout", node.commit_sha],
                        cwd=node_path,
                        check=True,
                        capture_output=True,
                    )

                    # Reinstall requirements after checkout
                    requirements_file = node_path / "requirements.txt"
                    if requirements_file.exists():
                        await asyncio.to_thread(
                            subprocess.run,
                            [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
                            check=True,
                            capture_output=True,
                        )

                    return SetupResult(
                        name=node.name,
                        success=True,
                        message=f"Updated to commit {node.commit_sha[:8]}",
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
                    [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
                    check=True,
                    capture_output=True,
                )

            # Install additional pip requirements if specified
            if node.pip_requirements:
                await asyncio.to_thread(
                    subprocess.run,
                    [sys.executable, "-m", "pip", "install", *node.pip_requirements],
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

    def install_extra_model_paths(self, content: str) -> SetupResult:
        """Install extra_model_paths.yaml configuration.

        Args:
            content: YAML content for extra_model_paths.yaml

        Returns:
            SetupResult indicating success or failure
        """
        target_path = self._settings.comfyui_path / "extra_model_paths.yaml"

        try:
            target_path.write_text(content)
            return SetupResult(
                name="extra_model_paths.yaml",
                success=True,
                message=f"Installed at {target_path}",
            )
        except OSError as e:
            return SetupResult(
                name="extra_model_paths.yaml",
                success=False,
                message=f"Failed to write: {e}",
            )

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

    # =========================================================================
    # Verification via /object_info
    # =========================================================================

    async def verify_nodes_available(
        self,
        expected_nodes: set[str],
    ) -> VerificationResult:
        """Verify that expected node types are available in ComfyUI.

        Starts ComfyUI, queries /object_info, and checks for expected nodes.

        Args:
            expected_nodes: Set of node type names expected to be available

        Returns:
            VerificationResult with available and missing nodes
        """
        if self._settings.no_verify:
            return VerificationResult(
                success=True,
                available_nodes=set(),
                missing_nodes=set(),
                message="Verification skipped (NO_VERIFY=true)",
            )

        self._console.print("[cyan]Starting ComfyUI for verification...[/cyan]")

        process: subprocess.Popen | None = None

        try:
            # Start ComfyUI
            process = subprocess.Popen(
                [
                    sys.executable,
                    "main.py",
                    "--listen",
                    self._settings.comfyui_host,
                    "--port",
                    str(self._settings.comfyui_port),
                    "--dont-print-server",
                ],
                cwd=self._settings.comfyui_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            # Wait for ComfyUI to be ready
            base_url = f"http://{self._settings.comfyui_host}:{self._settings.comfyui_port}"
            ready = await self._wait_for_comfyui(base_url)

            if not ready:
                return VerificationResult(
                    success=False,
                    available_nodes=set(),
                    missing_nodes=expected_nodes,
                    message=f"ComfyUI failed to start within {self._settings.comfyui_startup_timeout}s",
                )

            # Query /object_info
            available_nodes = await self._get_object_info(base_url)

            # Check for missing nodes
            missing_nodes = expected_nodes - available_nodes

            if missing_nodes:
                return VerificationResult(
                    success=False,
                    available_nodes=available_nodes,
                    missing_nodes=missing_nodes,
                    message=f"Missing {len(missing_nodes)} node(s): {', '.join(sorted(missing_nodes))}",
                )

            return VerificationResult(
                success=True,
                available_nodes=available_nodes,
                missing_nodes=set(),
                message=f"All {len(expected_nodes)} expected nodes available",
            )

        finally:
            # Cleanup: stop ComfyUI
            if process:
                self._console.print("[dim]Stopping ComfyUI verification server...[/dim]")
                try:
                    process.send_signal(signal.SIGTERM)
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()

    async def _wait_for_comfyui(self, base_url: str) -> bool:
        """Wait for ComfyUI to be ready to accept requests."""
        timeout = self._settings.comfyui_startup_timeout
        start_time = time.monotonic()

        async with httpx.AsyncClient() as client:
            while time.monotonic() - start_time < timeout:
                try:
                    response = await client.get(
                        f"{base_url}/system_stats",
                        timeout=5.0,
                    )
                    if response.status_code == 200:
                        self._console.print("[green]ComfyUI is ready[/green]")
                        return True
                except (httpx.RequestError, httpx.HTTPStatusError):
                    pass

                await asyncio.sleep(2)

        return False

    async def _get_object_info(self, base_url: str) -> set[str]:
        """Get available node types from ComfyUI /object_info endpoint."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{base_url}/object_info",
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()

            # object_info returns a dict with node type names as keys
            return set(data.keys())
