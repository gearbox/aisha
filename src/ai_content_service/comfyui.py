"""ComfyUI management for AI Content Service."""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from pathlib import Path

    from .config import CustomNodeConfig


class ComfyUIError(Exception):
    """Raised when ComfyUI operations fail."""

    pass


@dataclass
class ComfyUIStatus:
    """Status information about ComfyUI installation."""

    commit: str | None
    custom_node_count: int
    is_running: bool


class ComfyUIManager:
    """Manages ComfyUI installation, updates, and verification."""

    CUSTOM_NODES_DIR = "custom_nodes"
    OBJECT_INFO_ENDPOINT = "/object_info"
    DEFAULT_PORT = 8188

    def __init__(
        self,
        comfyui_path: Path,
        port: int = DEFAULT_PORT,
    ) -> None:
        self._comfyui_path = comfyui_path
        self._port = port

    async def checkout(self, commit: str) -> None:
        """Checkout ComfyUI to specific commit."""
        if not self._comfyui_path.exists():
            raise ComfyUIError(f"ComfyUI not found at {self._comfyui_path}")

        # Fetch latest
        await self._run_git(["fetch", "--all"])

        # Checkout specific commit
        await self._run_git(["checkout", commit])

    async def install_base_requirements(self) -> None:
        """Install ComfyUI base requirements."""
        requirements_path = self._comfyui_path / "requirements.txt"
        if not requirements_path.exists():
            raise ComfyUIError("ComfyUI requirements.txt not found")

        await self._run_pip(["install", "-r", str(requirements_path)])

    async def install_locked_requirements(self, requirements_path: Path) -> None:
        """Install locked requirements from pip freeze output."""
        if not requirements_path.exists():
            raise ComfyUIError(f"Requirements file not found: {requirements_path}")

        await self._run_pip(["install", "-r", str(requirements_path)])

    async def install_custom_node(self, node: CustomNodeConfig) -> None:
        """Install or update a custom node to specific commit."""
        custom_nodes_dir = self._comfyui_path / self.CUSTOM_NODES_DIR
        custom_nodes_dir.mkdir(exist_ok=True)

        node_dir = custom_nodes_dir / node.name

        if node_dir.exists():
            # Update existing node
            await self._run_git(["fetch", "--all"], cwd=node_dir)
        else:
            # Clone new node
            await self._run_git(
                ["clone", node.git_url, node.name],
                cwd=custom_nodes_dir,
            )
        await self._run_git(["checkout", node.commit_sha], cwd=node_dir)
        # Install node requirements if present
        requirements_path = node_dir / "requirements.txt"
        if requirements_path.exists():
            await self._run_pip(["install", "-r", str(requirements_path)])

        # Install explicit pip requirements
        if node.pip_requirements:
            await self._run_pip(["install", *node.pip_requirements])

    async def verify(self, timeout: float = 60.0) -> bool:
        """Verify ComfyUI is working by checking /object_info endpoint.

        Note: Assumes ComfyUI is already running.
        """
        url = f"http://127.0.0.1:{self._port}{self.OBJECT_INFO_ENDPOINT}"

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, timeout=timeout)
                return response.status_code == 200
            except httpx.RequestError:
                return False

    async def get_status(self) -> ComfyUIStatus:
        """Get current status of ComfyUI installation."""
        commit = await self._get_current_commit()
        custom_node_count = self._count_custom_nodes()
        is_running = await self._check_running()

        return ComfyUIStatus(
            commit=commit,
            custom_node_count=custom_node_count,
            is_running=is_running,
        )

    async def _get_current_commit(self) -> str | None:
        """Get current git commit SHA."""
        if not self._comfyui_path.exists():
            return None

        with contextlib.suppress(Exception):
            result = await asyncio.create_subprocess_exec(
                "git",
                "rev-parse",
                "HEAD",
                cwd=self._comfyui_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await result.communicate()
            if result.returncode == 0:
                return stdout.decode().strip()
        return None

    def _count_custom_nodes(self) -> int:
        """Count installed custom nodes."""
        custom_nodes_dir = self._comfyui_path / self.CUSTOM_NODES_DIR
        if not custom_nodes_dir.exists():
            return 0

        return sum(
            bool(p.is_dir() and not p.name.startswith(".")) for p in custom_nodes_dir.iterdir()
        )

    async def _check_running(self) -> bool:
        """Check if ComfyUI is running."""
        url = f"http://127.0.0.1:{self._port}{self.OBJECT_INFO_ENDPOINT}"
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, timeout=5.0)
                return response.status_code == 200
            except httpx.RequestError:
                return False

    async def _run_git(
        self,
        args: list[str],
        cwd: Path | None = None,
    ) -> None:
        """Run a git command."""
        work_dir = cwd or self._comfyui_path

        result = await asyncio.create_subprocess_exec(
            "git",
            *args,
            cwd=work_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await result.communicate()

        if result.returncode != 0:
            raise ComfyUIError(
                f"Git command failed: git {' '.join(args)}\n" f"stderr: {stderr.decode()}"
            )

    async def _run_pip(self, args: list[str]) -> None:
        """Run a pip command."""
        result = await asyncio.create_subprocess_exec(
            "pip",
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await result.communicate()

        if result.returncode != 0:
            raise ComfyUIError(
                f"Pip command failed: pip {' '.join(args)}\n" f"stderr: {stderr.decode()}"
            )
