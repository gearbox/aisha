"""Workflow management for AI Content Service."""

from __future__ import annotations

import json
import shutil
from pathlib import Path


class WorkflowError(Exception):
    """Raised when workflow operations fail."""

    pass


class WorkflowManager:
    """Manages ComfyUI workflow installation."""

    USER_WORKFLOWS_DIR = "user"

    def __init__(self, comfyui_path: Path) -> None:
        self._comfyui_path = comfyui_path

    async def install(
        self,
        workflow_path: Path,
        bundle_name: str,
    ) -> Path:
        """Install workflow to ComfyUI user workflows directory.

        Args:
            workflow_path: Path to source workflow JSON.
            bundle_name: Bundle name for organizing workflows.

        Returns:
            Path to installed workflow.
        """
        if not workflow_path.exists():
            raise WorkflowError(f"Workflow not found: {workflow_path}")

        # Validate JSON
        try:
            with Path.open(workflow_path) as f:
                json.load(f)
        except json.JSONDecodeError as e:
            raise WorkflowError(f"Invalid workflow JSON: {e}") from e

        # Create user workflows directory
        user_workflows_dir = self._comfyui_path / self.USER_WORKFLOWS_DIR
        user_workflows_dir.mkdir(parents=True, exist_ok=True)

        # Copy workflow with bundle prefix
        target_name = f"{bundle_name}_{workflow_path.name}"
        target_path = user_workflows_dir / target_name

        shutil.copy2(workflow_path, target_path)

        return target_path

    def list_workflows(self) -> list[Path]:
        """List installed user workflows."""
        user_workflows_dir = self._comfyui_path / self.USER_WORKFLOWS_DIR
        if not user_workflows_dir.exists():
            return []

        return sorted(user_workflows_dir.glob("*.json"))

    def remove_workflow(self, name: str) -> None:
        """Remove a workflow by name."""
        user_workflows_dir = self._comfyui_path / self.USER_WORKFLOWS_DIR
        workflow_path = user_workflows_dir / name

        if not workflow_path.exists():
            raise WorkflowError(f"Workflow not found: {name}")

        workflow_path.unlink()
