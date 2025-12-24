"""Workflow management for ComfyUI."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.console import Console

if TYPE_CHECKING:
    from ai_content_service.config import Settings, WorkflowDefinition


class WorkflowError(Exception):
    """Base exception for workflow errors."""


class WorkflowValidationError(WorkflowError):
    """Raised when workflow JSON is invalid."""


@dataclass
class WorkflowInfo:
    """Information about an installed workflow."""

    name: str
    filename: str
    path: Path
    description: str = ""
    node_count: int = 0
    required_models: list[str] | None = None


class WorkflowManager:
    """Manages ComfyUI workflow files."""

    def __init__(
        self,
        settings: Settings,
        console: Console | None = None,
    ) -> None:
        self._settings = settings
        self._console = console or Console()
        self._workflows_source_dir = settings.workflows_path
        self._workflows_target_dir = settings.user_workflows_path / "default" / "workflows"

    def ensure_directories(self) -> None:
        """Ensure workflow directories exist."""
        self._workflows_source_dir.mkdir(parents=True, exist_ok=True)
        self._workflows_target_dir.mkdir(parents=True, exist_ok=True)

    def validate_workflow_json(self, workflow_path: Path) -> dict[str, Any]:
        """Validate and parse workflow JSON file."""
        try:
            with Path.open(workflow_path) as f:
                data = json.load(f)

            # Basic structure validation
            if not isinstance(data, dict):
                raise WorkflowValidationError("Workflow must be a JSON object")

            # Check for required ComfyUI workflow structure
            # ComfyUI workflows have nodes as numbered keys or in a "nodes" array
            has_nodes = any(key.isdigit() for key in data) or "nodes" in data

            if not has_nodes and "last_node_id" not in data:
                self._console.print(
                    "[yellow]Warning: Workflow may not have standard ComfyUI structure[/yellow]"
                )

            return data

        except json.JSONDecodeError as e:
            raise WorkflowValidationError(f"Invalid JSON: {e}") from e

    def install_workflow(
        self,
        source_path: Path,
        definition: WorkflowDefinition | None = None,
    ) -> WorkflowInfo:
        """Install a workflow file to ComfyUI."""
        if not source_path.exists():
            raise WorkflowError(f"Workflow file not found: {source_path}")

        # Validate workflow
        workflow_data = self.validate_workflow_json(source_path)

        # Determine target filename
        target_filename = definition.filename if definition else source_path.name

        target_path = self._workflows_target_dir / target_filename

        # Copy workflow file
        self.ensure_directories()
        shutil.copy2(source_path, target_path)

        # Count nodes
        node_count = 0
        if "nodes" in workflow_data:
            node_count = len(workflow_data["nodes"])
        else:
            node_count = sum(1 for k in workflow_data if k.isdigit())

        info = WorkflowInfo(
            name=definition.name if definition else source_path.stem,
            filename=target_filename,
            path=target_path,
            description=definition.description if definition else "",
            node_count=node_count,
            required_models=definition.required_models if definition else None,
        )

        self._console.print(f"[green]✓ Installed workflow: {info.name}[/green]")
        return info

    def install_workflows_from_directory(self, source_dir: Path) -> list[WorkflowInfo]:
        """Install all workflow JSON files from a directory."""
        if not source_dir.exists():
            self._console.print(f"[yellow]Workflows directory not found: {source_dir}[/yellow]")
            return []

        installed = []
        for workflow_file in sorted(source_dir.glob("*.json")):
            try:
                info = self.install_workflow(workflow_file)
                installed.append(info)
            except WorkflowError as e:
                self._console.print(f"[red]✗ Failed to install {workflow_file.name}: {e}[/red]")

        return installed

    def list_installed_workflows(self) -> list[WorkflowInfo]:
        """List all installed workflows."""
        if not self._workflows_target_dir.exists():
            return []

        workflows = []
        for workflow_file in sorted(self._workflows_target_dir.glob("*.json")):
            try:
                workflow_data = self.validate_workflow_json(workflow_file)
                node_count = 0
                if "nodes" in workflow_data:
                    node_count = len(workflow_data["nodes"])
                else:
                    node_count = sum(1 for k in workflow_data if k.isdigit())

                workflows.append(
                    WorkflowInfo(
                        name=workflow_file.stem,
                        filename=workflow_file.name,
                        path=workflow_file,
                        node_count=node_count,
                    )
                )
            except WorkflowError:
                continue

        return workflows

    def remove_workflow(self, filename: str) -> bool:
        """Remove an installed workflow."""
        target_path = self._workflows_target_dir / filename
        if target_path.exists():
            target_path.unlink()
            self._console.print(f"[green]✓ Removed workflow: {filename}[/green]")
            return True
        return False

    def export_workflow(self, filename: str, target_path: Path) -> Path:
        """Export an installed workflow to a file."""
        source_path = self._workflows_target_dir / filename
        if not source_path.exists():
            raise WorkflowError(f"Workflow not found: {filename}")

        shutil.copy2(source_path, target_path)
        return target_path
