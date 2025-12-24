"""Deployment orchestrator - coordinates model installation, node setup, and workflow deployment."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ai_content_service.comfyui import ComfyUISetup, SetupResult
from ai_content_service.config import (
    CustomNode,
    DeploymentConfig,
    ModelDefinition,
    Settings,
    get_settings,
)
from ai_content_service.downloader import DownloadResult, ModelDownloader, RichDownloadReporter
from ai_content_service.workflows import WorkflowInfo, WorkflowManager


class DeploymentError(Exception):
    """Base exception for deployment errors."""


@dataclass
class DeploymentReport:
    """Complete deployment report."""

    models_downloaded: list[DownloadResult] = field(default_factory=list)
    custom_nodes_installed: list[SetupResult] = field(default_factory=list)
    workflows_installed: list[WorkflowInfo] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """Check if deployment was successful."""
        models_ok = all(r.success for r in self.models_downloaded)
        nodes_ok = all(r.success for r in self.custom_nodes_installed)
        return models_ok and nodes_ok and not self.errors

    @property
    def total_downloaded_bytes(self) -> int:
        """Get total bytes downloaded."""
        return sum(r.size_bytes for r in self.models_downloaded if r.success)


class Deployer:
    """Orchestrates the complete deployment process."""

    def __init__(
        self,
        settings: Settings | None = None,
        console: Console | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._console = console or Console()
        self._comfyui = ComfyUISetup(self._settings, self._console)
        self._workflow_manager = WorkflowManager(self._settings, self._console)

    def load_config(self, config_path: Path) -> DeploymentConfig:
        """Load deployment configuration from YAML file."""
        if not config_path.exists():
            raise DeploymentError(f"Configuration file not found: {config_path}")

        with Path.open(config_path) as f:
            data = yaml.safe_load(f)

        return DeploymentConfig.model_validate(data)

    async def deploy_models(
        self,
        models: list[ModelDefinition],
    ) -> list[DownloadResult]:
        """Deploy all model files."""
        all_results: list[DownloadResult] = []

        with RichDownloadReporter(self._console) as reporter:
            downloader = ModelDownloader(self._settings, reporter)

            for model in models:
                self._console.print(f"\n[bold cyan]Deploying model: {model.name}[/bold cyan]")
                if model.description:
                    self._console.print(f"[dim]{model.description}[/dim]")

                # Determine target directory
                target_dir = self._comfyui.get_model_target_path(
                    model.model_type.value,
                    model.subfolder,
                )
                target_dir.mkdir(parents=True, exist_ok=True)

                # Download all files for this model
                results = await downloader.download_model_files(model.files, target_dir)
                all_results.extend(results)

        return all_results

    async def deploy_custom_nodes(
        self,
        nodes: list[CustomNode],
    ) -> list[SetupResult]:
        """Deploy all custom nodes."""
        if not nodes:
            return []

        self._console.print("\n[bold cyan]Installing custom nodes...[/bold cyan]")
        return await self._comfyui.install_custom_nodes(nodes)

    def deploy_workflows(self, workflows_dir: Path | None = None) -> list[WorkflowInfo]:
        """Deploy workflow files."""
        source_dir = workflows_dir or self._settings.workflows_path

        if not source_dir.exists():
            self._console.print(f"[yellow]No workflows directory found at {source_dir}[/yellow]")
            return []

        self._console.print("\n[bold cyan]Installing workflows...[/bold cyan]")
        return self._workflow_manager.install_workflows_from_directory(source_dir)

    async def deploy(
        self,
        config: DeploymentConfig,
        workflows_dir: Path | None = None,
    ) -> DeploymentReport:
        """Execute complete deployment."""
        report = DeploymentReport()

        # Verify ComfyUI installation
        self._console.print(
            Panel.fit(
                "[bold]Starting Deployment[/bold]",
                border_style="cyan",
            )
        )

        if not self._comfyui.verify_installation():
            report.errors.append("ComfyUI installation not found or incomplete")
            return report

        # Ensure model directories exist
        self._comfyui.ensure_model_directories()

        # Install custom nodes first (models may depend on them)
        if config.custom_nodes:
            results = await self.deploy_custom_nodes(config.custom_nodes)
            report.custom_nodes_installed = results

        # Deploy models
        if config.models:
            results = await self.deploy_models(config.models)
            report.models_downloaded = results

        # Deploy workflows
        workflow_results = self.deploy_workflows(workflows_dir)
        report.workflows_installed = workflow_results

        # Print summary
        self._print_report(report)

        return report

    async def deploy_from_config_file(
        self,
        config_path: Path,
        workflows_dir: Path | None = None,
    ) -> DeploymentReport:
        """Deploy from a configuration file."""
        config = self.load_config(config_path)
        return await self.deploy(config, workflows_dir)

    def _print_report(self, report: DeploymentReport) -> None:
        """Print deployment summary report."""
        self._console.print("\n")

        # Models table
        if report.models_downloaded:
            table = Table(title="Model Downloads", show_header=True)
            table.add_column("File", style="cyan")
            table.add_column("Status", style="green")
            table.add_column("Size", justify="right")

            for result in report.models_downloaded:
                status = "[green]✓[/green]" if result.success else f"[red]✗ {result.error}[/red]"
                size = self._format_size(result.size_bytes) if result.success else "-"
                table.add_row(result.filename, status, size)

            self._console.print(table)

        # Custom nodes table
        if report.custom_nodes_installed:
            table = Table(title="Custom Nodes", show_header=True)
            table.add_column("Node", style="cyan")
            table.add_column("Status", style="green")
            table.add_column("Message")

            for result in report.custom_nodes_installed:
                status = "[green]✓[/green]" if result.success else "[red]✗[/red]"
                table.add_row(result.name, status, result.message)

            self._console.print(table)

        # Workflows table
        if report.workflows_installed:
            table = Table(title="Workflows", show_header=True)
            table.add_column("Name", style="cyan")
            table.add_column("Nodes", justify="right")
            table.add_column("Path")

            for info in report.workflows_installed:
                table.add_row(info.name, str(info.node_count), str(info.path))

            self._console.print(table)

        # Summary
        if report.success:
            self._console.print(
                Panel.fit(
                    f"[bold green]Deployment Successful[/bold green]\n"
                    f"Models: {len(report.models_downloaded)} | "
                    f"Nodes: {len(report.custom_nodes_installed)} | "
                    f"Workflows: {len(report.workflows_installed)}",
                    border_style="green",
                )
            )
        else:
            error_msg = "\n".join(report.errors) if report.errors else "See details above"
            self._console.print(
                Panel.fit(
                    f"[bold red]Deployment Failed[/bold red]\n{error_msg}",
                    border_style="red",
                )
            )

    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """Format bytes as human-readable size."""
        size = float(size_bytes)
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} PB"


async def quick_deploy(
    models: list[ModelDefinition],
    custom_nodes: list[CustomNode] | None = None,
    settings: Settings | None = None,
) -> DeploymentReport:
    """Convenience function for quick programmatic deployment."""
    config = DeploymentConfig(
        models=models,
        custom_nodes=custom_nodes or [],
    )
    deployer = Deployer(settings)
    return await deployer.deploy(config)
