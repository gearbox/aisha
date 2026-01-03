"""Deployment orchestrator - coordinates bundle deployment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ai_content_service.bundle import BundleFiles, BundleManager
from ai_content_service.comfyui import ComfyUISetup, SetupResult, VerificationResult
from ai_content_service.config import Settings, get_settings
from ai_content_service.downloader import DownloadResult, ModelDownloader, RichDownloadReporter
from ai_content_service.workflows import WorkflowInfo, WorkflowManager

if TYPE_CHECKING:
    from pathlib import Path


class DeploymentError(Exception):
    """Base exception for deployment errors."""


@dataclass
class DeploymentReport:
    """Complete deployment report."""

    bundle_name: str = ""
    bundle_version: str = ""
    comfyui_update: SetupResult | None = None
    base_requirements: SetupResult | None = None
    requirements_lock: SetupResult | None = None
    models_downloaded: list[DownloadResult] = field(default_factory=list)
    custom_nodes_installed: list[SetupResult] = field(default_factory=list)
    workflows_installed: list[WorkflowInfo] = field(default_factory=list)
    extra_model_paths: SetupResult | None = None
    verification: VerificationResult | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """Check if deployment was successful."""
        # Check ComfyUI update
        if self.comfyui_update and not self.comfyui_update.success:
            return False

        # Check base requirements
        if self.base_requirements and not self.base_requirements.success:
            return False

        # Check requirements lock
        if self.requirements_lock and not self.requirements_lock.success:
            return False

        # Check models
        if not all(r.success for r in self.models_downloaded):
            return False

        # Check custom nodes
        if not all(r.success for r in self.custom_nodes_installed):
            return False

        # Check verification
        if self.verification and not self.verification.success:
            return False

        # Check for any errors
        return not self.errors

    @property
    def total_downloaded_bytes(self) -> int:
        """Get total bytes downloaded."""
        return sum(r.size_bytes for r in self.models_downloaded if r.success)


class BundleDeployer:
    """Orchestrates the complete bundle deployment process."""

    def __init__(
        self,
        settings: Settings | None = None,
        console: Console | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._console = console or Console()
        self._bundle_manager = BundleManager(self._settings, self._console)
        self._comfyui = ComfyUISetup(self._settings, self._console)
        self._workflow_manager = WorkflowManager(self._settings, self._console)

    async def deploy_bundle(
        self,
        bundle_name: str | None = None,
        version: str | None = None,
    ) -> DeploymentReport:
        """Deploy a complete bundle.

        Args:
            bundle_name: Bundle name or None to use settings/env
            version: Specific version or None to use current symlink

        Returns:
            DeploymentReport with deployment results
        """
        report = DeploymentReport()

        # Resolve bundle name and version
        try:
            resolved_name, resolved_version = self._bundle_manager.resolve_bundle(
                bundle_name, version
            )
            report.bundle_name = resolved_name
            report.bundle_version = resolved_version
        except Exception as e:
            report.errors.append(str(e))
            return report

        # Print header
        self._console.print(
            Panel.fit(
                f"[bold]Deploying Bundle[/bold]\n"
                f"Name: {resolved_name}\n"
                f"Version: {resolved_version}",
                border_style="cyan",
            )
        )

        # Load bundle files
        try:
            bundle_files = self._bundle_manager.load_bundle(resolved_name, resolved_version)
        except Exception as e:
            report.errors.append(f"Failed to load bundle: {e}")
            return report

        # Verify ComfyUI installation
        if not self._comfyui.verify_installation():
            report.errors.append("ComfyUI installation not found or incomplete")
            return report

        # Step 1: Update ComfyUI to target commit
        self._console.print("\n[bold]Step 1/7: Updating ComfyUI[/bold]")
        report.comfyui_update = await self._comfyui.update_to_commit(
            bundle_files.bundle_config.comfyui.commit
        )
        if not report.comfyui_update.success:
            self._console.print(f"[red]✗ {report.comfyui_update.message}[/red]")
            return report
        self._console.print(f"[green]✓ {report.comfyui_update.message}[/green]")

        # Step 2: Install ComfyUI base requirements
        self._console.print("\n[bold]Step 2/7: Installing ComfyUI base requirements[/bold]")
        report.base_requirements = await self._comfyui.install_base_requirements()
        if not report.base_requirements.success:
            self._console.print(f"[red]✗ {report.base_requirements.message}[/red]")
            return report
        self._console.print(f"[green]✓ {report.base_requirements.message}[/green]")

        # Step 3: Install requirements.lock
        self._console.print("\n[bold]Step 3/7: Installing locked requirements[/bold]")
        report.requirements_lock = await self._comfyui.install_requirements_lock(
            bundle_files.requirements_lock
        )
        if not report.requirements_lock.success:
            self._console.print(f"[red]✗ {report.requirements_lock.message}[/red]")
            return report
        self._console.print(f"[green]✓ {report.requirements_lock.message}[/green]")

        # Step 4: Install custom nodes
        self._console.print("\n[bold]Step 4/7: Installing custom nodes[/bold]")
        if bundle_files.bundle_config.custom_nodes:
            report.custom_nodes_installed = await self._comfyui.install_custom_nodes(
                bundle_files.bundle_config.custom_nodes
            )
        else:
            self._console.print("[dim]No custom nodes to install[/dim]")

        # Step 5: Download models
        self._console.print("\n[bold]Step 5/7: Downloading models[/bold]")
        if bundle_files.bundle_config.models:
            report.models_downloaded = await self._deploy_models(bundle_files)
        else:
            self._console.print("[dim]No models to download[/dim]")

        # Step 6: Install workflow and extra_model_paths
        self._console.print("\n[bold]Step 6/7: Installing workflow and config[/bold]")
        workflow_info = await self._deploy_workflow_and_config(bundle_files, resolved_name)
        if workflow_info:
            report.workflows_installed = [workflow_info]

        # Install extra_model_paths if present
        if bundle_files.extra_model_paths:
            report.extra_model_paths = self._comfyui.install_extra_model_paths(
                bundle_files.extra_model_paths
            )
            if report.extra_model_paths.success:
                self._console.print(f"[green]✓ {report.extra_model_paths.message}[/green]")
            else:
                self._console.print(f"[red]✗ {report.extra_model_paths.message}[/red]")

        # Step 7: Verification
        self._console.print("\n[bold]Step 7/7: Verification[/bold]")
        if expected_nodes := bundle_files.expected_node_types:
            report.verification = await self._comfyui.verify_nodes_available(expected_nodes)
            if report.verification.success:
                self._console.print(f"[green]✓ {report.verification.message}[/green]")
            else:
                self._console.print(f"[red]✗ {report.verification.message}[/red]")
        else:
            self._console.print("[dim]No nodes to verify (empty workflow)[/dim]")

        # Print summary
        self._print_report(report)

        return report

    async def _deploy_models(self, bundle_files: BundleFiles) -> list[DownloadResult]:
        """Deploy all model files."""
        all_results: list[DownloadResult] = []

        # Ensure model directories exist
        self._comfyui.ensure_model_directories()

        with RichDownloadReporter(self._console) as reporter:
            downloader = ModelDownloader(self._settings, reporter)

            for model in bundle_files.bundle_config.models:
                self._console.print(f"\n[cyan]Deploying model: {model.name}[/cyan]")
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

    async def _deploy_workflow_and_config(
        self,
        bundle_files: BundleFiles,
        bundle_name: str,
    ) -> WorkflowInfo | None:
        """Deploy workflow file from bundle."""
        import json
        import tempfile
        from pathlib import Path

        # Write workflow to temp file and install
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            delete=False,
        ) as f:
            json.dump(bundle_files.workflow_json, f)
            temp_path = Path(f.name)

        try:
            # Create a definition for the workflow
            from ai_content_service.workflows import WorkflowDefinition

            definition = WorkflowDefinition(
                name=f"{bundle_name}_workflow",
                description=f"Workflow from bundle {bundle_name}",
                filename=f"{bundle_name}.json",
            )

            return self._workflow_manager.install_workflow(temp_path, definition)
        finally:
            temp_path.unlink(missing_ok=True)

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

        # Summary
        if report.success:
            self._console.print(
                Panel.fit(
                    f"[bold green]Deployment Successful[/bold green]\n"
                    f"Bundle: {report.bundle_name} v{report.bundle_version}\n"
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


async def deploy_bundle(
    bundle_name: str | None = None,
    version: str | None = None,
    settings: Settings | None = None,
) -> DeploymentReport:
    """Convenience function for quick programmatic deployment."""
    deployer = BundleDeployer(settings)
    return await deployer.deploy_bundle(bundle_name, version)
