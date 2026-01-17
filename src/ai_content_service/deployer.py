"""Deployment orchestration for AI Content Service."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import (
    BundleConfig,
    DeploymentPlan,
    DeployMode,
    Settings,
)

if TYPE_CHECKING:
    from pathlib import Path

    from .bundle import BundleManager
    from .comfyui import ComfyUIManager
    from .downloader import ModelDownloader
    from .workflows import WorkflowManager


console = Console()


class DeploymentError(Exception):
    """Raised when deployment fails."""

    pass


@dataclass
class DeploymentResult:
    """Result of a deployment operation."""

    success: bool
    plan: DeploymentPlan
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    # Step results
    comfyui_updated: bool = False
    base_requirements_installed: bool = False
    locked_requirements_installed: bool = False
    custom_nodes_installed: int = 0
    models_downloaded: int = 0
    workflow_installed: bool = False
    verification_passed: bool | None = None


class Deployer:
    """Orchestrates bundle deployment with mode-aware execution.

    Supports two deployment modes:
    - FULL: Complete deployment including ComfyUI checkout, requirements,
            custom nodes, models, and workflow.
    - MODELS_ONLY: Lightweight deployment that only downloads models and
                   installs the workflow. Use when ComfyUI is already set up.
    """

    def __init__(
        self,
        settings: Settings,
        bundle_manager: BundleManager,
        comfyui_manager: ComfyUIManager,
        model_downloader: ModelDownloader,
        workflow_manager: WorkflowManager,
    ) -> None:
        self._settings = settings
        self._bundle_manager = bundle_manager
        self._comfyui_manager = comfyui_manager
        self._model_downloader = model_downloader
        self._workflow_manager = workflow_manager

    async def deploy(
        self,
        bundle_name: str,
        version: str | None = None,
        mode: DeployMode = DeployMode.FULL,
        verify: bool = True,
        dry_run: bool = False,
    ) -> DeploymentResult:
        """Deploy a bundle with the specified mode.

        Args:
            bundle_name: Name of the bundle to deploy.
            version: Specific version or None for current.
            mode: Deployment mode (FULL or MODELS_ONLY).
            verify: Whether to verify deployment via ComfyUI.
            dry_run: If True, only show plan without executing.

        Returns:
            DeploymentResult with deployment outcome.
        """
        # Load bundle configuration
        bundle_path = self._bundle_manager.resolve_bundle_path(bundle_name, version)
        bundle = self._bundle_manager.load_bundle(bundle_path)
        _resolved_version = bundle_path.name

        # Create deployment plan
        plan = DeploymentPlan.from_bundle(bundle, mode, verify)

        # Display plan
        self._display_plan(plan)

        if dry_run:
            console.print("\n[yellow]Dry run - no changes made[/yellow]")
            return DeploymentResult(success=True, plan=plan)

        # Execute deployment
        result = DeploymentResult(success=True, plan=plan)

        try:
            await self._execute_deployment(bundle, bundle_path, plan, result)
        except Exception as e:
            result.success = False
            result.errors.append(str(e))
            console.print(f"\n[red]Deployment failed: {e}[/red]")

        self._display_result(result)
        return result

    async def _execute_deployment(
        self,
        bundle: BundleConfig,
        bundle_path: Path,
        plan: DeploymentPlan,
        result: DeploymentResult,
    ) -> None:
        """Execute deployment according to plan."""

        # Step 1: Update ComfyUI (FULL mode only)
        if plan.will_update_comfyui and bundle.comfyui:
            with console.status("[bold blue]Updating ComfyUI..."):
                await self._comfyui_manager.checkout(bundle.comfyui.commit)
                result.comfyui_updated = True
                console.print("[green]✓[/green] ComfyUI updated")

        # Step 2: Install base requirements (FULL mode only)
        if plan.will_install_base_requirements:
            with console.status("[bold blue]Installing base requirements..."):
                await self._comfyui_manager.install_base_requirements()
                result.base_requirements_installed = True
                console.print("[green]✓[/green] Base requirements installed")

        # Step 3: Install locked requirements (FULL mode only)
        if plan.will_install_locked_requirements and bundle.requirements_lock_file:
            with console.status("[bold blue]Installing locked requirements..."):
                requirements_path = bundle_path / bundle.requirements_lock_file
                await self._comfyui_manager.install_locked_requirements(requirements_path)
                result.locked_requirements_installed = True
                console.print("[green]✓[/green] Locked requirements installed")

        # Step 4: Install custom nodes (FULL mode only)
        if plan.will_install_custom_nodes:
            console.print(f"\n[bold]Installing {len(bundle.custom_nodes)} custom nodes...[/bold]")
            for node in bundle.custom_nodes:
                with console.status(f"[bold blue]Installing {node.name}..."):
                    await self._comfyui_manager.install_custom_node(node)
                    result.custom_nodes_installed += 1
                    console.print(f"[green]✓[/green] {node.name}")

        # Step 5: Download models (both modes)
        if plan.will_download_models:
            console.print(f"\n[bold]Downloading {plan.model_files_count} model files...[/bold]")
            downloaded = await self._model_downloader.download_all(
                bundle.models,
                self._settings.comfyui_path / "models",
            )
            result.models_downloaded = downloaded
            console.print(f"[green]✓[/green] {downloaded} models downloaded")

        # Step 6: Install workflow (both modes)
        if plan.will_install_workflow and bundle.workflow_file:
            with console.status("[bold blue]Installing workflow..."):
                workflow_path = bundle_path / bundle.workflow_file
                await self._workflow_manager.install(workflow_path, bundle.metadata.name)
                result.workflow_installed = True
                console.print("[green]✓[/green] Workflow installed")

        # Step 7: Verify (optional, both modes)
        if plan.will_verify:
            with console.status("[bold blue]Verifying deployment..."):
                result.verification_passed = await self._comfyui_manager.verify()
                if result.verification_passed:
                    console.print("[green]✓[/green] Verification passed")
                else:
                    result.warnings.append("Verification failed")
                    console.print("[yellow]⚠[/yellow] Verification failed")

    def _display_plan(self, plan: DeploymentPlan) -> None:
        """Display deployment plan to console."""
        mode_label = "Full Deployment" if plan.mode == DeployMode.FULL else "Models Only"
        mode_color = "green" if plan.mode == DeployMode.FULL else "cyan"

        table = Table(title=f"Deployment Plan: {plan.bundle_name} ({plan.bundle_version})")
        table.add_column("Step", style="bold")
        table.add_column("Action")
        table.add_column("Status")

        def status_icon(will_do: bool) -> str:
            return "[green]●[/green]" if will_do else "[dim]○[/dim]"

        table.add_row(
            "Mode",
            f"[{mode_color}]{mode_label}[/{mode_color}]",
            "",
        )
        table.add_row(
            "ComfyUI",
            "Checkout to pinned commit",
            status_icon(plan.will_update_comfyui),
        )
        table.add_row(
            "Base Requirements",
            "Install ComfyUI requirements.txt",
            status_icon(plan.will_install_base_requirements),
        )
        table.add_row(
            "Locked Requirements",
            "Install pip freeze overlay",
            status_icon(plan.will_install_locked_requirements),
        )
        table.add_row(
            "Custom Nodes",
            f"Install {plan.custom_nodes_count} nodes",
            status_icon(plan.will_install_custom_nodes),
        )
        table.add_row(
            "Models",
            f"Download {plan.model_files_count} files",
            status_icon(plan.will_download_models),
        )
        table.add_row(
            "Workflow",
            "Install workflow.json",
            status_icon(plan.will_install_workflow),
        )
        table.add_row(
            "Verify",
            "Check ComfyUI /object_info",
            status_icon(plan.will_verify),
        )

        console.print()
        console.print(table)
        console.print()

    def _display_result(self, result: DeploymentResult) -> None:
        """Display deployment result summary."""
        if result.success:
            console.print(
                Panel(
                    "[green]Deployment completed successfully[/green]",
                    title="Result",
                    border_style="green",
                )
            )
        else:
            error_text = "\n".join(f"• {e}" for e in result.errors)
            console.print(
                Panel(
                    f"[red]Deployment failed[/red]\n\n{error_text}",
                    title="Result",
                    border_style="red",
                )
            )

        if result.warnings:
            warning_text = "\n".join(f"• {w}" for w in result.warnings)
            console.print(f"\n[yellow]Warnings:[/yellow]\n{warning_text}")
