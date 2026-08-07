"""Deployment orchestration for AI Content Service."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .comfyui import MIN_CHECKPOINT_BYTES, ExpectedArtifact
from .config import (
    BundleConfig,
    DeploymentPlan,
    DeployMode,
    ModelType,
    Settings,
)
from .provisioning_reporter import ProvisioningReporter
from .provisioning_timing import PhaseId, ProvisioningTimer, build_env_context

if TYPE_CHECKING:
    from .bundle import BundleManager
    from .comfyui import ComfyUIManager
    from .downloader import ModelDownloader
    from .workflows import WorkflowManager


console = Console()
log = structlog.get_logger()

_MIN_EMBEDDING_BYTES = 1024  # Textual-inversion embeddings are often only a few KB
_MIN_SMALL_ARTIFACT_BYTES = 1 * 1024 * 1024  # 1 MB — floor for lightweight artifacts
_MIN_CONTROLNET_BYTES = 10 * 1024 * 1024

_MIN_BYTES_BY_MODEL_TYPE: dict[str, int] = {
    ModelType.CHECKPOINTS.value: MIN_CHECKPOINT_BYTES,
    ModelType.DIFFUSION.value: MIN_CHECKPOINT_BYTES,
    ModelType.CONTROLNET.value: _MIN_CONTROLNET_BYTES,
    ModelType.LORA.value: _MIN_SMALL_ARTIFACT_BYTES,
    ModelType.VAE.value: _MIN_SMALL_ARTIFACT_BYTES,
    ModelType.CLIP.value: _MIN_SMALL_ARTIFACT_BYTES,
    ModelType.UPSCALE.value: _MIN_SMALL_ARTIFACT_BYTES,
    ModelType.EMBEDDINGS.value: _MIN_EMBEDDING_BYTES,
}


def _verification_floor(model_type: str, declared: int | None) -> int:
    """Floor for ``verify``, bounded by any declared size.

    ``min()`` is load-bearing: a declared size may only relax the floor, never
    raise it. ``size_bytes`` is never itself a pass/fail threshold. A stale-high
    declaration therefore cannot fail a correct deployment, while a truthfully
    small artefact still passes.
    """
    floor = _MIN_BYTES_BY_MODEL_TYPE.get(model_type, MIN_CHECKPOINT_BYTES)
    return min(declared, floor) if declared is not None and declared > 0 else floor


class DeploymentError(Exception):
    """Raised when deployment fails."""


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
        reporter: ProvisioningReporter | None = None,
    ) -> None:
        self._settings = settings
        self._bundle_manager = bundle_manager
        self._comfyui_manager = comfyui_manager
        self._model_downloader = model_downloader
        self._workflow_manager = workflow_manager
        self._reporter: ProvisioningReporter = reporter or ProvisioningReporter.disabled()

    async def deploy_from_path(
        self,
        bundle_path: Path,
        *,
        mode: DeployMode = DeployMode.FULL,
        verify: bool = True,
        dry_run: bool = False,
    ) -> DeploymentResult:
        """Deploy a bundle from a pre-resolved path."""
        bundle = self._bundle_manager.load_bundle_config_from_path(bundle_path)
        plan = DeploymentPlan.from_bundle(bundle, mode, verify)
        self._display_plan(plan)

        if dry_run:
            console.print("\n[yellow]Dry run - no changes made[/yellow]")
            return DeploymentResult(success=True, plan=plan)

        result = DeploymentResult(success=True, plan=plan)
        timer = ProvisioningTimer()
        timer.record("bundle", bundle.metadata.name)
        timer.record("bundle_version", bundle.metadata.version)
        timer.record("mode", plan.mode.value)
        outcome = "ready"
        error: str | None = None
        try:
            await self._execute_deployment(bundle, bundle_path, plan, result, timer)
            await self._reporter.ready()
        except Exception as e:
            result.success = False
            result.errors.append(str(e))
            log.exception("deploy.failed")
            console.print(f"\n[red]Deployment failed: {e}[/red]")
            await self._reporter.failed(str(e))
            outcome = "failed"
            error = str(e)
        finally:
            await self._write_timing(bundle, plan, timer, outcome=outcome, error=error)

        self._display_result(result)
        return result

    async def _write_timing(
        self,
        bundle: BundleConfig,
        plan: DeploymentPlan,
        timer: ProvisioningTimer,
        *,
        outcome: str,
        error: str | None,
    ) -> None:
        """Record environment provenance and flush the timing record (B-L5).

        Instrumentation only: any failure here -- gathering the environment,
        writing the file -- is logged at WARNING and never allowed to turn a
        successful deployment into a failed one.
        """
        if not self._settings.provisioning_timing_enabled:
            return
        try:
            env_context = await asyncio.to_thread(
                build_env_context,
                self._settings,
                base_image=bundle.hardware.base_image if bundle.hardware else None,
                comfyui_source="bundle" if plan.will_update_comfyui else "image",
            )
            timer.record("env", env_context)
            timing_path = self._settings.provisioning_timing_path or (
                self._settings.cache_path / "provisioning-timings.jsonl"
            )
            timer.write(timing_path, outcome=outcome, error=error)
        except Exception:
            log.warning("provisioning_timing.failed", exc_info=True)

    async def deploy(
        self,
        bundle_name: str,
        version: str | None = None,
        *,
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
        bundle_path = self._bundle_manager.resolve_bundle_path(bundle_name, version)
        return await self.deploy_from_path(bundle_path, mode=mode, verify=verify, dry_run=dry_run)

    async def _execute_deployment(
        self,
        bundle: BundleConfig,
        bundle_path: Path,
        plan: DeploymentPlan,
        result: DeploymentResult,
        timer: ProvisioningTimer,
    ) -> None:
        """Execute deployment according to plan."""

        # Step 1: Update ComfyUI (FULL mode only)
        if plan.will_update_comfyui and bundle.comfyui:
            await self._reporter.phase("comfyui", "Updating ComfyUI")
            with timer.start(PhaseId.COMFYUI), console.status("[bold blue]Updating ComfyUI..."):
                await self._comfyui_manager.checkout(bundle.comfyui.commit)
                result.comfyui_updated = True
                console.print("[green]✓[/green] ComfyUI updated")
        else:
            timer.mark_skipped(PhaseId.COMFYUI)

        # Step 2: Install base requirements (FULL mode only)
        if plan.will_install_base_requirements:
            await self._reporter.phase("requirements_base", "Installing base requirements")
            with (
                timer.start(PhaseId.REQUIREMENTS_BASE),
                console.status("[bold blue]Installing base requirements..."),
            ):
                await self._comfyui_manager.install_base_requirements()
                result.base_requirements_installed = True
                console.print("[green]✓[/green] Base requirements installed")
        else:
            timer.mark_skipped(PhaseId.REQUIREMENTS_BASE)

        # Step 3: Install locked requirements (FULL mode only)
        if plan.will_install_locked_requirements and bundle.requirements_lock_file:
            await self._reporter.phase("requirements_locked", "Installing locked requirements")
            with (
                timer.start(PhaseId.REQUIREMENTS_LOCKED),
                console.status("[bold blue]Installing locked requirements..."),
            ):
                requirements_path = bundle_path / bundle.requirements_lock_file
                await self._comfyui_manager.install_locked_requirements(requirements_path)
                result.locked_requirements_installed = True
                console.print("[green]✓[/green] Locked requirements installed")
        else:
            timer.mark_skipped(PhaseId.REQUIREMENTS_LOCKED)

        # Step 4: Install custom nodes (FULL mode only)
        if plan.will_install_custom_nodes:
            await self._reporter.phase(
                "custom_nodes", f"Installing {len(bundle.custom_nodes)} custom nodes"
            )
            with timer.start(PhaseId.CUSTOM_NODES):
                console.print(
                    f"\n[bold]Installing {len(bundle.custom_nodes)} custom nodes...[/bold]"
                )
                for node in bundle.custom_nodes:
                    with console.status(f"[bold blue]Installing {node.name}..."):
                        await self._comfyui_manager.install_custom_node(node)
                        result.custom_nodes_installed += 1
                        console.print(f"[green]✓[/green] {node.name}")
        else:
            timer.mark_skipped(PhaseId.CUSTOM_NODES)

        # Step 5: Download models (both modes)
        if plan.will_download_models:
            await self._reporter.phase(
                "downloading", f"Downloading {plan.model_files_count} model files"
            )
            console.print(f"\n[bold]Downloading {plan.model_files_count} model files...[/bold]")

            models_bytes_total = 0

            async def _on_progress(
                bytes_done: int, bytes_total: int, files_done: int, files_total: int
            ) -> None:
                nonlocal models_bytes_total
                models_bytes_total = bytes_total
                await self._reporter.download_progress(
                    bytes_done, bytes_total, files_done, files_total
                )

            models_started = time.monotonic()
            with timer.start(PhaseId.MODELS):
                report = await self._model_downloader.download_all(
                    bundle.models,
                    self._settings.comfyui_path / "models",
                    on_progress=_on_progress,
                )
            models_duration = time.monotonic() - models_started
            mbps = (
                round((models_bytes_total / 1024 / 1024) / models_duration, 1)
                if models_duration > 0
                else None
            )
            timer.record(
                "models",
                {
                    "sources": dict(report.sources),
                    "bytes_total": models_bytes_total,
                    "mbps": mbps,
                },
            )
            result.models_downloaded = report.succeeded
            if not report.ok:
                detail = "; ".join(f"{f.filename}: {f.reason}" for f in report.failed)
                msg = f"{len(report.failed)}/{plan.model_files_count} model files failed: {detail}"
                console.print(f"[red]✗[/red] {msg}")
                raise DeploymentError(msg)
            console.print(f"[green]✓[/green] {report.succeeded} models downloaded")
        else:
            timer.mark_skipped(PhaseId.MODELS)

        # Step 6: Install workflow (both modes)
        if plan.will_install_workflow and bundle.workflow_file:
            await self._reporter.phase("workflow", "Installing workflow")
            with (
                timer.start(PhaseId.WORKFLOW),
                console.status("[bold blue]Installing workflow..."),
            ):
                workflow_path = bundle_path / bundle.workflow_file
                await self._workflow_manager.install(workflow_path, bundle.metadata.name)
                result.workflow_installed = True
                console.print("[green]✓[/green] Workflow installed")
        else:
            timer.mark_skipped(PhaseId.WORKFLOW)

        # Step 7: Verify (optional, both modes)
        if plan.will_verify:
            await self._reporter.phase("verifying", "Verifying deployment")
            with (
                timer.start(PhaseId.VERIFYING),
                console.status("[bold blue]Verifying deployment..."),
            ):
                expected = [
                    ExpectedArtifact(
                        relative_path=Path(model.target_subpath) / file.filename,
                        min_bytes=_verification_floor(model.model_type, file.size_bytes),
                        declared_bytes=file.size_bytes,
                    )
                    for model in bundle.models
                    for file in model.files
                ]
                problems = await self._comfyui_manager.verify(expected=expected)
                if problems:
                    detail = "; ".join(problems)
                    console.print(f"[red]✗[/red] Verification failed: {detail}")
                    raise DeploymentError(f"deployment verification failed: {detail}")
                result.verification_passed = True
                console.print("[green]✓[/green] Verification passed")
        else:
            timer.mark_skipped(PhaseId.VERIFYING)

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
            "Check all model files on disk",
            status_icon(plan.will_verify),
        )
        if plan.missing_url_files_count:
            table.add_row(
                "Missing URLs",
                f"{plan.missing_url_files_count} model file(s) have no source URL",
                "[red]✗[/red]",
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
