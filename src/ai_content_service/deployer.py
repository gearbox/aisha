"""Deployment orchestration for AI Content Service."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .comfyui import MIN_CHECKPOINT_BYTES, ExpectedArtifact, RequirementsLockMetrics
from .config import (
    BundleConfig,
    DeploymentPlan,
    DeployMode,
    ModelType,
    Settings,
    unwrap_secret,
)
from .operation_telemetry import OperationTarget, OperationTelemetry
from .provisioning_reporter import ProvisioningReporter
from .provisioning_timing import ProvisioningTimer, build_env_context
from .telemetry_contract import OperationKind, ProvisioningPhase

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
    ModelType.TEXT_ENCODERS.value: _MIN_SMALL_ARTIFACT_BYTES,
    ModelType.CLIP_VISION.value: _MIN_SMALL_ARTIFACT_BYTES,
    ModelType.STYLE_MODELS.value: _MIN_SMALL_ARTIFACT_BYTES,
    ModelType.GLIGEN.value: _MIN_SMALL_ARTIFACT_BYTES,
    ModelType.PHOTOMAKER.value: _MIN_SMALL_ARTIFACT_BYTES,
    ModelType.MODEL_PATCHES.value: _MIN_SMALL_ARTIFACT_BYTES,
    ModelType.AUDIO_ENCODERS.value: _MIN_SMALL_ARTIFACT_BYTES,
    ModelType.UNET.value: MIN_CHECKPOINT_BYTES,
    ModelType.BACKGROUND_REMOVAL.value: _MIN_SMALL_ARTIFACT_BYTES,
    ModelType.DETECTION.value: _MIN_SMALL_ARTIFACT_BYTES,
    ModelType.DIFFUSERS.value: _MIN_SMALL_ARTIFACT_BYTES,
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


def _telemetry_secrets(settings: Settings) -> tuple[str, ...]:
    """Configured secrets that must never be copied into timing JSONL errors."""
    candidates = (
        settings.hf_token,
        settings.civitai_api_token,
        settings.cf_tunnel_token,
        settings.apex_callback_token,
        settings.r2_readonly_secret_access_key,
        settings.r2_write_secret_access_key,
        settings.apex_admin_token,
        settings.github_token,
    )
    return tuple(value for secret in candidates if (value := unwrap_secret(secret)))


def _effective_mib_per_s(materialized_bytes: int, duration_s: float) -> float | None:
    """Final-materialization rate, not network throughput or decimal MB/s."""
    if duration_s <= 0:
        return None
    return round((materialized_bytes / (1024**2)) / duration_s, 3)


def _plan_snapshot(plan: DeploymentPlan) -> dict[str, object]:
    """Return the complete, plan-derived phase list used by the event start."""
    return {
        "phases": [
            {"phase": ProvisioningPhase.COMFYUI.value, "will_run": plan.will_update_comfyui},
            {
                "phase": ProvisioningPhase.REQUIREMENTS_BASE.value,
                "will_run": plan.will_install_base_requirements,
            },
            {
                "phase": ProvisioningPhase.REQUIREMENTS_LOCKED.value,
                "will_run": plan.will_install_locked_requirements,
            },
            {
                "phase": ProvisioningPhase.CUSTOM_NODES.value,
                "will_run": plan.will_install_custom_nodes,
            },
            {"phase": ProvisioningPhase.MODELS.value, "will_run": plan.will_download_models},
            {"phase": ProvisioningPhase.WORKFLOW.value, "will_run": plan.will_install_workflow},
            {"phase": ProvisioningPhase.VERIFYING.value, "will_run": plan.will_verify},
        ],
        "model_files": plan.model_files_count,
        "declared_model_bytes": plan.declared_model_bytes,
        "unknown_size_files": plan.unknown_size_files_count,
        "custom_nodes": plan.custom_nodes_count,
    }


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
    locked_requirements_delta: RequirementsLockMetrics | None = None
    custom_nodes_installed: int = 0
    custom_node_requirements: dict[str, RequirementsLockMetrics] = field(default_factory=dict)
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
        operation_id: str | None = None,
        operation_kind: OperationKind = OperationKind.BUNDLE_PROVISION,
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
        async with self._reporter.operation(
            operation_id=operation_id,
            kind=operation_kind,
            target=OperationTarget(
                bundle=bundle.metadata.name,
                bundle_version=bundle.metadata.version,
                mode=plan.mode.value,
            ),
        ) as operation:
            await operation.started(plan=_plan_snapshot(plan), message="Starting deployment")
            try:
                await self._execute_deployment(bundle, bundle_path, plan, result, timer, operation)
                # The deployment boundary ends before terminal notifications and
                # telemetry collection.  Those operations may be slow/fallible,
                # but must not inflate the deployment duration.
                timer.finish()
                await operation.succeeded(
                    summary=timer.snapshot(secrets=_telemetry_secrets(self._settings))
                )
            except Exception as e:
                result.success = False
                result.errors.append(str(e))
                outcome = "failed"
                error = str(e)
                # Preserve the original deployment failure and its duration before
                # a terminal callback has a chance to fail or block.
                timer.finish()
                log.exception("deploy.failed")
                console.print(f"\n[red]Deployment failed: {e}[/red]")
                try:
                    await operation.failed(
                        str(e),
                        summary=timer.snapshot(secrets=_telemetry_secrets(self._settings)),
                    )
                except Exception:
                    log.warning("deploy.failed_callback_failed", exc_info=True)
            finally:
                await self._write_timing(bundle, result, timer, outcome=outcome, error=error)

        self._display_result(result)
        return result

    async def _write_timing(
        self,
        bundle: BundleConfig,
        result: DeploymentResult,
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
                bundle_base_image=bundle.hardware.base_image if bundle.hardware else None,
                comfyui_source=(
                    "bundle_checkout" if result.comfyui_updated else "preexisting_unknown"
                ),
            )
            timer.record_env(env_context)
            timing_path = self._settings.provisioning_timing_path or (
                self._settings.cache_path / "provisioning-timings.jsonl"
            )
            timer.write(
                timing_path,
                outcome=outcome,
                error=error,
                secrets=_telemetry_secrets(self._settings),
            )
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
        operation_id: str | None = None,
        operation_kind: OperationKind = OperationKind.BUNDLE_PROVISION,
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
        return await self.deploy_from_path(
            bundle_path,
            mode=mode,
            verify=verify,
            dry_run=dry_run,
            operation_id=operation_id,
            operation_kind=operation_kind,
        )

    async def _execute_deployment(
        self,
        bundle: BundleConfig,
        bundle_path: Path,
        plan: DeploymentPlan,
        result: DeploymentResult,
        timer: ProvisioningTimer,
        operation: OperationTelemetry,
    ) -> None:
        """Execute deployment according to plan."""

        # Step 1: Update ComfyUI (FULL mode only)
        if plan.will_update_comfyui and bundle.comfyui:
            await operation.begin_phase(ProvisioningPhase.COMFYUI, "Updating ComfyUI")
            with (
                timer.start(ProvisioningPhase.COMFYUI),
                console.status("[bold blue]Updating ComfyUI..."),
            ):
                await self._comfyui_manager.checkout(bundle.comfyui.commit)
                result.comfyui_updated = True
                console.print("[green]✓[/green] ComfyUI updated")
        else:
            timer.mark_skipped(ProvisioningPhase.COMFYUI)

        # Step 2: Install base requirements (FULL mode only)
        if plan.will_install_base_requirements:
            await operation.begin_phase(
                ProvisioningPhase.REQUIREMENTS_BASE, "Installing base requirements"
            )
            with (
                timer.start(ProvisioningPhase.REQUIREMENTS_BASE),
                console.status("[bold blue]Installing base requirements..."),
            ):
                await self._comfyui_manager.install_base_requirements()
                result.base_requirements_installed = True
                console.print("[green]✓[/green] Base requirements installed")
        else:
            timer.mark_skipped(ProvisioningPhase.REQUIREMENTS_BASE)

        # Step 3: Install the bundle requirements overlay (FULL mode only).
        requirements_file = bundle.requirements_file()
        if plan.will_install_locked_requirements and requirements_file:
            await operation.begin_phase(
                ProvisioningPhase.REQUIREMENTS_LOCKED, "Installing locked requirements"
            )
            requirements_path = bundle_path / requirements_file
            requirements_source = (
                "overlay" if bundle.requirements_overlay_file is not None else "lock"
            )
            with (
                timer.start(ProvisioningPhase.REQUIREMENTS_LOCKED),
                console.status("[bold blue]Resolving locked requirements delta..."),
            ):
                if requirements_source == "overlay":
                    delta = await self._comfyui_manager.install_locked_requirements(
                        requirements_path, source="overlay"
                    )
                else:
                    delta = await self._comfyui_manager.install_locked_requirements(
                        requirements_path
                    )
            result.locked_requirements_delta = delta.metrics()
            timer.record_metric("requirements_locked", result.locked_requirements_delta)
            if delta.should_install:
                result.locked_requirements_installed = True
                console.print("[green]✓[/green] Locked requirements installed")
            else:
                timer.mark_skipped(ProvisioningPhase.REQUIREMENTS_LOCKED, replace_latest=True)
                console.print("[dim]○[/dim] Locked requirements already satisfied")
        else:
            timer.mark_skipped(ProvisioningPhase.REQUIREMENTS_LOCKED)

        # Step 4: Install custom nodes (FULL mode only)
        if plan.will_install_custom_nodes:
            await operation.begin_phase(
                ProvisioningPhase.CUSTOM_NODES,
                f"Installing {len(bundle.custom_nodes)} custom nodes",
            )
            with timer.start(ProvisioningPhase.CUSTOM_NODES):
                console.print(
                    f"\n[bold]Installing {len(bundle.custom_nodes)} custom nodes...[/bold]"
                )
                for node in bundle.custom_nodes:
                    with console.status(f"[bold blue]Installing {node.name}..."):
                        node_delta = await self._comfyui_manager.install_custom_node(node)
                        result.custom_nodes_installed += 1
                        if node_delta is not None:
                            result.custom_node_requirements[node.name] = node_delta.metrics()
                        console.print(f"[green]✓[/green] {node.name}")
            if result.custom_node_requirements:
                timer.record_metric("custom_node_requirements", result.custom_node_requirements)
        else:
            timer.mark_skipped(ProvisioningPhase.CUSTOM_NODES)

        # Step 5: Download models (both modes)
        if plan.will_download_models:
            await operation.begin_phase(
                ProvisioningPhase.MODELS, f"Downloading {plan.model_files_count} model files"
            )
            console.print(f"\n[bold]Downloading {plan.model_files_count} model files...[/bold]")

            with timer.start(ProvisioningPhase.MODELS):
                report = await self._model_downloader.download_all(
                    bundle.models,
                    self._settings.comfyui_path / "models",
                    on_progress=operation.progress,
                )
            effective_mib_per_s = _effective_mib_per_s(
                report.materialized_bytes, timer.duration_of(ProvisioningPhase.MODELS) or 0.0
            )
            timer.record_metric(
                "models",
                {
                    "sources": dict(report.sources),
                    "declared_bytes": report.declared_bytes,
                    "unknown_size_files": report.unknown_size_files,
                    "reused_bytes": report.reused_bytes,
                    "materialized_bytes": report.materialized_bytes,
                    "effective_mib_per_s": effective_mib_per_s,
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
            timer.mark_skipped(ProvisioningPhase.MODELS)

        # Step 6: Install workflow (both modes)
        if plan.will_install_workflow and bundle.workflow_file:
            await operation.begin_phase(ProvisioningPhase.WORKFLOW, "Installing workflow")
            with (
                timer.start(ProvisioningPhase.WORKFLOW),
                console.status("[bold blue]Installing workflow..."),
            ):
                workflow_path = bundle_path / bundle.workflow_file
                await self._workflow_manager.install(workflow_path, bundle.metadata.name)
                result.workflow_installed = True
                console.print("[green]✓[/green] Workflow installed")
        else:
            timer.mark_skipped(ProvisioningPhase.WORKFLOW)

        # Step 7: Verify (optional, both modes)
        if plan.will_verify:
            await operation.begin_phase(ProvisioningPhase.VERIFYING, "Verifying deployment")
            with (
                timer.start(ProvisioningPhase.VERIFYING),
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
            timer.mark_skipped(ProvisioningPhase.VERIFYING)

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
            f"Download {plan.model_files_count} files ({plan.declared_model_bytes} declared bytes)",
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
