"""CLI for AI Content Service."""

import asyncio
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import typer
import yaml
from pydantic import ValidationError
from rich import print as rprint
from rich.console import Console
from rich.table import Table

from . import cache_service
from .bundle_contract import ContractReport, Severity
from .bundle_contract_service import (
    BundleContractServiceError,
    EmptyBundleRegistryError,
    validate_bundle_contracts,
)
from .bundle_registry import BundleReference, BundleRegistry, BundleRegistryManager
from .bundle_resolution import (
    BundleResolutionError,
    ResolvedBundle,
    parse_bundle_reference,
    resolve_bundle,
)
from .cache_credentials import (
    ApexCacheCredentialProvider,
    CacheCredentialProvider,
    StaticCacheCredentialProvider,
)
from .cache_workflows import CacheWorkflowError, resolve_cache_targets, verify_cache_targets
from .config import (
    BundleConfig,
    DeployMode,
    Settings,
    get_settings,
    unwrap_secret,
)
from .logging_config import configure_logging
from .models_service import ModelFetchDownloadError, ModelsServiceError, fetch_model
from .r2_transfer import write_creds_from_settings
from .registry_service import create_registry_manager, get_or_default_registry
from .snapshot import CarryForwardReport

app = typer.Typer(
    name="acs",
    help="AI Content Service - Bundle-based deployment automation",
    no_args_is_help=True,
)
bundle_app = typer.Typer(
    name="bundle",
    help="Bundle management commands",
    no_args_is_help=True,
)
cache_app = typer.Typer(
    name="cache",
    help="Model weight cache management",
    no_args_is_help=True,
)
models_app = typer.Typer(
    name="models",
    help="Model bundle validation commands",
    no_args_is_help=True,
)
timings_app = typer.Typer(
    name="timings",
    help="Provisioning phase timing telemetry",
    no_args_is_help=True,
)
app.add_typer(bundle_app)
app.add_typer(cache_app)
app.add_typer(models_app)
app.add_typer(timings_app)

console = Console()


def _create_manager(
    settings: Settings, bundles_path: Path | None
) -> tuple[Settings, BundleRegistryManager]:
    """Apply the command-local bundles path override and build its manager."""
    if bundles_path:
        settings = settings.model_copy(update={"bundles_path": bundles_path})
    return settings, create_registry_manager(settings)


def _resolve_registry(manager: BundleRegistryManager, registry: str | None) -> BundleRegistry:
    """Resolve a named registry or the configured default."""
    resolved = manager.get(registry) if registry else manager.default
    if resolved is None:
        if registry:
            raise ValueError(f"Registry '{registry}' not found")
        raise ValueError("No default registry configured")
    return resolved


def version_callback(value: bool) -> None:
    """Show version and exit."""
    if value:
        from . import __version__

        console.print(f"acs version {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    _version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-v",
            help="Show version and exit.",
            callback=version_callback,
            is_eager=True,
        ),
    ] = False,
) -> None:
    """AI Content Service CLI."""
    if "--help" in sys.argv[1:]:
        # A subcommand's own --help is handled by Typer/Click after this
        # callback returns; settings/logging must stay untouched so --help
        # works even with a misconfigured environment.
        return

    try:
        settings = get_settings()
        configure_logging(settings.log_format, settings.log_level)
    except ValueError as e:
        console.print(f"[red]Error:[/red] Invalid configuration:\n{e}")
        raise typer.Exit(1) from e


@app.command()
def deploy(
    bundle: Annotated[
        str | None,
        typer.Option(
            "--bundle",
            "-b",
            help="Bundle reference (name, registry/name, or name:version). Falls back to ACS_BUNDLE env var.",
        ),
    ] = None,
    bundle_version: Annotated[
        str | None,
        typer.Option(
            "--bundle-version",
            "-V",
            help="Specific bundle version. Overrides a version embedded in --bundle.",
        ),
    ] = None,
    models_only: Annotated[
        bool,
        typer.Option("--models-only", "-m", help="Only deploy models and workflow"),
    ] = False,
    no_verify: Annotated[
        bool,
        typer.Option("--no-verify", help="Skip deployment verification"),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", "-n", help="Show deployment plan without executing"),
    ] = False,
    sync: Annotated[
        bool | None,
        typer.Option(
            "--sync/--no-sync",
            help="Sync registries before deploy (default: ACS_AUTO_SYNC_REGISTRIES)",
        ),
    ] = None,
    comfyui_path: Annotated[
        Path | None,
        typer.Option("--comfyui", "-c", help="Path to ComfyUI installation"),
    ] = None,
    bundles_path: Annotated[
        Path | None,
        typer.Option("--bundles-path", help="Override local bundles path"),
    ] = None,
    bundles_repo: Annotated[
        str | None,
        typer.Option("--bundles-repo", help="Override bundles repository URL"),
    ] = None,
) -> None:
    """Deploy a bundle from registry.

    Resolves the bundle reference via configured registries (remote git or
    local) and runs the full deployment pipeline.

    Examples:

        # Deploy latest from default registry (uses ACS_BUNDLE env var)
        acs deploy

        # Explicit bundle from default registry
        acs deploy -b wan_2.2_i2v

        # Pin a specific version
        acs deploy -b wan_2.2_i2v:260103-01

        # Deploy from a named registry
        acs deploy -b remote/wan_2.2_i2v

        # Models-only (skip ComfyUI setup and custom nodes)
        acs deploy -b wan_2.2_i2v --models-only

        # Dry run — show plan without executing
        acs deploy -b wan_2.2_i2v --dry-run
    """
    settings = get_settings()

    bundle_spec = bundle or settings.bundle
    if not bundle_spec:
        console.print(
            "[red]Error:[/red] No bundle specified. "
            "Use --bundle or set ACS_BUNDLE environment variable."
        )
        raise typer.Exit(1)

    overrides: dict[str, object] = {}
    if bundles_path:
        overrides["bundles_path"] = bundles_path
    if bundles_repo:
        overrides["bundles_repo"] = bundles_repo
    if comfyui_path:
        overrides["comfyui_path"] = comfyui_path
    if overrides:
        settings = settings.model_copy(update=overrides)

    ref = parse_bundle_reference(bundle_spec)
    version_override = bundle_version or settings.bundle_version
    if version_override and not ref.version:
        ref = BundleReference(name=ref.name, version=version_override, registry=ref.registry)

    verify = not (no_verify or settings.no_verify)
    mode = DeployMode.MODELS_ONLY if models_only else DeployMode.FULL
    if mode == DeployMode.MODELS_ONLY:
        console.print("[cyan]Models-only mode:[/cyan] Skipping ComfyUI setup and custom nodes\n")

    try:
        asyncio.run(
            _run_deploy(
                settings=settings,
                ref=ref,
                mode=mode,
                verify=verify,
                dry_run=dry_run,
                sync=sync,
            )
        )
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from e


async def _run_deploy(
    settings: Settings,
    ref: BundleReference,
    mode: DeployMode,
    verify: bool,
    dry_run: bool,
    sync: bool | None,
) -> None:
    """Async shim between the Typer command and the core deploy logic."""
    from .registry_service import run_deploy

    result = await run_deploy(
        settings=settings,
        ref=ref,
        mode=mode,
        verify=verify,
        dry_run=dry_run,
        sync=sync,
        console=console,
    )
    if not result.success:
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# bundle group
# ---------------------------------------------------------------------------


@bundle_app.command("list")
def bundle_list(
    registry: Annotated[
        str | None,
        typer.Option("--registry", "-r", help="Specific registry to list"),
    ] = None,
    tags: Annotated[
        str | None,
        typer.Option("--tags", "-t", help="Filter by tags (comma-separated)"),
    ] = None,
    sync: Annotated[
        bool,
        typer.Option("--sync/--no-sync", help="Sync before listing"),
    ] = False,
) -> None:
    """List available bundles across registries.

    Examples:

        # List all bundles from all registries
        acs bundle list

        # List from a specific registry
        acs bundle list --registry remote

        # Sync then list
        acs bundle list --sync
    """
    settings = get_settings()
    manager = create_registry_manager(settings)

    if registry and manager.get(registry) is None:
        rprint(f"[red]Registry '{registry}' not found[/red]")
        raise typer.Exit(1)

    async def _list() -> None:
        if sync:
            await manager.sync_all()

        registry_names = [registry] if registry else manager.list_registries()
        registries = [manager.get(r) for r in registry_names]

        tag_filter = {t.strip() for t in tags.split(",") if t.strip()} if tags else None

        table = Table(title="Available Bundles")
        table.add_column("Registry", style="cyan")
        table.add_column("Bundle", style="green")
        table.add_column("Description")
        table.add_column("Tags", style="dim")
        table.add_column("Default Version", style="yellow")

        for reg in registries:
            if reg is None:
                continue
            try:
                index = await reg.get_index()
                for b in index.bundles:
                    if tag_filter and not (b.tags and tag_filter.intersection(b.tags)):
                        continue
                    table.add_row(
                        reg.name,
                        b.name,
                        b.description or "-",
                        ", ".join(b.tags or []),
                        b.default_version or "-",
                    )
            except Exception as e:
                rprint(f"[yellow]Warning: Could not list {reg.name}: {e}[/yellow]")

        console.print(table)

    asyncio.run(_list())


@bundle_app.command("show")
def bundle_show(
    bundle: Annotated[
        str,
        typer.Argument(
            help="Bundle name or reference (e.g. wan_2.2_i2v, remote/wan_2.2_i2v:260101-01)"
        ),
    ],
    sync: Annotated[
        bool,
        typer.Option("--sync/--no-sync", help="Sync before showing"),
    ] = False,
) -> None:
    """Show detailed information about a bundle.

    Examples:

        acs bundle show wan_2.2_i2v
        acs bundle show remote/wan_2.2_i2v:260101-01
    """
    settings = get_settings()
    manager = create_registry_manager(settings)
    ref = parse_bundle_reference(bundle)

    async def _show() -> None:
        if sync:
            await manager.sync_all()

        try:
            reg = get_or_default_registry(manager, ref)
        except ValueError as e:
            rprint(f"[red]{e}[/red]")
            raise typer.Exit(1) from e

        try:
            path = await reg.resolve_bundle_path(ref.name, ref.version)
        except ValueError as e:
            rprint(f"[red]{e}[/red]")
            raise typer.Exit(1) from e

        bundle_yaml = path / "bundle.yaml"
        if not bundle_yaml.exists():
            rprint(f"[red]Bundle config not found at {path}[/red]")
            raise typer.Exit(1)

        raw = yaml.safe_load(bundle_yaml.read_text())
        if not isinstance(raw, dict):
            rprint(f"[red]Invalid bundle config at {bundle_yaml}: expected a mapping[/red]")
            raise typer.Exit(1)
        try:
            config = BundleConfig.model_validate(raw)
        except ValidationError as e:
            rprint(f"[red]Invalid bundle config at {bundle_yaml}:[/red]\n{e}")
            raise typer.Exit(1) from e

        rprint(f"\n[bold cyan]{ref.name}[/bold cyan]")
        rprint(f"  Path: {path}")
        rprint(f"  Registry: {reg.name}")
        rprint("\n[bold]Metadata:[/bold]")
        rprint(f"  Version: {config.metadata.version}")
        rprint(f"  Description: {config.metadata.description or '-'}")
        rprint(f"  Created: {config.metadata.created_at:%Y-%m-%d %H:%M UTC}")
        rprint(f"  Tested: {'Yes' if config.metadata.tested else 'No'}")

        if config.comfyui:
            rprint("\n[bold]ComfyUI:[/bold]")
            rprint(f"  Commit: {config.comfyui.commit[:12]}...")

        if config.custom_nodes:
            rprint(f"\n[bold]Custom Nodes ({len(config.custom_nodes)}):[/bold]")
            for node in config.custom_nodes:
                rprint(f"  • {node.name}")

        if config.models:
            total_files = sum(len(m.files) for m in config.models)
            rprint(f"\n[bold]Models ({len(config.models)} groups, {total_files} files):[/bold]")
            for model in config.models:
                rprint(f"  • {model.name} ({model.model_type})")

    asyncio.run(_show())


@bundle_app.command("versions")
def bundle_versions(
    bundle: Annotated[str, typer.Argument(help="Bundle name or reference")],
) -> None:
    """List available versions for a bundle.

    Example:

        acs bundle versions wan_2.2_i2v
    """
    settings = get_settings()
    manager = create_registry_manager(settings)
    ref = parse_bundle_reference(bundle)

    async def _versions() -> None:
        try:
            reg = get_or_default_registry(manager, ref)
        except ValueError as e:
            rprint(f"[red]{e}[/red]")
            raise typer.Exit(1) from e

        versions = await reg.list_versions(ref.name)
        rprint(f"\n[bold]Versions for {ref.name}:[/bold]")
        for version in versions:
            rprint(f"  • {version}")

    asyncio.run(_versions())


@bundle_app.command("sync")
def bundle_sync(
    registry: Annotated[
        str | None,
        typer.Option("--registry", "-r", help="Specific registry to sync"),
    ] = None,
) -> None:
    """Sync bundle registries (git pull).

    Examples:

        acs bundle sync
        acs bundle sync --registry remote
    """
    settings = get_settings()
    manager = create_registry_manager(settings)

    async def _sync() -> None:
        if registry:
            reg = manager.get(registry)
            if reg is None:
                rprint(f"[red]Registry '{registry}' not found[/red]")
                raise typer.Exit(1)
            await reg.sync()
            rprint(f"[green]✓[/green] Synced {registry}")
        else:
            await manager.sync_all()
            rprint("[green]✓[/green] Synced all registries")

    with console.status("[bold blue]Syncing registries..."):
        asyncio.run(_sync())


@bundle_app.command("set-current")
def bundle_set_current(
    name: Annotated[str, typer.Argument(help="Bundle name")],
    version: Annotated[str, typer.Argument(help="Version to set as current")],
) -> None:
    """Set the current version of a bundle.

    Example:

        acs bundle set-current wan_2.2_i2v 260101-02
    """
    from .bundle import BundleManager

    settings = get_settings()
    manager = BundleManager(settings)
    manager.set_current_version(name, version)
    console.print(f"[green]✓[/green] Set {name} current version to {version}")


@bundle_app.command("delete")
def bundle_delete(
    name: Annotated[str, typer.Argument(help="Bundle name")],
    version: Annotated[str, typer.Argument(help="Version to delete")],
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Skip confirmation"),
    ] = False,
) -> None:
    """Delete a bundle version.

    Example:

        acs bundle delete wan_2.2_i2v 260101-01
    """
    from .bundle import BundleManager

    settings = get_settings()
    manager = BundleManager(settings)

    if not force:
        confirm = typer.confirm(f"Delete {name} version {version}?")
        if not confirm:
            raise typer.Abort()

    manager.delete_version(name, version)
    console.print(f"[green]✓[/green] Deleted {name} version {version}")


def _render_contract_reports(reports: Sequence[ContractReport], *, json_output: bool) -> None:
    """Render bundle contract reports without coupling the pure validator to Rich."""
    if json_output:
        console.print_json(
            data={
                "ok": all(report.ok for report in reports),
                "bundles": [
                    {
                        "bundle_name": report.bundle_name,
                        "ok": report.ok,
                        "findings": [
                            {
                                "severity": finding.severity.value,
                                "check": finding.check,
                                "message": finding.message,
                                "location": finding.location,
                            }
                            for finding in report.findings
                        ],
                    }
                    for report in reports
                ],
            }
        )
        return

    if not reports:
        console.print(
            "[yellow]No bundles found; --allow-empty accepted this empty registry.[/yellow]"
        )
        return

    for severity in (Severity.ERROR, Severity.WARNING, Severity.INFO):
        table = Table(title=f"Bundle Contract Validation — {severity.value.title()}s")
        table.add_column("Bundle", style="cyan")
        table.add_column("Check")
        table.add_column("Location", style="dim")
        table.add_column("Message")
        count = 0
        for report in reports:
            for finding in report.findings:
                if finding.severity is severity:
                    table.add_row(
                        report.bundle_name, finding.check, finding.location, finding.message
                    )
                    count += 1
        if count:
            console.print(table)
    warnings = sum(
        finding.severity is Severity.WARNING for report in reports for finding in report.findings
    )
    if all(report.ok for report in reports):
        suffix = f" ({warnings} warnings)" if warnings else ""
        console.print(f"[green]✓[/green] Bundle contract is valid{suffix}")


@bundle_app.command("validate")
def bundle_validate(
    bundle: Annotated[
        str | None,
        typer.Argument(help="Bundle reference (e.g. wan_2.2_i2v or remote/wan_2.2_i2v:260101-01)"),
    ] = None,
    all_bundles: Annotated[
        bool,
        typer.Option("--all", help="Validate every bundle in the resolved registry"),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit a machine-readable report instead of Rich tables"),
    ] = False,
    sync: Annotated[
        bool,
        typer.Option("--sync/--no-sync", help="Sync registries before resolving"),
    ] = False,
    allow_empty: Annotated[
        bool,
        typer.Option("--allow-empty", help="Treat an empty registry as success when using --all"),
    ] = False,
    comfyui_url: Annotated[
        str | None,
        typer.Option(
            "--comfyui-url",
            help=(
                "Running ComfyUI URL for live workflow-class provider checks "
                "(defaults to ACS_COMFYUI_URL)"
            ),
        ),
    ] = None,
) -> None:
    """Validate a bundle contract, optionally against a running ComfyUI instance."""
    if bundle is not None and all_bundles:
        console.print("[red]Error:[/red] Specify exactly one of BUNDLE or --all (both given)")
        raise typer.Exit(1)
    if bundle is None and not all_bundles:
        console.print("[red]Error:[/red] Specify BUNDLE or --all")
        raise typer.Exit(1)

    settings = get_settings()
    manager = create_registry_manager(settings)

    try:
        reports = asyncio.run(
            validate_bundle_contracts(
                manager,
                bundle=bundle,
                all_bundles=all_bundles,
                sync=sync,
                allow_empty=allow_empty,
                comfyui_url=comfyui_url or settings.comfyui_url,
            )
        )
    except EmptyBundleRegistryError as exc:
        console.print(f"[red]Error:[/red] {exc}. Try --sync or pass --allow-empty.")
        raise typer.Exit(1) from exc
    except BundleContractServiceError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc
    _render_contract_reports(reports, json_output=json_output)
    if not all(report.ok for report in reports):
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# Top-level commands
# ---------------------------------------------------------------------------


def _print_carry_forward_report(source: ResolvedBundle, report: CarryForwardReport) -> None:
    """Render the non-fatal differences between a seed and its new snapshot."""
    console.print(f"\nCarried forward from {source.name}:{source.config.metadata.version}")
    if report.urls_carried:
        console.print(
            f"  urls           {len(report.urls_carried)}   {', '.join(report.urls_carried)}"
        )
    if report.blocks_carried:
        console.print(
            f"  blocks         {len(report.blocks_carried)}   {', '.join(report.blocks_carried)}"
        )
    if report.files_without_url:
        console.print(
            "[yellow]"
            f"  no url         {len(report.files_without_url)}   "
            f"{', '.join(report.files_without_url)}   -> url: '' # TODO"
            "[/yellow]"
        )
    if report.seed_files_unmatched:
        console.print(
            "[yellow]"
            f"  unmatched      {len(report.seed_files_unmatched)}   "
            f"{', '.join(report.seed_files_unmatched)}   "
            "(declared in seed, not on node)"
            "[/yellow]"
        )


def _print_custom_node_report(report: CarryForwardReport) -> None:
    """Show snapshot custom-node coverage beside the generated bundle path."""
    custom_nodes = report.custom_nodes
    console.print(
        "  Custom nodes: "
        f"captured {len(custom_nodes.captured)}, skipped {len(custom_nodes.skipped)}"
    )
    if custom_nodes.skipped:
        skipped = ", ".join(f"{node.name} ({node.reason})" for node in custom_nodes.skipped)
        console.print(f"[yellow]  skipped: {skipped}[/yellow]")


@app.command()
def snapshot(
    name: Annotated[
        str,
        typer.Option("--name", "-n", help="Bundle name"),
    ],
    workflow: Annotated[
        Path,
        typer.Option("--workflow", "-w", help="Path to workflow JSON file"),
    ],
    description: Annotated[
        str | None,
        typer.Option("--description", "-d", help="Bundle description"),
    ] = None,
    from_bundle: Annotated[
        str | None,
        typer.Option(
            "--from-bundle",
            help=(
                "Carry source URLs, metadata, hardware, generation and readiness_marker "
                "forward from an existing bundle ([registry/]name[:version]). Hashes, sizes, "
                "commits and the pip freeze always come from this node."
            ),
        ),
    ] = None,
    sync: Annotated[
        bool,
        typer.Option("--sync", help="Sync remote registries before resolving --from-bundle"),
    ] = False,
    extra_model_paths: Annotated[
        Path | None,
        typer.Option("--extra-model-paths", help="Path to extra_model_paths.yaml"),
    ] = None,
    scan_models: Annotated[
        bool,
        typer.Option(
            "--scan-models/--no-scan-models",
            help="Capture installed model sizes and SHA256 hashes (enabled by default)",
        ),
    ] = True,
    comfyui_path: Annotated[
        Path | None,
        typer.Option("--comfyui", "-c", help="Path to ComfyUI installation"),
    ] = None,
) -> None:
    """Create a snapshot bundle from a working ComfyUI setup.

    Captures the current state including:
    - ComfyUI commit SHA
    - Custom nodes with their commits
    - Python dependencies (pip freeze)
    - Installed model files with their local SHA256 hashes and sizes
    - Workflow JSON

    Use --from-bundle to carry authoring intent from a seed bundle. Source
    URLs, labels, metadata and Apex-facing fields carry forward; local byte
    metadata, commits and requirements are always captured from this node.

    Example:

        acs snapshot -n wan_2.2_i2v -w workflow.json -d "Initial setup"
    """
    from .snapshot import SnapshotManager

    settings = get_settings()
    if comfyui_path:
        settings = settings.model_copy(update={"comfyui_path": comfyui_path})

    manager = SnapshotManager(
        comfyui_path=settings.comfyui_path,
        bundles_path=settings.bundles_path,
        python_executable=settings.comfyui_python,
    )

    async def _run_snapshot() -> tuple[str, CarryForwardReport, ResolvedBundle | None]:
        resolved = (
            await resolve_bundle(settings, from_bundle, sync=sync)
            if from_bundle is not None
            else None
        )
        version, report = await manager.create_snapshot(
            name=name,
            workflow_path=workflow,
            description=description,
            extra_model_paths=extra_model_paths,
            scan_models=scan_models,
            carry_from=resolved.config if resolved is not None else None,
        )
        return version, report, resolved

    try:
        version, carry_report, resolved_bundle = asyncio.run(_run_snapshot())
    except BundleResolutionError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc

    console.print(f"\n[green]✓[/green] Created bundle {name} version {version}")
    console.print(f"  Path: {settings.bundles_path}/{name}/{version}/")
    _print_custom_node_report(carry_report)
    if resolved_bundle is not None:
        _print_carry_forward_report(resolved_bundle, carry_report)
    if scan_models:
        console.print("\n[yellow]Note:[/yellow] Add source URLs for the scanned model TODOs")


@app.command()
def status(
    comfyui_path: Annotated[
        Path | None,
        typer.Option("--comfyui", "-c", help="Path to ComfyUI installation"),
    ] = None,
) -> None:
    """Show deployment status of the ComfyUI installation."""
    from .comfyui import ComfyUIManager

    settings = get_settings()
    if comfyui_path:
        settings = settings.model_copy(update={"comfyui_path": comfyui_path})

    manager = ComfyUIManager(settings.comfyui_path, python_executable=settings.comfyui_python)
    status_info = asyncio.run(manager.get_status())

    console.print("\n[bold]ComfyUI Status[/bold]")
    console.print(f"Path: {settings.comfyui_path}")
    console.print(f"Commit: {status_info.commit or 'Unknown'}")
    console.print(f"Custom Nodes: {status_info.custom_node_count}")
    console.print(f"Running: {'Yes' if status_info.is_running else 'No'}")


# ---------------------------------------------------------------------------
# cache group
# ---------------------------------------------------------------------------


@cache_app.command("push")
def cache_push(
    bundle: Annotated[
        str,
        typer.Argument(help="Bundle reference (e.g. wan_2.2_i2v or remote/wan_2.2_i2v:260101-01)"),
    ],
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="Filename of the single model file to push"),
    ] = None,
    all_models: Annotated[
        bool,
        typer.Option("--all", "-a", help="Push all model files in the bundle"),
    ] = False,
    sync: Annotated[
        bool,
        typer.Option("--sync/--no-sync", help="Sync registries before resolving"),
    ] = False,
    direct: Annotated[
        bool,
        typer.Option(
            "--direct/--no-direct",
            help="Use operator-supplied R2 write credentials instead of Apex credential minting",
        ),
    ] = False,
) -> None:
    """Push model weights from local disk to the R2 model cache.

    Requires Apex credentials by default. Use --direct with explicitly
    configured write credentials when authoring a bundle without Apex running.
    Exactly one of --model or --all must be provided.

    Examples:

        # Push a single model file
        acs cache push wan_2.2_i2v --model wan2.1_i2vgen_480p_14f_fp8_e4m3fn.safetensors

        # Push all models in the bundle
        acs cache push wan_2.2_i2v --all
    """
    if not model and not all_models:
        console.print("[red]Error:[/red] Specify --model <filename> or --all")
        raise typer.Exit(1)
    if model and all_models:
        console.print("[red]Error:[/red] --model and --all are mutually exclusive")
        raise typer.Exit(1)

    settings = get_settings()

    admin_token: str | None = None
    if direct:
        if not settings.r2_write_access_key_id:
            console.print("[red]Error:[/red] ACS_R2_WRITE_ACCESS_KEY_ID is not set")
            raise typer.Exit(1)
        if not unwrap_secret(settings.r2_write_secret_access_key):
            console.print("[red]Error:[/red] ACS_R2_WRITE_SECRET_ACCESS_KEY is not set")
            raise typer.Exit(1)
    else:
        admin_token = unwrap_secret(settings.apex_admin_token)
        if not admin_token:
            console.print(
                "[red]Error:[/red] ACS_APEX_ADMIN_TOKEN is not set — refusing before any transfer"
            )
            raise typer.Exit(1)
        if not settings.apex_base_url:
            console.print("[red]Error:[/red] ACS_APEX_BASE_URL is not set")
            raise typer.Exit(1)

    if not settings.r2_s3_endpoint:
        console.print("[red]Error:[/red] ACS_R2_S3_ENDPOINT is not set")
        raise typer.Exit(1)

    manager = create_registry_manager(settings)
    ref = parse_bundle_reference(bundle)
    try:
        resolved = asyncio.run(
            resolve_cache_targets(
                settings,
                manager,
                ref,
                only_filename=model,
                sync=sync,
            )
        )
    except CacheWorkflowError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc

    provider: CacheCredentialProvider
    if direct:
        write_creds = write_creds_from_settings(settings)
        if write_creds is None:
            console.print("[red]Error:[/red] Direct R2 write credentials are incomplete")
            raise typer.Exit(1)
        provider = StaticCacheCredentialProvider(write_creds)
    else:
        if admin_token is None or settings.apex_base_url is None:
            msg = "Apex credential validation did not establish a required value"
            raise RuntimeError(msg)
        provider = ApexCacheCredentialProvider(
            base_url=settings.apex_base_url,
            admin_token=admin_token,
        )

    console.print(f"[dim]credential mode: {provider.name}[/dim]")
    try:
        report = cache_service.push_models(
            settings, list(resolved.targets), console, provider=provider
        )
    finally:
        provider.close()
    if not report.ok:
        raise typer.Exit(1)


@cache_app.command("verify")
def cache_verify(
    bundle: Annotated[
        str,
        typer.Argument(help="Bundle reference (e.g. wan_2.2_i2v or remote/wan_2.2_i2v:260101-01)"),
    ],
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="Filename of the single model file to verify"),
    ] = None,
    all_models: Annotated[
        bool,
        typer.Option("--all", "-a", help="Verify all model files in the bundle"),
    ] = False,
    deep: Annotated[
        bool,
        typer.Option("--deep", help="Pull every object to a temporary file and re-hash it"),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit a machine-readable report instead of a table"),
    ] = False,
    sync: Annotated[
        bool,
        typer.Option("--sync/--no-sync", help="Sync registries before resolving"),
    ] = False,
) -> None:
    """Verify cache objects with the read-only credential path."""
    if not model and not all_models:
        console.print("[red]Error:[/red] Specify --model <filename> or --all")
        raise typer.Exit(1)
    if model and all_models:
        console.print("[red]Error:[/red] --model and --all are mutually exclusive")
        raise typer.Exit(1)

    settings = get_settings()
    manager = create_registry_manager(settings)
    ref = parse_bundle_reference(bundle)
    try:
        report = asyncio.run(
            verify_cache_targets(
                settings,
                manager,
                ref,
                only_filename=model,
                sync=sync,
                deep=deep,
            )
        )
    except CacheWorkflowError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc
    if json_output:
        payload: dict[str, object] = {
            "ok": report.ok,
            "results": [
                {
                    "filename": result.filename,
                    "key": result.key,
                    "ok": result.ok,
                    "status": result.status,
                    "detail": result.detail,
                }
                for result in report.results
            ],
        }
        if report.configuration_error:
            payload["configuration_error"] = report.configuration_error
        console.print_json(data=payload)
    elif report.configuration_error:
        console.print(f"[red]Error:[/red] {report.configuration_error}")
        raise typer.Exit(1)
    else:
        table = Table(title="Model Cache Verification")
        table.add_column("Filename", style="cyan")
        table.add_column("Status")
        table.add_column("Key", style="dim")
        table.add_column("Detail")
        for result in report.results:
            style = "green" if result.ok else "red"
            table.add_row(
                result.filename,
                f"[{style}]{result.status}[/{style}]",
                result.key or "-",
                result.detail,
            )
        console.print(table)
    if not report.ok:
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# models group
# ---------------------------------------------------------------------------


@models_app.command("fetch")
def models_fetch(
    url: Annotated[str, typer.Option("--url", help="Model file HTTP(S) URL")],
    model_type: Annotated[
        str,
        typer.Option("--model-type", help="ComfyUI models subdirectory, e.g. checkpoints"),
    ],
    filename: Annotated[
        str,
        typer.Option("--filename", help="Plain filename to use beneath the model type"),
    ],
    subdirectory: Annotated[
        str | None,
        typer.Option("--subdirectory", help="Optional directory beneath --model-type"),
    ] = None,
    sha256: Annotated[
        str | None,
        typer.Option("--sha256", help="Optional expected SHA-256 checksum"),
    ] = None,
    comfyui_path: Annotated[
        Path | None,
        typer.Option("--comfyui", "-c", help="Path to the ComfyUI installation"),
    ] = None,
) -> None:
    """Fetch one weight through Aisha's normal downloader and print bundle YAML."""
    settings = get_settings()
    if comfyui_path:
        settings = settings.model_copy(update={"comfyui_path": comfyui_path})
    try:
        fetched = asyncio.run(
            fetch_model(
                settings,
                url=url,
                model_type=model_type,
                filename=filename,
                subdirectory=subdirectory,
                sha256=sha256,
            )
        )
    except ModelFetchDownloadError as exc:
        for failure in exc.failures:
            console.print(f"[red]✗[/red] {failure.filename}: {failure.reason}")
        if not exc.failures:
            console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc
    except ModelsServiceError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc

    console.print(f"  sha256:     {fetched.sha256}")
    console.print(f"  size_bytes: {fetched.size_bytes}")
    console.print()
    console.print(fetched.yaml_fragment, markup=False)


@models_app.command("check")
def models_check(
    bundle: Annotated[
        str | None,
        typer.Argument(help="Bundle reference (e.g. wan_2.2_i2v or wan_2.2_i2v:260105-01)"),
    ] = None,
    all_bundles: Annotated[
        bool,
        typer.Option("--all", help="Check every bundle in the resolved registry"),
    ] = False,
    offline: Annotated[
        bool,
        typer.Option(
            "--offline",
            help="Validate parsing and required fields without any network request",
        ),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit a machine-readable JSON report instead of a table"),
    ] = False,
    registry: Annotated[
        str | None,
        typer.Option(
            "--registry", "-r", help="Registry to use with --all (default: default registry)"
        ),
    ] = None,
    bundles_path: Annotated[
        Path | None,
        typer.Option("--bundles-path", help="Override local bundles path"),
    ] = None,
    sync: Annotated[
        bool,
        typer.Option("--sync/--no-sync", help="Sync registries before resolving"),
    ] = False,
    allow_empty: Annotated[
        bool,
        typer.Option("--allow-empty", help="Treat a registry with no bundles as success"),
    ] = False,
) -> None:
    """Authenticated Range-probe every model file in a bundle. Writes nothing to disk.

    Reports HTTP status, content-type, resolved filename, and size for each
    file, flagging HTML responses (wrong domain/expired token), 401/403, and
    civitai.com 404s that likely need civitai.red. Suitable as a CI gate for
    bundle PRs — exits 1 if any file fails.

    Exactly one of BUNDLE or --all is required. A bundle that fails to parse
    is reported as a failure row rather than aborting the run.

    Examples:

        acs models check wan_2.2_i2v
        acs models check wan_2.2_i2v:260105-01
        acs models check --all --offline
        acs models check --all --json
    """
    from .preflight import (
        MultiBundleReport,
        PreflightReport,
        check_all_bundles,
        check_bundle,
        multi_report_to_dict,
        render_multi_report,
        render_report,
        report_to_dict,
    )

    def _render_single_report(report: PreflightReport) -> None:
        if json_output:
            console.print_json(data=report_to_dict(report))
        else:
            render_report(report, console)

    def _render_multi_report(report: MultiBundleReport) -> None:
        if json_output:
            console.print_json(data=multi_report_to_dict(report))
        else:
            render_multi_report(report, console)

    if bundle is not None and all_bundles:
        console.print("[red]Error:[/red] Specify exactly one of BUNDLE or --all (both given)")
        raise typer.Exit(1)
    if bundle is None and not all_bundles:
        console.print("[red]Error:[/red] Specify BUNDLE or --all")
        raise typer.Exit(1)

    settings = get_settings()
    settings, manager = _create_manager(settings, bundles_path)

    if all_bundles:

        async def _check_all() -> MultiBundleReport:
            if sync:
                await manager.sync_all()
            reg = _resolve_registry(manager, registry)
            index = await reg.get_index()
            return await check_all_bundles(
                [entry.name for entry in index.bundles],
                settings,
                offline=offline,
                resolve_bundle_path=reg.resolve_bundle_path,
            )

        try:
            multi_report = asyncio.run(_check_all())
        except ValueError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1) from e

        _render_multi_report(multi_report)
        if multi_report.is_empty and not allow_empty:
            if not json_output:
                hint = "" if sync else " Try --sync."
                console.print(
                    "[red]Error:[/red] no bundles found in the resolved registry; "
                    f"nothing was checked.{hint} Pass --allow-empty to treat this as success."
                )
            raise typer.Exit(1)
        if not multi_report.ok:
            raise typer.Exit(1)
        return

    if bundle is None:
        console.print("[red]Error:[/red] Specify BUNDLE or --all")
        raise typer.Exit(1)
    ref = parse_bundle_reference(bundle)

    async def _run() -> PreflightReport:
        if sync:
            await manager.sync_all()
        bundle_path = await manager.resolve(ref)

        bundle_yaml = bundle_path / "bundle.yaml"
        if not bundle_yaml.exists():
            console.print(f"[red]Error:[/red] Bundle config not found at {bundle_path}")
            raise typer.Exit(1)

        raw = yaml.safe_load(bundle_yaml.read_text())
        try:
            bundle_config = BundleConfig.model_validate(raw)
        except ValidationError as e:
            console.print(f"[red]Error:[/red] Invalid bundle config:\n{e}")
            raise typer.Exit(1) from e

        return await check_bundle(bundle_config, settings, offline=offline)

    try:
        report = asyncio.run(_run())
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from e

    _render_single_report(report)
    if not report.ok:
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# timings group
# ---------------------------------------------------------------------------


def _timings_bundle_label(record: dict[str, object]) -> str:
    name = record.get("bundle")
    version = record.get("bundle_version")
    if name and version:
        return f"{name}:{version}"
    return str(name) if name else "-"


def _timings_number(value: object, *, precision: int = 1) -> str:
    return f"{value:.{precision}f}" if isinstance(value, (int, float)) else "-"


def _timings_phase_status(entry: dict[str, object]) -> str:
    """Read explicit schema-2 phase status or derive it from schema 1."""
    status = entry.get("status")
    if isinstance(status, str):
        return status
    return "skipped" if entry.get("skipped") else "completed"


def _timings_models_metrics(record: dict[str, object]) -> dict[str, object] | None:
    """Find schema-2 model metrics while keeping schema-1 records renderable."""
    metrics = record.get("metrics")
    if isinstance(metrics, dict):
        models = metrics.get("models")
        if isinstance(models, dict):
            return models
    models = record.get("models")
    return models if isinstance(models, dict) else None


def _render_timings_table(records: Sequence[dict[str, object]]) -> None:
    """Render timing JSONL records without coupling the pure reader to Rich.

    Uses its own wide, fixed-width `Console` rather than the module-level one
    -- one column per `PhaseId` plus identity/outcome/throughput columns is
    routinely 13+ columns wide, which a terminal-width-detecting console
    truncates into unreadable 1-2 character cells the moment stdout isn't a
    real wide TTY (piped output, CI logs, `CliRunner`). This is the artefact
    the next architecture decision gets made from, so it must stay readable.
    """
    from .provisioning_timing import PhaseId

    phase_ids = [phase.value for phase in PhaseId]
    table = Table(title="Provisioning Timings")
    table.add_column("Time", style="dim", overflow="fold")
    table.add_column("Bundle", overflow="fold")
    table.add_column("Mode")
    table.add_column("Outcome")
    for phase_id in phase_ids:
        table.add_column(phase_id, justify="right")
    table.add_column("Total (s)", justify="right")
    table.add_column("Effective MiB/s", justify="right")

    for record in records:
        phases_raw = record.get("phases")
        phases_by_id: dict[object, dict[str, object]] = (
            {p.get("phase"): p for p in phases_raw if isinstance(p, dict)}
            if isinstance(phases_raw, list)
            else {}
        )
        row = [
            str(record.get("started_at", record.get("ts", "-"))),
            _timings_bundle_label(record),
            str(record.get("mode", "-")),
            str(record.get("outcome", "-")),
        ]
        for phase_id in phase_ids:
            entry = phases_by_id.get(phase_id)
            if entry is None:
                row.append("-")
            elif _timings_phase_status(entry) == "skipped":
                row.append("skip")
            elif _timings_phase_status(entry) == "failed":
                row.append(f"failed ({_timings_number(entry.get('duration_s'))})")
            else:
                row.append(_timings_number(entry.get("duration_s")))
        row.append(_timings_number(record.get("total_s")))
        models = _timings_models_metrics(record)
        row.append(
            _timings_number(models.get("effective_mib_per_s") if models is not None else None)
        )

        style = "red" if record.get("outcome") == "failed" else None
        table.add_row(*row, style=style)

    Console(width=220).print(table)


@timings_app.command("show")
def timings_show(
    path: Annotated[
        Path | None,
        typer.Option(
            "--path",
            help=(
                "Timings JSONL file. Defaults to ACS_PROVISIONING_TIMING_PATH, or "
                "cache_path/'provisioning-timings.jsonl'"
            ),
        ),
    ] = None,
    last: Annotated[
        int | None,
        typer.Option("--last", help="Show only the most recent N runs"),
    ] = None,
    bundle: Annotated[
        str | None,
        typer.Option("--bundle", help="Filter to a single bundle name"),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit the raw JSONL records instead of a table"),
    ] = False,
) -> None:
    """Render per-deployment provisioning phase timings recorded by `acs deploy`.

    Reads the always-on JSONL sink written after every deployment (B-L1) --
    this is the record to answer "where did provisioning time go?" without a
    bespoke script and a rented GPU.

    Examples:

        acs timings show
        acs timings show --last 10
        acs timings show --bundle qwen_rapid_aio --json
    """
    from .provisioning_timing import read_records

    if last is not None and last < 1:
        raise typer.BadParameter("must be at least 1", param_hint="--last")

    settings = get_settings()
    timing_path = (
        path
        or settings.provisioning_timing_path
        or (settings.cache_path / "provisioning-timings.jsonl")
    )
    records = read_records(timing_path, bundle=bundle, last=last)

    if json_output:
        for record in records:
            typer.echo(json.dumps(record, separators=(",", ":")))
        return

    if not records:
        console.print(f"[yellow]No provisioning timing records found at {timing_path}[/yellow]")
        return

    _render_timings_table(records)


if __name__ == "__main__":
    app()
