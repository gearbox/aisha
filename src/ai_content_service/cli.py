"""CLI for AI Content Service."""

import asyncio
from pathlib import Path
from typing import Annotated

import typer
from rich import print as rprint
from rich.console import Console
from rich.table import Table

from .bundle_registry import BundleReference
from .config import DeployMode, Settings, get_settings
from .registry_service import create_registry_manager

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
app.add_typer(bundle_app)

console = Console()


def version_callback(value: bool) -> None:
    """Show version and exit."""
    if value:
        from . import __version__

        console.print(f"acs version {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    _version: Annotated[
        bool | None,
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
    pass


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
        bool,
        typer.Option("--sync/--no-sync", help="Sync registries before deploy"),
    ] = True,
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

    if bundles_path:
        settings.bundles_path = bundles_path
    if bundles_repo:
        settings.bundles_repo = bundles_repo
    if comfyui_path:
        settings.comfyui_path = comfyui_path
    if no_verify:
        settings.no_verify = True

    ref = BundleReference.parse(bundle_spec)
    if bundle_version and not ref.version:
        ref = BundleReference(name=ref.name, version=bundle_version, registry=ref.registry)

    mode = DeployMode.MODELS_ONLY if models_only else DeployMode.FULL
    if mode == DeployMode.MODELS_ONLY:
        console.print("[cyan]Models-only mode:[/cyan] Skipping ComfyUI setup and custom nodes\n")

    asyncio.run(
        _run_deploy(
            settings=settings,
            ref=ref,
            mode=mode,
            verify=not no_verify,
            dry_run=dry_run,
            sync=sync,
        )
    )


async def _run_deploy(
    settings: Settings,
    ref: BundleReference,
    mode: DeployMode,
    verify: bool,
    dry_run: bool,
    sync: bool,
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

    async def _list() -> None:
        if sync:
            await manager.sync_all()

        registry_names = [registry] if registry else manager.list_registries()
        registries = [manager.get(r) for r in registry_names]

        tag_filter = set(tags.split(",")) if tags else None

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
                    if tag_filter and b.tags and not tag_filter.intersection(b.tags):
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
    import yaml

    settings = get_settings()
    manager = create_registry_manager(settings)
    ref = BundleReference.parse(bundle)

    async def _show() -> None:
        if sync:
            await manager.sync_all()

        reg = manager.get(ref.registry) if ref.registry else manager.default
        if reg is None:
            rprint("[red]No registry available[/red]")
            raise typer.Exit(1)

        try:
            path = await reg.resolve_bundle_path(ref.name, ref.version)
            bundle_yaml = path / "bundle.yaml"

            if not bundle_yaml.exists():
                rprint(f"[red]Bundle config not found at {path}[/red]")
                raise typer.Exit(1)

            with bundle_yaml.open() as f:
                config = yaml.safe_load(f)

            rprint(f"\n[bold cyan]{ref.name}[/bold cyan]")
            rprint(f"  Path: {path}")
            rprint(f"  Registry: {reg.name}")

            if metadata := config.get("metadata"):
                rprint("\n[bold]Metadata:[/bold]")
                rprint(f"  Version: {metadata.get('version', 'N/A')}")
                rprint(f"  Description: {metadata.get('description', 'N/A')}")
                rprint(f"  Created: {metadata.get('created_at', 'N/A')}")
                rprint(f"  Tested: {metadata.get('tested', False)}")

            if comfyui := config.get("comfyui"):
                rprint("\n[bold]ComfyUI:[/bold]")
                rprint(f"  Commit: {comfyui.get('commit', 'N/A')[:12]}...")

            if nodes := config.get("custom_nodes"):
                rprint(f"\n[bold]Custom Nodes ({len(nodes)}):[/bold]")
                for node in nodes:
                    rprint(f"  • {node.get('name', 'Unknown')}")

            if models := config.get("models"):
                total_files = sum(len(m.get("files", [])) for m in models)
                rprint(f"\n[bold]Models ({len(models)} groups, {total_files} files):[/bold]")
                for model in models:
                    rprint(
                        f"  • {model.get('name', 'Unknown')} ({model.get('model_type', 'unknown')})"
                    )

        except ValueError as e:
            rprint(f"[red]{e}[/red]")
            raise typer.Exit(1) from e

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
    ref = BundleReference.parse(bundle)

    async def _versions() -> None:
        reg = manager.get(ref.registry) if ref.registry else manager.default
        if reg is None:
            rprint("[red]No registry available[/red]")
            raise typer.Exit(1)

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


# ---------------------------------------------------------------------------
# Top-level commands
# ---------------------------------------------------------------------------


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
        str,
        typer.Option("--description", "-d", help="Bundle description"),
    ] = "",
    extra_model_paths: Annotated[
        Path | None,
        typer.Option("--extra-model-paths", help="Path to extra_model_paths.yaml"),
    ] = None,
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
    - Workflow JSON

    Example:

        acs snapshot -n wan_2.2_i2v -w workflow.json -d "Initial setup"
    """
    from .snapshot import SnapshotManager

    settings = get_settings()
    if comfyui_path:
        settings.comfyui_path = comfyui_path

    manager = SnapshotManager(
        comfyui_path=settings.comfyui_path,
        bundles_path=settings.bundles_path,
    )

    version = asyncio.run(
        manager.create_snapshot(
            name=name,
            workflow_path=workflow,
            description=description,
            extra_model_paths=extra_model_paths,
        )
    )

    console.print(f"\n[green]✓[/green] Created bundle {name} version {version}")
    console.print(f"  Path: {settings.bundles_path}/{name}/{version}/")
    console.print("\n[yellow]Note:[/yellow] Edit bundle.yaml to add model definitions")


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
        settings.comfyui_path = comfyui_path

    manager = ComfyUIManager(settings.comfyui_path, python_executable=settings.comfyui_python)
    status_info = asyncio.run(manager.get_status())

    console.print("\n[bold]ComfyUI Status[/bold]")
    console.print(f"Path: {settings.comfyui_path}")
    console.print(f"Commit: {status_info.commit or 'Unknown'}")
    console.print(f"Custom Nodes: {status_info.custom_node_count}")
    console.print(f"Running: {'Yes' if status_info.is_running else 'No'}")


if __name__ == "__main__":
    app()
