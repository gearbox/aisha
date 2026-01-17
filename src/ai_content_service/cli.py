"""CLI for AI Content Service."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Annotated

import typer
from rich.console import Console
from rich.table import Table

from .config import DeployMode, Settings, get_settings

if TYPE_CHECKING:
    from pathlib import Path

app = typer.Typer(
    name="acs",
    help="AI Content Service - Bundle-based deployment automation",
    no_args_is_help=True,
)

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
            help="Bundle name to deploy. Falls back to ACS_BUNDLE env var.",
        ),
    ] = None,
    version: Annotated[
        str | None,
        typer.Option(
            "--version",
            "-V",
            help="Specific bundle version. Falls back to ACS_BUNDLE_VERSION or 'current'.",
        ),
    ] = None,
    models_only: Annotated[
        bool,
        typer.Option(
            "--models-only",
            "-m",
            help="Only download models and install workflow. Skip ComfyUI setup and custom nodes.",
        ),
    ] = False,
    no_verify: Annotated[
        bool,
        typer.Option(
            "--no-verify",
            help="Skip ComfyUI verification after deployment.",
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            "-n",
            help="Show deployment plan without executing.",
        ),
    ] = False,
    comfyui_path: Annotated[
        Path | None,
        typer.Option(
            "--comfyui",
            "-c",
            help="Path to ComfyUI installation.",
        ),
    ] = None,
) -> None:
    """Deploy a bundle to the ComfyUI installation.

    By default, performs a full deployment including ComfyUI checkout,
    requirements installation, custom nodes, models, and workflow.

    Use --models-only for lightweight deployments when you already have
    a working ComfyUI setup and just want to add models and workflow.

    Examples:

        # Full deployment using ACS_BUNDLE env var
        acs deploy

        # Full deployment with explicit bundle
        acs deploy --bundle wan_2.2_i2v

        # Models-only deployment (skip ComfyUI setup)
        acs deploy --bundle wan_2.2_i2v --models-only

        # Specific version with models-only
        acs deploy -b wan_2.2_i2v -V 260101-01 --models-only

        # Dry run to see deployment plan
        acs deploy --bundle wan_2.2_i2v --models-only --dry-run
    """
    settings = get_settings()

    # Resolve bundle name
    bundle_name = bundle or settings.bundle
    if not bundle_name:
        console.print(
            "[red]Error:[/red] No bundle specified. "
            "Use --bundle or set ACS_BUNDLE environment variable."
        )
        raise typer.Exit(1)

    # Resolve version
    bundle_version = version or settings.bundle_version

    # Override settings if CLI args provided
    if comfyui_path:
        settings.comfyui_path = comfyui_path
    if no_verify:
        settings.no_verify = True

    # Determine deployment mode
    mode = DeployMode.MODELS_ONLY if models_only else DeployMode.FULL

    # Display mode info
    if mode == DeployMode.MODELS_ONLY:
        console.print("[cyan]Models-only mode:[/cyan] Skipping ComfyUI setup and custom nodes\n")

    # Run deployment
    asyncio.run(
        _run_deploy(
            settings=settings,
            bundle_name=bundle_name,
            version=bundle_version,
            mode=mode,
            verify=not settings.no_verify,
            dry_run=dry_run,
        )
    )


async def _run_deploy(
    settings: Settings,
    bundle_name: str,
    version: str | None,
    mode: DeployMode,
    verify: bool,
    dry_run: bool,
) -> None:
    """Run async deployment."""
    # Import here to avoid circular imports and allow lazy loading
    from .bundle import BundleManager
    from .comfyui import ComfyUIManager
    from .deployer import Deployer
    from .downloader import ModelDownloader
    from .workflows import WorkflowManager

    # Create managers with dependency injection
    bundle_manager = BundleManager(settings.bundles_path)
    comfyui_manager = ComfyUIManager(settings.comfyui_path)
    model_downloader = ModelDownloader(
        max_concurrent=settings.max_concurrent_downloads,
        hf_token=settings.hf_token,
        civitai_token=settings.civitai_api_token,
    )
    workflow_manager = WorkflowManager(settings.comfyui_path)

    deployer = Deployer(
        settings=settings,
        bundle_manager=bundle_manager,
        comfyui_manager=comfyui_manager,
        model_downloader=model_downloader,
        workflow_manager=workflow_manager,
    )

    result = await deployer.deploy(
        bundle_name=bundle_name,
        version=version,
        mode=mode,
        verify=verify,
        dry_run=dry_run,
    )

    if not result.success:
        raise typer.Exit(1)


# Bundle subcommand group
bundle_app = typer.Typer(
    name="bundle",
    help="Bundle management commands",
    no_args_is_help=True,
)
app.add_typer(bundle_app)


@bundle_app.command("list")
def bundle_list(
    name: Annotated[
        str | None,
        typer.Argument(help="Bundle name to list versions for"),
    ] = None,
) -> None:
    """List bundles or versions of a specific bundle.

    Examples:

        # List all bundles
        acs bundle list

        # List versions of a specific bundle
        acs bundle list wan_2.2_i2v
    """
    from .bundle import BundleManager

    settings = get_settings()
    manager = BundleManager(settings.bundles_path)

    if name:
        # List versions of specific bundle
        versions = manager.list_versions(name)
        current = manager.get_current_version(name)

        table = Table(title=f"Versions of {name}")
        table.add_column("Version", style="cyan")
        table.add_column("Current", justify="center")
        table.add_column("Tested", justify="center")

        for v in versions:
            is_current = "●" if v.version == current else ""
            tested = "[green]✓[/green]" if v.tested else ""
            table.add_row(v.version, is_current, tested)

    else:
        # List all bundles
        bundles = manager.list_bundles()

        table = Table(title="Available Bundles")
        table.add_column("Name", style="cyan")
        table.add_column("Current Version")
        table.add_column("Versions", justify="right")

        for b in bundles:
            table.add_row(b.name, b.current_version or "-", str(b.version_count))

    console.print(table)


@bundle_app.command("show")
def bundle_show(
    name: Annotated[str, typer.Argument(help="Bundle name")],
    version: Annotated[
        str | None,
        typer.Option("--version", "-V", help="Specific version"),
    ] = None,
) -> None:
    """Show bundle details.

    Examples:

        # Show current version
        acs bundle show wan_2.2_i2v

        # Show specific version
        acs bundle show wan_2.2_i2v --version 260101-01
    """
    from .bundle import BundleManager

    settings = get_settings()
    manager = BundleManager(settings.bundles_path)

    bundle_path = manager.resolve_bundle_path(name, version)
    bundle = manager.load_bundle(bundle_path)

    # Display bundle info
    console.print(f"\n[bold cyan]{bundle.metadata.name}[/bold cyan]")
    console.print(f"Version: {bundle.metadata.version}")
    console.print(f"Description: {bundle.metadata.description}")
    console.print(f"Created: {bundle.metadata.created_at}")
    console.print(f"Tested: {'Yes' if bundle.metadata.tested else 'No'}")

    if bundle.comfyui:
        console.print(f"\n[bold]ComfyUI:[/bold] {bundle.comfyui.commit[:12]}")

    if bundle.custom_nodes:
        console.print(f"\n[bold]Custom Nodes ({len(bundle.custom_nodes)}):[/bold]")
        for node in bundle.custom_nodes:
            console.print(f"  • {node.name} ({node.commit_sha[:8]})")

    if bundle.models:
        total_files = sum(len(m.files) for m in bundle.models)
        console.print(f"\n[bold]Models ({len(bundle.models)} groups, {total_files} files):[/bold]")
        for model in bundle.models:
            console.print(f"  • {model.name} ({model.model_type})")
            for f in model.files:
                console.print(f"    - {f.filename}")


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
    manager = BundleManager(settings.bundles_path)

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
    manager = BundleManager(settings.bundles_path)

    if not force:
        confirm = typer.confirm(f"Delete {name} version {version}?")
        if not confirm:
            raise typer.Abort()

    manager.delete_version(name, version)
    console.print(f"[green]✓[/green] Deleted {name} version {version}")


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

    manager = ComfyUIManager(settings.comfyui_path)
    status = asyncio.run(manager.get_status())

    console.print("\n[bold]ComfyUI Status[/bold]")
    console.print(f"Path: {settings.comfyui_path}")
    console.print(f"Commit: {status.commit or 'Unknown'}")
    console.print(f"Custom Nodes: {status.custom_node_count}")
    console.print(f"Running: {'Yes' if status.is_running else 'No'}")


if __name__ == "__main__":
    app()
