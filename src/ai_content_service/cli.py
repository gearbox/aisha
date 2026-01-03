"""CLI for AI Content Service deployment and management."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ai_content_service import __version__
from ai_content_service.config import get_settings

app = typer.Typer(
    name="acs",
    help="AI Content Service - Bundle-based deployment automation",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

bundle_app = typer.Typer(
    name="bundle",
    help="Bundle management commands",
    no_args_is_help=True,
)

app.add_typer(bundle_app, name="bundle")

console = Console()


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"[cyan]AI Content Service[/cyan] version [green]{__version__}[/green]")
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
    ] = None,
) -> None:
    """AI Content Service - Bundle-based model deployment for ComfyUI."""


# =============================================================================
# Deploy Commands
# =============================================================================


@app.command()
def deploy(
    bundle_name: Annotated[
        str | None,
        typer.Option(
            "--bundle",
            "-b",
            help="Bundle name to deploy. Can also be set via ACS_BUNDLE env var.",
        ),
    ] = None,
    version: Annotated[
        str | None,
        typer.Option(
            "--version",
            help="Specific bundle version. Default: uses 'current' symlink.",
        ),
    ] = None,
    comfyui_path: Annotated[
        Path | None,
        typer.Option(
            "--comfyui",
            help="Path to ComfyUI installation.",
        ),
    ] = None,
    no_verify: Annotated[
        bool,
        typer.Option(
            "--no-verify",
            help="Skip ComfyUI verification after deployment.",
        ),
    ] = False,
) -> None:
    """Deploy a bundle to ComfyUI.

    Deploys the specified bundle, including ComfyUI update, custom nodes,
    models, and workflows.

    Example:
        acs deploy --bundle wan_2.2_i2v
        acs deploy --bundle wan_2.2_i2v --version 260101-01
    """
    from ai_content_service.deployer import BundleDeployer

    settings = get_settings()
    if comfyui_path:
        settings.comfyui_path = comfyui_path
    if no_verify:
        settings.no_verify = True

    deployer = BundleDeployer(settings, console)

    report = asyncio.run(deployer.deploy_bundle(bundle_name, version))

    if not report.success:
        raise typer.Exit(1)


# =============================================================================
# Snapshot Commands
# =============================================================================


@app.command()
def snapshot(
    name: Annotated[
        str,
        typer.Option(
            "--name",
            "-n",
            help="Bundle name for the snapshot (e.g., wan_2.2_i2v).",
        ),
    ],
    workflow: Annotated[
        Path,
        typer.Option(
            "--workflow",
            "-w",
            help="Path to workflow JSON file.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    description: Annotated[
        str,
        typer.Option(
            "--description",
            "-d",
            help="Description for this bundle version.",
        ),
    ] = "",
    extra_model_paths: Annotated[
        Path | None,
        typer.Option(
            "--extra-model-paths",
            help="Path to extra_model_paths.yaml file.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ] = None,
    comfyui_path: Annotated[
        Path | None,
        typer.Option(
            "--comfyui",
            help="Path to ComfyUI installation.",
        ),
    ] = None,
    no_set_current: Annotated[
        bool,
        typer.Option(
            "--no-set-current",
            help="Don't set this version as current.",
        ),
    ] = False,
) -> None:
    """Capture a snapshot from current ComfyUI installation.

    Creates a new bundle version by capturing:
    - ComfyUI commit SHA
    - All custom nodes with their commit SHAs
    - pip freeze output (requirements.lock)
    - Workflow JSON file
    - Optional extra_model_paths.yaml

    Example:
        acs snapshot --name wan_2.2_i2v --workflow workflow.json
        acs snapshot -n wan_2.2_i2v -w workflow.json -d "Initial WAN 2.2 setup"
    """
    from ai_content_service.bundle import BundleManager

    settings = get_settings()
    if comfyui_path:
        settings.comfyui_path = comfyui_path

    bundle_manager = BundleManager(settings, console)

    try:
        result = bundle_manager.create_snapshot(
            bundle_name=name,
            workflow_path=workflow,
            description=description,
            extra_model_paths_path=extra_model_paths,
            set_as_current=not no_set_current,
        )

        if result.success:
            console.print(
                Panel.fit(
                    f"[bold green]Snapshot Created[/bold green]\n"
                    f"Bundle: {result.bundle_name}\n"
                    f"Version: {result.version}\n"
                    f"Path: {result.path}",
                    border_style="green",
                )
            )
            console.print(
                "\n[yellow]Note:[/yellow] Models are not captured automatically. "
                "Edit bundle.yaml to add model definitions."
            )
        else:
            console.print(f"[red]✗ {result.message}[/red]")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]✗ Failed to create snapshot: {e}[/red]")
        raise typer.Exit(1) from e


# =============================================================================
# Bundle Management Commands
# =============================================================================


@bundle_app.command("list")
def bundle_list(
    bundle_name: Annotated[
        str | None,
        typer.Argument(
            help="Specific bundle name to show versions for.",
        ),
    ] = None,
) -> None:
    """List available bundles or versions of a specific bundle.

    Example:
        acs bundle list              # List all bundles
        acs bundle list wan_2.2_i2v  # List versions of wan_2.2_i2v
    """
    from ai_content_service.bundle import BundleManager, BundleNotFoundError

    settings = get_settings()
    bundle_manager = BundleManager(settings, console)

    if bundle_name:
        # Show versions for specific bundle
        try:
            bundle_info = bundle_manager.get_bundle(bundle_name)

            console.print(
                Panel.fit(
                    f"[bold]Bundle: {bundle_info.name}[/bold]",
                    border_style="cyan",
                )
            )

            if not bundle_info.versions:
                console.print("[yellow]No versions found[/yellow]")
                return

            table = Table(show_header=True)
            table.add_column("Version", style="cyan")
            table.add_column("Current", justify="center")

            for version in bundle_info.versions:
                is_current = "✓" if version == bundle_info.current_version else ""
                table.add_row(version, f"[green]{is_current}[/green]")

            console.print(table)

        except BundleNotFoundError as e:
            console.print(f"[red]✗ {e}[/red]")
            raise typer.Exit(1) from e

    else:
        # List all bundles
        bundles = bundle_manager.list_bundles()

        if not bundles:
            console.print("[yellow]No bundles found[/yellow]")
            console.print(f"[dim]Bundles directory: {settings.bundles_path}[/dim]")
            return

        table = Table(title="Available Bundles", show_header=True)
        table.add_column("Bundle", style="cyan")
        table.add_column("Current Version", style="green")
        table.add_column("Versions", justify="right")

        for bundle in bundles:
            current = bundle.current_version or "[dim]not set[/dim]"
            table.add_row(bundle.name, current, str(len(bundle.versions)))

        console.print(table)


@bundle_app.command("set-current")
def bundle_set_current(
    bundle_name: Annotated[
        str,
        typer.Argument(
            help="Bundle name.",
        ),
    ],
    version: Annotated[
        str,
        typer.Argument(
            help="Version to set as current.",
        ),
    ],
) -> None:
    """Set the current version for a bundle.

    Example:
        acs bundle set-current wan_2.2_i2v 260101-02
    """
    from ai_content_service.bundle import BundleManager, BundleNotFoundError

    settings = get_settings()
    bundle_manager = BundleManager(settings, console)

    try:
        bundle_manager.set_current_version(bundle_name, version)
    except BundleNotFoundError as e:
        console.print(f"[red]✗ {e}[/red]")
        raise typer.Exit(1) from e


@bundle_app.command("delete")
def bundle_delete(
    bundle_name: Annotated[
        str,
        typer.Argument(
            help="Bundle name.",
        ),
    ],
    version: Annotated[
        str,
        typer.Argument(
            help="Version to delete.",
        ),
    ],
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-f",
            help="Skip confirmation prompt.",
        ),
    ] = False,
) -> None:
    """Delete a specific bundle version.

    Example:
        acs bundle delete wan_2.2_i2v 260101-01
    """
    from ai_content_service.bundle import BundleError, BundleManager, BundleNotFoundError

    if not force:
        confirm = typer.confirm(f"Are you sure you want to delete {bundle_name}/{version}?")
        if not confirm:
            console.print("[yellow]Cancelled[/yellow]")
            raise typer.Exit(0)

    settings = get_settings()
    bundle_manager = BundleManager(settings, console)

    try:
        bundle_manager.delete_version(bundle_name, version)
    except (BundleNotFoundError, BundleError) as e:
        console.print(f"[red]✗ {e}[/red]")
        raise typer.Exit(1) from e


@bundle_app.command("show")
def bundle_show(
    bundle_name: Annotated[
        str,
        typer.Argument(
            help="Bundle name.",
        ),
    ],
    version: Annotated[
        str | None,
        typer.Option(
            "--version",
            help="Specific version. Default: current.",
        ),
    ] = None,
) -> None:
    """Show details of a bundle version.

    Example:
        acs bundle show wan_2.2_i2v
        acs bundle show wan_2.2_i2v --version 260101-01
    """
    from ai_content_service.bundle import BundleManager, BundleNotFoundError

    settings = get_settings()
    bundle_manager = BundleManager(settings, console)

    try:
        bundle_files = bundle_manager.load_bundle(bundle_name, version)
        config = bundle_files.bundle_config
        metadata = config.metadata

        console.print(
            Panel.fit(
                f"[bold]Bundle: {metadata.name}[/bold]\n"
                f"Version: {metadata.version}\n"
                f"Description: {metadata.description or '[dim]none[/dim]'}\n"
                f"Created: {metadata.created_at.isoformat()}\n"
                f"Tested: {'Yes' if metadata.tested else 'No'}",
                border_style="cyan",
            )
        )

        # ComfyUI info
        console.print(f"\n[bold]ComfyUI[/bold]: {config.comfyui.commit[:12]}")

        # Custom nodes
        if config.custom_nodes:
            console.print(f"\n[bold]Custom Nodes ({len(config.custom_nodes)}):[/bold]")
            for node in config.custom_nodes:
                console.print(
                    f"  • {node.name} @ {node.commit_sha[:8] if node.commit_sha else 'latest'}"
                )

        # Models
        if config.models:
            console.print(f"\n[bold]Models ({len(config.models)}):[/bold]")
            for model in config.models:
                console.print(f"  • {model.name} ({len(model.files)} files)")

        # Expected nodes from workflow
        expected_nodes = bundle_files.expected_node_types
        if expected_nodes:
            console.print(f"\n[bold]Workflow Node Types ({len(expected_nodes)}):[/bold]")
            for node_type in sorted(expected_nodes)[:10]:
                console.print(f"  • {node_type}")
            if len(expected_nodes) > 10:
                console.print(f"  ... and {len(expected_nodes) - 10} more")

    except BundleNotFoundError as e:
        console.print(f"[red]✗ {e}[/red]")
        raise typer.Exit(1) from e


# =============================================================================
# Status Command
# =============================================================================


@app.command()
def status(
    comfyui_path: Annotated[
        Path,
        typer.Option(
            "--comfyui",
            help="Path to ComfyUI installation.",
        ),
    ] = Path("/workspace/ComfyUI"),
) -> None:
    """Show current deployment status."""
    from ai_content_service.bundle import BundleManager
    from ai_content_service.comfyui import ComfyUISetup
    from ai_content_service.workflows import WorkflowManager

    settings = get_settings()
    settings.comfyui_path = comfyui_path

    comfyui = ComfyUISetup(settings, console)
    workflow_mgr = WorkflowManager(settings, console)
    bundle_mgr = BundleManager(settings, console)

    # Check installation
    console.print(Panel.fit("[bold]Deployment Status[/bold]", border_style="cyan"))

    if comfyui.verify_installation():
        console.print(f"[green]✓ ComfyUI found at {comfyui_path}[/green]")
        current_commit = comfyui.get_current_commit()
        if current_commit:
            console.print(f"  Commit: {current_commit[:12]}")
    else:
        console.print(f"[red]✗ ComfyUI not found at {comfyui_path}[/red]")
        raise typer.Exit(1)

    # List bundles
    bundles = bundle_mgr.list_bundles()
    if bundles:
        console.print(f"\n[bold]Available Bundles ({len(bundles)}):[/bold]")
        for bundle in bundles:
            current = (
                f" [green](current: {bundle.current_version})[/green]"
                if bundle.current_version
                else ""
            )
            console.print(f"  • {bundle.name}{current}")

    # List models
    models = comfyui.list_installed_models()
    if models:
        table = Table(title="Installed Models", show_header=True)
        table.add_column("Type", style="cyan")
        table.add_column("Count", justify="right")
        table.add_column("Files")

        for model_type, files in models.items():
            table.add_row(
                model_type,
                str(len(files)),
                ", ".join(files[:3]) + ("..." if len(files) > 3 else ""),
            )

        console.print(table)
    else:
        console.print("\n[yellow]No models installed[/yellow]")

    # List custom nodes
    nodes = comfyui.list_installed_custom_nodes()
    if nodes:
        console.print(f"\n[bold]Custom Nodes ({len(nodes)}):[/bold]")
        for node in nodes[:10]:
            console.print(f"  • {node}")
        if len(nodes) > 10:
            console.print(f"  ... and {len(nodes) - 10} more")
    else:
        console.print("\n[yellow]No custom nodes installed[/yellow]")

    # List workflows
    workflows = workflow_mgr.list_installed_workflows()
    if workflows:
        table = Table(title="Installed Workflows", show_header=True)
        table.add_column("Name", style="cyan")
        table.add_column("Nodes", justify="right")

        for wf in workflows:
            table.add_row(wf.name, str(wf.node_count))

        console.print(table)
    else:
        console.print("\n[yellow]No workflows installed[/yellow]")


if __name__ == "__main__":
    app()
