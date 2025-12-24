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
    help="AI Content Service - Cloud deployment automation",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

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
    """AI Content Service - Automated model deployment for Cloud + ComfyUI."""
    pass


@app.command()
def deploy(
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            help="Path to deployment configuration YAML file.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    workflows_dir: Annotated[
        Path | None,
        typer.Option(
            "--workflows",
            "-w",
            help="Path to workflows directory.",
            exists=True,
            file_okay=False,
            dir_okay=True,
        ),
    ] = None,
    comfyui_path: Annotated[
        Path | None,
        typer.Option(
            "--comfyui",
            help="Path to ComfyUI installation.",
        ),
    ] = None,
) -> None:
    """Deploy models, custom nodes, and workflows from configuration file."""
    from ai_content_service.deployer import Deployer

    settings = get_settings()
    if comfyui_path:
        settings.comfyui_path = comfyui_path

    deployer = Deployer(settings, console)

    console.print(
        Panel.fit(
            f"[bold]Deploying from[/bold] {config}",
            border_style="cyan",
        )
    )

    report = asyncio.run(deployer.deploy_from_config_file(config, workflows_dir))

    if not report.success:
        raise typer.Exit(1)


@app.command()
def deploy_wan(
    comfyui_path: Annotated[
        Path,
        typer.Option(
            "--comfyui",
            help="Path to ComfyUI installation.",
        ),
    ] = Path("/workspace/ComfyUI"),
    skip_existing: Annotated[
        bool,
        typer.Option(
            "--skip-existing/--force",
            help="Skip already downloaded files.",
        ),
    ] = True,
) -> None:
    """Deploy WAN 2.2 GGUF Q8 models (High Noise + Low Noise variants)."""
    from ai_content_service.config import CustomNode, ModelDefinition, ModelFile, ModelType
    from ai_content_service.deployer import quick_deploy

    settings = get_settings()
    settings.comfyui_path = comfyui_path
    settings.skip_existing = skip_existing

    console.print(
        Panel.fit(
            "[bold cyan]WAN 2.2 GGUF Q8 Deployment[/bold cyan]\n"
            "Models: dasiwaWAN22I2V14B High Noise + Low Noise\n"
            f"Target: {comfyui_path}",
            border_style="cyan",
        )
    )

    # Define WAN 2.2 models
    wan_model = ModelDefinition(
        name="dasiwaWAN22I2V14B-GGUF-Q8",
        description="WAN 2.2 Image-to-Video 14B GGUF Q8 quantized models",
        model_type=ModelType.DIFFUSION,
        files=[
            ModelFile(
                name="WAN 2.2 High Noise Q8",
                url="https://huggingface.co/Bedovyy/dasiwaWAN22I2V14B-GGUF/resolve/main/HighNoise/dasiwaWAN22I2V14B_midnightflirtHigh-Q8_0.gguf",
                filename="dasiwaWAN22I2V14B_midnightflirtHigh-Q8_0.gguf",
                sha256="0ab7f1fc4aa0f17de33877d1d87fef1c538b844c4a3a9decbcc88a741a3af7cd",
                size_bytes=None,
            ),
            ModelFile(
                name="WAN 2.2 Low Noise Q8",
                url="https://huggingface.co/Bedovyy/dasiwaWAN22I2V14B-GGUF/resolve/main/LowNoise/dasiwaWAN22I2V14B_midnightflirtLow-Q8_0.gguf",
                filename="dasiwaWAN22I2V14B_midnightflirtLow-Q8_0.gguf",
                sha256="3176b400b277be4533cfe4330afdeae3111a0cc6705701fe039fb4550bfa6246",
                size_bytes=None,
            ),
        ],
        custom_node_required="https://github.com/city96/ComfyUI-GGUF",
        subfolder=None
    )

    # ComfyUI-GGUF custom node is required for GGUF models
    gguf_node = CustomNode(
        name="ComfyUI-GGUF",
        git_url="https://github.com/city96/ComfyUI-GGUF",
        commit_sha=None,
    )

    report = asyncio.run(quick_deploy([wan_model], [gguf_node], settings))

    if not report.success:
        raise typer.Exit(1)

    console.print("\n[bold green]WAN 2.2 models ready for use in ComfyUI![/bold green]")


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
    from ai_content_service.comfyui import ComfyUISetup
    from ai_content_service.workflows import WorkflowManager

    settings = get_settings()
    settings.comfyui_path = comfyui_path

    comfyui = ComfyUISetup(settings, console)
    workflow_mgr = WorkflowManager(settings, console)

    # Check installation
    console.print(Panel.fit("[bold]Deployment Status[/bold]", border_style="cyan"))

    if comfyui.verify_installation():
        console.print(f"[green]✓ ComfyUI found at {comfyui_path}[/green]")
    else:
        console.print(f"[red]✗ ComfyUI not found at {comfyui_path}[/red]")
        raise typer.Exit(1)

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
        console.print("[yellow]No models installed[/yellow]")

    # List custom nodes
    nodes = comfyui.list_installed_custom_nodes()
    if nodes:
        console.print(f"\n[bold]Custom Nodes ({len(nodes)}):[/bold]")
        for node in nodes:
            console.print(f"  • {node}")
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


@app.command()
def install_workflow(
    workflow_file: Annotated[
        Path,
        typer.Argument(
            help="Path to workflow JSON file.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    comfyui_path: Annotated[
        Path,
        typer.Option(
            "--comfyui",
            help="Path to ComfyUI installation.",
        ),
    ] = Path("/workspace/ComfyUI"),
) -> None:
    """Install a single ComfyUI workflow file."""
    from ai_content_service.workflows import WorkflowError, WorkflowManager

    settings = get_settings()
    settings.comfyui_path = comfyui_path

    workflow_mgr = WorkflowManager(settings, console)

    try:
        info = workflow_mgr.install_workflow(workflow_file)
        console.print(f"[green]✓ Workflow installed:[/green] {info.name} ({info.node_count} nodes)")
    except WorkflowError as e:
        console.print(f"[red]✗ Failed to install workflow: {e}[/red]")
        raise typer.Exit(1) from e


@app.command()
def install_node(
    git_url: Annotated[
        str,
        typer.Argument(
            help="Git repository URL for the custom node.",
        ),
    ],
    name: Annotated[
        str | None,
        typer.Option(
            "--name",
            "-n",
            help="Custom node directory name (default: derived from URL).",
        ),
    ] = None,
    commit: Annotated[
        str | None,
        typer.Option(
            "--commit",
            help="Specific commit SHA to checkout.",
        ),
    ] = None,
    comfyui_path: Annotated[
        Path,
        typer.Option(
            "--comfyui",
            help="Path to ComfyUI installation.",
        ),
    ] = Path("/workspace/ComfyUI"),
) -> None:
    """Install a ComfyUI custom node from git repository."""
    from ai_content_service.comfyui import ComfyUISetup
    from ai_content_service.config import CustomNode

    settings = get_settings()
    settings.comfyui_path = comfyui_path

    # Derive name from URL if not provided
    if name is None:
        name = git_url.rstrip("/").split("/")[-1].replace(".git", "")

    node = CustomNode(
        name=name,
        git_url=git_url,
        commit_sha=commit,
    )

    comfyui = ComfyUISetup(settings, console)

    if not comfyui.verify_installation():
        console.print("[red]✗ ComfyUI not found[/red]")
        raise typer.Exit(1)

    result = asyncio.run(comfyui.install_custom_node(node))

    if result.success:
        console.print(f"[green]✓ {result.name}: {result.message}[/green]")
    else:
        console.print(f"[red]✗ {result.name}: {result.message}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
