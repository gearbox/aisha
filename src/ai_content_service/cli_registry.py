"""CLI commands for bundle registry management."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer
from rich import print as rprint
from rich.console import Console
from rich.table import Table

from .bundle_registry import (
    BundleReference,
    BundleRegistryManager,
    GitBundleRegistry,
    LocalBundleRegistry,
)
from .settings_registry import ExtendedSettings

console = Console()

# Create registry CLI group
registry_app = typer.Typer(
    name="registry",
    help="Manage bundle registries",
    no_args_is_help=True,
)


def get_settings() -> ExtendedSettings:
    """Get application settings."""
    return ExtendedSettings()


def create_registry_manager(settings: ExtendedSettings) -> BundleRegistryManager:
    """Create registry manager from settings."""
    manager = BundleRegistryManager()

    # Add local registry as fallback
    if settings.bundles_path.exists():
        local = LocalBundleRegistry(settings.bundles_path, "local")
        manager.register(local)

    # Add remote registry if configured
    if settings.has_remote_bundles():
        git = GitBundleRegistry(
            repo_url=settings.bundles_repo,  # type: ignore
            local_path=settings.get_bundles_cache_path(),
            name="remote",
            branch=settings.bundles_branch,
            auth_token=settings.github_token,
            ssh_key_path=settings.github_ssh_key,
        )
        manager.register(git, default=True)

    return manager


@registry_app.command("sync")
def sync_registries(
    registry: Annotated[
        str | None,
        typer.Option("--registry", "-r", help="Specific registry to sync"),
    ] = None,
) -> None:
    """Sync bundle registries (git pull)."""
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
            rprint(f"[green]✓[/green] Synced all registries")

    with console.status("[bold blue]Syncing registries..."):
        asyncio.run(_sync())


@registry_app.command("list")
def list_bundles(
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
    """List available bundles."""
    settings = get_settings()
    manager = create_registry_manager(settings)

    async def _list() -> None:
        if sync:
            await manager.sync_all()

        registries = (
            [manager.get(registry)]
            if registry
            else [manager.get(r) for r in manager.list_registries()]
        )

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
                for bundle in index.bundles:
                    # Filter by tags if specified
                    if tag_filter and bundle.tags:
                        if not tag_filter.intersection(bundle.tags):
                            continue

                    table.add_row(
                        reg.name,
                        bundle.name,
                        bundle.description or "-",
                        ", ".join(bundle.tags or []),
                        bundle.default_version or "-",
                    )
            except Exception as e:
                rprint(f"[yellow]Warning: Could not list {reg.name}: {e}[/yellow]")

        console.print(table)

    asyncio.run(_list())


@registry_app.command("versions")
def list_versions(
    bundle: Annotated[str, typer.Argument(help="Bundle name or reference")],
) -> None:
    """List available versions for a bundle."""
    settings = get_settings()
    manager = create_registry_manager(settings)
    ref = BundleReference.parse(bundle)

    async def _versions() -> None:
        registry = (
            manager.get(ref.registry) if ref.registry else manager.default
        )
        if registry is None:
            rprint("[red]No registry available[/red]")
            raise typer.Exit(1)

        versions = await registry.list_versions(ref.name)
        
        rprint(f"\n[bold]Versions for {ref.name}:[/bold]")
        for version in versions:
            rprint(f"  • {version}")

    asyncio.run(_versions())


@registry_app.command("info")
def bundle_info(
    bundle: Annotated[str, typer.Argument(help="Bundle name or reference")],
    sync: Annotated[
        bool,
        typer.Option("--sync/--no-sync", help="Sync before showing info"),
    ] = False,
) -> None:
    """Show detailed information about a bundle."""
    settings = get_settings()
    manager = create_registry_manager(settings)
    ref = BundleReference.parse(bundle)

    async def _info() -> None:
        if sync:
            await manager.sync_all()

        registry = (
            manager.get(ref.registry) if ref.registry else manager.default
        )
        if registry is None:
            rprint("[red]No registry available[/red]")
            raise typer.Exit(1)

        try:
            path = await registry.resolve_bundle_path(ref.name, ref.version)
            bundle_yaml = path / "bundle.yaml"

            if not bundle_yaml.exists():
                rprint(f"[red]Bundle config not found at {path}[/red]")
                raise typer.Exit(1)

            import yaml
            with open(bundle_yaml) as f:
                config = yaml.safe_load(f)

            rprint(f"\n[bold]Bundle: {ref.name}[/bold]")
            rprint(f"  Path: {path}")
            rprint(f"  Registry: {registry.name}")
            
            if metadata := config.get("metadata"):
                rprint(f"\n[bold]Metadata:[/bold]")
                rprint(f"  Version: {metadata.get('version', 'N/A')}")
                rprint(f"  Description: {metadata.get('description', 'N/A')}")
                rprint(f"  Created: {metadata.get('created_at', 'N/A')}")
                rprint(f"  Tested: {metadata.get('tested', False)}")

            if comfyui := config.get("comfyui"):
                rprint(f"\n[bold]ComfyUI:[/bold]")
                rprint(f"  Commit: {comfyui.get('commit', 'N/A')[:12]}...")

            if nodes := config.get("custom_nodes"):
                rprint(f"\n[bold]Custom Nodes ({len(nodes)}):[/bold]")
                for node in nodes:
                    rprint(f"  • {node.get('name', 'Unknown')}")

            if models := config.get("models"):
                total_files = sum(len(m.get("files", [])) for m in models)
                rprint(f"\n[bold]Models ({len(models)} groups, {total_files} files):[/bold]")
                for model in models:
                    rprint(f"  • {model.get('name', 'Unknown')} ({model.get('model_type', 'unknown')})")

        except ValueError as e:
            rprint(f"[red]{e}[/red]")
            raise typer.Exit(1)

    asyncio.run(_info())


# Enhanced deploy command that uses registries
def enhanced_deploy_command(
    bundle: Annotated[
        str,
        typer.Option(
            "--bundle", "-b",
            help="Bundle reference (name, registry/name, or name:version)"
        ),
    ],
    bundles_path: Annotated[
        Path | None,
        typer.Option("--bundles-path", help="Override local bundles path"),
    ] = None,
    bundles_repo: Annotated[
        str | None,
        typer.Option("--bundles-repo", help="Override bundles repository URL"),
    ] = None,
    version: Annotated[
        str | None,
        typer.Option("--version", "-v", help="Specific bundle version"),
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
        typer.Option("--dry-run", "-n", help="Show what would be deployed"),
    ] = False,
    sync: Annotated[
        bool,
        typer.Option("--sync/--no-sync", help="Sync registries before deploy"),
    ] = True,
    comfyui: Annotated[
        Path | None,
        typer.Option("--comfyui", help="Path to ComfyUI installation"),
    ] = None,
) -> None:
    """Deploy a bundle from registry.
    
    Examples:
        # Deploy from default registry
        acs deploy -b wan_2.2_i2v
        
        # Deploy specific version
        acs deploy -b wan_2.2_i2v:260103-01
        
        # Deploy from specific registry
        acs deploy -b remote/wan_2.2_i2v
        
        # Deploy models only (skip ComfyUI setup)
        acs deploy -b wan_2.2_i2v --models-only
    """
    from .config import DeployMode
    from .deployer import Deployer

    settings = get_settings()
    
    # Override settings if provided
    if bundles_path:
        settings.bundles_path = bundles_path
    if bundles_repo:
        settings.bundles_repo = bundles_repo
    if comfyui:
        settings.comfyui_path = comfyui

    manager = create_registry_manager(settings)
    ref = BundleReference.parse(bundle)
    
    # Apply version from option if not in reference
    if version and not ref.version:
        ref = BundleReference(name=ref.name, version=version, registry=ref.registry)

    async def _deploy() -> None:
        # Sync registries
        if sync:
            with console.status("[bold blue]Syncing registries..."):
                await manager.sync_all()
                console.print("[green]✓[/green] Registries synced")

        # Resolve bundle path
        bundle_path = await manager.resolve(ref)
        console.print(f"[green]✓[/green] Resolved bundle: {bundle_path}")

        # Create deployer and run
        mode = DeployMode.MODELS_ONLY if models_only else DeployMode.FULL
        
        # The existing Deployer class needs the bundle path
        # This integrates with the existing deployment infrastructure
        deployer = Deployer(settings)
        result = await deployer.deploy_from_path(
            bundle_path=bundle_path,
            mode=mode,
            verify=not no_verify,
            dry_run=dry_run,
        )
        
        if not result.success:
            raise typer.Exit(1)

    asyncio.run(_deploy())
