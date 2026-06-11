"""Registry management and deploy orchestration (Typer-free core)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .bundle_registry import (
    BundleReference,
    BundleRegistry,
    BundleRegistryManager,
    GitBundleRegistry,
    LocalBundleRegistry,
)

if TYPE_CHECKING:
    from rich.console import Console

    from .config import DeployMode, Settings
    from .deployer import DeploymentResult


def create_registry_manager(settings: Settings) -> BundleRegistryManager:
    """Build a BundleRegistryManager from settings.

    Registers a local registry when bundles_path exists, and a remote git
    registry when bundles_repo is configured (made the default).
    """
    manager = BundleRegistryManager()

    if settings.bundles_path.exists():
        local = LocalBundleRegistry(settings.bundles_path, "local")
        manager.register(local, default=not settings.has_remote_bundles())

    if settings.has_remote_bundles():
        git = GitBundleRegistry(
            repo_url=settings.bundles_repo,  # type: ignore[arg-type]
            local_path=settings.get_bundles_cache_path(),
            name="remote",
            branch=settings.bundles_branch,
            auth_token=settings.github_token,
            ssh_key_path=settings.github_ssh_key,
        )
        manager.register(git, default=True)

    return manager


def get_or_default_registry(manager: BundleRegistryManager, ref: BundleReference) -> BundleRegistry:
    """Return the registry for *ref* (named or default), or raise if none."""
    reg = manager.get(ref.registry) if ref.registry else manager.default
    if reg is None:
        raise ValueError("No registry available")
    return reg


async def run_deploy(
    settings: Settings,
    ref: BundleReference,
    mode: DeployMode,
    verify: bool,
    dry_run: bool,
    sync: bool | None = None,
    console: Console | None = None,
) -> DeploymentResult:
    """Execute a registry-aware deployment.

    Resolves *ref* to a local path via the configured registries, then runs
    the full Deployer pipeline.  All arguments are plain Python — no Typer
    types — so this function is testable without invoking the CLI.

    sync=None defers to settings.auto_sync_registries; True/False override it.
    """
    from rich.console import Console as _Console

    con: Console = console or _Console()
    manager = create_registry_manager(settings)

    if not manager.list_registries():
        raise ValueError(
            "No bundle registries configured. "
            "Set ACS_BUNDLES_PATH (local) or ACS_BUNDLES_REPO (remote)."
        )

    from .bundle import BundleManager
    from .comfyui import ComfyUIManager
    from .deployer import Deployer
    from .downloader import ModelDownloader
    from .provisioning_reporter import ProvisioningReporter
    from .workflows import WorkflowManager

    effective_sync = sync if sync is not None else settings.auto_sync_registries
    if effective_sync:
        with con.status("[bold blue]Syncing registries..."):
            await manager.sync_all()
        con.print("[green]✓[/green] Registries synced")

    bundle_path = await manager.resolve(ref)
    con.print(f"[green]✓[/green] Resolved bundle: {bundle_path}")

    deployer = Deployer(
        settings=settings,
        bundle_manager=BundleManager(settings),
        comfyui_manager=ComfyUIManager(
            settings.comfyui_path, python_executable=settings.comfyui_python
        ),
        model_downloader=ModelDownloader(settings),
        workflow_manager=WorkflowManager(settings.comfyui_path),
        reporter=ProvisioningReporter.from_settings(settings),
    )

    return await deployer.deploy_from_path(
        bundle_path=bundle_path,
        mode=mode,
        verify=verify,
        dry_run=dry_run,
    )
