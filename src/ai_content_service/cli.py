"""CLI for AI Content Service."""

import asyncio
import hashlib
from pathlib import Path
from typing import Annotated
from urllib.parse import parse_qs, urlencode, urlparse

import httpx
import typer
import yaml
from pydantic import ValidationError
from rich import print as rprint
from rich.console import Console
from rich.table import Table

from .bundle_registry import BundleReference
from .config import BundleConfig, DeployMode, Settings, get_settings
from .r2_transfer import R2TransferError, R2WriteCreds
from .r2_transfer import push as r2_push
from .registry_service import create_registry_manager, get_or_default_registry

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
app.add_typer(bundle_app)
app.add_typer(cache_app)

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

    ref = BundleReference.parse(bundle_spec)
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
    ref = BundleReference.parse(bundle)

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
    ref = BundleReference.parse(bundle)

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
        settings = settings.model_copy(update={"comfyui_path": comfyui_path})

    manager = SnapshotManager(
        comfyui_path=settings.comfyui_path,
        bundles_path=settings.bundles_path,
        python_executable=settings.comfyui_python,
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


def _compute_sha256(path: Path) -> str:
    """Compute SHA-256 of a file in chunks."""
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def _strip_token_from_url(url: str) -> str:
    """Remove the Civitai 'token' query param before sending to Apex."""
    parsed = urlparse(url)
    query = parse_qs(parsed.query, keep_blank_values=True)
    query.pop("token", None)
    return parsed._replace(query=urlencode(query, doseq=True)).geturl()


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
) -> None:
    """Push model weights from local disk to the R2 model cache.

    Requires ACS_APEX_ADMIN_TOKEN and ACS_APEX_BASE_URL to be set.
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

    if not settings.apex_admin_token:
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

    # Resolve bundle to get its config
    manager = create_registry_manager(settings)
    ref = BundleReference.parse(bundle)

    async def _resolve() -> Path:
        if sync:
            await manager.sync_all()
        return await manager.resolve(ref)

    try:
        bundle_path = asyncio.run(_resolve())
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from e

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

    # Collect target (model_config, file_config, disk_path) triples
    models_base = settings.comfyui_path / "models"
    targets = []
    for mc in bundle_config.models:
        model_dir = models_base / mc.model_type
        if mc.subdirectory:
            model_dir = model_dir / mc.subdirectory
        for fc in mc.files:
            if model and fc.filename != model:
                continue
            targets.append((mc, fc, model_dir / fc.filename))

    if not targets:
        console.print("[red]Error:[/red] No matching model files found in bundle")
        raise typer.Exit(1)

    admin_token = settings.apex_admin_token.get_secret_value()
    apex_base = settings.apex_base_url.rstrip("/")
    credentials_url = f"{apex_base}/v1/admin/model-cache/credentials"
    finalize_url = f"{apex_base}/v1/admin/model-cache/finalize"

    exit_code = 0
    with httpx.Client(timeout=30.0) as client:
        for mc, fc, file_path in targets:
            console.print(f"\n[bold]Pushing {fc.filename}[/bold]")

            resolved = file_path.resolve()
            if models_base.resolve() not in resolved.parents:
                console.print(f"  [red]✗[/red] Refusing path outside models dir: {fc.filename!r}")
                exit_code = 1
                continue

            if not file_path.exists():
                console.print(f"  [red]✗[/red] File not found on disk: {file_path}")
                exit_code = 1
                continue

            # Step 1: determine sha256 and size_bytes
            sha256 = fc.sha256 or _compute_sha256(file_path)
            size_bytes = file_path.stat().st_size

            # Step 2: mint write credentials from Apex
            source_url = _strip_token_from_url(fc.url)
            console.print("  Minting write credentials from Apex…")
            try:
                resp = client.post(
                    credentials_url,
                    json={
                        "sha256": sha256,
                        "filename": fc.filename,
                        "model_type": mc.model_type,
                        "source_url": source_url,
                    },
                    headers={"Authorization": f"Bearer {admin_token}"},
                )
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                console.print(f"  [red]✗[/red] Apex credentials request failed: {e}")
                exit_code = 1
                continue
            except httpx.HTTPError as e:
                console.print(f"  [red]✗[/red] Apex credentials request error: {e}")
                exit_code = 1
                continue

            cred_data = resp.json()
            r2_key: str = cred_data["r2_key"]
            raw_creds: dict[str, str] = cred_data["credentials"]
            write_creds = R2WriteCreds(
                access_key_id=raw_creds["access_key_id"],
                secret_access_key=raw_creds["secret_access_key"],
                session_token=raw_creds.get("session_token"),
            )

            # Step 3: push via rclone (rclone renders its own --progress to terminal)
            console.print(f"  Uploading to R2 ({size_bytes / 1024 / 1024:.1f} MiB)…")
            try:
                r2_push(
                    src_path=file_path,
                    key=r2_key,
                    creds=write_creds,
                    bucket=settings.r2_model_cache_bucket,
                    endpoint=settings.r2_s3_endpoint,
                    rclone_path=settings.rclone_path,
                    upload_concurrency=settings.rclone_upload_concurrency,
                    chunk_size_mb=settings.rclone_chunk_size_mb,
                    size_bytes=size_bytes,
                    max_timeout_s=settings.rclone_max_transfer_seconds,
                )
            except R2TransferError as e:
                console.print(f"  [red]✗[/red] rclone push failed: {e}")
                exit_code = 1
                continue

            # Step 4: finalize
            console.print("  Finalizing with Apex…")
            try:
                fin_resp = client.post(
                    finalize_url,
                    json={"sha256": sha256, "size_bytes": size_bytes},
                    headers={"Authorization": f"Bearer {admin_token}"},
                )
                if fin_resp.status_code in (409, 422):
                    console.print(
                        f"  [red]✗[/red] Apex finalize rejected ({fin_resp.status_code}): "
                        f"{fin_resp.text}"
                    )
                    exit_code = 1
                    continue
                fin_resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                console.print(f"  [red]✗[/red] Apex finalize failed: {e}")
                exit_code = 1
                continue
            except httpx.HTTPError as e:
                console.print(f"  [red]✗[/red] Apex finalize error: {e}")
                exit_code = 1
                continue

            console.print(f"  [green]✓[/green] cache.push.done — {fc.filename}")

    if exit_code != 0:
        raise typer.Exit(exit_code)


if __name__ == "__main__":
    app()
