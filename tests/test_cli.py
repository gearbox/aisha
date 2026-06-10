"""Smoke tests for CLI commands."""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from ai_content_service import __version__
from ai_content_service.bundle import BundleFiles
from ai_content_service.bundle_registry import BundleReference
from ai_content_service.cli import app
from ai_content_service.comfyui import ComfyUIStatus
from ai_content_service.config import (
    BundleConfig,
    BundleMetadata,
    DeployMode,
    Settings,
    reset_settings,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

runner = CliRunner()


@pytest.fixture(autouse=True)
def reset_settings_singleton() -> Iterator[None]:
    reset_settings()
    yield
    reset_settings()


@pytest.fixture
def temp_dir() -> Iterator[Path]:
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def settings(temp_dir: Path) -> Settings:
    return Settings(
        comfyui_path=temp_dir / "ComfyUI",
        bundles_path=temp_dir / "bundles",
    )


@pytest.fixture
def minimal_bundle_config() -> BundleConfig:
    return BundleConfig(
        metadata=BundleMetadata(
            name="test_bundle",
            version="260101-01",
            description="Test bundle",
            created_at=datetime.now(timezone.utc),
        ),
        models=[],
        workflow_file="workflow.json",
    )


@pytest.fixture
def mock_bundle_files(minimal_bundle_config: BundleConfig) -> BundleFiles:
    bf = MagicMock(spec=BundleFiles)
    bf.bundle_config = minimal_bundle_config
    return bf


class TestVersion:
    def test_version_flag_prints_version(self) -> None:
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert __version__ in result.output

    def test_short_version_flag(self) -> None:
        result = runner.invoke(app, ["-v"])
        assert result.exit_code == 0
        assert __version__ in result.output


class TestDeploy:
    def test_no_bundle_exits_with_error(self, settings: Settings) -> None:
        with patch("ai_content_service.cli.get_settings", return_value=settings):
            result = runner.invoke(app, ["deploy"])
        assert result.exit_code == 1
        assert "No bundle specified" in result.output

    def test_deploy_dry_run(self, settings: Settings) -> None:
        with (
            patch("ai_content_service.cli.get_settings", return_value=settings),
            patch("ai_content_service.cli._run_deploy", new=AsyncMock()) as mock_run,
        ):
            result = runner.invoke(app, ["deploy", "--bundle", "test_bundle", "--dry-run"])

        assert result.exit_code == 0
        mock_run.assert_called_once()
        _, kwargs = mock_run.call_args
        assert isinstance(kwargs["ref"], BundleReference)
        assert kwargs["ref"].name == "test_bundle"
        assert kwargs["dry_run"] is True
        assert kwargs["mode"] == DeployMode.FULL

    def test_deploy_models_only_shows_message(self, settings: Settings) -> None:
        with (
            patch("ai_content_service.cli.get_settings", return_value=settings),
            patch("ai_content_service.cli._run_deploy", new=AsyncMock()),
        ):
            result = runner.invoke(
                app, ["deploy", "--bundle", "test_bundle", "--models-only", "--dry-run"]
            )

        assert result.exit_code == 0
        assert "Models-only mode" in result.output

    def test_deploy_models_only_sets_mode(self, settings: Settings) -> None:
        with (
            patch("ai_content_service.cli.get_settings", return_value=settings),
            patch("ai_content_service.cli._run_deploy", new=AsyncMock()) as mock_run,
        ):
            runner.invoke(app, ["deploy", "--bundle", "test_bundle", "--models-only"])

        _, kwargs = mock_run.call_args
        assert kwargs["mode"] == DeployMode.MODELS_ONLY

    def test_deploy_passes_bundle_version(self, settings: Settings) -> None:
        with (
            patch("ai_content_service.cli.get_settings", return_value=settings),
            patch("ai_content_service.cli._run_deploy", new=AsyncMock()) as mock_run,
        ):
            runner.invoke(
                app, ["deploy", "--bundle", "test_bundle", "--bundle-version", "260101-01"]
            )

        _, kwargs = mock_run.call_args
        assert kwargs["ref"].version == "260101-01"

    def test_deploy_version_embedded_in_bundle_ref(self, settings: Settings) -> None:
        with (
            patch("ai_content_service.cli.get_settings", return_value=settings),
            patch("ai_content_service.cli._run_deploy", new=AsyncMock()) as mock_run,
        ):
            runner.invoke(app, ["deploy", "--bundle", "test_bundle:260101-01"])

        _, kwargs = mock_run.call_args
        assert kwargs["ref"].version == "260101-01"
        assert kwargs["ref"].name == "test_bundle"

    def test_deploy_failure_exits_nonzero(self, settings: Settings) -> None:
        with (
            patch("ai_content_service.cli.get_settings", return_value=settings),
            patch("ai_content_service.cli._run_deploy", new=AsyncMock(side_effect=SystemExit(1))),
        ):
            result = runner.invoke(app, ["deploy", "--bundle", "test_bundle"])

        assert result.exit_code != 0

    def test_deploy_uses_bundle_from_env(self, settings: Settings) -> None:
        settings_with_bundle = Settings(
            comfyui_path=settings.comfyui_path,
            bundles_path=settings.bundles_path,
            bundle="env_bundle",
        )
        with (
            patch("ai_content_service.cli.get_settings", return_value=settings_with_bundle),
            patch("ai_content_service.cli._run_deploy", new=AsyncMock()) as mock_run,
        ):
            result = runner.invoke(app, ["deploy"])

        assert result.exit_code == 0
        _, kwargs = mock_run.call_args
        assert kwargs["ref"].name == "env_bundle"

    def test_version_flag_does_not_conflict_with_deploy(self) -> None:
        result = runner.invoke(app, ["-v"])
        assert result.exit_code == 0
        assert __version__ in result.output

    def test_deploy_help_shows_bundle_version_not_version(self) -> None:
        result = runner.invoke(app, ["deploy", "--help"])
        assert result.exit_code == 0
        assert "--bundle-version" in result.output
        # --version must not appear as a deploy option (it lives at the app level)
        assert "  --version" not in result.output


class TestBundleList:
    def _make_manager(self) -> tuple[MagicMock, MagicMock]:
        """Return (mock_manager, mock_registry)."""
        entry = MagicMock()
        entry.name = "wan_i2v"
        entry.description = "WAN I2V"
        entry.tags = []
        entry.default_version = "260101-01"

        index = MagicMock()
        index.bundles = [entry]

        reg = AsyncMock()
        reg.name = "local"
        reg.get_index = AsyncMock(return_value=index)

        manager = MagicMock()
        manager.list_registries.return_value = ["local"]
        manager.get.return_value = reg

        return manager, reg

    def test_list_all_bundles(self, settings: Settings) -> None:
        mock_manager, _ = self._make_manager()

        with (
            patch("ai_content_service.cli.get_settings", return_value=settings),
            patch("ai_content_service.cli.create_registry_manager", return_value=mock_manager),
        ):
            result = runner.invoke(app, ["bundle", "list"])

        assert result.exit_code == 0
        assert "wan_i2v" in result.output

    def test_list_specific_registry(self, settings: Settings) -> None:
        mock_manager, mock_reg = self._make_manager()
        mock_manager.get.side_effect = lambda name: mock_reg if name == "local" else None

        with (
            patch("ai_content_service.cli.get_settings", return_value=settings),
            patch("ai_content_service.cli.create_registry_manager", return_value=mock_manager),
        ):
            result = runner.invoke(app, ["bundle", "list", "--registry", "local"])

        assert result.exit_code == 0
        assert "wan_i2v" in result.output


class TestBundleVersions:
    def test_list_versions_for_bundle(self, settings: Settings) -> None:
        mock_reg = AsyncMock()
        mock_reg.name = "local"
        mock_reg.list_versions = AsyncMock(return_value=["260101-02", "260101-01"])

        mock_manager = MagicMock()
        mock_manager.default = mock_reg
        mock_manager.get.return_value = None  # no named registry; use default

        with (
            patch("ai_content_service.cli.get_settings", return_value=settings),
            patch("ai_content_service.cli.create_registry_manager", return_value=mock_manager),
        ):
            result = runner.invoke(app, ["bundle", "versions", "wan_i2v"])

        assert result.exit_code == 0
        assert "260101-01" in result.output
        assert "260101-02" in result.output

    def test_no_registry_exits_with_error(self, settings: Settings) -> None:
        mock_manager = MagicMock()
        mock_manager.default = None
        mock_manager.get.return_value = None

        with (
            patch("ai_content_service.cli.get_settings", return_value=settings),
            patch("ai_content_service.cli.create_registry_manager", return_value=mock_manager),
        ):
            result = runner.invoke(app, ["bundle", "versions", "wan_i2v"])

        assert result.exit_code != 0


class TestBundleShow:
    def _make_manager_with_bundle(self, bundle_dir: Path) -> MagicMock:
        mock_reg = AsyncMock()
        mock_reg.name = "local"
        mock_reg.resolve_bundle_path = AsyncMock(return_value=bundle_dir)

        mock_manager = MagicMock()
        mock_manager.default = mock_reg
        mock_manager.get.return_value = None  # use default

        return mock_manager

    def test_show_prints_bundle_info(self, settings: Settings, temp_dir: Path) -> None:
        bundle_dir = temp_dir / "test_bundle_show"
        bundle_dir.mkdir()
        (bundle_dir / "bundle.yaml").write_text(
            "metadata:\n  name: test_bundle\n  version: '260101-01'\n  description: Test bundle\n"
        )

        mock_manager = self._make_manager_with_bundle(bundle_dir)

        with (
            patch("ai_content_service.cli.get_settings", return_value=settings),
            patch("ai_content_service.cli.create_registry_manager", return_value=mock_manager),
        ):
            result = runner.invoke(app, ["bundle", "show", "test_bundle"])

        assert result.exit_code == 0
        assert "test_bundle" in result.output
        assert "260101-01" in result.output

    def test_show_missing_bundle_yaml_exits(self, settings: Settings, temp_dir: Path) -> None:
        bundle_dir = temp_dir / "empty_bundle"
        bundle_dir.mkdir()

        mock_manager = self._make_manager_with_bundle(bundle_dir)

        with (
            patch("ai_content_service.cli.get_settings", return_value=settings),
            patch("ai_content_service.cli.create_registry_manager", return_value=mock_manager),
        ):
            result = runner.invoke(app, ["bundle", "show", "empty_bundle"])

        assert result.exit_code != 0


class TestBundleSync:
    def test_sync_all(self, settings: Settings) -> None:
        mock_manager = MagicMock()
        mock_manager.sync_all = AsyncMock()

        with (
            patch("ai_content_service.cli.get_settings", return_value=settings),
            patch("ai_content_service.cli.create_registry_manager", return_value=mock_manager),
        ):
            result = runner.invoke(app, ["bundle", "sync"])

        assert result.exit_code == 0
        mock_manager.sync_all.assert_called_once()

    def test_sync_specific_registry(self, settings: Settings) -> None:
        mock_reg = AsyncMock()
        mock_reg.sync = AsyncMock()

        mock_manager = MagicMock()
        mock_manager.get.return_value = mock_reg

        with (
            patch("ai_content_service.cli.get_settings", return_value=settings),
            patch("ai_content_service.cli.create_registry_manager", return_value=mock_manager),
        ):
            result = runner.invoke(app, ["bundle", "sync", "--registry", "remote"])

        assert result.exit_code == 0
        mock_reg.sync.assert_called_once()

    def test_sync_unknown_registry_exits(self, settings: Settings) -> None:
        mock_manager = MagicMock()
        mock_manager.get.return_value = None

        with (
            patch("ai_content_service.cli.get_settings", return_value=settings),
            patch("ai_content_service.cli.create_registry_manager", return_value=mock_manager),
        ):
            result = runner.invoke(app, ["bundle", "sync", "--registry", "nonexistent"])

        assert result.exit_code != 0


class TestBundleSetCurrent:
    def test_set_current_succeeds(self, settings: Settings) -> None:
        mock_manager = MagicMock()

        with (
            patch("ai_content_service.cli.get_settings", return_value=settings),
            patch("ai_content_service.bundle.BundleManager", return_value=mock_manager),
        ):
            result = runner.invoke(app, ["bundle", "set-current", "test_bundle", "260101-02"])

        assert result.exit_code == 0
        assert "260101-02" in result.output
        mock_manager.set_current_version.assert_called_once_with("test_bundle", "260101-02")


class TestBundleDelete:
    def test_delete_with_force(self, settings: Settings) -> None:
        mock_manager = MagicMock()

        with (
            patch("ai_content_service.cli.get_settings", return_value=settings),
            patch("ai_content_service.bundle.BundleManager", return_value=mock_manager),
        ):
            result = runner.invoke(app, ["bundle", "delete", "test_bundle", "260101-01", "--force"])

        assert result.exit_code == 0
        mock_manager.delete_version.assert_called_once_with("test_bundle", "260101-01")
        assert "Deleted" in result.output


class TestStatus:
    def test_status_shows_info(self, settings: Settings) -> None:
        status = ComfyUIStatus(commit="abc1234", custom_node_count=3, is_running=True)
        mock_manager = MagicMock()
        mock_manager.get_status = AsyncMock(return_value=status)

        with (
            patch("ai_content_service.cli.get_settings", return_value=settings),
            patch("ai_content_service.comfyui.ComfyUIManager", return_value=mock_manager),
        ):
            result = runner.invoke(app, ["status"])

        assert result.exit_code == 0
        assert "abc1234" in result.output
        assert "3" in result.output

    def test_status_not_running(self, settings: Settings) -> None:
        status = ComfyUIStatus(commit=None, custom_node_count=0, is_running=False)
        mock_manager = MagicMock()
        mock_manager.get_status = AsyncMock(return_value=status)

        with (
            patch("ai_content_service.cli.get_settings", return_value=settings),
            patch("ai_content_service.comfyui.ComfyUIManager", return_value=mock_manager),
        ):
            result = runner.invoke(app, ["status"])

        assert result.exit_code == 0
        assert "No" in result.output


class TestSnapshot:
    def test_snapshot_creates_bundle(self, settings: Settings, temp_dir: Path) -> None:
        workflow_file = temp_dir / "workflow.json"
        workflow_file.write_text("{}")

        mock_manager = MagicMock()
        mock_manager.create_snapshot = AsyncMock(return_value="260101-01")

        with (
            patch("ai_content_service.cli.get_settings", return_value=settings),
            patch("ai_content_service.snapshot.SnapshotManager", return_value=mock_manager),
        ):
            result = runner.invoke(
                app,
                [
                    "snapshot",
                    "--name",
                    "test_bundle",
                    "--workflow",
                    str(workflow_file),
                    "--description",
                    "Test snapshot",
                ],
            )

        assert result.exit_code == 0
        assert "test_bundle" in result.output
        assert "260101-01" in result.output
        mock_manager.create_snapshot.assert_called_once_with(
            name="test_bundle",
            workflow_path=workflow_file,
            description="Test snapshot",
            extra_model_paths=None,
        )


class TestRegistryService:
    """Tests for registry_service.py (Typer-free core)."""

    def test_create_registry_manager_importable_without_typer(self) -> None:
        from ai_content_service.registry_service import create_registry_manager

        assert callable(create_registry_manager)

    def test_run_deploy_importable_without_typer(self) -> None:
        from ai_content_service.registry_service import run_deploy

        assert callable(run_deploy)

    def test_settings_has_registry_fields(self) -> None:
        settings = Settings()
        assert hasattr(settings, "cache_path")
        assert hasattr(settings, "bundles_repo")
        assert hasattr(settings, "bundles_branch")
        assert hasattr(settings, "github_token")
        assert hasattr(settings, "github_ssh_key")
        assert hasattr(settings, "auto_sync_registries")
        assert settings.has_remote_bundles() is False

    def test_settings_get_bundles_cache_path(self, temp_dir: Path) -> None:
        settings = Settings(cache_path=temp_dir / "cache")
        assert settings.get_bundles_cache_path() == temp_dir / "cache" / "ai-bundles"

    def test_env_vars_populate_registry_fields(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ACS_BUNDLES_REPO", "https://github.com/test/repo")
        monkeypatch.setenv("ACS_GITHUB_TOKEN", "ghp_test")
        monkeypatch.setenv("ACS_BUNDLES_BRANCH", "dev")
        reset_settings()

        settings = Settings()
        assert settings.bundles_repo == "https://github.com/test/repo"
        assert settings.github_token == "ghp_test"
        assert settings.bundles_branch == "dev"
        assert settings.has_remote_bundles() is True

    def test_create_registry_manager_local_only(self, temp_dir: Path) -> None:
        from ai_content_service.registry_service import create_registry_manager

        bundles = temp_dir / "bundles"
        bundles.mkdir()
        settings = Settings(bundles_path=bundles)
        manager = create_registry_manager(settings)

        assert "local" in manager.list_registries()
        assert "remote" not in manager.list_registries()

    def test_create_registry_manager_no_registries_when_path_missing(self, temp_dir: Path) -> None:
        from ai_content_service.registry_service import create_registry_manager

        settings = Settings(bundles_path=temp_dir / "nonexistent")
        manager = create_registry_manager(settings)

        assert manager.list_registries() == []

    def test_acs_bundle_env_still_works(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ACS_BUNDLE", "wan_2.2_i2v")
        reset_settings()
        settings = Settings()
        assert settings.bundle == "wan_2.2_i2v"
