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
from ai_content_service.bundle import BundleFiles, BundleInfo, VersionInfo
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
        assert kwargs["bundle_name"] == "test_bundle"
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

    def test_deploy_passes_version(self, settings: Settings) -> None:
        with (
            patch("ai_content_service.cli.get_settings", return_value=settings),
            patch("ai_content_service.cli._run_deploy", new=AsyncMock()) as mock_run,
        ):
            runner.invoke(app, ["deploy", "--bundle", "test_bundle", "--version", "260101-01"])

        _, kwargs = mock_run.call_args
        assert kwargs["version"] == "260101-01"

    def test_deploy_failure_exits_nonzero(self, settings: Settings) -> None:
        async def failing_deploy(**_kwargs):
            raise SystemExit(1)

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
        assert kwargs["bundle_name"] == "env_bundle"


class TestBundleList:
    def test_list_all_bundles(self, settings: Settings) -> None:
        bundles = [
            BundleInfo(name="wan_i2v", current_version="260101-01", versions=["260101-01"]),
            BundleInfo(name="wan_t2v", current_version=None, versions=[]),
        ]
        mock_manager = MagicMock()
        mock_manager.list_bundles.return_value = bundles

        with (
            patch("ai_content_service.cli.get_settings", return_value=settings),
            patch("ai_content_service.bundle.BundleManager", return_value=mock_manager),
        ):
            result = runner.invoke(app, ["bundle", "list"])

        assert result.exit_code == 0
        assert "wan_i2v" in result.output
        assert "wan_t2v" in result.output

    def test_list_versions_for_bundle(self, settings: Settings) -> None:
        versions = [
            VersionInfo(version="260101-01", tested=True, description="First"),
            VersionInfo(version="260101-02", tested=False, description="Second"),
        ]
        mock_manager = MagicMock()
        mock_manager.list_versions.return_value = versions
        mock_manager.get_current_version.return_value = "260101-02"

        with (
            patch("ai_content_service.cli.get_settings", return_value=settings),
            patch("ai_content_service.bundle.BundleManager", return_value=mock_manager),
        ):
            result = runner.invoke(app, ["bundle", "list", "wan_i2v"])

        assert result.exit_code == 0
        assert "260101-01" in result.output
        assert "260101-02" in result.output


class TestBundleShow:
    def test_show_prints_bundle_info(
        self,
        settings: Settings,
        mock_bundle_files: BundleFiles,
    ) -> None:
        mock_manager = MagicMock()
        mock_manager.load_bundle.return_value = mock_bundle_files

        with (
            patch("ai_content_service.cli.get_settings", return_value=settings),
            patch("ai_content_service.bundle.BundleManager", return_value=mock_manager),
        ):
            result = runner.invoke(app, ["bundle", "show", "test_bundle"])

        assert result.exit_code == 0
        assert "test_bundle" in result.output
        assert "260101-01" in result.output

    def test_show_with_version(
        self,
        settings: Settings,
        mock_bundle_files: BundleFiles,
    ) -> None:
        mock_manager = MagicMock()
        mock_manager.load_bundle.return_value = mock_bundle_files

        with (
            patch("ai_content_service.cli.get_settings", return_value=settings),
            patch("ai_content_service.bundle.BundleManager", return_value=mock_manager),
        ):
            result = runner.invoke(app, ["bundle", "show", "test_bundle", "--version", "260101-01"])

        assert result.exit_code == 0
        mock_manager.load_bundle.assert_called_once_with("test_bundle", "260101-01")


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
