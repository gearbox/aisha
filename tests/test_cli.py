"""Smoke tests for CLI commands."""

from __future__ import annotations

import json
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from ai_content_service import __version__
from ai_content_service import preflight as preflight_module
from ai_content_service.bundle import BundleFiles
from ai_content_service.bundle_contract import ContractReport
from ai_content_service.bundle_registry import BundleReference
from ai_content_service.bundle_resolution import BundleResolutionError, ResolvedBundle
from ai_content_service.cli import app
from ai_content_service.comfyui import ComfyUIStatus
from ai_content_service.config import (
    BundleConfig,
    BundleMetadata,
    DeployMode,
    Settings,
    reset_settings,
)
from ai_content_service.preflight import BundleCheckResult
from ai_content_service.snapshot import CarryForwardReport, CustomNodeScanReport, CustomNodeSkip

if TYPE_CHECKING:
    import asyncio
    from collections.abc import Iterator

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

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
        cache_path=temp_dir / "cache",
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


class TestMainCallbackResilience:
    """The app callback resolves settings/logging before any subcommand runs.

    It must not do so on a --help invocation (Typer/Click show --help for a
    subcommand only after the parent callback has already run), and it must
    not surface a raw traceback when settings fail to validate.
    """

    def test_subcommand_help_ignores_broken_settings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ACS_COMFYUI_PYTHON", "/nonexistent/python")
        monkeypatch.setattr(sys, "argv", ["acs", "bundle", "list", "--help"])

        result = runner.invoke(app, ["bundle", "list", "--help"])

        assert result.exit_code == 0
        assert result.exception is None
        assert "Usage" in result.output

    def test_invalid_settings_gives_clean_error_not_traceback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ACS_COMFYUI_PYTHON", "/nonexistent/python")
        monkeypatch.setattr(sys, "argv", ["acs", "bundle", "list"])

        result = runner.invoke(app, ["bundle", "list"])

        assert result.exit_code != 0
        assert "Invalid configuration" in result.output
        assert "Traceback" not in result.output


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

        assert result.exit_code == 1

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
        # Strip ANSI codes — Rich may inject styling escapes inside option names.
        clean = _ANSI.sub("", result.output)
        assert "--bundle-version" in clean
        # --version must not appear as a deploy option (it lives at the app level)
        assert "  --version" not in clean

    def test_deploy_honors_settings_no_verify(self, settings: Settings) -> None:
        settings_no_verify = Settings(
            comfyui_path=settings.comfyui_path,
            bundles_path=settings.bundles_path,
            no_verify=True,
        )
        with (
            patch("ai_content_service.cli.get_settings", return_value=settings_no_verify),
            patch("ai_content_service.cli._run_deploy", new=AsyncMock()) as mock_run,
        ):
            result = runner.invoke(app, ["deploy", "--bundle", "test_bundle"])

        assert result.exit_code == 0
        _, kwargs = mock_run.call_args
        assert kwargs["verify"] is False

    def test_deploy_no_verify_flag(self, settings: Settings) -> None:
        with (
            patch("ai_content_service.cli.get_settings", return_value=settings),
            patch("ai_content_service.cli._run_deploy", new=AsyncMock()) as mock_run,
        ):
            result = runner.invoke(app, ["deploy", "--bundle", "test_bundle", "--no-verify"])

        assert result.exit_code == 0
        _, kwargs = mock_run.call_args
        assert kwargs["verify"] is False

    def test_deploy_sync_flag_tristate(self, settings: Settings) -> None:
        with (
            patch("ai_content_service.cli.get_settings", return_value=settings),
            patch("ai_content_service.cli._run_deploy", new=AsyncMock()) as mock_run,
        ):
            runner.invoke(app, ["deploy", "--bundle", "test_bundle"])
        _, kwargs = mock_run.call_args
        assert kwargs["sync"] is None

        with (
            patch("ai_content_service.cli.get_settings", return_value=settings),
            patch("ai_content_service.cli._run_deploy", new=AsyncMock()) as mock_run,
        ):
            runner.invoke(app, ["deploy", "--bundle", "test_bundle", "--sync"])
        _, kwargs = mock_run.call_args
        assert kwargs["sync"] is True

        with (
            patch("ai_content_service.cli.get_settings", return_value=settings),
            patch("ai_content_service.cli._run_deploy", new=AsyncMock()) as mock_run,
        ):
            runner.invoke(app, ["deploy", "--bundle", "test_bundle", "--no-sync"])
        _, kwargs = mock_run.call_args
        assert kwargs["sync"] is False

    def test_deploy_version_from_settings(self, settings: Settings) -> None:
        settings_with_version = Settings(
            comfyui_path=settings.comfyui_path,
            bundles_path=settings.bundles_path,
            bundle_version="260101-01",
        )
        with (
            patch("ai_content_service.cli.get_settings", return_value=settings_with_version),
            patch("ai_content_service.cli._run_deploy", new=AsyncMock()) as mock_run,
        ):
            runner.invoke(app, ["deploy", "--bundle", "test_bundle"])

        _, kwargs = mock_run.call_args
        assert kwargs["ref"].version == "260101-01"

    def test_deploy_does_not_mutate_singleton(self, settings: Settings, temp_dir: Path) -> None:
        original_path = settings.comfyui_path
        with (
            patch("ai_content_service.cli.get_settings", return_value=settings),
            patch("ai_content_service.cli._run_deploy", new=AsyncMock()),
        ):
            runner.invoke(app, ["deploy", "--bundle", "test_bundle", "--comfyui", str(temp_dir)])
        assert settings.comfyui_path == original_path

    def test_deploy_no_registries_friendly_error(self, temp_dir: Path) -> None:
        settings = Settings(bundles_path=temp_dir / "nonexistent")
        with patch("ai_content_service.cli.get_settings", return_value=settings):
            result = runner.invoke(app, ["deploy", "--bundle", "wan_2.2_i2v"])
        assert result.exit_code == 1
        assert "No bundle registries configured" in result.output
        assert "Traceback" not in result.output

    def test_deploy_unresolvable_bundle_friendly_error(self, settings: Settings) -> None:
        with (
            patch("ai_content_service.cli.get_settings", return_value=settings),
            patch(
                "ai_content_service.cli._run_deploy",
                new=AsyncMock(side_effect=ValueError("Bundle 'x' not found in registry 'local'")),
            ),
        ):
            result = runner.invoke(app, ["deploy", "--bundle", "x"])
        assert result.exit_code == 1
        assert "not found" in result.output
        assert "Traceback" not in result.output


class TestBundleList:
    def _make_manager(self) -> tuple[MagicMock, MagicMock]:
        """Return (mock_manager, mock_registry) with one tagged and one untagged bundle."""
        entry_tagged = MagicMock()
        entry_tagged.name = "wan_i2v"
        entry_tagged.description = "WAN I2V"
        entry_tagged.tags = ["video", "core"]
        entry_tagged.default_version = "260101-01"

        entry_untagged = MagicMock()
        entry_untagged.name = "other_bundle"
        entry_untagged.description = "Other"
        entry_untagged.tags = []
        entry_untagged.default_version = None

        index = MagicMock()
        index.bundles = [entry_tagged, entry_untagged]

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

    def test_list_filters_by_tags(self, settings: Settings) -> None:
        mock_manager, _ = self._make_manager()

        with (
            patch("ai_content_service.cli.get_settings", return_value=settings),
            patch("ai_content_service.cli.create_registry_manager", return_value=mock_manager),
        ):
            result = runner.invoke(app, ["bundle", "list", "--tags", "video"])

        assert result.exit_code == 0
        assert "wan_i2v" in result.output
        assert "other_bundle" not in result.output

    def test_list_tag_whitespace(self, settings: Settings) -> None:
        mock_manager, _ = self._make_manager()

        with (
            patch("ai_content_service.cli.get_settings", return_value=settings),
            patch("ai_content_service.cli.create_registry_manager", return_value=mock_manager),
        ):
            result = runner.invoke(app, ["bundle", "list", "--tags", "video, core"])

        assert result.exit_code == 0
        assert "wan_i2v" in result.output
        assert "other_bundle" not in result.output

    def test_list_sync_calls_sync_all(self, settings: Settings) -> None:
        mock_manager, _ = self._make_manager()
        mock_manager.sync_all = AsyncMock()

        with (
            patch("ai_content_service.cli.get_settings", return_value=settings),
            patch("ai_content_service.cli.create_registry_manager", return_value=mock_manager),
        ):
            result = runner.invoke(app, ["bundle", "list", "--sync"])

        assert result.exit_code == 0
        mock_manager.sync_all.assert_called_once()

    def test_list_unknown_registry_fails(self, settings: Settings) -> None:
        mock_manager, _ = self._make_manager()
        mock_manager.get.return_value = None

        with (
            patch("ai_content_service.cli.get_settings", return_value=settings),
            patch("ai_content_service.cli.create_registry_manager", return_value=mock_manager),
        ):
            result = runner.invoke(app, ["bundle", "list", "--registry", "nope"])

        assert result.exit_code != 0
        assert "not found" in result.output

    def test_list_registry_failure_warns(self, settings: Settings) -> None:
        mock_manager, mock_reg = self._make_manager()
        mock_reg.get_index = AsyncMock(side_effect=RuntimeError("connection refused"))

        with (
            patch("ai_content_service.cli.get_settings", return_value=settings),
            patch("ai_content_service.cli.create_registry_manager", return_value=mock_manager),
        ):
            result = runner.invoke(app, ["bundle", "list"])

        assert result.exit_code == 0
        assert "Warning: Could not list" in result.output


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

    def test_versions_qualified_ref_uses_named_registry(self, settings: Settings) -> None:
        remote_reg = AsyncMock()
        remote_reg.name = "remote"
        remote_reg.list_versions = AsyncMock(return_value=["260101-01"])

        default_reg = AsyncMock()
        default_reg.name = "local"
        default_reg.list_versions = AsyncMock(return_value=["260101-02"])

        mock_manager = MagicMock()
        mock_manager.default = default_reg
        mock_manager.get.side_effect = lambda name: remote_reg if name == "remote" else None

        with (
            patch("ai_content_service.cli.get_settings", return_value=settings),
            patch("ai_content_service.cli.create_registry_manager", return_value=mock_manager),
        ):
            result = runner.invoke(app, ["bundle", "versions", "remote/wan_i2v"])

        assert result.exit_code == 0
        remote_reg.list_versions.assert_awaited_once_with("wan_i2v")
        default_reg.list_versions.assert_not_awaited()


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

    def test_show_non_dict_yaml_exits(self, settings: Settings, temp_dir: Path) -> None:
        bundle_dir = temp_dir / "list_bundle"
        bundle_dir.mkdir()
        (bundle_dir / "bundle.yaml").write_text("- item1\n- item2\n")

        mock_manager = self._make_manager_with_bundle(bundle_dir)

        with (
            patch("ai_content_service.cli.get_settings", return_value=settings),
            patch("ai_content_service.cli.create_registry_manager", return_value=mock_manager),
        ):
            result = runner.invoke(app, ["bundle", "show", "list_bundle"])

        assert result.exit_code != 0
        # Rich may wrap error text across lines depending on terminal width.
        assert "expected a mapping" in " ".join(result.output.split())


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


class TestBundleValidate:
    def test_comfyui_url_option_overrides_environment_setting(self, settings: Settings) -> None:
        validate = AsyncMock(return_value=(ContractReport("demo", ()),))

        with (
            patch("ai_content_service.cli.get_settings", return_value=settings),
            patch("ai_content_service.cli.create_registry_manager"),
            patch("ai_content_service.cli.validate_bundle_contracts", new=validate),
        ):
            result = runner.invoke(
                app,
                ["bundle", "validate", "demo", "--comfyui-url", "http://localhost:18188"],
            )

        assert result.exit_code == 0
        assert validate.call_args.kwargs["comfyui_url"] == "http://localhost:18188"

    def test_comfyui_url_uses_environment_setting_when_option_is_omitted(
        self, settings: Settings
    ) -> None:
        configured = settings.model_copy(update={"comfyui_url": "http://comfy:18188"})
        validate = AsyncMock(return_value=(ContractReport("demo", ()),))

        with (
            patch("ai_content_service.cli.get_settings", return_value=configured),
            patch("ai_content_service.cli.create_registry_manager"),
            patch("ai_content_service.cli.validate_bundle_contracts", new=validate),
        ):
            result = runner.invoke(app, ["bundle", "validate", "demo"])

        assert result.exit_code == 0
        assert validate.call_args.kwargs["comfyui_url"] == "http://comfy:18188"


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

    def test_status_does_not_mutate_singleton(self, settings: Settings, temp_dir: Path) -> None:
        original_path = settings.comfyui_path
        status = ComfyUIStatus(commit=None, custom_node_count=0, is_running=False)
        mock_manager = MagicMock()
        mock_manager.get_status = AsyncMock(return_value=status)

        with (
            patch("ai_content_service.cli.get_settings", return_value=settings),
            patch("ai_content_service.comfyui.ComfyUIManager", return_value=mock_manager),
        ):
            runner.invoke(app, ["status", "--comfyui", str(temp_dir)])

        assert settings.comfyui_path == original_path


class TestSnapshot:
    def test_snapshot_creates_bundle(self, settings: Settings, temp_dir: Path) -> None:
        workflow_file = temp_dir / "workflow.json"
        workflow_file.write_text("{}")

        mock_manager = MagicMock()
        mock_manager.create_snapshot = AsyncMock(
            return_value=("260101-01", CarryForwardReport((), (), (), ()))
        )

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
            scan_models=True,
            carry_from=None,
        )

    def test_snapshot_renders_custom_node_summary(self, settings: Settings, temp_dir: Path) -> None:
        workflow_file = temp_dir / "workflow.json"
        workflow_file.write_text("{}")
        report = CarryForwardReport(
            (),
            (),
            (),
            (),
            custom_nodes=CustomNodeScanReport(
                captured=("captured",),
                skipped=(CustomNodeSkip("registry-node", "no_git_metadata"),),
            ),
        )
        mock_manager = MagicMock()
        mock_manager.create_snapshot = AsyncMock(return_value=("260101-01", report))

        with (
            patch("ai_content_service.cli.get_settings", return_value=settings),
            patch("ai_content_service.snapshot.SnapshotManager", return_value=mock_manager),
        ):
            result = runner.invoke(
                app,
                ["snapshot", "--name", "test_bundle", "--workflow", str(workflow_file)],
            )

        assert result.exit_code == 0
        assert "captured 1, skipped 1" in result.output
        assert "registry-node (no_git_metadata)" in result.output

    def test_from_bundle_resolution_failure_writes_no_snapshot(
        self, settings: Settings, temp_dir: Path
    ) -> None:
        workflow_file = temp_dir / "workflow.json"
        workflow_file.write_text("{}")
        mock_manager = MagicMock()
        mock_manager.create_snapshot = AsyncMock()

        with (
            patch("ai_content_service.cli.get_settings", return_value=settings),
            patch("ai_content_service.cli.resolve_bundle", new_callable=AsyncMock) as resolve,
            patch("ai_content_service.snapshot.SnapshotManager", return_value=mock_manager),
        ):
            resolve.side_effect = BundleResolutionError("Bundle 'missing' not found")
            result = runner.invoke(
                app,
                [
                    "snapshot",
                    "--name",
                    "test_bundle",
                    "--workflow",
                    str(workflow_file),
                    "--from-bundle",
                    "missing",
                ],
            )

        assert result.exit_code == 1
        assert "not found" in result.output
        mock_manager.create_snapshot.assert_not_awaited()
        assert not settings.bundles_path.exists()

    def test_from_bundle_renders_report_categories_in_yellow(
        self, settings: Settings, temp_dir: Path, minimal_bundle_config: BundleConfig
    ) -> None:
        workflow_file = temp_dir / "workflow.json"
        workflow_file.write_text("{}")
        resolved = ResolvedBundle(
            name="seed",
            path=temp_dir / "seed",
            config=minimal_bundle_config,
        )
        report = CarryForwardReport(
            urls_carried=("checkpoints/model.safetensors",),
            files_without_url=("loras/experiment.safetensors",),
            seed_files_unmatched=("vae/old.safetensors",),
            blocks_carried=("hardware",),
        )
        mock_manager = MagicMock()
        mock_manager.create_snapshot = AsyncMock(return_value=("260101-02", report))

        with (
            patch("ai_content_service.cli.get_settings", return_value=settings),
            patch("ai_content_service.cli.resolve_bundle", new=AsyncMock(return_value=resolved)),
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
                    "--from-bundle",
                    "seed",
                ],
            )

        assert result.exit_code == 0
        assert "Carried forward from seed:260101-01" in result.output
        assert "no url" in result.output
        assert "unmatched" in result.output

    def test_snapshot_can_disable_model_scanning(self, settings: Settings, temp_dir: Path) -> None:
        workflow_file = temp_dir / "workflow.json"
        workflow_file.write_text("{}")
        mock_manager = MagicMock()
        mock_manager.create_snapshot = AsyncMock(
            return_value=("260101-01", CarryForwardReport((), (), (), ()))
        )

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
                    "--no-scan-models",
                ],
            )

        assert result.exit_code == 0
        assert mock_manager.create_snapshot.call_args.kwargs["scan_models"] is False

    def test_snapshot_does_not_mutate_singleton(self, settings: Settings, temp_dir: Path) -> None:
        original_path = settings.comfyui_path
        workflow_file = temp_dir / "workflow.json"
        workflow_file.write_text("{}")
        mock_manager = MagicMock()
        mock_manager.create_snapshot = AsyncMock(
            return_value=("260101-01", CarryForwardReport((), (), (), ()))
        )

        with (
            patch("ai_content_service.cli.get_settings", return_value=settings),
            patch("ai_content_service.snapshot.SnapshotManager", return_value=mock_manager),
        ):
            runner.invoke(
                app,
                [
                    "snapshot",
                    "--name",
                    "test_bundle",
                    "--workflow",
                    str(workflow_file),
                    "--comfyui",
                    str(temp_dir / "other_comfyui"),
                ],
            )

        assert settings.comfyui_path == original_path


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
        assert settings.github_token is not None
        assert settings.github_token.get_secret_value() == "ghp_test"
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

    def test_create_registry_manager_unwraps_github_token(self, temp_dir: Path) -> None:
        """The composition root must unwrap SecretStr before it reaches GitBundleRegistry."""
        import base64

        from ai_content_service.registry_service import create_registry_manager

        settings = Settings(
            bundles_repo="https://github.com/test/repo",
            github_token="ghp_rawtoken",  # type: ignore[arg-type]
            cache_path=temp_dir / "cache",
        )
        manager = create_registry_manager(settings)

        git = manager.get("remote")
        assert git is not None
        b64_header = git._auth_header_b64()  # type: ignore[attr-defined]
        assert b64_header is not None
        decoded = base64.b64decode(b64_header).decode()
        assert decoded == "x-access-token:ghp_rawtoken"

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

    async def test_run_deploy_syncs_resolves_and_deploys(self, temp_dir: Path) -> None:
        from ai_content_service.registry_service import run_deploy

        settings = Settings(
            bundles_path=temp_dir / "bundles",
            auto_sync_registries=False,
        )
        ref = BundleReference(name="test_bundle")
        bundle_path = temp_dir / "bundle_dir"
        bundle_path.mkdir()

        mock_manager = MagicMock()
        mock_manager.sync_all = AsyncMock()
        mock_manager.resolve = AsyncMock(return_value=bundle_path)

        mock_deployer_instance = MagicMock()
        mock_deployer_instance.deploy_from_path = AsyncMock(return_value=MagicMock(success=True))
        mock_deployer_cls = MagicMock(return_value=mock_deployer_instance)

        with (
            patch(
                "ai_content_service.registry_service.create_registry_manager",
                return_value=mock_manager,
            ),
            patch("ai_content_service.deployer.Deployer", mock_deployer_cls),
            patch("ai_content_service.bundle.BundleManager"),
            patch("ai_content_service.comfyui.ComfyUIManager"),
            patch("ai_content_service.downloader.ModelDownloader"),
            patch("ai_content_service.workflows.WorkflowManager"),
        ):
            result = await run_deploy(
                settings=settings,
                ref=ref,
                mode=DeployMode.FULL,
                verify=True,
                dry_run=False,
                sync=True,
            )

        mock_manager.sync_all.assert_awaited_once()
        mock_manager.resolve.assert_awaited_once_with(ref)
        mock_deployer_instance.deploy_from_path.assert_awaited_once_with(
            bundle_path=bundle_path,
            mode=DeployMode.FULL,
            verify=True,
            dry_run=False,
        )
        assert result.success is True

    async def test_run_deploy_skips_sync_when_disabled(self, temp_dir: Path) -> None:
        from ai_content_service.registry_service import run_deploy

        settings = Settings(
            bundles_path=temp_dir / "bundles",
            auto_sync_registries=False,
        )
        ref = BundleReference(name="test_bundle")
        bundle_path = temp_dir / "bundle_dir"
        bundle_path.mkdir()

        mock_manager = MagicMock()
        mock_manager.sync_all = AsyncMock()
        mock_manager.resolve = AsyncMock(return_value=bundle_path)

        mock_deployer_instance = MagicMock()
        mock_deployer_instance.deploy_from_path = AsyncMock(return_value=MagicMock(success=True))

        with (
            patch(
                "ai_content_service.registry_service.create_registry_manager",
                return_value=mock_manager,
            ),
            patch("ai_content_service.deployer.Deployer", return_value=mock_deployer_instance),
            patch("ai_content_service.bundle.BundleManager"),
            patch("ai_content_service.comfyui.ComfyUIManager"),
            patch("ai_content_service.downloader.ModelDownloader"),
            patch("ai_content_service.workflows.WorkflowManager"),
        ):
            await run_deploy(
                settings=settings,
                ref=ref,
                mode=DeployMode.FULL,
                verify=True,
                dry_run=False,
                sync=None,
            )

        mock_manager.sync_all.assert_not_awaited()

    async def test_run_deploy_no_registries_raises(self) -> None:
        from ai_content_service.registry_service import run_deploy

        settings = Settings()
        ref = BundleReference(name="test_bundle")

        mock_manager = MagicMock()
        mock_manager.list_registries.return_value = []
        mock_manager.sync_all = AsyncMock()
        mock_manager.resolve = AsyncMock()

        with (
            patch(
                "ai_content_service.registry_service.create_registry_manager",
                return_value=mock_manager,
            ),
            pytest.raises(ValueError, match="No bundle registries configured"),
        ):
            await run_deploy(
                settings=settings,
                ref=ref,
                mode=DeployMode.FULL,
                verify=True,
                dry_run=False,
            )

        mock_manager.resolve.assert_not_awaited()
        mock_manager.sync_all.assert_not_awaited()

    def test_get_or_default_registry(self) -> None:
        from ai_content_service.registry_service import get_or_default_registry

        named_reg = MagicMock()
        named_reg.name = "remote"
        default_reg = MagicMock()
        default_reg.name = "local"

        manager = MagicMock()
        manager.get.side_effect = lambda name: named_reg if name == "remote" else None
        manager.default = default_reg

        assert (
            get_or_default_registry(manager, BundleReference(name="x", registry="remote"))
            is named_reg
        )
        assert get_or_default_registry(manager, BundleReference(name="x")) is default_reg

        empty_manager = MagicMock()
        empty_manager.get.return_value = None
        empty_manager.default = None
        with pytest.raises(ValueError, match="No registry available"):
            get_or_default_registry(empty_manager, BundleReference(name="x"))

    async def test_run_deploy_honors_explicit_sync_false(self, temp_dir: Path) -> None:
        from ai_content_service.registry_service import run_deploy

        settings = Settings(
            bundles_path=temp_dir / "bundles",
            auto_sync_registries=True,
        )
        ref = BundleReference(name="test_bundle")
        bundle_path = temp_dir / "bundle_dir"
        bundle_path.mkdir()

        mock_manager = MagicMock()
        mock_manager.sync_all = AsyncMock()
        mock_manager.resolve = AsyncMock(return_value=bundle_path)

        mock_deployer_instance = MagicMock()
        mock_deployer_instance.deploy_from_path = AsyncMock(return_value=MagicMock(success=True))

        with (
            patch(
                "ai_content_service.registry_service.create_registry_manager",
                return_value=mock_manager,
            ),
            patch("ai_content_service.deployer.Deployer", return_value=mock_deployer_instance),
            patch("ai_content_service.bundle.BundleManager"),
            patch("ai_content_service.comfyui.ComfyUIManager"),
            patch("ai_content_service.downloader.ModelDownloader"),
            patch("ai_content_service.workflows.WorkflowManager"),
        ):
            await run_deploy(
                settings=settings,
                ref=ref,
                mode=DeployMode.FULL,
                verify=True,
                dry_run=False,
                sync=False,
            )

        mock_manager.sync_all.assert_not_awaited()


class TestModelsCheck:
    """Tests for C4 -- `acs models check` gains --all/--offline/--json."""

    def _make_manager_with_bundle(self, bundle_dir: Path) -> MagicMock:
        mock_reg = AsyncMock()
        mock_reg.name = "local"
        mock_reg.resolve_bundle_path = AsyncMock(return_value=bundle_dir)

        mock_manager = MagicMock()
        mock_manager.default = mock_reg
        mock_manager.get.return_value = None  # use default
        # The single-bundle path resolves via `manager.resolve(ref)`, not `reg.resolve_bundle_path`.
        mock_manager.resolve = AsyncMock(return_value=bundle_dir)

        return mock_manager

    def test_neither_bundle_nor_all_is_an_error(self, settings: Settings) -> None:
        with patch("ai_content_service.cli.get_settings", return_value=settings):
            result = runner.invoke(app, ["models", "check"])

        assert result.exit_code == 1
        assert "specify bundle or --all" in result.output.lower()

    def test_both_bundle_and_all_is_an_error(self, settings: Settings) -> None:
        with patch("ai_content_service.cli.get_settings", return_value=settings):
            result = runner.invoke(app, ["models", "check", "test_bundle", "--all"])

        assert result.exit_code != 0
        assert "both given" in result.output.lower()

    def test_single_bundle_check_success(self, settings: Settings, temp_dir: Path) -> None:
        bundle_dir = temp_dir / "test_bundle_check"
        bundle_dir.mkdir()
        (bundle_dir / "bundle.yaml").write_text(
            "metadata:\n  name: test_bundle\n  version: '260101-01'\n"
            "models:\n"
            "  - name: m\n"
            "    model_type: checkpoints\n"
            "    files:\n"
            "      - name: f\n"
            "        url: https://example.com/f\n"
            "        filename: f.safetensors\n"
        )
        mock_manager = self._make_manager_with_bundle(bundle_dir)

        with (
            patch("ai_content_service.cli.get_settings", return_value=settings),
            patch("ai_content_service.cli.create_registry_manager", return_value=mock_manager),
            patch(
                "ai_content_service.preflight.check_bundle",
                new=AsyncMock(return_value=MagicMock(ok=True)),
            ),
        ):
            result = runner.invoke(app, ["models", "check", "test_bundle"])

        assert result.exit_code == 0

    def test_single_bundle_check_offline_flag_forwarded(
        self, settings: Settings, temp_dir: Path
    ) -> None:
        bundle_dir = temp_dir / "test_bundle_check"
        bundle_dir.mkdir()
        (bundle_dir / "bundle.yaml").write_text(
            "metadata:\n  name: test_bundle\n  version: '260101-01'\n"
        )
        mock_manager = self._make_manager_with_bundle(bundle_dir)
        mock_check = AsyncMock(return_value=MagicMock(ok=True))

        with (
            patch("ai_content_service.cli.get_settings", return_value=settings),
            patch("ai_content_service.cli.create_registry_manager", return_value=mock_manager),
            patch("ai_content_service.preflight.check_bundle", new=mock_check),
        ):
            result = runner.invoke(app, ["models", "check", "test_bundle", "--offline"])

        assert result.exit_code == 0
        assert mock_check.call_args.kwargs["offline"] is True

    def test_single_bundle_check_failure_exits_1(self, settings: Settings, temp_dir: Path) -> None:
        bundle_dir = temp_dir / "test_bundle_check"
        bundle_dir.mkdir()
        (bundle_dir / "bundle.yaml").write_text(
            "metadata:\n  name: test_bundle\n  version: '260101-01'\n"
        )
        mock_manager = self._make_manager_with_bundle(bundle_dir)

        with (
            patch("ai_content_service.cli.get_settings", return_value=settings),
            patch("ai_content_service.cli.create_registry_manager", return_value=mock_manager),
            patch(
                "ai_content_service.preflight.check_bundle",
                new=AsyncMock(return_value=MagicMock(ok=False)),
            ),
        ):
            result = runner.invoke(app, ["models", "check", "test_bundle"])

        assert result.exit_code == 1

    def test_all_checks_every_bundle_in_default_registry(
        self, settings: Settings, temp_dir: Path
    ) -> None:
        good_path = temp_dir / "good"
        good_path.mkdir()
        bad_path = temp_dir / "bad"
        bad_path.mkdir()

        good_entry = MagicMock()
        good_entry.name = "alpha-bundle"
        bad_entry = MagicMock()
        bad_entry.name = "beta-bundle"

        mock_reg = AsyncMock()
        mock_reg.get_index = AsyncMock(return_value=MagicMock(bundles=[good_entry, bad_entry]))
        mock_reg.resolve_bundle_path = AsyncMock(side_effect=[good_path, bad_path])

        mock_manager = MagicMock()
        mock_manager.default = mock_reg
        mock_manager.get.return_value = None

        async def fake_check_bundle_path(
            name: str,
            _path: Path,
            _settings: Settings,
            *,
            offline: bool = False,
            semaphore: asyncio.Semaphore | None = None,
        ) -> BundleCheckResult:
            assert not offline
            assert semaphore is not None
            return BundleCheckResult(
                bundle_name=name,
                parse_error=None if name == "alpha-bundle" else "synthetic failure",
            )

        with (
            patch("ai_content_service.cli.get_settings", return_value=settings),
            patch("ai_content_service.cli.create_registry_manager", return_value=mock_manager),
            patch(
                "ai_content_service.preflight.check_all_bundles",
                wraps=preflight_module.check_all_bundles,
            ) as mock_check_all,
            patch(
                "ai_content_service.preflight.check_bundle_path", side_effect=fake_check_bundle_path
            ),
        ):
            result = runner.invoke(app, ["models", "check", "--all"])

        assert result.exit_code == 1
        assert "alpha-bundle" in result.output
        assert "beta-bundle" in result.output
        assert "failed to parse" in result.output
        assert mock_check_all.await_args is not None
        assert mock_check_all.await_args.args[0] == ["alpha-bundle", "beta-bundle"]

    def test_empty_registry_fails_with_actionable_message(self, settings: Settings) -> None:
        mock_reg = AsyncMock()
        mock_reg.get_index = AsyncMock(return_value=MagicMock(bundles=[]))
        mock_manager = MagicMock(default=mock_reg)
        mock_manager.get.return_value = None

        with (
            patch("ai_content_service.cli.get_settings", return_value=settings),
            patch("ai_content_service.cli.create_registry_manager", return_value=mock_manager),
        ):
            result = runner.invoke(app, ["models", "check", "--all"])

        assert result.exit_code == 1
        assert "no bundles found" in result.output.lower()
        assert "--sync" in result.output
        assert "--allow-empty" in result.output

    def test_empty_registry_can_be_explicitly_allowed(self, settings: Settings) -> None:
        mock_reg = AsyncMock()
        mock_reg.get_index = AsyncMock(return_value=MagicMock(bundles=[]))
        mock_manager = MagicMock(default=mock_reg)
        mock_manager.get.return_value = None

        with (
            patch("ai_content_service.cli.get_settings", return_value=settings),
            patch("ai_content_service.cli.create_registry_manager", return_value=mock_manager),
        ):
            result = runner.invoke(app, ["models", "check", "--all", "--allow-empty"])

        assert result.exit_code == 0

    def test_all_with_no_registry_available_errors(self, settings: Settings) -> None:
        mock_manager = MagicMock()
        mock_manager.default = None
        mock_manager.get.return_value = None

        with (
            patch("ai_content_service.cli.get_settings", return_value=settings),
            patch("ai_content_service.cli.create_registry_manager", return_value=mock_manager),
        ):
            result = runner.invoke(app, ["models", "check", "--all"])

        assert result.exit_code == 1
        assert "no default registry" in result.output.lower()


class TestTimingsShow:
    """Tests for `acs timings show` -- Part B's read path for the timing JSONL."""

    def _write_records(self, path: Path, *lines: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n")

    _READY_RECORD = (
        '{"schema": 1, "ts": "2026-08-07T14:03:11Z", "outcome": "ready", "total_s": 214.7, '
        '"bundle": "qwen_rapid_aio", "bundle_version": "260805-01", "mode": "full", '
        '"phases": [{"phase": "comfyui", "duration_s": 41.2, "skipped": false}, '
        '{"phase": "requirements_base", "duration_s": 0.0, "skipped": true}], '
        '"models": {"sources": {"hf_xet": 3}, "bytes_total": 100, "mbps": 347.1}, '
        '"env": {"base_image": null, "gpu": null, "cpu_count": 8, "aisha_version": "0.13.0", '
        '"comfyui_source": "image", "hf_xet_enabled": true, "instance": null}}'
    )
    _FAILED_RECORD = (
        '{"schema": 1, "ts": "2026-08-07T15:00:00Z", "outcome": "failed", '
        '"error": "3/5 model files failed", "total_s": 12.3, "bundle": "other_bundle", '
        '"bundle_version": "260101-01", "mode": "models_only", '
        '"phases": [{"phase": "comfyui", "duration_s": 0.0, "skipped": true}, '
        '{"phase": "models", "duration_s": 12.3, "skipped": false}]}'
    )

    def test_missing_file_prints_message_without_raising(
        self, settings: Settings, temp_dir: Path
    ) -> None:
        with patch("ai_content_service.cli.get_settings", return_value=settings):
            result = runner.invoke(
                app, ["timings", "show", "--path", str(temp_dir / "absent.jsonl")]
            )

        assert result.exit_code == 0
        assert "no provisioning timing records" in result.output.lower()

    def test_single_run_renders_a_table(self, settings: Settings, temp_dir: Path) -> None:
        path = temp_dir / "timings.jsonl"
        self._write_records(path, self._READY_RECORD)

        with patch("ai_content_service.cli.get_settings", return_value=settings):
            result = runner.invoke(app, ["timings", "show", "--path", str(path)])

        assert result.exit_code == 0
        assert "qwen_rapid_aio" in result.output

    def test_mixed_set_renders_without_raising(self, settings: Settings, temp_dir: Path) -> None:
        path = temp_dir / "timings.jsonl"
        self._write_records(path, self._READY_RECORD, self._FAILED_RECORD)

        with patch("ai_content_service.cli.get_settings", return_value=settings):
            result = runner.invoke(app, ["timings", "show", "--path", str(path)])

        assert result.exit_code == 0
        assert "qwen_rapid_aio" in result.output
        assert "other_bundle" in result.output

    def test_json_output_emits_raw_records(self, settings: Settings, temp_dir: Path) -> None:
        path = temp_dir / "timings.jsonl"
        self._write_records(path, self._READY_RECORD)

        with patch("ai_content_service.cli.get_settings", return_value=settings):
            result = runner.invoke(app, ["timings", "show", "--path", str(path), "--json"])

        assert result.exit_code == 0
        lines = result.output.splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["bundle"] == "qwen_rapid_aio"
        assert not result.output.lstrip().startswith("[")

    def test_json_output_for_empty_selection_has_no_lines(
        self, settings: Settings, temp_dir: Path
    ) -> None:
        path = temp_dir / "timings.jsonl"
        self._write_records(path, self._READY_RECORD)

        with patch("ai_content_service.cli.get_settings", return_value=settings):
            result = runner.invoke(
                app,
                ["timings", "show", "--path", str(path), "--bundle", "absent", "--json"],
            )

        assert result.exit_code == 0
        assert result.output == ""

    def test_bundle_filter(self, settings: Settings, temp_dir: Path) -> None:
        path = temp_dir / "timings.jsonl"
        self._write_records(path, self._READY_RECORD, self._FAILED_RECORD)

        with patch("ai_content_service.cli.get_settings", return_value=settings):
            result = runner.invoke(
                app, ["timings", "show", "--path", str(path), "--bundle", "other_bundle"]
            )

        assert result.exit_code == 0
        assert "other_bundle" in result.output
        assert "qwen_rapid_aio" not in result.output

    def test_last_filter_limits_to_most_recent_n(self, settings: Settings, temp_dir: Path) -> None:
        path = temp_dir / "timings.jsonl"
        self._write_records(path, self._READY_RECORD, self._FAILED_RECORD)

        with patch("ai_content_service.cli.get_settings", return_value=settings):
            result = runner.invoke(app, ["timings", "show", "--path", str(path), "--last", "1"])

        assert result.exit_code == 0
        assert "other_bundle" in result.output
        assert "qwen_rapid_aio" not in result.output

    @pytest.mark.parametrize("value", ["0", "-1"])
    def test_last_requires_a_positive_integer(
        self, settings: Settings, temp_dir: Path, value: str
    ) -> None:
        path = temp_dir / "timings.jsonl"
        self._write_records(path, self._READY_RECORD)

        with patch("ai_content_service.cli.get_settings", return_value=settings):
            result = runner.invoke(app, ["timings", "show", "--path", str(path), "--last", value])

        assert result.exit_code != 0
        assert "at least 1" in result.output

    def test_schema_2_failed_phase_is_rendered_distinctly(
        self, settings: Settings, temp_dir: Path
    ) -> None:
        path = temp_dir / "timings.jsonl"
        self._write_records(
            path,
            json.dumps(
                {
                    "schema": 2,
                    "started_at": "2026-08-08T00:00:00Z",
                    "outcome": "failed",
                    "bundle": "demo",
                    "phases": [
                        {
                            "phase": "models",
                            "started_at": "2026-08-08T00:00:00Z",
                            "duration_s": 1.0,
                            "status": "failed",
                        }
                    ],
                    "metrics": {"models": {"effective_mib_per_s": 1.5}},
                }
            ),
        )

        with patch("ai_content_service.cli.get_settings", return_value=settings):
            result = runner.invoke(app, ["timings", "show", "--path", str(path)])

        assert result.exit_code == 0
        assert "failed (1.0)" in result.output
        assert "Effective MiB/s" in result.output

    def test_defaults_to_settings_cache_path_when_no_path_given(self, settings: Settings) -> None:
        default_path = settings.cache_path / "provisioning-timings.jsonl"
        self._write_records(default_path, self._READY_RECORD)

        with patch("ai_content_service.cli.get_settings", return_value=settings):
            result = runner.invoke(app, ["timings", "show"])

        assert result.exit_code == 0
        assert "qwen_rapid_aio" in result.output
