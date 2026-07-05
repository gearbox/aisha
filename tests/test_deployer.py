"""Tests for deployment orchestration."""

import tempfile
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_content_service.config import (
    BundleConfig,
    BundleMetadata,
    CustomNodeConfig,
    DeployMode,
    ModelConfig,
    ModelFileConfig,
    Settings,
)
from ai_content_service.deployer import Deployer, DeploymentResult


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
def minimal_bundle() -> BundleConfig:
    return BundleConfig(
        metadata=BundleMetadata(
            name="test_bundle",
            version="260101-01",
            description="Test",
            created_at=datetime.now(timezone.utc),
        ),
        models=[],
        workflow_file="workflow.json",
    )


@pytest.fixture
def full_bundle() -> BundleConfig:
    return BundleConfig(
        metadata=BundleMetadata(
            name="full_bundle",
            version="260101-01",
            description="Full test",
            created_at=datetime.now(timezone.utc),
        ),
        custom_nodes=[
            CustomNodeConfig(
                name="TestNode",
                git_url="https://github.com/test/node",
                commit_sha="abc123",
            )
        ],
        models=[
            ModelConfig(
                name="Test Model",
                model_type="checkpoints",
                files=[
                    ModelFileConfig(
                        name="Checkpoint",
                        url="https://huggingface.co/test/model.safetensors",
                        filename="model.safetensors",
                    )
                ],
            )
        ],
        workflow_file="workflow.json",
    )


@pytest.fixture
def mock_bundle_manager(minimal_bundle: BundleConfig, temp_dir: Path) -> MagicMock:
    mgr = MagicMock()
    bundle_path = temp_dir / "bundles" / "test_bundle" / "260101-01"
    bundle_path.mkdir(parents=True, exist_ok=True)
    mgr.resolve_bundle_path.return_value = bundle_path
    mgr.load_bundle_config_from_path.return_value = minimal_bundle
    return mgr


@pytest.fixture
def mock_comfyui_manager() -> AsyncMock:
    mgr = AsyncMock()
    mgr.checkout = AsyncMock()
    mgr.install_base_requirements = AsyncMock()
    mgr.install_locked_requirements = AsyncMock()
    mgr.install_custom_node = AsyncMock()
    mgr.verify = AsyncMock(return_value=True)
    return mgr


@pytest.fixture
def mock_model_downloader() -> AsyncMock:
    dl = AsyncMock()
    dl.download_all = AsyncMock(return_value=0)
    return dl


@pytest.fixture
def mock_workflow_manager() -> AsyncMock:
    wm = AsyncMock()
    wm.install = AsyncMock()
    return wm


@pytest.fixture
def deployer(
    settings: Settings,
    mock_bundle_manager: MagicMock,
    mock_comfyui_manager: AsyncMock,
    mock_model_downloader: AsyncMock,
    mock_workflow_manager: AsyncMock,
) -> Deployer:
    return Deployer(
        settings=settings,
        bundle_manager=mock_bundle_manager,
        comfyui_manager=mock_comfyui_manager,
        model_downloader=mock_model_downloader,
        workflow_manager=mock_workflow_manager,
    )


class TestDeployDryRun:
    async def test_dry_run_returns_success_without_executing(
        self, deployer: Deployer, mock_comfyui_manager: AsyncMock
    ) -> None:
        result = await deployer.deploy("test_bundle", dry_run=True)

        assert result.success is True
        mock_comfyui_manager.checkout.assert_not_called()
        mock_comfyui_manager.install_base_requirements.assert_not_called()

    async def test_dry_run_returns_deployment_result(self, deployer: Deployer) -> None:
        result = await deployer.deploy("test_bundle", dry_run=True)

        assert isinstance(result, DeploymentResult)
        assert result.errors == []

    async def test_dry_run_with_specific_version(self, deployer: Deployer) -> None:
        result = await deployer.deploy("test_bundle", version="260101-01", dry_run=True)
        assert result.success is True


class TestDeployExecution:
    async def test_captures_exception_in_result(
        self, deployer: Deployer, mock_comfyui_manager: AsyncMock
    ) -> None:
        mock_comfyui_manager.install_base_requirements.side_effect = RuntimeError("pip exploded")

        # Patch _execute_deployment to raise directly so we don't need to
        # set up the entire bundle with comfyui config
        with patch.object(
            deployer,
            "_execute_deployment",
            new=AsyncMock(side_effect=RuntimeError("deployment error")),
        ):
            result = await deployer.deploy("test_bundle")

        assert result.success is False
        assert any("deployment error" in e for e in result.errors)

    async def test_models_only_mode_propagated_to_plan(self, deployer: Deployer) -> None:
        result = await deployer.deploy("test_bundle", mode=DeployMode.MODELS_ONLY, dry_run=True)

        assert result.success is True
        assert result.plan.mode == DeployMode.MODELS_ONLY

    async def test_full_mode_propagated_to_plan(self, deployer: Deployer) -> None:
        result = await deployer.deploy("test_bundle", mode=DeployMode.FULL, dry_run=True)

        assert result.plan.mode == DeployMode.FULL

    async def test_deploy_delegates_to_deploy_from_path(
        self, deployer: Deployer, mock_bundle_manager: MagicMock
    ) -> None:
        """deploy() must resolve the bundle path then delegate to deploy_from_path."""
        expected_result = MagicMock()
        with patch.object(
            deployer, "deploy_from_path", new=AsyncMock(return_value=expected_result)
        ) as mock_deploy_from_path:
            result = await deployer.deploy(
                "test_bundle",
                version="260101-01",
                mode=DeployMode.MODELS_ONLY,
                verify=False,
                dry_run=True,
            )

        mock_bundle_manager.resolve_bundle_path.assert_called_once_with("test_bundle", "260101-01")
        resolved_path = mock_bundle_manager.resolve_bundle_path.return_value
        mock_deploy_from_path.assert_called_once_with(
            resolved_path, mode=DeployMode.MODELS_ONLY, verify=False, dry_run=True
        )
        assert result is expected_result


class TestDeployFromPath:
    async def test_dry_run_returns_success(
        self,
        deployer: Deployer,
        temp_dir: Path,
    ) -> None:
        bundle_path = temp_dir / "bundles" / "test_bundle" / "260101-01"
        result = await deployer.deploy_from_path(bundle_path, dry_run=True)

        assert result.success is True
        assert result.errors == []

    async def test_captures_exception_in_result(
        self,
        deployer: Deployer,
        temp_dir: Path,
    ) -> None:
        bundle_path = temp_dir / "bundles" / "test_bundle" / "260101-01"

        with patch.object(
            deployer,
            "_execute_deployment",
            new=AsyncMock(side_effect=RuntimeError("path deploy failed")),
        ):
            result = await deployer.deploy_from_path(bundle_path)

        assert result.success is False
        assert any("path deploy failed" in e for e in result.errors)

    async def test_models_only_mode_skips_comfyui_steps(
        self,
        deployer: Deployer,
        mock_comfyui_manager: AsyncMock,
        temp_dir: Path,
    ) -> None:
        bundle_path = temp_dir / "bundles" / "test_bundle" / "260101-01"

        result = await deployer.deploy_from_path(
            bundle_path, mode=DeployMode.MODELS_ONLY, dry_run=True
        )

        assert result.plan.will_update_comfyui is False
        assert result.plan.will_install_base_requirements is False
        mock_comfyui_manager.checkout.assert_not_called()


class TestRunDeployUsesFromSettings:
    async def test_run_deploy_builds_reporter_from_settings(self, tmp_path: Path) -> None:
        from unittest.mock import AsyncMock, MagicMock, patch

        from ai_content_service import registry_service
        from ai_content_service.provisioning_reporter import ProvisioningReporter

        settings = Settings()
        ref = MagicMock()

        mock_manager = MagicMock()
        mock_manager.list_registries.return_value = ["fake"]
        mock_manager.sync_all = AsyncMock()
        mock_manager.resolve = AsyncMock(return_value=tmp_path)

        mock_deployer = MagicMock()
        mock_deployer.deploy_from_path = AsyncMock(return_value=MagicMock(success=True, errors=[]))
        mock_reporter = ProvisioningReporter.disabled()

        with (
            patch(
                "ai_content_service.registry_service.create_registry_manager",
                return_value=mock_manager,
            ),
            patch("ai_content_service.deployer.Deployer", return_value=mock_deployer),
            patch("ai_content_service.bundle.BundleManager"),
            patch("ai_content_service.comfyui.ComfyUIManager"),
            patch("ai_content_service.downloader.ModelDownloader"),
            patch("ai_content_service.workflows.WorkflowManager"),
            patch.object(
                ProvisioningReporter, "from_settings", return_value=mock_reporter
            ) as mock_from_settings,
            patch.object(ProvisioningReporter, "from_env") as mock_from_env,
        ):
            await registry_service.run_deploy(
                settings=settings,
                ref=ref,
                mode=DeployMode.FULL,
                verify=True,
                dry_run=False,
                sync=False,
            )

        mock_from_settings.assert_called_once_with(settings)
        mock_from_env.assert_not_called()


class TestVerificationBehavior:
    async def test_verify_success_sets_passed_and_no_warning(
        self, deployer: Deployer, mock_comfyui_manager: AsyncMock
    ) -> None:
        mock_comfyui_manager.verify = AsyncMock(return_value=True)
        result = await deployer.deploy("test_bundle", verify=True)
        assert result.verification_passed is True
        assert "Verification failed" not in result.warnings

    async def test_verify_failure_adds_warning(
        self, deployer: Deployer, mock_comfyui_manager: AsyncMock
    ) -> None:
        mock_comfyui_manager.verify = AsyncMock(return_value=False)
        result = await deployer.deploy("test_bundle", verify=True)
        assert result.verification_passed is False
        assert "Verification failed" in result.warnings

    async def test_no_verify_skips_verify_call(
        self, deployer: Deployer, mock_comfyui_manager: AsyncMock
    ) -> None:
        result = await deployer.deploy("test_bundle", verify=False)
        mock_comfyui_manager.verify.assert_not_called()
        assert result.verification_passed is None


class TestDeploymentResult:
    def test_initial_state(self) -> None:
        from ai_content_service.config import DeploymentPlan

        plan = MagicMock(spec=DeploymentPlan)
        result = DeploymentResult(success=True, plan=plan)

        assert result.comfyui_updated is False
        assert result.base_requirements_installed is False
        assert result.locked_requirements_installed is False
        assert result.custom_nodes_installed == 0
        assert result.models_downloaded == 0
        assert result.workflow_installed is False
        assert result.verification_passed is None
        assert result.errors == []
        assert result.warnings == []
