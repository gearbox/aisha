"""Tests for deployment mode and planning."""

from datetime import datetime, timezone

import pytest

from ai_content_service.config import (
    BundleConfig,
    BundleMetadata,
    ComfyUIConfig,
    CustomNodeConfig,
    DeploymentPlan,
    DeployMode,
    ModelConfig,
    ModelFileConfig,
)


@pytest.fixture
def full_bundle() -> BundleConfig:
    """Create a complete bundle config for testing."""
    return BundleConfig(
        metadata=BundleMetadata(
            name="test_bundle",
            version="260101-01",
            description="Test bundle",
            created_at=datetime.now(timezone.utc),
            tested=True,
        ),
        comfyui=ComfyUIConfig(
            repo="https://github.com/comfyanonymous/ComfyUI",
            commit="abc123def456",
        ),
        custom_nodes=[
            CustomNodeConfig(
                name="ComfyUI-GGUF",
                git_url="https://github.com/city96/ComfyUI-GGUF",
                commit_sha="def456789",
            ),
            CustomNodeConfig(
                name="ComfyUI-VideoHelper",
                git_url="https://github.com/example/VideoHelper",
                commit_sha="789abcdef",
            ),
        ],
        models=[
            ModelConfig(
                name="WAN Model",
                model_type="diffusion_models",
                files=[
                    ModelFileConfig(
                        name="Model Q8",
                        url="https://huggingface.co/test/model.gguf",
                        filename="model.gguf",
                        sha256="abc123",
                    ),
                    ModelFileConfig(
                        name="Model Q4",
                        url="https://huggingface.co/test/model_q4.gguf",
                        filename="model_q4.gguf",
                        sha256="def456",
                    ),
                ],
            ),
        ],
        requirements_lock_file="requirements.lock",
        workflow_file="workflow.json",
    )


@pytest.fixture
def minimal_bundle() -> BundleConfig:
    """Create a minimal bundle (models + workflow only)."""
    return BundleConfig(
        metadata=BundleMetadata(
            name="minimal_bundle",
            version="260101-01",
            description="Minimal test bundle",
            created_at=datetime.now(timezone.utc),
            tested=False,
        ),
        comfyui=None,  # No ComfyUI config
        custom_nodes=[],  # No custom nodes
        models=[
            ModelConfig(
                name="Single Model",
                model_type="checkpoints",
                files=[
                    ModelFileConfig(
                        name="Checkpoint",
                        url="https://example.com/model.safetensors",
                        filename="model.safetensors",
                    ),
                ],
            ),
        ],
        requirements_lock_file=None,  # No locked requirements
        workflow_file="workflow.json",
    )


class TestDeployMode:
    """Tests for DeployMode enum."""

    def test_full_mode_value(self) -> None:
        """Test FULL mode has correct string value."""
        assert DeployMode.FULL.value == "full"

    def test_models_only_mode_value(self) -> None:
        """Test MODELS_ONLY mode has correct string value."""
        assert DeployMode.MODELS_ONLY.value == "models_only"

    def test_mode_from_string(self) -> None:
        """Test creating mode from string."""
        assert DeployMode("full") == DeployMode.FULL
        assert DeployMode("models_only") == DeployMode.MODELS_ONLY


class TestDeploymentPlanFullMode:
    """Tests for DeploymentPlan in FULL mode."""

    def test_full_mode_with_complete_bundle(self, full_bundle: BundleConfig) -> None:
        """Test FULL mode enables all deployment steps."""
        plan = DeploymentPlan.from_bundle(full_bundle, DeployMode.FULL)

        assert plan.mode == DeployMode.FULL
        assert plan.bundle_name == "test_bundle"
        assert plan.bundle_version == "260101-01"

        # All steps should be enabled
        assert plan.will_update_comfyui is True
        assert plan.will_install_base_requirements is True
        assert plan.will_install_locked_requirements is True
        assert plan.will_install_custom_nodes is True
        assert plan.will_download_models is True
        assert plan.will_install_workflow is True
        assert plan.will_verify is True

        # Counts
        assert plan.custom_nodes_count == 2
        assert plan.models_count == 1
        assert plan.model_files_count == 2

    def test_full_mode_with_minimal_bundle(self, minimal_bundle: BundleConfig) -> None:
        """Test FULL mode with minimal bundle only enables relevant steps."""
        plan = DeploymentPlan.from_bundle(minimal_bundle, DeployMode.FULL)

        assert plan.mode == DeployMode.FULL

        # Steps depend on bundle content
        assert plan.will_update_comfyui is False  # No comfyui config
        assert plan.will_install_base_requirements is False  # No comfyui config
        assert plan.will_install_locked_requirements is False  # No requirements file
        assert plan.will_install_custom_nodes is False  # No custom nodes
        assert plan.will_download_models is True  # Has models
        assert plan.will_install_workflow is True  # Has workflow

        assert plan.custom_nodes_count == 0
        assert plan.models_count == 1
        assert plan.model_files_count == 1


class TestDeploymentPlanModelsOnlyMode:
    """Tests for DeploymentPlan in MODELS_ONLY mode."""

    def test_models_only_skips_comfyui_setup(self, full_bundle: BundleConfig) -> None:
        """Test MODELS_ONLY mode skips ComfyUI and custom node setup."""
        plan = DeploymentPlan.from_bundle(full_bundle, DeployMode.MODELS_ONLY)

        assert plan.mode == DeployMode.MODELS_ONLY
        assert plan.bundle_name == "test_bundle"

        # These should be skipped
        assert plan.will_update_comfyui is False
        assert plan.will_install_base_requirements is False
        assert plan.will_install_locked_requirements is False
        assert plan.will_install_custom_nodes is False

        # These should still happen
        assert plan.will_download_models is True
        assert plan.will_install_workflow is True
        assert plan.will_verify is True

        # Custom nodes count should be 0 in plan (won't be installed)
        assert plan.custom_nodes_count == 0
        # But models should still be counted
        assert plan.models_count == 1
        assert plan.model_files_count == 2

    def test_models_only_without_verify(self, full_bundle: BundleConfig) -> None:
        """Test MODELS_ONLY mode can disable verification."""
        plan = DeploymentPlan.from_bundle(full_bundle, DeployMode.MODELS_ONLY, verify=False)

        assert plan.will_verify is False
        assert plan.will_download_models is True
        assert plan.will_install_workflow is True

    def test_models_only_with_empty_models(self) -> None:
        """Test MODELS_ONLY mode with bundle having no models."""
        bundle = BundleConfig(
            metadata=BundleMetadata(
                name="workflow_only",
                version="260101-01",
                description="Just a workflow",
                created_at=datetime.now(timezone.utc),
            ),
            models=[],
            workflow_file="workflow.json",
        )

        plan = DeploymentPlan.from_bundle(bundle, DeployMode.MODELS_ONLY)

        assert plan.will_download_models is False
        assert plan.will_install_workflow is True
        assert plan.models_count == 0
        assert plan.model_files_count == 0


class TestBundleConfigHelpers:
    """Tests for BundleConfig helper methods."""

    def test_requires_comfyui_setup(
        self, full_bundle: BundleConfig, minimal_bundle: BundleConfig
    ) -> None:
        """Test requires_comfyui_setup detection."""
        assert full_bundle.requires_comfyui_setup() is True
        assert minimal_bundle.requires_comfyui_setup() is False

    def test_requires_custom_nodes(
        self, full_bundle: BundleConfig, minimal_bundle: BundleConfig
    ) -> None:
        """Test requires_custom_nodes detection."""
        assert full_bundle.requires_custom_nodes() is True
        assert minimal_bundle.requires_custom_nodes() is False

    def test_requires_models(self, full_bundle: BundleConfig, minimal_bundle: BundleConfig) -> None:
        """Test requires_models detection."""
        assert full_bundle.requires_models() is True
        assert minimal_bundle.requires_models() is True

    def test_get_all_model_files(self, full_bundle: BundleConfig) -> None:
        """Test flattening model files."""
        files = full_bundle.get_all_model_files()
        assert len(files) == 2

        model, file = files[0]
        assert model.name == "WAN Model"
        assert file.filename == "model.gguf"
