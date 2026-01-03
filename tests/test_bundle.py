"""Tests for bundle management."""

import json
import tempfile
from pathlib import Path

import pytest
import yaml

from ai_content_service.bundle import (
    BundleError,
    BundleInfo,
    BundleManager,
    BundleNotFoundError,
    BundleValidationError,
)
from ai_content_service.config import Settings


@pytest.fixture
def temp_bundles_dir() -> Path:
    """Create a temporary bundles directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def settings(temp_bundles_dir: Path) -> Settings:
    """Create settings with temp bundles path."""
    return Settings(
        bundles_path=temp_bundles_dir,
        comfyui_path=Path("/workspace/ComfyUI"),
    )


@pytest.fixture
def bundle_manager(settings: Settings) -> BundleManager:
    """Create a bundle manager instance."""
    return BundleManager(settings)


def create_test_bundle(
    bundles_dir: Path,
    bundle_name: str,
    version: str,
    with_current: bool = True,
) -> Path:
    """Helper to create a test bundle structure."""
    version_path = bundles_dir / bundle_name / version
    version_path.mkdir(parents=True, exist_ok=True)

    # Create bundle.yaml
    bundle_config = {
        "metadata": {
            "name": bundle_name,
            "version": version,
            "description": "Test bundle",
        },
        "comfyui": {
            "commit": "abc123def456789",
        },
        "custom_nodes": [
            {
                "name": "TestNode",
                "git_url": "https://github.com/test/node",
                "commit_sha": "def456",
            }
        ],
        "models": [],
    }

    with (version_path / "bundle.yaml").open("w") as f:
        yaml.dump(bundle_config, f)

    # Create requirements.lock
    (version_path / "requirements.lock").write_text("torch==2.1.0\n")

    # Create workflow.json
    workflow = {
        "nodes": [
            {"id": 1, "type": "KSampler"},
            {"id": 2, "type": "CLIPTextEncode"},
        ]
    }
    with (version_path / "workflow.json").open("w") as f:
        json.dump(workflow, f)

    # Create current symlink
    if with_current:
        current_link = bundles_dir / bundle_name / "current"
        if current_link.exists():
            current_link.unlink()
        current_link.symlink_to(version)

    return version_path


class TestBundleManager:
    """Tests for BundleManager."""

    def test_list_bundles_empty(self, bundle_manager: BundleManager) -> None:
        """Test listing bundles when none exist."""
        bundles = bundle_manager.list_bundles()
        assert bundles == []

    def test_list_bundles(
        self,
        bundle_manager: BundleManager,
        temp_bundles_dir: Path,
    ) -> None:
        """Test listing bundles."""
        create_test_bundle(temp_bundles_dir, "wan_2.2_i2v", "260101-01")
        create_test_bundle(temp_bundles_dir, "ltx_i2v", "260102-01")

        bundles = bundle_manager.list_bundles()
        assert len(bundles) == 2
        names = [b.name for b in bundles]
        assert "wan_2.2_i2v" in names
        assert "ltx_i2v" in names

    def test_get_bundle(
        self,
        bundle_manager: BundleManager,
        temp_bundles_dir: Path,
    ) -> None:
        """Test getting a specific bundle."""
        create_test_bundle(temp_bundles_dir, "wan_2.2_i2v", "260101-01")
        create_test_bundle(temp_bundles_dir, "wan_2.2_i2v", "260101-02", with_current=False)

        bundle_info = bundle_manager.get_bundle("wan_2.2_i2v")

        assert bundle_info.name == "wan_2.2_i2v"
        assert len(bundle_info.versions) == 2
        assert "260101-01" in bundle_info.versions
        assert "260101-02" in bundle_info.versions
        assert bundle_info.current_version == "260101-01"

    def test_get_bundle_not_found(self, bundle_manager: BundleManager) -> None:
        """Test getting a non-existent bundle."""
        with pytest.raises(BundleNotFoundError):
            bundle_manager.get_bundle("nonexistent")

    def test_load_bundle(
        self,
        bundle_manager: BundleManager,
        temp_bundles_dir: Path,
    ) -> None:
        """Test loading a bundle."""
        create_test_bundle(temp_bundles_dir, "wan_2.2_i2v", "260101-01")

        bundle_files = bundle_manager.load_bundle("wan_2.2_i2v")

        assert bundle_files.bundle_config.metadata.name == "wan_2.2_i2v"
        assert bundle_files.bundle_config.comfyui.commit == "abc123def456789"
        assert "torch==2.1.0" in bundle_files.requirements_lock
        assert len(bundle_files.workflow_json["nodes"]) == 2

    def test_load_bundle_specific_version(
        self,
        bundle_manager: BundleManager,
        temp_bundles_dir: Path,
    ) -> None:
        """Test loading a specific version."""
        create_test_bundle(temp_bundles_dir, "wan_2.2_i2v", "260101-01")
        create_test_bundle(temp_bundles_dir, "wan_2.2_i2v", "260101-02", with_current=False)

        bundle_files = bundle_manager.load_bundle("wan_2.2_i2v", "260101-02")

        assert bundle_files.bundle_config.metadata.version == "260101-02"

    def test_load_bundle_version_not_found(
        self,
        bundle_manager: BundleManager,
        temp_bundles_dir: Path,
    ) -> None:
        """Test loading a non-existent version."""
        create_test_bundle(temp_bundles_dir, "wan_2.2_i2v", "260101-01")

        with pytest.raises(BundleNotFoundError):
            bundle_manager.load_bundle("wan_2.2_i2v", "260199-99")

    def test_load_bundle_no_current(
        self,
        bundle_manager: BundleManager,
        temp_bundles_dir: Path,
    ) -> None:
        """Test loading when no current version is set."""
        create_test_bundle(
            temp_bundles_dir,
            "wan_2.2_i2v",
            "260101-01",
            with_current=False,
        )

        with pytest.raises(BundleNotFoundError) as exc_info:
            bundle_manager.load_bundle("wan_2.2_i2v")

        assert "no current version" in str(exc_info.value).lower()

    def test_set_current_version(
        self,
        bundle_manager: BundleManager,
        temp_bundles_dir: Path,
    ) -> None:
        """Test setting current version."""
        create_test_bundle(temp_bundles_dir, "wan_2.2_i2v", "260101-01")
        create_test_bundle(temp_bundles_dir, "wan_2.2_i2v", "260101-02", with_current=False)

        bundle_manager.set_current_version("wan_2.2_i2v", "260101-02")

        bundle_info = bundle_manager.get_bundle("wan_2.2_i2v")
        assert bundle_info.current_version == "260101-02"

    def test_set_current_version_not_found(
        self,
        bundle_manager: BundleManager,
        temp_bundles_dir: Path,
    ) -> None:
        """Test setting non-existent version as current."""
        create_test_bundle(temp_bundles_dir, "wan_2.2_i2v", "260101-01")

        with pytest.raises(BundleNotFoundError):
            bundle_manager.set_current_version("wan_2.2_i2v", "260199-99")

    def test_delete_version(
        self,
        bundle_manager: BundleManager,
        temp_bundles_dir: Path,
    ) -> None:
        """Test deleting a version."""
        create_test_bundle(temp_bundles_dir, "wan_2.2_i2v", "260101-01")
        create_test_bundle(temp_bundles_dir, "wan_2.2_i2v", "260101-02", with_current=False)

        bundle_manager.delete_version("wan_2.2_i2v", "260101-02")

        bundle_info = bundle_manager.get_bundle("wan_2.2_i2v")
        assert "260101-02" not in bundle_info.versions
        assert "260101-01" in bundle_info.versions

    def test_delete_current_version_fails(
        self,
        bundle_manager: BundleManager,
        temp_bundles_dir: Path,
    ) -> None:
        """Test that deleting current version fails."""
        create_test_bundle(temp_bundles_dir, "wan_2.2_i2v", "260101-01")

        with pytest.raises(BundleError) as exc_info:
            bundle_manager.delete_version("wan_2.2_i2v", "260101-01")

        assert "current version" in str(exc_info.value).lower()

    def test_expected_node_types(
        self,
        bundle_manager: BundleManager,
        temp_bundles_dir: Path,
    ) -> None:
        """Test extracting expected node types from workflow."""
        create_test_bundle(temp_bundles_dir, "wan_2.2_i2v", "260101-01")

        bundle_files = bundle_manager.load_bundle("wan_2.2_i2v")
        expected_nodes = bundle_files.expected_node_types

        assert "KSampler" in expected_nodes
        assert "CLIPTextEncode" in expected_nodes

    def test_resolve_bundle_from_args(
        self,
        bundle_manager: BundleManager,
        temp_bundles_dir: Path,
    ) -> None:
        """Test resolving bundle from explicit arguments."""
        create_test_bundle(temp_bundles_dir, "wan_2.2_i2v", "260101-01")

        name, version = bundle_manager.resolve_bundle("wan_2.2_i2v", "260101-01")

        assert name == "wan_2.2_i2v"
        assert version == "260101-01"

    def test_resolve_bundle_uses_current(
        self,
        bundle_manager: BundleManager,
        temp_bundles_dir: Path,
    ) -> None:
        """Test resolving bundle uses current symlink."""
        create_test_bundle(temp_bundles_dir, "wan_2.2_i2v", "260101-01")

        name, version = bundle_manager.resolve_bundle("wan_2.2_i2v", None)

        assert name == "wan_2.2_i2v"
        assert version == "260101-01"  # from current symlink

    def test_resolve_bundle_no_name(
        self,
        bundle_manager: BundleManager,
    ) -> None:
        """Test resolving bundle without name fails."""
        with pytest.raises(BundleError) as exc_info:
            bundle_manager.resolve_bundle(None, None)

        assert "no bundle specified" in str(exc_info.value).lower()


class TestBundleValidation:
    """Tests for bundle validation."""

    def test_missing_bundle_yaml(
        self,
        bundle_manager: BundleManager,
        temp_bundles_dir: Path,
    ) -> None:
        """Test that missing bundle.yaml fails validation."""
        version_path = temp_bundles_dir / "test" / "260101-01"
        version_path.mkdir(parents=True)

        # Create only requirements.lock and workflow.json
        (version_path / "requirements.lock").write_text("torch==2.1.0\n")
        (version_path / "workflow.json").write_text("{}")

        # Create current symlink
        (temp_bundles_dir / "test" / "current").symlink_to("260101-01")

        with pytest.raises(BundleValidationError) as exc_info:
            bundle_manager.load_bundle("test")

        assert "bundle.yaml" in str(exc_info.value).lower()

    def test_missing_requirements_lock(
        self,
        bundle_manager: BundleManager,
        temp_bundles_dir: Path,
    ) -> None:
        """Test that missing requirements.lock fails validation."""
        version_path = temp_bundles_dir / "test" / "260101-01"
        version_path.mkdir(parents=True)

        # Create bundle.yaml and workflow.json only
        bundle_config = {
            "metadata": {"name": "test", "version": "260101-01"},
            "comfyui": {"commit": "abc123"},
        }
        with (version_path / "bundle.yaml").open("w") as f:
            yaml.dump(bundle_config, f)
        (version_path / "workflow.json").write_text("{}")

        # Create current symlink
        (temp_bundles_dir / "test" / "current").symlink_to("260101-01")

        with pytest.raises(BundleValidationError) as exc_info:
            bundle_manager.load_bundle("test")

        assert "requirements.lock" in str(exc_info.value).lower()

    def test_missing_workflow_json(
        self,
        bundle_manager: BundleManager,
        temp_bundles_dir: Path,
    ) -> None:
        """Test that missing workflow.json fails validation."""
        version_path = temp_bundles_dir / "test" / "260101-01"
        version_path.mkdir(parents=True)

        # Create bundle.yaml and requirements.lock only
        bundle_config = {
            "metadata": {"name": "test", "version": "260101-01"},
            "comfyui": {"commit": "abc123"},
        }
        with (version_path / "bundle.yaml").open("w") as f:
            yaml.dump(bundle_config, f)
        (version_path / "requirements.lock").write_text("torch==2.1.0\n")

        # Create current symlink
        (temp_bundles_dir / "test" / "current").symlink_to("260101-01")

        with pytest.raises(BundleValidationError) as exc_info:
            bundle_manager.load_bundle("test")

        assert "workflow.json" in str(exc_info.value).lower()
