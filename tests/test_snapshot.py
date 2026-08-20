"""Tests for snapshot management."""

import hashlib
import json
import os
import re
import tempfile
import threading
from collections.abc import Iterator, Mapping
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import yaml
from pydantic import ValidationError

from ai_content_service.bundle_contract import Severity, check_bundle_contract
from ai_content_service.bundle_registry import LocalBundleRegistry
from ai_content_service.config import (
    BundleConfig,
    BundleMetadata,
    BundleVersion,
    CustomNodeConfig,
    ModelConfig,
    ModelFileConfig,
    WorkflowNodeConfig,
)
from ai_content_service.snapshot import (
    CarryForwardReport,
    SnapshotError,
    SnapshotManager,
    _hash_model_file,
    _HashResult,
    _render_bundle_yaml,
    _write_bundle_files,
)
from ai_content_service.workflow_map import _normalize_workflow_comment


@pytest.fixture
def temp_dir() -> Iterator[Path]:
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def comfyui_path(temp_dir: Path) -> Path:
    path = temp_dir / "ComfyUI"
    path.mkdir()
    return path


@pytest.fixture
def bundles_path(temp_dir: Path) -> Path:
    path = temp_dir / "bundles"
    path.mkdir()
    return path


@pytest.fixture
def python_executable(temp_dir: Path) -> Path:
    path = temp_dir / "python"
    path.write_text("")
    return path


@pytest.fixture
def snapshot_manager(
    comfyui_path: Path, bundles_path: Path, python_executable: Path
) -> SnapshotManager:
    return SnapshotManager(comfyui_path, bundles_path, python_executable=python_executable)


@pytest.fixture
def workflow_file(temp_dir: Path) -> Path:
    wf = temp_dir / "workflow.json"
    wf.write_text(json.dumps({"nodes": []}))
    return wf


def make_mock_process(returncode: int = 0, stdout: bytes = b"") -> MagicMock:
    proc = MagicMock()
    proc.returncode = returncode
    proc.communicate = AsyncMock(return_value=(stdout, b""))
    return proc


class TestCreateSnapshotValidation:
    async def test_raises_when_comfyui_not_found(
        self, temp_dir: Path, bundles_path: Path, workflow_file: Path, python_executable: Path
    ) -> None:
        manager = SnapshotManager(
            temp_dir / "nonexistent", bundles_path, python_executable=python_executable
        )
        with pytest.raises(SnapshotError, match="ComfyUI not found"):
            await manager.create_snapshot("mybundle", workflow_file)

    async def test_raises_when_workflow_not_found(
        self, snapshot_manager: SnapshotManager, temp_dir: Path
    ) -> None:
        with pytest.raises(SnapshotError, match="Workflow not found"):
            await snapshot_manager.create_snapshot("mybundle", temp_dir / "missing.json")


class TestRequiredCustomNodeSnapshot:
    @staticmethod
    def _registry_node(comfyui_path: Path, name: str, repository: str) -> None:
        node_dir = comfyui_path / "custom_nodes" / name
        node_dir.mkdir(parents=True)
        (node_dir / "pyproject.toml").write_text(
            "[project]\n"
            f'name = "{name}"\n'
            'version = "1.5.0"\n'
            "[project.urls]\n"
            f'Repository = "{repository}"\n'
        )

    @staticmethod
    async def _capture(
        manager: SnapshotManager,
        workflow_file: Path,
        name: str,
        api_graph: Mapping[str, object],
        object_info: Mapping[str, object] | None,
        *,
        allow_unverified_custom_nodes: bool = False,
    ) -> tuple[str, CarryForwardReport]:
        with (
            patch.object(manager, "_git", new=AsyncMock(return_value=(1, "", ""))),
            patch.object(manager, "_resolve_registry_pin", new=AsyncMock(return_value=None)),
            patch.object(
                manager, "_snapshot_workflow_api", new=AsyncMock(return_value=(api_graph, ()))
            ),
            patch.object(
                manager,
                "_fetch_object_info",
                new=AsyncMock(return_value=(object_info, None)),
            ),
        ):
            return await manager.create_snapshot(
                name,
                workflow_file,
                scan_models=False,
                include_workflow_map=False,
                allow_unverified_custom_nodes=allow_unverified_custom_nodes,
            )

    async def test_required_skipped_node_aborts_and_removes_bundle(
        self,
        snapshot_manager: SnapshotManager,
        comfyui_path: Path,
        workflow_file: Path,
        bundles_path: Path,
    ) -> None:
        self._registry_node(
            comfyui_path,
            "comfyui-kjnodes",
            "https://github.com/kijai/ComfyUI-KJNodes",
        )
        api_graph = {"72": {"class_type": "PatchFlashAttentionKJ", "inputs": {}}}
        object_info = {"PatchFlashAttentionKJ": {"python_module": "custom_nodes.comfyui-kjnodes"}}

        with pytest.raises(SnapshotError) as error:
            await self._capture(snapshot_manager, workflow_file, "probe", api_graph, object_info)

        message = str(error.value)
        assert "PatchFlashAttentionKJ" in message
        assert "comfyui-kjnodes" in message
        assert "no_git_metadata" in message
        assert "git clone https://github.com/kijai/ComfyUI-KJNodes" in message
        assert not (bundles_path / "probe").exists()

    async def test_required_skipped_node_abort_is_not_suppressed_by_allow_unverified(
        self,
        snapshot_manager: SnapshotManager,
        comfyui_path: Path,
        workflow_file: Path,
        bundles_path: Path,
    ) -> None:
        """A positively-identified missing provider is never overridable (P1 regression)."""
        self._registry_node(
            comfyui_path,
            "comfyui-kjnodes",
            "https://github.com/kijai/ComfyUI-KJNodes",
        )
        api_graph = {"72": {"class_type": "PatchFlashAttentionKJ", "inputs": {}}}
        object_info = {"PatchFlashAttentionKJ": {"python_module": "custom_nodes.comfyui-kjnodes"}}

        with pytest.raises(SnapshotError) as error:
            await self._capture(
                snapshot_manager,
                workflow_file,
                "probe",
                api_graph,
                object_info,
                allow_unverified_custom_nodes=True,
            )

        assert "PatchFlashAttentionKJ" in str(error.value)
        assert not (bundles_path / "probe").exists()

    async def test_reports_every_required_skipped_node(
        self,
        snapshot_manager: SnapshotManager,
        comfyui_path: Path,
        workflow_file: Path,
    ) -> None:
        self._registry_node(comfyui_path, "first-node", "https://github.com/example/first")
        self._registry_node(comfyui_path, "second-node", "https://github.com/example/second")
        api_graph = {
            "1": {"class_type": "FirstClass", "inputs": {}},
            "2": {"class_type": "SecondClass", "inputs": {}},
        }
        object_info = {
            "FirstClass": {"python_module": "custom_nodes.first-node"},
            "SecondClass": {"python_module": "custom_nodes.second-node"},
        }

        with pytest.raises(SnapshotError) as error:
            await self._capture(snapshot_manager, workflow_file, "probe", api_graph, object_info)

        assert "FirstClass" in str(error.value)
        assert "SecondClass" in str(error.value)

    async def test_unrelated_skipped_node_succeeds_when_all_classes_are_core(
        self,
        snapshot_manager: SnapshotManager,
        comfyui_path: Path,
        workflow_file: Path,
        bundles_path: Path,
    ) -> None:
        self._registry_node(comfyui_path, "registry-node", "https://github.com/example/node")
        version, report = await self._capture(
            snapshot_manager,
            workflow_file,
            "probe",
            {"2": {"class_type": "KSampler", "inputs": {}}},
            {"KSampler": {"python_module": "nodes"}},
        )

        assert report.custom_nodes.skipped[0].name == "registry-node"
        assert not report.has_unverified_custom_nodes
        assert (bundles_path / "probe" / version).is_dir()

    async def test_unreachable_object_info_writes_unverified_bundle(
        self,
        snapshot_manager: SnapshotManager,
        comfyui_path: Path,
        workflow_file: Path,
        bundles_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        self._registry_node(comfyui_path, "registry-node", "https://github.com/example/node")
        api_graph = {"2": {"class_type": "KSampler", "inputs": {}}}
        with (
            patch.object(snapshot_manager, "_git", new=AsyncMock(return_value=(1, "", ""))),
            patch.object(
                snapshot_manager, "_resolve_registry_pin", new=AsyncMock(return_value=None)
            ),
            patch.object(
                snapshot_manager,
                "_snapshot_workflow_api",
                new=AsyncMock(return_value=(api_graph, ())),
            ),
            patch.object(
                snapshot_manager,
                "_fetch_object_info",
                new=AsyncMock(return_value=(None, "ComfyUI is unavailable")),
            ),
            caplog.at_level("WARNING", logger="ai_content_service.snapshot"),
        ):
            version, report = await snapshot_manager.create_snapshot(
                "probe", workflow_file, scan_models=False, include_workflow_map=False
            )

        assert report.has_unverified_custom_nodes
        assert (bundles_path / "probe" / version).is_dir()
        assert not (bundles_path / "probe" / "current").exists()
        events = [
            record.msg
            for record in caplog.records
            if isinstance(record.msg, dict)
            and record.msg.get("event") == "snapshot.custom_node_skipped_unverified"
        ]
        assert events[0]["name"] == "registry-node"
        assert events[0]["reason"] == "ComfyUI is unavailable"

    async def test_missing_python_module_is_unverified_and_can_be_explicitly_allowed(
        self,
        snapshot_manager: SnapshotManager,
        comfyui_path: Path,
        workflow_file: Path,
        bundles_path: Path,
    ) -> None:
        self._registry_node(comfyui_path, "registry-node", "https://github.com/example/node")
        version, report = await self._capture(
            snapshot_manager,
            workflow_file,
            "probe",
            {"2": {"class_type": "KSampler", "inputs": {}}},
            {"KSampler": {}},
            allow_unverified_custom_nodes=True,
        )

        assert report.has_unverified_custom_nodes
        assert "no python_module" in report.custom_nodes.unverified[0].reason
        assert (bundles_path / "probe" / "current").resolve().name == version

    async def test_seed_pin_covers_skipped_workflow_provider(
        self,
        snapshot_manager: SnapshotManager,
        comfyui_path: Path,
        workflow_file: Path,
        bundles_path: Path,
    ) -> None:
        self._registry_node(
            comfyui_path,
            "comfyui-kjnodes",
            "https://github.com/kijai/ComfyUI-KJNodes",
        )
        seed_node = CustomNodeConfig(
            name="ComfyUI-KJNodes",
            git_url="https://github.com/kijai/ComfyUI-KJNodes",
            commit_sha="3f20054214fec9f9234fd3841ae6f1e4287948f6",
        )
        seed = BundleConfig(
            metadata=BundleMetadata(name="seed", version="260101-01"),
            custom_nodes=[seed_node],
            workflow_file="workflow.json",
        )
        api_graph = {"72": {"class_type": "PatchFlashAttentionKJ", "inputs": {}}}
        object_info = {"PatchFlashAttentionKJ": {"python_module": "custom_nodes.comfyui-kjnodes"}}

        with (
            patch.object(snapshot_manager, "_git", new=AsyncMock(return_value=(1, "", ""))),
            patch.object(
                snapshot_manager, "_resolve_registry_pin", new=AsyncMock(return_value=None)
            ),
            patch.object(
                snapshot_manager,
                "_snapshot_workflow_api",
                new=AsyncMock(return_value=(api_graph, ())),
            ),
            patch.object(
                snapshot_manager,
                "_fetch_object_info",
                new=AsyncMock(return_value=(object_info, None)),
            ),
        ):
            version, report = await snapshot_manager.create_snapshot(
                "probe",
                workflow_file,
                scan_models=False,
                include_workflow_map=False,
                carry_from=seed,
            )

        bundle = BundleConfig.model_validate(
            yaml.safe_load((bundles_path / "probe" / version / "bundle.yaml").read_text())
        )
        assert bundle.custom_nodes == [seed_node]
        assert report.custom_nodes.carried == ("ComfyUI-KJNodes",)
        assert report.custom_nodes.attributed[0].class_name == "PatchFlashAttentionKJ"
        assert report.custom_nodes.required == ()


class TestGenerateVersion:
    def test_new_bundle_gets_first_version(self, snapshot_manager: SnapshotManager) -> None:
        today = datetime.now(timezone.utc).strftime("%y%m%d")
        version = snapshot_manager._generate_version("new_bundle")
        assert version == f"{today}-01"

    def test_increments_sequence_for_existing_today_versions(
        self, snapshot_manager: SnapshotManager, bundles_path: Path
    ) -> None:
        today = datetime.now(timezone.utc).strftime("%y%m%d")
        bundle_dir = bundles_path / "mybundle"
        (bundle_dir / f"{today}-01").mkdir(parents=True)

        version = snapshot_manager._generate_version("mybundle")
        assert version == f"{today}-02"

    def test_increments_past_existing_max_sequence(
        self, snapshot_manager: SnapshotManager, bundles_path: Path
    ) -> None:
        today = datetime.now(timezone.utc).strftime("%y%m%d")
        bundle_dir = bundles_path / "mybundle"
        (bundle_dir / f"{today}-01").mkdir(parents=True)
        (bundle_dir / f"{today}-05").mkdir(parents=True)

        version = snapshot_manager._generate_version("mybundle")
        assert version == f"{today}-06"

    def test_previous_day_versions_do_not_affect_sequence(
        self, snapshot_manager: SnapshotManager, bundles_path: Path
    ) -> None:
        today = datetime.now(timezone.utc).strftime("%y%m%d")
        bundle_dir = bundles_path / "mybundle"
        (bundle_dir / "250101-99").mkdir(parents=True)  # old date

        version = snapshot_manager._generate_version("mybundle")
        assert version == f"{today}-01"

    def test_skips_gaps_after_deletion(
        self, snapshot_manager: SnapshotManager, bundles_path: Path
    ) -> None:
        """Regression guard: version generation must use max-sequence, not count."""
        today = datetime.now(timezone.utc).strftime("%y%m%d")
        bundle_dir = bundles_path / "mybundle"
        # Simulate versions -01 and -03 existing after -02 was deleted.
        (bundle_dir / f"{today}-01").mkdir(parents=True)
        (bundle_dir / f"{today}-03").mkdir(parents=True)

        version = snapshot_manager._generate_version("mybundle")
        assert version == f"{today}-04"

    def test_generate_version_uses_utc(self, snapshot_manager: SnapshotManager) -> None:
        """_generate_version must delegate to BundleVersion.create_new (UTC-based)."""
        with patch.object(
            BundleVersion, "create_new", return_value=BundleVersion(version="990101-01")
        ) as mock_create_new:
            version = snapshot_manager._generate_version("new_bundle")

        mock_create_new.assert_called_once_with([])
        assert version == "990101-01"


class TestCreateSnapshotSuccess:
    async def test_indexed_registry_snapshot_lands_in_bundles_and_resolves(
        self,
        comfyui_path: Path,
        workflow_file: Path,
        python_executable: Path,
        temp_dir: Path,
    ) -> None:
        registry_root = temp_dir / "ai-bundles"
        registry_root.mkdir()
        (registry_root / "bundles").mkdir()
        (registry_root / "bundle-index.yaml").write_text(
            yaml.safe_dump({"bundles": [{"name": "snapshot", "path": "bundles/snapshot"}]})
        )
        manager = SnapshotManager(
            comfyui_path,
            registry_root,
            python_executable=python_executable,
        )
        ok = make_mock_process(returncode=0, stdout=b"abc123\n")

        with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=ok)):
            version, _ = await manager.create_snapshot("snapshot", workflow_file)

        bundle_dir = registry_root / "bundles" / "snapshot" / version
        assert bundle_dir.is_dir()
        assert not (registry_root / "snapshot").exists()
        current = bundle_dir.parent / "current"
        assert current.resolve() == bundle_dir.resolve()
        resolved = await LocalBundleRegistry(registry_root).resolve_bundle_path("snapshot")
        assert resolved == bundle_dir.resolve()

    async def test_creates_bundle_directory(
        self,
        snapshot_manager: SnapshotManager,
        workflow_file: Path,
        bundles_path: Path,
    ) -> None:
        ok_commit = make_mock_process(returncode=0, stdout=b"deadbeef\n")
        ok_pip = make_mock_process(returncode=0, stdout=b"torch==2.1.0\n")

        call_count = 0

        async def mock_exec(*args: object, **_kwargs: object) -> MagicMock:
            nonlocal call_count
            call_count += 1
            # pip freeze returns requirements, git returns commit hash
            return ok_pip if "freeze" in args else ok_commit

        with patch("asyncio.create_subprocess_exec", new=mock_exec):
            version, _ = await snapshot_manager.create_snapshot("mybundle", workflow_file)

        bundle_dir = bundles_path / "mybundle" / version
        assert bundle_dir.is_dir()

    async def test_writes_bundle_yaml(
        self,
        snapshot_manager: SnapshotManager,
        workflow_file: Path,
        bundles_path: Path,
    ) -> None:
        ok = make_mock_process(returncode=0, stdout=b"abc123\n")
        with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=ok)):
            version, _ = await snapshot_manager.create_snapshot(
                "mybundle", workflow_file, description="test"
            )

        config_path = bundles_path / "mybundle" / version / "bundle.yaml"
        assert config_path.exists()
        config = yaml.safe_load(config_path.read_text())
        assert config["metadata"]["name"] == "mybundle"
        assert config["metadata"]["description"] == "test"

    async def test_created_snapshot_passes_bundle_contract(
        self,
        snapshot_manager: SnapshotManager,
        bundles_path: Path,
        temp_dir: Path,
    ) -> None:
        gui_graph = {
            "nodes": [
                {
                    "id": 9,
                    "type": "EmptyLatentImage",
                    "inputs": [{"name": "width", "widget": {}}],
                    "widgets_values": [512],
                },
                {
                    "id": 3,
                    "type": "CLIPTextEncode",
                    "inputs": [{"name": "text", "widget": {}}],
                    "widgets_values": ["hello"],
                },
                {
                    "id": 2,
                    "type": "KSampler",
                    "inputs": [
                        {"name": "positive"},
                        {"name": "latent_image"},
                        {"name": "steps", "widget": {}},
                    ],
                    "widgets_values": [8],
                },
            ],
            "links": [
                [1, 3, 0, 2, 0, "CONDITIONING"],
                [2, 9, 0, 2, 1, "LATENT"],
            ],
        }
        api_graph = {
            "9": {"class_type": "EmptyLatentImage", "inputs": {"width": 512}},
            "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "hello"}},
            "2": {
                "class_type": "KSampler",
                "inputs": {"positive": ["3", 0], "latent_image": ["9", 0], "steps": 8},
            },
        }
        workflow_path = temp_dir / "workflow.json"
        workflow_path.write_text(json.dumps(gui_graph))

        async def convert(
            _workflow_path: Path, _rejected_path: Path
        ) -> tuple[object, tuple[str, ...]]:
            return api_graph, ()

        carry_from = BundleConfig.model_validate(
            {
                "metadata": {"name": "seed", "version": "260101-01"},
                "hardware": {
                    "gpu_whitelist": ["RTX 4090"],
                    "min_disk_gb": 100,
                    "min_network_upload_mbps": 100,
                    "min_network_download_mbps": 100,
                    "cuda_min_version": "12.1",
                    "num_gpus": 1,
                    "comfyui_port": 18188,
                },
            }
        )
        ok = make_mock_process(returncode=0, stdout=b"abc123\n")
        with (
            patch.object(snapshot_manager, "_snapshot_workflow_api", new=convert),
            patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=ok)),
        ):
            version, _ = await snapshot_manager.create_snapshot(
                "demo", workflow_path, scan_models=False, carry_from=carry_from
            )

        bundle = bundles_path / "demo" / version
        raw_bundle = yaml.safe_load((bundle / "bundle.yaml").read_text())
        report = check_bundle_contract(
            "demo",
            bundle,
            raw_bundle,
            bundle_root=bundle.parent,
            index_entries=({"name": "demo", "model_type": "aisha-image"},),
        )

        assert not [finding for finding in report.findings if finding.severity is Severity.ERROR]

    async def test_created_snapshot_exercises_image_and_model_inputs_passes_bundle_contract(
        self,
        snapshot_manager: SnapshotManager,
        comfyui_path: Path,
        bundles_path: Path,
        temp_dir: Path,
    ) -> None:
        """P1-1: a Qwen-shaped, split-loader bundle round-trips with zero ERROR.

        This is the acceptance test for the whole arc -- a bundle aisha writes
        must be a bundle aisha accepts -- exercised against the two workflow
        map shapes real bundles actually use: an image input wired into the
        positive prompt, and a checkpoint split across three loaders.
        """
        gui_graph = {
            "nodes": [
                {
                    "id": 9,
                    "type": "EmptyLatentImage",
                    "inputs": [{"name": "width", "widget": {}}],
                    "widgets_values": [1024],
                },
                {
                    "id": 3,
                    "type": "TextEncodeQwenImageEditPlus",
                    "inputs": [{"name": "prompt", "widget": {}}, {"name": "image1"}],
                    "widgets_values": ["hello"],
                },
                {
                    "id": 4,
                    "type": "LoadImage",
                    "inputs": [{"name": "image", "widget": {}}],
                    "widgets_values": ["input.png"],
                },
                {
                    "id": 2,
                    "type": "KSampler",
                    "inputs": [
                        {"name": "positive"},
                        {"name": "latent_image"},
                        {"name": "steps", "widget": {}},
                    ],
                    "widgets_values": [8],
                },
                {
                    "id": 10,
                    "type": "UNETLoader",
                    "inputs": [{"name": "unet_name", "widget": {}}],
                    "widgets_values": ["unet.safetensors"],
                },
                {
                    "id": 11,
                    "type": "CLIPLoader",
                    "inputs": [{"name": "clip_name", "widget": {}}],
                    "widgets_values": ["clip.safetensors"],
                },
                {
                    "id": 12,
                    "type": "VAELoader",
                    "inputs": [{"name": "vae_name", "widget": {}}],
                    "widgets_values": ["vae.safetensors"],
                },
            ],
            "links": [
                [1, 3, 0, 2, 0, "CONDITIONING"],
                [2, 9, 0, 2, 1, "LATENT"],
                [3, 4, 0, 3, 1, "IMAGE"],
            ],
        }
        api_graph = {
            "9": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024}},
            "3": {
                "class_type": "TextEncodeQwenImageEditPlus",
                "inputs": {"prompt": "hello", "image1": ["4", 0]},
            },
            "4": {"class_type": "LoadImage", "inputs": {"image": "input.png"}},
            "2": {
                "class_type": "KSampler",
                "inputs": {"positive": ["3", 0], "latent_image": ["9", 0], "steps": 8},
            },
            "10": {"class_type": "UNETLoader", "inputs": {"unet_name": "unet.safetensors"}},
            "11": {"class_type": "CLIPLoader", "inputs": {"clip_name": "clip.safetensors"}},
            "12": {"class_type": "VAELoader", "inputs": {"vae_name": "vae.safetensors"}},
        }
        workflow_path = temp_dir / "workflow.json"
        workflow_path.write_text(json.dumps(gui_graph))

        (comfyui_path / "models" / "diffusion_models").mkdir(parents=True)
        (comfyui_path / "models" / "text_encoders").mkdir(parents=True)
        (comfyui_path / "models" / "vae").mkdir(parents=True)
        (comfyui_path / "models" / "diffusion_models" / "unet.safetensors").write_bytes(b"unet")
        (comfyui_path / "models" / "text_encoders" / "clip.safetensors").write_bytes(b"clip")
        (comfyui_path / "models" / "vae" / "vae.safetensors").write_bytes(b"vae")

        async def convert(
            _workflow_path: Path, _rejected_path: Path
        ) -> tuple[object, tuple[str, ...]]:
            return api_graph, ()

        carry_from = BundleConfig.model_validate(
            {
                "metadata": {"name": "seed", "version": "260101-01"},
                "hardware": {
                    "gpu_whitelist": ["RTX 4090"],
                    "min_disk_gb": 100,
                    "min_network_upload_mbps": 100,
                    "min_network_download_mbps": 100,
                    "cuda_min_version": "12.1",
                    "num_gpus": 1,
                    "comfyui_port": 18188,
                },
            }
        )
        ok = make_mock_process(returncode=0, stdout=b"abc123\n")
        with (
            patch.object(snapshot_manager, "_snapshot_workflow_api", new=convert),
            patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=ok)),
        ):
            version, _ = await snapshot_manager.create_snapshot(
                "demo", workflow_path, carry_from=carry_from
            )

        bundle = bundles_path / "demo" / version
        raw_bundle = yaml.safe_load((bundle / "bundle.yaml").read_text())
        config = BundleConfig.model_validate(raw_bundle)
        assert config.workflow is not None
        assert len(config.workflow.image_inputs) == 1
        assert len(config.workflow.model_inputs) == 3

        report = check_bundle_contract(
            "demo",
            bundle,
            raw_bundle,
            bundle_root=bundle.parent,
            index_entries=({"name": "demo", "model_type": "aisha-image"},),
        )

        assert not [finding for finding in report.findings if finding.severity is Severity.ERROR]

    @pytest.mark.parametrize(
        "converter_mode",
        (
            "status_400",
            "status_413",
            "status_500",
            "connection_error",
            "timeout",
            "non_json",
            "non_api_shaped",
            "empty_dict",
            "null",
            "sync_rejection",
            "valid",
        ),
    )
    async def test_converter_modes_still_write_parseable_bundle_yaml(
        self,
        converter_mode: str,
        comfyui_path: Path,
        bundles_path: Path,
        python_executable: Path,
        temp_dir: Path,
    ) -> None:
        workflow_path = temp_dir / "workflow.json"
        workflow_path.write_text(
            json.dumps(
                {
                    "nodes": [{"id": 1, "type": "KSampler", "inputs": [], "widgets_values": []}],
                    "links": [],
                }
            )
        )
        manager = SnapshotManager(
            comfyui_path,
            bundles_path,
            python_executable=python_executable,
            comfyui_url="http://comfyui.local",
        )
        response = MagicMock()
        response.status_code = {
            "status_400": 400,
            "status_413": 413,
            "status_500": 500,
        }.get(converter_mode, 200)
        if converter_mode == "non_json":
            response.json.side_effect = ValueError("not json")
        elif converter_mode == "non_api_shaped":
            response.json.return_value = {"bad": "shape"}
        elif converter_mode == "empty_dict":
            response.json.return_value = {}
        elif converter_mode == "null":
            response.json.return_value = None
        elif converter_mode == "sync_rejection":
            # API-shaped and link-clean, but disagrees with the GUI graph's
            # class_type -- a 200 that check_workflow_sync must still reject,
            # a different branch than any transport failure above.
            response.json.return_value = {"1": {"class_type": "VAEDecode", "inputs": {}}}
        else:
            response.json.return_value = {"1": {"class_type": "KSampler", "inputs": {}}}
        client = MagicMock()
        client.get = AsyncMock(side_effect=httpx.HTTPError("no version endpoint"))
        if converter_mode == "connection_error":
            client.post = AsyncMock(side_effect=httpx.ConnectError("connection refused"))
        elif converter_mode == "timeout":
            client.post = AsyncMock(side_effect=httpx.ReadTimeout("timed out"))
        else:
            client.post = AsyncMock(return_value=response)
        ok = make_mock_process(returncode=0, stdout=b"abc123\n")

        with (
            patch(
                "ai_content_service.snapshot.httpx.AsyncClient", return_value=_make_async_cm(client)
            ),
            patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=ok)),
        ):
            version, _ = await manager.create_snapshot(
                f"converter-{converter_mode}", workflow_path, scan_models=False
            )

        bundle_yaml = bundles_path / f"converter-{converter_mode}" / version / "bundle.yaml"
        parsed = yaml.safe_load(bundle_yaml.read_text())
        assert isinstance(parsed, dict)
        BundleConfig.model_validate(parsed)

    async def test_rejected_response_paths_are_versioned_outside_bundles(
        self,
        snapshot_manager: SnapshotManager,
        workflow_file: Path,
        bundles_path: Path,
        temp_dir: Path,
    ) -> None:
        rejected_dirs = [temp_dir / "diagnostic-one", temp_dir / "diagnostic-two"]
        for directory in rejected_dirs:
            directory.mkdir()
        rejected_paths: list[Path] = []

        async def reject_api(
            _workflow_path: Path, rejected_path: Path
        ) -> tuple[None, tuple[str, ...]]:
            rejected_path.write_text("{}\n")
            rejected_paths.append(rejected_path)
            return None, ()

        ok = make_mock_process(returncode=0, stdout=b"abc123\n")
        with (
            patch(
                "ai_content_service.snapshot.tempfile.mkdtemp",
                side_effect=[str(directory) for directory in rejected_dirs],
            ),
            patch.object(snapshot_manager, "_snapshot_workflow_api", new=reject_api),
            patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=ok)),
        ):
            first_version, _ = await snapshot_manager.create_snapshot("demo", workflow_file)
            second_version, _ = await snapshot_manager.create_snapshot("demo", workflow_file)

        assert first_version != second_version
        assert len(rejected_paths) == 2
        assert rejected_paths[0] != rejected_paths[1]
        assert all(
            path.exists() and not path.is_relative_to(bundles_path) for path in rejected_paths
        )
        assert first_version in rejected_paths[0].name
        assert second_version in rejected_paths[1].name

    async def test_writes_additive_requirements_overlay(
        self,
        snapshot_manager: SnapshotManager,
        workflow_file: Path,
        bundles_path: Path,
        temp_dir: Path,
    ) -> None:
        base_manifest = temp_dir / "base-manifest.json"
        base_manifest.write_text(json.dumps({"packages": {"torch": "2.1.0"}}))
        pip_output = b"torch==2.1.0\nnumpy==1.24.0\n"
        ok_commit = make_mock_process(returncode=0, stdout=b"abc123\n")
        ok_pip = make_mock_process(returncode=0, stdout=pip_output)

        async def mock_exec(*args: object, **_kwargs: object) -> MagicMock:
            return ok_pip if "freeze" in args else ok_commit

        with patch("asyncio.create_subprocess_exec", new=mock_exec):
            version, _ = await snapshot_manager.create_snapshot(
                "mybundle", workflow_file, base_manifest=base_manifest
            )

        req_path = bundles_path / "mybundle" / version / "requirements.overlay.txt"
        assert req_path.exists()
        assert req_path.read_text() == "numpy==1.24.0\n"
        config = yaml.safe_load((req_path.parent / "bundle.yaml").read_text())
        assert config["requirements_overlay_file"] == "requirements.overlay.txt"
        assert "requirements_lock_file" not in config

    async def test_missing_base_manifest_warns_and_writes_no_requirements_file(
        self,
        snapshot_manager: SnapshotManager,
        workflow_file: Path,
        bundles_path: Path,
        temp_dir: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        with caplog.at_level("WARNING", logger="ai_content_service.snapshot"):
            version, _ = await snapshot_manager.create_snapshot(
                "mybundle", workflow_file, base_manifest=temp_dir / "missing-manifest.json"
            )

        bundle_dir = bundles_path / "mybundle" / version
        config = yaml.safe_load((bundle_dir / "bundle.yaml").read_text())
        assert "requirements_overlay_file" not in config
        assert "requirements_lock_file" not in config
        assert not list(bundle_dir.glob("requirements.*"))
        warnings = [
            record.msg
            for record in caplog.records
            if isinstance(record.msg, dict)
            and record.msg.get("event") == "snapshot.overlay_skipped"
        ]
        assert len(warnings) == 1
        warning = warnings[0]
        assert warning["message"] == (
            "No usable base manifest was found; snapshot will carry no requirements file."
        )
        assert warning["base_manifest"] == str(temp_dir / "missing-manifest.json")
        assert "No such file or directory" in str(warning["error"])

    async def test_overlay_is_sorted_and_excludes_missing_local_file_reference(
        self,
        snapshot_manager: SnapshotManager,
        workflow_file: Path,
        bundles_path: Path,
        temp_dir: Path,
    ) -> None:
        base_manifest = temp_dir / "base-manifest.json"
        base_manifest.write_text(json.dumps({"packages": {"numpy": "1.24.0", "torch": "2.1.0"}}))
        pip_output = (
            b"torch==2.1.0\nzebra==3.0\nnumpy==2.0.0\npackaging @ file:///conda-builder/packaging\n"
        )
        ok_commit = make_mock_process(returncode=0, stdout=b"abc123\n")
        ok_pip = make_mock_process(returncode=0, stdout=pip_output)

        async def mock_exec(*args: object, **_kwargs: object) -> MagicMock:
            return ok_pip if "freeze" in args else ok_commit

        with patch("asyncio.create_subprocess_exec", new=mock_exec):
            version, _ = await snapshot_manager.create_snapshot(
                "mybundle", workflow_file, base_manifest=base_manifest
            )

        overlay = bundles_path / "mybundle" / version / "requirements.overlay.txt"
        assert overlay.read_text() == "numpy==2.0.0\nzebra==3.0\n"

    async def test_overlay_retains_portable_git_direct_reference_verbatim(
        self,
        snapshot_manager: SnapshotManager,
        workflow_file: Path,
        bundles_path: Path,
        temp_dir: Path,
    ) -> None:
        """R2: a git-referenced dependency is exactly what an unreleased custom-node

        fork produces, and must survive into the only requirements artifact a
        snapshot now emits — not be silently dropped as a "base-image conda
        reference".
        """
        base_manifest = temp_dir / "base-manifest.json"
        base_manifest.write_text(json.dumps({"packages": {"torch": "2.1.0"}}))
        git_ref = "ddt @ git+https://github.com/datadriventests/ddt@" + "a" * 40
        pip_output = f"torch==2.1.0\n{git_ref}\n".encode()
        ok_commit = make_mock_process(returncode=0, stdout=b"abc123\n")
        ok_pip = make_mock_process(returncode=0, stdout=pip_output)

        async def mock_exec(*args: object, **_kwargs: object) -> MagicMock:
            return ok_pip if "freeze" in args else ok_commit

        with patch("asyncio.create_subprocess_exec", new=mock_exec):
            version, _ = await snapshot_manager.create_snapshot(
                "mybundle", workflow_file, base_manifest=base_manifest
            )

        overlay = bundles_path / "mybundle" / version / "requirements.overlay.txt"
        assert overlay.read_text() == f"{git_ref}\n"

    async def test_overlay_retains_wheel_url_direct_reference_verbatim(
        self,
        snapshot_manager: SnapshotManager,
        workflow_file: Path,
        bundles_path: Path,
        temp_dir: Path,
    ) -> None:
        base_manifest = temp_dir / "base-manifest.json"
        base_manifest.write_text(json.dumps({"packages": {}}))
        wheel_ref = (
            "other-dep @ https://example.com/other_dep-1.2.0-py3-none-any.whl#sha256=" + "b" * 64
        )
        pip_output = f"{wheel_ref}\n".encode()
        ok_commit = make_mock_process(returncode=0, stdout=b"abc123\n")
        ok_pip = make_mock_process(returncode=0, stdout=pip_output)

        async def mock_exec(*args: object, **_kwargs: object) -> MagicMock:
            return ok_pip if "freeze" in args else ok_commit

        with patch("asyncio.create_subprocess_exec", new=mock_exec):
            version, _ = await snapshot_manager.create_snapshot(
                "mybundle", workflow_file, base_manifest=base_manifest
            )

        overlay = bundles_path / "mybundle" / version / "requirements.overlay.txt"
        assert overlay.read_text() == f"{wheel_ref}\n"

    async def test_overlay_retains_existing_local_file_reference(
        self,
        snapshot_manager: SnapshotManager,
        workflow_file: Path,
        bundles_path: Path,
        temp_dir: Path,
    ) -> None:
        artifact = temp_dir / "package.whl"
        artifact.write_bytes(b"wheel")
        base_manifest = temp_dir / "base-manifest.json"
        base_manifest.write_text(json.dumps({"packages": {}}))
        file_ref = f"example-package @ {artifact.as_uri()}"
        pip_output = f"{file_ref}\n".encode()
        ok_commit = make_mock_process(returncode=0, stdout=b"abc123\n")
        ok_pip = make_mock_process(returncode=0, stdout=pip_output)

        async def mock_exec(*args: object, **_kwargs: object) -> MagicMock:
            return ok_pip if "freeze" in args else ok_commit

        with patch("asyncio.create_subprocess_exec", new=mock_exec):
            version, _ = await snapshot_manager.create_snapshot(
                "mybundle", workflow_file, base_manifest=base_manifest
            )

        overlay = bundles_path / "mybundle" / version / "requirements.overlay.txt"
        assert overlay.read_text() == f"{file_ref}\n"

    async def test_overlay_direct_reference_overrides_base_even_at_matching_name(
        self,
        snapshot_manager: SnapshotManager,
        workflow_file: Path,
        bundles_path: Path,
        temp_dir: Path,
    ) -> None:
        """A direct reference is an override regardless of the base version, since

        pip would install the referenced artifact rather than the base package.
        """
        base_manifest = temp_dir / "base-manifest.json"
        base_manifest.write_text(json.dumps({"packages": {"ddt": "1.0.0"}}))
        git_ref = "ddt @ git+https://github.com/datadriventests/ddt@" + "c" * 40
        pip_output = f"{git_ref}\n".encode()
        ok_commit = make_mock_process(returncode=0, stdout=b"abc123\n")
        ok_pip = make_mock_process(returncode=0, stdout=pip_output)

        async def mock_exec(*args: object, **_kwargs: object) -> MagicMock:
            return ok_pip if "freeze" in args else ok_commit

        with patch("asyncio.create_subprocess_exec", new=mock_exec):
            version, _ = await snapshot_manager.create_snapshot(
                "mybundle", workflow_file, base_manifest=base_manifest
            )

        overlay = bundles_path / "mybundle" / version / "requirements.overlay.txt"
        assert overlay.read_text() == f"{git_ref}\n"

    async def test_overlay_dropped_lines_are_counted_and_logged(
        self,
        snapshot_manager: SnapshotManager,
        workflow_file: Path,
        bundles_path: Path,
        temp_dir: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        base_manifest = temp_dir / "base-manifest.json"
        base_manifest.write_text(json.dumps({"packages": {}}))
        pip_output = b"torch>=1.2,<2\nnumpy==1.0; python_version<'3.11'\nnot a requirement\n"
        ok_commit = make_mock_process(returncode=0, stdout=b"abc123\n")
        ok_pip = make_mock_process(returncode=0, stdout=pip_output)

        async def mock_exec(*args: object, **_kwargs: object) -> MagicMock:
            return ok_pip if "freeze" in args else ok_commit

        with (
            patch("asyncio.create_subprocess_exec", new=mock_exec),
            caplog.at_level("WARNING", logger="ai_content_service.snapshot"),
        ):
            version, report = await snapshot_manager.create_snapshot(
                "mybundle", workflow_file, base_manifest=base_manifest
            )

        overlay = bundles_path / "mybundle" / version / "requirements.overlay.txt"
        assert overlay.read_text() == ""
        assert report.overlay_dropped_lines == (
            "torch>=1.2,<2",
            "numpy==1.0; python_version<'3.11'",
            "not a requirement",
        )
        warnings = [
            record.msg
            for record in caplog.records
            if isinstance(record.msg, dict)
            and record.msg.get("event") == "snapshot.overlay_lines_dropped"
        ]
        assert len(warnings) == 1
        assert warnings[0]["count"] == 3
        assert warnings[0]["samples"] == [
            "torch>=1.2,<2",
            "numpy==1.0; python_version<'3.11'",
            "not a requirement",
        ]

    async def test_copies_workflow_json(
        self,
        snapshot_manager: SnapshotManager,
        workflow_file: Path,
        bundles_path: Path,
    ) -> None:
        ok = make_mock_process(returncode=0, stdout=b"abc123\n")
        with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=ok)):
            version, _ = await snapshot_manager.create_snapshot("mybundle", workflow_file)

        installed = bundles_path / "mybundle" / version / "workflow.json"
        assert installed.exists()
        assert json.loads(installed.read_text()) == {"nodes": []}

    async def test_sets_current_symlink_for_first_version(
        self,
        snapshot_manager: SnapshotManager,
        workflow_file: Path,
        bundles_path: Path,
    ) -> None:
        ok = make_mock_process(returncode=0, stdout=b"abc123\n")
        with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=ok)):
            version, _ = await snapshot_manager.create_snapshot("mybundle", workflow_file)

        current_link = bundles_path / "mybundle" / "current"
        assert current_link.is_symlink()
        assert current_link.resolve().name == version

    async def test_does_not_overwrite_current_for_subsequent_version(
        self,
        snapshot_manager: SnapshotManager,
        workflow_file: Path,
        bundles_path: Path,
    ) -> None:
        ok = make_mock_process(returncode=0, stdout=b"abc123\n")

        with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=ok)):
            first, _ = await snapshot_manager.create_snapshot("mybundle", workflow_file)

        # Manually simulate a second version being present before creating third
        today = datetime.now(timezone.utc).strftime("%y%m%d")
        second_dir = bundles_path / "mybundle" / f"{today}-02"
        second_dir.mkdir()

        with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=ok)):
            await snapshot_manager.create_snapshot("mybundle", workflow_file)

        # Current symlink should still point to the first version created
        current_link = bundles_path / "mybundle" / "current"
        assert current_link.resolve().name == first

    async def test_snapshot_sets_current_only_when_absent(
        self,
        snapshot_manager: SnapshotManager,
        workflow_file: Path,
        bundles_path: Path,
    ) -> None:
        """Setting current must key off symlink absence, not a directory-count of 1.

        Pre-seed two version directories (no snapshot call, so no `current`
        symlink exists yet) before creating a third via create_snapshot — the
        old `len(iterdir()) == 1` check would skip setting `current` here.
        """
        today = datetime.now(timezone.utc).strftime("%y%m%d")
        bundle_dir = bundles_path / "mybundle"
        (bundle_dir / f"{today}-01").mkdir(parents=True)
        (bundle_dir / f"{today}-02").mkdir(parents=True)

        ok = make_mock_process(returncode=0, stdout=b"abc123\n")
        with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=ok)):
            version, _ = await snapshot_manager.create_snapshot("mybundle", workflow_file)

        current_link = bundle_dir / "current"
        assert current_link.is_symlink()
        assert current_link.resolve().name == version


class TestPipFreeze:
    async def test_pip_freeze_targets_comfyui_python(
        self, snapshot_manager: SnapshotManager, python_executable: Path
    ) -> None:
        ok = make_mock_process(returncode=0, stdout=b"torch==2.1.0\n")
        with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=ok)) as mock_exec:
            result = await snapshot_manager._pip_freeze()

        args = mock_exec.call_args[0]
        assert args[0] == str(python_executable)
        assert args[1] == "-m"
        assert args[2] == "pip"
        assert args[3] == "freeze"
        assert result == "torch==2.1.0\n"

    async def test_pip_freeze_nonzero_raises_snapshot_error(
        self, snapshot_manager: SnapshotManager
    ) -> None:
        failed = make_mock_process(returncode=1, stdout=b"")
        failed.communicate = AsyncMock(return_value=(b"", b"pip is broken"))
        with (
            patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=failed)),
            pytest.raises(SnapshotError, match="pip is broken"),
        ):
            await snapshot_manager._pip_freeze()


class TestScanModels:
    async def test_discovers_hashes_sizes_and_subdirectories(
        self, snapshot_manager: SnapshotManager, comfyui_path: Path
    ) -> None:
        checkpoints = comfyui_path / "models" / "checkpoints"
        nested_lora = comfyui_path / "models" / "loras" / "characters"
        checkpoints.mkdir(parents=True)
        nested_lora.mkdir(parents=True)
        checkpoint = checkpoints / "base.safetensors"
        lora = nested_lora / "hero.gguf"
        checkpoint.write_bytes(b"checkpoint bytes")
        lora.write_bytes(b"lora bytes")

        models = await snapshot_manager._scan_models(None)

        assert [(model.model_type, model.subdirectory) for model in models] == [
            ("loras", "characters"),
            ("checkpoints", None),
        ]
        by_filename = {file.filename: file for model in models for file in model.files}
        assert by_filename["base.safetensors"].size_bytes == len(b"checkpoint bytes")
        assert (
            by_filename["base.safetensors"].sha256
            == hashlib.sha256(b"checkpoint bytes").hexdigest()
        )
        assert by_filename["hero.gguf"].sha256 == hashlib.sha256(b"lora bytes").hexdigest()
        assert all(file.url == "" for file in by_filename.values())

    async def test_skips_non_model_files_cache_git_and_partial_transfers(
        self, snapshot_manager: SnapshotManager, comfyui_path: Path
    ) -> None:
        models_dir = comfyui_path / "models" / "checkpoints"
        (models_dir / ".cache").mkdir(parents=True)
        (models_dir / ".git").mkdir()
        (models_dir / "keep.bin").write_bytes(b"keep")
        (models_dir / "readme.txt").write_text("not a model")
        (models_dir / "incomplete.safetensors.part").write_bytes(b"partial")
        (models_dir / "retry.bin.r2tmp").write_bytes(b"partial")
        (models_dir / ".cache" / "cached.safetensors").write_bytes(b"cache")
        (models_dir / ".git" / "object.ckpt").write_bytes(b"git")

        result = await snapshot_manager._scan_models(None)

        assert len(result) == 1
        assert [file.filename for file in result[0].files] == ["keep.bin"]

    async def test_skips_zero_byte_models_with_a_warning(
        self, snapshot_manager: SnapshotManager, comfyui_path: Path
    ) -> None:
        model = comfyui_path / "models" / "checkpoints" / "empty.safetensors"
        model.parent.mkdir(parents=True)
        model.write_bytes(b"")

        with patch("ai_content_service.snapshot.console.print") as warning:
            result = await snapshot_manager._scan_models(None)

        assert result == []
        assert "zero-byte model file" in str(warning.call_args)

    async def test_warns_about_model_files_in_unknown_top_level_directory(
        self, snapshot_manager: SnapshotManager, comfyui_path: Path
    ) -> None:
        unknown_directory = comfyui_path / "models" / "future_models"
        model = unknown_directory / "nested" / "weight.safetensors"
        model.parent.mkdir(parents=True)
        model.write_bytes(b"model")

        with patch("ai_content_service.snapshot.log.warning") as warning:
            result = await snapshot_manager._scan_models(None)

        assert result == []
        warning.assert_called_once_with(
            "snapshot.unknown_model_dir",
            directory=str(unknown_directory),
            file_count=1,
        )

    async def test_honours_extra_model_paths(
        self, snapshot_manager: SnapshotManager, temp_dir: Path
    ) -> None:
        external = temp_dir / "shared-models"
        external.mkdir()
        model_file = external / "external.pth"
        model_file.write_bytes(b"external model")
        config = temp_dir / "extra_model_paths.yaml"
        config.write_text(f"shared:\n  base_path: {temp_dir}\n  checkpoints: shared-models\n")

        result = await snapshot_manager._scan_models(config)

        assert len(result) == 1
        assert result[0].model_type == "checkpoints"
        assert result[0].files[0].filename == "external.pth"

    async def test_snapshot_writes_url_todos_for_scanned_models(
        self,
        snapshot_manager: SnapshotManager,
        comfyui_path: Path,
        workflow_file: Path,
        bundles_path: Path,
    ) -> None:
        model_file = comfyui_path / "models" / "checkpoints" / "base.safetensors"
        model_file.parent.mkdir(parents=True)
        model_file.write_bytes(b"model")
        ok = make_mock_process(returncode=0, stdout=b"abc123\n")

        with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=ok)):
            version, _ = await snapshot_manager.create_snapshot("mybundle", workflow_file)

        bundle_yaml = (bundles_path / "mybundle" / version / "bundle.yaml").read_text()
        parsed = yaml.safe_load(bundle_yaml)
        file = parsed["models"][0]["files"][0]
        assert file["url"] == ""
        assert file["size_bytes"] == len(b"model")
        assert file["sha256"] == hashlib.sha256(b"model").hexdigest()
        assert "url: ''  # TODO: source URL" in bundle_yaml

    async def test_no_scan_models_bypasses_model_discovery(
        self, snapshot_manager: SnapshotManager, workflow_file: Path
    ) -> None:
        ok = make_mock_process(returncode=0, stdout=b"abc123\n")
        with (
            patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=ok)),
            patch.object(snapshot_manager, "_scan_models", new_callable=AsyncMock) as scan,
        ):
            await snapshot_manager.create_snapshot("mybundle", workflow_file, scan_models=False)

        scan.assert_not_awaited()


class TestSnapshotCarryForward:
    @staticmethod
    def _seed_bundle() -> BundleConfig:
        return BundleConfig.model_validate(
            {
                "metadata": {
                    "name": "seed",
                    "version": "260101-01",
                    "description": "Seed description",
                    "tested": True,
                    "author": "https://example.com/author",
                    "notes": "Keep this note",
                    "tags": ["seed", "qwen"],
                },
                "models": [
                    {
                        "name": "Seed checkpoint",
                        "description": "Checkpoint description",
                        "custom_node_required": "CheckpointLoader",
                        "model_type": "checkpoints",
                        "files": [
                            {
                                "name": "Checkpoint display label",
                                "url": "https://example.com/checkpoint",
                                "filename": "shared.safetensors",
                                "sha256": "a" * 64,
                                "size_bytes": 1,
                            }
                        ],
                    },
                    {
                        "name": "Seed CLIP",
                        "model_type": "clip",
                        "files": [
                            {
                                "name": "CLIP display label",
                                "url": "https://example.com/clip",
                                "filename": "shared.safetensors",
                            }
                        ],
                    },
                    {
                        "name": "Missing VAE",
                        "model_type": "vae",
                        "files": [
                            {
                                "name": "Old VAE",
                                "url": "https://example.com/old-vae",
                                "filename": "old.safetensors",
                            }
                        ],
                    },
                ],
                "hardware": {"min_disk_gb": 80, "gpu_whitelist": ["H100"]},
                "generation": {"defaults": {"steps": 30, "scheduler": "normal"}},
                "readiness_marker": {"node_class": "QwenLoader"},
            }
        )

    async def test_carries_seed_intent_and_reports_model_differences(
        self,
        snapshot_manager: SnapshotManager,
        comfyui_path: Path,
        workflow_file: Path,
        bundles_path: Path,
    ) -> None:
        for model_type, content in (
            ("checkpoints", b"checkpoint bytes"),
            ("clip", b"clip bytes"),
            ("loras", b"unseeded bytes"),
        ):
            path = comfyui_path / "models" / model_type / "shared.safetensors"
            if model_type == "loras":
                path = path.with_name("experiment.safetensors")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)

        ok = make_mock_process(returncode=0, stdout=b"abc123\n")
        with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=ok)):
            version, report = await snapshot_manager.create_snapshot(
                "snapshot", workflow_file, carry_from=self._seed_bundle()
            )

        assert report.urls_carried == ("checkpoints/shared.safetensors", "clip/shared.safetensors")
        assert report.files_without_url == ("loras/experiment.safetensors",)
        assert report.seed_files_unmatched == ("vae/old.safetensors",)
        assert report.blocks_carried == ("hardware", "generation", "readiness_marker")

        config = yaml.safe_load((bundles_path / "snapshot" / version / "bundle.yaml").read_text())
        by_target = {
            f"{model['model_type']}/{model['subdirectory']}"
            if model.get("subdirectory")
            else model["model_type"]: model
            for model in config["models"]
        }
        checkpoint = by_target["checkpoints"]
        clip = by_target["clip"]
        assert checkpoint["description"] == "Checkpoint description"
        assert checkpoint["custom_node_required"] == "CheckpointLoader"
        assert checkpoint["files"][0]["name"] == "Checkpoint display label"
        assert checkpoint["files"][0]["url"] == "https://example.com/checkpoint"
        assert clip["files"][0]["name"] == "CLIP display label"
        assert clip["files"][0]["url"] == "https://example.com/clip"
        assert checkpoint["files"][0]["sha256"] == hashlib.sha256(b"checkpoint bytes").hexdigest()
        assert checkpoint["files"][0]["size_bytes"] == len(b"checkpoint bytes")
        assert config["metadata"]["description"] == "Seed description"
        assert config["metadata"]["author"] == "https://example.com/author"
        assert config["metadata"]["notes"] == "Keep this note"
        assert config["metadata"]["tags"] == ["seed", "qwen"]
        assert config["metadata"]["tested"] is False
        assert config["metadata"]["version"] == version
        assert config["hardware"] == {"gpu_whitelist": ["H100"], "min_disk_gb": 80}
        assert config["generation"] == {"defaults": {"scheduler": "normal", "steps": 30}}
        assert config["readiness_marker"] == {"node_class": "QwenLoader"}

    async def test_overlay_base_mismatch_is_logged_when_hardware_disagrees(
        self,
        snapshot_manager: SnapshotManager,
        workflow_file: Path,
        temp_dir: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        base_manifest = temp_dir / "base-manifest.json"
        base_manifest.write_text(json.dumps({"packages": {}, "base_image": "vastai/comfy:v0.34.0"}))
        seed = BundleConfig.model_validate(
            {
                "metadata": {"name": "seed", "version": "260101-01"},
                "hardware": {"base_image": "vastai/comfy:v0.32.0"},
            }
        )
        ok = make_mock_process(returncode=0, stdout=b"abc123\n")

        with (
            patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=ok)),
            caplog.at_level("WARNING", logger="ai_content_service.snapshot"),
        ):
            await snapshot_manager.create_snapshot(
                "snapshot", workflow_file, carry_from=seed, base_manifest=base_manifest
            )

        warnings = [
            record.msg
            for record in caplog.records
            if isinstance(record.msg, dict)
            and record.msg.get("event") == "snapshot.overlay_base_mismatch"
        ]
        assert len(warnings) == 1
        assert warnings[0]["manifest_base_image"] == "vastai/comfy:v0.34.0"
        assert warnings[0]["hardware_base_image"] == "vastai/comfy:v0.32.0"

    async def test_non_pristine_base_manifest_is_reported_but_still_used(
        self,
        snapshot_manager: SnapshotManager,
        workflow_file: Path,
        temp_dir: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        base_manifest = temp_dir / "base-manifest.json"
        base_manifest.write_text(
            json.dumps({"captured_before_install": False, "packages": {"torch": "2.1.0"}})
        )
        ok_commit = make_mock_process(returncode=0, stdout=b"abc123\n")
        ok_pip = make_mock_process(returncode=0, stdout=b"torch==2.1.0\nnumpy==1.0\n")

        async def mock_exec(*args: object, **_kwargs: object) -> MagicMock:
            return ok_pip if "freeze" in args else ok_commit

        with (
            patch("asyncio.create_subprocess_exec", new=mock_exec),
            caplog.at_level("WARNING", logger="ai_content_service.snapshot"),
        ):
            await snapshot_manager.create_snapshot(
                "snapshot", workflow_file, base_manifest=base_manifest
            )

        warnings = [
            record.msg
            for record in caplog.records
            if isinstance(record.msg, dict)
            and record.msg.get("event") == "snapshot.base_manifest_not_pristine"
        ]
        assert len(warnings) == 1
        assert warnings[0]["base_manifest"] == str(base_manifest)

    async def test_overlay_base_mismatch_not_logged_when_manifest_base_image_absent(
        self,
        snapshot_manager: SnapshotManager,
        workflow_file: Path,
        temp_dir: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A null manifest base_image is normal (env-delegation C3) and must not warn."""
        base_manifest = temp_dir / "base-manifest.json"
        base_manifest.write_text(json.dumps({"packages": {}, "base_image": None}))
        seed = BundleConfig.model_validate(
            {
                "metadata": {"name": "seed", "version": "260101-01"},
                "hardware": {"base_image": "vastai/comfy:v0.32.0"},
            }
        )
        ok = make_mock_process(returncode=0, stdout=b"abc123\n")

        with (
            patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=ok)),
            caplog.at_level("WARNING", logger="ai_content_service.snapshot"),
        ):
            await snapshot_manager.create_snapshot(
                "snapshot", workflow_file, carry_from=seed, base_manifest=base_manifest
            )

        assert not [
            record.msg
            for record in caplog.records
            if isinstance(record.msg, dict)
            and record.msg.get("event") == "snapshot.overlay_base_mismatch"
        ]

    async def test_overlay_base_match_not_logged(
        self,
        snapshot_manager: SnapshotManager,
        workflow_file: Path,
        temp_dir: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        base_manifest = temp_dir / "base-manifest.json"
        base_manifest.write_text(json.dumps({"packages": {}, "base_image": "vastai/comfy:v0.32.0"}))
        seed = BundleConfig.model_validate(
            {
                "metadata": {"name": "seed", "version": "260101-01"},
                "hardware": {"base_image": "vastai/comfy:v0.32.0"},
            }
        )
        ok = make_mock_process(returncode=0, stdout=b"abc123\n")

        with (
            patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=ok)),
            caplog.at_level("WARNING", logger="ai_content_service.snapshot"),
        ):
            await snapshot_manager.create_snapshot(
                "snapshot", workflow_file, carry_from=seed, base_manifest=base_manifest
            )

        assert not [
            record.msg
            for record in caplog.records
            if isinstance(record.msg, dict)
            and record.msg.get("event") == "snapshot.overlay_base_mismatch"
        ]

    async def test_explicit_description_wins_and_blank_seed_url_stays_a_todo(
        self,
        snapshot_manager: SnapshotManager,
        comfyui_path: Path,
        workflow_file: Path,
        bundles_path: Path,
    ) -> None:
        model = comfyui_path / "models" / "checkpoints" / "model.safetensors"
        model.parent.mkdir(parents=True)
        model.write_bytes(b"model")
        seed = self._seed_bundle().model_copy(
            update={
                "models": [
                    ModelConfig(
                        name="Seed checkpoint",
                        model_type="checkpoints",
                        files=[
                            ModelFileConfig(
                                name="Seed label",
                                url="",
                                filename="model.safetensors",
                            )
                        ],
                    )
                ]
            }
        )
        ok = make_mock_process(returncode=0, stdout=b"abc123\n")

        with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=ok)):
            version, report = await snapshot_manager.create_snapshot(
                "snapshot",
                workflow_file,
                description="Explicit description",
                carry_from=seed,
            )

        bundle_yaml = (bundles_path / "snapshot" / version / "bundle.yaml").read_text()
        config = yaml.safe_load(bundle_yaml)
        assert report.urls_carried == ()
        assert report.files_without_url == ()
        assert config["metadata"]["description"] == "Explicit description"
        assert config["models"][0]["files"][0]["name"] == "Seed label"
        assert config["models"][0]["files"][0]["url"] == ""
        assert "url: ''  # TODO: source URL" in bundle_yaml


class TestExtraModelPathCompatibility:
    def test_parses_native_sections_multiline_paths_and_expansions(
        self, snapshot_manager: SnapshotManager, temp_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        workspace = temp_dir / "workspace"
        home = temp_dir / "home"
        config_dir = temp_dir / "config"
        config_dir.mkdir()
        monkeypatch.setenv("SNAPSHOT_WORKSPACE", str(workspace))
        monkeypatch.setenv("HOME", str(home))
        config = config_dir / "extra_model_paths.yaml"
        config.write_text(
            "comfyui:\n"
            "  base_path: $SNAPSHOT_WORKSPACE\n"
            "  is_default: true\n"
            "  diffusion_models: |\n"
            "    models/diffusion_models\n"
            "\n"
            "    models/unet\n"
            "shared:\n"
            "  base_path: ./shared\n"
            "  checkpoints: |\n"
            "    checkpoints\n"
            "    alternate-checkpoints\n"
            "standalone:\n"
            "  loras: ~/loras\n"
        )

        roots = snapshot_manager._extra_model_roots(config)

        assert [(root.model_type.value, root.path, root.is_default) for root in roots] == [
            ("diffusion_models", workspace / "models" / "diffusion_models", True),
            ("diffusion_models", workspace / "models" / "unet", True),
            ("checkpoints", config_dir / "shared" / "checkpoints", False),
            ("checkpoints", config_dir / "shared" / "alternate-checkpoints", False),
            ("loras", home / "loras", False),
        ]

    def test_rejects_malformed_sections_and_model_values(
        self, snapshot_manager: SnapshotManager, temp_dir: Path
    ) -> None:
        config = temp_dir / "extra_model_paths.yaml"
        config.write_text("broken: not-a-section\n")
        with pytest.raises(SnapshotError, match="broken"):
            snapshot_manager._extra_model_roots(config)

        config.write_text("shared:\n  checkpoints: [not, a, string]\n")
        with pytest.raises(SnapshotError, match="checkpoints"):
            snapshot_manager._extra_model_roots(config)

    def test_does_not_treat_nonstandard_wrapper_as_a_root_section(
        self, snapshot_manager: SnapshotManager, temp_dir: Path
    ) -> None:
        config = temp_dir / "extra_model_paths.yaml"
        config.write_text("extra_model_paths:\n  shared:\n    checkpoints: elsewhere\n")

        assert snapshot_manager._extra_model_roots(config) == []

    async def test_default_extra_root_takes_comfyui_precedence_and_warns(
        self,
        snapshot_manager: SnapshotManager,
        comfyui_path: Path,
        temp_dir: Path,
    ) -> None:
        built_in = comfyui_path / "models" / "checkpoints"
        default_root = temp_dir / "default"
        ordinary_root = temp_dir / "ordinary"
        for directory, content in (
            (built_in, b"built-in"),
            (default_root, b"default"),
            (ordinary_root, b"ordinary"),
        ):
            directory.mkdir(parents=True)
            (directory / "duplicate.safetensors").write_bytes(content)
        config = temp_dir / "extra_model_paths.yaml"
        config.write_text(
            "ordinary:\n"
            f"  checkpoints: {ordinary_root}\n"
            "preferred:\n"
            "  is_default: true\n"
            f"  checkpoints: {default_root}\n"
        )

        with patch("ai_content_service.snapshot.console.print") as warning:
            models = await snapshot_manager._scan_models(config)

        file = models[0].files[0]
        assert file.sha256 == hashlib.sha256(b"default").hexdigest()
        assert file.size_bytes == len(b"default")
        warning_output = "\n".join(str(call) for call in warning.call_args_list)
        assert "duplicate.safetensors" in warning_output
        assert str(default_root) in warning_output
        assert str(ordinary_root) in warning_output


class TestSafeModelTraversal:
    @staticmethod
    def _symlink_or_skip(link: Path, target: Path) -> None:
        try:
            link.symlink_to(target, target_is_directory=True)
        except (NotImplementedError, OSError) as e:
            pytest.skip(f"directory symlinks are unavailable: {e}")

    async def test_follows_symlinked_directories_once_and_prunes_hidden_dirs(
        self, snapshot_manager: SnapshotManager, comfyui_path: Path, temp_dir: Path
    ) -> None:
        root = comfyui_path / "models" / "checkpoints"
        root.mkdir(parents=True)
        shared = temp_dir / "shared"
        shared.mkdir()
        (shared / "model.safetensors").write_bytes(b"linked")
        (shared / ".cache").mkdir()
        (shared / ".cache" / "ignored.safetensors").write_bytes(b"ignored")
        (shared / ".git").mkdir()
        (shared / ".git" / "ignored.ckpt").write_bytes(b"ignored")
        self._symlink_or_skip(root / "a-link", shared)
        self._symlink_or_skip(root / "b-alias", shared)
        self._symlink_or_skip(shared / "cycle", shared)

        models = await snapshot_manager._scan_models(None)

        assert len(models) == 1
        assert models[0].subdirectory == "a-link"
        assert [file.filename for file in models[0].files] == ["model.safetensors"]

    async def test_directory_enumeration_and_hash_failures_are_not_suppressed(
        self, snapshot_manager: SnapshotManager, comfyui_path: Path
    ) -> None:
        checkpoints = comfyui_path / "models" / "checkpoints"
        checkpoints.mkdir(parents=True)
        (checkpoints / "model.safetensors").write_bytes(b"content")

        with (
            patch("ai_content_service.snapshot.os.scandir", side_effect=PermissionError("denied")),
            pytest.raises(SnapshotError, match="checkpoints"),
        ):
            await snapshot_manager._scan_models(None)

        with (
            patch(
                "ai_content_service.snapshot._hash_model_file",
                side_effect=PermissionError("denied"),
            ),
            pytest.raises(SnapshotError, match=re.escape("model.safetensors")),
        ):
            await snapshot_manager._scan_models(None)


class TestSnapshotHashing:
    def test_hash_result_uses_one_stable_open_file(self, temp_dir: Path) -> None:
        model = temp_dir / "stable.safetensors"
        model.write_bytes(b"stable bytes")

        result = _hash_model_file(model, lambda _count: None)

        assert result.bytes_read == len(b"stable bytes")
        assert result.initial_size == result.final_size == result.bytes_read
        assert result.sha256 == hashlib.sha256(b"stable bytes").hexdigest()

    def test_hash_rejects_file_mutation(self, temp_dir: Path) -> None:
        model = temp_dir / "changed.safetensors"
        model.write_bytes(b"content")
        before = model.stat()
        changed_values = list(before)
        changed_values[6] += 1
        changed = os.stat_result(changed_values)

        with (
            patch("ai_content_service.snapshot.os.fstat", side_effect=[before, changed]),
            pytest.raises(SnapshotError, match=re.escape("changed.safetensors")),
        ):
            _hash_model_file(model, lambda _count: None)

    async def test_hashing_is_bounded_and_progress_stays_on_event_loop_thread(
        self, snapshot_manager: SnapshotManager, comfyui_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checkpoints = comfyui_path / "models" / "checkpoints"
        checkpoints.mkdir(parents=True)
        for number in range(5):
            (checkpoints / f"model-{number}.safetensors").write_bytes(b"x")

        active = 0
        maximum_active = 0
        starts = 0
        lock = threading.Lock()
        barrier = threading.Barrier(2)
        progress_threads: set[int] = set()
        loop_thread = threading.get_ident()

        class RecordingProgress:
            def __init__(self, *_args: object, **_kwargs: object) -> None:
                self.total: int | None = None
                self.completed = 0

            def __enter__(self) -> "RecordingProgress":
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def add_task(self, *_args: object, total: int) -> int:
                self.total = total
                return 1

            def advance(self, _task_id: int, delta: int) -> None:
                progress_threads.add(threading.get_ident())
                self.completed += delta

            def update(self, _task_id: int, *, total: int, completed: int) -> None:
                progress_threads.add(threading.get_ident())
                self.total = total
                self.completed = completed

        def fake_hash(_path: Path, on_chunk: object) -> _HashResult:
            nonlocal active, maximum_active, starts
            with lock:
                active += 1
                starts += 1
                maximum_active = max(maximum_active, active)
                wait_for_peer = starts <= 2
            try:
                if wait_for_peer:
                    barrier.wait()
                assert callable(on_chunk)
                on_chunk(1)
                return _HashResult("a" * 64, 1, 1, 1)
            finally:
                with lock:
                    active -= 1

        monkeypatch.setattr("ai_content_service.snapshot._MAX_CONCURRENT_HASHES", 2)
        with (
            patch("ai_content_service.snapshot.Progress", RecordingProgress),
            patch("ai_content_service.snapshot._hash_model_file", side_effect=fake_hash),
        ):
            models = await snapshot_manager._scan_models(None)

        assert maximum_active == 2
        assert maximum_active <= 2
        assert progress_threads == {loop_thread}
        assert [file.filename for file in models[0].files] == [
            "model-0.safetensors",
            "model-1.safetensors",
            "model-2.safetensors",
            "model-3.safetensors",
            "model-4.safetensors",
        ]

    async def test_hash_result_size_is_authoritative_over_candidate_estimate(
        self, snapshot_manager: SnapshotManager, comfyui_path: Path
    ) -> None:
        model = comfyui_path / "models" / "checkpoints" / "model.safetensors"
        model.parent.mkdir(parents=True)
        model.write_bytes(b"old")
        result = _HashResult("b" * 64, bytes_read=7, initial_size=7, final_size=7)

        with patch("ai_content_service.snapshot._hash_model_file", return_value=result):
            models = await snapshot_manager._scan_models(None)

        assert models[0].files[0].size_bytes == 7


class TestSnapshotYamlAnnotations:
    @staticmethod
    def _bundle_with_urls(*urls: str) -> BundleConfig:
        return BundleConfig(
            metadata=BundleMetadata(name="snapshot", version="260101-01", description="url: ''"),
            models=[
                ModelConfig(
                    name="checkpoints",
                    model_type="checkpoints",
                    files=[
                        ModelFileConfig(
                            name=f"model-{index}",
                            filename=f"model-{index}.safetensors",
                            url=url,
                        )
                        for index, url in enumerate(urls)
                    ],
                )
            ],
        )

    def test_annotates_only_empty_model_urls_and_round_trips(self) -> None:
        rendered = _render_bundle_yaml(self._bundle_with_urls("", "https://example.com/model"))

        parsed = yaml.safe_load(rendered)
        assert rendered.count("# TODO: source URL") == 1
        assert parsed["metadata"]["description"] == "url: ''"
        assert [file["url"] for file in parsed["models"][0]["files"]] == [
            "",
            "https://example.com/model",
        ]

    def test_annotation_count_mismatch_fails_loudly(self) -> None:
        class NoReplacementPattern:
            @staticmethod
            def subn(_value: object, text: str) -> tuple[str, int]:
                return text, 0

        with (
            patch("ai_content_service.snapshot.re.compile", return_value=NoReplacementPattern()),
            pytest.raises(SnapshotError) as error,
        ):
            _render_bundle_yaml(self._bundle_with_urls(""))

        assert "placeholders" in str(error.value)

    def test_omits_optional_readiness_marker_from_snapshot_yaml(self) -> None:
        rendered = _render_bundle_yaml(self._bundle_with_urls("https://example.com/model"))
        assert "readiness_marker" not in rendered

    def test_multiline_comment_is_fully_commented_and_round_trips(self) -> None:
        config = self._bundle_with_urls("https://example.com/model")

        rendered = _render_bundle_yaml(
            config, workflow_comments=("first line\nsecond line\nthird line",)
        )

        assert "# first line\n# second line\n# third line\n" in rendered
        assert yaml.safe_load(rendered) == config.model_dump(
            mode="json", by_alias=True, exclude_none=True
        )

    def test_validation_error_comment_is_single_line_ascii_and_bounded(self) -> None:
        with pytest.raises(ValidationError) as error:
            WorkflowNodeConfig.model_validate({"id": "3", "class": " ", "inputs": {}})

        comment = _normalize_workflow_comment(error.value)

        assert "\n" not in comment
        assert comment.isascii()
        assert len(comment) <= 200

    def test_multiline_fallback_annotation_cannot_escape_its_yaml_line(self) -> None:
        config = BundleConfig(
            metadata=BundleMetadata(name="snapshot", version="260101-01"),
            workflow_file="workflow.json",
            workflow_api_file="workflow.api.json",
        )
        comment = "TODO: export via Graph -> Export (API) first line\nsecond line"

        rendered = _render_bundle_yaml(config, workflow_comments=(comment,))

        assert "workflow_api_file: workflow.api.json  # TODO: export" in rendered
        assert "# second line\n" in rendered
        assert yaml.safe_load(rendered) == config.model_dump(
            mode="json", by_alias=True, exclude_none=True
        )


class TestScanCustomNodes:
    @staticmethod
    def _events(caplog: pytest.LogCaptureFixture, event: str) -> list[dict[str, object]]:
        return [
            record.msg
            for record in caplog.records
            if isinstance(record.msg, dict) and record.msg.get("event") == event
        ]

    async def test_returns_empty_when_no_custom_nodes_dir(
        self, snapshot_manager: SnapshotManager
    ) -> None:
        result = await snapshot_manager._scan_custom_nodes()
        assert result == []

    async def test_skips_non_git_directories(
        self, snapshot_manager: SnapshotManager, comfyui_path: Path
    ) -> None:
        custom_nodes = comfyui_path / "custom_nodes"
        custom_nodes.mkdir()
        (custom_nodes / "not_a_repo").mkdir()  # no .git subdir

        result = await snapshot_manager._scan_custom_nodes()
        assert result == []

    async def test_skips_hidden_directories(
        self, snapshot_manager: SnapshotManager, comfyui_path: Path
    ) -> None:
        custom_nodes = comfyui_path / "custom_nodes"
        custom_nodes.mkdir()
        hidden = custom_nodes / ".hidden_repo"
        hidden.mkdir()
        (hidden / ".git").mkdir()

        result = await snapshot_manager._scan_custom_nodes()
        assert result == []

    async def test_collects_git_repos_with_remote(
        self, snapshot_manager: SnapshotManager, comfyui_path: Path
    ) -> None:
        custom_nodes = comfyui_path / "custom_nodes"
        custom_nodes.mkdir()
        node_dir = custom_nodes / "MyNode"
        node_dir.mkdir()
        (node_dir / ".git").mkdir()

        ok_root = make_mock_process(returncode=0, stdout=f"{node_dir}\n".encode())
        ok_remote = make_mock_process(returncode=0, stdout=b"https://github.com/test/node\n")
        ok_commit = make_mock_process(returncode=0, stdout=b"deadbeef\n")

        async def mock_exec(*args: object, **_kwargs: object) -> MagicMock:
            if "--show-toplevel" in args:
                return ok_root
            return ok_remote if "remote" in args else ok_commit

        with patch("asyncio.create_subprocess_exec", new=mock_exec):
            result = await snapshot_manager._scan_custom_nodes()

        assert len(result) == 1
        assert result[0].name == "MyNode"
        assert result[0].git_url == "https://github.com/test/node"
        assert result[0].commit_sha == "deadbeef"

    async def test_registry_directory_never_uses_ancestor_git_metadata(
        self,
        snapshot_manager: SnapshotManager,
        comfyui_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        custom_nodes = comfyui_path / "custom_nodes"
        custom_nodes.mkdir()
        (comfyui_path / ".git").mkdir()
        registry_node = custom_nodes / "comfyui-kjnodes"
        registry_node.mkdir()
        (registry_node / ".github").mkdir()
        (registry_node / ".gitignore").write_text("*.pyc\n")
        (registry_node / "pyproject.toml").write_text(
            "[project]\n"
            'name = "comfyui-kjnodes"\n'
            'version = "1.5.0"\n'
            "[project.urls]\n"
            'Repository = "https://github.com/kijai/ComfyUI-KJNodes"\n'
        )
        git = AsyncMock()

        with (
            patch.object(snapshot_manager, "_git", new=git),
            patch.object(
                snapshot_manager, "_resolve_registry_pin", new=AsyncMock(return_value=None)
            ),
            patch("ai_content_service.snapshot.console.print") as printed,
            caplog.at_level("WARNING", logger="ai_content_service.snapshot"),
        ):
            result = await snapshot_manager._scan_custom_nodes()

        assert result == []
        git.assert_not_awaited()
        unsupported = self._events(caplog, "snapshot.custom_node_unsupported_source")
        assert len(unsupported) == 1
        assert unsupported[0]["directory"] == "comfyui-kjnodes"
        assert unsupported[0]["project_name"] == "comfyui-kjnodes"
        assert unsupported[0]["version"] == "1.5.0"
        assert unsupported[0]["repository"] == "https://github.com/kijai/ComfyUI-KJNodes"
        skipped = self._events(caplog, "snapshot.custom_node_skipped")
        assert skipped[0]["reason"] == "no_git_metadata"
        console_output = "\n".join(str(call.args[0]) for call in printed.call_args_list)
        assert "rm -rf custom_nodes/comfyui-kjnodes" in console_output
        assert "git clone https://github.com/kijai/ComfyUI-KJNodes" in console_output

    async def test_registry_install_captures_a_tag_derived_pin(
        self,
        snapshot_manager: SnapshotManager,
        comfyui_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        node_dir = comfyui_path / "custom_nodes" / "comfyui-kjnodes"
        node_dir.mkdir(parents=True)
        (node_dir / "pyproject.toml").write_text(
            "[project]\n"
            'name = "comfyui-kjnodes"\n'
            'version = "1.5.0"\n'
            "[project.urls]\n"
            'Repository = "https://github.com/kijai/ComfyUI-KJNodes"\n'
        )
        response = MagicMock(status_code=200)
        response.json.return_value = {"object": {"type": "commit", "sha": "tag-sha"}}
        client = MagicMock()
        client.get = AsyncMock(return_value=response)

        with (
            patch(
                "ai_content_service.snapshot.httpx.AsyncClient",
                return_value=_make_async_cm(client),
            ),
            caplog.at_level("INFO", logger="ai_content_service.snapshot"),
        ):
            result = await snapshot_manager._scan_custom_nodes()

        assert result[0].name == "comfyui-kjnodes"
        assert result[0].commit_sha == "tag-sha"
        assert snapshot_manager._last_custom_node_scan.skipped == ()
        assert "/git/ref/tags/v1.5.0" in str(client.get.await_args.args[0])
        events = self._events(caplog, "snapshot.custom_node_pinned_from_registry")
        assert events[0]["commit_sha"] == "tag-sha"
        assert "tag-derived" in str(events[0]["pin_source"])

    async def test_registry_tag_resolver_dereferences_annotated_tags(
        self, snapshot_manager: SnapshotManager
    ) -> None:
        tag_ref = MagicMock(status_code=200)
        tag_ref.json.return_value = {"object": {"type": "tag", "sha": "annotated-sha"}}
        annotated_tag = MagicMock(status_code=200)
        annotated_tag.json.return_value = {"object": {"type": "commit", "sha": "commit-sha"}}
        client = MagicMock()
        client.get = AsyncMock(side_effect=[tag_ref, annotated_tag])

        with patch(
            "ai_content_service.snapshot.httpx.AsyncClient", return_value=_make_async_cm(client)
        ):
            result = await snapshot_manager._resolve_registry_pin(
                "https://github.com/example/node", "1.5.0"
            )

        assert result == "commit-sha"
        assert "/git/tags/annotated-sha" in str(client.get.await_args_list[1].args[0])

    async def test_registry_pin_attaches_bearer_token_when_configured(
        self, comfyui_path: Path, bundles_path: Path, python_executable: Path
    ) -> None:
        manager = SnapshotManager(
            comfyui_path,
            bundles_path,
            python_executable=python_executable,
            github_token="ghp_configured",
        )
        response = MagicMock(status_code=200)
        response.json.return_value = {"object": {"type": "commit", "sha": "sha"}}
        client = MagicMock()
        client.get = AsyncMock(return_value=response)

        with patch(
            "ai_content_service.snapshot.httpx.AsyncClient", return_value=_make_async_cm(client)
        ):
            await manager._resolve_registry_pin("https://github.com/example/node", "1.5.0")

        assert client.get.await_args.kwargs["headers"]["Authorization"] == "Bearer ghp_configured"

    async def test_registry_pin_omits_auth_header_when_token_unset(
        self, snapshot_manager: SnapshotManager
    ) -> None:
        response = MagicMock(status_code=200)
        response.json.return_value = {"object": {"type": "commit", "sha": "sha"}}
        client = MagicMock()
        client.get = AsyncMock(return_value=response)

        with patch(
            "ai_content_service.snapshot.httpx.AsyncClient", return_value=_make_async_cm(client)
        ):
            await snapshot_manager._resolve_registry_pin("https://github.com/example/node", "1.5.0")

        assert "Authorization" not in client.get.await_args.kwargs["headers"]
        assert client.get.await_count == 1

    async def test_registry_pin_ignores_raw_environment_token(
        self,
        comfyui_path: Path,
        bundles_path: Path,
        python_executable: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("ACS_GITHUB_TOKEN", "env-token-must-be-ignored")
        manager = SnapshotManager(
            comfyui_path, bundles_path, python_executable=python_executable, github_token=None
        )
        response = MagicMock(status_code=200)
        response.json.return_value = {"object": {"type": "commit", "sha": "sha"}}
        client = MagicMock()
        client.get = AsyncMock(return_value=response)

        with patch(
            "ai_content_service.snapshot.httpx.AsyncClient", return_value=_make_async_cm(client)
        ):
            await manager._resolve_registry_pin("https://github.com/example/node", "1.5.0")

        assert "Authorization" not in client.get.await_args.kwargs["headers"]

    async def test_registry_pin_miss_logs_repository_not_github(
        self, snapshot_manager: SnapshotManager, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("INFO", logger="ai_content_service.snapshot"):
            result = await snapshot_manager._resolve_registry_pin("https://gitlab.com/x/y", "1.0.0")

        assert result is None
        events = self._events(caplog, "snapshot.registry_pin_miss")
        assert events[0]["reason"] == "repository_not_github"
        assert events[0]["repository"] == "https://gitlab.com/x/y"

    async def test_registry_pin_miss_logs_repository_unparseable(
        self, snapshot_manager: SnapshotManager, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("INFO", logger="ai_content_service.snapshot"):
            result = await snapshot_manager._resolve_registry_pin(
                "https://github.com/onlyowner", "1.0.0"
            )

        assert result is None
        events = self._events(caplog, "snapshot.registry_pin_miss")
        assert events[0]["reason"] == "repository_unparseable"

    async def test_registry_pin_miss_logs_version_missing(
        self, snapshot_manager: SnapshotManager, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("INFO", logger="ai_content_service.snapshot"):
            result = await snapshot_manager._resolve_registry_pin(
                "https://github.com/kijai/ComfyUI-KJNodes", None
            )

        assert result is None
        events = self._events(caplog, "snapshot.registry_pin_miss")
        assert events[0]["reason"] == "version_missing"

    async def test_registry_pin_miss_logs_tag_not_found(
        self, snapshot_manager: SnapshotManager, caplog: pytest.LogCaptureFixture
    ) -> None:
        response = MagicMock(status_code=404, headers={})
        client = MagicMock()
        client.get = AsyncMock(return_value=response)

        with (
            patch(
                "ai_content_service.snapshot.httpx.AsyncClient", return_value=_make_async_cm(client)
            ),
            caplog.at_level("INFO", logger="ai_content_service.snapshot"),
        ):
            result = await snapshot_manager._resolve_registry_pin(
                "https://github.com/kijai/ComfyUI-KJNodes", "1.5.0"
            )

        assert result is None
        events = self._events(caplog, "snapshot.registry_pin_miss")
        assert events[0]["reason"] == "tag_not_found"
        assert events[0]["status_code"] == 404
        assert events[0]["tags"] == ("v1.5.0", "1.5.0")

    async def test_registry_pin_miss_logs_rate_limited(
        self, snapshot_manager: SnapshotManager, caplog: pytest.LogCaptureFixture
    ) -> None:
        response = MagicMock(status_code=403, headers={"x-ratelimit-remaining": "0"})
        response.json.return_value = {"message": "API rate limit exceeded"}
        client = MagicMock()
        client.get = AsyncMock(return_value=response)

        with (
            patch(
                "ai_content_service.snapshot.httpx.AsyncClient", return_value=_make_async_cm(client)
            ),
            caplog.at_level("INFO", logger="ai_content_service.snapshot"),
        ):
            result = await snapshot_manager._resolve_registry_pin(
                "https://github.com/kijai/ComfyUI-KJNodes", "1.5.0"
            )

        assert result is None
        events = self._events(caplog, "snapshot.registry_pin_miss")
        assert events[0]["reason"] == "rate_limited"
        assert events[0]["status_code"] == 403

    async def test_registry_pin_miss_logs_http_error(
        self, snapshot_manager: SnapshotManager, caplog: pytest.LogCaptureFixture
    ) -> None:
        response = MagicMock(status_code=500, headers={})
        client = MagicMock()
        client.get = AsyncMock(return_value=response)

        with (
            patch(
                "ai_content_service.snapshot.httpx.AsyncClient", return_value=_make_async_cm(client)
            ),
            caplog.at_level("INFO", logger="ai_content_service.snapshot"),
        ):
            result = await snapshot_manager._resolve_registry_pin(
                "https://github.com/kijai/ComfyUI-KJNodes", "1.5.0"
            )

        assert result is None
        events = self._events(caplog, "snapshot.registry_pin_miss")
        assert events[0]["reason"] == "http_error"
        assert events[0]["status_code"] == 500

    async def test_registry_pin_miss_logs_network_error(
        self, snapshot_manager: SnapshotManager, caplog: pytest.LogCaptureFixture
    ) -> None:
        client = MagicMock()
        client.get = AsyncMock(side_effect=httpx.ConnectError("boom"))

        with (
            patch(
                "ai_content_service.snapshot.httpx.AsyncClient", return_value=_make_async_cm(client)
            ),
            caplog.at_level("INFO", logger="ai_content_service.snapshot"),
        ):
            result = await snapshot_manager._resolve_registry_pin(
                "https://github.com/kijai/ComfyUI-KJNodes", "1.5.0"
            )

        assert result is None
        events = self._events(caplog, "snapshot.registry_pin_miss")
        assert events[0]["reason"] == "network_error"
        assert events[0]["exception"] == "ConnectError"

    async def test_skipped_registry_node_carries_seed_pin(
        self,
        snapshot_manager: SnapshotManager,
        comfyui_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        node_dir = comfyui_path / "custom_nodes" / "comfyui-kjnodes"
        node_dir.mkdir(parents=True)
        (node_dir / "pyproject.toml").write_text(
            "[project]\nversion = " + '"1.5.0"\n'
            "[project.urls]\n"
            'Repository = "https://github.com/kijai/ComfyUI-KJNodes"\n'
        )
        seed_node = CustomNodeConfig(
            name="ComfyUI-KJNodes",
            git_url="https://github.com/kijai/ComfyUI-KJNodes",
            commit_sha="3f20054214fec9f9234fd3841ae6f1e4287948f6",
        )
        seed = BundleConfig(
            metadata=BundleMetadata(name="demo", version="260101-01"),
            custom_nodes=[seed_node],
            workflow_file="workflow.json",
        )

        with (
            patch.object(
                snapshot_manager, "_resolve_registry_pin", new=AsyncMock(return_value=None)
            ),
            caplog.at_level("INFO", logger="ai_content_service.snapshot"),
        ):
            result = await snapshot_manager._scan_custom_nodes(seed)

        assert result == [seed_node]
        assert snapshot_manager._last_custom_node_scan.captured == ()
        assert snapshot_manager._last_custom_node_scan.carried == ("ComfyUI-KJNodes",)
        events = self._events(caplog, "snapshot.custom_node_carried")
        assert events[0]["commit_sha"] == seed_node.commit_sha
        assert events[0]["reason"] == "no_git_metadata"

    async def test_clean_live_scan_wins_over_seed_pin(
        self, snapshot_manager: SnapshotManager, comfyui_path: Path
    ) -> None:
        node_dir = comfyui_path / "custom_nodes" / "node"
        node_dir.mkdir(parents=True)
        (node_dir / ".git").mkdir()
        seed = BundleConfig(
            metadata=BundleMetadata(name="demo", version="260101-01"),
            custom_nodes=[
                CustomNodeConfig(
                    name="node",
                    git_url="https://github.com/example/node",
                    commit_sha="seed-sha",
                )
            ],
            workflow_file="workflow.json",
        )

        async def git(_node_dir: Path, *args: str) -> tuple[int, str, str]:
            if args == ("rev-parse", "--show-toplevel"):
                return 0, str(node_dir), ""
            if args == ("remote", "get-url", "origin"):
                return 0, "https://github.com/example/node", ""
            return 0, "live-sha", ""

        with patch.object(snapshot_manager, "_git", new=git):
            result = await snapshot_manager._scan_custom_nodes(seed)

        assert result[0].commit_sha == "live-sha"
        assert snapshot_manager._last_custom_node_scan.carried == ()

    async def test_git_root_must_be_the_node_directory(
        self,
        snapshot_manager: SnapshotManager,
        comfyui_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        custom_nodes = comfyui_path / "custom_nodes"
        custom_nodes.mkdir()
        node_dir = custom_nodes / "worktree-node"
        node_dir.mkdir()
        (node_dir / ".git").write_text("gitdir: elsewhere\n")

        git = AsyncMock(return_value=(0, str(comfyui_path), "ancestor repository"))
        with (
            patch.object(snapshot_manager, "_git", new=git),
            caplog.at_level("WARNING", logger="ai_content_service.snapshot"),
        ):
            result = await snapshot_manager._scan_custom_nodes()

        assert result == []
        git.assert_awaited_once_with(node_dir, "rev-parse", "--show-toplevel")
        skipped = self._events(caplog, "snapshot.custom_node_skipped")
        assert skipped[0]["reason"] == "not_repo_root"
        assert skipped[0]["stderr"] == "ancestor repository"

    async def test_helper_files_do_not_emit_skip_warnings(
        self,
        snapshot_manager: SnapshotManager,
        comfyui_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        custom_nodes = comfyui_path / "custom_nodes"
        custom_nodes.mkdir()
        (custom_nodes / "__pycache__").mkdir()
        (custom_nodes / "helper.py").write_text("# helper\n")
        (custom_nodes / "example.example").write_text("example\n")
        (custom_nodes / "actual-node").mkdir()

        with caplog.at_level("WARNING", logger="ai_content_service.snapshot"):
            result = await snapshot_manager._scan_custom_nodes()

        assert result == []
        skipped = self._events(caplog, "snapshot.custom_node_skipped")
        assert [event["name"] for event in skipped] == ["actual-node"]

    async def test_records_requirements_and_reports_uncovered_pyproject_dependencies(
        self,
        snapshot_manager: SnapshotManager,
        comfyui_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        custom_nodes = comfyui_path / "custom_nodes"
        custom_nodes.mkdir()
        node_dir = custom_nodes / "node"
        node_dir.mkdir()
        (node_dir / ".git").mkdir()
        (node_dir / "requirements.txt").write_text("pillow>=10.3.0\ncolor-matcher\n")
        (node_dir / "pyproject.toml").write_text(
            '[project]\ndependencies = ["pillow>=10.3.0", "color-matcher", "matplotlib"]\n'
        )

        async def git(repo_path: Path, *args: str) -> tuple[int, str, str]:
            if args == ("rev-parse", "--show-toplevel"):
                return 0, str(repo_path), ""
            if args == ("remote", "get-url", "origin"):
                return 0, "https://github.com/test/node", ""
            return 0, "deadbeef", ""

        with (
            patch.object(snapshot_manager, "_git", new=git),
            caplog.at_level("INFO", logger="ai_content_service.snapshot"),
        ):
            result = await snapshot_manager._scan_custom_nodes()

        assert result[0].pip_requirements == []
        requirements = self._events(caplog, "snapshot.custom_node_requirements")
        assert requirements[0]["name"] == "node"
        assert requirements[0]["count"] == 2
        dependencies = self._events(caplog, "snapshot.custom_node_pyproject_deps")
        assert dependencies[0]["uncovered_dependencies"] == ["matplotlib"]

    async def test_directive_line_produces_empty_pip_requirements(
        self,
        snapshot_manager: SnapshotManager,
        comfyui_path: Path,
    ) -> None:
        """A ``-r``/``-c``/``-e`` directive line must never reach pip_requirements."""
        custom_nodes = comfyui_path / "custom_nodes"
        custom_nodes.mkdir()
        node_dir = custom_nodes / "node"
        node_dir.mkdir()
        (node_dir / ".git").mkdir()
        (node_dir / "requirements.txt").write_text("-r extras.txt\n")

        async def git(repo_path: Path, *args: str) -> tuple[int, str, str]:
            if args == ("rev-parse", "--show-toplevel"):
                return 0, str(repo_path), ""
            if args == ("remote", "get-url", "origin"):
                return 0, "https://github.com/test/node", ""
            return 0, "deadbeef", ""

        with patch.object(snapshot_manager, "_git", new=git):
            result = await snapshot_manager._scan_custom_nodes()

        assert result[0].pip_requirements == []

    async def test_plain_pinned_list_also_produces_empty_pip_requirements(
        self,
        snapshot_manager: SnapshotManager,
        comfyui_path: Path,
    ) -> None:
        """A requirements.txt with only plain pins is still never copied -- it
        would double-install correctly, but a scan never captures a
        repository-owned file either way."""
        custom_nodes = comfyui_path / "custom_nodes"
        custom_nodes.mkdir()
        node_dir = custom_nodes / "node"
        node_dir.mkdir()
        (node_dir / ".git").mkdir()
        (node_dir / "requirements.txt").write_text("torch==2.1.0\nnumpy==1.26.0\n")

        async def git(repo_path: Path, *args: str) -> tuple[int, str, str]:
            if args == ("rev-parse", "--show-toplevel"):
                return 0, str(repo_path), ""
            if args == ("remote", "get-url", "origin"):
                return 0, "https://github.com/test/node", ""
            return 0, "deadbeef", ""

        with patch.object(snapshot_manager, "_git", new=git):
            result = await snapshot_manager._scan_custom_nodes()

        assert result[0].pip_requirements == []

    async def test_from_bundle_carries_forward_explicit_pip_requirements(
        self,
        snapshot_manager: SnapshotManager,
        comfyui_path: Path,
    ) -> None:
        """A seed bundle's hand-authored pip_requirements survive a rescan."""
        custom_nodes = comfyui_path / "custom_nodes"
        custom_nodes.mkdir()
        node_dir = custom_nodes / "node"
        node_dir.mkdir()
        (node_dir / ".git").mkdir()
        (node_dir / "requirements.txt").write_text("color-matcher\n")

        async def git(repo_path: Path, *args: str) -> tuple[int, str, str]:
            if args == ("rev-parse", "--show-toplevel"):
                return 0, str(repo_path), ""
            if args == ("remote", "get-url", "origin"):
                return 0, "https://github.com/test/node", ""
            return 0, "deadbeef", ""

        seed = BundleConfig(
            metadata=BundleMetadata(name="demo", version="260101-01"),
            custom_nodes=[
                CustomNodeConfig(
                    name="node",
                    git_url="https://github.com/test/node",
                    commit_sha="deadbeef",
                    pip_requirements=["extra-package==1.0"],
                )
            ],
            workflow_file="workflow.json",
        )

        with patch.object(snapshot_manager, "_git", new=git):
            result = await snapshot_manager._scan_custom_nodes(seed)

        assert result[0].pip_requirements == ["extra-package==1.0"]

    @pytest.mark.parametrize("pyproject", [None, "this is not valid toml = ["])
    async def test_unreadable_registry_metadata_still_warns_without_raising(
        self,
        snapshot_manager: SnapshotManager,
        comfyui_path: Path,
        caplog: pytest.LogCaptureFixture,
        pyproject: str | None,
    ) -> None:
        custom_nodes = comfyui_path / "custom_nodes"
        custom_nodes.mkdir()
        node_dir = custom_nodes / "registry-node"
        node_dir.mkdir()
        if pyproject is not None:
            (node_dir / "pyproject.toml").write_text(pyproject)

        with caplog.at_level("WARNING", logger="ai_content_service.snapshot"):
            result = await snapshot_manager._scan_custom_nodes()

        assert result == []
        unsupported = self._events(caplog, "snapshot.custom_node_unsupported_source")
        assert unsupported[0]["project_name"] is None
        assert unsupported[0]["version"] is None
        assert unsupported[0]["repository"] is None

    async def test_summary_reports_captured_and_skipped_counts(
        self,
        snapshot_manager: SnapshotManager,
        comfyui_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        custom_nodes = comfyui_path / "custom_nodes"
        custom_nodes.mkdir()
        captured = custom_nodes / "captured"
        captured.mkdir()
        (captured / ".git").mkdir()
        (custom_nodes / "skipped").mkdir()

        async def git(repo_path: Path, *args: str) -> tuple[int, str, str]:
            if args == ("rev-parse", "--show-toplevel"):
                return 0, str(repo_path), ""
            if args == ("remote", "get-url", "origin"):
                return 0, "https://github.com/test/captured", ""
            return 0, "deadbeef", ""

        with (
            patch.object(snapshot_manager, "_git", new=git),
            caplog.at_level("INFO", logger="ai_content_service.snapshot"),
        ):
            result = await snapshot_manager._scan_custom_nodes()

        assert [node.name for node in result] == ["captured"]
        assert snapshot_manager._last_custom_node_scan.captured == ("captured",)
        assert snapshot_manager._last_custom_node_scan.skipped[0].name == "skipped"
        summaries = self._events(caplog, "snapshot.custom_nodes_summary")
        assert summaries[0]["captured"] == 1
        assert summaries[0]["skipped"] == 1


class TestGithubRepositoryParts:
    @pytest.mark.parametrize(
        "repository",
        [
            "https://github.com/kijai/ComfyUI-KJNodes",
            "https://github.com/kijai/ComfyUI-KJNodes.git",
            "https://github.com/kijai/ComfyUI-KJNodes/",
            "https://www.github.com/kijai/ComfyUI-KJNodes",
            "git+https://github.com/kijai/ComfyUI-KJNodes.git",
            "https://github.com/kijai/ComfyUI-KJNodes/tree/main",
            "git@github.com:kijai/ComfyUI-KJNodes.git",
        ],
    )
    def test_github_repository_parts_resolves_owner_and_repo(self, repository: str) -> None:
        assert SnapshotManager._github_repository_parts(repository) == (
            "kijai",
            "ComfyUI-KJNodes",
        )

    def test_github_repository_parts_rejects_non_github_host(self) -> None:
        assert SnapshotManager._github_repository_parts("https://gitlab.com/x/y") is None


def _make_async_cm(return_value: object) -> MagicMock:
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=return_value)
    cm.__aexit__ = AsyncMock(return_value=None)
    return cm


class TestSnapshotWorkflowApiConversion:
    """P0-2/P2-3: a note-bearing GUI graph must not be rejected by the converter."""

    async def test_markdown_note_does_not_cause_converter_rejection(
        self,
        comfyui_path: Path,
        bundles_path: Path,
        python_executable: Path,
        temp_dir: Path,
    ) -> None:
        gui_graph = {
            "nodes": [
                {"id": 1, "type": "MarkdownNote", "inputs": [], "widgets_values": ["## docs"]},
                {
                    "id": 2,
                    "type": "KSampler",
                    "inputs": [{"name": "steps", "widget": {}}],
                    "widgets_values": [8],
                },
            ],
            "links": [],
        }
        api_graph = {"2": {"class_type": "KSampler", "inputs": {"steps": 8}}}
        workflow_path = temp_dir / "workflow.json"
        workflow_path.write_text(json.dumps(gui_graph))

        manager = SnapshotManager(
            comfyui_path,
            bundles_path,
            python_executable=python_executable,
            comfyui_url="http://comfyui.local",
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = api_graph

        mock_client = MagicMock()
        mock_client.get = AsyncMock(side_effect=httpx.HTTPError("no version endpoint"))
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch(
            "ai_content_service.snapshot.httpx.AsyncClient",
            return_value=_make_async_cm(mock_client),
        ):
            graph, comments = await manager._snapshot_workflow_api(
                workflow_path, temp_dir / "rejected.json"
            )

        assert graph == api_graph
        assert comments == ()
        assert not (temp_dir / "rejected.json").exists()

    async def test_real_sync_error_still_causes_converter_rejection(
        self,
        comfyui_path: Path,
        bundles_path: Path,
        python_executable: Path,
        temp_dir: Path,
    ) -> None:
        gui_graph = {
            "nodes": [
                {
                    "id": 2,
                    "type": "KSampler",
                    "inputs": [{"name": "steps", "widget": {}}],
                    "widgets_values": [8],
                },
            ],
            "links": [],
        }
        # The converter response disagrees with the GUI graph's committed value.
        api_graph = {"2": {"class_type": "KSampler", "inputs": {"steps": 4}}}
        workflow_path = temp_dir / "workflow.json"
        workflow_path.write_text(json.dumps(gui_graph))

        manager = SnapshotManager(
            comfyui_path,
            bundles_path,
            python_executable=python_executable,
            comfyui_url="http://comfyui.local",
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = api_graph

        mock_client = MagicMock()
        mock_client.get = AsyncMock(side_effect=httpx.HTTPError("no version endpoint"))
        mock_client.post = AsyncMock(return_value=mock_response)

        rejected_path = temp_dir / "rejected.json"
        with patch(
            "ai_content_service.snapshot.httpx.AsyncClient",
            return_value=_make_async_cm(mock_client),
        ):
            graph, comments = await manager._snapshot_workflow_api(workflow_path, rejected_path)

        assert graph is None
        assert comments
        assert rejected_path.exists()

    async def test_snapshot_rejects_conversion_missing_the_output_node(
        self,
        comfyui_path: Path,
        bundles_path: Path,
        python_executable: Path,
        temp_dir: Path,
    ) -> None:
        """P0-1: a converter response dropping the terminal SaveImage node must be
        rejected, not silently committed as a graph that produces no output."""
        gui_graph = {
            "nodes": [
                {"id": 2, "type": "KSampler", "inputs": [], "widgets_values": []},
                {
                    "id": 9,
                    "type": "SaveImage",
                    "inputs": [{"name": "images", "link": 1}],
                    "widgets_values": [],
                },
            ],
            "links": [[1, 2, 0, 9, 0, "IMAGE"]],
        }
        # The converter response omits node 9 -- the SaveImage output.
        api_graph = {"2": {"class_type": "KSampler", "inputs": {}}}
        workflow_path = temp_dir / "workflow.json"
        workflow_path.write_text(json.dumps(gui_graph))
        rejected_path = temp_dir / "rejected.json"
        manager = SnapshotManager(
            comfyui_path,
            bundles_path,
            python_executable=python_executable,
            comfyui_url="http://comfyui.local",
        )

        response = MagicMock()
        response.status_code = 200
        response.json.return_value = api_graph
        client = MagicMock()
        client.get = AsyncMock(side_effect=httpx.HTTPError("no version endpoint"))
        client.post = AsyncMock(return_value=response)

        with patch(
            "ai_content_service.snapshot.httpx.AsyncClient",
            return_value=_make_async_cm(client),
        ):
            graph, comments = await manager._snapshot_workflow_api(workflow_path, rejected_path)

        assert graph is None
        assert comments
        assert rejected_path.exists()

    async def test_dangling_api_link_causes_converter_rejection(
        self,
        comfyui_path: Path,
        bundles_path: Path,
        python_executable: Path,
        temp_dir: Path,
    ) -> None:
        gui_graph = {
            "nodes": [
                {"id": 2, "type": "KSampler", "inputs": [], "widgets_values": []},
            ],
            "links": [],
        }
        api_graph = {"2": {"class_type": "KSampler", "inputs": {"seed": ["missing", 0]}}}
        workflow_path = temp_dir / "workflow.json"
        workflow_path.write_text(json.dumps(gui_graph))
        rejected_path = temp_dir / "rejected.json"
        manager = SnapshotManager(
            comfyui_path,
            bundles_path,
            python_executable=python_executable,
            comfyui_url="http://comfyui.local",
        )

        response = MagicMock()
        response.status_code = 200
        response.json.return_value = api_graph
        client = MagicMock()
        client.get = AsyncMock(side_effect=httpx.HTTPError("no version endpoint"))
        client.post = AsyncMock(return_value=response)

        with patch(
            "ai_content_service.snapshot.httpx.AsyncClient",
            return_value=_make_async_cm(client),
        ):
            graph, comments = await manager._snapshot_workflow_api(workflow_path, rejected_path)

        assert graph is None
        assert comments
        assert rejected_path.exists()

    async def test_warning_only_sync_result_still_accepts_converter_response(
        self,
        comfyui_path: Path,
        bundles_path: Path,
        python_executable: Path,
        temp_dir: Path,
    ) -> None:
        gui_graph = {
            "nodes": [
                {
                    "id": 2,
                    "type": "KSampler",
                    "inputs": [{"name": "steps"}],
                    "widgets_values": [8],
                },
            ],
            "links": [],
        }
        api_graph = {"2": {"class_type": "KSampler", "inputs": {"steps": 8}}}
        workflow_path = temp_dir / "workflow.json"
        workflow_path.write_text(json.dumps(gui_graph))
        manager = SnapshotManager(
            comfyui_path,
            bundles_path,
            python_executable=python_executable,
            comfyui_url="http://comfyui.local",
        )
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = api_graph
        client = MagicMock()
        client.get = AsyncMock(side_effect=httpx.HTTPError("no version endpoint"))
        client.post = AsyncMock(return_value=response)

        with patch(
            "ai_content_service.snapshot.httpx.AsyncClient",
            return_value=_make_async_cm(client),
        ):
            graph, comments = await manager._snapshot_workflow_api(
                workflow_path, temp_dir / "rejected.json"
            )

        assert graph == api_graph
        assert comments == ()


class TestWriteRejectedApi:
    """P2-4: a rejected-response write failure must not fail the snapshot."""

    async def test_write_failure_logs_warning_and_does_not_raise(
        self,
        snapshot_manager: SnapshotManager,
        temp_dir: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        rejected_path = temp_dir / "workflow.api.json.rejected"

        with (
            patch.object(Path, "write_text", side_effect=OSError("disk full")),
            caplog.at_level("WARNING", logger="ai_content_service.snapshot"),
        ):
            await snapshot_manager._write_rejected_api(rejected_path, {"1": {}})

        warnings = [
            record.msg
            for record in caplog.records
            if isinstance(record.msg, dict)
            and record.msg.get("event") == "snapshot.workflow_api_rejected_write_failed"
        ]
        assert len(warnings) == 1
        assert warnings[0]["path"] == str(rejected_path)
        assert not rejected_path.exists()


class TestBundleYamlAsciiLocale:
    """P0-3: bundle.yaml must write correctly regardless of the preferred locale encoding."""

    def test_write_bundle_files_never_consults_preferred_encoding(self, tmp_path: Path) -> None:
        """The crash was `open("w")` falling back to locale.getpreferredencoding(False),
        which is 'ascii' under the bare LC_ALL=C a Vast.ai container actually runs
        with. Passing encoding="utf-8" explicitly means that call is never made at
        all -- assert that directly rather than relying on some byte tripping the
        codec, since yaml.safe_dump already escapes non-ASCII values to ASCII."""
        config = BundleConfig(
            metadata=BundleMetadata(name="demo", version="260101-01", description="café"),
        )
        config_path = tmp_path / "bundle.yaml"
        requirements_path = tmp_path / "requirements.overlay.txt"
        fallback_comment = (
            "TODO: export via Graph -> Export (API) and commit alongside workflow.json"
        )

        def _boom(*_args: object, **_kwargs: object) -> str:
            raise AssertionError("locale.getpreferredencoding must not be consulted")

        with patch("locale.getpreferredencoding", side_effect=_boom):
            _write_bundle_files(
                config_path,
                config,
                requirements_path,
                "torch==2.1.0\n",
                workflow_comments=(fallback_comment,),
            )

        assert config_path.exists()
        assert requirements_path.read_text(encoding="utf-8") == "torch==2.1.0\n"
        assert fallback_comment in config_path.read_text(encoding="utf-8")

    def test_generated_comment_survives_as_trailing_comment_when_field_absent(self) -> None:
        """P2-5: workflow_api_file is omitted on the fallback path, so the fallback
        TODO must degrade to a trailing comment rather than raising."""
        config = BundleConfig(
            metadata=BundleMetadata(name="demo", version="260101-01"),
        )
        fallback_comment = (
            "TODO: export via Graph -> Export (API) and commit alongside workflow.json"
        )

        rendered = _render_bundle_yaml(config, workflow_comments=(fallback_comment,))

        assert "workflow_api_file" not in rendered
        assert f"# {fallback_comment}" in rendered
