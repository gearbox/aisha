"""Tests for snapshot management."""

import json
import tempfile
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from ai_content_service.snapshot import SnapshotError, SnapshotManager


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
def snapshot_manager(comfyui_path: Path, bundles_path: Path) -> SnapshotManager:
    return SnapshotManager(comfyui_path, bundles_path)


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
        self, temp_dir: Path, bundles_path: Path, workflow_file: Path
    ) -> None:
        manager = SnapshotManager(temp_dir / "nonexistent", bundles_path)
        with pytest.raises(SnapshotError, match="ComfyUI not found"):
            await manager.create_snapshot("mybundle", workflow_file)

    async def test_raises_when_workflow_not_found(
        self, snapshot_manager: SnapshotManager, temp_dir: Path
    ) -> None:
        with pytest.raises(SnapshotError, match="Workflow not found"):
            await snapshot_manager.create_snapshot("mybundle", temp_dir / "missing.json")


class TestGenerateVersion:
    def test_new_bundle_gets_first_version(self, snapshot_manager: SnapshotManager) -> None:
        today = datetime.now().strftime("%y%m%d")
        version = snapshot_manager._generate_version("new_bundle")
        assert version == f"{today}-01"

    def test_increments_sequence_for_existing_today_versions(
        self, snapshot_manager: SnapshotManager, bundles_path: Path
    ) -> None:
        today = datetime.now().strftime("%y%m%d")
        bundle_dir = bundles_path / "mybundle"
        (bundle_dir / f"{today}-01").mkdir(parents=True)

        version = snapshot_manager._generate_version("mybundle")
        assert version == f"{today}-02"

    def test_increments_past_existing_max_sequence(
        self, snapshot_manager: SnapshotManager, bundles_path: Path
    ) -> None:
        today = datetime.now().strftime("%y%m%d")
        bundle_dir = bundles_path / "mybundle"
        (bundle_dir / f"{today}-01").mkdir(parents=True)
        (bundle_dir / f"{today}-05").mkdir(parents=True)

        version = snapshot_manager._generate_version("mybundle")
        assert version == f"{today}-06"

    def test_previous_day_versions_do_not_affect_sequence(
        self, snapshot_manager: SnapshotManager, bundles_path: Path
    ) -> None:
        today = datetime.now().strftime("%y%m%d")
        bundle_dir = bundles_path / "mybundle"
        (bundle_dir / "250101-99").mkdir(parents=True)  # old date

        version = snapshot_manager._generate_version("mybundle")
        assert version == f"{today}-01"


class TestCreateSnapshotSuccess:
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
            return ok_pip if args[0] == "pip" else ok_commit

        with patch("asyncio.create_subprocess_exec", new=mock_exec):
            version = await snapshot_manager.create_snapshot("mybundle", workflow_file)

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
            version = await snapshot_manager.create_snapshot(
                "mybundle", workflow_file, description="test"
            )

        config_path = bundles_path / "mybundle" / version / "bundle.yaml"
        assert config_path.exists()
        config = yaml.safe_load(config_path.read_text())
        assert config["metadata"]["name"] == "mybundle"
        assert config["metadata"]["description"] == "test"

    async def test_writes_requirements_lock(
        self,
        snapshot_manager: SnapshotManager,
        workflow_file: Path,
        bundles_path: Path,
    ) -> None:
        pip_output = b"torch==2.1.0\nnumpy==1.24.0\n"
        ok_commit = make_mock_process(returncode=0, stdout=b"abc123\n")
        ok_pip = make_mock_process(returncode=0, stdout=pip_output)

        async def mock_exec(*args: object, **_kwargs: object) -> MagicMock:
            return ok_pip if args[0] == "pip" else ok_commit

        with patch("asyncio.create_subprocess_exec", new=mock_exec):
            version = await snapshot_manager.create_snapshot("mybundle", workflow_file)

        req_path = bundles_path / "mybundle" / version / "requirements.lock"
        assert req_path.exists()
        assert "torch==2.1.0" in req_path.read_text()

    async def test_copies_workflow_json(
        self,
        snapshot_manager: SnapshotManager,
        workflow_file: Path,
        bundles_path: Path,
    ) -> None:
        ok = make_mock_process(returncode=0, stdout=b"abc123\n")
        with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=ok)):
            version = await snapshot_manager.create_snapshot("mybundle", workflow_file)

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
            version = await snapshot_manager.create_snapshot("mybundle", workflow_file)

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
            first = await snapshot_manager.create_snapshot("mybundle", workflow_file)

        # Manually simulate a second version being present before creating third
        today = datetime.now().strftime("%y%m%d")
        second_dir = bundles_path / "mybundle" / f"{today}-99"
        second_dir.mkdir()

        with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=ok)):
            await snapshot_manager.create_snapshot("mybundle", workflow_file)

        # Current symlink should still point to the first version created
        current_link = bundles_path / "mybundle" / "current"
        assert current_link.resolve().name == first


class TestScanCustomNodes:
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

        ok_remote = make_mock_process(returncode=0, stdout=b"https://github.com/test/node\n")
        ok_commit = make_mock_process(returncode=0, stdout=b"deadbeef\n")

        call_index = 0

        async def mock_exec(*args: object, **_kwargs: object) -> MagicMock:
            nonlocal call_index
            call_index += 1
            # Alternate: first call is remote, second is commit
            return ok_remote if "remote" in args else ok_commit

        with patch("asyncio.create_subprocess_exec", new=mock_exec):
            result = await snapshot_manager._scan_custom_nodes()

        assert len(result) == 1
        assert result[0].name == "MyNode"
        assert result[0].git_url == "https://github.com/test/node"
        assert result[0].commit_sha == "deadbeef"
