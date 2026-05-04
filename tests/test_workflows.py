"""Tests for workflow management."""

import json
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from ai_content_service.workflows import WorkflowError, WorkflowManager


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
def workflow_manager(comfyui_path: Path) -> WorkflowManager:
    return WorkflowManager(comfyui_path)


@pytest.fixture
def valid_workflow(temp_dir: Path) -> Path:
    workflow = temp_dir / "workflow.json"
    workflow.write_text(json.dumps({"nodes": [{"id": 1, "type": "KSampler"}]}))
    return workflow


class TestWorkflowManagerInstall:
    async def test_install_raises_when_workflow_not_found(
        self, workflow_manager: WorkflowManager, temp_dir: Path
    ) -> None:
        with pytest.raises(WorkflowError, match="Workflow not found"):
            await workflow_manager.install(temp_dir / "missing.json", "bundle")

    async def test_install_raises_on_invalid_json(
        self, workflow_manager: WorkflowManager, temp_dir: Path
    ) -> None:
        bad = temp_dir / "bad.json"
        bad.write_text("{not valid json")
        with pytest.raises(WorkflowError, match="Invalid workflow JSON"):
            await workflow_manager.install(bad, "bundle")

    async def test_install_returns_path_with_bundle_prefix(
        self, workflow_manager: WorkflowManager, valid_workflow: Path
    ) -> None:
        result = await workflow_manager.install(valid_workflow, "mybundle")
        assert result.name == "mybundle_workflow.json"

    async def test_install_creates_user_dir(
        self, workflow_manager: WorkflowManager, valid_workflow: Path
    ) -> None:
        await workflow_manager.install(valid_workflow, "mybundle")
        assert (workflow_manager._comfyui_path / "user").is_dir()

    async def test_install_copies_content_correctly(
        self, workflow_manager: WorkflowManager, valid_workflow: Path
    ) -> None:
        expected = json.loads(valid_workflow.read_text())
        result = await workflow_manager.install(valid_workflow, "mybundle")
        assert json.loads(result.read_text()) == expected

    async def test_install_overwrites_existing(
        self, workflow_manager: WorkflowManager, valid_workflow: Path
    ) -> None:
        user_dir = workflow_manager._comfyui_path / "user"
        user_dir.mkdir()
        existing = user_dir / "mybundle_workflow.json"
        existing.write_text('{"old": true}')

        await workflow_manager.install(valid_workflow, "mybundle")

        assert json.loads(existing.read_text()) != {"old": True}


class TestWorkflowManagerList:
    def test_list_returns_empty_when_no_user_dir(self, workflow_manager: WorkflowManager) -> None:
        assert workflow_manager.list_workflows() == []

    def test_list_returns_empty_for_empty_user_dir(self, workflow_manager: WorkflowManager) -> None:
        (workflow_manager._comfyui_path / "user").mkdir()
        assert workflow_manager.list_workflows() == []

    def test_list_returns_sorted_json_files(self, workflow_manager: WorkflowManager) -> None:
        user_dir = workflow_manager._comfyui_path / "user"
        user_dir.mkdir()
        (user_dir / "b_workflow.json").write_text("{}")
        (user_dir / "a_workflow.json").write_text("{}")

        result = workflow_manager.list_workflows()

        assert len(result) == 2
        assert result[0].name == "a_workflow.json"
        assert result[1].name == "b_workflow.json"

    def test_list_ignores_non_json_files(self, workflow_manager: WorkflowManager) -> None:
        user_dir = workflow_manager._comfyui_path / "user"
        user_dir.mkdir()
        (user_dir / "workflow.json").write_text("{}")
        (user_dir / "notes.txt").write_text("ignore")

        result = workflow_manager.list_workflows()
        assert len(result) == 1
        assert result[0].name == "workflow.json"


class TestWorkflowManagerRemove:
    def test_remove_raises_when_not_found(self, workflow_manager: WorkflowManager) -> None:
        with pytest.raises(WorkflowError, match="Workflow not found"):
            workflow_manager.remove_workflow("missing.json")

    def test_remove_deletes_file(self, workflow_manager: WorkflowManager) -> None:
        user_dir = workflow_manager._comfyui_path / "user"
        user_dir.mkdir()
        target = user_dir / "mybundle_workflow.json"
        target.write_text("{}")

        workflow_manager.remove_workflow("mybundle_workflow.json")

        assert not target.exists()

    def test_remove_leaves_other_workflows_intact(self, workflow_manager: WorkflowManager) -> None:
        user_dir = workflow_manager._comfyui_path / "user"
        user_dir.mkdir()
        (user_dir / "a_workflow.json").write_text("{}")
        (user_dir / "b_workflow.json").write_text("{}")

        workflow_manager.remove_workflow("a_workflow.json")

        assert not (user_dir / "a_workflow.json").exists()
        assert (user_dir / "b_workflow.json").exists()
