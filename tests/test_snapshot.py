"""Tests for snapshot management."""

import hashlib
import json
import os
import re
import tempfile
import threading
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from ai_content_service.config import (
    BundleConfig,
    BundleMetadata,
    BundleVersion,
    ModelConfig,
    ModelFileConfig,
)
from ai_content_service.snapshot import (
    SnapshotError,
    SnapshotManager,
    _hash_model_file,
    _HashResult,
    _render_bundle_yaml,
)


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
            return ok_pip if "freeze" in args else ok_commit

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
            version = await snapshot_manager.create_snapshot("mybundle", workflow_file)

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
            version = await snapshot_manager.create_snapshot("mybundle", workflow_file)

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
