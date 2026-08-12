"""Tests for ComfyUI management."""

import json
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import ANY, AsyncMock, MagicMock, call, patch

import httpx
import pytest

from ai_content_service.comfyui import (
    MIN_CHECKPOINT_BYTES,
    ComfyUIError,
    ComfyUIManager,
    ComfyUIStatus,
    ExpectedArtifact,
)
from ai_content_service.config import CustomNodeConfig


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
def manager(comfyui_path: Path) -> ComfyUIManager:
    return ComfyUIManager(comfyui_path, python_executable=Path(sys.executable))


def make_mock_process(returncode: int = 0, stdout: bytes = b"", stderr: bytes = b"") -> MagicMock:
    proc = MagicMock()
    proc.returncode = returncode
    proc.communicate = AsyncMock(return_value=(stdout, stderr))
    return proc


class TestCheckout:
    async def test_raises_when_path_missing(self, temp_dir: Path) -> None:
        manager = ComfyUIManager(temp_dir / "nonexistent", python_executable=Path(sys.executable))
        with pytest.raises(ComfyUIError, match="ComfyUI not found"):
            await manager.checkout("abc123")

    async def test_raises_on_git_failure(self, manager: ComfyUIManager) -> None:
        failing = make_mock_process(returncode=1, stderr=b"not a git repo")
        with (
            patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=failing)),
            pytest.raises(ComfyUIError, match="Git command failed"),
        ):
            await manager.checkout("abc123")

    async def test_succeeds_on_git_success(self, manager: ComfyUIManager) -> None:
        ok = make_mock_process(returncode=0)
        with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=ok)):
            await manager.checkout("abc123")  # should not raise


class TestInstallBaseRequirements:
    async def test_raises_when_requirements_missing(self, manager: ComfyUIManager) -> None:
        with pytest.raises(ComfyUIError, match=r"requirements\.txt not found"):
            await manager.install_base_requirements()

    async def test_raises_on_pip_failure(self, manager: ComfyUIManager, comfyui_path: Path) -> None:
        (comfyui_path / "requirements.txt").write_text("torch")
        failing = make_mock_process(returncode=1, stderr=b"pip error")
        with (
            patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=failing)),
            pytest.raises(ComfyUIError, match="Pip command failed"),
        ):
            await manager.install_base_requirements()

    async def test_succeeds_when_requirements_present(
        self, manager: ComfyUIManager, comfyui_path: Path
    ) -> None:
        (comfyui_path / "requirements.txt").write_text("torch")
        ok = make_mock_process(returncode=0)
        pip_list = make_mock_process(stdout=b"[]")

        async def capture(*args: object, **_kwargs: object) -> MagicMock:
            return pip_list if "list" in args else ok

        with patch("asyncio.create_subprocess_exec", new=capture):
            await manager.install_base_requirements()  # should not raise

    async def test_identical_requirements_skip_pip_install(
        self, manager: ComfyUIManager, comfyui_path: Path
    ) -> None:
        (comfyui_path / "requirements.txt").write_text("torch==2.1.0\n")
        pip_list = make_mock_process(stdout=b'[{"name": "torch", "version": "2.1.0"}]')
        calls: list[tuple[object, ...]] = []

        async def capture(*args: object, **_kwargs: object) -> MagicMock:
            calls.append(args)
            return pip_list

        with patch("asyncio.create_subprocess_exec", new=capture):
            await manager.install_base_requirements()

        assert len(calls) == 1
        assert calls[0][3:] == ("list", "--format=json", "--disable-pip-version-check")


class TestLockedRequirementsPipCommands:
    async def test_raises_when_file_missing(self, manager: ComfyUIManager, temp_dir: Path) -> None:
        with pytest.raises(ComfyUIError, match="Requirements file not found"):
            await manager.install_locked_requirements(temp_dir / "missing.lock")

    async def test_succeeds_with_existing_file(
        self, manager: ComfyUIManager, temp_dir: Path
    ) -> None:
        req_file = temp_dir / "requirements.lock"
        req_file.write_text("torch==2.1.0\n")
        installed = make_mock_process(
            returncode=0,
            stdout=b'[{"name": "torch", "version": "2.1.0"}]',
        )
        with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=installed)):
            delta = await manager.install_locked_requirements(req_file)

        assert delta.should_install is False

    async def test_identical_lock_skips_pip_install_and_logs_delta(
        self, manager: ComfyUIManager, temp_dir: Path
    ) -> None:
        req_file = temp_dir / "requirements.lock"
        req_file.write_text("torch==2.1.0\npackaging==1.0\n")
        pip_list = make_mock_process(
            stdout=b'[{"name": "Torch", "version": "2.1.0"}, '
            b'{"name": "packaging", "version": "1.0.0"}]'
        )
        calls: list[tuple[object, ...]] = []

        async def capture(*args: object, **_kwargs: object) -> MagicMock:
            calls.append(args)
            return pip_list

        with (
            patch("asyncio.create_subprocess_exec", new=capture),
            patch("ai_content_service.comfyui.log.info") as info,
        ):
            delta = await manager.install_locked_requirements(req_file)

        assert delta.metrics() == {
            "total": 2,
            "satisfied": 2,
            "missing": 0,
            "conflicting": 0,
            "conflicting_sample": [],
            "unparseable": 0,
            "outcome": "skipped",
        }
        assert len(calls) == 1
        assert calls[0][3:] == ("list", "--format=json", "--disable-pip-version-check")
        info.assert_called_once_with("requirements.lock.delta", **delta.metrics())

    async def test_delta_file_installs_only_missing_and_conflicting_packages(
        self, manager: ComfyUIManager, temp_dir: Path
    ) -> None:
        req_file = temp_dir / "requirements.lock"
        satisfied_names = [f"satisfied-package-{index}" for index in range(157)]
        req_file.write_text(
            "".join(f"{name}==1.0\n" for name in satisfied_names)
            + "missing-package==3.0\nconflicting-package[feature]==4.0\n"
        )
        pip_list = make_mock_process(
            stdout=json.dumps(
                [
                    *({"name": name, "version": "1.0"} for name in satisfied_names),
                    {"name": "conflicting-package", "version": "5.0"},
                ]
            ).encode()
        )
        pip_install = make_mock_process()
        calls: list[tuple[object, ...]] = []
        pip_requirement_files: list[Path] = []
        pip_requirements: list[str] = []

        async def capture(*args: object, **_kwargs: object) -> MagicMock:
            calls.append(args)
            if "list" in args:
                return pip_list
            requirement_file = Path(str(args[-1]))
            pip_requirement_files.append(requirement_file)
            pip_requirements.append(requirement_file.read_text())
            return pip_install

        with patch("asyncio.create_subprocess_exec", new=capture):
            delta = await manager.install_locked_requirements(req_file)

        assert delta.metrics()["missing"] == 1
        assert delta.metrics()["conflicting"] == 1
        assert len(calls) == 2
        assert calls[1][3:-1] == ("install", "-r")
        assert pip_requirement_files[0] != req_file
        assert pip_requirements == ["missing-package==3.0\nconflicting-package[feature]==4.0\n"]
        assert all(name not in pip_requirements[0] for name in satisfied_names)
        assert not pip_requirement_files[0].exists()

    async def test_conflicting_package_warns_before_install(
        self, manager: ComfyUIManager, temp_dir: Path
    ) -> None:
        req_file = temp_dir / "requirements.lock"
        req_file.write_text("torch==2.1.0\n")
        pip_list = make_mock_process(stdout=b'[{"name": "torch", "version": "2.2.0"}]')
        pip_install = make_mock_process()

        async def capture(*args: object, **_kwargs: object) -> MagicMock:
            return pip_list if "list" in args else pip_install

        with (
            patch("asyncio.create_subprocess_exec", new=capture),
            patch("ai_content_service.comfyui.log.warning") as warning,
        ):
            delta = await manager.install_locked_requirements(req_file)

        assert delta.metrics()["conflicting"] == 1
        assert delta.metrics()["outcome"] == "installed"
        assert delta.metrics()["conflicting_sample"] == [
            {"name": "torch", "locked": "2.1.0", "installed": "2.2.0"}
        ]
        warning.assert_called_once_with(
            "requirements.lock.conflict",
            packages={"torch": {"locked": "2.1.0", "installed": "2.2.0"}},
        )

    async def test_conflicting_install_failure_warns_and_records_outcome(
        self, manager: ComfyUIManager, temp_dir: Path
    ) -> None:
        req_file = temp_dir / "requirements.lock"
        req_file.write_text("torch==2.1.0\n")
        pip_list = make_mock_process(stdout=b'[{"name": "torch", "version": "2.2.0"}]')
        pip_install = make_mock_process(returncode=1, stderr=b"conda uninstall failed")

        async def capture(*args: object, **_kwargs: object) -> MagicMock:
            return pip_list if "list" in args else pip_install

        with (
            patch("asyncio.create_subprocess_exec", new=capture),
            patch("ai_content_service.comfyui.log.warning") as warning,
        ):
            delta = await manager.install_locked_requirements(req_file)

        assert delta.metrics()["outcome"] == "conflict_install_failed"
        warning.assert_has_calls(
            [
                call(
                    "requirements.lock.conflict",
                    packages={"torch": {"locked": "2.1.0", "installed": "2.2.0"}},
                ),
                call(
                    "requirements.lock.conflict_install_failed",
                    error=ANY,
                    packages=["torch"],
                ),
            ]
        )

    async def test_missing_package_install_failure_raises_and_removes_delta_file(
        self, manager: ComfyUIManager, temp_dir: Path
    ) -> None:
        req_file = temp_dir / "requirements.lock"
        req_file.write_text("missing-package==3.0\n")
        pip_list = make_mock_process(stdout=b"[]")
        pip_install = make_mock_process(returncode=1, stderr=b"pip error")
        pip_requirement_files: list[Path] = []

        async def capture(*args: object, **_kwargs: object) -> MagicMock:
            if "list" in args:
                return pip_list
            requirement_file = Path(str(args[-1]))
            pip_requirement_files.append(requirement_file)
            assert requirement_file.exists()
            return pip_install

        with (
            patch("asyncio.create_subprocess_exec", new=capture),
            pytest.raises(ComfyUIError, match="Pip command failed"),
        ):
            await manager.install_locked_requirements(req_file)

        assert len(pip_requirement_files) == 1
        assert not pip_requirement_files[0].exists()

    async def test_unparseable_lock_line_never_skips_install(
        self, manager: ComfyUIManager, temp_dir: Path
    ) -> None:
        req_file = temp_dir / "requirements.lock"
        req_file.write_text("torch==2.1.0\ngit+https://example.test/project.git\n")
        pip_list = make_mock_process(stdout=b'[{"name": "torch", "version": "2.1.0"}]')
        pip_install = make_mock_process()
        calls: list[tuple[object, ...]] = []

        async def capture(*args: object, **_kwargs: object) -> MagicMock:
            calls.append(args)
            return pip_list if "list" in args else pip_install

        with patch("asyncio.create_subprocess_exec", new=capture):
            delta = await manager.install_locked_requirements(req_file)

        assert delta.metrics()["unparseable"] == 1
        assert delta.should_install is True
        assert delta.metrics()["outcome"] == "installed"
        assert calls[1][3:] == ("install", "-r", str(req_file))


class TestInstallCustomNode:
    async def test_raises_when_no_commit_sha(self, manager: ComfyUIManager) -> None:
        node = CustomNodeConfig(
            name="TestNode",
            git_url="https://github.com/test/node",
            commit_sha=None,
        )
        ok = make_mock_process(returncode=0)
        with (
            patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=ok)),
            pytest.raises(ComfyUIError, match="No commit SHA"),
        ):
            await manager.install_custom_node(node)

    async def test_clones_new_node(self, manager: ComfyUIManager) -> None:
        node = CustomNodeConfig(
            name="TestNode",
            git_url="https://github.com/test/node",
            commit_sha="abc123",
        )
        ok = make_mock_process(returncode=0)
        calls: list[tuple] = []

        async def capture(*args: object, **_kwargs: object) -> MagicMock:
            calls.append(args)
            return ok

        with patch("asyncio.create_subprocess_exec", new=capture):
            await manager.install_custom_node(node)

        # First call should be git clone
        assert calls[0][0] == "git"
        assert "clone" in calls[0]

    async def test_fetches_existing_node(self, manager: ComfyUIManager, comfyui_path: Path) -> None:
        node_dir = comfyui_path / "custom_nodes" / "TestNode"
        node_dir.mkdir(parents=True)
        node = CustomNodeConfig(
            name="TestNode",
            git_url="https://github.com/test/node",
            commit_sha="abc123",
        )
        ok = make_mock_process(returncode=0)
        calls: list[tuple] = []

        async def capture(*args: object, **_kwargs: object) -> MagicMock:
            calls.append(args)
            return ok

        with patch("asyncio.create_subprocess_exec", new=capture):
            await manager.install_custom_node(node)

        # First call should be git fetch (node already exists)
        assert calls[0][0] == "git"
        assert "fetch" in calls[0]


def make_mock_http_client(
    status_code: int | None = 200, error: Exception | None = None
) -> AsyncMock:
    """Build an AsyncMock for httpx.AsyncClient that works as an async context manager."""
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    if error is not None:
        mock_client.get.side_effect = error
    else:
        mock_client.get.return_value = MagicMock(status_code=status_code)
    return mock_client


class TestVerify:
    def _make_ckpt_dir(self, comfyui_dir: Path) -> Path:
        ckpt_dir = comfyui_dir / "models" / "checkpoints"
        ckpt_dir.mkdir(parents=True)
        return ckpt_dir

    async def test_returns_no_problems_when_all_artifacts_present(
        self, manager: ComfyUIManager, comfyui_path: Path
    ) -> None:
        ckpt_dir = self._make_ckpt_dir(comfyui_path)
        (ckpt_dir / "model.safetensors").write_bytes(b"x" * MIN_CHECKPOINT_BYTES)
        problems = await manager.verify(
            expected=[
                ExpectedArtifact(
                    relative_path=Path("checkpoints/model.safetensors"),
                    min_bytes=MIN_CHECKPOINT_BYTES,
                )
            ]
        )
        assert problems == []

    @pytest.mark.parametrize(
        ("relative_path", "size", "min_bytes"),
        [
            (Path("upscale_models/upscale.bin"), 4 * 1024 * 1024, 1 * 1024 * 1024),
            (Path("embeddings/embedding.pt"), 20 * 1024, 1024),
        ],
    )
    async def test_small_valid_artifacts_pass_their_lightweight_floors(
        self,
        manager: ComfyUIManager,
        comfyui_path: Path,
        relative_path: Path,
        size: int,
        min_bytes: int,
    ) -> None:
        destination = comfyui_path / "models" / relative_path
        destination.parent.mkdir(parents=True)
        destination.write_bytes(b"x" * size)

        problems = await manager.verify(
            expected=[ExpectedArtifact(relative_path=relative_path, min_bytes=min_bytes)]
        )

        assert problems == []

    async def test_declared_size_mismatch_warns_but_does_not_fail(
        self, manager: ComfyUIManager, comfyui_path: Path
    ) -> None:
        ckpt_dir = self._make_ckpt_dir(comfyui_path)
        actual_size = MIN_CHECKPOINT_BYTES + 1
        (ckpt_dir / "model.safetensors").write_bytes(b"x" * actual_size)
        with patch("ai_content_service.comfyui.log.warning") as warning:
            problems = await manager.verify(
                expected=[
                    ExpectedArtifact(
                        relative_path=Path("checkpoints/model.safetensors"),
                        min_bytes=MIN_CHECKPOINT_BYTES,
                        declared_bytes=actual_size + 3,
                    )
                ]
            )

        assert problems == []
        warning.assert_called_once_with(
            "verify.size.declared_mismatch",
            path=str(ckpt_dir / "model.safetensors"),
            declared=actual_size + 3,
            actual=actual_size,
        )

    async def test_returns_no_problems_for_empty_expected_list(
        self, manager: ComfyUIManager, comfyui_path: Path
    ) -> None:
        self._make_ckpt_dir(comfyui_path)
        problems = await manager.verify(expected=[])
        assert problems == []

    async def test_reports_missing_artifact(
        self, manager: ComfyUIManager, comfyui_path: Path
    ) -> None:
        self._make_ckpt_dir(comfyui_path)
        problems = await manager.verify(
            expected=[
                ExpectedArtifact(relative_path=Path("checkpoints/missing.safetensors"), min_bytes=1)
            ]
        )
        assert len(problems) == 1
        assert "checkpoints/missing.safetensors" in problems[0]

    async def test_reports_truncated_artifact(
        self, manager: ComfyUIManager, comfyui_path: Path
    ) -> None:
        ckpt_dir = self._make_ckpt_dir(comfyui_path)
        (ckpt_dir / "tiny.safetensors").write_bytes(b"x" * 1024)  # well below floor
        problems = await manager.verify(
            expected=[
                ExpectedArtifact(
                    relative_path=Path("checkpoints/tiny.safetensors"),
                    min_bytes=MIN_CHECKPOINT_BYTES,
                )
            ]
        )
        assert len(problems) == 1
        assert "checkpoints/tiny.safetensors" in problems[0]

    async def test_reports_artifact_under_a_subdirectory(
        self, manager: ComfyUIManager, comfyui_path: Path
    ) -> None:
        """A model with `subdirectory` set: verify must look under
        `models/<type>/<subdir>/<filename>`, not `models/<type>/<filename>` (C1/C2)."""
        sub_dir = comfyui_path / "models" / "checkpoints" / "Wan" / "22"
        sub_dir.mkdir(parents=True)
        (sub_dir / "model.safetensors").write_bytes(b"x" * MIN_CHECKPOINT_BYTES)
        problems = await manager.verify(
            expected=[
                ExpectedArtifact(
                    relative_path=Path("checkpoints/Wan/22/model.safetensors"),
                    min_bytes=MIN_CHECKPOINT_BYTES,
                )
            ]
        )
        assert problems == []

    async def test_names_every_problem_in_one_call(
        self, manager: ComfyUIManager, comfyui_path: Path
    ) -> None:
        self._make_ckpt_dir(comfyui_path)
        problems = await manager.verify(
            expected=[
                ExpectedArtifact(relative_path=Path("checkpoints/a.safetensors"), min_bytes=1),
                ExpectedArtifact(relative_path=Path("checkpoints/b.safetensors"), min_bytes=1),
            ]
        )
        assert len(problems) == 2

    async def test_missing_artifact_treated_as_oserror(
        self, manager: ComfyUIManager, comfyui_path: Path
    ) -> None:
        self._make_ckpt_dir(comfyui_path)
        with patch("pathlib.Path.stat", side_effect=OSError("permission denied")):
            problems = await manager.verify(
                expected=[
                    ExpectedArtifact(
                        relative_path=Path("checkpoints/model.safetensors"), min_bytes=1
                    )
                ]
            )
        assert len(problems) == 1

    async def test_no_network_calls_during_verify(
        self, manager: ComfyUIManager, comfyui_path: Path
    ) -> None:
        ckpt_dir = self._make_ckpt_dir(comfyui_path)
        (ckpt_dir / "model.safetensors").write_bytes(b"x" * MIN_CHECKPOINT_BYTES)
        with patch("httpx.AsyncClient") as mock_http:
            await manager.verify(
                expected=[
                    ExpectedArtifact(
                        relative_path=Path("checkpoints/model.safetensors"),
                        min_bytes=MIN_CHECKPOINT_BYTES,
                    )
                ]
            )
        mock_http.assert_not_called()

    async def test_never_hashes_only_stats(
        self, manager: ComfyUIManager, comfyui_path: Path
    ) -> None:
        """Pitfall #5: verify must `stat`, never read/hash file contents."""
        ckpt_dir = self._make_ckpt_dir(comfyui_path)
        (ckpt_dir / "model.safetensors").write_bytes(b"x" * MIN_CHECKPOINT_BYTES)
        with patch("pathlib.Path.open") as mock_open:
            problems = await manager.verify(
                expected=[
                    ExpectedArtifact(
                        relative_path=Path("checkpoints/model.safetensors"),
                        min_bytes=MIN_CHECKPOINT_BYTES,
                    )
                ]
            )
        assert problems == []
        mock_open.assert_not_called()


class TestCountCustomNodes:
    def test_returns_zero_when_no_custom_nodes_dir(self, manager: ComfyUIManager) -> None:
        assert manager._count_custom_nodes() == 0

    def test_counts_directories_only(self, manager: ComfyUIManager, comfyui_path: Path) -> None:
        custom_nodes = comfyui_path / "custom_nodes"
        custom_nodes.mkdir()
        (custom_nodes / "NodeA").mkdir()
        (custom_nodes / "NodeB").mkdir()
        (custom_nodes / "readme.txt").write_text("not a node")

        assert manager._count_custom_nodes() == 2

    def test_ignores_hidden_directories(self, manager: ComfyUIManager, comfyui_path: Path) -> None:
        custom_nodes = comfyui_path / "custom_nodes"
        custom_nodes.mkdir()
        (custom_nodes / "NodeA").mkdir()
        (custom_nodes / ".hidden").mkdir()

        assert manager._count_custom_nodes() == 1


class TestGetStatus:
    async def test_returns_comfyui_status(self, manager: ComfyUIManager) -> None:
        ok = make_mock_process(returncode=0, stdout=b"deadbeef\n")
        mock_client = make_mock_http_client(status_code=200)

        with (
            patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=ok)),
            patch("httpx.AsyncClient", return_value=mock_client),
        ):
            status = await manager.get_status()

        assert isinstance(status, ComfyUIStatus)
        assert status.commit == "deadbeef"
        assert status.is_running is True

    async def test_returns_none_commit_when_git_fails(self, manager: ComfyUIManager) -> None:
        failing = make_mock_process(returncode=1)
        error = httpx.RequestError("refused", request=MagicMock())
        mock_client = make_mock_http_client(error=error)

        with (
            patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=failing)),
            patch("httpx.AsyncClient", return_value=mock_client),
        ):
            status = await manager.get_status()

        assert status.commit is None
        assert status.is_running is False


class TestCheckRunningProbesLoopbackForWildcardHost:
    async def test_check_running_probes_loopback_for_wildcard_host(
        self, comfyui_path: Path
    ) -> None:
        """The default host (0.0.0.0) must not be dialed directly; probe 127.0.0.1 instead."""
        manager = ComfyUIManager(comfyui_path, python_executable=Path(sys.executable))
        mock_client = make_mock_http_client(status_code=200)

        with patch("httpx.AsyncClient", return_value=mock_client):
            await manager._check_running()

        requested_url = mock_client.get.call_args.args[0]
        assert requested_url.startswith("http://127.0.0.1:")

    async def test_check_running_probes_ipv6_wildcard_as_loopback(self, comfyui_path: Path) -> None:
        manager = ComfyUIManager(comfyui_path, python_executable=Path(sys.executable), host="::")
        mock_client = make_mock_http_client(status_code=200)

        with patch("httpx.AsyncClient", return_value=mock_client):
            await manager._check_running()

        requested_url = mock_client.get.call_args.args[0]
        assert requested_url.startswith("http://127.0.0.1:")

    async def test_check_running_passes_through_explicit_host(self, comfyui_path: Path) -> None:
        manager = ComfyUIManager(
            comfyui_path, python_executable=Path(sys.executable), host="192.168.1.50"
        )
        mock_client = make_mock_http_client(status_code=200)

        with patch("httpx.AsyncClient", return_value=mock_client):
            await manager._check_running()

        requested_url = mock_client.get.call_args.args[0]
        assert requested_url.startswith("http://192.168.1.50:")


class TestRunPipUsesConfiguredInterpreter:
    async def test_pip_command_invokes_configured_python_with_dash_m(self, tmp_path: Path) -> None:
        """_run_pip must invoke `<python_executable> -m pip ...`, not bare `pip`."""
        fake_python = tmp_path / "fake-venv" / "bin" / "python"
        fake_python.parent.mkdir(parents=True)
        fake_python.write_text("")
        fake_python.chmod(0o755)

        manager = ComfyUIManager(
            comfyui_path=tmp_path / "ComfyUI",
            python_executable=fake_python,
        )

        captured: dict[str, tuple] = {}

        async def fake_exec(*args: str, **_kwargs: object) -> object:
            captured["args"] = args
            return make_mock_process(returncode=0)

        with patch("asyncio.create_subprocess_exec", new=fake_exec):
            await manager._run_pip(["install", "foo"])

        assert captured["args"][0] == str(fake_python)
        assert captured["args"][1] == "-m"
        assert captured["args"][2] == "pip"
        assert "install" in captured["args"]
        assert "foo" in captured["args"]


class TestLockedRequirementInstallCommand:
    async def test_locked_requirements_install_does_not_force_reinstall(
        self, manager: ComfyUIManager, temp_dir: Path
    ) -> None:
        """A non-authoritative lock still installs, without forcing a reinstall."""
        req_file = temp_dir / "requirements.lock"
        req_file.write_text(
            "--extra-index-url https://download.pytorch.org/whl/cu129\n"
            "torch==2.8.0+cu129\n"
            "wheel==0.45.1\n"
        )

        captured: list[tuple[str, ...]] = []

        async def fake_exec(*args: str, **_kwargs: object) -> object:
            captured.append(args)
            if "list" in args:
                return make_mock_process(
                    returncode=0,
                    stdout=(
                        b'[{"name": "torch", "version": "2.8.0+cu129"}, '
                        b'{"name": "wheel", "version": "0.45.1"}]'
                    ),
                )
            return make_mock_process(returncode=0)

        with patch("asyncio.create_subprocess_exec", new=fake_exec):
            await manager.install_locked_requirements(req_file)

        assert len(captured) == 2
        assert captured[1][3:] == ("install", "-r", str(req_file))

    async def test_base_requirements_installs_its_delta_without_forcing_reinstall(
        self, manager: ComfyUIManager, comfyui_path: Path
    ) -> None:
        """Base requirements use the same measured-delta path as bundle locks."""
        req_file = comfyui_path / "requirements.txt"
        req_file.write_text("numpy==2.0.0\n")

        captured: list[tuple[str, ...]] = []

        async def fake_exec(*args: str, **_kwargs: object) -> object:
            captured.append(args)
            if "list" in args:
                return make_mock_process(returncode=0, stdout=b"[]")
            return make_mock_process(returncode=0)

        with patch("asyncio.create_subprocess_exec", new=fake_exec):
            await manager.install_base_requirements()

        assert len(captured) == 2
        assert "--ignore-installed" not in captured[1]
        assert captured[1][3:-1] == ("install", "-r")
        assert Path(captured[1][-1]) != req_file
