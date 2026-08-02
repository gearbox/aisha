"""Tests for ComfyUI management."""

import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from ai_content_service.comfyui import (
    _MIN_CHECKPOINT_BYTES,
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
        with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=ok)):
            await manager.install_base_requirements()  # should not raise


class TestInstallLockedRequirements:
    async def test_raises_when_file_missing(self, manager: ComfyUIManager, temp_dir: Path) -> None:
        with pytest.raises(ComfyUIError, match="Requirements file not found"):
            await manager.install_locked_requirements(temp_dir / "missing.lock")

    async def test_succeeds_with_existing_file(
        self, manager: ComfyUIManager, temp_dir: Path
    ) -> None:
        req_file = temp_dir / "requirements.lock"
        req_file.write_text("torch==2.1.0\n")
        ok = make_mock_process(returncode=0)
        with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=ok)):
            await manager.install_locked_requirements(req_file)  # should not raise


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
        (ckpt_dir / "model.safetensors").write_bytes(b"x" * _MIN_CHECKPOINT_BYTES)
        problems = await manager.verify(
            expected=[
                ExpectedArtifact(
                    relative_path=Path("checkpoints/model.safetensors"),
                    min_bytes=_MIN_CHECKPOINT_BYTES,
                )
            ]
        )
        assert problems == []

    async def test_declared_size_mismatch_warns_but_does_not_fail(
        self, manager: ComfyUIManager, comfyui_path: Path
    ) -> None:
        ckpt_dir = self._make_ckpt_dir(comfyui_path)
        actual_size = _MIN_CHECKPOINT_BYTES + 1
        (ckpt_dir / "model.safetensors").write_bytes(b"x" * actual_size)
        with patch("ai_content_service.comfyui.log.warning") as warning:
            problems = await manager.verify(
                expected=[
                    ExpectedArtifact(
                        relative_path=Path("checkpoints/model.safetensors"),
                        min_bytes=_MIN_CHECKPOINT_BYTES,
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
                    min_bytes=_MIN_CHECKPOINT_BYTES,
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
        (sub_dir / "model.safetensors").write_bytes(b"x" * _MIN_CHECKPOINT_BYTES)
        problems = await manager.verify(
            expected=[
                ExpectedArtifact(
                    relative_path=Path("checkpoints/Wan/22/model.safetensors"),
                    min_bytes=_MIN_CHECKPOINT_BYTES,
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
        (ckpt_dir / "model.safetensors").write_bytes(b"x" * _MIN_CHECKPOINT_BYTES)
        with patch("httpx.AsyncClient") as mock_http:
            await manager.verify(
                expected=[
                    ExpectedArtifact(
                        relative_path=Path("checkpoints/model.safetensors"),
                        min_bytes=_MIN_CHECKPOINT_BYTES,
                    )
                ]
            )
        mock_http.assert_not_called()

    async def test_never_hashes_only_stats(
        self, manager: ComfyUIManager, comfyui_path: Path
    ) -> None:
        """Pitfall #5: verify must `stat`, never read/hash file contents."""
        ckpt_dir = self._make_ckpt_dir(comfyui_path)
        (ckpt_dir / "model.safetensors").write_bytes(b"x" * _MIN_CHECKPOINT_BYTES)
        with patch("pathlib.Path.open") as mock_open:
            problems = await manager.verify(
                expected=[
                    ExpectedArtifact(
                        relative_path=Path("checkpoints/model.safetensors"),
                        min_bytes=_MIN_CHECKPOINT_BYTES,
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


class TestInstallLockedRequirementsIgnoresInstalled:
    async def test_locked_requirements_install_uses_ignore_installed(
        self, manager: ComfyUIManager, temp_dir: Path
    ) -> None:
        """Locked overlay must bypass uninstall to survive debian-managed packages.

        Regression guard for the Phase 1 v0.6.2 failure on vastai/comfy where
        `pip install -r requirements.lock` errored with
        'Cannot uninstall wheel 0.42.0, RECORD file not found' because the
        image installs `wheel` via apt without pip metadata.
        """
        req_file = temp_dir / "requirements.lock"
        req_file.write_text(
            "--extra-index-url https://download.pytorch.org/whl/cu129\n"
            "torch==2.8.0+cu129\n"
            "wheel==0.45.1\n"
        )

        captured: dict[str, tuple] = {}

        async def fake_exec(*args: str, **_kwargs: object) -> object:
            captured["args"] = args
            return make_mock_process(returncode=0)

        with patch("asyncio.create_subprocess_exec", new=fake_exec):
            await manager.install_locked_requirements(req_file)

        assert "--ignore-installed" in captured["args"], (
            "install_locked_requirements must use --ignore-installed to bypass "
            "uninstall of debian-managed packages with missing RECORD files"
        )
        ignore_idx = captured["args"].index("--ignore-installed")
        install_idx = captured["args"].index("install")
        assert ignore_idx > install_idx

    async def test_base_requirements_does_not_use_ignore_installed(
        self, manager: ComfyUIManager, comfyui_path: Path
    ) -> None:
        """Only the locked overlay needs --ignore-installed; base requirements don't."""
        (comfyui_path / "requirements.txt").write_text("numpy")

        captured: dict[str, tuple] = {}

        async def fake_exec(*args: str, **_kwargs: object) -> object:
            captured["args"] = args
            return make_mock_process(returncode=0)

        with patch("asyncio.create_subprocess_exec", new=fake_exec):
            await manager.install_base_requirements()

        assert "--ignore-installed" not in captured["args"]
