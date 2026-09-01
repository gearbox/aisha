"""Tests for ComfyUI management."""

import ast
import io
import json
import stat
import sys
import tempfile
import zipfile
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import ANY, AsyncMock, MagicMock, call, patch

import httpx
import pytest

from ai_content_service import comfyui as comfyui_module
from ai_content_service.comfyui import (
    MIN_CHECKPOINT_BYTES,
    ComfyUIError,
    ComfyUIManager,
    ComfyUIStatus,
    ExpectedArtifact,
)
from ai_content_service.config import CustomNodeConfig


def _build_zip(entries: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    return buffer.getvalue()


class _StreamingResponse:
    """Small streaming-response double that fails if production reads .content."""

    def __init__(
        self,
        chunks: list[bytes],
        *,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.chunks = chunks
        self.status_code = status_code
        self.headers = headers or {}
        self.iterated = False

    @property
    def content(self) -> bytes:
        raise AssertionError("registry downloads must not materialize response.content")

    async def aiter_bytes(self, _chunk_size: int):
        self.iterated = True
        for chunk in self.chunks:
            yield chunk


def _make_registry_client(
    get_side_effect: list[object], download_response: _StreamingResponse | None = None
) -> AsyncMock:
    """An AsyncMock usable as `async with httpx.AsyncClient(...) as client`."""
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.get = AsyncMock(side_effect=get_side_effect)
    if download_response is not None:
        stream_context = MagicMock()
        stream_context.__aenter__ = AsyncMock(return_value=download_response)
        stream_context.__aexit__ = AsyncMock(return_value=False)
        client.stream = MagicMock(return_value=stream_context)
    return client


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


def test_every_comfyui_manager_construction_passes_port_and_host() -> None:
    """Keep every composition root on Settings' configured ComfyUI endpoint."""
    source_root = Path(__file__).parents[1] / "src" / "ai_content_service"
    calls = [
        node
        for path in source_root.rglob("*.py")
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ComfyUIManager"
    ]

    assert len(calls) == 3
    assert all({keyword.arg for keyword in call.keywords} >= {"port", "host"} for call in calls)


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


class TestRestartAndWait:
    async def test_restart_and_wait_returns_when_node_class_appears(
        self, manager: ComfyUIManager
    ) -> None:
        process = make_mock_process()
        response = MagicMock(status_code=200)
        response.json.return_value = {"ReadyNode": {}}
        client = AsyncMock()
        client.__aenter__.return_value = client
        client.get = AsyncMock(return_value=response)

        with (
            patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)),
            patch.object(manager, "_check_running", new=AsyncMock(return_value=True)),
            patch("httpx.AsyncClient", return_value=client),
        ):
            await manager.restart_and_wait(
                node_class="ReadyNode",
                restart_command=("supervisorctl", "restart", "comfyui"),
                timeout_s=1.0,
                poll_interval_s=0.5,
            )

    async def test_restart_waits_for_the_process_to_go_down_first(
        self, manager: ComfyUIManager
    ) -> None:
        process = make_mock_process()
        states = iter((True, False, True))
        events: list[str] = []

        async def check_running() -> bool:
            running = next(states)
            events.append("up" if running else "down")
            return running

        async def node_class_available(_node_class: str) -> tuple[bool, bool]:
            assert events == ["up", "down", "up"]
            return True, False

        with (
            patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)),
            patch.object(manager, "_check_running", new=check_running),
            patch.object(manager, "_node_class_available", new=node_class_available),
            patch("asyncio.sleep", new=AsyncMock()),
        ):
            await manager.restart_and_wait(
                node_class="ReadyNode",
                restart_command=("supervisorctl", "restart", "comfyui"),
                timeout_s=1.0,
                poll_interval_s=0.5,
            )

        assert events == ["up", "down", "up"]

    async def test_restart_warns_when_no_down_transition_is_observed(
        self, manager: ComfyUIManager
    ) -> None:
        process = make_mock_process()

        with (
            patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)),
            patch.object(manager, "_check_running", new=AsyncMock(return_value=True)),
            patch.object(comfyui_module.log, "warning") as warning,
        ):
            await manager.restart_and_wait(
                node_class=None,
                restart_command=("supervisorctl", "restart", "comfyui"),
                timeout_s=0.0,
                poll_interval_s=0.5,
            )

        warning.assert_called_once_with("comfyui.restart.no_down_transition", down_window_s=0.0)

    async def test_restart_does_not_pass_readiness_against_the_pre_restart_process(
        self, manager: ComfyUIManager
    ) -> None:
        process = make_mock_process()
        readiness = AsyncMock(return_value=(True, False))

        with (
            patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)),
            patch.object(
                manager, "_check_running", new=AsyncMock(side_effect=(True, False, False))
            ),
            patch.object(manager, "_node_class_available", new=readiness),
            pytest.raises(ComfyUIError, match="did not come up"),
        ):
            await manager.restart_and_wait(
                node_class="ReadyNode",
                restart_command=("supervisorctl", "restart", "comfyui"),
                timeout_s=0.0,
                poll_interval_s=0.5,
            )

        readiness.assert_not_awaited()

    async def test_restart_and_wait_distinguishes_process_down_from_class_missing(
        self, manager: ComfyUIManager
    ) -> None:
        process = make_mock_process()

        with (
            patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)),
            patch.object(manager, "_check_running", new=AsyncMock(return_value=False)),
            pytest.raises(ComfyUIError, match="did not come up"),
        ):
            await manager.restart_and_wait(
                node_class="ReadyNode",
                restart_command=("supervisorctl", "restart", "comfyui"),
                timeout_s=0.0,
                poll_interval_s=0.5,
            )

        response = MagicMock(status_code=200)
        response.json.return_value = {"OtherNode": {}}
        client = AsyncMock()
        client.__aenter__.return_value = client
        client.get = AsyncMock(return_value=response)
        with (
            patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)),
            patch.object(manager, "_check_running", new=AsyncMock(return_value=True)),
            patch("httpx.AsyncClient", return_value=client),
            pytest.raises(ComfyUIError, match="custom node may have failed to import"),
        ):
            await manager.restart_and_wait(
                node_class="ReadyNode",
                restart_command=("supervisorctl", "restart", "comfyui"),
                timeout_s=0.0,
                poll_interval_s=0.5,
            )

    async def test_restart_and_wait_falls_back_to_full_object_info_on_404(
        self, manager: ComfyUIManager
    ) -> None:
        process = make_mock_process()
        class_response = MagicMock(status_code=404)
        full_response = MagicMock(status_code=200)
        full_response.json.return_value = {"ReadyNode": {}}
        client = AsyncMock()
        client.__aenter__.return_value = client
        client.get = AsyncMock(side_effect=[class_response, full_response])

        with (
            patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)),
            patch.object(manager, "_check_running", new=AsyncMock(return_value=True)),
            patch("httpx.AsyncClient", return_value=client),
        ):
            await manager.restart_and_wait(
                node_class="ReadyNode",
                restart_command=("supervisorctl", "restart", "comfyui"),
                timeout_s=1.0,
                poll_interval_s=0.5,
            )

        assert client.get.await_count == 2

    async def test_restart_command_is_never_shell_interpreted(
        self, manager: ComfyUIManager
    ) -> None:
        process = make_mock_process()
        command = ("supervisorctl; touch /tmp/not-run", "restart", "comfyui")

        with (
            patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)) as create,
            patch.object(manager, "_check_running", new=AsyncMock(return_value=True)),
        ):
            await manager.restart_and_wait(
                node_class=None,
                restart_command=command,
                timeout_s=1.0,
                poll_interval_s=0.5,
            )

        await_args = create.await_args
        assert await_args is not None
        assert await_args.args == command


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

    async def test_on_conflict_fail_raises_before_pip_runs(
        self, manager: ComfyUIManager, temp_dir: Path
    ) -> None:
        req_file = temp_dir / "requirements.lock"
        req_file.write_text("torch==2.1.0\n")
        pip_list = make_mock_process(stdout=b'[{"name": "torch", "version": "2.2.0"}]')
        calls: list[tuple[object, ...]] = []

        async def capture(*args: object, **_kwargs: object) -> MagicMock:
            calls.append(args)
            return pip_list

        with (
            patch("asyncio.create_subprocess_exec", new=capture),
            pytest.raises(ComfyUIError, match=r"torch: locked=2\.1\.0 installed=2\.2\.0"),
        ):
            await manager.install_locked_requirements(req_file, on_conflict="fail")

        assert len(calls) == 1
        assert "list" in calls[0]

    async def test_on_conflict_install_preserves_existing_behaviour(
        self, manager: ComfyUIManager, temp_dir: Path
    ) -> None:
        req_file = temp_dir / "requirements.lock"
        req_file.write_text("torch==2.1.0\n")
        pip_list = make_mock_process(stdout=b'[{"name": "torch", "version": "2.2.0"}]')
        pip_install = make_mock_process()

        async def capture(*args: object, **_kwargs: object) -> MagicMock:
            return pip_list if "list" in args else pip_install

        with patch("asyncio.create_subprocess_exec", new=capture):
            delta = await manager.install_locked_requirements(req_file, on_conflict="install")

        assert delta.metrics()["outcome"] == "installed"

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

    async def test_missing_file_reference_is_treated_as_satisfied_and_never_reaches_pip(
        self, manager: ComfyUIManager, temp_dir: Path
    ) -> None:
        req_file = temp_dir / "requirements.lock"
        req_file.write_text(
            "packaging @ file:///conda/feedstock_root/build_artifacts/packaging\ntorch==2.1.0\n"
        )
        pip_list = make_mock_process(stdout=b'[{"name": "torch", "version": "2.1.0"}]')
        calls: list[tuple[object, ...]] = []

        async def capture(*args: object, **_kwargs: object) -> MagicMock:
            calls.append(args)
            return pip_list

        with (
            patch("asyncio.create_subprocess_exec", new=capture),
            patch("ai_content_service.comfyui.log.warning") as warning,
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
        warning.assert_called_once_with(
            "requirements.lock.unresolvable_reference", package="packaging"
        )

    async def test_existing_file_reference_is_installed_as_a_requirement(
        self, manager: ComfyUIManager, temp_dir: Path
    ) -> None:
        artifact = temp_dir / "package.whl"
        artifact.write_bytes(b"wheel")
        req_file = temp_dir / "requirements.lock"
        requirement = f"example-package @ {artifact.as_uri()}"
        req_file.write_text(f"{requirement}\n")
        pip_list = make_mock_process(stdout=b'[{"name": "example-package", "version": "1.0"}]')
        pip_install = make_mock_process()
        installed_requirements: list[str] = []

        async def capture(*args: object, **_kwargs: object) -> MagicMock:
            if "list" in args:
                return pip_list
            delta_path = Path(str(args[-1]))
            installed_requirements.append(delta_path.read_text())
            return pip_install

        with patch("asyncio.create_subprocess_exec", new=capture):
            delta = await manager.install_locked_requirements(req_file)

        assert delta.metrics()["unparseable"] == 0
        assert delta.metrics()["missing"] == 1
        assert installed_requirements == [f"{requirement}\n"]

    async def test_https_reference_is_installed_as_a_requirement(
        self, manager: ComfyUIManager, temp_dir: Path
    ) -> None:
        req_file = temp_dir / "requirements.lock"
        requirement = "example-package @ https://example.test/package.whl#sha256=abc123"
        req_file.write_text(f"{requirement}\n")
        pip_list = make_mock_process(stdout=b'[{"name": "example-package", "version": "1.0"}]')
        pip_install = make_mock_process()
        installed_requirements: list[str] = []

        async def capture(*args: object, **_kwargs: object) -> MagicMock:
            if "list" in args:
                return pip_list
            delta_path = Path(str(args[-1]))
            installed_requirements.append(delta_path.read_text())
            return pip_install

        with patch("asyncio.create_subprocess_exec", new=capture):
            delta = await manager.install_locked_requirements(req_file)

        assert delta.metrics()["unparseable"] == 0
        assert delta.metrics()["missing"] == 1
        assert installed_requirements == [f"{requirement}\n"]

    async def test_overlay_uses_distinct_delta_log_event(
        self, manager: ComfyUIManager, temp_dir: Path
    ) -> None:
        req_file = temp_dir / "requirements.overlay.txt"
        req_file.write_text("overlay==1.0\n")
        pip_list = make_mock_process(stdout=b'[{"name": "overlay", "version": "1.0"}]')

        with (
            patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=pip_list)),
            patch("ai_content_service.comfyui.log.info") as info,
        ):
            delta = await manager.install_locked_requirements(req_file, source="overlay")

        info.assert_called_once_with("requirements.overlay.delta", **delta.metrics())


class TestInstallCustomNode:
    async def test_raises_when_no_commit_sha(self, manager: ComfyUIManager) -> None:
        """CustomNode.validate_source_fields now rejects this at construction;
        model_construct bypasses it to confirm the deploy-time guard (needed
        for mypy narrowing and defense-in-depth) still catches a malformed
        node built some other way.
        """
        node = CustomNodeConfig.model_construct(
            name="TestNode",
            source="git",
            git_url="https://github.com/test/node",
            commit_sha=None,
            node_id=None,
            version=None,
            archive_sha256=None,
            pip_requirements=[],
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
            commit_sha="a" * 40,
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
            commit_sha="a" * 40,
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

    async def test_requirements_txt_installs_exactly_once(
        self, manager: ComfyUIManager, comfyui_path: Path
    ) -> None:
        """A directive line (``-r``) is unparseable as a lock pin, so the delta
        machinery installs directly from the file -- once, from the node's own
        directory, where the directive resolves correctly."""
        node_dir = comfyui_path / "custom_nodes" / "TestNode"
        node_dir.mkdir(parents=True)
        (node_dir / "requirements.txt").write_text("-r extras.txt\n")
        node = CustomNodeConfig(
            name="TestNode",
            git_url="https://github.com/test/node",
            commit_sha="a" * 40,
        )
        pip_list = make_mock_process(returncode=0, stdout=b"[]")
        pip_install = make_mock_process(returncode=0)
        install_calls: list[tuple[object, ...]] = []

        async def capture(*args: object, **_kwargs: object) -> MagicMock:
            if args[0] == "git":
                return make_mock_process(returncode=0)
            if "list" in args:
                return pip_list
            install_calls.append(args)
            return pip_install

        with patch("asyncio.create_subprocess_exec", new=capture):
            delta = await manager.install_custom_node(node)

        assert len(install_calls) == 1
        assert install_calls[0][3:5] == ("install", "-r")
        assert install_calls[0][5] == str(node_dir / "requirements.txt")
        assert delta is not None

    async def test_satisfied_node_requirements_skip_pip_and_route_through_delta(
        self, manager: ComfyUIManager, comfyui_path: Path
    ) -> None:
        """A fully-pinned requirements.txt the image already satisfies costs no
        pip call, and logs through the ``custom_node`` delta source."""
        node_dir = comfyui_path / "custom_nodes" / "TestNode"
        node_dir.mkdir(parents=True)
        (node_dir / "requirements.txt").write_text("torch==2.1.0\n")
        node = CustomNodeConfig(
            name="TestNode",
            git_url="https://github.com/test/node",
            commit_sha="a" * 40,
        )
        pip_list = make_mock_process(
            returncode=0, stdout=b'[{"name": "torch", "version": "2.1.0"}]'
        )
        install_calls: list[tuple[object, ...]] = []

        async def capture(*args: object, **_kwargs: object) -> MagicMock:
            if args[0] == "git":
                return make_mock_process(returncode=0)
            if "list" in args:
                return pip_list
            install_calls.append(args)
            return make_mock_process(returncode=0)

        with (
            patch("asyncio.create_subprocess_exec", new=capture),
            patch("ai_content_service.comfyui.log.info") as info,
        ):
            delta = await manager.install_custom_node(node)

        assert not install_calls
        assert delta is not None
        assert delta.should_install is False
        info.assert_any_call("requirements.custom_node.delta", **delta.metrics())

    async def test_custom_node_requirements_honour_on_conflict(
        self, manager: ComfyUIManager, comfyui_path: Path
    ) -> None:
        node_dir = comfyui_path / "custom_nodes" / "TestNode"
        node_dir.mkdir(parents=True)
        (node_dir / "requirements.txt").write_text("torch==2.1.0\n")
        node = CustomNodeConfig(
            name="TestNode",
            git_url="https://github.com/test/node",
            commit_sha="a" * 40,
        )
        pip_list = make_mock_process(stdout=b'[{"name": "torch", "version": "2.2.0"}]')
        calls: list[tuple[object, ...]] = []

        async def capture(*args: object, **_kwargs: object) -> MagicMock:
            calls.append(args)
            return make_mock_process() if args[0] == "git" else pip_list

        with (
            patch("asyncio.create_subprocess_exec", new=capture),
            pytest.raises(ComfyUIError, match="requirements conflict"),
        ):
            await manager.install_custom_node(node, on_conflict="fail")

        assert any("list" in call_args for call_args in calls)
        assert all("install" not in call_args for call_args in calls)

    async def test_no_requirements_txt_returns_none_delta(self, manager: ComfyUIManager) -> None:
        node = CustomNodeConfig(
            name="TestNode",
            git_url="https://github.com/test/node",
            commit_sha="a" * 40,
        )
        ok = make_mock_process(returncode=0)

        with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=ok)):
            delta = await manager.install_custom_node(node)

        assert delta is None

    async def test_pip_requirements_install_separately_from_requirements_txt(
        self, manager: ComfyUIManager, comfyui_path: Path
    ) -> None:
        """Pass 1 (the node's own file) and pass 2 (the bundle author's
        additions) must both run, as two distinct pip calls."""
        node_dir = comfyui_path / "custom_nodes" / "TestNode"
        node_dir.mkdir(parents=True)
        (node_dir / "requirements.txt").write_text("-r extras.txt\n")
        node = CustomNodeConfig(
            name="TestNode",
            git_url="https://github.com/test/node",
            commit_sha="a" * 40,
            pip_requirements=["extra-package==1.0"],
        )
        pip_list = make_mock_process(returncode=0, stdout=b"[]")
        install_calls: list[tuple[object, ...]] = []

        async def capture(*args: object, **_kwargs: object) -> MagicMock:
            if args[0] == "git":
                return make_mock_process(returncode=0)
            if "list" in args:
                return pip_list
            install_calls.append(args)
            return make_mock_process(returncode=0)

        with patch("asyncio.create_subprocess_exec", new=capture):
            await manager.install_custom_node(node)

        assert len(install_calls) == 2
        assert install_calls[0][3:5] == ("install", "-r")
        assert install_calls[1][3:] == ("install", "extra-package==1.0")

    async def test_git_source_never_touches_the_registry_http_path(
        self, manager: ComfyUIManager
    ) -> None:
        """P3 regression: a git entry (the default `source`) must not construct
        an httpx client at all -- branching is on `node.source`, not a URL."""
        node = CustomNodeConfig(
            name="TestNode", git_url="https://github.com/test/node", commit_sha="a" * 40
        )
        ok = make_mock_process(returncode=0)

        with (
            patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=ok)),
            patch("httpx.AsyncClient") as mock_ctor,
        ):
            await manager.install_custom_node(node)

        mock_ctor.assert_not_called()


class TestInstallRegistryCustomNode:
    """P3: deploy-time installation from an immutable Comfy Registry version."""

    async def test_install_registry_node_downloads_extracts_and_installs_requirements_through_delta(
        self, manager: ComfyUIManager, comfyui_path: Path
    ) -> None:
        node = CustomNodeConfig(
            name="comfyui-kjnodes",
            source="registry",
            node_id="comfyui-kjnodes",
            version="1.5.0",
        )
        zip_bytes = _build_zip(
            {"__init__.py": b"# node\n", "requirements.txt": b"pillow==10.3.0\n"}
        )
        version_response = MagicMock(status_code=200)
        version_response.json.return_value = {"downloadUrl": "https://cdn.comfy.org/node.zip"}
        download_response = _StreamingResponse([zip_bytes])
        client = _make_registry_client([version_response], download_response)

        pip_list = make_mock_process(returncode=0, stdout=b"[]")
        install_calls: list[tuple[object, ...]] = []

        async def capture(*args: object, **_kwargs: object) -> MagicMock:
            if "list" in args:
                return pip_list
            install_calls.append(args)
            return make_mock_process(returncode=0)

        with (
            patch("httpx.AsyncClient", return_value=client),
            patch("asyncio.create_subprocess_exec", new=capture),
        ):
            delta = await manager.install_custom_node(node)

        node_dir = comfyui_path / "custom_nodes" / "comfyui-kjnodes"
        assert (node_dir / "__init__.py").read_text() == "# node\n"
        assert (node_dir / "requirements.txt").exists()
        assert delta is not None
        assert install_calls  # the missing pin was installed through the delta path
        assert [p.name for p in (comfyui_path / "custom_nodes").iterdir()] == ["comfyui-kjnodes"]

    async def test_install_registry_node_nested_archive_prefix_still_extracts_to_node_name(
        self, manager: ComfyUIManager, comfyui_path: Path
    ) -> None:
        node = CustomNodeConfig(
            name="kjnodes", source="registry", node_id="comfyui-kjnodes", version="1.5.0"
        )
        zip_bytes = _build_zip({"comfyui-kjnodes-1.5.0/__init__.py": b"# node\n"})
        version_response = MagicMock(status_code=200)
        version_response.json.return_value = {"downloadUrl": "https://cdn.comfy.org/node.zip"}
        download_response = _StreamingResponse([zip_bytes])
        client = _make_registry_client([version_response], download_response)

        with patch("httpx.AsyncClient", return_value=client):
            await manager.install_custom_node(node)

        node_dir = comfyui_path / "custom_nodes" / "kjnodes"
        assert (node_dir / "__init__.py").read_text() == "# node\n"
        assert not (node_dir / "comfyui-kjnodes-1.5.0").exists()

    async def test_install_registry_node_digest_mismatch_raises_and_leaves_nothing_behind(
        self, manager: ComfyUIManager, comfyui_path: Path
    ) -> None:
        node = CustomNodeConfig(
            name="kjnodes",
            source="registry",
            node_id="comfyui-kjnodes",
            version="1.5.0",
            archive_sha256="0" * 64,
        )
        zip_bytes = _build_zip({"__init__.py": b"# node\n"})
        version_response = MagicMock(status_code=200)
        version_response.json.return_value = {"downloadUrl": "https://cdn.comfy.org/node.zip"}
        download_response = _StreamingResponse([zip_bytes])
        client = _make_registry_client([version_response], download_response)

        with (
            patch("httpx.AsyncClient", return_value=client),
            pytest.raises(ComfyUIError, match="digest mismatch"),
        ):
            await manager.install_custom_node(node)

        assert not list((comfyui_path / "custom_nodes").iterdir())

    async def test_install_registry_node_version_404_names_node_id_and_version(
        self, manager: ComfyUIManager
    ) -> None:
        node = CustomNodeConfig(
            name="kjnodes", source="registry", node_id="comfyui-kjnodes", version="9.9.9"
        )
        client = _make_registry_client([MagicMock(status_code=404)])

        with (
            patch("httpx.AsyncClient", return_value=client),
            pytest.raises(ComfyUIError) as exc_info,
        ):
            await manager.install_custom_node(node)

        assert "comfyui-kjnodes" in str(exc_info.value)
        assert "9.9.9" in str(exc_info.value)

    async def test_install_registry_node_matching_pyproject_without_marker_reinstalls(
        self, manager: ComfyUIManager, comfyui_path: Path
    ) -> None:
        node_dir = comfyui_path / "custom_nodes" / "kjnodes"
        node_dir.mkdir(parents=True)
        (node_dir / "pyproject.toml").write_text('[project]\nversion = "1.5.0"\n')
        (node_dir / "stale.py").write_text("# stale\n")
        node = CustomNodeConfig(
            name="kjnodes", source="registry", node_id="comfyui-kjnodes", version="1.5.0"
        )
        zip_bytes = _build_zip({"__init__.py": b"# replacement\n"})
        version_response = MagicMock(status_code=200)
        version_response.json.return_value = {"downloadUrl": "https://cdn.comfy.org/node.zip"}
        client = _make_registry_client([version_response], _StreamingResponse([zip_bytes]))

        with (
            patch("httpx.AsyncClient", return_value=client),
            patch("ai_content_service.comfyui.log.info") as info,
        ):
            delta = await manager.install_custom_node(node)

        assert delta is None
        assert (node_dir / "__init__.py").read_text() == "# replacement\n"
        assert not (node_dir / "stale.py").exists()
        info.assert_any_call(
            "custom_node.registry.no_provenance",
            name="kjnodes",
            installed_version="1.5.0",
            reason="marker_absent",
        )

    async def test_install_registry_node_matching_marker_skips_download_without_pyproject(
        self, manager: ComfyUIManager, comfyui_path: Path
    ) -> None:
        node_dir = comfyui_path / "custom_nodes" / "kjnodes"
        node_dir.mkdir(parents=True)
        (node_dir / ".aisha-registry-version").write_text("1.5.0\n")
        node = CustomNodeConfig(
            name="kjnodes", source="registry", node_id="comfyui-kjnodes", version="1.5.0"
        )

        with patch("httpx.AsyncClient") as mock_ctor:
            delta = await manager.install_custom_node(node)

        mock_ctor.assert_not_called()
        assert delta is None

    async def test_install_registry_node_different_installed_version_reinstalls_cleanly(
        self, manager: ComfyUIManager, comfyui_path: Path
    ) -> None:
        node_dir = comfyui_path / "custom_nodes" / "kjnodes"
        node_dir.mkdir(parents=True)
        (node_dir / "pyproject.toml").write_text('[project]\nversion = "1.4.0"\n')
        (node_dir / ".aisha-registry-version").write_text("1.4.0\n")
        (node_dir / "stale.py").write_text("# old\n")
        node = CustomNodeConfig(
            name="kjnodes", source="registry", node_id="comfyui-kjnodes", version="1.5.0"
        )
        zip_bytes = _build_zip({"__init__.py": b"# new\n"})
        version_response = MagicMock(status_code=200)
        version_response.json.return_value = {"downloadUrl": "https://cdn.comfy.org/node.zip"}
        download_response = _StreamingResponse([zip_bytes])
        client = _make_registry_client([version_response], download_response)

        with patch("httpx.AsyncClient", return_value=client):
            await manager.install_custom_node(node)

        assert (node_dir / "__init__.py").read_text() == "# new\n"
        assert not (node_dir / "stale.py").exists()
        assert (node_dir / ".aisha-registry-version").read_text() == "1.5.0\n"

    async def test_install_registry_node_unreadable_marker_reinstalls(
        self, manager: ComfyUIManager, comfyui_path: Path
    ) -> None:
        node_dir = comfyui_path / "custom_nodes" / "kjnodes"
        node_dir.mkdir(parents=True)
        (node_dir / ".aisha-registry-version").mkdir()
        (node_dir / "stale.py").write_text("# old\n")
        node = CustomNodeConfig(
            name="kjnodes", source="registry", node_id="comfyui-kjnodes", version="1.5.0"
        )
        zip_bytes = _build_zip({"__init__.py": b"# new\n"})
        version_response = MagicMock(status_code=200)
        version_response.json.return_value = {"downloadUrl": "https://cdn.comfy.org/node.zip"}
        client = _make_registry_client([version_response], _StreamingResponse([zip_bytes]))

        with patch("httpx.AsyncClient", return_value=client):
            await manager.install_custom_node(node)

        assert (node_dir / "__init__.py").read_text() == "# new\n"
        assert not (node_dir / "stale.py").exists()

    async def test_registry_archive_rejects_oversize_content_length_before_reading_body(
        self, manager: ComfyUIManager, comfyui_path: Path
    ) -> None:
        (comfyui_path / "custom_nodes").mkdir()
        response = _StreamingResponse([b"would not fit"], headers={"Content-Length": "11"})
        client = _make_registry_client([], response)

        with (
            patch("httpx.AsyncClient", return_value=client),
            patch.object(comfyui_module, "_REGISTRY_ARCHIVE_MAX_BYTES", 10),
            pytest.raises(ComfyUIError, match="maximum allowed size of 10 bytes"),
        ):
            await manager._download_registry_archive("kjnodes", "https://cdn.comfy.org/node.zip")

        assert not response.iterated
        assert not list((comfyui_path / "custom_nodes").iterdir())

    async def test_registry_archive_rejects_oversize_stream_and_removes_temp_file(
        self, manager: ComfyUIManager, comfyui_path: Path
    ) -> None:
        (comfyui_path / "custom_nodes").mkdir()
        response = _StreamingResponse([b"1234", b"5678"])
        client = _make_registry_client([], response)

        with (
            patch("httpx.AsyncClient", return_value=client),
            patch.object(comfyui_module, "_REGISTRY_ARCHIVE_MAX_BYTES", 6),
            pytest.raises(ComfyUIError, match="maximum allowed size of 6 bytes"),
        ):
            await manager._download_registry_archive("kjnodes", "https://cdn.comfy.org/node.zip")

        assert response.iterated
        assert not list((comfyui_path / "custom_nodes").iterdir())

    async def test_registry_archive_streams_a_response_without_accessing_content(
        self, manager: ComfyUIManager, comfyui_path: Path
    ) -> None:
        (comfyui_path / "custom_nodes").mkdir()
        response = _StreamingResponse([b"first", b"second"])
        client = _make_registry_client([], response)

        with patch("httpx.AsyncClient", return_value=client):
            archive_path = await manager._download_registry_archive(
                "kjnodes", "https://cdn.comfy.org/node.zip"
            )

        assert response.iterated
        assert archive_path.read_bytes() == b"firstsecond"
        assert archive_path.parent == comfyui_path.parent
        archive_path.unlink()

    async def test_registry_archive_prefers_configured_cache_staging_directory(
        self, comfyui_path: Path, temp_dir: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        cache_path = temp_dir / "cache"
        configured_manager = ComfyUIManager(
            comfyui_path,
            python_executable=Path(sys.executable),
            registry_archive_dir=cache_path,
        )
        response = _StreamingResponse([b"archive"])
        client = _make_registry_client([], response)

        with (
            patch("httpx.AsyncClient", return_value=client),
            caplog.at_level("INFO", logger="ai_content_service.comfyui"),
        ):
            archive_path = await configured_manager._download_registry_archive(
                "kjnodes", "https://cdn.comfy.org/node.zip"
            )

        assert archive_path.parent == cache_path
        assert any(
            isinstance(record.msg, dict)
            and record.msg.get("event") == "custom_node.registry.archive_staging"
            and record.msg.get("directory") == str(cache_path)
            for record in caplog.records
        )
        archive_path.unlink()

    def test_registry_archive_strips_only_wrapper_shaped_prefixes(
        self, manager: ComfyUIManager
    ) -> None:
        assert (
            manager._shared_archive_prefix(
                ["__init__.py"], node_name="comfyui-kjnodes", version="1.5.0"
            )
            is None
        )
        assert (
            manager._shared_archive_prefix(
                ["comfyui-kjnodes-1.5.0/__init__.py"],
                node_name="comfyui-kjnodes",
                version="1.5.0",
            )
            == "comfyui-kjnodes-1.5.0"
        )
        assert (
            manager._shared_archive_prefix(
                ["nodes/__init__.py"], node_name="whatever", version="1.5.0"
            )
            is None
        )
        assert (
            manager._shared_archive_prefix(
                ["one/__init__.py", "two/node.py"],
                node_name="one",
                version="1.5.0",
            )
            is None
        )

    def test_registry_archive_restores_safe_unix_permissions(
        self, manager: ComfyUIManager, comfyui_path: Path
    ) -> None:
        archive_path = comfyui_path / "registry.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            executable = zipfile.ZipInfo("run.sh")
            executable.create_system = 3
            executable.external_attr = 0o755 << 16
            archive.writestr(executable, b"#!/bin/sh\n")
            privileged = zipfile.ZipInfo("privileged.sh")
            privileged.create_system = 3
            privileged.external_attr = 0o4755 << 16
            archive.writestr(privileged, b"#!/bin/sh\n")
            dos_member = zipfile.ZipInfo("windows.bat")
            dos_member.create_system = 0
            dos_member.external_attr = 0o755 << 16
            archive.writestr(dos_member, b"echo off\n")

        node_dir = comfyui_path / "custom_nodes" / "kjnodes"
        node_dir.parent.mkdir()
        manager._extract_registry_archive(archive_path, node_dir, "1.5.0")

        assert node_dir.joinpath("run.sh").stat().st_mode & 0o777 == 0o755
        assert node_dir.joinpath("privileged.sh").stat().st_mode & stat.S_ISUID == 0
        assert node_dir.joinpath("privileged.sh").stat().st_mode & 0o777 == 0o755
        assert node_dir.joinpath("windows.bat").stat().st_mode & 0o111 == 0

    async def test_invalid_registry_node_name_cannot_escape_or_delete_comfyui(
        self, manager: ComfyUIManager, comfyui_path: Path
    ) -> None:
        models_dir = comfyui_path / "models"
        models_dir.mkdir()
        models_dir.joinpath("model.safetensors").write_text("keep")
        custom_nodes_dir = comfyui_path / "custom_nodes"
        custom_nodes_dir.mkdir()
        custom_nodes_dir.joinpath("IMPORTANT.txt").write_text("keep")
        node = CustomNodeConfig.model_construct(
            name="..",
            source="registry",
            node_id="comfyui-kjnodes",
            version="1.5.0",
            archive_sha256=None,
            pip_requirements=[],
        )

        with (
            patch("httpx.AsyncClient") as mock_ctor,
            pytest.raises(ComfyUIError, match="outside ComfyUI custom_nodes"),
        ):
            await manager.install_custom_node(node)

        mock_ctor.assert_not_called()
        assert models_dir.joinpath("model.safetensors").read_text() == "keep"
        assert custom_nodes_dir.joinpath("IMPORTANT.txt").read_text() == "keep"

    def test_registry_archive_refuses_destination_outside_custom_nodes_before_removal(
        self, manager: ComfyUIManager, comfyui_path: Path
    ) -> None:
        archive_path = comfyui_path / "registry.zip"
        archive_path.write_bytes(_build_zip({"__init__.py": b"# node\n"}))
        expected_custom_nodes = comfyui_path / "custom_nodes"
        expected_custom_nodes.mkdir()
        outside_node_dir = comfyui_path / "outside" / "kjnodes"
        outside_node_dir.mkdir(parents=True)
        outside_node_dir.joinpath("IMPORTANT.txt").write_text("keep")

        with pytest.raises(ComfyUIError, match="outside ComfyUI custom_nodes"):
            manager._extract_registry_archive(archive_path, outside_node_dir, "1.5.0")

        assert outside_node_dir.joinpath("IMPORTANT.txt").read_text() == "keep"
        assert not list(expected_custom_nodes.iterdir())

    def test_registry_archive_rejects_declared_uncompressed_zip_bomb_before_extracting(
        self, manager: ComfyUIManager, comfyui_path: Path
    ) -> None:
        archive_path = comfyui_path / "registry.zip"
        archive_path.write_bytes(_build_zip({"large.bin": b"x" * 11}))
        custom_nodes_dir = comfyui_path / "custom_nodes"
        custom_nodes_dir.mkdir()

        with (
            patch.object(comfyui_module, "_REGISTRY_ARCHIVE_MAX_UNCOMPRESSED_BYTES", 10),
            pytest.raises(
                ComfyUIError, match=r"declares 11 uncompressed bytes, exceeding the cap of 10"
            ),
        ):
            manager._extract_registry_archive(archive_path, custom_nodes_dir / "kjnodes", "1.5.0")

        assert not list(custom_nodes_dir.iterdir())

    def test_registry_archive_stops_when_actual_content_exceeds_uncompressed_cap(
        self, manager: ComfyUIManager, comfyui_path: Path
    ) -> None:
        archive_path = comfyui_path / "registry.zip"
        archive_path.write_bytes(_build_zip({"small.bin": b"x"}))
        custom_nodes_dir = comfyui_path / "custom_nodes"
        custom_nodes_dir.mkdir()
        original_open = zipfile.ZipFile.open

        def oversized_open(
            archive: zipfile.ZipFile,
            name: str | zipfile.ZipInfo,
            mode: str = "r",
            pwd: bytes | None = None,
            *,
            force_zip64: bool = False,
        ) -> io.BytesIO:
            if mode == "r":
                return io.BytesIO(b"x" * 11)
            return original_open(archive, name, mode, pwd, force_zip64=force_zip64)  # type: ignore[return-value]

        with (
            patch.object(zipfile.ZipFile, "open", new=oversized_open),
            patch.object(comfyui_module, "_REGISTRY_ARCHIVE_MAX_UNCOMPRESSED_BYTES", 10),
            pytest.raises(ComfyUIError, match="wrote more than 10 uncompressed bytes"),
        ):
            manager._extract_registry_archive(archive_path, custom_nodes_dir / "kjnodes", "1.5.0")

        assert not list(custom_nodes_dir.iterdir())

    def test_registry_archive_rejects_excessive_member_count(
        self, manager: ComfyUIManager, comfyui_path: Path
    ) -> None:
        archive_path = comfyui_path / "registry.zip"
        archive_path.write_bytes(_build_zip({f"node-{index}.py": b"x" for index in range(4)}))
        custom_nodes_dir = comfyui_path / "custom_nodes"
        custom_nodes_dir.mkdir()

        with (
            patch.object(comfyui_module, "_REGISTRY_ARCHIVE_MAX_MEMBERS", 3),
            pytest.raises(ComfyUIError, match="has 4 members, exceeding the cap of 3"),
        ):
            manager._extract_registry_archive(archive_path, custom_nodes_dir / "kjnodes", "1.5.0")

        assert not list(custom_nodes_dir.iterdir())

    def test_registry_archive_rejects_symlink_members(
        self, manager: ComfyUIManager, comfyui_path: Path
    ) -> None:
        archive_path = comfyui_path / "registry.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            symlink = zipfile.ZipInfo("linked-file")
            symlink.create_system = 3
            symlink.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(symlink, b"/etc/passwd")
        custom_nodes_dir = comfyui_path / "custom_nodes"
        custom_nodes_dir.mkdir()

        with pytest.raises(ComfyUIError, match="symlink member: 'linked-file'"):
            manager._extract_registry_archive(archive_path, custom_nodes_dir / "kjnodes", "1.5.0")

        assert not list(custom_nodes_dir.iterdir())


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
