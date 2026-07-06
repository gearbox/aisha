"""Tests for bundle registry module."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from ai_content_service.bundle_registry import (
    BundleIndex,
    BundleIndexEntry,
    BundleReference,
    BundleRegistryManager,
    GitBundleRegistry,
    LocalBundleRegistry,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_dir() -> Iterator[Path]:
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


def _make_bundle_dir(root: Path, name: str, version: str, with_current: bool = True) -> Path:
    """Create a minimal bundle version directory under root/<name>/<version>/."""
    version_path = root / name / version
    version_path.mkdir(parents=True, exist_ok=True)
    (version_path / "bundle.yaml").write_text(f"name: {name}\nversion: {version}\n")
    if with_current:
        current = root / name / "current"
        if current.exists() or current.is_symlink():
            current.unlink()
        current.symlink_to(version)
    return version_path


# ---------------------------------------------------------------------------
# BundleReference
# ---------------------------------------------------------------------------


class TestBundleReferenceParse:
    def test_name_only(self) -> None:
        ref = BundleReference.parse("my_bundle")
        assert ref.name == "my_bundle"
        assert ref.version is None
        assert ref.registry is None

    def test_name_and_version(self) -> None:
        ref = BundleReference.parse("my_bundle:1.0")
        assert ref.name == "my_bundle"
        assert ref.version == "1.0"
        assert ref.registry is None

    def test_registry_and_name(self) -> None:
        ref = BundleReference.parse("myreg/my_bundle")
        assert ref.name == "my_bundle"
        assert ref.version is None
        assert ref.registry == "myreg"

    def test_all_fields(self) -> None:
        ref = BundleReference.parse("myreg/my_bundle:2.5")
        assert ref.name == "my_bundle"
        assert ref.version == "2.5"
        assert ref.registry == "myreg"


# ---------------------------------------------------------------------------
# BundleIndexEntry
# ---------------------------------------------------------------------------


class TestBundleIndexEntry:
    def test_from_dict_full(self) -> None:
        data = {
            "name": "b1",
            "path": "bundles/b1",
            "description": "desc",
            "tags": ["tag1"],
            "default_version": "v1",
        }
        entry = BundleIndexEntry.from_dict(data)
        assert entry.name == "b1"
        assert entry.path == "bundles/b1"
        assert entry.description == "desc"
        assert entry.tags == ["tag1"]
        assert entry.default_version == "v1"

    def test_from_dict_minimal(self) -> None:
        entry = BundleIndexEntry.from_dict({"name": "b2", "path": "bundles/b2"})
        assert entry.description == ""
        assert entry.tags is None
        assert entry.default_version is None


# ---------------------------------------------------------------------------
# BundleIndex
# ---------------------------------------------------------------------------


class TestBundleIndex:
    def test_from_yaml(self) -> None:
        content = yaml.dump(
            {
                "version": "2",
                "bundles": [
                    {"name": "alpha", "path": "bundles/alpha"},
                    {"name": "beta", "path": "bundles/beta", "description": "b"},
                ],
            }
        )
        idx = BundleIndex.from_yaml(content)
        assert idx.version == "2"
        assert len(idx.bundles) == 2

    def test_from_yaml_defaults(self) -> None:
        idx = BundleIndex.from_yaml(yaml.dump({}))
        assert idx.version == "1"
        assert idx.bundles == []

    def test_find_existing(self) -> None:
        idx = BundleIndex(bundles=[BundleIndexEntry(name="x", path="p/x")])
        entry = idx.find("x")
        assert entry is not None
        assert entry.name == "x"

    def test_find_missing(self) -> None:
        idx = BundleIndex(bundles=[])
        assert idx.find("nope") is None


# ---------------------------------------------------------------------------
# LocalBundleRegistry
# ---------------------------------------------------------------------------


class TestLocalBundleRegistry:
    def test_name_and_path(self, tmp_dir: Path) -> None:
        reg = LocalBundleRegistry(tmp_dir, name="mylocal")
        assert reg.name == "mylocal"
        assert reg.path == tmp_dir

    def test_sync_invalidates_cache(self, tmp_dir: Path) -> None:
        reg = LocalBundleRegistry(tmp_dir)
        reg._index = MagicMock()  # inject a fake cached index
        asyncio.get_event_loop().run_until_complete(reg.sync())
        assert reg._index is None

    def test_get_index_from_yaml_file(self, tmp_dir: Path) -> None:
        index_content = yaml.dump({"bundles": [{"name": "b1", "path": "bundles/b1"}]})
        (tmp_dir / "bundle-index.yaml").write_text(index_content)

        idx = asyncio.get_event_loop().run_until_complete(LocalBundleRegistry(tmp_dir).get_index())
        assert idx.find("b1") is not None

    def test_get_index_cached(self, tmp_dir: Path) -> None:
        reg = LocalBundleRegistry(tmp_dir)
        fake = BundleIndex(bundles=[])
        reg._index = fake
        idx = asyncio.get_event_loop().run_until_complete(reg.get_index())
        assert idx is fake

    def test_get_index_auto_discover(self, tmp_dir: Path) -> None:
        _make_bundle_dir(tmp_dir, "wan", "v1")
        idx = asyncio.get_event_loop().run_until_complete(LocalBundleRegistry(tmp_dir).get_index())
        assert idx.find("wan") is not None

    def test_get_index_auto_discover_bundles_subdir(self, tmp_dir: Path) -> None:
        bundles_dir = tmp_dir / "bundles"
        bundles_dir.mkdir()
        _make_bundle_dir(bundles_dir, "wan", "v1")
        idx = asyncio.get_event_loop().run_until_complete(LocalBundleRegistry(tmp_dir).get_index())
        assert idx.find("wan") is not None

    def test_resolve_via_current_symlink(self, tmp_dir: Path) -> None:
        version_path = _make_bundle_dir(tmp_dir, "wan", "v1", with_current=True)
        reg = LocalBundleRegistry(tmp_dir)
        result = asyncio.get_event_loop().run_until_complete(reg.resolve_bundle_path("wan"))
        assert result == version_path.resolve()

    def test_resolve_specific_version(self, tmp_dir: Path) -> None:
        _make_bundle_dir(tmp_dir, "wan", "v1")
        _make_bundle_dir(tmp_dir, "wan", "v2", with_current=False)
        reg = LocalBundleRegistry(tmp_dir)
        result = asyncio.get_event_loop().run_until_complete(reg.resolve_bundle_path("wan", "v2"))
        assert result == tmp_dir / "wan" / "v2"

    def test_resolve_uses_default_version(self, tmp_dir: Path) -> None:
        _make_bundle_dir(tmp_dir, "wan", "v1", with_current=False)
        # Write an index that sets default_version
        index_content = yaml.dump(
            {"bundles": [{"name": "wan", "path": "wan", "default_version": "v1"}]}
        )
        (tmp_dir / "bundle-index.yaml").write_text(index_content)
        reg = LocalBundleRegistry(tmp_dir)
        result = asyncio.get_event_loop().run_until_complete(reg.resolve_bundle_path("wan"))
        assert result == tmp_dir / "wan" / "v1"

    def test_resolve_fallback_to_bundle_dir(self, tmp_dir: Path) -> None:
        bundle_dir = tmp_dir / "flat"
        bundle_dir.mkdir()
        (bundle_dir / "bundle.yaml").write_text("name: flat\n")
        index_content = yaml.dump({"bundles": [{"name": "flat", "path": "flat"}]})
        (tmp_dir / "bundle-index.yaml").write_text(index_content)
        reg = LocalBundleRegistry(tmp_dir)
        result = asyncio.get_event_loop().run_until_complete(reg.resolve_bundle_path("flat"))
        assert result == bundle_dir

    def test_resolve_bundle_not_found(self, tmp_dir: Path) -> None:
        reg = LocalBundleRegistry(tmp_dir)
        with pytest.raises(ValueError, match="not found in registry"):
            asyncio.get_event_loop().run_until_complete(reg.resolve_bundle_path("ghost"))

    def test_resolve_version_not_found(self, tmp_dir: Path) -> None:
        _make_bundle_dir(tmp_dir, "wan", "v1", with_current=False)
        reg = LocalBundleRegistry(tmp_dir)
        with pytest.raises(ValueError, match="not found for bundle"):
            asyncio.get_event_loop().run_until_complete(reg.resolve_bundle_path("wan", "v99"))

    def test_list_versions(self, tmp_dir: Path) -> None:
        _make_bundle_dir(tmp_dir, "wan", "v1")
        _make_bundle_dir(tmp_dir, "wan", "v2", with_current=False)
        reg = LocalBundleRegistry(tmp_dir)
        versions = asyncio.get_event_loop().run_until_complete(reg.list_versions("wan"))
        assert set(versions) == {"v1", "v2"}

    def test_list_versions_bundle_not_found(self, tmp_dir: Path) -> None:
        reg = LocalBundleRegistry(tmp_dir)
        with pytest.raises(ValueError, match="not found"):
            asyncio.get_event_loop().run_until_complete(reg.list_versions("ghost"))


# ---------------------------------------------------------------------------
# GitBundleRegistry
# ---------------------------------------------------------------------------


def _make_mock_process(returncode: int = 0) -> MagicMock:
    proc = MagicMock()
    proc.returncode = returncode
    proc.communicate = AsyncMock(return_value=(b"", b"error output"))
    return proc


class TestGitBundleRegistry:
    def _reg(self, tmp_dir: Path, **kwargs) -> GitBundleRegistry:
        return GitBundleRegistry(
            repo_url="https://github.com/example/bundles.git",
            local_path=tmp_dir / "repo",
            name="git",
            **kwargs,
        )

    def test_name_and_path(self, tmp_dir: Path) -> None:
        reg = self._reg(tmp_dir)
        assert reg.name == "git"
        assert reg.path == tmp_dir / "repo"

    def test_auth_args_empty_without_token(self, tmp_dir: Path) -> None:
        reg = self._reg(tmp_dir)
        assert reg._auth_args() == []

    def test_auth_args_with_token(self, tmp_dir: Path) -> None:
        reg = GitBundleRegistry(
            repo_url="https://github.com/example/bundles.git",
            local_path=tmp_dir / "repo",
            auth_token="mytoken",
        )
        args = reg._auth_args()
        assert args[0] == "-c"
        assert args[1].startswith("http.extraHeader=Authorization: Basic ")
        assert "mytoken" not in args[1]  # token is base64-encoded, not raw

    def test_git_ssh_command_with_key(self, tmp_dir: Path) -> None:
        key = tmp_dir / "id_rsa"
        reg = GitBundleRegistry(
            repo_url="git@github.com:example/bundles.git",
            local_path=tmp_dir / "repo",
            ssh_key_path=key,
        )
        cmd = reg._get_git_ssh_command()
        assert cmd is not None
        assert str(key) in cmd

    def test_git_ssh_command_without_key(self, tmp_dir: Path) -> None:
        reg = self._reg(tmp_dir)
        assert reg._get_git_ssh_command() is None

    def test_sync_clone(self, tmp_dir: Path) -> None:
        repo_path = tmp_dir / "repo"
        reg = GitBundleRegistry(
            repo_url="https://github.com/example/bundles.git",
            local_path=repo_path,
            name="git",
        )
        mock_proc = _make_mock_process(returncode=0)

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            asyncio.get_event_loop().run_until_complete(reg.sync())
            # First call should be a clone (repo_path didn't exist)
            first_call_args = mock_exec.call_args_list[0][0]
            assert "clone" in first_call_args

        assert reg._local_registry is None  # cache was invalidated

    def test_sync_pull_success(self, tmp_dir: Path) -> None:
        repo_path = tmp_dir / "repo"
        repo_path.mkdir()
        reg = GitBundleRegistry(
            repo_url="https://github.com/example/bundles.git",
            local_path=repo_path,
            name="git",
        )
        mock_proc = _make_mock_process(returncode=0)

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            asyncio.get_event_loop().run_until_complete(reg.sync())

    def test_sync_pull_falls_back_to_fetch_reset(self, tmp_dir: Path) -> None:
        repo_path = tmp_dir / "repo"
        repo_path.mkdir()
        reg = GitBundleRegistry(
            repo_url="https://github.com/example/bundles.git",
            local_path=repo_path,
            name="git",
            branch="main",
        )

        fail_proc = _make_mock_process(returncode=1)
        ok_proc = _make_mock_process(returncode=0)
        call_results = [fail_proc, ok_proc, ok_proc]

        async def fake_exec(*_args, **_kwargs):
            return call_results.pop(0)

        with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
            asyncio.get_event_loop().run_until_complete(reg.sync())

    def test_sync_clone_failure_raises(self, tmp_dir: Path) -> None:
        repo_path = tmp_dir / "repo"
        reg = GitBundleRegistry(
            repo_url="https://github.com/example/bundles.git",
            local_path=repo_path,
            name="git",
        )
        fail_proc = _make_mock_process(returncode=1)

        with (
            patch("asyncio.create_subprocess_exec", return_value=fail_proc),
            pytest.raises(RuntimeError, match="git clone"),
        ):
            asyncio.get_event_loop().run_until_complete(reg.sync())

    def test_sync_with_ssh_key(self, tmp_dir: Path) -> None:
        repo_path = tmp_dir / "repo"
        key_path = tmp_dir / "id_rsa"
        reg = GitBundleRegistry(
            repo_url="git@github.com:example/bundles.git",
            local_path=repo_path,
            name="git",
            ssh_key_path=key_path,
        )
        ok_proc = _make_mock_process(returncode=0)

        with patch("asyncio.create_subprocess_exec", return_value=ok_proc) as mock_exec:
            asyncio.get_event_loop().run_until_complete(reg.sync())
            kwargs = mock_exec.call_args_list[0][1]
            assert "GIT_SSH_COMMAND" in kwargs.get("env", {})

    def test_sync_diverged_branch_awaits_reset_and_succeeds(self, tmp_dir: Path) -> None:
        """Pull fails, fetch+reset succeed — sync() must await every subprocess to completion."""
        repo_path = tmp_dir / "repo"
        repo_path.mkdir()
        reg = GitBundleRegistry(
            repo_url="https://github.com/example/bundles.git",
            local_path=repo_path,
            name="git",
            branch="main",
        )

        fail_proc = _make_mock_process(returncode=1)
        fetch_proc = _make_mock_process(returncode=0)
        reset_proc = _make_mock_process(returncode=0)
        call_results = [fail_proc, fetch_proc, reset_proc]

        async def fake_exec(*_args, **_kwargs):
            return call_results.pop(0)

        with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
            asyncio.get_event_loop().run_until_complete(reg.sync())

        fail_proc.communicate.assert_awaited_once()
        fetch_proc.communicate.assert_awaited_once()
        reset_proc.communicate.assert_awaited_once()

    def test_sync_reset_failure_raises(self, tmp_dir: Path) -> None:
        repo_path = tmp_dir / "repo"
        repo_path.mkdir()
        reg = GitBundleRegistry(
            repo_url="https://github.com/example/bundles.git",
            local_path=repo_path,
            name="git",
            branch="main",
        )

        fail_proc = _make_mock_process(returncode=1)
        fetch_proc = _make_mock_process(returncode=0)
        reset_fail_proc = _make_mock_process(returncode=1)
        call_results = [fail_proc, fetch_proc, reset_fail_proc]

        async def fake_exec(*_args, **_kwargs):
            return call_results.pop(0)

        with (
            patch("asyncio.create_subprocess_exec", side_effect=fake_exec),
            pytest.raises(RuntimeError, match="reset --hard"),
        ):
            asyncio.get_event_loop().run_until_complete(reg.sync())

    def test_sync_fallback_passes_env_to_fetch_and_reset(self, tmp_dir: Path) -> None:
        repo_path = tmp_dir / "repo"
        repo_path.mkdir()
        key_path = tmp_dir / "id_rsa"
        reg = GitBundleRegistry(
            repo_url="https://github.com/example/bundles.git",
            local_path=repo_path,
            name="git",
            branch="main",
            ssh_key_path=key_path,
        )

        fail_proc = _make_mock_process(returncode=1)
        fetch_proc = _make_mock_process(returncode=0)
        reset_proc = _make_mock_process(returncode=0)
        call_results = [fail_proc, fetch_proc, reset_proc]

        async def fake_exec(*_args, **_kwargs):
            return call_results.pop(0)

        with patch("asyncio.create_subprocess_exec", side_effect=fake_exec) as mock_exec:
            asyncio.get_event_loop().run_until_complete(reg.sync())

        for call in mock_exec.call_args_list:
            assert "GIT_SSH_COMMAND" in call.kwargs.get("env", {})

    def test_clone_url_contains_no_token(self, tmp_dir: Path) -> None:
        repo_path = tmp_dir / "repo"
        reg = GitBundleRegistry(
            repo_url="https://github.com/example/bundles.git",
            local_path=repo_path,
            name="git",
            auth_token="ghp_secrettoken",
        )
        ok_proc = _make_mock_process(returncode=0)

        with patch("asyncio.create_subprocess_exec", return_value=ok_proc) as mock_exec:
            asyncio.get_event_loop().run_until_complete(reg.sync())

        clone_args = mock_exec.call_args_list[0][0]
        assert "https://github.com/example/bundles.git" in clone_args
        assert all("ghp_secrettoken" not in arg for arg in clone_args if arg != clone_args[0])
        for arg in clone_args:
            assert "ghp_secrettoken" not in arg or "http.extraHeader" in arg

    def test_git_config_never_contains_token(self, tmp_dir: Path) -> None:
        """The clone URL argument itself must be the clean URL — token only ever
        travels via the -c http.extraHeader flag, never persisted to .git/config."""
        repo_path = tmp_dir / "repo"
        reg = GitBundleRegistry(
            repo_url="https://github.com/example/bundles.git",
            local_path=repo_path,
            name="git",
            auth_token="ghp_secrettoken",
        )
        ok_proc = _make_mock_process(returncode=0)

        with patch("asyncio.create_subprocess_exec", return_value=ok_proc) as mock_exec:
            asyncio.get_event_loop().run_until_complete(reg.sync())

        clone_args = mock_exec.call_args_list[0][0]
        url_arg = clone_args[-2]  # ["git", *auth_args, "clone", ..., url, local_path]
        assert url_arg == "https://github.com/example/bundles.git"

    def test_run_git_error_redacts_token(self, tmp_dir: Path) -> None:
        repo_path = tmp_dir / "repo"
        reg = GitBundleRegistry(
            repo_url="https://github.com/example/bundles.git",
            local_path=repo_path,
            name="git",
            auth_token="ghp_secrettoken",
        )
        fail_proc = MagicMock()
        fail_proc.returncode = 1
        fail_proc.communicate = AsyncMock(
            return_value=(b"", b"fatal: https://ghp_secrettoken@github.com/x not found")
        )

        with (
            patch("asyncio.create_subprocess_exec", return_value=fail_proc),
            pytest.raises(RuntimeError) as exc_info,
        ):
            asyncio.get_event_loop().run_until_complete(reg.sync())

        message = str(exc_info.value)
        assert "ghp_secrettoken" not in message
        assert "https://ghp_secrettoken@" not in message

    def test_run_git_error_redacts_b64_auth_header_in_stderr(self, tmp_dir: Path) -> None:
        """stderr echoing the b64-encoded Authorization header must still be redacted."""
        repo_path = tmp_dir / "repo"
        reg = GitBundleRegistry(
            repo_url="https://github.com/example/bundles.git",
            local_path=repo_path,
            name="git",
            auth_token="ghp_secrettoken",
        )
        b64 = reg._auth_header_b64()
        assert b64 is not None
        fail_proc = MagicMock()
        fail_proc.returncode = 1
        fail_proc.communicate = AsyncMock(
            return_value=(b"", f"fatal: auth header rejected: Basic {b64}".encode())
        )

        with (
            patch("asyncio.create_subprocess_exec", return_value=fail_proc),
            pytest.raises(RuntimeError) as exc_info,
        ):
            asyncio.get_event_loop().run_until_complete(reg.sync())

        message = str(exc_info.value)
        assert "ghp_secrettoken" not in message
        assert b64 not in message

    def test_run_git_error_excludes_auth_header_from_message(self, tmp_dir: Path) -> None:
        """The raised message is built only from caller args + redacted stderr — never
        from the injected auth argv — so no token representation can leak into it."""
        repo_path = tmp_dir / "repo"
        reg = GitBundleRegistry(
            repo_url="https://github.com/example/bundles.git",
            local_path=repo_path,
            name="git",
            auth_token="ghp_secrettoken",
        )
        b64 = reg._auth_header_b64()
        assert b64 is not None
        fail_proc = MagicMock()
        fail_proc.returncode = 1
        fail_proc.communicate = AsyncMock(return_value=(b"", b"fatal: not found"))

        with (
            patch("asyncio.create_subprocess_exec", return_value=fail_proc),
            pytest.raises(RuntimeError) as exc_info,
        ):
            asyncio.get_event_loop().run_until_complete(reg.sync())

        message = str(exc_info.value)
        assert "ghp_secrettoken" not in message
        assert b64 not in message
        assert "http.extraHeader" not in message

    def test_run_git_injects_auth_args_into_argv(self, tmp_dir: Path) -> None:
        """_run_git injects auth flags into the executed argv even though callers
        pass plain args without them."""
        repo_path = tmp_dir / "repo"
        reg = GitBundleRegistry(
            repo_url="https://github.com/example/bundles.git",
            local_path=repo_path,
            name="git",
            auth_token="ghp_secrettoken",
        )
        mock_proc = _make_mock_process(returncode=0)

        caller_args = ["-C", str(repo_path), "status"]
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            asyncio.get_event_loop().run_until_complete(reg._run_git(caller_args))

        executed_args = mock_exec.call_args_list[0][0]
        assert "-c" in executed_args
        assert any("http.extraHeader" in str(arg) for arg in executed_args)
        assert "-c" not in caller_args

    def test_sync_call_sites_pass_plain_args(self, tmp_dir: Path) -> None:
        """pull/fetch/clone call sites must not embed auth flags in their own
        args list — injection happens exclusively inside _run_git."""
        repo_path = tmp_dir / "repo"
        repo_path.mkdir()
        reg = GitBundleRegistry(
            repo_url="https://github.com/example/bundles.git",
            local_path=repo_path,
            name="git",
            branch="main",
            auth_token="ghp_secrettoken",
        )

        original_run_git = reg._run_git
        captured_args: list[list[str]] = []

        async def spy_run_git(args: list[str], *, env: dict[str, str] | None = None) -> None:
            captured_args.append(args)
            await original_run_git(args, env=env)

        fail_proc = _make_mock_process(returncode=1)
        fetch_proc = _make_mock_process(returncode=0)
        reset_proc = _make_mock_process(returncode=0)
        call_results = [fail_proc, fetch_proc, reset_proc]

        async def fake_exec(*_args: object, **_kwargs: object):
            return call_results.pop(0)

        with (
            patch.object(reg, "_run_git", side_effect=spy_run_git),
            patch("asyncio.create_subprocess_exec", side_effect=fake_exec),
        ):
            asyncio.get_event_loop().run_until_complete(reg.sync())

        for args in captured_args:
            assert "-c" not in args
            assert all("http.extraHeader" not in arg for arg in args)

    def test_resolve_without_default_registry_message(self) -> None:
        mgr = BundleRegistryManager()
        ref = BundleReference.parse("b1:v1")
        with pytest.raises(ValueError) as exc_info:
            asyncio.get_event_loop().run_until_complete(mgr.resolve(ref))
        assert "None" not in str(exc_info.value)

    def test_resolve_bundle_path_no_version_message(self, tmp_dir: Path) -> None:
        _make_bundle_dir(tmp_dir, "wan", "v1", with_current=False)
        reg = LocalBundleRegistry(tmp_dir)
        with pytest.raises(ValueError) as exc_info:
            asyncio.get_event_loop().run_until_complete(reg.resolve_bundle_path("wan"))
        assert "None" not in str(exc_info.value)

    def test_delegates_to_local_registry(self, tmp_dir: Path) -> None:
        repo_path = tmp_dir / "repo"
        _make_bundle_dir(repo_path, "b1", "v1")
        reg = GitBundleRegistry(
            repo_url="https://example.com/repo.git",
            local_path=repo_path,
            name="git",
        )
        idx = asyncio.get_event_loop().run_until_complete(reg.get_index())
        assert idx.find("b1") is not None

        result = asyncio.get_event_loop().run_until_complete(reg.resolve_bundle_path("b1", "v1"))
        assert result == repo_path / "b1" / "v1"

        versions = asyncio.get_event_loop().run_until_complete(reg.list_versions("b1"))
        assert "v1" in versions

    def test_local_registry_uses_bundles_subdir(self, tmp_dir: Path) -> None:
        repo_path = tmp_dir / "repo"
        bundles_path = repo_path / "bundles"
        _make_bundle_dir(bundles_path, "b1", "v1")
        reg = GitBundleRegistry(
            repo_url="https://example.com/repo.git",
            local_path=repo_path,
            name="git",
        )
        local = reg._get_local_registry()
        assert local.path == bundles_path


# ---------------------------------------------------------------------------
# BundleRegistryManager
# ---------------------------------------------------------------------------


class TestBundleRegistryManager:
    def _local(self, tmp_dir: Path, name: str = "local") -> LocalBundleRegistry:
        return LocalBundleRegistry(tmp_dir, name=name)

    def test_register_and_get(self, tmp_dir: Path) -> None:
        mgr = BundleRegistryManager()
        reg = self._local(tmp_dir, "r1")
        mgr.register(reg)
        assert mgr.get("r1") is reg

    def test_get_missing(self) -> None:
        mgr = BundleRegistryManager()
        assert mgr.get("nope") is None

    def test_first_registered_becomes_default(self, tmp_dir: Path) -> None:
        mgr = BundleRegistryManager()
        reg = self._local(tmp_dir, "r1")
        mgr.register(reg)
        assert mgr.default is reg

    def test_explicit_default_flag(self, tmp_dir: Path) -> None:
        mgr = BundleRegistryManager()
        r1 = self._local(tmp_dir, "r1")
        r2 = self._local(tmp_dir, "r2")
        mgr.register(r1)
        mgr.register(r2, default=True)
        assert mgr.default is r2

    def test_constructor_default(self, tmp_dir: Path) -> None:
        reg = self._local(tmp_dir, "r1")
        mgr = BundleRegistryManager(default_registry=reg)
        assert mgr.default is reg
        assert mgr.get("r1") is reg

    def test_list_registries(self, tmp_dir: Path) -> None:
        mgr = BundleRegistryManager()
        mgr.register(self._local(tmp_dir, "r1"))
        mgr.register(self._local(tmp_dir, "r2"))
        assert set(mgr.list_registries()) == {"r1", "r2"}

    def test_sync_all(self, tmp_dir: Path) -> None:
        mgr = BundleRegistryManager()
        r1 = self._local(tmp_dir, "r1")
        r2 = self._local(tmp_dir, "r2")
        r1._index = MagicMock()
        r2._index = MagicMock()
        mgr.register(r1)
        mgr.register(r2)
        asyncio.get_event_loop().run_until_complete(mgr.sync_all())
        assert r1._index is None
        assert r2._index is None

    def test_resolve_with_default_registry(self, tmp_dir: Path) -> None:
        _make_bundle_dir(tmp_dir, "b1", "v1")
        mgr = BundleRegistryManager(default_registry=self._local(tmp_dir, "local"))
        ref = BundleReference.parse("b1:v1")
        result = asyncio.get_event_loop().run_until_complete(mgr.resolve(ref))
        assert result == tmp_dir / "b1" / "v1"

    def test_resolve_with_named_registry(self, tmp_dir: Path) -> None:
        _make_bundle_dir(tmp_dir, "b1", "v1")
        mgr = BundleRegistryManager()
        mgr.register(self._local(tmp_dir, "mylocal"))
        ref = BundleReference.parse("mylocal/b1:v1")
        result = asyncio.get_event_loop().run_until_complete(mgr.resolve(ref))
        assert result == tmp_dir / "b1" / "v1"

    def test_resolve_unknown_registry_raises(self) -> None:
        mgr = BundleRegistryManager()
        ref = BundleReference.parse("unknown/b1:v1")
        with pytest.raises(ValueError, match="not found"):
            asyncio.get_event_loop().run_until_complete(mgr.resolve(ref))

    def test_resolve_no_default_raises(self) -> None:
        mgr = BundleRegistryManager()
        ref = BundleReference.parse("b1:v1")
        with pytest.raises(ValueError):
            asyncio.get_event_loop().run_until_complete(mgr.resolve(ref))
