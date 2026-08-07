"""Tests for HfXetTransport (Phase 2a, C3)."""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_content_service.config import Settings
from ai_content_service.download_auth import CredentialEgressError
from ai_content_service.download_transport import (
    TransportFetchError,
    TransportRequest,
    TransportUnavailableError,
)
from ai_content_service.hf_xet_transport import HfXetTransport, _dir_size, _parse_hf_url

_HF_ENV_KEYS = (
    "HF_HOME",
    "HF_XET_HIGH_PERFORMANCE",
    "HF_XET_NUM_CONCURRENT_RANGE_GETS",
    "HF_HUB_DISABLE_PROGRESS_BARS",
)


@pytest.fixture(autouse=True)
def _restore_hf_env():
    """HfXetTransport.__init__ mutates the real process environment (by design --
    some hf_xet builds read these at import time). Snapshot and restore so tests
    do not leak env state into each other."""
    saved = {k: os.environ.get(k) for k in _HF_ENV_KEYS}
    yield
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    overrides.setdefault("hf_cache_path", tmp_path / "hf-home")
    return Settings(**overrides)  # type: ignore[arg-type]


def _transport(tmp_path: Path, **overrides: object) -> HfXetTransport:
    return HfXetTransport(_settings(tmp_path, **overrides))


# ---------------------------------------------------------------------------
# URL parsing
# ---------------------------------------------------------------------------


class TestParseHfUrl:
    def test_model_form(self) -> None:
        parsed = _parse_hf_url("https://huggingface.co/owner/repo/resolve/main/model.safetensors")
        assert parsed is not None
        assert parsed.repo_type == "model"
        assert parsed.repo_id == "owner/repo"
        assert parsed.revision == "main"
        assert parsed.path_in_repo == "model.safetensors"

    def test_dataset_form(self) -> None:
        parsed = _parse_hf_url(
            "https://huggingface.co/datasets/owner/repo/resolve/main/data/train.parquet"
        )
        assert parsed is not None
        assert parsed.repo_type == "dataset"
        assert parsed.repo_id == "owner/repo"
        assert parsed.revision == "main"
        assert parsed.path_in_repo == "data/train.parquet"

    def test_nested_path_with_slashes(self) -> None:
        parsed = _parse_hf_url(
            "https://huggingface.co/owner/repo/resolve/abc123/onnx/model_fp16.onnx"
        )
        assert parsed is not None
        assert parsed.path_in_repo == "onnx/model_fp16.onnx"

    def test_revision_is_not_rewritten_to_main(self) -> None:
        """Pitfall: silently rewriting revision changes which weights are fetched."""
        parsed = _parse_hf_url(
            "https://huggingface.co/owner/repo/resolve/a1b2c3d4/model.safetensors"
        )
        assert parsed is not None
        assert parsed.revision == "a1b2c3d4"

    def test_query_string_is_stripped(self) -> None:
        parsed = _parse_hf_url(
            "https://huggingface.co/owner/repo/resolve/main/model.safetensors?download=true"
        )
        assert parsed is not None
        assert parsed.path_in_repo == "model.safetensors"

    def test_page_url_returns_none(self) -> None:
        assert _parse_hf_url("https://huggingface.co/owner/repo") is None

    def test_tree_url_returns_none(self) -> None:
        assert _parse_hf_url("https://huggingface.co/owner/repo/tree/main") is None


# ---------------------------------------------------------------------------
# can_handle
# ---------------------------------------------------------------------------


class TestCanHandle:
    def test_model_resolve_form_accepted(self, tmp_path: Path) -> None:
        t = _transport(tmp_path)
        assert t.can_handle("https://huggingface.co/owner/repo/resolve/main/model.safetensors")

    def test_dataset_resolve_form_accepted(self, tmp_path: Path) -> None:
        t = _transport(tmp_path)
        assert t.can_handle("https://huggingface.co/datasets/owner/repo/resolve/main/d.json")

    def test_hf_co_shortlink_accepted(self, tmp_path: Path) -> None:
        t = _transport(tmp_path)
        assert t.can_handle("https://hf.co/owner/repo/resolve/main/model.safetensors")

    def test_page_url_rejected(self, tmp_path: Path) -> None:
        t = _transport(tmp_path)
        assert not t.can_handle("https://huggingface.co/owner/repo")

    def test_non_hf_host_rejected(self, tmp_path: Path) -> None:
        t = _transport(tmp_path)
        assert not t.can_handle("https://civitai.com/api/download/models/123")

    def test_lookalike_domain_rejected(self, tmp_path: Path) -> None:
        t = _transport(tmp_path)
        assert not t.can_handle(
            "https://huggingface.co.evil.com/owner/repo/resolve/main/model.safetensors"
        )

    def test_subdomain_of_configured_domain_accepted(self, tmp_path: Path) -> None:
        t = _transport(tmp_path, hf_domains="hf-mirror.internal.example.com")
        assert t.can_handle(
            "https://cdn.hf-mirror.internal.example.com/owner/repo/resolve/main/f.bin"
        )

    def test_single_label_domain_config_rejected(self, tmp_path: Path) -> None:
        """L2: a single-label entry would make every host under that suffix eligible."""
        with pytest.raises(Exception, match="invalid hf domain"):
            _settings(tmp_path, hf_domains="co")


# ---------------------------------------------------------------------------
# probe_digest
# ---------------------------------------------------------------------------


def _sibling(rfilename: str, sha256: str | None) -> MagicMock:
    sib = MagicMock()
    sib.rfilename = rfilename
    sib.lfs = {"sha256": sha256} if sha256 is not None else None
    return sib


class TestProbeDigest:
    URL = "https://huggingface.co/owner/repo/resolve/main/model.safetensors"

    async def test_returns_lfs_sha256(self, tmp_path: Path) -> None:
        t = _transport(tmp_path)
        info = MagicMock()
        info.siblings = [_sibling("model.safetensors", "a" * 64)]
        api = MagicMock()
        api.model_info.return_value = info

        with patch("ai_content_service.hf_xet_transport.HfApi", return_value=api):
            digest = await t.probe_digest(self.URL)

        assert digest == "a" * 64

    async def test_dataset_url_calls_dataset_info(self, tmp_path: Path) -> None:
        t = _transport(tmp_path)
        url = "https://huggingface.co/datasets/owner/repo/resolve/main/data.parquet"
        info = MagicMock()
        info.siblings = [_sibling("data.parquet", "b" * 64)]
        api = MagicMock()
        api.dataset_info.return_value = info

        with patch("ai_content_service.hf_xet_transport.HfApi", return_value=api):
            digest = await t.probe_digest(url)

        assert digest == "b" * 64
        api.dataset_info.assert_called_once()
        api.model_info.assert_not_called()

    async def test_non_lfs_file_returns_none(self, tmp_path: Path) -> None:
        t = _transport(tmp_path)
        info = MagicMock()
        info.siblings = [_sibling("model.safetensors", None)]
        api = MagicMock()
        api.model_info.return_value = info

        with patch("ai_content_service.hf_xet_transport.HfApi", return_value=api):
            digest = await t.probe_digest(self.URL)

        assert digest is None

    async def test_missing_entry_returns_none(self, tmp_path: Path) -> None:
        t = _transport(tmp_path)
        info = MagicMock()
        info.siblings = [_sibling("other-file.bin", "a" * 64)]
        api = MagicMock()
        api.model_info.return_value = info

        with patch("ai_content_service.hf_xet_transport.HfApi", return_value=api):
            digest = await t.probe_digest(self.URL)

        assert digest is None

    async def test_api_error_returns_none_not_raise(self, tmp_path: Path) -> None:
        t = _transport(tmp_path)
        api = MagicMock()
        api.model_info.side_effect = RuntimeError("boom")

        with patch("ai_content_service.hf_xet_transport.HfApi", return_value=api):
            digest = await t.probe_digest(self.URL)

        assert digest is None

    async def test_non_resolve_url_returns_none_without_api_call(self, tmp_path: Path) -> None:
        t = _transport(tmp_path)
        with patch("ai_content_service.hf_xet_transport.HfApi") as api_cls:
            digest = await t.probe_digest("https://huggingface.co/owner/repo")
        assert digest is None
        api_cls.assert_not_called()

    async def test_mirror_host_is_used_as_endpoint(self, tmp_path: Path) -> None:
        """A configured mirror must actually be queried -- not silently
        substituted for the real huggingface.co."""
        t = _transport(tmp_path, hf_domains="hf-mirror.internal.example.com")
        url = "https://hf-mirror.internal.example.com/owner/repo/resolve/main/model.safetensors"
        info = MagicMock()
        info.siblings = [_sibling("model.safetensors", "a" * 64)]
        api_cls = MagicMock()
        api_cls.return_value.model_info.return_value = info

        with patch("ai_content_service.hf_xet_transport.HfApi", api_cls):
            digest = await t.probe_digest(url)

        assert digest == "a" * 64
        api_cls.assert_called_once_with(
            endpoint="https://hf-mirror.internal.example.com", token=None
        )

    async def test_host_outside_policy_is_not_queried(self, tmp_path: Path) -> None:
        """`_parse_hf_url` only looks at the path, so a resolve-shaped URL on
        a host outside `hf_domains` must still be refused -- probe_digest's
        no-raise contract means this surfaces as None, not an exception."""
        t = _transport(tmp_path, hf_token="secret-hf-token")
        url = "https://not-configured.example.com/owner/repo/resolve/main/model.safetensors"

        with patch("ai_content_service.hf_xet_transport.HfApi") as api_cls:
            digest = await t.probe_digest(url)

        assert digest is None
        api_cls.assert_not_called()


# ---------------------------------------------------------------------------
# fetch: atomic destination, temp cleanup, availability
# ---------------------------------------------------------------------------


def _fake_hf_hub_download(content: bytes = b"weights") -> MagicMock:
    def _download(*, filename: str, local_dir: Path, **_kwargs: object) -> str:
        dest = Path(local_dir) / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
        return str(dest)

    return MagicMock(side_effect=_download)


class TestFetch:
    URL = "https://huggingface.co/owner/repo/resolve/main/model.safetensors"

    def _request(self, tmp_path: Path) -> TransportRequest:
        return TransportRequest(
            url=self.URL,
            destination=tmp_path / "models" / "checkpoints" / "model.safetensors",
            expected_sha256=None,
            expected_size=None,
        )

    async def test_success_moves_file_into_place_and_removes_temp(self, tmp_path: Path) -> None:
        t = _transport(tmp_path)
        request = self._request(tmp_path)
        request.destination.parent.mkdir(parents=True, exist_ok=True)
        temp_dir = request.destination.with_name(f"{request.destination.name}.hfxet")

        with patch(
            "ai_content_service.hf_xet_transport.hf_hub_download",
            _fake_hf_hub_download(b"weights-data"),
        ):
            result = await t.fetch(request, None)

        assert result.transport == "hf_xet"
        assert result.bytes_written == len(b"weights-data")
        assert request.destination.read_bytes() == b"weights-data"
        assert not temp_dir.exists()

    async def test_mirror_host_is_used_as_endpoint(self, tmp_path: Path) -> None:
        """A configured mirror must actually be fetched from -- not silently
        substituted for the real huggingface.co."""
        t = _transport(tmp_path, hf_domains="hf-mirror.internal.example.com")
        url = "https://hf-mirror.internal.example.com/owner/repo/resolve/main/model.safetensors"
        request = TransportRequest(
            url=url,
            destination=tmp_path / "models" / "checkpoints" / "model.safetensors",
            expected_sha256=None,
            expected_size=None,
        )
        request.destination.parent.mkdir(parents=True, exist_ok=True)
        download = _fake_hf_hub_download(b"weights-data")

        with patch("ai_content_service.hf_xet_transport.hf_hub_download", download):
            await t.fetch(request, None)

        assert download.call_args.kwargs["endpoint"] == "https://hf-mirror.internal.example.com"

    async def test_host_outside_policy_raises_without_calling_hf_hub_download(
        self, tmp_path: Path
    ) -> None:
        """`_parse_hf_url` only looks at the path, so a resolve-shaped URL on
        a host outside `hf_domains` must still be refused before the token
        is ever handed to hf_hub_download -- callers that skip
        `select_transport`/`can_handle` must not be able to route the HF
        token to an arbitrary host."""
        t = _transport(tmp_path, hf_token="secret-hf-token")
        url = "https://not-configured.example.com/owner/repo/resolve/main/model.safetensors"
        request = TransportRequest(
            url=url,
            destination=tmp_path / "models" / "checkpoints" / "model.safetensors",
            expected_sha256=None,
            expected_size=None,
        )
        request.destination.parent.mkdir(parents=True, exist_ok=True)

        with (
            patch("ai_content_service.hf_xet_transport.hf_hub_download") as download,
            pytest.raises(CredentialEgressError),
        ):
            await t.fetch(request, None)

        download.assert_not_called()

    async def test_nested_repo_path_is_resolved_from_return_value(self, tmp_path: Path) -> None:
        """Pitfall: local_dir reproduces the repo path -- must not assume
        `<temp>/<destination.name>`."""
        url = "https://huggingface.co/owner/repo/resolve/main/onnx/model.onnx"
        request = TransportRequest(
            url=url,
            destination=tmp_path / "models" / "checkpoints" / "renamed.onnx",
            expected_sha256=None,
            expected_size=None,
        )
        request.destination.parent.mkdir(parents=True, exist_ok=True)
        t = _transport(tmp_path)

        with patch(
            "ai_content_service.hf_xet_transport.hf_hub_download",
            _fake_hf_hub_download(b"onnx-bytes"),
        ):
            result = await t.fetch(request, None)

        assert result.bytes_written == len(b"onnx-bytes")
        assert request.destination.read_bytes() == b"onnx-bytes"

    async def test_failure_removes_temp_and_leaves_no_destination(self, tmp_path: Path) -> None:
        t = _transport(tmp_path)
        request = self._request(tmp_path)
        request.destination.parent.mkdir(parents=True, exist_ok=True)
        temp_dir = request.destination.with_name(f"{request.destination.name}.hfxet")

        with (
            patch(
                "ai_content_service.hf_xet_transport.hf_hub_download",
                MagicMock(side_effect=RuntimeError("network exploded")),
            ),
            pytest.raises(TransportFetchError),
        ):
            await t.fetch(request, None)

        assert not temp_dir.exists()
        assert not request.destination.exists()

    async def test_cancellation_removes_temp_and_propagates(self, tmp_path: Path) -> None:
        t = _transport(tmp_path)
        request = self._request(tmp_path)
        request.destination.parent.mkdir(parents=True, exist_ok=True)
        temp_dir = request.destination.with_name(f"{request.destination.name}.hfxet")

        with (
            patch(
                "ai_content_service.hf_xet_transport.hf_hub_download",
                MagicMock(side_effect=asyncio.CancelledError()),
            ),
            pytest.raises(asyncio.CancelledError),
        ):
            await t.fetch(request, None)

        assert not temp_dir.exists()

    async def test_xet_load_failure_raises_transport_unavailable(self, tmp_path: Path) -> None:
        """A native load failure (glibc mismatch, unsupported arch) is probed
        once at construction and must not fall through to hf_hub_download,
        which would silently use its own httpx path and hide the regression."""
        with patch(
            "ai_content_service.hf_xet_transport._probe_xet_load",
            return_value="libc.so.6: version `GLIBC_2.35' not found",
        ):
            t = _transport(tmp_path)
        request = self._request(tmp_path)

        with (
            patch("ai_content_service.hf_xet_transport.hf_hub_download") as download,
            pytest.raises(TransportUnavailableError, match=r"GLIBC_2\.35"),
        ):
            await t.fetch(request, None)

        download.assert_not_called()

    async def test_progress_callback_invoked_during_fetch(self, tmp_path: Path) -> None:
        import ai_content_service.hf_xet_transport as hf_xet_transport_module

        with patch.object(hf_xet_transport_module, "_PROGRESS_POLL_INTERVAL_S", 0.01):
            t = _transport(tmp_path)
            request = self._request(tmp_path)
            request.destination.parent.mkdir(parents=True, exist_ok=True)

            def _slow_download(*, filename: str, local_dir: Path, **_kwargs: object) -> str:
                dest = Path(local_dir) / filename
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(b"a")
                time.sleep(0.05)
                dest.write_bytes(b"aa")
                return str(dest)

            on_progress = AsyncMock()
            with patch(
                "ai_content_service.hf_xet_transport.hf_hub_download",
                MagicMock(side_effect=_slow_download),
            ):
                await t.fetch(request, on_progress)

        assert on_progress.await_count >= 1

    async def test_hf_hub_download_runs_off_the_event_loop(self, tmp_path: Path) -> None:
        """Regression guard: hf_hub_download is synchronous. Called directly on
        the event loop it would serialise concurrent fetches (25x -> ~1x)."""
        t = _transport(tmp_path)

        def _blocking_download(*, filename: str, local_dir: Path, **_kwargs: object) -> str:
            time.sleep(0.2)
            dest = Path(local_dir) / filename
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"x")
            return str(dest)

        def _request(name: str) -> TransportRequest:
            dest = tmp_path / "models" / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            return TransportRequest(
                url=self.URL, destination=dest, expected_sha256=None, expected_size=None
            )

        with patch(
            "ai_content_service.hf_xet_transport.hf_hub_download",
            MagicMock(side_effect=_blocking_download),
        ):
            start = time.monotonic()
            await asyncio.gather(
                t.fetch(_request("a.safetensors"), None),
                t.fetch(_request("b.safetensors"), None),
            )
            elapsed = time.monotonic() - start

        assert elapsed < 0.35, f"expected overlap, took {elapsed:.2f}s for two 0.2s fetches"


# ---------------------------------------------------------------------------
# Environment set once at construction
# ---------------------------------------------------------------------------


class TestEnvironment:
    def test_env_vars_reach_process_environment(self, tmp_path: Path) -> None:
        _transport(tmp_path, hf_xet_concurrent_range_gets=16)

        assert os.environ["HF_XET_HIGH_PERFORMANCE"] == "1"
        assert os.environ["HF_XET_NUM_CONCURRENT_RANGE_GETS"] == "16"
        assert os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] == "1"
        assert os.environ["HF_HOME"] == str(tmp_path / "hf-home")

    def test_hf_home_directory_is_created(self, tmp_path: Path) -> None:
        hf_home = tmp_path / "custom-hf-home"
        _transport(tmp_path, hf_cache_path=hf_home)
        assert hf_home.is_dir()

    def test_default_hf_home_under_cache_path(self, tmp_path: Path) -> None:
        settings = Settings(cache_path=tmp_path / "cache")
        HfXetTransport(settings)
        assert os.environ["HF_HOME"] == str(tmp_path / "cache" / "hf")
        assert (tmp_path / "cache" / "hf").is_dir()

    def test_unwritable_hf_home_does_not_raise(self, tmp_path: Path) -> None:
        """L6: constructing the transport must never fail deployment -- a node
        whose cache_path isn't writable yet still deploys, and every later
        fetch degrades to the next candidate on its own."""
        settings = Settings(hf_cache_path=tmp_path / "hf-home")
        with patch.object(Path, "mkdir", side_effect=OSError("read-only file system")):
            HfXetTransport(settings)  # must not raise


# ---------------------------------------------------------------------------
# _dir_size helper
# ---------------------------------------------------------------------------


class TestDirSize:
    def test_sums_nested_files(self, tmp_path: Path) -> None:
        (tmp_path / "a.bin").write_bytes(b"12345")
        nested = tmp_path / "sub"
        nested.mkdir()
        (nested / "b.bin").write_bytes(b"1234567890")

        assert _dir_size(tmp_path) == 15

    def test_missing_directory_returns_zero(self, tmp_path: Path) -> None:
        assert _dir_size(tmp_path / "does-not-exist") == 0
