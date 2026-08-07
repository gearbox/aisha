"""HuggingFace transport backed by `hf_xet`.

HuggingFace's Xet-backed `/resolve/` endpoint performs badly over a single
httpx connection (measured 16.2 MB/s). `hf_xet` talks to the Xet CAS instead
and reaches 300-400+ MB/s on the same node. This module never decides *when*
to run -- `can_handle` answers that from `Settings.hf_domains`, and the
composition root (`downloader.build_transports`) decides whether to register
it at all.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import structlog

from .config import unwrap_secret
from .download_auth import (
    BoundCredential,
    assert_no_credential_egress,
    build_credentials,
    build_huggingface_policy,
    redact_url,
)
from .download_transport import (
    TransportFetchError,
    TransportRequest,
    TransportResult,
    TransportUnavailableError,
)

if TYPE_CHECKING:
    # Real runtime binding happens in HfXetTransport.__init__, deferred until
    # after HF_HOME et al. are set on the environment (see comment there).
    # Ruff's static analysis can't see that -- it only sees HfApi/hf_hub_download
    # called below and wants the import promoted out of TYPE_CHECKING, which
    # would reintroduce the premature-import bug this guards against.
    from huggingface_hub import HfApi, hf_hub_download  # noqa: TC004

    from .config import Settings
    from .download_transport import ProgressCallback

log = structlog.get_logger()

_PROGRESS_POLL_INTERVAL_S = 2.0

_ENV_HF_HOME = "HF_HOME"
_ENV_XET_HIGH_PERFORMANCE = "HF_XET_HIGH_PERFORMANCE"
_ENV_XET_CONCURRENT_RANGE_GETS = "HF_XET_NUM_CONCURRENT_RANGE_GETS"
_ENV_DISABLE_PROGRESS_BARS = "HF_HUB_DISABLE_PROGRESS_BARS"

# <owner>/<repo>/resolve/<revision>/<path...>
_MODEL_RESOLVE_RE = re.compile(
    r"^/(?P<owner>[^/]+)/(?P<repo>[^/]+)/resolve/(?P<revision>[^/]+)/(?P<path>.+)$"
)
# /datasets/<owner>/<repo>/resolve/<revision>/<path...>
_DATASET_RESOLVE_RE = re.compile(
    r"^/datasets/(?P<owner>[^/]+)/(?P<repo>[^/]+)/resolve/(?P<revision>[^/]+)/(?P<path>.+)$"
)


@dataclass(frozen=True, slots=True)
class _ParsedHfUrl:
    """A HuggingFace `/resolve/` URL, broken into repo coordinates."""

    repo_type: str  # "model" or "dataset"
    repo_id: str  # "<owner>/<repo>"
    revision: str
    path_in_repo: str


def _parse_hf_url(url: str) -> _ParsedHfUrl | None:
    """Parse a HuggingFace resolve URL, or return None for anything else.

    A HuggingFace *page* URL (no `/resolve/<revision>/`) is not a weight and
    must return None so the caller falls through rather than failing.
    """
    path = urlparse(url).path

    dataset_match = _DATASET_RESOLVE_RE.match(path)
    if dataset_match is not None:
        groups = dataset_match.groupdict()
        return _ParsedHfUrl(
            repo_type="dataset",
            repo_id=f"{groups['owner']}/{groups['repo']}",
            revision=groups["revision"],
            path_in_repo=groups["path"],
        )

    model_match = _MODEL_RESOLVE_RE.match(path)
    if model_match is not None:
        groups = model_match.groupdict()
        return _ParsedHfUrl(
            repo_type="model",
            repo_id=f"{groups['owner']}/{groups['repo']}",
            revision=groups["revision"],
            path_in_repo=groups["path"],
        )

    return None


def _endpoint_for(url: str) -> str:
    """The scheme+host `HfApi`/`hf_hub_download` should target for *url*.

    Without this, both default to the real huggingface.co regardless of
    which host `can_handle` actually matched -- a configured mirror would be
    silently ignored in favour of the canonical hub.
    """
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}"


def _dir_size(path: Path) -> int:
    """Best-effort total bytes under *path*. Missing/racing files are skipped."""
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            with contextlib.suppress(OSError):
                total += (Path(root) / name).stat().st_size
    return total


def _probe_xet_load() -> str | None:
    """Import `hf_xet` now and report a load failure, or None.

    Called from `__init__`, after `HF_XET_*` are set on the environment --
    never at module scope, which would import `hf_xet` (if nothing else in
    the process already has) before those env vars exist. `huggingface_hub`
    itself never raises on a failed native load; it silently falls back to
    its own httpx path, which would hide the 25x regression instead of
    surfacing it through the transport fallback chain (L6).
    """
    try:
        import hf_xet  # noqa: F401
    except ImportError as exc:
        return str(exc)
    return None


class HfXetTransport:
    """Fetches HuggingFace `/resolve/` URLs through `hf_xet`.

    `HF_XET_*` and `HF_HOME` are set on the process environment once, at
    construction, because some `hf_xet` builds read them at import time --
    setting them per call would silently do nothing.
    """

    def __init__(self, settings: Settings) -> None:
        self._token = unwrap_secret(settings.hf_token)

        # The same policy `build_registry` uses for the httpx path (E4/D4) --
        # a single source of truth for "which hosts may see the HF token"
        # means `can_handle`'s notion of an eligible host and the credential
        # guard's notion can never diverge.
        self._policy = build_huggingface_policy(settings)
        self._credentials: tuple[BoundCredential, ...] = build_credentials(
            (self._policy,), {"huggingface": self._token}
        )

        hf_home = settings.hf_home
        os.environ[_ENV_HF_HOME] = str(hf_home)
        os.environ[_ENV_XET_HIGH_PERFORMANCE] = "1"
        os.environ[_ENV_XET_CONCURRENT_RANGE_GETS] = str(settings.hf_xet_concurrent_range_gets)
        os.environ[_ENV_DISABLE_PROGRESS_BARS] = "1"

        # `huggingface_hub.constants` reads HF_HOME/HF_HUB_CACHE/HF_XET_CACHE/
        # HF_HUB_DISABLE_PROGRESS_BARS once, at first import of the package,
        # and never again. Importing it at module scope would snapshot
        # whatever those vars happened to be when this file was first
        # imported (before any Settings-driven env mutation ever ran),
        # silently sending the Xet chunk cache to the library's own default
        # (~/.cache/huggingface) instead of the configured hf_home. Deferring
        # the import to here, after the env vars above are set, is what makes
        # the snapshot pick up the right values.
        global HfApi, hf_hub_download
        from huggingface_hub import HfApi, hf_hub_download

        # Must run after the env vars above are set, and before hf_hub_download
        # gets a chance to import hf_xet itself under whatever ordering it
        # chooses.
        self._xet_load_error = _probe_xet_load()

        # Best-effort: constructing this transport must never fail deployment
        # (L6). A node whose cache_path isn't writable yet still deploys --
        # every later fetch degrades to the next candidate on its own.
        try:
            hf_home.mkdir(parents=True, exist_ok=True)
            free = shutil.disk_usage(hf_home).free
            log.info("hf_xet.home", path=str(hf_home), free_bytes=free)
        except OSError as exc:
            log.warning("hf_xet.home.unavailable", path=str(hf_home), error=str(exc))

    @property
    def name(self) -> str:
        return "hf_xet"

    def can_handle(self, url: str) -> bool:
        try:
            netloc = urlparse(url).netloc
        except ValueError:
            return False
        if not self._policy.matches(netloc):
            return False
        return _parse_hf_url(url) is not None

    def _check_egress(self, url: str) -> None:
        """Refuse to attach the HF token to a host `self._policy` does not cover.

        `_parse_hf_url` only looks at the path, not the host, so this is the
        one place that actually stops the token reaching an unintended host --
        `can_handle` uses the identical policy, but `fetch`/`probe_digest` are
        reachable directly (e.g. by a caller that skips `select_transport`),
        so the check is repeated here rather than trusted from upstream.
        """
        headers = {"Authorization": f"Bearer {self._token}"} if self._token else {}
        assert_no_credential_egress(url, headers, self._credentials)

    async def probe_digest(self, url: str) -> str | None:
        """Upstream's declared `lfs.sha256` for this path, or None.

        Never raises: a probe is an optimisation, and a wrong answer here
        would silently reroute a download or waste a transfer.
        """
        parsed = _parse_hf_url(url)
        if parsed is None:
            return None
        try:
            self._check_egress(url)
            endpoint = _endpoint_for(url)
            return await asyncio.to_thread(self._probe_digest_sync, parsed, endpoint)
        except Exception:
            log.debug("hf_xet.probe_digest.error", url=redact_url(url), exc_info=True)
            return None

    def _probe_digest_sync(self, parsed: _ParsedHfUrl, endpoint: str) -> str | None:
        api = HfApi(endpoint=endpoint, token=self._token)
        info = (
            api.dataset_info(parsed.repo_id, revision=parsed.revision, files_metadata=True)
            if parsed.repo_type == "dataset"
            else api.model_info(parsed.repo_id, revision=parsed.revision, files_metadata=True)
        )
        for sibling in info.siblings or ():
            if sibling.rfilename != parsed.path_in_repo:
                continue
            lfs = getattr(sibling, "lfs", None)
            if lfs is None:
                return None  # not an LFS file -- no Xet sha256 to compare
            sha256 = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
            return sha256 if isinstance(sha256, str) else None
        return None

    async def fetch(
        self, request: TransportRequest, on_progress: ProgressCallback | None
    ) -> TransportResult:
        parsed = _parse_hf_url(request.url)
        if parsed is None:
            raise TransportFetchError(f"not a HuggingFace resolve URL: {redact_url(request.url)}")

        if self._xet_load_error is not None:
            # Raise rather than call hf_hub_download anyway: huggingface_hub
            # would silently fall back to its own httpx path, hiding the 25x
            # regression instead of surfacing it through the fallback chain.
            raise TransportUnavailableError(
                f"hf_xet native component failed to load: {self._xet_load_error}"
            )

        # Outside the try/except below on purpose: a CredentialEgressError is
        # a policy violation, not a transport failure, and must not be
        # rewrapped into a TransportFetchError that _try_transport logs as a
        # routine fall-through-to-httpx case (which would just re-fail the
        # same way, since httpx uses the identical policy).
        self._check_egress(request.url)
        endpoint = _endpoint_for(request.url)

        temp_dir = request.destination.with_name(f"{request.destination.name}.hfxet")
        temp_dir.mkdir(parents=True, exist_ok=True)

        progress_task: asyncio.Task[None] | None = None
        if on_progress is not None:
            progress_task = asyncio.create_task(
                self._poll_progress(temp_dir, request.expected_size, on_progress)
            )

        try:
            downloaded = await asyncio.to_thread(
                hf_hub_download,
                repo_id=parsed.repo_id,
                filename=parsed.path_in_repo,
                repo_type=parsed.repo_type,
                revision=parsed.revision,
                token=self._token,
                local_dir=temp_dir,
                endpoint=endpoint,
            )
            source = Path(downloaded)
            bytes_written = source.stat().st_size
            request.destination.parent.mkdir(parents=True, exist_ok=True)
            source.replace(request.destination)
            return TransportResult(bytes_written=bytes_written, transport=self.name)
        except Exception as exc:
            secrets = (self._token,) if self._token else ()
            raise TransportFetchError(
                f"hf_xet fetch failed for {redact_url(request.url, secrets=secrets)}: {exc}"
            ) from exc
        finally:
            if progress_task is not None:
                progress_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await progress_task
            await asyncio.to_thread(shutil.rmtree, temp_dir, ignore_errors=True)

    async def _poll_progress(
        self,
        temp_dir: Path,
        expected_size: int | None,
        on_progress: ProgressCallback,
    ) -> None:
        total = expected_size or 0
        while True:
            await asyncio.sleep(_PROGRESS_POLL_INTERVAL_S)
            try:
                bytes_so_far = await asyncio.to_thread(_dir_size, temp_dir)
                await on_progress(bytes_so_far, total)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.debug("hf_xet.progress_poll.error", exc_info=True)
