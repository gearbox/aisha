"""Typer-free authoring operation for fetching one model and rendering its fragment."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

import yaml
from pydantic import ValidationError

from .config import ModelConfig, ModelFileConfig, Settings
from .downloader import DownloadError, FileFailure, ModelDownloader, build_transports
from .file_hashes import compute_file_sha256
from .url_sanitizer import strip_credential_query_params

if TYPE_CHECKING:
    from pathlib import Path


class ModelsServiceError(Exception):
    """Base exception for a model-fetch operation that can be shown by the CLI."""


class ModelFetchInputError(ModelsServiceError):
    """The user-supplied model description failed Pydantic validation."""


class ModelFetchDownloadError(ModelsServiceError):
    """The downloader failed or returned one or more file failures."""

    def __init__(self, message: str, failures: tuple[FileFailure, ...] = ()) -> None:
        super().__init__(message)
        self.failures = failures


class ModelFetchInspectionError(ModelsServiceError):
    """A completed download could not be inspected on local disk."""


@dataclass(frozen=True, slots=True)
class FetchedModel:
    """The verified local model metadata and safe bundle-YAML fragment."""

    sha256: str
    size_bytes: int
    path: Path
    yaml_fragment: str


async def fetch_model(
    settings: Settings,
    *,
    url: str,
    model_type: str,
    filename: str,
    subdirectory: str | None,
    sha256: str | None,
) -> FetchedModel:
    """Fetch one model with the normal downloader and produce safe YAML metadata."""
    try:
        model = ModelConfig(
            name=filename,
            model_type=model_type,
            subdirectory=subdirectory,
            files=[
                ModelFileConfig(
                    name=filename,
                    url=url,
                    filename=filename,
                    sha256=sha256,
                )
            ],
        )
    except ValidationError as exc:
        raise ModelFetchInputError(f"Invalid model input:\n{exc}") from exc

    try:
        downloader = ModelDownloader(settings, build_transports(settings))
        report = await downloader.download_all([model], settings.models_path)
    except DownloadError as exc:
        raise ModelFetchDownloadError(str(exc)) from exc
    if not report.ok:
        raise ModelFetchDownloadError("Model download failed", report.failed)

    path = settings.models_path / model.target_subpath / filename
    try:
        digest = await asyncio.to_thread(compute_file_sha256, path)
        size_bytes = path.stat().st_size
    except OSError as exc:
        raise ModelFetchInspectionError(
            f"Download completed but cannot inspect {path}: {exc}"
        ) from exc

    fragment = yaml.safe_dump(
        [
            {
                "name": filename,
                "url": strip_credential_query_params(url),
                "filename": filename,
                "sha256": digest,
                "size_bytes": size_bytes,
            }
        ],
        default_flow_style=False,
        sort_keys=False,
    )
    return FetchedModel(
        sha256=digest,
        size_bytes=size_bytes,
        path=path,
        yaml_fragment=fragment,
    )
