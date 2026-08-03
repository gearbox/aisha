"""Tests for Typer-free single-model authoring."""

from __future__ import annotations

import asyncio
import hashlib
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import pytest

from ai_content_service.config import Settings
from ai_content_service.downloader import DownloadError, DownloadReport, FileFailure
from ai_content_service.models_service import (
    ModelFetchDownloadError,
    ModelFetchInputError,
    ModelFetchInspectionError,
    fetch_model,
)

if TYPE_CHECKING:
    from collections.abc import Coroutine
    from pathlib import Path
    from typing import TypeVar

    _Result = TypeVar("_Result")


def _run(coroutine: Coroutine[object, object, _Result]) -> _Result:
    """Run an isolated coroutine without leaving pytest without a default loop."""
    try:
        return asyncio.run(coroutine)
    finally:
        asyncio.set_event_loop(asyncio.new_event_loop())


def _settings(tmp_path: Path) -> Settings:
    return Settings(comfyui_path=tmp_path / "ComfyUI")


def test_fetch_model_rejects_pydantic_input_before_downloader(tmp_path: Path) -> None:
    with pytest.raises(ModelFetchInputError, match="Invalid model input"):
        _run(
            fetch_model(
                _settings(tmp_path),
                url="not-a-url",
                model_type="checkpoints",
                filename="model",
                subdirectory=None,
                sha256=None,
            )
        )


def test_fetch_model_translates_downloader_exception(tmp_path: Path) -> None:
    downloader = AsyncMock()
    downloader.download_all.side_effect = DownloadError("network failed")
    with (
        patch("ai_content_service.models_service.ModelDownloader", return_value=downloader),
        pytest.raises(ModelFetchDownloadError, match="network failed"),
    ):
        _run(
            fetch_model(
                _settings(tmp_path),
                url="https://example.com/model",
                model_type="checkpoints",
                filename="model",
                subdirectory=None,
                sha256=None,
            )
        )


def test_fetch_model_exposes_downloader_report_failures(tmp_path: Path) -> None:
    failure = FileFailure("model", "https://example.com/model", "failed")
    downloader = AsyncMock()
    downloader.download_all.return_value = DownloadReport(succeeded=0, failed=(failure,))
    with (
        patch("ai_content_service.models_service.ModelDownloader", return_value=downloader),
        pytest.raises(ModelFetchDownloadError) as error,
    ):
        _run(
            fetch_model(
                _settings(tmp_path),
                url="https://example.com/model",
                model_type="checkpoints",
                filename="model",
                subdirectory=None,
                sha256=None,
            )
        )

    assert error.value.failures == (failure,)


def test_fetch_model_inspects_file_and_sanitizes_fragment(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    path = settings.models_path / "checkpoints" / "model"
    content = b"model-bytes"
    path.parent.mkdir(parents=True)
    path.write_bytes(content)
    downloader = AsyncMock()
    downloader.download_all.return_value = DownloadReport(succeeded=1, failed=())

    with patch("ai_content_service.models_service.ModelDownloader", return_value=downloader):
        fetched = _run(
            fetch_model(
                settings,
                url="https://civitai.com/api/download/models/1?token=super-secret&a=1&a=2",
                model_type="checkpoints",
                filename="model",
                subdirectory=None,
                sha256=None,
            )
        )

    assert fetched.sha256 == hashlib.sha256(content).hexdigest()
    assert fetched.size_bytes == len(content)
    assert "super-secret" not in fetched.yaml_fragment
    assert "a=1&a=2" in fetched.yaml_fragment


def test_fetch_model_reports_completed_but_missing_file(tmp_path: Path) -> None:
    downloader = AsyncMock()
    downloader.download_all.return_value = DownloadReport(succeeded=1, failed=())
    with (
        patch("ai_content_service.models_service.ModelDownloader", return_value=downloader),
        pytest.raises(ModelFetchInspectionError, match="cannot inspect"),
    ):
        _run(
            fetch_model(
                _settings(tmp_path),
                url="https://example.com/model",
                model_type="checkpoints",
                filename="model",
                subdirectory=None,
                sha256=None,
            )
        )
