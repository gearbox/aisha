"""Tests for Typer-free cache command preparation."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from ai_content_service import cache_service
from ai_content_service.bundle_registry import BundleReference
from ai_content_service.cache_workflows import (
    CacheWorkflowError,
    resolve_cache_targets,
    verify_cache_targets,
)
from ai_content_service.config import Settings
from ai_content_service.registry_service import create_registry_manager

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


def _settings_and_bundle(tmp_path: Path, yaml_text: str | None = None) -> Settings:
    bundles = tmp_path / "bundles"
    version = bundles / "demo" / "260101-01"
    version.mkdir(parents=True)
    (bundles / "demo" / "current").symlink_to("260101-01")
    (version / "bundle.yaml").write_text(
        yaml_text
        or (
            "metadata:\n  name: demo\n  version: '260101-01'\nmodels:\n"
            "  - name: model\n    model_type: checkpoints\n    files:\n"
            "      - name: model\n        url: https://example.com/model\n        filename: model\n"
        )
    )
    return Settings(comfyui_path=tmp_path / "ComfyUI", bundles_path=bundles)


def test_resolve_cache_targets_loads_validated_selected_model(tmp_path: Path) -> None:
    settings = _settings_and_bundle(tmp_path)
    resolved = _run(
        resolve_cache_targets(
            settings,
            create_registry_manager(settings),
            BundleReference.parse("demo"),
            only_filename="model",
            sync=False,
        )
    )

    assert resolved.config.metadata.name == "demo"
    assert [target.file.filename for target in resolved.targets] == ["model"]


@pytest.mark.parametrize(
    "yaml_text",
    [
        "metadata: [not, a, mapping",
        "metadata:\n  name: demo\n  version: '260101-01'\n  typo: true\n",
    ],
)
def test_resolve_cache_targets_translates_parse_and_schema_errors(
    tmp_path: Path, yaml_text: str
) -> None:
    settings = _settings_and_bundle(tmp_path, yaml_text)

    with pytest.raises(CacheWorkflowError, match="Invalid bundle config"):
        _run(
            resolve_cache_targets(
                settings,
                create_registry_manager(settings),
                BundleReference.parse("demo"),
                only_filename=None,
                sync=False,
            )
        )


def test_resolve_cache_targets_rejects_missing_selector_match(tmp_path: Path) -> None:
    settings = _settings_and_bundle(tmp_path)

    with pytest.raises(CacheWorkflowError, match="No matching"):
        _run(
            resolve_cache_targets(
                settings,
                create_registry_manager(settings),
                BundleReference.parse("demo"),
                only_filename="missing",
                sync=False,
            )
        )


def test_verify_cache_targets_reuses_resolution_flow(tmp_path: Path) -> None:
    settings = _settings_and_bundle(tmp_path)
    expected = cache_service.VerifyReport()
    with patch(
        "ai_content_service.cache_workflows.cache_service.verify_models", return_value=expected
    ) as verify:
        report = _run(
            verify_cache_targets(
                settings,
                create_registry_manager(settings),
                BundleReference.parse("demo"),
                only_filename=None,
                sync=False,
                deep=True,
            )
        )

    assert report is expected
    assert verify.call_args.kwargs["deep"] is True
