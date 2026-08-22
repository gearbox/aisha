"""Tests for the package-derived public version."""

from __future__ import annotations

import importlib
from importlib.metadata import PackageNotFoundError, version
from unittest.mock import patch

import ai_content_service


def test_package_version_matches_installed_distribution() -> None:
    assert ai_content_service.__version__ == version("aisha")


def test_package_version_has_a_source_tree_fallback() -> None:
    with patch("importlib.metadata.version", side_effect=PackageNotFoundError):
        fallback_module = importlib.reload(ai_content_service)
        assert fallback_module.__version__ == "0.0.0+unknown"

    importlib.reload(ai_content_service)
