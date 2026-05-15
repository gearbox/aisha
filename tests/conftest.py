"""pytest configuration and shared fixtures."""

from __future__ import annotations

import os

import pytest

from ai_content_service.config import Settings


@pytest.fixture(autouse=True)
def isolate_settings_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent .env and real ACS_ env vars from leaking into Settings during tests.

    Without this, pydantic-settings reads the project .env, which sets
    ACS_BUNDLE, ACS_CF_TUNNEL_TOKEN, ACS_COMFYUI_PATH, etc., causing
    tests that expect defaults to fail.
    """
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    for key in [k for k in os.environ if k.startswith("ACS_")]:
        monkeypatch.delenv(key, raising=False)
