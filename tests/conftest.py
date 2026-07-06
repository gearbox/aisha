"""pytest configuration and shared fixtures."""

from __future__ import annotations

import os

import pytest
import structlog

from ai_content_service.config import Settings


@pytest.fixture(scope="session", autouse=True)
def _structlog_stdlib_bridge() -> None:
    """Route structlog through stdlib logging for the whole test session.

    Without this, `structlog.get_logger()` uses structlog's default
    PrintLogger (bypassing the `logging` module entirely) until something
    calls `configure_logging()` — which previously happened only by accident,
    depending on alphabetical test-file collection order. Any test file
    relying on `caplog` would then fail when run in isolation. This mirrors
    the stdlib-bridge half of `logging_config.configure_logging()` (no
    handler is installed — `caplog` manages its own).
    """
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


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
