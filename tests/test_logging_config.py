"""Tests for structlog configuration (JSON headless / console TTY routing)."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import structlog

from ai_content_service.logging_config import configure_logging

if TYPE_CHECKING:
    import pytest


class TestJsonFormat:
    def test_json_format_emits_valid_json(self, capsys: pytest.CaptureFixture[str]) -> None:
        configure_logging(fmt="json")
        log = structlog.get_logger("test.logging_config")
        log.info("test.event", foo="bar")

        line = capsys.readouterr().err.strip().splitlines()[-1]
        data = json.loads(line)

        assert data["event"] == "test.event"
        assert data["foo"] == "bar"
        assert "level" in data
        assert "timestamp" in data

    def test_auto_uses_json_when_not_tty(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("sys.stderr.isatty", lambda: False)
        configure_logging(fmt="auto")
        log = structlog.get_logger("test.logging_config")
        log.info("test.auto_event")

        line = capsys.readouterr().err.strip().splitlines()[-1]
        data = json.loads(line)
        assert data["event"] == "test.auto_event"


class TestStdlibBridge:
    def test_stdlib_loggers_render_through_structlog(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        configure_logging(fmt="json")
        logging.getLogger("httpx").warning("connection pool exhausted")

        line = capsys.readouterr().err.strip().splitlines()[-1]
        data = json.loads(line)
        assert data["event"] == "connection pool exhausted"
        assert data["logger"] == "httpx"


class TestIdempotentConfiguration:
    def test_configure_idempotent_handlers(self) -> None:
        configure_logging(fmt="json")
        configure_logging(fmt="json")

        assert len(logging.getLogger().handlers) == 1
