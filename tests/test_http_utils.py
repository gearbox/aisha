"""Tests for defensive HTTP header parsing."""

from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

from ai_content_service.http_utils import (
    parse_content_length,
    parse_content_range_total,
    parse_retry_after,
    safe_int,
)


class TestSafeInt:
    def test_valid_integer(self) -> None:
        assert safe_int("1234") == 1234

    def test_missing_or_malformed_value(self) -> None:
        assert safe_int(None) is None
        assert safe_int("") is None
        assert safe_int("  ") is None
        assert safe_int("1234, 1234") is None
        assert safe_int("chunked") is None


class TestContentHeaders:
    def test_content_length(self) -> None:
        assert parse_content_length({"Content-Length": "1234"}) == 1234
        assert parse_content_length({}) is None
        assert parse_content_length({"content-length": ""}) is None
        assert parse_content_length({"content-length": "1234, 1234"}) is None
        assert parse_content_length({"content-length": "chunked"}) is None

    def test_content_range_total(self) -> None:
        assert parse_content_range_total({"content-range": "bytes 0-0/14203980000"}) == 14203980000
        assert parse_content_range_total({"content-range": "bytes 0-0/*"}) is None
        assert parse_content_range_total({"content-range": "banana"}) is None
        assert parse_content_range_total({}) is None


class TestRetryAfter:
    def test_delta_seconds_are_parsed_and_clamped(self) -> None:
        now = datetime(2026, 7, 31, tzinfo=UTC)
        assert parse_retry_after("30", now=now, max_seconds=120) == 30
        assert parse_retry_after("999999", now=now, max_seconds=120) == 120

    def test_http_date_and_past_date(self) -> None:
        now = datetime(2026, 7, 31, 12, 0, tzinfo=UTC)
        future = format_datetime(now + timedelta(seconds=30), usegmt=True)
        past = format_datetime(now - timedelta(seconds=30), usegmt=True)
        assert parse_retry_after(future, now=now, max_seconds=120) == 30
        assert parse_retry_after(past, now=now, max_seconds=120) == 0

    def test_absent_or_garbage_is_unknown(self) -> None:
        now = datetime(2026, 7, 31, tzinfo=UTC)
        assert parse_retry_after(None, now=now, max_seconds=120) is None
        assert parse_retry_after("later", now=now, max_seconds=120) is None
