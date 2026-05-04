"""Tests for Content-Disposition header parsing utility."""

from ai_content_service.content_disposition_utils import parse_content_disposition


class TestParseContentDisposition:
    def test_none_returns_none(self) -> None:
        assert parse_content_disposition(None) is None

    def test_empty_string_returns_none(self) -> None:
        assert parse_content_disposition("") is None

    def test_attachment_only_returns_none(self) -> None:
        assert parse_content_disposition("attachment") is None

    def test_quoted_filename(self) -> None:
        assert (
            parse_content_disposition('attachment; filename="model.safetensors"')
            == "model.safetensors"
        )

    def test_unquoted_filename(self) -> None:
        assert (
            parse_content_disposition("attachment; filename=model.safetensors")
            == "model.safetensors"
        )

    def test_empty_filename_returns_none(self) -> None:
        assert parse_content_disposition("attachment; filename=") is None

    def test_utf8_filename_star_decoded(self) -> None:
        header = "attachment; filename*=UTF-8''model%20name%20v2.safetensors"
        assert parse_content_disposition(header) == "model name v2.safetensors"

    def test_utf8_lowercase_charset(self) -> None:
        assert (
            parse_content_disposition("attachment; filename*=utf-8''test.safetensors")
            == "test.safetensors"
        )

    def test_filename_star_preferred_over_filename(self) -> None:
        header = (
            "attachment; filename=\"fallback.bin\"; filename*=UTF-8''preferred%20name.safetensors"
        )
        assert parse_content_disposition(header) == "preferred name.safetensors"

    def test_filename_star_without_double_quote_falls_through(self) -> None:
        # filename*= value without '' marker is skipped; filename= is used as fallback
        header = 'attachment; filename*=malformed; filename="fallback.bin"'
        assert parse_content_disposition(header) == "fallback.bin"

    def test_case_insensitive_filename_key(self) -> None:
        header = 'attachment; FILENAME="model.safetensors"'
        assert parse_content_disposition(header) == "model.safetensors"

    def test_case_insensitive_filename_star_key(self) -> None:
        header = "attachment; FILENAME*=UTF-8''encoded.safetensors"
        assert parse_content_disposition(header) == "encoded.safetensors"

    def test_whitespace_around_parts(self) -> None:
        header = 'attachment ;  filename="model.safetensors" '
        assert parse_content_disposition(header) == "model.safetensors"

    def test_inline_disposition(self) -> None:
        assert parse_content_disposition('inline; filename="file.json"') == "file.json"
