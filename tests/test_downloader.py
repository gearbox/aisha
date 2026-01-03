"""Tests for model downloader including Civitai support."""

import pytest

from ai_content_service.config import Settings
from ai_content_service.downloader import ModelDownloader


@pytest.fixture
def settings() -> Settings:
    """Create settings for testing."""
    return Settings(
        civitai_api_token="test_civitai_token_123",
        hf_token="test_hf_token_456",
    )


@pytest.fixture
def settings_no_tokens() -> Settings:
    """Create settings without API tokens."""
    return Settings()


@pytest.fixture
def downloader(settings: Settings) -> ModelDownloader:
    """Create a downloader instance with tokens."""
    return ModelDownloader(settings)


@pytest.fixture
def downloader_no_tokens(settings_no_tokens: Settings) -> ModelDownloader:
    """Create a downloader instance without tokens."""
    return ModelDownloader(settings_no_tokens)


class TestCivitaiUrlDetection:
    """Tests for Civitai URL detection."""

    def test_civitai_url_detected(self, downloader: ModelDownloader) -> None:
        """Test that civitai.com URLs are detected."""
        assert downloader._is_civitai_url("https://civitai.com/api/download/models/123")
        assert downloader._is_civitai_url("https://www.civitai.com/api/download/models/456")

    def test_civitai_url_case_insensitive(self, downloader: ModelDownloader) -> None:
        """Test that detection is case insensitive."""
        assert downloader._is_civitai_url("https://CIVITAI.COM/api/download/models/123")
        assert downloader._is_civitai_url("https://CivitAI.com/api/download/models/123")

    def test_non_civitai_urls_not_detected(self, downloader: ModelDownloader) -> None:
        """Test that non-Civitai URLs are not detected."""
        assert not downloader._is_civitai_url("https://huggingface.co/model/download")
        assert not downloader._is_civitai_url("https://example.com/civitai.com/fake")
        assert not downloader._is_civitai_url("https://notcivitai.com/models/123")


class TestCivitaiUrlPreparation:
    """Tests for Civitai URL preparation with token."""

    def test_civitai_url_gets_token_appended(self, downloader: ModelDownloader) -> None:
        """Test that token is appended to Civitai URLs."""
        url = "https://civitai.com/api/download/models/123"
        prepared = downloader._prepare_download_url(url)

        assert "token=test_civitai_token_123" in prepared
        assert prepared.startswith("https://civitai.com/api/download/models/123")

    def test_civitai_url_with_existing_query_params(self, downloader: ModelDownloader) -> None:
        """Test that token is added to URLs with existing query params."""
        url = "https://civitai.com/api/download/models/123?type=Model"
        prepared = downloader._prepare_download_url(url)

        assert "token=test_civitai_token_123" in prepared
        assert "type=Model" in prepared

    def test_civitai_url_token_overwrites_existing(self, downloader: ModelDownloader) -> None:
        """Test that existing token is overwritten."""
        url = "https://civitai.com/api/download/models/123?token=old_token"
        prepared = downloader._prepare_download_url(url)

        assert "token=test_civitai_token_123" in prepared
        assert "old_token" not in prepared

    def test_civitai_url_without_token_unchanged(
        self, downloader_no_tokens: ModelDownloader
    ) -> None:
        """Test that Civitai URLs without token setting are unchanged."""
        url = "https://civitai.com/api/download/models/123"
        prepared = downloader_no_tokens._prepare_download_url(url)

        assert prepared == url
        assert "token=" not in prepared

    def test_non_civitai_url_unchanged(self, downloader: ModelDownloader) -> None:
        """Test that non-Civitai URLs are not modified."""
        url = "https://huggingface.co/model/download"
        prepared = downloader._prepare_download_url(url)

        assert prepared == url
        assert "token=" not in prepared


class TestAuthHeaders:
    """Tests for authentication headers."""

    def test_huggingface_gets_auth_header(self, downloader: ModelDownloader) -> None:
        """Test that HuggingFace URLs get Authorization header."""
        url = "https://huggingface.co/model/download"
        headers = downloader._get_auth_headers(url)

        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer test_hf_token_456"

    def test_civitai_no_auth_header(self, downloader: ModelDownloader) -> None:
        """Test that Civitai URLs don't get auth header (uses URL token)."""
        url = "https://civitai.com/api/download/models/123"
        headers = downloader._get_auth_headers(url)

        # Civitai uses URL token, not header
        assert "Authorization" not in headers

    def test_other_urls_no_auth_header(self, downloader: ModelDownloader) -> None:
        """Test that other URLs don't get auth headers."""
        url = "https://example.com/model.safetensors"
        headers = downloader._get_auth_headers(url)

        assert headers == {}

    def test_no_token_no_auth_header(self, downloader_no_tokens: ModelDownloader) -> None:
        """Test that no header is added when no token is configured."""
        url = "https://huggingface.co/model/download"
        headers = downloader_no_tokens._get_auth_headers(url)

        assert headers == {}


class TestContentDispositionParsing:
    """Tests for Content-Disposition header parsing."""

    def test_simple_filename(self) -> None:
        """Test parsing simple filename."""
        header = 'attachment; filename="model.safetensors"'
        result = ModelDownloader._parse_content_disposition(header)
        assert result == "model.safetensors"

    def test_filename_without_quotes(self) -> None:
        """Test parsing filename without quotes."""
        header = "attachment; filename=model.safetensors"
        result = ModelDownloader._parse_content_disposition(header)
        assert result == "model.safetensors"

    def test_utf8_encoded_filename(self) -> None:
        """Test parsing UTF-8 encoded filename."""
        header = "attachment; filename*=UTF-8''model%20name.safetensors"
        result = ModelDownloader._parse_content_disposition(header)
        assert result == "model name.safetensors"

    def test_utf8_lowercase(self) -> None:
        """Test parsing utf-8 lowercase encoding."""
        header = "attachment; filename*=utf-8''test.safetensors"
        result = ModelDownloader._parse_content_disposition(header)
        assert result == "test.safetensors"

    def test_none_header(self) -> None:
        """Test handling None header."""
        result = ModelDownloader._parse_content_disposition(None)
        assert result is None

    def test_empty_header(self) -> None:
        """Test handling empty header."""
        result = ModelDownloader._parse_content_disposition("")
        assert result is None

    def test_header_without_filename(self) -> None:
        """Test handling header without filename."""
        header = "attachment"
        result = ModelDownloader._parse_content_disposition(header)
        assert result is None

    def test_complex_civitai_header(self) -> None:
        """Test parsing realistic Civitai Content-Disposition header."""
        header = "attachment; filename=\"v1-5-pruned-emaonly.safetensors\"; filename*=UTF-8''v1-5-pruned-emaonly.safetensors"
        result = ModelDownloader._parse_content_disposition(header)
        # Should prefer filename*= (UTF-8 encoded)
        assert result == "v1-5-pruned-emaonly.safetensors"


class TestSettingsCivitaiToken:
    """Tests for Civitai token in settings."""

    def test_civitai_token_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that CIVITAI_API_TOKEN env var is read."""
        monkeypatch.setenv("ACS_CIVITAI_API_TOKEN", "env_token_xyz")
        settings = Settings()
        assert settings.civitai_api_token == "env_token_xyz"

    def test_civitai_token_default_none(self) -> None:
        """Test that Civitai token defaults to None."""
        settings = Settings()
        assert settings.civitai_api_token is None
