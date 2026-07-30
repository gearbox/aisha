"""Tests for download_auth: pure per-host auth-policy resolution and token attachment."""

from __future__ import annotations

from urllib.parse import parse_qs, urlparse

import pytest

from ai_content_service.config import Settings
from ai_content_service.download_auth import (
    AuthTransport,
    HostAuthPolicy,
    apply_auth,
    build_registry,
    redact_url,
    resolve_policy,
)

# ---------------------------------------------------------------------------
# resolve_policy / HostAuthPolicy.matches
# ---------------------------------------------------------------------------


@pytest.fixture
def registry() -> tuple[HostAuthPolicy, ...]:
    return build_registry(Settings())


class TestResolvePolicy:
    @pytest.mark.parametrize(
        "netloc",
        ["civitai.com", "civitai.red", "civitai.green", "www.civitai.red", "CIVITAI.RED"],
    )
    def test_civitai_domains_resolve_to_civitai_policy(
        self, registry: tuple[HostAuthPolicy, ...], netloc: str
    ) -> None:
        policy = resolve_policy(registry, f"https://{netloc}/api/download/models/1")
        assert policy is not None
        assert policy.name == "civitai"

    @pytest.mark.parametrize(
        "netloc",
        ["civitai.com.evil.com", "notcivitai.com", "civitai.red.evil.com"],
    )
    def test_lookalike_domains_get_no_policy(
        self, registry: tuple[HostAuthPolicy, ...], netloc: str
    ) -> None:
        """Substring/suffix lookalikes must not match — token exfiltration guard."""
        policy = resolve_policy(registry, f"https://{netloc}/x")
        assert policy is None

    @pytest.mark.parametrize("netloc", ["huggingface.co", "hf.co"])
    def test_huggingface_domains_resolve_to_hf_policy(
        self, registry: tuple[HostAuthPolicy, ...], netloc: str
    ) -> None:
        policy = resolve_policy(registry, f"https://{netloc}/model/download")
        assert policy is not None
        assert policy.name == "huggingface"

    def test_unknown_host_gets_no_policy(self, registry: tuple[HostAuthPolicy, ...]) -> None:
        assert resolve_policy(registry, "https://example.com/model.safetensors") is None

    def test_subdomain_matches(self, registry: tuple[HostAuthPolicy, ...]) -> None:
        policy = resolve_policy(registry, "https://cdn-lfs.huggingface.co/x")
        assert policy is not None
        assert policy.name == "huggingface"

    def test_userinfo_and_port_stripped(self, registry: tuple[HostAuthPolicy, ...]) -> None:
        policy = resolve_policy(registry, "https://foo@huggingface.co:443/x")
        assert policy is not None
        assert policy.name == "huggingface"

    def test_host_in_userinfo_does_not_match(self, registry: tuple[HostAuthPolicy, ...]) -> None:
        policy = resolve_policy(registry, "https://huggingface.co@evil.com/x")
        assert policy is None


# ---------------------------------------------------------------------------
# apply_auth
# ---------------------------------------------------------------------------


class TestApplyAuth:
    @pytest.fixture
    def civitai_policy(self) -> HostAuthPolicy:
        return HostAuthPolicy(
            name="civitai",
            domains=("civitai.com", "civitai.red", "civitai.green"),
            primary=AuthTransport.BEARER_HEADER,
            fallback=AuthTransport.QUERY_TOKEN,
        )

    def test_bearer_header_puts_token_in_header_not_url(
        self, civitai_policy: HostAuthPolicy
    ) -> None:
        url = "https://civitai.red/api/download/models/123"
        new_url, headers = apply_auth(
            civitai_policy, AuthTransport.BEARER_HEADER, url, {}, "secret-token"
        )

        assert new_url == url
        assert headers["Authorization"] == "Bearer secret-token"
        assert "token" not in new_url

    def test_query_token_puts_token_in_query_not_headers(
        self, civitai_policy: HostAuthPolicy
    ) -> None:
        url = "https://civitai.red/api/download/models/123"
        new_url, headers = apply_auth(
            civitai_policy, AuthTransport.QUERY_TOKEN, url, {}, "secret-token"
        )

        query = parse_qs(urlparse(new_url).query)
        assert query["token"] == ["secret-token"]
        assert headers == {}

    def test_query_token_preserves_existing_params(self, civitai_policy: HostAuthPolicy) -> None:
        url = "https://civitai.red/api/download/models/123?type=Model"
        new_url, _headers = apply_auth(
            civitai_policy, AuthTransport.QUERY_TOKEN, url, {}, "secret-token"
        )

        query = parse_qs(urlparse(new_url).query)
        assert query["type"] == ["Model"]
        assert query["token"] == ["secret-token"]

    def test_query_token_overwrites_existing_token(self, civitai_policy: HostAuthPolicy) -> None:
        url = "https://civitai.red/api/download/models/123?token=old"
        new_url, _headers = apply_auth(
            civitai_policy, AuthTransport.QUERY_TOKEN, url, {}, "secret-token"
        )

        query = parse_qs(urlparse(new_url).query)
        assert query["token"] == ["secret-token"]

    def test_none_token_leaves_url_and_headers_unchanged(
        self, civitai_policy: HostAuthPolicy
    ) -> None:
        url = "https://civitai.red/api/download/models/123"
        new_url, headers = apply_auth(civitai_policy, AuthTransport.BEARER_HEADER, url, {}, None)

        assert new_url == url
        assert headers == {}

    def test_transport_not_declared_on_policy_raises(self) -> None:
        hf_policy = HostAuthPolicy(
            name="huggingface",
            domains=("huggingface.co",),
            primary=AuthTransport.BEARER_HEADER,
            fallback=None,
        )
        with pytest.raises(ValueError, match="not configured"):
            apply_auth(hf_policy, AuthTransport.QUERY_TOKEN, "https://huggingface.co/x", {}, "t")

    def test_existing_headers_preserved(self, civitai_policy: HostAuthPolicy) -> None:
        _url, headers = apply_auth(
            civitai_policy,
            AuthTransport.BEARER_HEADER,
            "https://civitai.red/x",
            {"User-Agent": "test-ua"},
            "secret-token",
        )
        assert headers["User-Agent"] == "test-ua"
        assert headers["Authorization"] == "Bearer secret-token"


# ---------------------------------------------------------------------------
# redact_url
# ---------------------------------------------------------------------------


class TestRedactUrl:
    def test_masks_token(self) -> None:
        redacted = redact_url("https://civitai.com/api/download/models/1?token=SUPER_SECRET")
        assert "SUPER_SECRET" not in redacted
        assert "token=***" in redacted

    def test_masks_api_key(self) -> None:
        redacted = redact_url("https://example.com/x?api_key=SUPER_SECRET")
        assert "SUPER_SECRET" not in redacted
        assert "api_key=***" in redacted

    def test_masks_access_token(self) -> None:
        redacted = redact_url("https://example.com/x?access_token=SUPER_SECRET")
        assert "SUPER_SECRET" not in redacted
        assert "access_token=***" in redacted

    def test_leaves_other_params_and_path_intact(self) -> None:
        redacted = redact_url("https://civitai.com/api/download/models/1?type=Model&token=SECRET")
        assert "type=Model" in redacted
        assert "/api/download/models/1" in redacted
        assert "SECRET" not in redacted

    def test_noop_on_token_free_url(self) -> None:
        url = "https://huggingface.co/model/download?revision=main"
        assert redact_url(url) == url

    def test_masks_token_embedded_in_exception_text(self) -> None:
        """redact_url is also used on stringified exceptions (D12), not just clean URLs."""
        text = (
            "Client error '403 Forbidden' for url "
            "'https://civitai.com/api/download/models/123?type=Model&token=SUPER_SECRET_TOKEN'"
        )
        redacted = redact_url(text)
        assert "SUPER_SECRET_TOKEN" not in redacted
        assert "token=***" in redacted

    def test_does_not_raise_on_malformed_input(self) -> None:
        assert redact_url("not a url at all ??%%") == "not a url at all ??%%"


# ---------------------------------------------------------------------------
# build_registry
# ---------------------------------------------------------------------------


class TestBuildRegistry:
    def test_default_civitai_domains(self) -> None:
        registry = build_registry(Settings())
        civitai = next(p for p in registry if p.name == "civitai")
        assert civitai.domains == ("civitai.com", "civitai.red", "civitai.green")

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ACS_CIVITAI_DOMAINS", "civitai.com, Civitai.Red")
        registry = build_registry(Settings())
        civitai = next(p for p in registry if p.name == "civitai")
        assert civitai.domains == ("civitai.com", "civitai.red")

    def test_huggingface_has_no_fallback(self) -> None:
        registry = build_registry(Settings())
        hf = next(p for p in registry if p.name == "huggingface")
        assert hf.fallback is None

    def test_civitai_fallback_is_query_token(self) -> None:
        registry = build_registry(Settings())
        civitai = next(p for p in registry if p.name == "civitai")
        assert civitai.fallback == AuthTransport.QUERY_TOKEN
