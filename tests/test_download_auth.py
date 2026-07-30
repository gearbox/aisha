"""Tests for download_auth: pure per-host auth-policy resolution and token attachment."""

from __future__ import annotations

from urllib.parse import parse_qs, urlparse

import httpx
import pytest

from ai_content_service.config import Settings
from ai_content_service.download_auth import (
    AuthTransport,
    CredentialEgressError,
    HostAuthPolicy,
    apply_auth,
    assert_no_credential_egress,
    attempt_with_auth,
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

    def test_civitai_opt_out_disables_fallback(self) -> None:
        registry = build_registry(Settings(civitai_allow_query_token_fallback=False))
        civitai = next(p for p in registry if p.name == "civitai")
        assert civitai.fallback is None


# ---------------------------------------------------------------------------
# attempt_with_auth (M4a)
# ---------------------------------------------------------------------------


class TestAttemptWithAuth:
    @pytest.fixture
    def civitai_policy(self) -> HostAuthPolicy:
        return HostAuthPolicy(
            name="civitai",
            domains=("civitai.com", "civitai.red", "civitai.green"),
            primary=AuthTransport.BEARER_HEADER,
            fallback=AuthTransport.QUERY_TOKEN,
        )

    async def test_200_first_try_one_send_primary_transport(
        self, civitai_policy: HostAuthPolicy
    ) -> None:
        calls: list[tuple[str, dict[str, str]]] = []

        async def send(url: str, headers: dict[str, str]) -> int:
            calls.append((url, headers))
            return 200

        result, transport = await attempt_with_auth(
            civitai_policy, "tok", "https://civitai.red/x", {}, send=send, status_of=lambda r: r
        )

        assert result == 200
        assert transport == AuthTransport.BEARER_HEADER
        assert len(calls) == 1
        assert calls[0][1]["Authorization"] == "Bearer tok"
        assert "token" not in calls[0][0]

    async def test_401_then_200_falls_back_exactly_once(
        self, civitai_policy: HostAuthPolicy
    ) -> None:
        responses = iter([401, 200])
        calls: list[tuple[str, dict[str, str]]] = []

        async def send(url: str, headers: dict[str, str]) -> int:
            calls.append((url, headers))
            return next(responses)

        result, transport = await attempt_with_auth(
            civitai_policy, "tok", "https://civitai.red/x", {}, send=send, status_of=lambda r: r
        )

        assert result == 200
        assert transport == AuthTransport.QUERY_TOKEN
        assert len(calls) == 2
        assert "Authorization" in calls[0][1]
        assert "token" not in calls[0][0]
        assert "Authorization" not in calls[1][1]
        assert "token=tok" in calls[1][0]

    async def test_401_then_401_returns_second_response_with_fallback_transport(
        self, civitai_policy: HostAuthPolicy
    ) -> None:
        calls: list[tuple[str, dict[str, str]]] = []

        async def send(url: str, headers: dict[str, str]) -> int:
            calls.append((url, headers))
            return 401

        result, transport = await attempt_with_auth(
            civitai_policy, "tok", "https://civitai.red/x", {}, send=send, status_of=lambda r: r
        )

        assert result == 401
        assert transport == AuthTransport.QUERY_TOKEN
        assert len(calls) == 2  # never a third attempt

    async def test_policy_with_no_fallback_issues_a_single_send(self) -> None:
        hf_policy = HostAuthPolicy(
            name="huggingface",
            domains=("huggingface.co",),
            primary=AuthTransport.BEARER_HEADER,
            fallback=None,
        )
        calls: list[tuple[str, dict[str, str]]] = []

        async def send(url: str, headers: dict[str, str]) -> int:
            calls.append((url, headers))
            return 401

        result, transport = await attempt_with_auth(
            hf_policy, "tok", "https://huggingface.co/x", {}, send=send, status_of=lambda r: r
        )

        assert result == 401
        assert transport == AuthTransport.BEARER_HEADER
        assert len(calls) == 1

    async def test_civitai_with_fallback_opted_out_never_builds_query_token(self) -> None:
        """Settings(civitai_allow_query_token_fallback=False) nulls policy.fallback;
        attempt_with_auth must then never build a ?token= URL, even on 403."""
        opted_out_policy = HostAuthPolicy(
            name="civitai",
            domains=("civitai.red",),
            primary=AuthTransport.BEARER_HEADER,
            fallback=None,
        )
        calls: list[tuple[str, dict[str, str]]] = []

        async def send(url: str, headers: dict[str, str]) -> int:
            calls.append((url, headers))
            return 403

        result, transport = await attempt_with_auth(
            opted_out_policy,
            "tok",
            "https://civitai.red/x",
            {},
            send=send,
            status_of=lambda r: r,
        )

        assert result == 403
        assert transport == AuthTransport.BEARER_HEADER
        assert len(calls) == 1
        assert all("token=" not in url for url, _ in calls)

    async def test_no_policy_sends_unauthenticated_once(self) -> None:
        calls: list[tuple[str, dict[str, str]]] = []

        async def send(url: str, headers: dict[str, str]) -> int:
            calls.append((url, headers))
            return 200

        result, transport = await attempt_with_auth(
            None, None, "https://example.com/x", {}, send=send, status_of=lambda r: r
        )

        assert result == 200
        assert transport == AuthTransport.NONE
        assert len(calls) == 1
        assert calls[0][1] == {}


# ---------------------------------------------------------------------------
# assert_no_credential_egress / CredentialEgressError (R3a)
# ---------------------------------------------------------------------------


class TestAssertNoCredentialEgress:
    @pytest.fixture
    def civitai_policy(self) -> HostAuthPolicy:
        return HostAuthPolicy(
            name="civitai",
            domains=("civitai.com", "civitai.red", "civitai.green"),
            primary=AuthTransport.BEARER_HEADER,
            fallback=AuthTransport.QUERY_TOKEN,
        )

    def test_raises_for_bearer_header_bound_to_foreign_cdn(
        self, civitai_policy: HostAuthPolicy
    ) -> None:
        with pytest.raises(CredentialEgressError):
            assert_no_credential_egress(
                civitai_policy, "https://cdn.example.com/x", {"Authorization": "Bearer t"}
            )

    @pytest.mark.parametrize("host", ["civitai.red", "www.civitai.red", "civitai.com"])
    def test_passes_for_policy_domains(self, civitai_policy: HostAuthPolicy, host: str) -> None:
        assert_no_credential_egress(
            civitai_policy, f"https://{host}/x", {"Authorization": "Bearer t"}
        )  # must not raise

    def test_raises_for_lookalike_domain(self, civitai_policy: HostAuthPolicy) -> None:
        with pytest.raises(CredentialEgressError):
            assert_no_credential_egress(
                civitai_policy,
                "https://civitai.com.evil.com/x",
                {"Authorization": "Bearer t"},
            )

    def test_raises_for_query_token_on_foreign_host(self, civitai_policy: HostAuthPolicy) -> None:
        with pytest.raises(CredentialEgressError):
            assert_no_credential_egress(civitai_policy, "https://evil.com/x?token=SECRET", {})

    def test_no_credential_present_never_raises_even_on_foreign_host(
        self, civitai_policy: HostAuthPolicy
    ) -> None:
        assert_no_credential_egress(civitai_policy, "https://cdn.example.com/x", {})  # no raise

    def test_none_policy_with_credential_raises(self) -> None:
        with pytest.raises(CredentialEgressError):
            assert_no_credential_egress(
                None, "https://civitai.red/x", {"Authorization": "Bearer t"}
            )

    def test_never_raises_on_malformed_url(self, civitai_policy: HostAuthPolicy) -> None:
        # urlparse itself raises ValueError on a malformed IPv6 host -- the
        # guard must swallow that rather than let it crash the request.
        assert_no_credential_egress(
            civitai_policy, "http://[::1", {"Authorization": "Bearer t"}
        )  # must not raise


# ---------------------------------------------------------------------------
# R3a regression guard, do not delete (see pitfall #8 in the remediation prompt):
# pins the httpx behaviour D4 depends on -- if a future httpx release changes
# redirect semantics, this fails instead of the token silently shipping to a CDN.
# ---------------------------------------------------------------------------


class TestRedirectCredentialHandling:
    async def test_cross_origin_absolute_redirect_strips_auth_and_drops_query_token(self) -> None:
        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            if len(captured) == 1:
                return httpx.Response(
                    302,
                    headers={"Location": "https://cdn.example-r2.com/blob/abc?X-Amz-Signature=sig"},
                )
            return httpx.Response(200)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport, follow_redirects=True) as client:
            await client.get(
                "https://civitai.red/api/download/models/1?token=SECRET123",
                headers={"Authorization": "Bearer SECRET123", "Range": "bytes=0-0"},
            )

        assert len(captured) == 2
        _first, second = captured
        assert second.url.host == "cdn.example-r2.com"
        assert "SECRET123" not in str(second.url)
        assert "Authorization" not in second.headers
        assert second.headers.get("Range") == "bytes=0-0"

    async def test_protocol_relative_redirect_strips_auth_and_drops_query_token(self) -> None:
        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            if len(captured) == 1:
                return httpx.Response(302, headers={"Location": "//othercdn.example/blob"})
            return httpx.Response(200)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport, follow_redirects=True) as client:
            await client.get(
                "https://civitai.red/dl2?token=SECRET123",
                headers={"Authorization": "Bearer SECRET123", "Range": "bytes=0-0"},
            )

        assert len(captured) == 2
        _first, second = captured
        assert second.url.host == "othercdn.example"
        assert "SECRET123" not in str(second.url)
        assert "Authorization" not in second.headers
        assert second.headers.get("Range") == "bytes=0-0"

    async def test_relative_same_host_redirect_drops_query_token_regardless_of_origin(
        self,
    ) -> None:
        """A relative Location never carries the original query string forward --
        httpx rebuilds the redirect URL from Location alone. Authorization is
        preserved here because same-host redirects are same-origin, which is
        correct and not a leak."""
        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            if len(captured) == 1:
                return httpx.Response(302, headers={"Location": "/redirected/path"})
            return httpx.Response(200)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport, follow_redirects=True) as client:
            await client.get(
                "https://civitai.red/dl?token=SECRET123",
                headers={"Authorization": "Bearer SECRET123", "Range": "bytes=0-0"},
            )

        assert len(captured) == 2
        _first, second = captured
        assert second.url.host == "civitai.red"
        assert second.url.path == "/redirected/path"
        assert "SECRET123" not in str(second.url)
        assert second.headers.get("Range") == "bytes=0-0"
