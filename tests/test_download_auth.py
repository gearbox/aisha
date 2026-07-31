"""Tests for download_auth: pure per-host auth-policy resolution and token attachment."""

from __future__ import annotations

import logging
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

from ai_content_service.config import Settings
from ai_content_service.download_auth import (
    AuthTransport,
    BoundCredential,
    CredentialEgressError,
    HostAuthPolicy,
    apply_auth,
    assert_no_credential_egress,
    attempt_with_auth,
    build_credentials,
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

    def test_empty_string_token_leaves_url_and_headers_unchanged(
        self, civitai_policy: HostAuthPolicy
    ) -> None:
        """E3a: apply_auth guards on truthiness, not `is None` -- defence in
        depth even if a caller passes an unsanitised blank token."""
        url = "https://civitai.red/api/download/models/123"
        new_url, headers = apply_auth(civitai_policy, AuthTransport.BEARER_HEADER, url, {}, "")

        assert new_url == url
        assert headers == {}
        assert "Authorization" not in headers

    def test_empty_string_token_query_transport_leaves_url_unchanged(
        self, civitai_policy: HostAuthPolicy
    ) -> None:
        url = "https://civitai.red/api/download/models/123"
        new_url, _headers = apply_auth(civitai_policy, AuthTransport.QUERY_TOKEN, url, {}, "")

        assert new_url == url
        assert "token" not in new_url

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

    def test_secret_masked_despite_non_matching_parameter_name(self) -> None:
        """MY-5a: a token we hold must be masked regardless of what query
        parameter name it rides under."""
        redacted = redact_url("https://example.com/x?key=OURTOKEN", secrets=["OURTOKEN"])
        assert "OURTOKEN" not in redacted
        assert "***" in redacted

    def test_bundle_own_token_still_masked_by_name_pass_when_not_in_secrets(self) -> None:
        """A bundle's own embedded ?token= is not one of our secrets, but the
        name-based backstop must still mask it."""
        redacted = redact_url(
            "https://mirror.example/x?token=BUNDLE_OWN_SECRET", secrets=["OURTOKEN"]
        )
        assert "BUNDLE_OWN_SECRET" not in redacted
        assert "token=***" in redacted

    def test_no_secrets_argument_behaves_exactly_as_before(self) -> None:
        url = "https://civitai.com/api/download/models/1?token=SUPER_SECRET"
        assert redact_url(url) == redact_url(url, secrets=())

    def test_short_secret_below_floor_leaves_unrelated_text_untouched(self) -> None:
        """MY-7a: a short token used as a value-based secret must not corrupt
        unrelated text it happens to appear as a substring of."""
        url = "https://cdn-lfs.huggingface.co/repos/model-v1.safetensors"
        assert redact_url(url, secrets=["1"]) == url

    def test_secret_exactly_at_floor_is_still_redacted(self) -> None:
        redacted = redact_url("https://example.com/x?key=8CHARSEC", secrets=["8CHARSEC"])
        assert "8CHARSEC" not in redacted


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
# build_credentials (MY-8a, MY-7b)
# ---------------------------------------------------------------------------


class TestBuildCredentials:
    @pytest.fixture
    def registry(self) -> tuple[HostAuthPolicy, ...]:
        return build_registry(Settings())

    def test_pairs_each_policy_with_its_token(self, registry: tuple[HostAuthPolicy, ...]) -> None:
        credentials = build_credentials(
            registry, {"huggingface": "hf_token_value", "civitai": "civitai_token_value"}
        )
        by_policy = {c.policy.name: c.token for c in credentials}
        assert by_policy == {"huggingface": "hf_token_value", "civitai": "civitai_token_value"}

    def test_blank_tokens_are_dropped(self, registry: tuple[HostAuthPolicy, ...]) -> None:
        credentials = build_credentials(registry, {"huggingface": None, "civitai": "tok"})
        assert [c.policy.name for c in credentials] == ["civitai"]

    def test_no_tokens_yields_no_credentials(self, registry: tuple[HostAuthPolicy, ...]) -> None:
        credentials = build_credentials(registry, {})
        assert credentials == ()

    def test_short_token_warns_with_policy_and_length_not_value(
        self, registry: tuple[HostAuthPolicy, ...], caplog: pytest.LogCaptureFixture
    ) -> None:
        """MY-7b: the operator is told a token looks malformed, but the value
        itself must never appear in the log."""
        with caplog.at_level(logging.WARNING, logger="ai_content_service.download_auth"):
            credentials = build_credentials(registry, {"civitai": "abcde"})

        assert len(credentials) == 1
        assert credentials[0].token == "abcde"

        warnings = [r for r in caplog.records if "auth.token.too_short" in r.getMessage()]
        assert len(warnings) == 1
        record = warnings[0]
        assert "abcde" not in record.getMessage()
        # structlog's stdlib bridge (see conftest) packs the event dict into
        # `record.msg` rather than individual LogRecord attributes.
        event = record.msg
        assert isinstance(event, dict)
        assert event["policy"] == "civitai"
        assert event["length"] == 5

    def test_long_token_does_not_warn(
        self, registry: tuple[HostAuthPolicy, ...], caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="ai_content_service.download_auth"):
            build_credentials(registry, {"civitai": "a_perfectly_long_token"})

        assert not any("auth.token.too_short" in r.getMessage() for r in caplog.records)


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
    """R3b: the guard keys on credential *values*, not parameter/header names.

    01r's version keyed on names (`token=`, `Authorization` present at all)
    and broke both a direct URL that merely happens to use `?token=` (E1) and
    a provider's own presigned redirect that reuses the same name (MY-4).

    E4a: each credential is checked against its own issuing policy, not
    against whichever policy happens to match the destination -- `02r`'s flat
    `secrets: Iterable[str]` made that distinction impossible to enforce.
    """

    @pytest.fixture
    def civitai_policy(self) -> HostAuthPolicy:
        return HostAuthPolicy(
            name="civitai",
            domains=("civitai.com", "civitai.red", "civitai.green"),
            primary=AuthTransport.BEARER_HEADER,
            fallback=AuthTransport.QUERY_TOKEN,
        )

    @pytest.fixture
    def hf_policy(self) -> HostAuthPolicy:
        return HostAuthPolicy(
            name="huggingface",
            domains=("huggingface.co", "hf.co"),
            primary=AuthTransport.BEARER_HEADER,
            fallback=None,
        )

    @pytest.fixture
    def our_civitai_cred(self, civitai_policy: HostAuthPolicy) -> BoundCredential:
        return BoundCredential(civitai_policy, "OUR_TOKEN")

    # -- E1: a credential that isn't ours must never be blocked -------------

    @pytest.mark.parametrize("param_name", ["token", "api_key", "access_token"])
    def test_e1_foreign_credential_on_unregistered_host_never_raises(
        self, our_civitai_cred: BoundCredential, param_name: str
    ) -> None:
        """A direct/presigned URL's own credential, under any of the
        sensitive-looking param names, is not ours and is none of our
        business — regardless of param name."""
        assert_no_credential_egress(
            f"https://mirror.internal.example/weights/model.safetensors?{param_name}=PRESIGNED",
            {},
            [our_civitai_cred],
        )  # must not raise

    # -- still raises: our exact value, wherever it appears ------------------

    def test_our_token_in_authorization_header_on_foreign_host_raises(
        self, our_civitai_cred: BoundCredential
    ) -> None:
        with pytest.raises(CredentialEgressError):
            assert_no_credential_egress(
                "https://cdn.example.com/x",
                {"Authorization": "Bearer OUR_TOKEN"},
                [our_civitai_cred],
            )

    def test_our_token_in_query_param_on_foreign_host_raises(
        self, our_civitai_cred: BoundCredential
    ) -> None:
        with pytest.raises(CredentialEgressError):
            assert_no_credential_egress(
                "https://evil.com/x?token=OUR_TOKEN",
                {},
                [our_civitai_cred],
            )

    def test_our_token_embedded_in_path_segment_raises(
        self, our_civitai_cred: BoundCredential
    ) -> None:
        with pytest.raises(CredentialEgressError):
            assert_no_credential_egress(
                "https://evil.com/creds/OUR_TOKEN/blob",
                {},
                [our_civitai_cred],
            )

    def test_raises_for_lookalike_domain(self, our_civitai_cred: BoundCredential) -> None:
        """The look-alike-host guard must survive the value-based rewrite."""
        with pytest.raises(CredentialEgressError):
            assert_no_credential_egress(
                "https://civitai.com.evil.com/x",
                {"Authorization": "Bearer OUR_TOKEN"},
                [our_civitai_cred],
            )

    # -- passes: our value, on our own domains -------------------------------

    @pytest.mark.parametrize("host", ["civitai.red", "www.civitai.red", "civitai.green"])
    def test_our_token_on_civitai_domains_passes(
        self, our_civitai_cred: BoundCredential, host: str
    ) -> None:
        assert_no_credential_egress(
            f"https://{host}/x",
            {"Authorization": "Bearer OUR_TOKEN"},
            [our_civitai_cred],
        )  # must not raise

    @pytest.mark.parametrize("host", ["cdn-lfs.huggingface.co", "cas-bridge.xethub.hf.co"])
    def test_hf_token_on_hf_domains_passes(self, hf_policy: HostAuthPolicy, host: str) -> None:
        assert_no_credential_egress(
            f"https://{host}/x",
            {"Authorization": "Bearer HF_TOKEN"},
            [BoundCredential(hf_policy, "HF_TOKEN")],
        )  # must not raise

    def test_no_credential_present_never_raises_even_on_foreign_host(
        self, our_civitai_cred: BoundCredential
    ) -> None:
        assert_no_credential_egress("https://cdn.example.com/x", {}, [our_civitai_cred])  # no raise

    def test_credential_on_unregistered_host_raises(
        self, our_civitai_cred: BoundCredential
    ) -> None:
        with pytest.raises(CredentialEgressError):
            assert_no_credential_egress(
                "https://totally-unknown-host.example/x",
                {"Authorization": "Bearer OUR_TOKEN"},
                [our_civitai_cred],
            )

    # -- E4: a credential is checked against its OWN issuing policy ---------

    def test_civitai_token_on_huggingface_host_raises(self, civitai_policy: HostAuthPolicy) -> None:
        """E4: our Civitai token must not be allowed onto huggingface.co just
        because huggingface.co matches *some* policy in the registry."""
        civitai_cred = BoundCredential(civitai_policy, "CIV_TOKEN_BBB")
        with pytest.raises(CredentialEgressError):
            assert_no_credential_egress(
                "https://huggingface.co/repos/x/model.safetensors?token=CIV_TOKEN_BBB",
                {},
                [civitai_cred],
            )

    def test_hf_token_on_civitai_host_raises(self, hf_policy: HostAuthPolicy) -> None:
        """E4, reverse direction."""
        hf_cred = BoundCredential(hf_policy, "HF_TOKEN_AAA")
        with pytest.raises(CredentialEgressError):
            assert_no_credential_egress(
                "https://civitai.red/api/download/models/1?token=HF_TOKEN_AAA",
                {},
                [hf_cred],
            )

    def test_each_credential_still_passes_on_its_own_domain_when_both_configured(
        self, civitai_policy: HostAuthPolicy, hf_policy: HostAuthPolicy
    ) -> None:
        """Both tokens configured simultaneously: each is fine on its own
        provider and only its own provider."""
        credentials = [
            BoundCredential(civitai_policy, "CIV_TOKEN_BBB"),
            BoundCredential(hf_policy, "HF_TOKEN_AAA"),
        ]
        assert_no_credential_egress(
            "https://civitai.red/x", {"Authorization": "Bearer CIV_TOKEN_BBB"}, credentials
        )  # must not raise
        assert_no_credential_egress(
            "https://huggingface.co/x", {"Authorization": "Bearer HF_TOKEN_AAA"}, credentials
        )  # must not raise

    def test_message_names_the_offending_policy(self, civitai_policy: HostAuthPolicy) -> None:
        civitai_cred = BoundCredential(civitai_policy, "CIV_TOKEN_BBB")
        with pytest.raises(CredentialEgressError, match="civitai"):
            assert_no_credential_egress(
                "https://huggingface.co/x?token=CIV_TOKEN_BBB", {}, [civitai_cred]
            )

    def test_multi_offender_message_names_both_policies(
        self, civitai_policy: HostAuthPolicy, hf_policy: HostAuthPolicy
    ) -> None:
        """Pitfall #5: if both a Civitai and an HF credential somehow match
        the same foreign request, both offending policy names must appear —
        not just the first."""
        shared = "SHARED_VALUE_TOKEN"
        credentials = [
            BoundCredential(civitai_policy, shared),
            BoundCredential(hf_policy, shared),
        ]
        with pytest.raises(CredentialEgressError) as exc_info:
            assert_no_credential_egress(f"https://evil.example/x?token={shared}", {}, credentials)
        msg = str(exc_info.value)
        assert "civitai" in msg
        assert "huggingface" in msg
        assert shared not in msg

    # -- no-op / fail-open discipline ----------------------------------------

    def test_empty_credentials_is_noop_even_with_authorization_present(self) -> None:
        """R3c: with no tokens configured, the guard is a no-op by design —
        the download will separately fail on auth, which is a distinct
        signal (pitfall #4)."""
        assert_no_credential_egress(
            "https://evil.example/x", {"Authorization": "Bearer whatever"}
        )  # credentials defaults to () -- must not raise

    def test_blank_token_credentials_are_filtered_out(self, civitai_policy: HostAuthPolicy) -> None:
        assert_no_credential_egress(
            "https://evil.example/x",
            {"Authorization": "Bearer whatever"},
            [BoundCredential(civitai_policy, "")],
        )  # must not raise

    def test_malformed_url_without_matching_credential_does_not_raise(
        self, our_civitai_cred: BoundCredential
    ) -> None:
        """A syntactically malformed URL must not crash a probe that carries
        no matching credential -- OUR_TOKEN appears nowhere in headers/url,
        so no match is ever established regardless of the parse failure."""
        assert_no_credential_egress("http://[::1", {}, [our_civitai_cred])  # must not raise

    def test_non_ascii_token_still_detected_via_fallback_compare(
        self, civitai_policy: HostAuthPolicy
    ) -> None:
        """hmac.compare_digest raises TypeError on non-ASCII str operands;
        the guard must fall back to a plain compare rather than let that
        TypeError propagate and defeat detection (pitfall #2)."""
        token = "tökén-secret"
        with pytest.raises(CredentialEgressError):
            assert_no_credential_egress(
                "https://evil.example/x",
                {"Authorization": f"Bearer {token}"},
                [BoundCredential(civitai_policy, token)],
            )

    def test_error_message_never_contains_the_secret(self, civitai_policy: HostAuthPolicy) -> None:
        with pytest.raises(CredentialEgressError) as exc_info:
            assert_no_credential_egress(
                "https://evil.example/x",
                {"Authorization": "Bearer OUR_SUPER_SECRET_TOKEN"},
                [BoundCredential(civitai_policy, "OUR_SUPER_SECRET_TOKEN")],
            )
        assert "OUR_SUPER_SECRET_TOKEN" not in str(exc_info.value)

    # -- MY-7a: entropy floor on the substring scan --------------------------

    def test_short_token_credential_free_cdn_url_does_not_raise(
        self, civitai_policy: HostAuthPolicy
    ) -> None:
        """A short token must not turn a credential-free CDN URL into a
        false-positive credential-egress failure, even if the token's
        characters happen to appear in it."""
        short_cred = BoundCredential(civitai_policy, "1234")
        assert_no_credential_egress(
            "https://cdn-b2.example/file/blob-1234/model.safetensors",
            {},
            [short_cred],
        )  # must not raise -- below the substring-scan floor

    def test_short_token_exact_query_value_match_still_raises(
        self, civitai_policy: HostAuthPolicy
    ) -> None:
        """Exact query-value comparison applies at any length -- only the
        substring scan has a floor."""
        short_cred = BoundCredential(civitai_policy, "1234")
        with pytest.raises(CredentialEgressError):
            assert_no_credential_egress("https://evil.example/x?token=1234", {}, [short_cred])

    def test_short_token_exact_bearer_match_still_raises(
        self, civitai_policy: HostAuthPolicy
    ) -> None:
        short_cred = BoundCredential(civitai_policy, "1234")
        with pytest.raises(CredentialEgressError):
            assert_no_credential_egress(
                "https://evil.example/x", {"Authorization": "Bearer 1234"}, [short_cred]
            )

    # -- MY-9a: fail closed once a credential is matched ---------------------

    def test_urlparse_failure_with_matching_credential_raises_not_returns(
        self, civitai_policy: HostAuthPolicy, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A credential in the Authorization header (which is not
        URL-derived), with `urlparse` forced to raise, must still raise
        CredentialEgressError -- silently allowing a credential-bearing
        request whose host can't be resolved is exactly the fail-open bug
        MY-9a closes."""
        import ai_content_service.download_auth as download_auth_module

        def _boom(_url: str) -> None:
            raise ValueError("boom")

        monkeypatch.setattr(download_auth_module, "urlparse", _boom)

        with pytest.raises(CredentialEgressError):
            assert_no_credential_egress(
                "https://anything/x",
                {"Authorization": "Bearer OUR_TOKEN"},
                [BoundCredential(civitai_policy, "OUR_TOKEN")],
            )


class TestValueBasedGuardEndToEnd:
    """MY-4: the redirect chain that would have caught 01r's design error
    before it shipped."""

    async def test_civitai_redirect_to_provider_presigned_url_completes(self) -> None:
        """civitai.red redirects to a CDN whose presigned URL happens to use
        the same `token=` name as ours, with a different value. The chain
        must complete -- the provider's credential is none of our business."""
        registry = build_registry(Settings())
        civitai_policy = next(p for p in registry if p.name == "civitai")
        credentials = (BoundCredential(civitai_policy, "OUR_BEARER_TOKEN"),)

        async def guard(request: httpx.Request) -> None:
            assert_no_credential_egress(str(request.url), request.headers, credentials)

        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            if len(captured) == 1:
                return httpx.Response(
                    302,
                    headers={"Location": "https://cdn-b2.example/blob?token=PROVIDER_PRESIGNED"},
                )
            return httpx.Response(200)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(
            transport=transport, follow_redirects=True, event_hooks={"request": [guard]}
        ) as client:
            response = await client.get(
                "https://civitai.red/api/download/models/1",
                headers={"Authorization": f"Bearer {credentials[0].token}"},
            )

        assert response.status_code == 200
        assert len(captured) == 2
        assert captured[1].url.host == "cdn-b2.example"
        assert "token=PROVIDER_PRESIGNED" in str(captured[1].url)

    async def test_civitai_redirect_to_huggingface_with_civitai_token_raises(self) -> None:
        """E4 reachability: a compromised/misbehaving redirect target that
        happens to be a *different* provider we also hold a credential for
        must not receive that credential -- even though huggingface.co is a
        host this registry knows about."""
        registry = build_registry(Settings())
        civitai_policy = next(p for p in registry if p.name == "civitai")
        credentials = (BoundCredential(civitai_policy, "OUR_CIVITAI_TOKEN"),)

        async def guard(request: httpx.Request) -> None:
            assert_no_credential_egress(str(request.url), request.headers, credentials)

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.host == "civitai.red":
                return httpx.Response(
                    302,
                    headers={"Location": "https://huggingface.co/repos/x?token=OUR_CIVITAI_TOKEN"},
                )
            return httpx.Response(200)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(
            transport=transport, follow_redirects=True, event_hooks={"request": [guard]}
        ) as client:
            with pytest.raises(CredentialEgressError):
                await client.get(
                    "https://civitai.red/api/download/models/1",
                    headers={"Authorization": "Bearer OUR_CIVITAI_TOKEN"},
                )


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
