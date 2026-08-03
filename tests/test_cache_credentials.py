"""Tests for cache-write credential policies."""

from unittest.mock import MagicMock

import httpx
import pytest

from ai_content_service.cache_credentials import (
    ApexCacheCredentialProvider,
    CacheCredentialError,
    StaticCacheCredentialProvider,
)
from ai_content_service.r2_transfer import R2WriteCreds


def test_static_provider_derives_key_and_skips_finalize() -> None:
    digest = "a" * 64
    creds = R2WriteCreds(access_key_id="KEY", secret_access_key="SECRET")
    provider = StaticCacheCredentialProvider(creds)

    minted = provider.mint(
        sha256=digest,
        filename="model.safetensors",
        model_type="checkpoints",
        source_url="https://example.com/model.safetensors",
    )

    assert minted.r2_key == f"models/by-sha256/{digest}"
    assert minted.creds is creds
    provider.finalize(sha256=digest, size_bytes=42)
    provider.close()


def test_apex_provider_preserves_mint_and_finalize_payloads() -> None:
    digest = "b" * 64
    mint_response = MagicMock()
    mint_response.json.return_value = {
        "r2_key": f"models/by-sha256/{digest}",
        "credentials": {"access_key_id": "TEMP", "secret_access_key": "SECRET"},
    }
    finalize_response = MagicMock(status_code=200)
    client = MagicMock()
    client.post.side_effect = [mint_response, finalize_response]
    provider = ApexCacheCredentialProvider(
        base_url="https://api.example.com/",
        admin_token="ADMIN",
        client=client,
    )

    minted = provider.mint(
        sha256=digest,
        filename="model.safetensors",
        model_type="checkpoints",
        source_url="https://example.com/model.safetensors",
    )
    provider.finalize(sha256=digest, size_bytes=42)

    assert minted.creds.access_key_id == "TEMP"
    assert (
        client.post.call_args_list[0].args[0]
        == "https://api.example.com/v1/admin/model-cache/credentials"
    )
    assert client.post.call_args_list[0].kwargs["json"]["sha256"] == digest
    assert client.post.call_args_list[1].kwargs["json"] == {"sha256": digest, "size_bytes": 42}


@pytest.mark.parametrize(
    "payload",
    [
        {"credentials": {"access_key_id": "KEY", "secret_access_key": "SECRET"}},
        {
            "r2_key": 42,
            "credentials": {"access_key_id": "KEY", "secret_access_key": "SECRET"},
        },
        {
            "r2_key": "models/by-sha256/" + "0" * 64,
            "credentials": {"access_key_id": "KEY", "secret_access_key": "SECRET"},
        },
        {
            "r2_key": "models/by-sha256/" + "a" * 64,
            "credentials": {"access_key_id": "", "secret_access_key": "SECRET"},
        },
    ],
)
def test_apex_provider_rejects_invalid_or_mismatched_credential_response(payload: object) -> None:
    response = MagicMock()
    response.json.return_value = payload
    client = MagicMock()
    client.post.return_value = response
    provider = ApexCacheCredentialProvider(
        base_url="https://api.example.com",
        admin_token="ADMIN",
        client=client,
    )

    with pytest.raises(CacheCredentialError, match="Apex credentials response invalid"):
        provider.mint(
            sha256="a" * 64,
            filename="model.safetensors",
            model_type="checkpoints",
            source_url="https://example.com/model.safetensors",
        )


def test_apex_provider_rejects_invalid_session_token_type() -> None:
    response = MagicMock()
    response.json.return_value = {
        "r2_key": "models/by-sha256/" + "a" * 64,
        "credentials": {
            "access_key_id": "KEY",
            "secret_access_key": "SECRET",
            "session_token": 1,
        },
    }
    client = MagicMock()
    client.post.return_value = response
    provider = ApexCacheCredentialProvider(
        base_url="https://api.example.com", admin_token="ADMIN", client=client
    )

    with pytest.raises(CacheCredentialError, match="session_token"):
        provider.mint(
            sha256="a" * 64,
            filename="model.safetensors",
            model_type="checkpoints",
            source_url="https://example.com/model.safetensors",
        )


def test_apex_provider_closes_only_owned_client() -> None:
    borrowed = MagicMock()
    ApexCacheCredentialProvider(
        base_url="https://api.example.com", admin_token="ADMIN", client=borrowed
    ).close()
    borrowed.close.assert_not_called()

    owned = MagicMock()
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            "ai_content_service.cache_credentials.httpx.Client", lambda **_kwargs: owned
        )
        provider = ApexCacheCredentialProvider(
            base_url="https://api.example.com", admin_token="ADMIN"
        )
    provider.close()
    owned.close.assert_called_once()


@pytest.mark.parametrize("status_code", [409, 422])
def test_apex_provider_surfaces_finalize_rejections(status_code: int) -> None:
    response = MagicMock(status_code=status_code, text="rejected")
    client = MagicMock()
    client.post.return_value = response
    provider = ApexCacheCredentialProvider(
        base_url="https://api.example.com",
        admin_token="ADMIN",
        client=client,
    )

    with pytest.raises(CacheCredentialError, match=str(status_code)):
        provider.finalize(sha256="a" * 64, size_bytes=1)


def test_apex_provider_wraps_http_failures() -> None:
    request = httpx.Request("POST", "https://api.example.com")
    response = httpx.Response(401, request=request)
    client = MagicMock()
    client.post.return_value.raise_for_status.side_effect = httpx.HTTPStatusError(
        "unauthorized", request=request, response=response
    )
    provider = ApexCacheCredentialProvider(
        base_url="https://api.example.com",
        admin_token="ADMIN",
        client=client,
    )

    with pytest.raises(CacheCredentialError, match="credentials request failed"):
        provider.mint(
            sha256="a" * 64,
            filename="model.safetensors",
            model_type="checkpoints",
            source_url="https://example.com/model.safetensors",
        )


def test_apex_provider_reports_invalid_digest_as_caller_input() -> None:
    client = MagicMock()
    provider = ApexCacheCredentialProvider(
        base_url="https://api.example.com",
        admin_token="ADMIN",
        client=client,
    )

    with pytest.raises(ValueError):
        provider.mint(
            sha256="not-a-digest",
            filename="model.safetensors",
            model_type="checkpoints",
            source_url="https://example.com/model.safetensors",
        )

    client.post.assert_not_called()
