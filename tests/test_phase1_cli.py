"""Command-level regression tests for Phase 1 authoring commands."""

from __future__ import annotations

import json
from io import StringIO
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

from rich.console import Console
from typer.testing import CliRunner

from ai_content_service import cache_service
from ai_content_service.bundle_contract import ContractReport, Finding, Severity
from ai_content_service.cli import _render_contract_reports, app
from ai_content_service.config import Settings, reset_settings
from ai_content_service.models_service import FetchedModel

if TYPE_CHECKING:
    from pathlib import Path

    from pytest import MonkeyPatch


runner = CliRunner()


def _bundle_settings(tmp_path: Path, *, yaml_text: str | None = None) -> Settings:
    bundles = tmp_path / "bundles"
    version = bundles / "demo" / "260101-01"
    version.mkdir(parents=True)
    (bundles / "demo" / "current").symlink_to("260101-01")
    (version / "bundle.yaml").write_text(
        yaml_text
        or (
            "metadata:\n  name: demo\n  version: '260101-01'\nmodels:\n"
            "  - name: model\n    model_type: checkpoints\n    files:\n"
            "      - name: model\n        url: https://example.com/model\n        filename: model\n"
        )
    )
    return Settings(
        comfyui_path=tmp_path / "ComfyUI",
        bundles_path=bundles,
        apex_base_url="https://api.example.com",
        apex_admin_token="admin",  # type: ignore[arg-type]
        r2_s3_endpoint="https://account.r2.cloudflarestorage.com",
    )


def test_bundle_validate_all_rejects_empty_registry_and_allow_empty_is_explicit(
    tmp_path: Path,
) -> None:
    settings = Settings(bundles_path=tmp_path / "bundles")
    settings.bundles_path.mkdir()
    with patch("ai_content_service.cli.get_settings", return_value=settings):
        failed = runner.invoke(app, ["bundle", "validate", "--all"])
        allowed = runner.invoke(app, ["bundle", "validate", "--all", "--allow-empty"])

    assert failed.exit_code == 1
    assert "no bundles found" in failed.output.lower()
    assert "contract is valid" not in failed.output.lower()
    assert allowed.exit_code == 0
    assert "no bundles found" in allowed.output.lower()


def test_cache_push_does_not_construct_provider_for_malformed_yaml(tmp_path: Path) -> None:
    settings = _bundle_settings(tmp_path, yaml_text="metadata: [broken")
    with (
        patch("ai_content_service.cli.get_settings", return_value=settings),
        patch("ai_content_service.cli.ApexCacheCredentialProvider") as provider,
        patch("ai_content_service.cache_service.push_models") as push,
    ):
        result = runner.invoke(app, ["cache", "push", "demo", "--all"])

    assert result.exit_code == 1
    assert "invalid bundle config" in result.output.lower()
    provider.assert_not_called()
    push.assert_not_called()


def test_cache_push_closes_provider_after_success_and_failure(tmp_path: Path) -> None:
    settings = _bundle_settings(tmp_path)
    for report in (
        cache_service.PushReport([cache_service.PushFileResult("model", ok=True)]),
        cache_service.PushReport([cache_service.PushFileResult("model", ok=False, detail="bad")]),
    ):
        provider = MagicMock()
        provider.name = "apex"
        with (
            patch("ai_content_service.cli.get_settings", return_value=settings),
            patch("ai_content_service.cli.ApexCacheCredentialProvider", return_value=provider),
            patch("ai_content_service.cache_service.push_models", return_value=report),
        ):
            result = runner.invoke(app, ["cache", "push", "demo", "--all"])

        assert result.exit_code == (0 if report.ok else 1)
        provider.close.assert_called_once()


def test_cache_verify_renders_json_and_propagates_failure(tmp_path: Path) -> None:
    settings = _bundle_settings(tmp_path)
    report = cache_service.VerifyReport(
        [cache_service.VerifyFileResult("model", "key", False, "MISSING")]
    )
    with (
        patch("ai_content_service.cli.get_settings", return_value=settings),
        patch(
            "ai_content_service.cli.verify_cache_targets",
            new=AsyncMock(return_value=report),
        ),
    ):
        result = runner.invoke(app, ["cache", "verify", "demo", "--all", "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["ok"] is False
    assert payload["results"][0]["status"] == "MISSING"


def test_cache_verify_renders_configuration_error_as_json(tmp_path: Path) -> None:
    settings = _bundle_settings(tmp_path)
    report = cache_service.VerifyReport(
        [],
        configuration_error="ACS_R2_READONLY_ACCESS_KEY_ID is not set",
    )
    with (
        patch("ai_content_service.cli.get_settings", return_value=settings),
        patch(
            "ai_content_service.cli.verify_cache_targets",
            new=AsyncMock(return_value=report),
        ),
    ):
        result = runner.invoke(app, ["cache", "verify", "demo", "--all", "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload == {
        "ok": False,
        "results": [],
        "configuration_error": "ACS_R2_READONLY_ACCESS_KEY_ID is not set",
    }


def test_models_fetch_renders_sanitized_service_fragment(tmp_path: Path) -> None:
    settings = _bundle_settings(tmp_path)
    fragment = "- name: model\n  url: https://civitai.com/model?a=1\n"
    fetched = FetchedModel("a" * 64, 42, tmp_path / "model", fragment)
    with (
        patch("ai_content_service.cli.get_settings", return_value=settings),
        patch("ai_content_service.cli.fetch_model", new=AsyncMock(return_value=fetched)),
    ):
        result = runner.invoke(
            app,
            [
                "models",
                "fetch",
                "--url",
                "https://civitai.com/model?token=secret&a=1",
                "--model-type",
                "checkpoints",
                "--filename",
                "model",
            ],
        )

    assert result.exit_code == 0
    assert "secret" not in result.output
    assert fragment in result.output


def test_contract_renderer_confirms_warnings_only_validation(monkeypatch: MonkeyPatch) -> None:
    output = StringIO()
    warning = Finding(Severity.WARNING, "metadata.tested_false", "Not tested", "bundle.yaml")
    report = ContractReport("demo", (warning,))
    console = Console(file=output, force_terminal=False)
    monkeypatch.setattr("ai_content_service.cli.console", console)

    _render_contract_reports((report,), json_output=False)

    assert "Bundle contract is valid (1 warnings)" in output.getvalue()


def teardown_module() -> None:
    reset_settings()
