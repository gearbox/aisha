"""Tests for configuration models."""

from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from ai_content_service.config import (
    BundleConfig,
    BundleMetadata,
    BundleVersion,
    ComfyUIConfig,
    CustomNode,
    DeploymentPlan,
    DeployMode,
    ModelConfig,
    ModelFileConfig,
    Settings,
    unwrap_secret,
)
from ai_content_service.download_auth import build_credentials, build_registry


class TestUnwrapSecret:
    """Tests for E3 (config.py unwrap_secret) — a blank secret is no secret."""

    def test_none_stays_none(self) -> None:
        assert unwrap_secret(None) is None

    def test_empty_string_becomes_none(self) -> None:
        assert unwrap_secret(SecretStr("")) is None

    def test_whitespace_only_becomes_none(self) -> None:
        assert unwrap_secret(SecretStr("   ")) is None

    def test_non_blank_value_returned_unchanged(self) -> None:
        assert unwrap_secret(SecretStr("tok")) == "tok"

    def test_surrounding_whitespace_is_not_stripped(self) -> None:
        """Only the emptiness check uses .strip() -- a token with meaningful
        surrounding whitespace must pass through unmutated."""
        assert unwrap_secret(SecretStr("  tok  ")) == "  tok  "

    def test_empty_civitai_token_yields_no_credential_end_to_end(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ACS_CIVITAI_API_TOKEN", "")
        settings = Settings()
        registry = build_registry(settings)
        tokens = {
            "huggingface": unwrap_secret(settings.hf_token),
            "civitai": unwrap_secret(settings.civitai_api_token),
        }
        credentials = build_credentials(registry, tokens)
        assert all(c.policy.name != "civitai" for c in credentials)


class TestCustomNode:
    """Tests for CustomNode model."""

    def test_valid_custom_node(self) -> None:
        """Test creating a valid custom node."""
        node = CustomNode(
            name="ComfyUI-GGUF",
            git_url="https://github.com/city96/ComfyUI-GGUF",
            commit_sha="a" * 40,
        )
        assert node.name == "ComfyUI-GGUF"
        assert node.source == "git"
        assert node.commit_sha == "a" * 40

    @pytest.mark.parametrize(
        "name", ["..", ".", "", "   ", "a/b", "a\\b", "/abs", "../../etc", "-node"]
    )
    def test_name_rejects_unsafe_directory_components(self, name: str) -> None:
        with pytest.raises(ValidationError, match="custom node name"):
            CustomNode(name=name, git_url="https://github.com/example/node", commit_sha="a" * 40)

    @pytest.mark.parametrize("name", ["ok-node", "ComfyUI-KJNodes", "comfyui_kjnodes"])
    def test_name_accepts_safe_directory_components(self, name: str) -> None:
        node = CustomNode(name=name, git_url="https://github.com/example/node", commit_sha="a" * 40)

        assert node.name == name


class TestReadinessMarkerConfig:
    """Bundle-level validation for Apex's optional readiness evidence."""

    def _bundle(self, marker: object | None) -> dict[str, object]:
        bundle: dict[str, object] = {
            "metadata": {"name": "test", "version": "260101-01"},
        }
        if marker is not None:
            bundle["readiness_marker"] = marker
        return bundle

    def test_readiness_marker_is_optional_and_validated_when_present(self) -> None:
        assert BundleConfig.model_validate(self._bundle(None)).readiness_marker is None
        marker = BundleConfig.model_validate(self._bundle({"node_class": "KSampler"}))
        assert marker.readiness_marker is not None
        assert marker.readiness_marker.node_class == "KSampler"

    @pytest.mark.parametrize(
        "marker",
        [
            {"node_class": ""},
            {"node_class": "   "},
            {"node_class": "KSampler", "typo": 1},
        ],
    )
    def test_readiness_marker_rejects_empty_or_unknown_values(self, marker: object) -> None:
        with pytest.raises(ValidationError):
            BundleConfig.model_validate(self._bundle(marker))

    def test_readiness_marker_strips_surrounding_whitespace(self) -> None:
        marker = BundleConfig.model_validate(self._bundle({"node_class": " KSampler "}))
        assert marker.readiness_marker is not None
        assert marker.readiness_marker.node_class == "KSampler"

    def test_custom_node_with_commit(self) -> None:
        """Test custom node with pinned commit."""
        node = CustomNode(
            name="ComfyUI-GGUF",
            git_url="https://github.com/city96/ComfyUI-GGUF",
            commit_sha="abc123def",
        )
        assert node.commit_sha == "abc123def"


class TestBundleVersion:
    """Tests for BundleVersion model."""

    def test_valid_version(self) -> None:
        """Test creating a valid bundle version."""
        version = BundleVersion(version="260101-01")
        assert version.version == "260101-01"
        assert str(version) == "260101-01"

    def test_invalid_version_format(self) -> None:
        """Test that invalid version formats are rejected."""
        with pytest.raises(ValidationError):
            BundleVersion(version="2025-01-01")

        with pytest.raises(ValidationError):
            BundleVersion(version="260101")

        with pytest.raises(ValidationError):
            BundleVersion(version="260101-1")

    def test_create_new_first_of_day(self) -> None:
        """Test creating first version of the day."""
        version = BundleVersion.create_new([])
        # Should end with -01
        assert version.version.endswith("-01")

    def test_create_new_increment(self) -> None:
        """Test creating incremented version."""
        # Get today's date prefix
        today = datetime.now(tz=timezone.utc).strftime("%y%m%d")
        existing = [f"{today}-01", f"{today}-02"]

        version = BundleVersion.create_new(existing)
        assert version.version == f"{today}-03"

    def test_create_new_ignores_other_dates(self) -> None:
        """Test that versions from other dates are ignored."""
        today = datetime.now(tz=timezone.utc).strftime("%y%m%d")
        existing = ["240101-01", "240101-02", f"{today}-01"]

        version = BundleVersion.create_new(existing)
        assert version.version == f"{today}-02"

    def test_create_new_skips_gaps_after_deletion(self) -> None:
        """Regression guard: must use max-sequence, not count (collides after deletions)."""
        today = datetime.now(tz=timezone.utc).strftime("%y%m%d")
        existing = [f"{today}-01", f"{today}-03"]

        version = BundleVersion.create_new(existing)
        assert version.version == f"{today}-04"


class TestBundleMetadata:
    """Tests for BundleMetadata model."""

    def test_valid_metadata(self) -> None:
        """Test creating valid metadata."""
        metadata = BundleMetadata(
            name="wan_2.2_i2v",
            version="260101-01",
            description="WAN 2.2 Image to Video",
        )
        assert metadata.name == "wan_2.2_i2v"
        assert metadata.version == "260101-01"
        assert metadata.tested is False

    def test_created_at_default(self) -> None:
        """Test that created_at has a default value."""
        metadata = BundleMetadata(
            name="test",
            version="260101-01",
        )
        assert metadata.created_at is not None
        assert isinstance(metadata.created_at, datetime)


class TestComfyUIConfig:
    """Tests for ComfyUIConfig model."""

    def test_valid_config(self) -> None:
        """Test creating valid ComfyUI config."""
        config = ComfyUIConfig(commit="abc123def456")
        assert config.commit == "abc123def456"
        assert config.repo == "https://github.com/comfyanonymous/ComfyUI"

    def test_custom_repo(self) -> None:
        """Test with custom repo URL."""
        config = ComfyUIConfig(
            repo="https://github.com/fork/ComfyUI",
            commit="abc123",
        )
        assert config.repo == "https://github.com/fork/ComfyUI"


class TestBundleConfig:
    """Tests for BundleConfig model."""

    def test_valid_bundle_config(self) -> None:
        """Test creating a valid bundle config."""
        config = BundleConfig(
            metadata=BundleMetadata(
                name="test",
                version="260101-01",
            ),
            comfyui=ComfyUIConfig(commit="abc123"),
            custom_nodes=[
                CustomNode(
                    name="TestNode",
                    git_url="https://github.com/test/node",
                    commit_sha="def456",
                ),
            ],
        )
        assert config.metadata.name == "test"
        assert config.comfyui is not None
        assert config.comfyui.commit == "abc123"
        assert len(config.custom_nodes) == 1

    def test_custom_nodes_require_commit_sha(self) -> None:
        """Test that custom nodes in bundles must have commit_sha."""
        with pytest.raises(ValidationError) as exc_info:
            BundleConfig(
                metadata=BundleMetadata(
                    name="test",
                    version="260101-01",
                ),
                comfyui=ComfyUIConfig(commit="abc123"),
                custom_nodes=[
                    CustomNode(
                        name="TestNode",
                        git_url="https://github.com/test/node",
                        # Missing commit_sha
                    ),
                ],
            )
        assert "commit_sha" in str(exc_info.value)


class TestCustomNodeSource:
    """P1: CustomNode's ``source`` discriminator (git default, registry addition)."""

    def test_custom_node_source_source_absent_defaults_to_git(self) -> None:
        node = CustomNode.model_validate(
            {
                "name": "ComfyUI-GGUF",
                "git_url": "https://github.com/city96/ComfyUI-GGUF",
                "commit_sha": "a" * 40,
            }
        )
        assert node.source == "git"

    def test_custom_node_source_git_without_commit_sha_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="commit_sha"):
            CustomNode.model_validate(
                {"name": "n", "source": "git", "git_url": "https://github.com/x/y"}
            )

    def test_custom_node_source_registry_with_node_id_and_version_is_valid(self) -> None:
        node = CustomNode.model_validate(
            {
                "name": "comfyui-kjnodes",
                "source": "registry",
                "node_id": "comfyui-kjnodes",
                "version": "1.5.0",
            }
        )
        assert node.commit_sha is None
        assert node.git_url is None
        assert node.node_id == "comfyui-kjnodes"
        assert node.version == "1.5.0"

    def test_custom_node_source_registry_forbids_git_fields(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            CustomNode.model_validate(
                {
                    "name": "n",
                    "source": "registry",
                    "node_id": "n",
                    "version": "1.0.0",
                    "git_url": "https://github.com/x/y",
                    "commit_sha": "a" * 40,
                }
            )
        message = str(exc_info.value)
        assert "git_url" in message
        assert "commit_sha" in message

    def test_custom_node_source_git_forbids_registry_fields(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            CustomNode.model_validate(
                {
                    "name": "n",
                    "source": "git",
                    "git_url": "https://github.com/x/y",
                    "commit_sha": "a" * 40,
                    "node_id": "n",
                    "version": "1.0.0",
                }
            )
        message = str(exc_info.value)
        assert "node_id" in message
        assert "version" in message

    def test_custom_node_source_registry_without_version_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="version"):
            CustomNode.model_validate({"name": "n", "source": "registry", "node_id": "n"})

    def test_custom_node_source_unknown_source_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CustomNode.model_validate(
                {"name": "n", "source": "bogus", "git_url": "https://github.com/x/y"}
            )

    def test_custom_node_source_require_pinned_custom_nodes_accepts_registry_without_commit_sha(
        self,
    ) -> None:
        config = BundleConfig.model_validate(
            {
                "metadata": {"name": "test", "version": "260101-01"},
                "custom_nodes": [
                    {
                        "name": "comfyui-kjnodes",
                        "source": "registry",
                        "node_id": "comfyui-kjnodes",
                        "version": "1.5.0",
                    }
                ],
            }
        )
        assert config.custom_nodes[0].commit_sha is None

    def test_custom_node_source_require_pinned_custom_nodes_still_rejects_unpinned_git(
        self,
    ) -> None:
        with pytest.raises(ValidationError):
            BundleConfig.model_validate(
                {
                    "metadata": {"name": "test", "version": "260101-01"},
                    "custom_nodes": [
                        CustomNode.model_construct(
                            name="n",
                            source="git",
                            git_url="https://github.com/x/y",
                            commit_sha=None,
                            node_id=None,
                            version=None,
                            archive_sha256=None,
                            pip_requirements=[],
                        )
                    ],
                }
            )


class TestSettings:
    """Tests for Settings model."""

    def test_default_settings(self) -> None:
        """Test default settings values."""
        settings = Settings()
        assert settings.comfyui_path == Path("/workspace/ComfyUI")
        assert settings.max_concurrent_downloads == 3
        assert settings.verify_checksums is True
        assert settings.skip_existing is True
        assert settings.no_verify is False

    def test_models_path_property(self) -> None:
        """Test models_path derived property."""
        settings = Settings(comfyui_path=Path("/test/ComfyUI"))
        assert settings.models_path == Path("/test/ComfyUI/models")

    def test_custom_nodes_path_property(self) -> None:
        """Test custom_nodes_path derived property."""
        settings = Settings(comfyui_path=Path("/test/ComfyUI"))
        assert settings.custom_nodes_path == Path("/test/ComfyUI/custom_nodes")

    def test_bundles_path_default(self) -> None:
        """Test default bundles path."""
        settings = Settings()
        assert settings.bundles_path == Path("config/bundles")

    def test_settings_rejects_nonexistent_comfyui_python(self, tmp_path: Path) -> None:
        """Settings must fail-fast if comfyui_python points at a missing file."""
        bogus = tmp_path / "definitely-not-a-real-python"
        with pytest.raises(ValueError, match="comfyui_python"):
            Settings(comfyui_python=bogus)

    def test_settings_accepts_real_python(self, tmp_path: Path) -> None:
        """Settings must accept a comfyui_python path that exists and is a file."""
        real_python = tmp_path / "python"
        real_python.write_text("")
        real_python.chmod(0o755)
        settings = Settings(comfyui_python=real_python)
        assert settings.comfyui_python == real_python

    def test_rclone_max_transfer_seconds_default(self) -> None:
        settings = Settings()
        assert settings.rclone_max_transfer_seconds == 3600

    def test_settings_rclone_max_transfer_seconds_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ACS_RCLONE_MAX_TRANSFER_SECONDS", "120")
        settings = Settings()
        assert settings.rclone_max_transfer_seconds == 120

    def test_civitai_domains_default(self) -> None:
        settings = Settings()
        assert settings.civitai_domains == ("civitai.com", "civitai.red", "civitai.green")

    def test_civitai_domains_env_override_comma_separated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """D2/pitfall #6: a plain comma-separated string, not JSON — must not be JSON-decoded."""
        monkeypatch.setenv("ACS_CIVITAI_DOMAINS", "civitai.com, CIVITAI.RED , ,civitai.green")
        settings = Settings()
        assert settings.civitai_domains == ("civitai.com", "civitai.red", "civitai.green")

    @pytest.mark.parametrize(
        "bad_domains",
        [
            "com",
            "",
            "*.civitai.red",
            "https://civitai.red",
            "civitai.red:443",
            "civitai..red",
            "civitai red",
        ],
    )
    def test_invalid_civitai_domains_are_rejected(self, bad_domains: str) -> None:
        with pytest.raises(ValidationError, match="invalid civitai domain") as exc_info:
            Settings(civitai_domains=bad_domains)  # type: ignore[arg-type]
        assert bad_domains.strip().lower() in str(exc_info.value)

    def test_civitai_allow_query_token_fallback_default_false(self) -> None:
        settings = Settings()
        assert settings.civitai_allow_query_token_fallback is False

    def test_civitai_allow_query_token_fallback_env_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ACS_CIVITAI_ALLOW_QUERY_TOKEN_FALLBACK", "false")
        settings = Settings()
        assert settings.civitai_allow_query_token_fallback is False

    def test_civitai_allow_query_token_fallback_disabled_nulls_policy_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from ai_content_service.download_auth import build_registry

        monkeypatch.setenv("ACS_CIVITAI_ALLOW_QUERY_TOKEN_FALLBACK", "false")
        settings = Settings()
        registry = build_registry(settings)
        civitai = next(p for p in registry if p.name == "civitai")
        assert civitai.fallback is None

    def test_download_user_agent_default_looks_like_a_browser(self) -> None:
        settings = Settings()
        assert "Mozilla" in settings.download_user_agent

    def test_download_user_agent_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ACS_DOWNLOAD_USER_AGENT", "custom-agent/1.0")
        settings = Settings()
        assert settings.download_user_agent == "custom-agent/1.0"

    def test_hf_domains_default(self) -> None:
        settings = Settings()
        assert settings.hf_domains == ("huggingface.co", "hf.co")

    def test_hf_domains_env_override_comma_separated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """L2: a plain comma-separated string, not JSON — mirrors civitai_domains (D2)."""
        monkeypatch.setenv("ACS_HF_DOMAINS", "huggingface.co, HF.CO , ,hf-mirror.com")
        settings = Settings()
        assert settings.hf_domains == ("huggingface.co", "hf.co", "hf-mirror.com")

    @pytest.mark.parametrize(
        "bad_domains",
        [
            "co",
            "",
            "*.hf.co",
            "https://hf.co",
            "hf.co:443",
            "hf..co",
            "hugging face.co",
        ],
    )
    def test_invalid_hf_domains_are_rejected(self, bad_domains: str) -> None:
        with pytest.raises(ValidationError, match="invalid hf domain") as exc_info:
            Settings(hf_domains=bad_domains)  # type: ignore[arg-type]
        assert bad_domains.strip().lower() in str(exc_info.value)

    def test_hf_xet_enabled_default_true(self) -> None:
        settings = Settings()
        assert settings.hf_xet_enabled is True

    def test_hf_xet_enabled_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ACS_HF_XET_ENABLED", "false")
        settings = Settings()
        assert settings.hf_xet_enabled is False

    def test_hf_xet_concurrent_range_gets_default(self) -> None:
        settings = Settings()
        assert settings.hf_xet_concurrent_range_gets == 32

    def test_hf_xet_concurrent_range_gets_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            Settings(hf_xet_concurrent_range_gets=0)

    def test_hf_cache_path_default_none(self) -> None:
        settings = Settings()
        assert settings.hf_cache_path is None

    def test_hf_home_defaults_under_cache_path(self) -> None:
        settings = Settings(cache_path=Path("/workspace/.aisha-cache"))
        assert settings.hf_home == Path("/workspace/.aisha-cache/hf")

    def test_hf_home_uses_hf_cache_path_when_set(self) -> None:
        settings = Settings(
            cache_path=Path("/workspace/.aisha-cache"),
            hf_cache_path=Path("/mnt/big-disk/hf"),
        )
        assert settings.hf_home == Path("/mnt/big-disk/hf")


class TestModelFileConfigSha256Normalization:
    """Tests for D5 (config.py normalize_sha256) — fixes B2/B3."""

    def test_uppercase_sha256_normalized_to_lowercase(self) -> None:
        digest = "ABCDEF0123456789" * 4  # 64 uppercase hex chars
        cfg = ModelFileConfig(
            name="f", url="https://example.com/f", filename="f.safetensors", sha256=digest
        )
        assert cfg.sha256 is not None
        assert cfg.sha256 == digest.lower()
        assert len(cfg.sha256) == 64

    def test_sha256_with_surrounding_whitespace_is_stripped(self) -> None:
        digest = "a" * 64
        cfg = ModelFileConfig(
            name="f", url="https://example.com/f", filename="f.safetensors", sha256=f"  {digest}  "
        )
        assert cfg.sha256 == digest

    def test_sha256_none_stays_none(self) -> None:
        cfg = ModelFileConfig(name="f", url="https://example.com/f", filename="f.safetensors")
        assert cfg.sha256 is None

    @pytest.mark.parametrize(
        "bad_hash",
        [
            "not-a-hash",
            "a" * 63,
            "a" * 65,
            "g" * 64,  # non-hex character
        ],
    )
    def test_invalid_sha256_rejected(self, bad_hash: str) -> None:
        with pytest.raises(ValidationError, match="64 hexadecimal"):
            ModelFileConfig(
                name="f", url="https://example.com/f", filename="f.safetensors", sha256=bad_hash
            )


class TestModelFileConfigSizeBytes:
    """Declared model file sizes must be positive when present."""

    @pytest.mark.parametrize("size_bytes", [0, -1])
    def test_non_positive_size_bytes_rejected(self, size_bytes: int) -> None:
        with pytest.raises(ValidationError, match="greater than 0"):
            ModelFileConfig(
                name="f",
                url="https://example.com/f",
                filename="f.safetensors",
                size_bytes=size_bytes,
            )

    @pytest.mark.parametrize("size_bytes", [None, 1])
    def test_none_or_positive_size_bytes_accepted(self, size_bytes: int | None) -> None:
        config = ModelFileConfig(
            name="f",
            url="https://example.com/f",
            filename="f.safetensors",
            size_bytes=size_bytes,
        )
        assert config.size_bytes == size_bytes


class TestModelFileConfigUrlValidation:
    """Tests for MY-6a (config.py validate_url) — fixes E2 at the boundary."""

    def test_malformed_ipv6_url_rejected_naming_url_field(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            ModelFileConfig(name="f", url="http://[::1", filename="f.safetensors")
        assert "url" in str(exc_info.value)

    @pytest.mark.parametrize(
        "bad_url",
        ["model.safetensors", "ftp://host/f", "https://"],
    )
    def test_invalid_urls_rejected(self, bad_url: str) -> None:
        with pytest.raises(ValidationError, match="url"):
            ModelFileConfig(name="f", url=bad_url, filename="f.safetensors")

    def test_empty_url_is_accepted_as_a_snapshot_placeholder(self) -> None:
        cfg = ModelFileConfig(name="f", url="", filename="f.safetensors")
        assert cfg.url == ""

    @pytest.mark.parametrize(
        "good_url",
        [
            "https://civitai.red/api/download/models/1?type=Model",
            "http://localhost:8080/f.safetensors",
            "https://user:pass@host.example:8443/f.safetensors",
        ],
    )
    def test_valid_urls_accepted(self, good_url: str) -> None:
        cfg = ModelFileConfig(name="f", url=good_url, filename="f.safetensors")
        assert cfg.url == good_url

    def test_validator_does_not_normalize_or_strip_query(self) -> None:
        """apply_auth owns URL mutation; a validator that normalises could
        silently invalidate a presigned signature (pitfall #7)."""
        url = "https://civitai.red/api/download/models/1?Type=Model&X-Amz-Signature=AbC123"
        cfg = ModelFileConfig(name="f", url=url, filename="f.safetensors")
        assert cfg.url == url


class TestModelFileConfigPathTraversal:
    """Tests for path-traversal validators on the live ModelFileConfig class."""

    @pytest.mark.parametrize("bad_filename", ["../x", "a/b", "a\\b", "..", ".", ""])
    def test_model_file_config_filename_path_traversal_rejected(self, bad_filename: str) -> None:
        with pytest.raises(ValidationError):
            ModelFileConfig(
                name="Test",
                url="https://example.com/model.gguf",
                filename=bad_filename,
            )

    def test_model_file_config_valid_filename_accepted(self) -> None:
        cfg = ModelFileConfig(
            name="Test",
            url="https://example.com/model.gguf",
            filename="model.gguf",
        )
        assert cfg.filename == "model.gguf"


class TestModelConfigPathTraversal:
    """Tests for path-traversal validators on the live ModelConfig class."""

    @pytest.mark.parametrize("bad_subdirectory", ["../loras", "/abs", "a/../b"])
    def test_model_config_subdirectory_traversal_rejected(self, bad_subdirectory: str) -> None:
        with pytest.raises(ValidationError):
            ModelConfig(
                name="m",
                model_type="loras",
                files=[
                    ModelFileConfig(name="f", url="https://example.com/f.gguf", filename="f.gguf")
                ],
                subdirectory=bad_subdirectory,
            )

    @pytest.mark.parametrize("bad_model_type", ["../etc", "/abs", "a/../b", ""])
    def test_model_config_model_type_traversal_rejected(self, bad_model_type: str) -> None:
        with pytest.raises(ValidationError):
            ModelConfig(
                name="m",
                model_type=bad_model_type,
                files=[
                    ModelFileConfig(name="f", url="https://example.com/f.gguf", filename="f.gguf")
                ],
            )

    def test_model_config_valid_values_accepted(self) -> None:
        config = ModelConfig(
            name="m",
            model_type="loras",
            files=[ModelFileConfig(name="f", url="https://example.com/f.gguf", filename="f.gguf")],
            subdirectory="sdxl",
        )
        assert config.model_type == "loras"
        assert config.subdirectory == "sdxl"

    def test_model_config_no_subdirectory_accepted(self) -> None:
        config = ModelConfig(
            name="m",
            model_type="vae",
            files=[ModelFileConfig(name="f", url="https://example.com/f.gguf", filename="f.gguf")],
        )
        assert config.subdirectory is None

    def test_target_subpath_without_subdirectory(self) -> None:
        config = ModelConfig(
            name="m",
            model_type="diffusion_models",
            files=[ModelFileConfig(name="f", url="https://example.com/f.gguf", filename="f.gguf")],
        )
        assert config.target_subpath == "diffusion_models"

    def test_target_subpath_with_subdirectory(self) -> None:
        config = ModelConfig(
            name="m",
            model_type="loras",
            files=[ModelFileConfig(name="f", url="https://example.com/f.gguf", filename="f.gguf")],
            subdirectory="anime",
        )
        assert config.target_subpath == "loras/anime"


class TestModelConfigSubfolderAlias:
    """Tests for C1 -- `subfolder:` (every real bundle) and `subdirectory:` (the
    canonical name) must parse identically, and `target_subpath` must reflect it."""

    def test_subfolder_key_populates_subdirectory(self) -> None:
        """The C1 regression test: every ai-bundles bundle uses `subfolder:`."""
        config = ModelConfig.model_validate(
            {
                "name": "m",
                "model_type": "checkpoints",
                "files": [{"name": "f", "url": "https://example.com/f.gguf", "filename": "f.gguf"}],
                "subfolder": "Wan/22",
            }
        )
        assert config.subdirectory == "Wan/22"
        assert config.target_subpath == "checkpoints/Wan/22"

    def test_subdirectory_key_still_works(self) -> None:
        config = ModelConfig.model_validate(
            {
                "name": "m",
                "model_type": "checkpoints",
                "files": [{"name": "f", "url": "https://example.com/f.gguf", "filename": "f.gguf"}],
                "subdirectory": "Wan/22",
            }
        )
        assert config.subdirectory == "Wan/22"
        assert config.target_subpath == "checkpoints/Wan/22"

    def test_both_keys_present_with_different_values_is_ambiguous(self) -> None:
        with pytest.raises(ValidationError):
            ModelConfig.model_validate(
                {
                    "name": "m",
                    "model_type": "checkpoints",
                    "files": [
                        {"name": "f", "url": "https://example.com/f.gguf", "filename": "f.gguf"}
                    ],
                    "subdirectory": "a",
                    "subfolder": "b",
                }
            )

    def test_subfolder_traversal_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ModelConfig.model_validate(
                {
                    "name": "m",
                    "model_type": "loras",
                    "files": [
                        {"name": "f", "url": "https://example.com/f.gguf", "filename": "f.gguf"}
                    ],
                    "subfolder": "../etc",
                }
            )

    def test_populate_by_name_still_accepts_field_name_kwarg(self) -> None:
        """C1 pitfall #2: constructing with the field name in code/tests must keep working."""
        config = ModelConfig(
            name="m",
            model_type="loras",
            files=[ModelFileConfig(name="f", url="https://example.com/f.gguf", filename="f.gguf")],
            subdirectory="sdxl",
        )
        assert config.subdirectory == "sdxl"


class TestBundleSchemaStrictness:
    """Tests for C1b -- unknown keys in a bundle file are an error, not a silent drop."""

    def test_model_config_unknown_key_rejected(self) -> None:
        with pytest.raises(ValidationError, match="size_gb"):
            ModelConfig.model_validate(
                {
                    "name": "m",
                    "model_type": "checkpoints",
                    "size_gb": 14.5,
                    "files": [
                        {"name": "f", "url": "https://example.com/f.gguf", "filename": "f.gguf"}
                    ],
                }
            )

    def test_model_file_config_unknown_key_rejected(self) -> None:
        with pytest.raises(ValidationError, match="size_gb"):
            ModelFileConfig.model_validate(
                {
                    "name": "f",
                    "url": "https://example.com/f.gguf",
                    "filename": "f.gguf",
                    "size_gb": 14.5,
                }
            )

    def test_bundle_config_unknown_key_rejected(self) -> None:
        with pytest.raises(ValidationError, match="not_a_real_key"):
            BundleConfig.model_validate(
                {
                    "metadata": {"name": "test", "version": "260101-01"},
                    "not_a_real_key": True,
                }
            )

    def test_bundle_metadata_unknown_key_rejected(self) -> None:
        with pytest.raises(ValidationError, match="not_a_real_key"):
            BundleMetadata.model_validate(
                {"name": "test", "version": "260101-01", "not_a_real_key": True}
            )

    def test_custom_node_unknown_key_rejected(self) -> None:
        with pytest.raises(ValidationError, match="not_a_real_key"):
            CustomNode.model_validate(
                {
                    "name": "TestNode",
                    "git_url": "https://github.com/test/node",
                    "commit_sha": "abc123",
                    "not_a_real_key": True,
                }
            )

    def test_description_and_custom_node_required_are_accepted(self) -> None:
        config = ModelConfig.model_validate(
            {
                "name": "m",
                "model_type": "diffusion_models",
                "description": "A test model",
                "custom_node_required": "ComfyUI-GGUF",
                "files": [{"name": "f", "url": "https://example.com/f.gguf", "filename": "f.gguf"}],
            }
        )
        assert config.description == "A test model"
        assert config.custom_node_required == "ComfyUI-GGUF"

    def test_description_and_custom_node_required_default_to_none(self) -> None:
        config = ModelConfig(
            name="m",
            model_type="diffusion_models",
            files=[ModelFileConfig(name="f", url="https://example.com/f.gguf", filename="f.gguf")],
        )
        assert config.description is None
        assert config.custom_node_required is None

    def test_bundle_metadata_accepts_author_notes_tags(self) -> None:
        """Discovered while validating real ai-bundles data: these three
        fields were silently dropped the same way `subfolder` was."""
        metadata = BundleMetadata.model_validate(
            {
                "name": "test",
                "version": "260101-01",
                "author": "https://civitai.com/models/1",
                "notes": "Initial bundle creation",
                "tags": ["i2v", "wan"],
            }
        )
        assert metadata.author == "https://civitai.com/models/1"
        assert metadata.notes == "Initial bundle creation"
        assert metadata.tags == ["i2v", "wan"]

    def test_bundle_config_accepts_hardware_and_generation_sections(self) -> None:
        """Apex-consumed sections are typed while remaining optional to Aisha."""
        config = BundleConfig.model_validate(
            {
                "metadata": {"name": "test", "version": "260101-01"},
                "hardware": {"gpu_whitelist": ["RTX 5090"], "num_gpus": 1},
                "generation": {"defaults": {"resolution": "1024x1024"}},
            }
        )
        assert config.hardware is not None
        assert config.hardware.gpu_whitelist == ["RTX 5090"]
        assert config.hardware.num_gpus == 1
        assert config.generation is not None
        assert config.generation.defaults is not None
        assert config.generation.defaults.resolution == "1024x1024"

    def test_hardware_base_image_is_optional_and_typed(self) -> None:
        """Part C: base_image is advisory -- present or absent, both validate."""
        with_image = BundleConfig.model_validate(
            {
                "metadata": {"name": "test", "version": "260101-01"},
                "hardware": {"base_image": "vastai/comfy:v0.30.0-cuda-13.2-py312"},
            }
        )
        assert with_image.hardware is not None
        assert with_image.hardware.base_image == "vastai/comfy:v0.30.0-cuda-13.2-py312"

        without_image = BundleConfig.model_validate(
            {
                "metadata": {"name": "test", "version": "260101-01"},
                "hardware": {"num_gpus": 1},
            }
        )
        assert without_image.hardware is not None
        assert without_image.hardware.base_image is None

    def test_hardware_typo_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="min_netwrok_download_mbps"):
            BundleConfig.model_validate(
                {
                    "metadata": {"name": "test", "version": "260101-01"},
                    "hardware": {"min_netwrok_download_mbps": 300},
                }
            )

    def test_hardware_and_generation_are_optional(self) -> None:
        config = BundleConfig.model_validate({"metadata": {"name": "test", "version": "260101-01"}})
        assert config.hardware is None
        assert config.generation is None

    def test_settings_still_ignores_unknown_acs_env_vars(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pitfall #1: Settings must keep `extra='ignore'` -- an unrelated ACS_*
        env var must never break startup."""
        monkeypatch.setenv("ACS_SOME_UNRELATED_FUTURE_VARIABLE", "x")
        Settings()  # must not raise


class TestDeploymentPlanMissingUrls:
    """Tests for C3b -- the plan surfaces missing source URLs before anything runs."""

    def _bundle(self, urls: list[str]) -> BundleConfig:
        return BundleConfig(
            metadata=BundleMetadata(name="test", version="260101-01"),
            models=[
                ModelConfig(
                    name="m",
                    model_type="checkpoints",
                    files=[
                        ModelFileConfig(name=f"f{i}", url=url, filename=f"f{i}.safetensors")
                        for i, url in enumerate(urls)
                    ],
                )
            ],
        )

    def test_no_missing_urls(self) -> None:
        plan = DeploymentPlan.from_bundle(
            self._bundle(["https://example.com/a"]), DeployMode.MODELS_ONLY
        )
        assert plan.missing_url_files_count == 0

    def test_counts_every_missing_url(self) -> None:
        plan = DeploymentPlan.from_bundle(
            self._bundle(["", "https://example.com/a", ""]), DeployMode.MODELS_ONLY
        )
        assert plan.missing_url_files_count == 2
