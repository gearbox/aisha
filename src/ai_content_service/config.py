"""Configuration models and settings for AI Content Service."""

from __future__ import annotations

import contextlib
import re
import sys
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Annotated, Final, Literal
from urllib.parse import urlparse

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


def unwrap_secret(secret: SecretStr | None) -> str | None:
    """Unwrap an optional SecretStr to its plain value, for use at composition roots.

    A blank or whitespace-only secret is normalised to ``None``: an unset
    variable and an empty one both mean "no credential". Attaching an empty
    bearer or ``?token=`` is worse than attaching nothing — providers may answer
    401 for an artefact they would have served anonymously (E3).
    """
    if secret is None:
        return None
    value = secret.get_secret_value()
    return value if value.strip() else None


_DEFAULT_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
_HOSTNAME_RE = re.compile(r"^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?$")


def _invalid_domain_error(field_name: str, entry: str, example: str) -> ValueError:
    return ValueError(
        f"invalid {field_name} domain {entry!r}: expected a hostname with at least "
        f"two labels (e.g. {example!r}). A single-label entry would make every host "
        f"under that suffix a valid destination for the {field_name} token."
    )


def _normalize_domain_entry(
    raw: object,
    *,
    field_name: str,
    example: str,
    allow_blank: bool,
) -> str | None:
    if not isinstance(raw, str):
        raise ValueError(f"invalid {field_name} domain {raw!r}: expected a hostname")
    entry = raw.strip().lower()
    if not entry:
        if allow_blank:
            return None
        raise _invalid_domain_error(field_name, entry, example)
    labels = entry.split(".")
    if len(labels) < 2 or not all(_HOSTNAME_RE.fullmatch(label) for label in labels):
        raise _invalid_domain_error(field_name, entry, example)
    return entry


def _normalize_domain_list(value: object, *, field_name: str, example: str) -> object:
    """Shared before-validator body for `civitai_domains` and `hf_domains`.

    Accepts a plain comma-separated string (never JSON-decoded -- these
    fields use `NoDecode`) or a list/tuple, and rejects any entry with fewer
    than two labels: a single-label entry would make every host under that
    suffix a valid destination for the field's API token.
    """
    if isinstance(value, str):
        parts: list[str] = value.split(",")
        if not value.strip():
            parts = [value]
    elif isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        return value

    normalized: list[str] = []
    allow_blank = isinstance(value, str) and len(parts) > 1
    for raw in parts:
        entry = _normalize_domain_entry(
            raw,
            field_name=field_name,
            example=example,
            allow_blank=allow_blank,
        )
        if entry is not None:
            normalized.append(entry)

    return tuple(normalized)


class DeployMode(str, Enum):
    """Deployment mode controlling which components are installed."""

    FULL = "full"
    """Full deployment: ComfyUI, custom nodes, requirements, models, workflow."""

    MODELS_ONLY = "models_only"
    """Models-only deployment: Only downloads models and installs workflow.

    Use this when you already have a working ComfyUI setup and just want to
    add a new workflow with its required models.
    """


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="ACS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_nested_delimiter="__",
    )

    # Paths
    comfyui_path: Path = Field(
        default=Path("/workspace/ComfyUI"),
        description="Path to ComfyUI installation",
    )
    bundles_path: Path = Field(
        default=Path("config/bundles"),
        description=(
            "Path to the ai-bundles repository root containing bundle-index.yaml. "
            "Pointing at its bundles/ subdirectory disables index-driven resolution."
        ),
    )
    comfyui_python: Path = Field(
        default_factory=lambda: Path(sys.executable),
        description=(
            "Python interpreter that owns the ComfyUI venv. All pip operations for "
            "ComfyUI base requirements, locked overlay, and custom-node deps target "
            "this interpreter's site-packages. Defaults to sys.executable, which is "
            "correct for development. Override via ACS_COMFYUI_PYTHON in production; "
            "on Vast.ai's vastai/comfy image this is /venv/main/bin/python."
        ),
    )

    @field_validator("comfyui_python")
    @classmethod
    def _check_python_exists(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(
                f"comfyui_python does not exist: {v}. "
                f"Set ACS_COMFYUI_PYTHON to a valid interpreter path."
            )
        if not v.is_file():
            raise ValueError(f"comfyui_python is not a file: {v}")
        return v

    # Bundle selection
    bundle: str | None = Field(
        default=None,
        description="Bundle name to deploy",
    )
    bundle_version: str | None = Field(
        default=None,
        description="Specific bundle version (default: current)",
    )

    # Authentication tokens
    hf_token: SecretStr | None = Field(
        default=None,
        description="Hugging Face API token for private/gated models",
    )
    civitai_api_token: SecretStr | None = Field(
        default=None,
        description="Civitai API token for model downloads",
    )
    civitai_domains: Annotated[tuple[str, ...], NoDecode] = Field(
        default=("civitai.com", "civitai.red", "civitai.green"),
        description="Civitai front-door domains eligible for API-token auth",
    )
    civitai_allow_query_token_fallback: bool = Field(
        default=False,
        description=(
            "On 401/403 from a Civitai host, retry once with the token as a "
            "?token= query param. Off by default: Civitai documents the "
            "Authorization header as fully supported for downloads and warns "
            "that query params are recorded in edge and proxy access logs. "
            "Enable only if header auth is rejected."
        ),
    )

    @field_validator("civitai_domains", mode="before")
    @classmethod
    def _split_civitai_domains(cls, v: object) -> object:
        return _normalize_domain_list(v, field_name="civitai", example="civitai.red")

    hf_domains: Annotated[tuple[str, ...], NoDecode] = Field(
        default=("huggingface.co", "hf.co"),
        description=(
            "HuggingFace front-door domains routed to the hf_xet transport and "
            "eligible for HuggingFace token auth (also the allowlist "
            "build_huggingface_policy uses for where the token may be sent)"
        ),
    )
    hf_xet_enabled: bool = Field(
        default=True,
        description=(
            "Route HuggingFace URLs through hf_xet. Disable to force the httpx "
            "path -- measured 25x slower on Xet-backed repos, so only for "
            "debugging."
        ),
    )
    hf_xet_concurrent_range_gets: int = Field(
        default=32,
        gt=0,
        description=(
            "HF_XET_NUM_CONCURRENT_RANGE_GETS. 32 measured 414 MB/s vs 322 at "
            "the library default on a 4090 node."
        ),
    )
    hf_cache_path: Path | None = Field(
        default=None,
        description=(
            "HF_HOME for the Xet chunk cache. Defaults to cache_path/'hf'. Must "
            "be on a filesystem with room for the chunk cache alongside the "
            "weights themselves."
        ),
    )

    @field_validator("hf_domains", mode="before")
    @classmethod
    def _split_hf_domains(cls, v: object) -> object:
        return _normalize_domain_list(v, field_name="hf", example="hf.co")

    # Download settings
    max_concurrent_downloads: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum number of concurrent model downloads",
    )
    download_user_agent: str = Field(
        default=_DEFAULT_BROWSER_UA,
        description=("User-Agent for model downloads; Cloudflare challenges default library UAs"),
    )
    download_max_attempts: int = Field(
        default=6,
        ge=1,
        le=20,
        description="Maximum attempts for a model transfer, including the initial request",
    )
    download_max_retry_after_seconds: float = Field(
        default=120.0,
        gt=0,
        description="Maximum delay honoured from an HTTP Retry-After response header",
    )

    # Download settings (verification / skip)
    verify_checksums: bool = Field(
        default=True,
        description="Verify SHA256 checksums after download",
    )
    skip_existing: bool = Field(
        default=True,
        description="Skip download if file already exists and checksum matches",
    )

    # Deployment options
    no_verify: bool = Field(
        default=False,
        description="Skip ComfyUI verification after deployment",
    )
    deploy_mode: DeployMode = Field(
        default=DeployMode.FULL,
        description="Deployment mode (full or models_only)",
    )

    # Logging
    log_format: Literal["auto", "json", "console"] = Field(
        default="auto", description="Log output format; auto = json when stderr is not a TTY"
    )
    log_level: str = Field(default="INFO", description="Root log level")

    # ComfyUI runtime (used by supervisord config generation in onstart.sh)
    comfyui_port: int = Field(
        default=8188,
        ge=1,
        le=65535,
        description="ComfyUI listen port; must match apex's bundle.hardware.comfyui_port",
    )
    comfyui_host: str = Field(
        # cloudflared reaches ComfyUI over the container bridge
        default="0.0.0.0",  # noqa: S104
        description="ComfyUI listen interface; must remain 0.0.0.0 for cloudflared to reach it",
    )
    comfyui_extra_args: str = Field(
        default="",
        description="Extra args appended to python main.py when supervisord launches ComfyUI",
    )
    comfyui_url: str | None = Field(
        default=None,
        description=(
            "Optional running ComfyUI base URL used by `acs bundle validate` for live "
            "workflow provider checks (for example, http://localhost:18188)."
        ),
    )

    # Cloudflare tunnel
    cf_tunnel_token: SecretStr | None = Field(
        default=None,
        description="Cloudflare tunnel token; supervisord launches cloudflared when this is set",
    )

    # Apex provisioning callbacks — consumed by ProvisioningReporter
    apex_session_id: str = Field(
        default="",
        description="Session ID from apex; forwarded in every provisioning callback payload",
    )
    apex_callback_url: str = Field(
        default="",
        description="Base URL for provisioning callbacks; consumed by ProvisioningReporter",
    )
    apex_callback_token: SecretStr | None = Field(
        default=None,
        description="Bearer token for provisioning callback auth; consumed by ProvisioningReporter",
    )

    # Provisioning phase timing telemetry (Phase 2b-lite) — always on,
    # independent of the Apex callback fields above, which are unset on every
    # manual node, benchmark, and local run: exactly when this record matters most.
    provisioning_timing_path: Path | None = Field(
        default=None,
        description=(
            "JSONL file for per-deployment phase timings. Defaults to "
            "cache_path/'provisioning-timings.jsonl'. Local concurrent writers "
            "append atomically; shared/network filesystem guarantees depend on "
            "that filesystem and are not provided by Aisha."
        ),
    )
    provisioning_timing_enabled: bool = Field(
        default=True,
        description="Write a provisioning-timings.jsonl record after every deployment.",
    )

    # R2 Model Cache — read path (B1: baked read-only token from Vast.ai template env)
    r2_model_cache_bucket: str = Field(
        default="apex-model-cache",
        description="R2 bucket name for model cache",
    )
    r2_s3_endpoint: str | None = Field(
        default=None,
        description="R2 S3-compatible endpoint URL (e.g. https://<account>.r2.cloudflarestorage.com)",
    )
    r2_readonly_access_key_id: str | None = Field(
        default=None,
        description="Read-only R2 access key ID baked into the Vast.ai template",
    )
    r2_readonly_secret_access_key: SecretStr | None = Field(
        default=None,
        description="Read-only R2 secret access key baked into the Vast.ai template",
    )

    # R2 Model Cache -- direct write path (operator-supplied by `cache push --direct`).
    # Distinct from r2_readonly_* on purpose: the write token is never present on
    # a GPU node, and the read token must never be able to write.
    r2_write_access_key_id: str | None = Field(
        default=None,
        description="R2 access key ID with write access to the model cache bucket",
    )
    r2_write_secret_access_key: SecretStr | None = Field(
        default=None,
        description="R2 secret access key with write access to the model cache bucket",
    )

    # R2 Model Cache — Apex credential broker (default cache push mode)
    apex_base_url: str | None = Field(
        default=None,
        description="Apex base URL for admin model-cache API (e.g. https://api.example.com)",
    )
    apex_admin_token: SecretStr | None = Field(
        default=None,
        description="Apex admin bearer token; required for acs cache push",
    )

    # rclone settings
    rclone_path: str = Field(
        default="rclone",
        description="Path to rclone binary; defaults to rclone on PATH",
    )

    # Multi-GB uploads fail with `501 NotImplemented` error;
    # That matches the known R2 multipart flakiness, and the documented stabilizer is lowering part concurrency;
    # https://forum.rclone.org/t/rclone-fails-to-switch-to-multi-part-uploads-when-a-file-is-too-large/36259/7
    # https://developers.cloudflare.com/r2/examples/rclone/#a-note-about-multipart-upload-part-sizes
    rclone_upload_concurrency: int = Field(
        default=4,
        ge=1,
        description="S3 multipart upload concurrency for rclone push",
    )
    rclone_chunk_size_mb: int = Field(
        default=128,
        ge=5,
        description="S3 multipart chunk size in MiB for rclone push",
    )
    rclone_multi_thread_streams: int = Field(
        default=4,
        ge=1,
        description="Number of parallel streams for rclone multi-thread download",
    )
    rclone_max_transfer_seconds: int = Field(
        default=3600,
        ge=60,
        description="Wall-clock cap (seconds) for a single rclone pull/push subprocess",
    )

    # Supervisor
    supervisor_log_dir: Path = Field(
        default=Path("/var/log/aisha"),
        description="Directory where supervisord and child process logs are written",
    )

    # ----------------------------------------------------------------
    # Bundle registry
    # ----------------------------------------------------------------
    cache_path: Path = Field(
        default=Path("/workspace/.aisha-cache"),
        description="Cache directory for cloned registries",
    )
    bundles_repo: str | None = Field(
        default=None,
        description="Git URL for bundles repository (e.g., https://github.com/gearbox/ai-bundles)",
    )
    bundles_branch: str = Field(
        default="main",
        description="Branch to use for bundles repository",
    )
    github_token: SecretStr | None = Field(
        default=None,
        description="GitHub Personal Access Token for private repos",
    )
    github_ssh_key: Path | None = Field(
        default=None,
        description="Path to GitHub SSH private key",
    )
    auto_sync_registries: bool = Field(
        default=True,
        description="Automatically sync registries on deploy",
    )

    @property
    def hf_home(self) -> Path:
        """HF_HOME for the Xet chunk cache: `hf_cache_path` if set, else under `cache_path`."""
        return self.hf_cache_path or self.cache_path / "hf"

    @property
    def models_path(self) -> Path:
        return self.comfyui_path / "models"

    @property
    def custom_nodes_path(self) -> Path:
        return self.comfyui_path / "custom_nodes"

    def has_remote_bundles(self) -> bool:
        """Return True when a remote bundles repository is configured."""
        return self.bundles_repo is not None

    def get_bundles_cache_path(self) -> Path:
        """Return the local path where the bundles repo will be cloned."""
        return self.cache_path / "ai-bundles"


# Bundle configuration models

_VERSION_RE = re.compile(r"^\d{6}-\d{2}$")


class BundleVersion(BaseModel):
    """Bundle version in YYMMDD-NN format."""

    version: str

    @field_validator("version")
    @classmethod
    def validate_format(cls, v: str) -> str:
        if not _VERSION_RE.match(v):
            msg = f"Version must match YYMMDD-NN format, got: {v!r}"
            raise ValueError(msg)
        return v

    def __str__(self) -> str:
        return self.version

    @classmethod
    def create_new(cls, existing: list[str]) -> BundleVersion:
        today = datetime.now(tz=timezone.utc).strftime("%y%m%d")
        max_seq = 0
        for v in existing:
            if v.startswith(f"{today}-"):
                with contextlib.suppress(IndexError, ValueError):
                    max_seq = max(max_seq, int(v.split("-")[1]))
        return cls(version=f"{today}-{max_seq + 1:02d}")


class ModelType(str, Enum):
    """ComfyUI model subdirectory types."""

    DIFFUSION = "diffusion_models"
    LORA = "loras"
    CLIP = "clip"
    TEXT_ENCODERS = "text_encoders"
    CLIP_VISION = "clip_vision"
    VAE = "vae"
    CONTROLNET = "controlnet"
    UPSCALE = "upscale_models"
    EMBEDDINGS = "embeddings"
    CHECKPOINTS = "checkpoints"
    STYLE_MODELS = "style_models"
    UNET = "unet"
    GLIGEN = "gligen"
    PHOTOMAKER = "photomaker"
    MODEL_PATCHES = "model_patches"
    AUDIO_ENCODERS = "audio_encoders"
    BACKGROUND_REMOVAL = "background_removal"
    DETECTION = "detection"
    DIFFUSERS = "diffusers"


_CUSTOM_NODE_GIT_FIELDS: Final[tuple[str, ...]] = ("git_url", "commit_sha")
_CUSTOM_NODE_REGISTRY_FIELDS: Final[tuple[str, ...]] = ("node_id", "version", "archive_sha256")
_COMMIT_SHA_RE: Final = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_GIT_URL_PREFIXES: Final[tuple[str, ...]] = ("https://",)


def _blank(value: object) -> bool:
    """Return whether an optional source field has no usable value."""
    return value is None or (isinstance(value, str) and not value.strip())


def validate_custom_node_name(value: str) -> str:
    """Keep a custom-node installation inside ComfyUI's custom_nodes directory."""
    if (
        value != value.strip()
        or not value
        or "/" in value
        or "\\" in value
        or "\x00" in value
        or value in {".", ".."}
        or value.startswith("-")
    ):
        raise ValueError(
            "custom node name must be a plain relative directory name "
            "(no leading or trailing whitespace, path separators, or NUL; not empty, '.' "
            "or '..'; and not starting with '-')"
        )

    base = Path("x").resolve()
    try:
        (base / value).resolve().relative_to(base)
    except (OSError, ValueError) as exc:
        raise ValueError("custom node name must resolve inside its custom_nodes directory") from exc
    return value


def is_exact_commit_sha(value: str) -> bool:
    """Return whether a Git revision is a stable, full object name."""
    return _COMMIT_SHA_RE.fullmatch(value) is not None


def is_supported_git_url(value: str) -> bool:
    """Return whether deployment can anonymously clone a custom node URL.

    GPU-node provisioning supplies neither credentials nor SSH key material,
    so the custom-node clone path supports HTTPS only.
    """
    return value.startswith(_GIT_URL_PREFIXES)


class CustomNode(BaseModel):
    """Custom node configuration.

    A node is pinned by exactly one of two equally authoritative sources:

    - ``source: git`` (the default, so every bundle predating this field still
      parses unchanged): ``git_url`` + ``commit_sha`` pin an exact commit.
    - ``source: registry``: ``node_id`` + ``version`` pin an immutable Comfy
      Registry release. This exists because many custom nodes (for example
      ComfyUI-KJNodes) publish registry versions from ``pyproject.toml``
      without ever tagging a git release, so no commit can represent what was
      installed and tested.

    The Comfy Registry's version endpoint (``GET /nodes/{node_id}/versions/{version}``,
    checked against ``comfyui-kjnodes`` 1.5.0 on 2026-08-20) exposes no
    integrity digest -- no sha256, no content hash, nothing beyond
    ``downloadUrl``. ``archive_sha256`` is therefore a reserved field today:
    registry installs are trusted transport-only unless an author supplies a
    digest manually. This is a real weakening relative to a git commit's
    exact-content guarantee, and the field remains available for a future
    Registry digest or snapshot-time archive hash.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    source: Literal["git", "registry"] = "git"

    # source == "git"
    git_url: str | None = None
    commit_sha: str | None = None

    # source == "registry"
    node_id: str | None = None
    version: str | None = None
    archive_sha256: str | None = None

    pip_requirements: list[str] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        """Require one safe custom_nodes directory component at the model boundary."""
        return validate_custom_node_name(value)

    @field_validator("node_id", "version", "git_url", "commit_sha", "archive_sha256", mode="before")
    @classmethod
    def strip_source_fields(cls, value: object) -> object:
        """Normalise source identifiers before source-specific validation."""
        return value.strip() if isinstance(value, str) else value

    @field_validator("git_url")
    @classmethod
    def validate_git_url(cls, value: str | None) -> str | None:
        """Require a URL the credential-free deployment clone can use."""
        if value and not is_supported_git_url(value):
            raise ValueError("git_url must start with https://")
        return value

    @field_validator("commit_sha")
    @classmethod
    def validate_commit_sha(cls, value: str | None) -> str | None:
        """Require an immutable full Git object name, never a ref or abbreviation."""
        if value and not is_exact_commit_sha(value):
            raise ValueError(
                "commit_sha must be an exact 40- or 64-character lowercase hexadecimal object name"
            )
        return value

    @model_validator(mode="after")
    def validate_source_fields(self) -> CustomNode:
        required, forbidden_fields = (
            (_CUSTOM_NODE_GIT_FIELDS, _CUSTOM_NODE_REGISTRY_FIELDS)
            if self.source == "git"
            else (("node_id", "version"), _CUSTOM_NODE_GIT_FIELDS)
        )
        missing = [field for field in required if _blank(getattr(self, field))]
        present = [field for field in forbidden_fields if getattr(self, field) is not None]
        errors: list[str] = []
        if missing:
            errors.append(f"source={self.source!r} requires {', '.join(missing)}")
        if present:
            errors.append(f"source={self.source!r} forbids {', '.join(present)}")
        if errors:
            raise ValueError(f"custom node {self.name!r}: {'; '.join(errors)}")
        return self


class BundleMetadata(BaseModel):
    """Bundle metadata."""

    model_config = ConfigDict(extra="forbid")

    name: str
    version: str
    description: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    tested: bool = False
    author: str | None = Field(
        default=None, description="Advisory: source URL or attribution for this bundle."
    )
    notes: str | None = Field(default=None, description="Advisory: free-form authoring notes.")
    tags: list[str] | None = Field(
        default=None, description="Advisory: free-form labels for discovery/filtering."
    )


class ComfyUIConfig(BaseModel):
    """Optional bundle-level ComfyUI override.

    A template normally owns ComfyUI, CUDA, Python, and the base package set.
    Use this only when a bundle genuinely needs a ComfyUI revision unavailable
    from a published template; duplicating the template pin adds deployment
    time and creates a second source of environment truth.
    """

    model_config = ConfigDict(extra="forbid")

    repo: str = Field(
        default="https://github.com/comfyanonymous/ComfyUI",
        description="Repository for the exceptional bundle-level ComfyUI override.",
    )
    commit: str = Field(
        description=(
            "Exceptional ComfyUI commit override. Prefer pinning "
            "hardware.template_hash_id so the tested template owns ComfyUI, CUDA, Python, "
            "and base packages."
        )
    )


CustomNodeConfig = CustomNode


class ModelFileConfig(BaseModel):
    """Individual model file configuration."""

    model_config = ConfigDict(extra="forbid")

    name: str
    url: str
    filename: str
    sha256: str | None = None
    size_bytes: int | None = Field(
        default=None,
        gt=0,
        description="Declared file size. Advisory: relaxes verification floors, never tightens them.",
    )

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Fail loud at the bundle boundary, once, so a malformed URL (E2)
        never reaches runtime code three modules away. Does not normalise or
        strip the query — `apply_auth` owns URL mutation, and this must not
        silently invalidate a presigned signature. An empty string is the
        explicit snapshot placeholder for a source URL that must be supplied
        before deployment; `acs models check` reports it as an actionable row.
        """
        if not v:
            return v
        try:
            parsed = urlparse(v)
        except ValueError as e:
            msg = f"url is not parseable: {e}"
            raise ValueError(msg) from e
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            msg = "url must be an absolute http(s) URL with a host"
            raise ValueError(msg)
        return v

    @field_validator("filename")
    @classmethod
    def no_path_separators(cls, v: str) -> str:
        if "/" in v or "\\" in v or v in {"", ".", ".."}:
            msg = "filename must be a plain file name (no path separators, not empty, not '.' or '..')"
            raise ValueError(msg)
        return v

    @field_validator("sha256")
    @classmethod
    def normalize_sha256(cls, v: str | None) -> str | None:
        if v is None:
            return None
        normalized = v.strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", normalized):
            msg = "sha256 must be exactly 64 hexadecimal characters"
            raise ValueError(msg)
        return normalized


class ModelConfig(BaseModel):
    """Model group configuration."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    name: str
    description: str | None = None
    model_type: str = Field(
        description="ComfyUI model subdirectory (e.g., 'diffusion_models', 'clip', 'vae')"
    )
    custom_node_required: str | None = Field(
        default=None,
        description="Advisory: custom node this model's loader requires. Not yet enforced (F9).",
    )
    files: list[ModelFileConfig]
    subdirectory: str | None = Field(
        default=None,
        validation_alias=AliasChoices("subdirectory", "subfolder"),
        description="Optional subdirectory within model_type folder",
    )

    @field_validator("model_type")
    @classmethod
    def model_type_no_traversal(cls, v: str) -> str:
        if (
            not v
            or v.startswith("/")
            or Path(v).is_absolute()
            or "\\" in v
            or ".." in Path(v).parts
        ):
            msg = "model_type must not be an absolute path or contain '..' segments"
            raise ValueError(msg)
        return v

    @field_validator("subdirectory")
    @classmethod
    def subdirectory_no_traversal(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if v.startswith("/") or Path(v).is_absolute() or "\\" in v or ".." in Path(v).parts:
            msg = "subdirectory must not be an absolute path or contain '..' segments"
            raise ValueError(msg)
        return v

    @property
    def target_subpath(self) -> str:
        """Path of this model's directory relative to the models root."""
        return f"{self.model_type}/{self.subdirectory}" if self.subdirectory else self.model_type


class HardwareConfig(BaseModel):
    """Hardware requirements consumed by Apex's ``BundleIndexService``.

    Aisha does not interpret these fields, but it validates them at the bundle
    boundary so a typo cannot silently remove an Apex provisioning filter.
    """

    model_config = ConfigDict(extra="forbid")

    gpu_whitelist: list[str] | None = None
    min_disk_gb: int | None = None
    min_network_upload_mbps: int | None = None
    min_network_download_mbps: int | None = None
    cuda_min_version: str | None = None
    num_gpus: int | None = None
    comfyui_port: int | None = None
    template_hash_id: str | None = Field(
        default=None,
        description=(
            "Recommended complete-environment pin: the Vast.ai template tested with this "
            "bundle. The template owns ComfyUI, CUDA, Python, and the base package set."
        ),
    )
    base_image: str | None = Field(
        default=None,
        description=(
            "Vast.ai image this bundle was tested against (e.g. "
            "'vastai/comfy:v0.30.0-cuda-13.2-py312'). Advisory: aisha does not "
            "act on it. Telemetry records it as bundle_base_image, never as "
            "observed runtime-image provenance; Apex can eventually use it to "
            "pick a template."
        ),
    )


class GenerationDefaultsConfig(BaseModel):
    """Optional generation defaults consumed by Apex."""

    model_config = ConfigDict(extra="forbid")

    resolution: str | None = None
    steps: int | None = None
    cfg: float | None = None
    sampler: str | None = None
    scheduler: str | None = None
    denoise: float | None = None


class GenerationConstraintsConfig(BaseModel):
    """Optional generation constraints consumed by Apex."""

    model_config = ConfigDict(extra="forbid")

    max_megapixels: float | None = None
    latent_multiple: int | None = None
    max_edge: int | None = None
    min_steps: int | None = None
    max_steps: int | None = None
    min_cfg: float | None = None
    max_cfg: float | None = None
    allowed_samplers: list[str] | None = None
    allowed_schedulers: list[str] | None = None


class GenerationConfig(BaseModel):
    """Optional generation configuration consumed by Apex."""

    model_config = ConfigDict(extra="forbid")

    defaults: GenerationDefaultsConfig | None = None
    constraints: GenerationConstraintsConfig | None = None


class ReadinessMarkerConfig(BaseModel):
    """Bundle-declared readiness evidence consumed by Apex.

    Aisha does not interpret this, but validates it at the bundle boundary so a
    typo cannot silently disable Apex's provisioning gate -- the same posture
    used for HardwareConfig and GenerationConfig.
    """

    model_config = ConfigDict(extra="forbid")

    node_class: str = Field(
        min_length=1,
        description=(
            "ComfyUI class name that must appear in /object_info before Apex "
            "promotes the session to active."
        ),
    )

    @field_validator("node_class")
    @classmethod
    def normalize_node_class(cls, value: str) -> str:
        """Match Apex's non-blank readiness-node requirement at our boundary."""
        if normalized := value.strip():
            return normalized
        raise ValueError("node_class must not be empty or whitespace-only")


class WorkflowRole(str, Enum):
    """The closed set of workflow roles that Apex can address.

    ``StrEnum`` would express this a little more directly, but Aisha supports
    Python 3.10 where it is not available.  A ``str, Enum`` has the same YAML
    and JSON behaviour on every supported Python version.
    """

    LATENT = "latent"
    POSITIVE_PROMPT = "positive_prompt"
    NEGATIVE_PROMPT = "negative_prompt"
    SAMPLER = "sampler"
    SAVE = "save"
    PREVIEW = "preview"


_WORKFLOW_ROLE_PARAMETERS: dict[WorkflowRole, frozenset[str]] = {
    WorkflowRole.LATENT: frozenset({"width", "height", "batch_size"}),
    WorkflowRole.POSITIVE_PROMPT: frozenset({"text"}),
    WorkflowRole.NEGATIVE_PROMPT: frozenset({"text"}),
    WorkflowRole.SAMPLER: frozenset({"seed", "steps", "cfg", "sampler", "scheduler", "denoise"}),
    WorkflowRole.SAVE: frozenset({"filename_prefix"}),
    WorkflowRole.PREVIEW: frozenset(),
}
_REQUIRED_WORKFLOW_ROLES: frozenset[WorkflowRole] = frozenset(
    {
        WorkflowRole.LATENT,
        WorkflowRole.POSITIVE_PROMPT,
        WorkflowRole.SAMPLER,
    }
)


def _normalize_workflow_node_id(value: object) -> str:
    """Normalise ComfyUI node ids to the string keys used by API workflows."""
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError("id must be an integer or string")
    if normalized := str(value):
        return normalized
    raise ValueError("id must not be empty")


def _non_blank_workflow_class(value: str) -> str:
    if not value.strip():
        raise ValueError("class must not be empty or whitespace-only")
    return value


class WorkflowNodeConfig(BaseModel):
    """One addressable workflow node and its writable API inputs."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: str
    class_: str = Field(alias="class", serialization_alias="class")
    inputs: dict[str, str] = Field(default_factory=dict)

    @field_validator("id", mode="before")
    @classmethod
    def normalize_id(cls, value: object) -> str:
        return _normalize_workflow_node_id(value)

    @field_validator("class_")
    @classmethod
    def validate_class(cls, value: str) -> str:
        return _non_blank_workflow_class(value)

    @field_validator("inputs")
    @classmethod
    def validate_inputs(cls, value: dict[str, str]) -> dict[str, str]:
        for parameter, input_name in value.items():
            if not input_name.strip():
                raise ValueError(f"inputs[{parameter!r}] must not be empty or whitespace-only")
        return value


class WorkflowImageInputConfig(BaseModel):
    """A LoadImage node wired to an input on the positive-prompt node.

    ``target_input`` deliberately names the input on the *positive_prompt*
    node, rather than an input on the loader itself.  Qwen image-edit prompt
    nodes receive ``image1``/``image2`` directly from LoadImage nodes.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: str
    class_: str = Field(alias="class", serialization_alias="class")
    target_input: str

    @field_validator("id", mode="before")
    @classmethod
    def normalize_id(cls, value: object) -> str:
        return _normalize_workflow_node_id(value)

    @field_validator("class_")
    @classmethod
    def validate_class(cls, value: str) -> str:
        return _non_blank_workflow_class(value)

    @field_validator("target_input")
    @classmethod
    def validate_target_input(cls, value: str) -> str:
        if not value:
            raise ValueError("target_input must not be empty")
        return value


class WorkflowModelInputConfig(BaseModel):
    """A loader input whose model filename Apex may replace."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: str
    class_: str = Field(alias="class", serialization_alias="class")
    input: str
    model_type: str | None = None
    filename: str | None = None
    label: str | None = None

    @field_validator("id", mode="before")
    @classmethod
    def normalize_id(cls, value: object) -> str:
        return _normalize_workflow_node_id(value)

    @field_validator("class_")
    @classmethod
    def validate_class(cls, value: str) -> str:
        return _non_blank_workflow_class(value)

    @field_validator("input")
    @classmethod
    def validate_input(cls, value: str) -> str:
        if not value:
            raise ValueError("input must not be empty")
        return value

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("filename must not be empty or whitespace-only")
        return value

    @model_validator(mode="after")
    def require_model_reference(self) -> WorkflowModelInputConfig:
        if self.model_type is None and self.filename is None:
            raise ValueError("model_inputs entries require at least one of model_type or filename")
        return self


class WorkflowMapConfig(BaseModel):
    """Bundle-specific mapping from Apex's parameter vocabulary to API nodes."""

    model_config = ConfigDict(extra="forbid")

    nodes: dict[WorkflowRole, WorkflowNodeConfig]
    image_inputs: list[WorkflowImageInputConfig] = Field(default_factory=list)
    model_inputs: list[WorkflowModelInputConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_map(self) -> WorkflowMapConfig:
        errors: list[str] = []
        missing_roles = sorted(role.value for role in _REQUIRED_WORKFLOW_ROLES - self.nodes.keys())
        if missing_roles:
            errors.append(f"workflow.nodes is missing required role(s): {', '.join(missing_roles)}")

        for role, node in self.nodes.items():
            if unsupported := sorted(set(node.inputs) - _WORKFLOW_ROLE_PARAMETERS[role]):
                errors.append(
                    f"workflow.nodes.{role.value}.inputs has unsupported parameter key(s): "
                    f"{', '.join(unsupported)}"
                )

        for role in (WorkflowRole.POSITIVE_PROMPT, WorkflowRole.NEGATIVE_PROMPT):
            prompt_node = self.nodes.get(role)
            if prompt_node is not None and "text" not in prompt_node.inputs:
                errors.append(f"workflow.nodes.{role.value}.inputs must include 'text'")

        id_owners: dict[str, list[str]] = {}
        for role, node in self.nodes.items():
            id_owners.setdefault(node.id, []).append(f"nodes.{role.value}")
        for index, image_input in enumerate(self.image_inputs):
            id_owners.setdefault(image_input.id, []).append(f"image_inputs[{index}]")
        for index, model_input in enumerate(self.model_inputs):
            id_owners.setdefault(model_input.id, []).append(f"model_inputs[{index}]")
        errors.extend(
            f"workflow node id {node_id!r} is reused by {', '.join(owners)}"
            for node_id, owners in sorted(id_owners.items())
            if len(owners) > 1
        )
        if errors:
            raise ValueError("; ".join(errors))
        return self


def _validate_bundle_filename(value: str | None) -> str | None:
    """Keep bundle-declared artifacts inside their resolved version directory."""
    if value is None:
        return None
    if not value.strip() or "/" in value or "\\" in value or value in {".", ".."}:
        raise ValueError(
            "workflow file must be a plain relative filename "
            "(no path separators, not empty or whitespace-only, not '.' or '..')"
        )
    return value


class BundleConfig(BaseModel):
    """Complete bundle configuration."""

    model_config = ConfigDict(extra="forbid")

    metadata: BundleMetadata
    comfyui: ComfyUIConfig | None = None
    custom_nodes: list[CustomNode] = Field(default_factory=list)
    models: list[ModelConfig] = Field(default_factory=list)

    # Bundle files
    requirements_lock_file: str | None = Field(
        default=None,
        description=(
            "Deprecated full pip freeze retained for existing bundles. Use "
            "requirements_overlay_file for dependencies the selected template does not provide."
        ),
    )
    requirements_overlay_file: str | None = Field(
        default=None,
        description=(
            "Optional additive pip overlay generated against the selected base image. "
            "It contains only packages absent from, or different in, that image."
        ),
    )
    workflow_file: str | None = None
    workflow_api_file: str | None = None
    workflow: WorkflowMapConfig | None = None
    extra_model_paths_file: str | None = None

    # Consumed by Apex, not by aisha. `hardware.comfyui_port` in particular
    # must match `Settings.comfyui_port` -- see that field's docstring.
    hardware: HardwareConfig | None = None
    generation: GenerationConfig | None = None
    readiness_marker: ReadinessMarkerConfig | None = None

    @model_validator(mode="after")
    def require_pinned_custom_nodes(self) -> BundleConfig:
        """Every custom node must carry a reproducible pin for its own source.

        ``CustomNode.validate_source_fields`` already enforces this per-node,
        so this is a second, bundle-level backstop -- the same defense-in-depth
        posture as the rest of this file's ``model_validator`` chain, not a
        substitute for the field-level check.
        """
        for node in self.custom_nodes:
            if node.source == "git" and node.commit_sha is None:
                msg = f"commit_sha is required for bundle node '{node.name}'"
                raise ValueError(msg)
            if node.source == "registry" and node.version is None:
                msg = f"version is required for bundle node '{node.name}'"
                raise ValueError(msg)
        return self

    @field_validator("workflow_file", "workflow_api_file")
    @classmethod
    def validate_workflow_filenames(cls, value: str | None) -> str | None:
        return _validate_bundle_filename(value)

    @model_validator(mode="after")
    def validate_workflow_references(self) -> BundleConfig:
        """Validate map references that need the enclosing ``models`` list."""
        if self.workflow is None:
            return self

        errors: list[str] = []
        if self.workflow_api_file is None:
            errors.append("workflow_api_file is required when workflow is declared")

        groups_by_type: dict[str, list[ModelConfig]] = {}
        for model in self.models:
            groups_by_type.setdefault(model.model_type, []).append(model)

        for index, model_input in enumerate(self.workflow.model_inputs):
            if model_input.model_type is None:
                continue
            groups = groups_by_type.get(model_input.model_type, [])
            location = f"workflow.model_inputs[{index}]"
            if not groups:
                errors.append(
                    f"{location}.model_type {model_input.model_type!r} names no models group"
                )
                continue
            if len(groups) != 1:
                if model_input.filename is None:
                    errors.append(
                        f"{location}.model_type {model_input.model_type!r} names "
                        f"{len(groups)} model groups; filename is required to disambiguate"
                    )
                    continue
                candidates = [
                    file
                    for group in groups
                    for file in group.files
                    if file.filename == model_input.filename
                ]
                if len(candidates) != 1:
                    errors.append(
                        f"{location}.filename {model_input.filename!r} matches "
                        f"{len(candidates)} files across the {len(groups)} model_type "
                        f"{model_input.model_type!r} groups; expected exactly one"
                    )
                continue
            files = groups[0].files
            if model_input.filename is None:
                if len(files) != 1:
                    errors.append(
                        f"{location}.model_type {model_input.model_type!r} resolves to "
                        f"{len(files)} files; filename is required unless exactly one file exists"
                    )
                continue
            if all(file.filename != model_input.filename for file in files):
                errors.append(
                    f"{location}.filename {model_input.filename!r} is not in model_type "
                    f"{model_input.model_type!r}"
                )

        if errors:
            raise ValueError("; ".join(errors))
        return self

    def get_all_model_files(self) -> list[tuple[ModelConfig, ModelFileConfig]]:
        """Get flat list of all model files with their parent config."""
        result: list[tuple[ModelConfig, ModelFileConfig]] = []
        for model in self.models:
            result.extend((model, file) for file in model.files)
        return result

    def requires_comfyui_setup(self) -> bool:
        """Check if this bundle requires ComfyUI setup (commit checkout)."""
        return self.comfyui is not None

    def requires_custom_nodes(self) -> bool:
        """Check if this bundle has custom nodes to install."""
        return len(self.custom_nodes) > 0

    def requires_models(self) -> bool:
        """Check if this bundle has models to download."""
        return len(self.models) > 0

    def requirements_file(self) -> str | None:
        """Return the one requirements artifact a bundle may install."""
        if self.requirements_lock_file is not None and self.requirements_overlay_file is not None:
            raise ValueError(
                "bundle declares both requirements_lock_file and requirements_overlay_file"
            )
        return self.requirements_overlay_file or self.requirements_lock_file


class DeploymentPlan(BaseModel):
    """Deployment plan showing what will be installed."""

    mode: DeployMode
    bundle_name: str
    bundle_version: str

    # What will be done
    will_update_comfyui: bool = False
    will_install_base_requirements: bool = False
    will_install_locked_requirements: bool = False
    will_install_custom_nodes: bool = False
    will_download_models: bool = False
    will_install_workflow: bool = False
    will_verify: bool = False

    # Counts
    custom_nodes_count: int = 0
    models_count: int = 0
    model_files_count: int = 0
    missing_url_files_count: int = 0

    @classmethod
    def from_bundle(
        cls,
        bundle: BundleConfig,
        mode: DeployMode,
        verify: bool = True,
    ) -> DeploymentPlan:
        """Create deployment plan from bundle config and mode."""
        model_files = bundle.get_all_model_files()
        is_full = mode == DeployMode.FULL
        needs_comfyui_setup = is_full and bundle.requires_comfyui_setup()

        return cls(
            mode=mode,
            bundle_name=bundle.metadata.name,
            bundle_version=bundle.metadata.version,
            will_update_comfyui=needs_comfyui_setup,
            will_install_base_requirements=needs_comfyui_setup,
            will_install_locked_requirements=is_full and bundle.requirements_file() is not None,
            will_install_custom_nodes=is_full and bundle.requires_custom_nodes(),
            will_download_models=bundle.requires_models(),
            will_install_workflow=bundle.workflow_file is not None,
            will_verify=verify,
            custom_nodes_count=len(bundle.custom_nodes) if is_full else 0,
            models_count=len(bundle.models),
            model_files_count=len(model_files),
            missing_url_files_count=sum(not f.url for _m, f in model_files),
        )


# Singleton settings instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get application settings (singleton)."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Reset settings singleton (useful for testing)."""
    global _settings
    _settings = None
