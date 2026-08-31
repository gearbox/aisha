"""Vocabulary shared by provisioning telemetry producers and consumers."""

from __future__ import annotations

from enum import StrEnum

from uuid_utils.compat import uuid7

SCHEMA_VERSION = 2


class OperationKind(StrEnum):
    """The provisioning-like activity represented by an operation stream."""

    SESSION_BOOTSTRAP = "session_bootstrap"
    BUNDLE_PROVISION = "bundle_provision"
    BUNDLE_REMOVAL = "bundle_removal"
    COMFYUI_RESTART = "comfyui_restart"


class OperationStatus(StrEnum):
    """Lifecycle state for an operation."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ProvisioningPhase(StrEnum):
    """Stable identifiers for individual provisioning phases."""

    PREFLIGHT = "preflight"
    COMFYUI = "comfyui"
    REQUIREMENTS_BASE = "requirements_base"
    REQUIREMENTS_LOCKED = "requirements_locked"
    CUSTOM_NODES = "custom_nodes"
    MODELS = "models"
    WORKFLOW = "workflow"
    VERIFYING = "verifying"
    RESTART = "restart"


class WorkUnit(StrEnum):
    """Units used in generic progress envelopes."""

    BYTES = "bytes"
    FILES = "files"
    ITEMS = "items"


class EtaBasis(StrEnum):
    """The derivation used for a reported ETA."""

    LIVE_THROUGHPUT = "live_throughput"


def new_id() -> str:
    """Return a fresh UUIDv7 for an operation or event identifier."""
    # ``compat`` returns a stdlib uuid.UUID, which Apex's model layer requires.
    return str(uuid7())
