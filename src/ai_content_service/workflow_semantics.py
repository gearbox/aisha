"""Shared, fail-closed vocabulary for workflow contract v2.

The workflow-map schema, graph inference, offline validation, and snapshot
authoring all consume these semantics.  Keeping them here prevents a loader
or Apex family from acquiring different meanings at those boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final


class WorkflowMedia(str, Enum):
    """The graph-level media shape addressed by this workflow contract."""

    IMAGE = "image"
    VIDEO = "video"


class WorkflowMediaKind(str, Enum):
    """The kind of uploaded asset accepted by a media-loader node."""

    IMAGE = "image"
    VIDEO = "video"


class WorkflowMediaSlot(str, Enum):
    """The semantic position of an uploaded asset in the generation request."""

    REFERENCE = "reference"
    FIRST_FRAME = "first_frame"
    LAST_FRAME = "last_frame"
    SOURCE = "source"


@dataclass(frozen=True, slots=True)
class MediaLoaderSpec:
    """The contract-relevant behaviour of a supported ComfyUI media loader."""

    kind: WorkflowMediaKind
    input_name: str
    output_slots: frozenset[int]


# Output slots are verified against each loader's ComfyUI ``/object_info``
# declaration.  In particular, LoadImage's mask is output 1 and must never
# certify an image media edge; LoadImageMask's only MASK output is slot 0.
MEDIA_LOADER_SPECS: Final[dict[str, MediaLoaderSpec]] = {
    "LoadImage": MediaLoaderSpec(WorkflowMediaKind.IMAGE, "image", frozenset({0})),
    "LoadImageMask": MediaLoaderSpec(WorkflowMediaKind.IMAGE, "image", frozenset({0})),
    "LoadVideo": MediaLoaderSpec(WorkflowMediaKind.VIDEO, "file", frozenset({0})),
    "VHS_LoadVideo": MediaLoaderSpec(WorkflowMediaKind.VIDEO, "video", frozenset({0})),
}

# Request slots describe API capability, not arbitrary topology.  There is no
# video reference slot: reference/edit conditioning is an image input.
MEDIA_SLOT_KINDS: Final[dict[WorkflowMediaSlot, WorkflowMediaKind]] = {
    WorkflowMediaSlot.REFERENCE: WorkflowMediaKind.IMAGE,
    WorkflowMediaSlot.FIRST_FRAME: WorkflowMediaKind.IMAGE,
    WorkflowMediaSlot.LAST_FRAME: WorkflowMediaKind.IMAGE,
    WorkflowMediaSlot.SOURCE: WorkflowMediaKind.VIDEO,
}

# This is the one source for the Apex bundle-index family vocabulary and its
# workflow-media meaning.  Contract validation derives its accepted set from
# this mapping; snapshot authoring uses the same values.
APEX_MODEL_TYPE_MEDIA: Final[dict[str, WorkflowMedia]] = {
    "aisha-image": WorkflowMedia.IMAGE,
    "aisha-video": WorkflowMedia.VIDEO,
}


def media_from_apex_model_type(model_type: str) -> WorkflowMedia:
    """Return the workflow media implied by a known Apex model family.

    ``None`` is intentionally not accepted here: callers must explicitly
    preserve their documented no-family fallback, while invalid non-null
    values fail rather than becoming image workflows by accident.
    """
    try:
        return APEX_MODEL_TYPE_MEDIA[model_type]
    except KeyError as exc:
        raise ValueError(f"Unknown Apex model_type {model_type!r}.") from exc
