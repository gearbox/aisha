"""Pure collision checks for shared-node additive bundle deployment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .config import BundleConfig
    from .residency import ResidentBundle


@dataclass(frozen=True, slots=True)
class PreflightFinding:
    """One stable preflight decision and its operator-facing explanation."""

    code: str
    detail: str


@dataclass(frozen=True, slots=True)
class AdditivePreflightReport:
    """All additive safety findings, split by whether they block deployment."""

    blocking: tuple[PreflightFinding, ...] = ()
    advisory: tuple[PreflightFinding, ...] = ()

    @property
    def ok(self) -> bool:
        """Whether no safety issue prevents an additive deployment."""
        return not self.blocking


def check_additive(
    bundle: BundleConfig,
    *,
    resident: Mapping[str, ResidentBundle],
    current_comfyui_commit: str | None,
) -> AdditivePreflightReport:
    """Report all known collisions without touching the filesystem or network."""
    blocking: list[PreflightFinding] = []
    advisory: list[PreflightFinding] = []

    if bundle.requirements_lock_file is not None and bundle.requirements_overlay_file is None:
        blocking.append(
            PreflightFinding(
                code="requirements_full_lock",
                detail=(
                    f"bundle {bundle.metadata.name!r} declares requirements_lock_file without "
                    "requirements_overlay_file; regenerate an overlay against the template before "
                    "an additive deploy"
                ),
            )
        )

    if bundle.comfyui is not None:
        expected_commit = bundle.comfyui.commit
        if current_comfyui_commit is None:
            advisory.append(
                PreflightFinding(
                    code="comfyui_revision_mismatch",
                    detail=(
                        f"bundle {bundle.metadata.name!r} requires ComfyUI {expected_commit}, "
                        "but the current revision cannot be determined because this node is not a "
                        "git checkout"
                    ),
                )
            )
        elif expected_commit != current_comfyui_commit:
            blocking.append(
                PreflightFinding(
                    code="comfyui_revision_mismatch",
                    detail=(
                        f"bundle {bundle.metadata.name!r} requires ComfyUI {expected_commit}, "
                        f"but this node is running {current_comfyui_commit}"
                    ),
                )
            )

    for node in bundle.custom_nodes:
        # BundleConfig validates the source-specific pin.  Avoid substituting
        # an empty string here: two malformed declarations must not compare as equal.
        pin = cast("str", node.commit_sha if node.source == "git" else node.version)
        for resident_bundle in resident.values():
            for resident_node in resident_bundle.custom_nodes:
                if resident_node.name != node.name:
                    continue
                if resident_node.source != node.source or resident_node.pin != pin:
                    blocking.append(
                        PreflightFinding(
                            code="custom_node_pin_conflict",
                            detail=(
                                f"resident bundle {resident_bundle.name!r} pins custom node "
                                f"{node.name!r} as {resident_node.source}@{resident_node.pin}, "
                                f"but {bundle.metadata.name!r} requires {node.source}@{pin}"
                            ),
                        )
                    )

    for model, file in bundle.get_all_model_files():
        path = f"{model.target_subpath}/{file.filename}"
        for resident_bundle in resident.values():
            for resident_file in resident_bundle.model_files:
                if resident_file.path != path:
                    continue
                if resident_file.sha256 is None or file.sha256 is None:
                    advisory.append(
                        PreflightFinding(
                            code="model_path_unverifiable",
                            detail=(
                                f"bundle {bundle.metadata.name!r} and resident bundle "
                                f"{resident_bundle.name!r} both declare model path {path!r}, but "
                                "one or both declarations have no sha256"
                            ),
                        )
                    )
                elif resident_file.sha256 != file.sha256:
                    blocking.append(
                        PreflightFinding(
                            code="model_sha_collision",
                            detail=(
                                f"bundle {bundle.metadata.name!r} declares {path!r} with sha256 "
                                f"{file.sha256}, but resident bundle {resident_bundle.name!r} "
                                f"declares {resident_file.sha256}"
                            ),
                        )
                    )

    return AdditivePreflightReport(blocking=tuple(blocking), advisory=tuple(advisory))
