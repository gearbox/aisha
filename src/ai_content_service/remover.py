"""Reference-counted model removal for resident bundles.

Custom nodes and pip packages are deliberately never removed: they are cheap
to retain, almost never interfere, and reference-counting either could break a
live bundle for negligible disk savings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog

from .workflows import WorkflowError

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

    from .config import Settings
    from .residency import ResidencyStore
    from .workflows import WorkflowManager


log = structlog.get_logger()


class RemovalError(Exception):
    """Raised when a bundle cannot be safely removed."""


@dataclass(frozen=True, slots=True)
class RemovalResult:
    """The disk effects (or dry-run projection) of removing one bundle."""

    bundle: str
    files_removed: tuple[str, ...]
    files_retained: tuple[str, ...]
    bytes_freed: int
    workflow_removed: bool
    directories_pruned: tuple[str, ...]


class BundleRemover:
    """Remove only model files no other resident bundle still declares."""

    def __init__(
        self,
        settings: Settings,
        *,
        residency: ResidencyStore,
        workflow_manager: WorkflowManager,
    ) -> None:
        self._settings = settings
        self._residency = residency
        self._workflow_manager = workflow_manager

    async def remove(
        self,
        bundle_name: str,
        *,
        retain_bundles: Sequence[str] | None = None,
        dry_run: bool = False,
    ) -> RemovalResult:
        """Remove unshared bundle files and its workflow from this node."""
        resident = self._residency.load()
        target = resident.get(bundle_name)
        if target is None:
            names = ", ".join(sorted(resident)) or "none"
            raise RemovalError(
                f"bundle {bundle_name!r} is not recorded in residency; resident bundles: {names}"
            )

        retained = {name for name in resident if name != bundle_name}
        if retain_bundles is not None:
            supplied = set(retain_bundles) - {bundle_name}
            if supplied != retained:
                log.warning(
                    "residency.retain_mismatch",
                    manifest_bundles=sorted(retained),
                    apex_bundles=sorted(supplied),
                )
                retained |= supplied
        retained_paths = {
            file.path
            for name, bundle in resident.items()
            if name in retained
            for file in bundle.model_files
        }
        target_paths = tuple(file.path for file in target.model_files)
        files_retained = tuple(path for path in target_paths if path in retained_paths)
        files_removed = tuple(path for path in target_paths if path not in retained_paths)

        full_paths = {path: self._model_path(path) for path in files_removed}
        bytes_freed = sum(self._file_size(path) for path in full_paths.values())
        directories_pruned = self._would_prune(full_paths.values())
        workflow_removed = target.workflow_filename is not None

        if dry_run:
            return RemovalResult(
                bundle=bundle_name,
                files_removed=files_removed,
                files_retained=files_retained,
                bytes_freed=bytes_freed,
                workflow_removed=workflow_removed,
                directories_pruned=directories_pruned,
            )

        actual_pruned: list[str] = []
        for relative, path in full_paths.items():
            try:
                path.unlink()
            except FileNotFoundError:
                log.debug("removal.model_missing", bundle=bundle_name, path=relative)
            actual_pruned.extend(self._prune_empty_parents(path.parent))

        if target.workflow_filename is not None:
            try:
                self._workflow_manager.remove_workflow(target.workflow_filename)
            except WorkflowError:
                log.debug(
                    "removal.workflow_missing",
                    bundle=bundle_name,
                    workflow=target.workflow_filename,
                )
        self._residency.forget(bundle_name)
        return RemovalResult(
            bundle=bundle_name,
            files_removed=files_removed,
            files_retained=files_retained,
            bytes_freed=bytes_freed,
            workflow_removed=workflow_removed,
            directories_pruned=tuple(actual_pruned),
        )

    def _model_path(self, relative: str) -> Path:
        """Resolve a manifest path under models, refusing a future unsafe entry."""
        root = self._settings.models_path
        path = root / relative
        try:
            path.resolve().relative_to(root.resolve())
        except ValueError as exc:
            raise RemovalError(f"refusing to remove model outside {root}: {relative!r}") from exc
        return path

    @staticmethod
    def _file_size(path: Path) -> int:
        try:
            return path.stat().st_size
        except OSError:
            return 0

    def _would_prune(self, paths: Iterable[Path]) -> tuple[str, ...]:
        """Project post-removal empty directories without mutating the filesystem."""
        deleted = set(paths)
        roots = {path.parent for path in deleted}
        pruned: list[Path] = []
        for directory in sorted(roots, key=lambda candidate: len(candidate.parts), reverse=True):
            current = directory
            while (
                current != self._settings.models_path
                and current not in pruned
                and self._would_be_empty(current, deleted)
            ):
                pruned.append(current)
                current = current.parent
        return tuple(self._relative(path) for path in pruned)

    def _would_be_empty(self, directory: Path, deleted: set[Path]) -> bool:
        try:
            entries = tuple(directory.iterdir())
        except OSError:
            return False
        return all(
            entry in deleted or (entry.is_dir() and self._would_be_empty(entry, deleted))
            for entry in entries
        )

    def _prune_empty_parents(self, directory: Path) -> list[str]:
        """Remove empty ancestors, never the models root itself."""
        pruned: list[str] = []
        while directory != self._settings.models_path:
            try:
                directory.rmdir()
            except OSError:
                break
            pruned.append(self._relative(directory))
            directory = directory.parent
        return pruned

    def _relative(self, path: Path) -> str:
        return path.relative_to(self._settings.models_path).as_posix()
