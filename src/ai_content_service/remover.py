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
        self._models_root = settings.models_path.resolve()
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
            if unknown := supplied - set(resident):
                log.error(
                    "residency.retain_unresolvable",
                    unknown_bundles=sorted(unknown),
                    manifest_bundles=sorted(retained),
                    apex_bundles=sorted(supplied),
                )
                unknown_names = ", ".join(sorted(unknown))
                raise RemovalError(
                    f"cannot safely remove {bundle_name!r}: apex retains bundle(s) "
                    f"absent from this node's residency manifest: {unknown_names}. "
                    "The residency manifest and apex disagree about what is on this node."
                )
            if retained - supplied:
                log.warning(
                    "residency.retain_mismatch",
                    manifest_bundles=sorted(retained),
                    apex_bundles=sorted(supplied),
                )
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
        projected_prunes = self._would_prune(full_paths.values())

        if dry_run:
            return RemovalResult(
                bundle=bundle_name,
                files_removed=files_removed,
                files_retained=files_retained,
                bytes_freed=bytes_freed,
                workflow_removed=self._workflow_exists(target.workflow_filename),
                directories_pruned=tuple(self._relative(path) for path in projected_prunes),
            )

        for relative, path in full_paths.items():
            try:
                path.unlink()
            except FileNotFoundError:
                log.debug("removal.model_missing", bundle=bundle_name, path=relative)
        actual_pruned: list[str] = []
        for directory in projected_prunes:
            try:
                directory.rmdir()
            except OSError:
                log.warning("removal.directory_prune_failed", path=str(directory))
            else:
                actual_pruned.append(self._relative(directory))

        workflow_removed = False
        if target.workflow_filename is not None:
            try:
                self._workflow_manager.remove_workflow(target.workflow_filename)
            except WorkflowError:
                log.debug(
                    "removal.workflow_missing",
                    bundle=bundle_name,
                    workflow=target.workflow_filename,
                )
            else:
                workflow_removed = True
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
        root = self._models_root
        path = root / relative
        try:
            path.resolve().relative_to(root)
        except ValueError as exc:
            raise RemovalError(f"refusing to remove model outside {root}: {relative!r}") from exc
        return path

    @staticmethod
    def _file_size(path: Path) -> int:
        try:
            return path.stat().st_size
        except OSError:
            return 0

    def _would_prune(self, paths: Iterable[Path]) -> tuple[Path, ...]:
        """Project empty directories and order them for safe ``rmdir`` calls."""
        deleted = set(paths)
        roots = {path.parent for path in deleted}
        pruned: set[Path] = set()
        checked: dict[Path, bool] = {}

        def would_be_empty(directory: Path) -> bool:
            if directory in checked:
                return checked[directory]
            try:
                entries = tuple(directory.iterdir())
            except OSError:
                checked[directory] = False
                return False
            for entry in entries:
                if entry in deleted:
                    continue
                if not entry.is_dir() or not would_be_empty(entry):
                    checked[directory] = False
                    return False
            checked[directory] = True
            return True

        for directory in sorted(roots, key=lambda candidate: len(candidate.parts), reverse=True):
            current = directory
            while self._is_model_subdirectory(current) and would_be_empty(current):
                pruned.add(current)
                current = current.parent
        return tuple(sorted(pruned, key=lambda candidate: len(candidate.parts), reverse=True))

    def _is_model_subdirectory(self, directory: Path) -> bool:
        """Return whether a directory is safely below, but not equal to, models root."""
        resolved = directory.resolve()
        try:
            resolved.relative_to(self._models_root)
        except ValueError:
            return False
        return resolved != self._models_root

    def _workflow_exists(self, workflow_filename: str | None) -> bool:
        """Return whether the recorded workflow is currently installed."""
        return workflow_filename is not None and any(
            path.name == workflow_filename for path in self._workflow_manager.list_workflows()
        )

    def _relative(self, path: Path) -> str:
        return path.relative_to(self._models_root).as_posix()
