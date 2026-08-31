"""Node-local bundle residency manifest.

The manifest is intentionally a small, dependency-free record of bundle
declarations.  It has no file locking: Aisha executes one deployment at a
time on a node, and Phase G preserves that one-in-flight-command invariant.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Literal, cast

RESIDENCY_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class ResidentModelFile:
    """One model file a resident bundle declares beneath ``models/``."""

    path: str
    sha256: str | None
    size_bytes: int | None


@dataclass(frozen=True, slots=True)
class ResidentCustomNode:
    """One pinned custom node a resident bundle declares."""

    name: str
    source: Literal["git", "registry"]
    pin: str


@dataclass(frozen=True, slots=True)
class ResidentBundle:
    """The declaration recorded for one resident bundle name."""

    name: str
    version: str
    registry: str | None
    mode: str
    deployed_at: str
    model_files: tuple[ResidentModelFile, ...]
    custom_nodes: tuple[ResidentCustomNode, ...]
    workflow_filename: str | None
    readiness_node_class: str | None
    pending_restart: bool


class ResidencyError(Exception):
    """Raised when residency cannot safely be loaded or stored."""


class ResidencyStore:
    """Read and atomically update the node-local residency manifest."""

    def __init__(self, path: Path) -> None:
        self._path = path

    @property
    def path(self) -> Path:
        """Path to the backing manifest, useful for CLI presentation."""
        return self._path

    def load(self) -> dict[str, ResidentBundle]:
        """Return all resident bundles, refusing corrupt state loudly."""
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise self._error(f"could not read valid JSON ({exc})") from exc
        try:
            return self._parse_payload(payload)
        except (TypeError, ValueError) as exc:
            raise self._error(f"failed validation ({exc})") from exc

    def record(self, bundle: ResidentBundle) -> None:
        """Replace the entry for a bundle name and atomically persist it."""
        resident = self.load()
        resident[bundle.name] = bundle
        self._save(resident)

    def forget(self, bundle_name: str) -> ResidentBundle | None:
        """Forget and return one bundle, or ``None`` when it was absent."""
        resident = self.load()
        removed = resident.pop(bundle_name, None)
        if removed is not None:
            self._save(resident)
        return removed

    def mark_all_restarted(self) -> None:
        """Clear the advisory restart flag after a successful ComfyUI restart."""
        resident = self.load()
        updated = {
            name: replace(bundle, pending_restart=False) for name, bundle in resident.items()
        }
        if any(bundle.pending_restart for bundle in resident.values()):
            self._save(updated)

    def _save(self, resident: dict[str, ResidentBundle]) -> None:
        payload = {
            "schema_version": RESIDENCY_SCHEMA_VERSION,
            "bundles": {name: self._bundle_to_dict(bundle) for name, bundle in resident.items()},
        }
        tmp_path = self._path.with_name(f"{self._path.name}.tmp")
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with tmp_path.open("w", encoding="utf-8") as file:
                json.dump(payload, file, indent=2, sort_keys=True)
                file.write("\n")
            tmp_path.replace(self._path)
        except OSError as exc:
            raise ResidencyError(f"could not save residency manifest {self._path}: {exc}") from exc

    def _error(self, reason: str) -> ResidencyError:
        return ResidencyError(
            f"residency manifest {self._path} {reason}; delete the file only if you accept "
            "that this deployment's collision checks cannot run"
        )

    @staticmethod
    def _bundle_to_dict(bundle: ResidentBundle) -> dict[str, object]:
        return cast("dict[str, object]", asdict(bundle))

    @classmethod
    def _parse_payload(cls, payload: object) -> dict[str, ResidentBundle]:
        root = cls._mapping(payload, "manifest")
        if root.get("schema_version") != RESIDENCY_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {RESIDENCY_SCHEMA_VERSION}, got {root.get('schema_version')!r}"
            )
        bundles = cls._mapping(root.get("bundles"), "bundles")
        parsed: dict[str, ResidentBundle] = {}
        for name, value in bundles.items():
            if not isinstance(name, str) or not name:
                raise ValueError("bundle keys must be non-empty strings")
            bundle = cls._parse_bundle(value)
            if bundle.name != name:
                raise ValueError(f"bundle key {name!r} does not match entry name {bundle.name!r}")
            parsed[name] = bundle
        return parsed

    @classmethod
    def _parse_bundle(cls, value: object) -> ResidentBundle:
        data = cls._mapping(value, "bundle")
        return ResidentBundle(
            name=cls._string(data.get("name"), "bundle.name"),
            version=cls._string(data.get("version"), "bundle.version"),
            registry=cls._optional_string(data.get("registry"), "bundle.registry"),
            mode=cls._string(data.get("mode"), "bundle.mode"),
            deployed_at=cls._string(data.get("deployed_at"), "bundle.deployed_at"),
            model_files=tuple(
                cls._parse_model_file(item)
                for item in cls._list(data.get("model_files"), "bundle.model_files")
            ),
            custom_nodes=tuple(
                cls._parse_custom_node(item)
                for item in cls._list(data.get("custom_nodes"), "bundle.custom_nodes")
            ),
            workflow_filename=cls._optional_string(
                data.get("workflow_filename"), "bundle.workflow_filename"
            ),
            readiness_node_class=cls._optional_string(
                data.get("readiness_node_class"), "bundle.readiness_node_class"
            ),
            pending_restart=cls._boolean(data.get("pending_restart"), "bundle.pending_restart"),
        )

    @classmethod
    def _parse_model_file(cls, value: object) -> ResidentModelFile:
        data = cls._mapping(value, "model file")
        path = cls._string(data.get("path"), "model file.path")
        cls._validate_model_path(path)
        size = data.get("size_bytes")
        if size is not None and (isinstance(size, bool) or not isinstance(size, int) or size < 0):
            raise ValueError("model file.size_bytes must be a non-negative integer or null")
        return ResidentModelFile(
            path=path,
            sha256=cls._optional_string(data.get("sha256"), "model file.sha256"),
            size_bytes=cast("int | None", size),
        )

    @classmethod
    def _parse_custom_node(cls, value: object) -> ResidentCustomNode:
        data = cls._mapping(value, "custom node")
        source = data.get("source")
        if source not in {"git", "registry"}:
            raise ValueError("custom node.source must be 'git' or 'registry'")
        return ResidentCustomNode(
            name=cls._string(data.get("name"), "custom node.name"),
            source=cast("Literal['git', 'registry']", source),
            pin=cls._string(data.get("pin"), "custom node.pin"),
        )

    @staticmethod
    def _mapping(value: object, label: str) -> dict[str, object]:
        if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
            raise ValueError(f"{label} must be an object")
        return cast("dict[str, object]", value)

    @staticmethod
    def _list(value: object, label: str) -> list[object]:
        if not isinstance(value, list):
            raise ValueError(f"{label} must be an array")
        return value

    @staticmethod
    def _string(value: object, label: str) -> str:
        if not isinstance(value, str) or not value:
            raise ValueError(f"{label} must be a non-empty string")
        return value

    @classmethod
    def _optional_string(cls, value: object, label: str) -> str | None:
        return None if value is None else cls._string(value, label)

    @staticmethod
    def _boolean(value: object, label: str) -> bool:
        if not isinstance(value, bool):
            raise ValueError(f"{label} must be a boolean")
        return value

    @staticmethod
    def _validate_model_path(path: str) -> None:
        parsed = PurePosixPath(path)
        if (
            parsed.is_absolute()
            or path != parsed.as_posix()
            or path in {"", "."}
            or ".." in parsed.parts
        ):
            raise ValueError("model file.path must be a POSIX path relative to models/")
