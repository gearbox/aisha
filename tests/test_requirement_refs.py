"""Tests for the shared direct-reference predicate.

Both the overlay writer (snapshot.py) and the requirements installer
(comfyui.py) import ``is_missing_local_reference`` from here; see R2 of
agent_prompts/overlay-directref-remediation-prompt.md.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ai_content_service.requirement_refs import is_missing_local_reference

if TYPE_CHECKING:
    from pathlib import Path


def test_non_file_scheme_is_never_missing() -> None:
    assert is_missing_local_reference("https://example.com/pkg.whl") is False
    assert is_missing_local_reference("git+https://github.com/owner/repo@" + "a" * 40) is False


def test_file_reference_to_nonexistent_path_is_missing() -> None:
    assert is_missing_local_reference("file:///conda-builder/does-not-exist") is True


def test_file_reference_to_existing_path_is_not_missing(tmp_path: Path) -> None:
    artifact = tmp_path / "package.whl"
    artifact.write_bytes(b"wheel")
    assert is_missing_local_reference(artifact.as_uri()) is False


def test_file_reference_with_localhost_netloc(tmp_path: Path) -> None:
    artifact = tmp_path / "package.whl"
    artifact.write_bytes(b"wheel")
    assert is_missing_local_reference(f"file://localhost{artifact}") is False


def test_file_reference_with_nonlocal_netloc_is_missing() -> None:
    assert is_missing_local_reference("file://remotehost/some/path") is True
