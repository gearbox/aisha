"""Regression checks for the CI-based SonarQube Cloud integration."""

from __future__ import annotations

import re
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).parent.parent


def _properties(path: Path) -> dict[str, str]:
    properties: dict[str, str] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", maxsplit=1)
            properties[key.strip()] = value.strip()
    return properties


def test_sonar_python_versions_match_supported_minors() -> None:
    pyproject = REPOSITORY_ROOT.joinpath("pyproject.toml").read_text()
    properties = _properties(REPOSITORY_ROOT / "sonar-project.properties")

    requires_python = re.search(r'^requires-python\s*=\s*"([^"]+)"$', pyproject, re.MULTILINE)
    assert requires_python is not None
    assert requires_python[1] == ">=3.10,<3.13"
    assert properties["sonar.python.version"] == "3.10,3.11,3.12"


def test_sonar_imports_the_canonical_coverage_report() -> None:
    properties = _properties(REPOSITORY_ROOT / "sonar-project.properties")
    workflow = REPOSITORY_ROOT.joinpath(".github/workflows/ci.yml").read_text()

    assert properties["sonar.python.coverage.reportPaths"] == "coverage.xml"
    assert "relative_files = true" in REPOSITORY_ROOT.joinpath("pyproject.toml").read_text()
    assert "name: python-coverage-sonar" in workflow
    assert "name: SonarQube Cloud" in workflow
    assert "needs: [lint, test]" in workflow
    assert (
        "uses: SonarSource/sonarqube-scan-action@7006c4492b2e0ee0f816d36501671557c97f5995"
        in workflow
    )
