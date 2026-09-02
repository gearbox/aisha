"""Regression coverage for the shared workflow contract version."""

from __future__ import annotations

from ai_content_service import bundle_contract, workflow_map
from ai_content_service.config import WORKFLOW_CONTRACT_VERSION, BundleConfig
from tests.workflow_map_helpers import _raw_bundle


def test_contract_version_constant_matches_every_consumer() -> None:
    assert WORKFLOW_CONTRACT_VERSION == 2
    assert bundle_contract.WORKFLOW_CONTRACT_VERSION == WORKFLOW_CONTRACT_VERSION
    assert workflow_map.WORKFLOW_CONTRACT_VERSION == WORKFLOW_CONTRACT_VERSION

    config = BundleConfig.model_validate(_raw_bundle())
    assert config.workflow is not None
    assert config.workflow.contract_version == WORKFLOW_CONTRACT_VERSION
