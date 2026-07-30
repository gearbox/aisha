# ==============================================================================
# AISHA Makefile
# ==============================================================================
# Common operations for development and deployment
# ==============================================================================

.PHONY: help install dev test lint sync deploy list clean

# Default target
help:
	@echo "AISHA - AI Content Service Deployment"
	@echo ""
	@echo "Usage: make <target>"
	@echo ""
	@echo "Development:"
	@echo "  install     Install aisha in production mode"
	@echo "  dev         Install with development dependencies"
	@echo "  test        Run test suite"
	@echo "  lint        Run linters (ruff, mypy)"
	@echo ""
	@echo "Registry:"
	@echo "  sync        Sync all bundle registries"
	@echo "  list        List available bundles"
	@echo ""
	@echo "Deployment:"
	@echo "  deploy      Deploy default bundle (set ACS_BUNDLE)"
	@echo "  deploy-wan  Deploy WAN 2.2 I2V bundle"
	@echo ""
	@echo "Utilities:"
	@echo "  clean       Clean caches and build artifacts"
	@echo "  env         Show current configuration"

# -----------------------------------------------------------------------------
# Development
# -----------------------------------------------------------------------------

install:
	uv pip install -e . --system

dev:
	uv pip install -e ".[dev]" --system
	pre-commit install

test:
	pytest tests/ -v --cov=ai_content_service --cov-report=term-missing

lint:
	uv run ruff check
	uv run ruff format --check
	uv run mypy --strict src scripts
	uv run pyright

format:
	uv run ruff check --fix
	uv run ruff format

# -----------------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------------

sync:
	acs bundle sync

list:
	acs bundle list

# -----------------------------------------------------------------------------
# Deployment
# -----------------------------------------------------------------------------

deploy:
ifndef ACS_BUNDLE
	$(error ACS_BUNDLE is not set. Use: make deploy ACS_BUNDLE=wan_2.2_i2v)
endif
	acs deploy -b $(ACS_BUNDLE)

deploy-wan:
	acs deploy -b wan_2.2_i2v

deploy-models-only:
ifndef ACS_BUNDLE
	$(error ACS_BUNDLE is not set)
endif
	acs deploy -b $(ACS_BUNDLE) --models-only

deploy-dry-run:
ifndef ACS_BUNDLE
	$(error ACS_BUNDLE is not set)
endif
	acs deploy -b $(ACS_BUNDLE) --dry-run

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

clean:
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

env:
	@echo "Current Configuration:"
	@echo "  ACS_COMFYUI_PATH=${ACS_COMFYUI_PATH:-/workspace/ComfyUI}"
	@echo "  ACS_BUNDLES_REPO=${ACS_BUNDLES_REPO:-<not set>}"
	@echo "  ACS_BUNDLE=${ACS_BUNDLE:-<not set>}"
	@echo "  ACS_GITHUB_TOKEN=${ACS_GITHUB_TOKEN:+<set>}"
	@echo "  ACS_HF_TOKEN=${ACS_HF_TOKEN:+<set>}"

# -----------------------------------------------------------------------------
# Release
# -----------------------------------------------------------------------------

build:
	python -m build

publish-test:
	twine upload --repository testpypi dist/*

publish:
	twine upload dist/*
