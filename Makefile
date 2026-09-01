SHELL := /bin/bash
.DEFAULT_GOAL := help

UV ?= uv
QUICKSTART_CONFIG ?= configs/quickstart.yaml
RELEASE_CONFIG ?= configs/release.yaml

.PHONY: help install format lint test quickstart reproduce demo check

help: ## Show available project commands
	@awk 'BEGIN {FS = ":.*## "; printf "Offer-to-Exit commands:\n\n"} /^[a-zA-Z_-]+:.*## / {printf "  %-14s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Create the environment and install project plus development tools
	$(UV) sync --all-extras

format: ## Apply deterministic Python formatting and safe lint fixes
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

lint: ## Check formatting, lint rules, and static types
	$(UV) run ruff format --check .
	$(UV) run ruff check .
	$(UV) run mypy

test: ## Run the automated test suite with coverage
	$(UV) run pytest

quickstart: ## Run the small generated fixture end to end
	$(UV) run offer-to-exit run --config $(QUICKSTART_CONFIG)

reproduce: ## Rebuild the deterministic release experiment
	$(UV) run offer-to-exit run --config $(RELEASE_CONFIG)

demo: ## Build the static local decision explorer
	$(UV) run offer-to-exit demo --config $(QUICKSTART_CONFIG)

check: lint test quickstart ## Run the full local quality checks
