SHELL := /bin/bash
.DEFAULT_GOAL := help

UV ?= uv
QUICKSTART_CONFIG ?= configs/quickstart.yaml
RELEASE_CONFIG ?= configs/release.yaml
FLORIDA_RAW_DIR ?= data/raw
FLORIDA_PROCESSED_DIR ?= data/processed

.PHONY: help install format lint test florida-fetch florida-prepare florida-data florida-study release quickstart reproduce demo check

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

florida-fetch: ## Download Tampa and Orlando county sales into the ignored raw-data directory
	$(UV) run offer-to-exit fetch --raw-dir $(FLORIDA_RAW_DIR)

florida-prepare: ## Remove direct identifiers and link named-iBuyer inventory episodes
	$(UV) run offer-to-exit prepare --raw-dir $(FLORIDA_RAW_DIR) --processed-dir $(FLORIDA_PROCESSED_DIR)

florida-data: florida-fetch florida-prepare ## Rebuild the two-market, privacy-safe Florida inputs

florida-study: ## Fit in Tampa, score Orlando, and rebuild the aggregate Florida evidence
	$(UV) run offer-to-exit florida-study --transactions $(FLORIDA_PROCESSED_DIR)/florida_transactions_safe.csv.gz --episodes $(FLORIDA_PROCESSED_DIR)/named_ibuyer_episodes_safe.csv.gz --output-dir artifacts/release

release: florida-data florida-study reproduce ## Rebuild all real-data and controlled-experiment evidence

quickstart: ## Run the small generated fixture end to end
	$(UV) run offer-to-exit run --config $(QUICKSTART_CONFIG)

reproduce: ## Rebuild the deterministic release experiment
	$(UV) run offer-to-exit run --config $(RELEASE_CONFIG)

demo: ## Build the static local decision explorer
	$(UV) run offer-to-exit demo --config $(QUICKSTART_CONFIG)

check: lint test quickstart ## Run the full local quality checks
