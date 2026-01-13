"""
Makefile for Systematic Alpha Generation project
Automates common tasks: setup, installation, testing, running
"""

.PHONY: help install dev-setup test test-coverage run-full run-data run-features run-train run-backtest clean docs lint format

help:
	@echo "Systematic Alpha Generation - Available Commands"
	@echo ""
	@echo "Setup Commands:"
	@echo "  make install          Install project dependencies"
	@echo "  make dev-setup        Setup development environment with pre-commit hooks"
	@echo ""
	@echo "Execution Commands:"
	@echo "  make run-full         Run complete pipeline (data → features → model → backtest)"
	@echo "  make run-data         Download data from Yahoo Finance"
	@echo "  make run-features     Engineer features"
	@echo "  make run-train        Train ensemble model"
	@echo "  make run-backtest     Run walk-forward backtest"
	@echo ""
	@echo "Testing & Quality:"
	@echo "  make test             Run unit tests"
	@echo "  make test-coverage    Run tests with coverage report"
	@echo "  make lint             Run code linter (flake8)"
	@echo "  make format           Format code with black"
	@echo ""
	@echo "Utilities:"
	@echo "  make docs             Generate API documentation"
	@echo "  make clean            Clean generated files and cache"
	@echo "  make install-dev      Install with dev dependencies (testing, linting)"

# Installation
install:
	pip install --upgrade pip
	pip install -r requirements.txt
	@echo "✓ Dependencies installed successfully"

install-dev: install
	pip install pytest pytest-cov flake8 black sphinx
	@echo "✓ Development dependencies installed"

dev-setup: install-dev
	pre-commit install 2>/dev/null || echo "Note: Install pre-commit for automated checks"
	@echo "✓ Development environment setup complete"

# Execution
run-full:
	python main.py --config config/config.yaml --run-full --verbose

run-data:
	python main.py --stage data-fetch --verbose

run-features:
	python main.py --stage feature-engineering --verbose

run-train:
	python main.py --stage train-model --verbose

run-backtest:
	python main.py --stage backtest --walk-forward --verbose

run-analysis:
	python main.py --stage analysis --generate-report --verbose

# Testing
test:
	pytest tests/ -v --tb=short

test-coverage:
	pytest tests/ --cov=src --cov-report=html --cov-report=term
	@echo "✓ Coverage report generated in htmlcov/index.html"

# Code Quality
lint:
	flake8 src/ --max-line-length=100 --ignore=E501,W503
	@echo "✓ Linting complete"

format:
	black src/ tests/ main.py --line-length=100
	@echo "✓ Code formatted"

# Documentation
docs:
	cd docs && make html
	@echo "✓ Documentation built in docs/_build/html"

# Cleaning
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .coverage htmlcov build dist
	@echo "✓ Cache and temporary files cleaned"

clean-data:
	rm -rf data/processed/* data/results/*
	@echo "✓ Processed data and results cleared (keeping raw data)"

clean-all: clean clean-data
	rm -rf data/raw/*
	@echo "✓ All data cleared - full reset"

# Quick aliases
t: test
c: clean
f: format
l: lint
