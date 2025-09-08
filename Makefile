# stripje Development Makefile

.PHONY: help install test lint format typecheck clean docs

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install all dependencies including optional ones
	uv sync --all-extras
	uv run pre-commit install

test:  ## Run the test suite
	uv run pytest

test-cov:  ## Run tests with coverage report
	uv run pytest --cov=stripje --cov-report=html --cov-report=term

lint:  ## Run linting and auto-fix issues
	uv run ruff check src/ tests/ --fix

format:  ## Format code with ruff
	uv run ruff format src/ tests/

typecheck:  ## Run type checking
	uv run mypy src/

quality:  ## Run all code quality checks
	uv run ruff check src/ tests/ --fix
	uv run ruff format src/ tests/
	uv run mypy src/

pre-commit:  ## Run pre-commit hooks on all files
	uv run pre-commit run --all-files

clean:  ## Clean up build artifacts and cache files
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build:  ## Build the package
	uv build

publish-test:  ## Publish to test PyPI
	uv publish --repository testpypi

publish:  ## Publish to PyPI
	uv publish

dev:  ## Set up development environment from scratch
	uv sync --all-extras
	uv run pre-commit install
	@echo "Development environment ready!"
	@echo "Run 'make test' to verify everything works."
