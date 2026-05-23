# Makefile for Cambium

.PHONY: help install install-dev test lint format clean build publish-test publish

help:
	@echo "Cambium Development Commands"
	@echo "=============================="
	@echo "install       - Install the package"
	@echo "install-dev   - Install in development mode with all dependencies"
	@echo "test          - Run all tests"
	@echo "test-cov      - Run tests with coverage"
	@echo "lint          - Run linters (black, isort, mypy)"
	@echo "format        - Format code with black and isort"
	@echo "clean         - Clean build artifacts"
	@echo "build         - Build package distribution"
	@echo "check         - Check the distribution"
	@echo "publish-test  - Publish to TestPyPI"
	@echo "publish       - Publish to PyPI"
	@echo "docs          - Build documentation"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev,train]"
	pre-commit install

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=cambium --cov-report=html --cov-report=term

lint:
	black --check cambium/ tests/
	isort --check-only cambium/ tests/
	mypy cambium/

format:
	black cambium/ tests/
	isort cambium/ tests/

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf htmlcov
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

check:
	twine check dist/*

publish-test: build check
	twine upload --repository testpypi dist/*

publish: build check
	twine upload dist/*

docs:
	cd docs && sphinx-build -b html . _build/html

# Alternative if you have make available inside docs
docs-make:
	cd docs && $(MAKE) html

bump-patch:
	bumpver update --patch

bump-minor:
	bumpver update --minor

bump-major:
	bumpver update --major
