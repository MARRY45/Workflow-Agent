.PHONY: install lint format typecheck test cov check demo clean

PY ?= python

install:
	$(PY) -m pip install -e ".[dev]"

lint:
	ruff check src tests examples
	ruff format --check src tests examples

format:
	ruff check --fix src tests examples
	ruff format src tests examples

typecheck:
	mypy src

test:
	pytest

cov:
	pytest --cov --cov-report=term-missing

check: lint typecheck cov

demo:
	workflow-agent demo

clean:
	rm -rf build dist .mypy_cache .ruff_cache .pytest_cache .coverage htmlcov
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
