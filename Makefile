PYTHON=python
PIP=pip

install:
	$(PIP) install -U pip
	$(PIP) install -e .[dev]
	pre-commit install

test:
	pytest -q --cov=src/addrnorm --cov-fail-under=85

format:
	black src tests
	isort src tests || true

lint:
	ruff src tests
	mypy src/addrnorm

run-api:
	uvicorn src.addrnorm.api.main:app --reload

build-docker:
	docker build -t addrnorm-api .
