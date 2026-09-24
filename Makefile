.PHONY: venv install install-dev init migrate migrate-create run-api run-frontend test test-coverage test-cli lint format backup restore clean prefetch-benchmarks

# Everything runs through the venv's interpreter, so no activation is needed.
# PYTHON_BOOTSTRAP only creates the venv; it must be 3.10-3.12 (see PYTHON_VERSION.md).
PYTHON_BOOTSTRAP ?= python3.11
PY ?= venv/bin/python

# Installation
venv:
	@if [ ! -x $(PY) ]; then $(PYTHON_BOOTSTRAP) -m venv venv; fi

install: venv
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e .

# Warm Hugging Face cache for all registered benchmarks (one row each; requires datasets + network)
prefetch-benchmarks:
	$(PY) -m vivasecuris.aiasylum.benchmarks.prefetch

install-dev: venv
	$(PY) -m pip install -r requirements.txt -r requirements-dev.txt
	$(PY) -m pip install -e .
	@if [ -f .pre-commit-config.yaml ]; then $(PY) -m pre_commit install; fi

# Database
init:
	$(PY) -m alembic upgrade head

migrate:
	$(PY) -m alembic upgrade head

migrate-create:
	@if [ -z "$(MESSAGE)" ]; then \
		echo "Usage: make migrate-create MESSAGE='description'"; \
		exit 1; \
	fi
	$(PY) -m alembic revision --autogenerate -m "$(MESSAGE)"

# Running
run-api:
	$(PY) -m uvicorn vivasecuris.aiasylum.api.main:app --reload --host 0.0.0.0 --port 8000

run-frontend:
	cd frontend && npm run dev

# Testing
test:
	$(PY) -m pytest tests/ -v

test-coverage:
	$(PY) -m pytest tests/ --cov=vivasecuris --cov-report=html --cov-report=term

test-cli:
	$(PY) scripts/test_cli.py

# Linting & Formatting
lint:
	$(PY) -m flake8 vivasecuris tests scripts
	$(PY) -m mypy vivasecuris

format:
	$(PY) -m black vivasecuris tests scripts
	$(PY) -m isort vivasecuris tests scripts

# Database Management
backup:
	@if [ -f data/aiasylum.db ]; then \
		cp data/aiasylum.db data/aiasylum.db.backup.$$(date +%Y%m%d_%H%M%S); \
		echo "Backup created"; \
	else \
		echo "Database not found"; \
	fi

restore:
	@if [ -z "$(BACKUP_FILE)" ]; then \
		echo "Usage: make restore BACKUP_FILE=path/to/backup.db"; \
		exit 1; \
	fi
	cp $(BACKUP_FILE) data/aiasylum.db
	echo "Database restored from $(BACKUP_FILE)"

# Cleanup
clean:
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
