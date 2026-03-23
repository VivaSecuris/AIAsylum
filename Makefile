.PHONY: install install-dev init migrate migrate-create run-api run-frontend test test-coverage test-cli lint format backup restore clean prefetch-benchmarks

# Installation
install:
	python3 -m pip install -r requirements.txt

# Warm Hugging Face cache for all registered benchmarks (one row each; requires datasets + network)
prefetch-benchmarks:
	python3 -m vivasecuris.aiasylum.benchmarks.prefetch

install-dev:
	python3 -m pip install -r requirements.txt -r requirements-dev.txt
	pre-commit install

# Database
init:
	alembic upgrade head

migrate:
	alembic upgrade head

migrate-create:
	@if [ -z "$(MESSAGE)" ]; then \
		echo "Usage: make migrate-create MESSAGE='description'"; \
		exit 1; \
	fi
	alembic revision --autogenerate -m "$(MESSAGE)"

# Running
run-api:
	uvicorn vivasecuris.aiasylum.api.main:app --reload --host 0.0.0.0 --port 8000

run-frontend:
	cd frontend && npm run dev

# Testing
test:
	pytest tests/ -v

test-coverage:
	pytest tests/ --cov=vivasecuris --cov-report=html --cov-report=term

test-cli:
	python scripts/test_cli.py

test-cli:
	python scripts/test_cli.py

# Linting & Formatting
lint:
	flake8 vivasecuris tests scripts
	mypy vivasecuris

format:
	black vivasecuris tests scripts
	isort vivasecuris tests scripts

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
