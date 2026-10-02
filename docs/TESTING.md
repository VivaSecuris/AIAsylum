# Testing Guide

This guide explains how to test AI Asylum to ensure everything works correctly.

## Quick Test (Recommended First Step)

The quickest way to test your setup:

```bash
# Run the quick test script (no API keys needed)
python scripts/quick_test.py
```

This tests:
- Module imports
- Database models
- Configuration loading
- Test framework
- Provider registry

## Comprehensive Test Script

For a full system check:

```bash
# Run comprehensive test script
./scripts/test_setup.sh
```

This script tests:
- Python version
- Virtual environment
- Dependencies
- Project structure
- Configuration files
- Environment setup
- Database initialization
- Python imports
- Docker configuration
- Ansible configuration
- Unit tests (if pytest is installed)

## Running Unit Tests

### Install Test Dependencies

```bash
# Activate virtual environment
source venv/bin/activate

# Install test dependencies
pip install -r requirements-dev.txt
```

### Run All Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=vivasecuris --cov-report=html

# Run specific test file
pytest tests/test_models.py -v

# Run specific test
pytest tests/test_models.py::TestModelProviders::test_get_provider_ollama -v
```

### Test Categories

- **Unit Tests**: Fast, isolated tests (`tests/test_*.py`)
- **Integration Tests**: End-to-end workflow tests (`tests/test_integration.py`)

## Manual Testing

### Test API Server

```bash
# Start API server
make run-api

# In another terminal, test endpoints
curl http://localhost:8000/health
curl http://localhost:8000/api/v1/test-runs/
```

### Test CLI

```bash
# Activate virtual environment
source venv/bin/activate

# Test CLI commands
python -m vivasecuris.aiasylum.cli list-benchmarks
python -m vivasecuris.aiasylum.cli results
```

### Test with Ollama (No API Keys Needed)

```bash
# Start Ollama (if not running)
ollama serve

# Pull a model
ollama pull llama2

# Run a test
python -m vivasecuris.aiasylum.cli run \
  --doctor-provider ollama \
  --doctor-model llama2 \
  --patient-provider ollama \
  --patient-model llama2 \
  --test-type conversation
```

## Docker Testing

### Test Docker Build

```bash
# Build image
docker-compose build

# Test container
docker-compose up -d
docker-compose logs -f

# Test API
curl http://localhost:8000/health

# Clean up
docker-compose down
```

## Continuous Integration

For CI/CD pipelines, use:

```bash
# Install dependencies
pip install -r requirements.txt -r requirements-dev.txt

# Run tests
pytest tests/ -v --cov=vivasecuris --cov-report=xml

# Run quick test
python scripts/quick_test.py
```

## Test Coverage Goals

- **Unit Tests**: >80% coverage
- **Integration Tests**: Cover main workflows
- **Critical Paths**: 100% coverage (authentication, database operations)

## Troubleshooting Tests

### Import Errors

If you see import errors:
```bash
# Ensure you're in the project root
cd /path/to/asylum

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Database Errors

If database tests fail:
```bash
# Initialize database
make init

# Or manually
alembic upgrade head
```

### Missing Dependencies

If tests fail due to missing packages:
```bash
# Install all dependencies
pip install -r requirements.txt -r requirements-dev.txt
```

## Writing New Tests

### Unit Test Example

```python
# tests/test_example.py
import pytest
from vivasecuris.aiasylum.example import ExampleClass

def test_example_function():
    """Test example function."""
    result = ExampleClass.example_function()
    assert result == expected_value
```

### Integration Test Example

```python
# tests/test_integration.py
@pytest.mark.asyncio
async def test_full_workflow():
    """Test complete workflow."""
    # Setup
    # Execute
    # Assert
    pass
```

## Best Practices

1. **Test Isolation**: Each test should be independent
2. **Use Fixtures**: Share setup code via pytest fixtures
3. **Mock External Services**: Don't call real APIs in unit tests
4. **Clear Test Names**: Test names should describe what they test
5. **Fast Tests**: Unit tests should run quickly
6. **Documentation**: Add docstrings to test functions

## Test Data

Test data is stored in:
- `tests/fixtures/` - Test data files
- In-memory databases for unit tests
- Mock objects for external services
