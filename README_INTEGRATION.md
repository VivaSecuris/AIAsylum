# Integration Testing

## Quick Run

```bash
# Run all integration tests
python integration.py
```

This script will:
- ✅ Test all imports
- ✅ Test database models
- ✅ Test database initialization
- ✅ Test configuration
- ✅ Test model providers
- ✅ Test test framework
- ✅ Run pytest unit tests
- ✅ Test CLI commands
- ✅ Check API server (if running)

## What It Tests

### 1. Imports
Verifies all core modules can be imported:
- Models, Doctor, Patient, Runner
- Test framework classes
- Database models
- Analysis service
- Configuration

### 2. Database
- Model definitions
- Database initialization
- Connection settings

### 3. Configuration
- Settings loading
- Environment variables
- Default values

### 4. Model Providers
- Provider registry
- Provider initialization
- API key validation (warns if missing)

### 5. Test Framework
- Test class definitions
- Test initialization
- Test categories

### 6. Unit Tests
Runs pytest test suite:
- Model tests
- Doctor/Patient tests
- Test framework tests
- Database tests
- Integration tests

### 7. CLI Commands
Tests command-line interface:
- Command availability
- Command execution
- Output validation

### 8. API Server
Checks if API is running:
- Health endpoint
- Server availability
- Connection test

## Output

The script provides colored output:
- 🟢 **Green (✓)**: Test passed
- 🔴 **Red (✗)**: Test failed
- 🟡 **Yellow (⚠)**: Warning (not critical)

## Exit Codes

- **0**: All critical tests passed
- **1**: Some tests failed

## Examples

### Successful Run
```
✓ All critical tests passed!
AI Asylum is ready to use.
```

### With Warnings
```
⚠ Provider: openai (API key not set)
⚠ API server not running (start with: make run-api)
✓ All critical tests passed!
```

### With Failures
```
✗ Import: vivasecuris.aiasylum.models (ModuleNotFoundError)
✗ Some tests failed. Please review the errors above.
```

## Integration with CI/CD

Use in CI pipelines:

```bash
# Run integration tests
python integration.py

# Exit code indicates success/failure
if [ $? -eq 0 ]; then
    echo "All tests passed"
else
    echo "Tests failed"
    exit 1
fi
```

## Troubleshooting

### Import Errors
```bash
# Install dependencies
pip install -r requirements-core.txt
```

### Database Errors
```bash
# Initialize database
make init
```

### Pytest Not Found
```bash
# Install pytest
pip install pytest pytest-asyncio pytest-mock
```

### CLI Errors
```bash
# Make sure you're in project root
cd /path/to/asylum
python integration.py
```

## Running Specific Tests

The script runs all tests by default. To test specific areas:

```python
# Edit integration.py and comment out unwanted tests
# Or run pytest directly for unit tests only:
pytest tests/ -v
```
