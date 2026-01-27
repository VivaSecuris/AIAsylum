# CLI Test Script

This directory contains test scripts for the AI Asylum CLI.

## test_cli.py

Comprehensive test script that validates all CLI commands and functionality.

### Usage

```bash
# Run all tests
python scripts/test_cli.py

# Or make it executable and run directly
chmod +x scripts/test_cli.py
./scripts/test_cli.py
```

### What It Tests

1. **Help Command** - Verifies the main help output and command listing
2. **List Benchmarks** - Tests the `list-benchmarks` command
3. **Results Command** - Tests the `results` command (may skip if database not initialized)
4. **Run Command Validation** - Tests that required arguments are validated
5. **Run Benchmark Validation** - Tests benchmark command validation
6. **Ollama Commands** - Tests Ollama subcommands (may skip if Ollama not running)
7. **Invalid Command Handling** - Tests error handling for invalid commands
8. **Command Help** - Tests help output for individual commands

### Test Results

The script provides color-coded output:
- ✓ Green: Tests that passed
- ✗ Red: Tests that failed
- ⚠ Yellow: Tests that were skipped or have warnings

### Notes

- Some tests may be skipped if dependencies (database, Ollama) are not configured
- The script does NOT run actual test runs (which would require API keys and models)
- The script focuses on CLI interface validation, not end-to-end functionality

### Exit Codes

- `0`: All critical tests passed
- `1`: Some tests failed

## Integration with CI/CD

You can integrate this into your CI/CD pipeline:

```yaml
# Example GitHub Actions
- name: Test CLI
  run: python scripts/test_cli.py
```

## Manual Testing

For manual testing of actual functionality, you can use:

```bash
# Test help
python -m vivasecuris.aiasylum.cli --help

# Test list benchmarks
python -m vivasecuris.aiasylum.cli list-benchmarks

# Test results (requires database)
python -m vivasecuris.aiasylum.cli results --limit 5

# Test Ollama commands (requires Ollama running)
python -m vivasecuris.aiasylum.cli ollama list
python -m vivasecuris.aiasylum.cli ollama check llama2
```
