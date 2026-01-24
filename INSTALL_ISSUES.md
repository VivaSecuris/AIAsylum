# Installation Issues and Solutions

## Python 3.14 Compatibility Issues

### pydantic-core Build Failure

**Error:**
```
TypeError: ForwardRef._evaluate() missing 1 required keyword-only argument: 'recursive_guard'
ERROR: Failed building wheel for pydantic-core
```

**Cause:**
Python 3.14 is very new and `pydantic-core` doesn't have pre-built wheels yet. When it tries to build from source, it fails due to Python 3.14 API changes.

**Solution:**
**You MUST use Python 3.11 or 3.12.** Python 3.14 is not supported yet.

```bash
# Install Python 3.11
brew install python@3.11

# Create new virtual environment
python3.11 -m venv venv311
source venv311/bin/activate

# Install dependencies
pip install -r requirements-core.txt
```

### Why Python 3.14 Doesn't Work

1. **No Pre-built Wheels**: Most packages don't have wheels for Python 3.14
2. **Build Failures**: Source builds fail due to API changes
3. **Dependency Chain**: `pydantic-core` → `pydantic` → `fastapi` → entire stack fails

### Recommended Python Versions

| Version | Status | Recommendation |
|---------|--------|----------------|
| Python 3.10 | ✅ Fully supported | Good choice |
| Python 3.11 | ✅ Fully supported | **Recommended** |
| Python 3.12 | ✅ Fully supported | **Recommended** |
| Python 3.13 | ⚠️ Mostly works | Some packages may have issues |
| Python 3.14 | ❌ Not supported | **Do not use** |

## Other Common Issues

### psycopg2-binary Build Failure

**Error:**
```
Error: pg_config executable not found
```

**Solution:**
- Use SQLite instead (default, no setup needed)
- Or install PostgreSQL: `brew install postgresql` (macOS)

### pandas Build Failure

**Error:**
```
error: metadata-generation-failed
```

**Solution:**
- pandas is optional (commented out in requirements.txt)
- Use `requirements-core.txt` which doesn't include pandas
- Or use Python 3.11/3.12 where pandas has wheels

## Quick Fix Summary

**If you're on Python 3.14:**

1. **Switch to Python 3.11** (recommended):
   ```bash
   brew install python@3.11
   python3.11 -m venv venv
   source venv/bin/activate
   pip install -r requirements-core.txt
   ```

2. **Or use Docker** (handles Python version automatically):
   ```bash
   docker-compose build
   docker-compose up -d
   ```

3. **Or use pyenv** to manage Python versions:
   ```bash
   pyenv install 3.11.9
   pyenv local 3.11.9
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements-core.txt
   ```

## Verification

After switching Python versions:

```bash
python --version  # Should show 3.10, 3.11, or 3.12
python scripts/quick_test.py  # Should pass
python integration.py  # Should pass
```

## Getting Help

If you're still having issues:

1. Check Python version: `python3 --version`
2. If it's 3.14, switch to 3.11 or 3.12
3. Verify virtual environment: `which python`
4. Try Docker if local setup is problematic
5. Check [PYTHON_VERSION.md](PYTHON_VERSION.md) for more details
