# Python Version Compatibility

## Recommended Python Versions

- **Python 3.10** ✅ Fully supported (recommended)
- **Python 3.11** ✅ Fully supported (recommended)
- **Python 3.12** ✅ Fully supported
- **Python 3.13** ⚠️ Mostly supported (some packages may have issues)
- **Python 3.14** ⚠️ Experimental (many packages not yet compatible)

## Current Status

You're running **Python 3.14.2**, which is very new. Some packages may not have wheels built yet and will try to compile from source, which can fail.

**Known Issues:**
- `pydantic-core` fails to build from source on Python 3.14 (missing `recursive_guard` argument)
- Many packages don't have pre-built wheels for Python 3.14 yet
- Compilation from source often fails due to compatibility issues

## Solutions

### Option 1: Use Python 3.11 or 3.12 (Recommended)

```bash
# Install Python 3.11 via Homebrew (macOS)
brew install python@3.11

# Create new virtual environment with Python 3.11
python3.11 -m venv venv311
source venv311/bin/activate

# Install dependencies
pip install -r requirements-core.txt
```

### Option 2: Use pyenv to Manage Python Versions

```bash
# Install pyenv
brew install pyenv

# Install Python 3.11
pyenv install 3.11.9

# Set local Python version
pyenv local 3.11.9

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements-core.txt
```

### Option 3: Try Latest Package Versions (May Not Work)

If you want to stick with Python 3.14, try installing latest versions:

```bash
# Upgrade pip first
pip install --upgrade pip setuptools wheel

# Try installing pydantic from source (may still fail)
pip install --no-binary pydantic-core pydantic pydantic-settings

# If that fails, pydantic is not compatible with Python 3.14 yet
# You MUST use Python 3.11 or 3.12
```

**Note**: As of now, `pydantic-core` does NOT work with Python 3.14. You must use Python 3.11 or 3.12.

### Option 4: Use Docker (No Python Version Issues)

```bash
# Docker handles Python version automatically
docker-compose build
docker-compose up -d
```

## Checking Your Python Version

```bash
python3 --version
# or
python --version
```

## Why This Matters

- **Python 3.10-3.12**: All packages have pre-built wheels
- **Python 3.13**: Most packages work, some may need compilation
- **Python 3.14**: Very new, many packages don't have wheels yet

## Quick Fix

The fastest solution is to use Python 3.11:

```bash
# macOS with Homebrew
brew install python@3.11
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements-core.txt
```

## Verification

After switching Python versions:

```bash
python --version  # Should show 3.10, 3.11, or 3.12
python scripts/quick_test.py  # Should pass
```
