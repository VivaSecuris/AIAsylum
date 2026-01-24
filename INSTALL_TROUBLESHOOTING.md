# Installation Troubleshooting

Common installation issues and solutions.

## psycopg2-binary Installation Error

### Error Message
```
Error: pg_config executable not found.
pg_config is required to build psycopg2 from source.
```

### Solution

**Option 1: Use SQLite (Recommended for Development)**

SQLite works out of the box and doesn't require any system dependencies:

```bash
# Just install the base requirements (psycopg2 is optional)
pip install -r requirements.txt
```

The default configuration uses SQLite, so you don't need PostgreSQL for development.

**Option 2: Install PostgreSQL (For Production)**

If you need PostgreSQL:

**macOS:**
```bash
# Install PostgreSQL via Homebrew
brew install postgresql

# Then install psycopg2
pip install psycopg2-binary
```

**Ubuntu/Debian:**
```bash
# Install PostgreSQL development libraries
sudo apt-get update
sudo apt-get install libpq-dev postgresql-dev

# Then install psycopg2
pip install psycopg2-binary
```

**Windows:**
1. Download and install PostgreSQL from https://www.postgresql.org/download/windows/
2. Make sure PostgreSQL's `bin` directory is in your PATH
3. Then install: `pip install psycopg2-binary`

**Option 3: Skip PostgreSQL Dependencies**

If you're only using SQLite (which is fine for development):

```bash
# Install without PostgreSQL dependencies
pip install -r requirements.txt

# The installation will skip psycopg2-binary (it's commented out)
```

## Other Common Issues

### Python Version

**Error:** `Python 3.10+ required`

**Solution:**
```bash
# Check your Python version
python3 --version

# If < 3.10, install a newer version
# macOS: brew install python@3.11
# Or use pyenv: pyenv install 3.11
```

### Virtual Environment Issues

**Error:** `ModuleNotFoundError` or `command not found`

**Solution:**
```bash
# Make sure virtual environment is activated
source venv/bin/activate  # Mac/Linux
# or
venv\Scripts\activate     # Windows

# Verify you're using the venv Python
which python  # Should point to venv/bin/python
```

### Permission Errors

**Error:** `Permission denied` when installing

**Solution:**
```bash
# Don't use sudo with pip in virtual environments
# Make sure venv is activated first
source venv/bin/activate
pip install -r requirements.txt
```

### Missing System Dependencies

**Error:** Various build errors

**Solution (macOS):**
```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install Homebrew if not installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

**Solution (Ubuntu/Debian):**
```bash
# Install build essentials
sudo apt-get update
sudo apt-get install build-essential python3-dev
```

## Quick Fix: SQLite Only Setup

If you just want to get started quickly without PostgreSQL:

```bash
# 1. Activate virtual environment
source venv/bin/activate

# 2. Install base requirements (SQLite only)
pip install -r requirements.txt

# 3. Verify installation
python scripts/quick_test.py

# 4. You're ready! SQLite will be used automatically
```

## Verification

After installation, verify everything works:

```bash
# Quick test
python scripts/quick_test.py

# Comprehensive test
./scripts/test_setup.sh

# Or run unit tests
pytest tests/ -v
```

## Getting Help

If you're still having issues:

1. Check Python version: `python3 --version` (need 3.10+)
2. Verify venv is activated: `which python`
3. Check error messages carefully
4. Review this troubleshooting guide
5. Check [README.md](README.md) for more details
