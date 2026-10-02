# Quick Start Guide

Get AI Asylum up and running in minutes!

## Step 1: Check Python Version

```bash
python3 --version
```

**Important**: Python 3.10, 3.11, or 3.12 is required. Python 3.13 and newer cannot build the pinned dependencies. See [INSTALL_TROUBLESHOOTING.md](INSTALL_TROUBLESHOOTING.md).

## Step 2: Verify Setup

```bash
# Run the quick test (no dependencies needed)
python scripts/quick_test.py
```

This will tell you what's missing. Don't worry if some tests fail - we'll fix that next.

## Step 3: Install Dependencies

```bash
# Activate virtual environment (if not already)
source venv/bin/activate  # On Mac/Linux
# or
venv\Scripts\activate     # On Windows

# Install core dependencies (recommended first)
pip install -r requirements-core.txt

# OR install all dependencies (some may have issues on Python 3.14+)
pip install -r requirements.txt
```

**Notes**: 
- `psycopg2-binary` is optional - SQLite works by default
- On Python 3.13 or newer, install 3.11 or 3.12 instead. See [INSTALL_TROUBLESHOOTING.md](INSTALL_TROUBLESHOOTING.md).
- See [INSTALL_TROUBLESHOOTING.md](INSTALL_TROUBLESHOOTING.md) for help

## Step 4: Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your settings. See docs/API_KEYS.md.
# At minimum, you need ONE of:
# - OPENAI_API_KEY=sk-...
# - ANTHROPIC_API_KEY=sk-ant-...
# - GOOGLE_API_KEY=...
# OR use Ollama (no API keys needed!)
```

## Step 5: Initialize Database

```bash
# Create data directory
mkdir -p data

# Run migrations
make init
# or manually:
alembic upgrade head
```

## Step 6: Test Everything

```bash
# Run comprehensive test
./scripts/test_setup.sh

# Or run quick test again (should pass now)
python scripts/quick_test.py

# Or run unit tests
pytest tests/ -v
```

## Step 7: Start Using AI Asylum

### Option A: Use Ollama (No API Keys!)

```bash
# Start Ollama (if not running)
ollama serve

# Pull a model
ollama pull llama2

# Run your first test!
python -m vivasecuris.aiasylum.cli run \
  --doctor-provider ollama \
  --doctor-model llama2 \
  --patient-provider ollama \
  --patient-model llama2 \
  --test-type conversation
```

### Option B: Use API Server

```bash
# Start API server
make run-api

# In another terminal, test it
curl http://localhost:8000/health

# View API docs
open http://localhost:8000/docs
```

### Option C: Use Docker

```bash
# Build and start
docker-compose up -d

# Check logs
docker-compose logs -f

# Test API
curl http://localhost:8000/health
```

## Common Issues

### "Module not found" errors
- Make sure virtual environment is activated
- Run `pip install -r requirements.txt`

### Database errors
- Run `make init` to initialize database
- Check that `data/` directory exists and is writable

### API key errors
- You don't need API keys if using Ollama!
- Set `OLLAMA_BASE_URL=http://localhost:11434` in `.env`
- Make sure Ollama is running: `ollama serve`

### Port already in use
- Change ports in `docker-compose.yml` or `.env`
- Kill process using port: `lsof -ti:8000 | xargs kill`

## Next Steps

- Read [README.md](../README.md) for full documentation
- Check [TESTING.md](TESTING.md) for the testing guide
- The rest of this directory is the user documentation

## Getting Help

1. Run `python scripts/validate_setup.py` to diagnose issues
2. Check logs for error messages
3. Review [Troubleshooting](README.md#troubleshooting) section
4. Open an issue with error details
