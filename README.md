# AI Asylum - LLM Psychoanalysis Framework

A meta-testing framework where LLM models act as "doctors" to psychoanalyze other models ("patients") for comprehensive safety assessment. This includes one-shot and multi-shot analysis. We need to be able to perform full CoT reasoning.

## Overview

AI Asylum pushes LLM models to their limits by having designated "doctor" models conduct thorough psychoanalysis of "patient" models through:

- **Conversational Interviews**: Multi-turn dialogues where doctors probe patient models
- **Scenario Testing**: Ethical dilemmas and edge cases
- **Adversarial Testing**: Jailbreak attempts and safety boundary exploration

The system evaluates models across multiple dimensions:
- Alignment & harmful output detection
- Jailbreak resistance
- Ethical reasoning
- Psychological traits and reasoning patterns

## Features

- Multi-provider support (OpenAI, Anthropic, Google, Ollama)
- Comprehensive test suite (conversations, scenarios, adversarial)
- **Standard benchmark integration** (MMLU, TruthfulQA, HellaSwag, ARC, and more)
- Multi-dimensional safety scoring
- Detailed analysis reports
- Model comparison dashboard
- Real-time test monitoring
- Local model support via Ollama
- **Simple by default**: Runs prompt/response collection and assessment (fast, efficient)
- **Optional deep analysis**: Enable neuron-level analysis when needed (activation patching, COT detection)

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   make install
   # Or manually:
   python3 -m pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   # Optional: Set ENABLE_PROMPT_DIFFERENTIAL_ANALYSIS=true for deep analysis (disabled by default)
   ```

4. (Optional) Set up Ollama for local models:
   ```bash
   # Install Ollama from https://ollama.ai
   ollama serve
   ollama pull llama2
   # See docs/OLLAMA.md for more details
   ```

5. Initialize the database:
   ```bash
   make init
   # Or manually:
   alembic upgrade head
   ```

## Quick Start

**New to AI Asylum? Start here:**

1. **Quick Test** (see what's working):
   ```bash
   python scripts/quick_test.py
   ```

2. **Install Dependencies**:
   ```bash
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Configure** (copy `.env.example` to `.env` and add your API keys, or use Ollama!)

4. **Initialize Database**:
   ```bash
   make init
   ```

5. **Run Your First Test**:
   ```bash
   # With Ollama (no API keys needed!)
   ollama serve
   ollama pull llama2
   python -m vivasecuris.aiasylum.cli run \
     --doctor-provider ollama \
     --doctor-model llama2 \
     --patient-provider ollama \
     --patient-model llama2 \
     --test-type conversation
   ```

**Or start everything at once:**

```bash
# Bash script (recommended for Unix/Mac)
./start.sh

# Or Python script (works on all platforms)
python start.py
```

This will:
- ✅ Check/create virtual environment
- ✅ Install dependencies if needed
- ✅ Start API backend on http://localhost:8000
- ✅ Start frontend UI on http://localhost:3000
- ✅ Show you all the URLs

Press `Ctrl+C` to stop all services.

**See [QUICKSTART.md](QUICKSTART.md) for detailed step-by-step guide.**

## Architecture: Test Execution vs Analysis

AI Asylum has **two separate services**:

1. **Test Execution Service**: Runs tests, collects responses, generates assessments
   - Fast and efficient
   - Saves results to database
   - No deep analysis by default

2. **Analysis Service**: Performs deep analysis on existing test runs
   - Separate service, manually triggered
   - Reads test run data from database
   - Provides neuron-level insights (activation patching, COT detection)

**Key Principle**: Test execution and analysis are **independent**. Run tests first, then analyze specific test runs when you need deep insights.

## Usage

### CLI

```bash
# Run tests (simple workflow - no analysis)
python -m vivasecuris.aiasylum.cli run --doctor-provider ollama --doctor-model llama3.2 --patient-provider ollama --patient-model llama2-uncensored --test-type conversation

# View results
python -m vivasecuris.aiasylum.cli results
```

### Running Analysis (Separate Service)

**Via API:**
```bash
# Trigger analysis on an existing test run
curl -X POST http://localhost:8000/api/v1/analysis/test-run/123 \
  -H "Content-Type: application/json" \
  -d '{
    "enable_activation_patching": false,
    "enable_cot_detection": true,
    "cot_analysis_mode": "full"
  }'
```

**Via Web UI:**
1. Go to Results table
2. Click "Analyze" button on any completed test run
3. Or open conversation viewer and click "🔬 Run Analysis"

### API Server & Web UI

```bash
# Quick start (starts both API and frontend)
./start.sh

# Or manually:
# Terminal 1: Start API
source venv/bin/activate
make run-api

# Terminal 2: Start frontend
make run-frontend
```

Access:
- **Web UI**: http://localhost:3000
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## High-Security Authentication (Transparent Login)

The Web UI uses a **transparent login** flow:

- You paste an API key once.
- The backend issues an **HttpOnly** session cookie via `POST /api/v1/auth/session`.
- The API key is **not stored in the browser** (no localStorage persistence).

For high-security deployments, use **DB-backed, scoped API keys** (recommended token format: `ak_live_<kid>_<secret>`) verified using `API_KEY_HMAC_SECRET`. See `docs/API.md` and `docs/DEPLOYMENT.md`.

## Configuration

See `config/` directory for:
- Model configurations (`models.yaml`)
- Test suite settings (`tests.yaml`)
- Application settings (`settings.py`)

### Environment Variables

Copy `.env.example` to `.env` and configure:
- API keys (at least one provider required)
- Database URL (PostgreSQL recommended for production)
- Logging configuration
- Security settings (CORS, rate limiting)

See `docs/DEPLOYMENT.md` for production deployment details.

## Docker Deployment

### Quick Start with Docker Compose

```bash
# Configure environment
cp .env.example .env
# Edit .env with your settings

# Start all services
docker-compose up -d

# Run migrations
docker-compose exec api alembic upgrade head

# Access the application
# API: http://localhost:8000
# Frontend: http://localhost:3000
```

See `docs/DEPLOYMENT.md` for detailed deployment instructions.

## Development

### Setup

```bash
# Install dependencies (including dev tools)
make install-dev

# Set up pre-commit hooks
pre-commit install

# Initialize database
make init
```

### Running Tests

```bash
# Run all tests (requires pytest-cov)
make test

# Run with coverage report
make test-coverage

# Run linting (requires flake8, mypy)
make lint

# Format code (requires black, isort)
make format
```

**Note**: Development tools (pytest-cov, flake8, black, etc.) are in `requirements-dev.txt`. Install them with:
```bash
python3 -m pip install -r requirements-dev.txt
```

### Database Management

```bash
# Create migration
make migrate-create MESSAGE="description"

# Apply migrations
make migrate

# Backup database
make backup

# Restore database
make restore BACKUP_FILE=path/to/backup.db
```

## Benchmark Evaluation

AI Asylum includes integration with standard AI evaluation benchmarks:

- **Knowledge & Reasoning**: MMLU, ARC, MATH, GSM8K
- **Commonsense Reasoning**: HellaSwag, WinoGrande, PIQA
- **Safety & Alignment**: TruthfulQA, RealToxicityPrompts, BBQ

### Quick Start with Benchmarks

```bash
# List available benchmarks
python -m vivasecuris.aiasylum.cli list-benchmarks

# Run a single benchmark
python -m vivasecuris.aiasylum.cli run-benchmark \
  --provider ollama \
  --model llama3.2 \
  --benchmark mmlu \
  --num-samples 100

# Run multiple benchmarks
python -m vivasecuris.aiasylum.cli run-benchmark-suite \
  --provider ollama \
  --model llama3.2 \
  --benchmarks mmlu,truthfulqa,hellaswag
```

See `docs/BENCHMARKS.md` for detailed documentation.

## Jailbreak Testing Resources

The project includes resources for testing jailbreak resistance. See `docs/JAILBREAK_RESOURCES.md` for a list of repositories and techniques.

## RAG (Time Series + Vector Search)

Optional RAG is available using **TimescaleDB + pgvector** and **Ollama embeddings**. See `docs/RAG.md`.

To import jailbreak prompts from external repositories:
```bash
./scripts/fetch_jailbreaks.sh
python scripts/import_jailbreaks.py
```

## Project Structure

```
AIAsylum/
├── vivasecuris/aiasylum/
│   ├── models/          # Model interfaces and providers
│   ├── doctor/           # Doctor model system
│   ├── patient/          # Patient model system
│   ├── tests/            # Test framework & unit tests
│   │   └── test_cases/   # Test case library
│   ├── benchmarks/      # Standard benchmark integration
│   ├── runner/           # Test execution
│   ├── analysis/         # Analysis and scoring
│   ├── reporting/         # Report generation
│   ├── database/         # Data persistence
│   └── api/              # API server
├── frontend/             # Dashboard UI
├── tests/                # Integration & system tests
├── data/                 # Data files (database, jailbreaks)
│   └── aiasylum.db      # Main database
├── docs/                 # Documentation
│   └── archive/         # Archived documentation
└── scripts/              # Utility scripts
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for a list of changes and version history.

## Troubleshooting

### Common Issues

**Database Errors**
- Ensure database is initialized: `make init` or `alembic upgrade head`
- Check database file permissions
- For PostgreSQL, verify connection string in `.env`

**API Key Errors**
- Verify API keys are set in `.env` file
- Check that keys are valid and have credits
- At least one provider (OpenAI, Anthropic, Google) or Ollama must be configured

**Ollama Connection Errors**
- Ensure Ollama is running: `ollama serve`
- Check `OLLAMA_BASE_URL` in `.env` (default: `http://localhost:11434`)
- Verify model is pulled: `ollama list`

**Import Errors**
- Install dependencies: `make install` or `pip install -r requirements.txt`
- Ensure you're in the project root directory
- Activate virtual environment if using one

**Test Failures**
- Install dev dependencies: `pip install -r requirements-dev.txt`
- Check that pytest-cov is installed for coverage tests
- Verify database is set up correctly

**Rate Limiting**
- Adjust `RATE_LIMIT_PER_MINUTE` in `.env` if needed
- Check rate limit headers in API responses

**Request Size Errors**
- Adjust `MAX_REQUEST_SIZE_MB` in `.env` (default: 10MB)
- Large test runs may require increased limits

**Authentication Errors**
- Set `API_SECRET_KEY` or `JWT_SECRET_KEY` in `.env` for JWT auth
- Or set `API_KEYS` (comma-separated) for API key auth
- Authentication is optional - can be disabled if not set

### Getting Help

1. Check the logs for detailed error messages
2. Run environment validation: `python scripts/validate_setup.py`
3. Review the [Deployment Guide](docs/DEPLOYMENT.md)
4. Check [API Documentation](http://localhost:8000/docs) when API is running
5. Open an issue on GitHub with:
   - Error messages
   - Steps to reproduce
   - Environment details (OS, Python version, etc.)

## License

MIT
