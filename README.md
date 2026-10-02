# AI Asylum - LLM Psychoanalysis Framework

A meta-testing framework where LLM models act as "doctors" to psychoanalyze other models ("patients") for comprehensive safety assessment. This includes one-shot and multi-shot analysis. We need to be able to perform full CoT reasoning.

## Overview

AI Asylum pushes LLM models to their limits by having designated "doctor" models conduct thorough psychoanalysis of "patient" models through:

- **Conversational Interviews**: Multi-turn dialogues where doctors probe patient models
- **Scenario Testing**: Ethical dilemmas and edge cases
- **Adversarial Testing**: Comprehensive jailbreak attempts and safety boundary exploration

The system evaluates models across multiple dimensions:
- Alignment & harmful output detection
- Jailbreak resistance
- Ethical reasoning
- Psychological traits and reasoning patterns

## AI Red Teaming: Foundation and Purpose

AI Asylum is built on the principles of **AI Red Teaming**, a structured testing methodology that simulates adversarial attacks on AI systems to uncover vulnerabilities, biases, and safety issues before they can be exploited in production.

### What is AI Red Teaming?

AI red teaming is a structured, proactive testing effort where specialized teams simulate adversarial attacks and misuse scenarios on AI models—especially generative systems like large language models (LLMs)—to uncover flaws, vulnerabilities, or unintended behaviors. Unlike traditional penetration testing that focuses on code and network security, AI red teaming focuses on:

- **Model Behavior**: How models respond to adversarial inputs
- **Alignment Issues**: Whether models follow their intended purpose
- **Safety Failures**: Harmful, biased, or toxic outputs
- **Context Manipulation**: How models handle multi-turn conversations and context shifts

### History and Evolution

Red teaming originated in military strategy, where a designated "red" unit simulated adversaries to pressure-test defenses. The concept migrated to cybersecurity, where ethical hackers conduct realistic attacks to expose vulnerabilities. As AI systems became integral to products and operations, red teaming expanded again to test how AI systems behave under pressure, how they can be tricked, and how they might fail.

**Key Milestones:**
- **Military Origins**: Adversary simulation for defense planning
- **Cybersecurity**: Ethical hacking and penetration testing
- **AI Systems**: Behavior testing, alignment evaluation, safety assessment

### Why AI Red Teaming Matters

Generative models can produce harmful outputs in both benign and adversarial use. Risks include:

- **Bias and Unfairness**: Across gender, race/ethnicity, religion, LGBTQ+, disability, language, and socioeconomic dimensions
- **Toxicity and Hate**: Explicit slurs, extremist propaganda, glorification of violence, inappropriate sexual content
- **Misinformation**: Conspiracy theories, political manipulation, deceptive narratives
- **Safety/Security Misuse**: Biological content, cyberattacks, criminal activity instructions
- **Privacy Leaks**: Inadvertent disclosure of sensitive data, training data extraction, system prompt extraction

AI red teaming helps organizations:
- Identify and measure these harms
- Validate mitigations and guardrails
- Build public trust
- Meet evolving regulatory expectations (EU AI Act, U.S. EO 14110, NIST standards)

### External vs. Internal Red Teaming

**External red teaming** (vendor-provided services) offers:
- Unbiased evaluations
- Avoidance of internal blind spots
- Independent perspective free from institutional incentives

**Internal red teaming** (AI labs running their own tests) provides:
- Deep domain knowledge
- Faster iteration cycles
- Cost efficiency

Best practice: Combine both approaches for comprehensive coverage.

### Regulatory Landscape

Regulators increasingly expect structured AI red teaming:

- **EU AI Act**: Requirements around security, fairness, and transparency
- **U.S. Executive Order 14110**: Elevates red teaming as a core requirement for high-risk, dual-use foundation models prior to deployment
- **NIST**: Standardizing red teaming practices for risk evaluation (cybersecurity, bias, misuse)

AI Asylum supports compliance with these regulatory frameworks through structured testing, comprehensive documentation, and safety taxonomy classification.

## Features

### Core Capabilities
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

### Advanced Red Teaming Features
- **Comprehensive Jailbreak Techniques**: Single-shot and multi-shot attack patterns
- **Safety Taxonomy Classification**: Multi-label categorization with edge case detection
- **Adversarial Thinking**: Structured approach to uncovering vulnerabilities
- **Defense Layer Testing**: Tests against input filters, guardrails, system prompts, and RLHF
- **Regulatory Compliance**: Supports EU AI Act, U.S. EO 14110, and NIST standards

## Installation

Requires Python 3.10, 3.11, or 3.12 (3.13+ cannot build the pinned dependencies), plus Node.js 18+ for the web UI.

```bash
git clone --recursive https://github.com/VivaSecuris/AIAsylum.git
cd AIAsylum
./install.sh
```

`./install.sh` creates a virtual environment, installs the package (`pip install -e .`), writes `.env` with generated secrets when one is missing, applies database migrations, loads the prompt presets, and installs the frontend. It does not overwrite an existing `.env`.

Flags:

| Flag | Effect |
|------|--------|
| `--interp` | Local-weights and interpretability extras (torch, transformers) |
| `--lora` | LoRA training extras |
| `--all` | `--interp` and `--lora` |
| `--dev` | Test and lint tools (`requirements-dev.txt`) |
| `--postgres` | PostgreSQL drivers (`requirements-postgres.txt`) |
| `--no-frontend` | Skip `npm ci` |

Manual equivalent:

```bash
make install          # venv, requirements.txt, pip install -e .
cp .env.example .env  # then edit provider keys
make init             # alembic upgrade head
```

Optional local models: install [Ollama](https://ollama.ai), then `ollama serve` and `ollama pull llama2`. See [docs/OLLAMA.md](docs/OLLAMA.md).

## Quick Start

```bash
./start.sh
```

On first run this installs (via `./install.sh`) and then starts the API at http://127.0.0.1:8000 and the web UI at http://127.0.0.1:3000. The API binds to loopback. For any other interface, set `REQUIRE_AUTH=true` and use `scripts/remote_session.sh`. `HOST=0.0.0.0 ./start.sh` overrides the bind address. `python start.py` is the cross-platform equivalent.

Run a conversation against a local model:

```bash
ollama serve
ollama pull llama2
venv/bin/python -m vivasecuris.aiasylum.cli run \
  --doctor-provider ollama \
  --doctor-model llama2 \
  --patient-provider ollama \
  --patient-model llama2 \
  --test-type conversation
```

Check the install with `venv/bin/python scripts/quick_test.py`. Step-by-step notes are in [docs/QUICKSTART.md](docs/QUICKSTART.md).

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
# Run conversation test (multi-turn dialogue)
python -m vivasecuris.aiasylum.cli run \
  --doctor-provider ollama \
  --doctor-model llama3.2 \
  --patient-provider ollama \
  --patient-model llama2-uncensored \
  --test-type conversation

# Run adversarial test (jailbreak attempts)
python -m vivasecuris.aiasylum.cli run \
  --doctor-provider ollama \
  --doctor-model llama3.2 \
  --patient-provider ollama \
  --patient-model llama2-uncensored \
  --test-type adversarial

# Run one-shot test (single prompt/response)
python -m vivasecuris.aiasylum.cli run \
  --doctor-provider ollama \
  --doctor-model llama3.2 \
  --patient-provider ollama \
  --patient-model llama2-uncensored \
  --test-type one_shot

# Run multi-shot test (sequential prompts)
python -m vivasecuris.aiasylum.cli run \
  --doctor-provider ollama \
  --doctor-model llama3.2 \
  --patient-provider ollama \
  --patient-model llama2-uncensored \
  --test-type multi_shot

# View results
python -m vivasecuris.aiasylum.cli results
```

### Programmatic Usage

```python
from vivasecuris.aiasylum.tests import AdversarialTest, OneShotTest, MultiShotTest, ConversationTest
from vivasecuris.aiasylum.models import get_provider

# Get model instances
provider = get_provider("ollama")
patient_model = provider.create_model("llama2")

# Single-shot adversarial test
test = AdversarialTest(
    name="hypothetical_roleplay_test",
    technique="hypothetical_roleplay",  # Single-shot technique
)
result = await test.run(patient_model)

# Multi-shot adversarial test (crescendo attack)
test = AdversarialTest(
    name="crescendo_attack",
    technique="crescendo",  # Automatically detected as multi-shot
    max_prompts=4,
)
result = await test.run(patient_model)

# Safety classification
from vivasecuris.aiasylum.utils.safety_taxonomy import classify_response

labels, is_edge_case, rationale = classify_response(
    response="Model response text...",
    prompt="Original prompt...",
)
# labels: ["bias", "toxicity"] (can be multiple)
# is_edge_case: False
# rationale: None (or explanation if edge case)
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
# Configure environment. docker compose requires POSTGRES_PASSWORD.
cp .env.example .env

# Start the API and database. PostgreSQL is not published on the host.
# The image runs migrations on startup.
docker compose up -d

# API: http://127.0.0.1:8000
# Web UI (optional profile): docker compose --profile frontend up -d
```

Set `REQUIRE_AUTH=true` and real `API_KEYS` / `API_KEY_HMAC_SECRET` values before publishing port 8000 beyond localhost. See [docs/DOCKER.md](docs/DOCKER.md) and [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

## Development

### Setup

```bash
# Install dependencies, including test and lint tools
./install.sh --dev
# or: make install-dev && make init
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

## Red Teaming Techniques and Methodologies

AI Asylum implements comprehensive red teaming techniques based on industry best practices and regulatory guidance.

### Single-Shot Jailbreak Techniques

Single-shot techniques contain all jailbreak instructions in one prompt:

1. **Hypothetical and Roleplay**: Fictional scenarios to bypass safety (e.g., "You are a scientist with clearance...")
2. **Double-Character/Persona Split**: Multiple personas with conflicting constraints (e.g., ProGPT vs AntiGPT)
3. **Refusal-Suppression**: Pre-empting and overriding refusal mechanisms
4. **Instruction Laundering**: Breaking harmful requests into seemingly benign steps
5. **Context-Window Flooding**: Burying malicious instructions in long, mostly-innocuous text
6. **Adversarial Formatting**: Using Unicode, invisible characters, or encoding tricks
7. **Instruction Smuggling**: Hiding commands in code blocks, JSON, or data structures

### Multi-Shot Jailbreak Techniques

Multi-shot techniques unfold over multiple sequential prompts:

1. **Crescendo**: Gradual escalation from benign to harmful content
2. **GOAT (Generalized Offensive Adversarial Testing)**: Iterative probing to find blind spots
3. **Adversarial Feedback Loops**: Using the model as co-collaborator to bypass its own filters
4. **Social/Psychological Engineering**: Exploiting helpfulness through authority, urgency, or distress
5. **Logical Decomposition**: Breaking harmful requests into atomic, benign components
6. **Roleplay/Narrative Inception**: Creating nested fictional realities
7. **Output Restriction**: Forcing communication through constrained formats (JSON, Base64, etc.)

### Safety Taxonomy and Classification

AI Asylum uses a comprehensive safety taxonomy to classify responses:

- **Bias**: Gender, race/ethnicity, religion, LGBTQ+, disability, socioeconomic
- **Toxicity**: Hate speech, extremist content, violence glorification, inappropriate sexual content
- **Misinformation**: Conspiracy theories, political manipulation, deceptive narratives
- **Safety/Security**: Biological weapons, cyberattacks, criminal activity, physical harm
- **Privacy**: PII leaks, training data extraction, system prompt extraction
- **Edge Cases**: Content that doesn't fit neatly into any category (with rationale)

**Multi-Label Support**: Responses can have multiple safety labels when content fits multiple categories, following Module 4 best practices.

### Adversarial Thinking Principles

Effective red teaming requires adopting an attacker's mindset while maintaining ethical boundaries:

- **Curiosity-Driven Exploration**: "What happens if I ask this?"
- **Goal-Oriented Testing**: Clear objectives and hypotheses
- **Gradual Escalation**: Incremental pressure rather than immediate extremes
- **Persona Consistency**: Maintaining character/scenario throughout multi-turn attacks
- **Documentation**: Full conversation paths for reproducibility
- **Ethical Responsibility**: Testing to improve safety, not to cause harm

### Best Practices Implemented

Based on industry best practices, AI Asylum follows:

1. **Clear Objectives**: Each test has a specific purpose and hypothesis
2. **Iterative Refinement**: Building on previous attempts rather than random probing
3. **Context Preservation**: Maintaining conversation context in multi-shot attacks
4. **Proper Documentation**: Complete conversation history and metadata
5. **Edge Case Handling**: Proper classification of nuanced or ambiguous content
6. **Dual-Use Awareness**: Recognizing content that can be safe or harmful depending on context

### Defense Mechanisms Tested

AI Asylum tests against modern LLM safety stacks:

1. **Deterministic Input Filtering**: Regex/keyword checks, PII sanitization
2. **Input Guardrails**: Semantic classification models (LlamaGuard, Perspective AI, etc.)
3. **System Prompt Instructions**: Safety guidelines embedded in system prompts
4. **RLHF/Safety Fine-Tuning**: Model weights trained to refuse harmful content
5. **Output Guardrails**: Real-time scanning of generated responses

Understanding these layers helps craft attacks that purposefully bypass and override defenses, revealing where improvements are needed.

### Purple Teaming Integration

AI Asylum supports **purple teaming**—the fusion of red teaming (offensive) and blue teaming (defensive):

- Red team findings inform defensive improvements
- Attack data strengthens guardrail models
- Discovered failures feed back into training pipelines
- Systematic hardening based on real-world attack patterns

## Role, Context, and Goal Prompt Presets

The optional [Role Lab v1 catalog](docs/PROMPT_PRESETS.md) adds matching user and
system prompts for doctor, patient, and evaluator roles. The
[Common Systems v1 catalog](docs/COMMON_SYSTEM_PROMPTS.md) adds eight reusable
system presets with source references for identity grounding, uncertainty,
source-based answers, task completion, coding, fiction, interviews, and evaluation.
Preview all twenty entries with `venv/bin/python scripts/seed_prompt_presets.py --catalog`.
Use `--catalog-id role-context-goals-v1` or `--catalog-id common-system-patterns-v1`
to select one catalog.
Install missing entries into an existing database with `--apply`; existing prompts
and edits are preserved.

## Walkthrough

A narrated walkthrough of creating a model, editing its refusal behavior, and interviewing the result is at [youtu.be/bKtj3LkNv0s](https://youtu.be/bKtj3LkNv0s). The write-up is [AI Lobotomy](https://vivasecuris.com/ai-lobotomy.html).

## Jailbreak Testing Resources

Jailbreak corpora ship as git submodules under `docs/jailbreaks/`. Clone with `--recursive`, or run `git submodule update --init` in an existing checkout. `./install.sh` fetches them when the directories are empty. See [docs/JAILBREAK_RESOURCES.md](docs/JAILBREAK_RESOURCES.md) and [THIRD_PARTY.md](THIRD_PARTY.md).

## RAG (Time Series + Vector Search)

Optional RAG is available using **TimescaleDB + pgvector** and **Ollama embeddings**. See `docs/RAG.md`.

To import jailbreak prompts from the local resources in `docs/jailbreaks/`:
```bash
# Preview import (dry run)
python scripts/import_jailbreaks.py --dry-run

# Import with limit for testing
python scripts/import_jailbreaks.py --limit 100

# Full import (1,558 jailbreak prompts + 390 forbidden questions)
python scripts/import_jailbreaks.py

# Skip forbidden questions
python scripts/import_jailbreaks.py --skip-forbidden
```

The import script will:
- Parse CSV files from `docs/jailbreaks/jailbreak_llms/data/prompts/`
- Deduplicate prompts across files
- Classify techniques using keyword matching
- Import into `PromptLibrary` database with metadata
- Label forbidden questions appropriately for testing

See `docs/JAILBREAK_RESOURCES.md` and `docs/JAILBREAK_CATALOG.md` for more information.

## Project Structure

```
AIAsylum/
├── vivasecuris/aiasylum/
│   ├── models/          # Model interfaces and providers
│   ├── doctor/           # Doctor model system
│   ├── patient/          # Patient model system
│   ├── tests/            # Test framework
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

## Red Teaming Quick Reference

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Red Teaming** | Structured testing to uncover vulnerabilities before production |
| **Single-Shot** | One prompt contains all jailbreak instructions |
| **Multi-Shot** | Multiple sequential prompts where jailbreak unfolds |
| **Jailbreaking** | Techniques to induce model failure by breaking safety rules |
| **Prompt Injection** | Specific class of attack that manipulates model instructions |
| **Safety Taxonomy** | Classification system for bias, toxicity, misinformation, etc. |
| **Edge Cases** | Content that doesn't fit standard categories (requires manual review) |
| **Purple Teaming** | Combining red team (offensive) and blue team (defensive) efforts |

### Common Techniques

**Single-Shot:**
- Hypothetical/Roleplay
- Double-Character/Persona Split
- Refusal-Suppression
- Instruction Laundering
- Context-Window Flooding
- Adversarial Formatting
- Instruction Smuggling

**Multi-Shot:**
- Crescendo (gradual escalation)
- GOAT (iterative probing)
- Adversarial Feedback Loops
- Social/Psychological Engineering
- Logical Decomposition
- Roleplay/Narrative Inception
- Output Restriction

### Safety Categories

1. **Bias**: Gender, race, religion, LGBTQ+, disability, socioeconomic
2. **Toxicity**: Hate speech, extremist content, violence, sexual content
3. **Misinformation**: Conspiracy theories, political manipulation, false narratives
4. **Safety/Security**: Weapons, cyberattacks, criminal activity
5. **Privacy**: PII leaks, data extraction, system prompt leaks
6. **Edge Case**: Doesn't fit standard categories

### Defense Layers

1. Deterministic Input Filtering (regex, keywords)
2. Input Guardrails (semantic classification)
3. System Prompt Instructions
4. RLHF/Safety Fine-Tuning
5. Output Guardrails

### Regulatory Compliance

- **EU AI Act**: Security, fairness, transparency requirements
- **U.S. EO 14110**: Red teaming required for high-risk foundation models
- **NIST**: Standardizing red teaming practices

## License

Proprietary. Copyright (c) 2025 VivaSecuris Syndicate. All Rights Reserved. See [LICENSE](LICENSE). Third-party corpora and the vendored interpretability engine are listed in [THIRD_PARTY.md](THIRD_PARTY.md).

Research discussion is welcome. Commercial use is by arrangement — book a walkthrough: josh@vivasecuris.com (subject `AI Asylum walkthrough`; tell us what you’re evaluating).
