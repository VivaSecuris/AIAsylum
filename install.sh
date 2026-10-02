#!/usr/bin/env bash
# AI Asylum installer (macOS / Linux).
# Idempotent: safe to re-run. Does not overwrite an existing .env.
#
#   ./install.sh                  core API, CLI, database, frontend
#   ./install.sh --interp         also local-weights / interpretability extras
#   ./install.sh --lora           also LoRA training extras
#   ./install.sh --all            interp + lora
#   ./install.sh --dev            also test and lint tools
#   ./install.sh --postgres       also PostgreSQL drivers
#   ./install.sh --no-frontend    skip npm ci

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

WITH_INTERP=0
WITH_LORA=0
WITH_DEV=0
WITH_POSTGRES=0
WITH_FRONTEND=1

usage() {
    sed -n '2,12p' "$0"
    exit 2
}

while [ $# -gt 0 ]; do
    case "$1" in
        --interp) WITH_INTERP=1 ;;
        --lora) WITH_LORA=1 ;;
        --all) WITH_INTERP=1; WITH_LORA=1 ;;
        --dev) WITH_DEV=1 ;;
        --postgres) WITH_POSTGRES=1 ;;
        --no-frontend) WITH_FRONTEND=0 ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1" >&2; usage ;;
    esac
    shift
done

python_in_range() {
    "$1" -c 'import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] < (3, 13) else 1)' >/dev/null 2>&1
}

pick_python() {
    local candidate
    for candidate in python3.11 python3.12 python3.10 python3; do
        if command -v "$candidate" >/dev/null 2>&1 && python_in_range "$candidate"; then
            command -v "$candidate"
            return 0
        fi
    done
    return 1
}

echo "AI Asylum install"
echo "  root: $ROOT"

if ! PYTHON_CMD="$(pick_python)"; then
    FOUND="$(python3 --version 2>/dev/null || echo 'python3 not found')"
    echo "Python 3.10, 3.11, or 3.12 is required (found: $FOUND)." >&2
    echo "3.13 and newer cannot build the pinned pydantic-core wheels." >&2
    echo "  macOS:  brew install python@3.11" >&2
    echo "  Debian: sudo apt-get install python3.11 python3.11-venv" >&2
    exit 1
fi
echo "  python: $PYTHON_CMD ($("$PYTHON_CMD" --version))"

if [ -d .git ]; then
    NEED_SUBMODULES=0
    for sub in docs/jailbreaks/Awesome-LLM-Jailbreak \
               docs/jailbreaks/LLM-Jailbreaks \
               docs/jailbreaks/Prompt-Hacking-Resources \
               docs/jailbreaks/jailbreak_llms; do
        if [ -d "$sub" ] && [ -z "$(ls -A "$sub" 2>/dev/null || true)" ]; then
            NEED_SUBMODULES=1
        fi
    done
    if [ "$NEED_SUBMODULES" -eq 1 ]; then
        if [ -f .gitmodules ]; then
            echo "Fetching jailbreak corpus submodules..."
            git submodule update --init --recursive
        else
            echo "Jailbreak corpus directories are empty and .gitmodules is missing." >&2
            echo "See docs/JAILBREAK_RESOURCES.md." >&2
        fi
    fi
fi

if [ ! -x venv/bin/python ] || ! python_in_range venv/bin/python; then
    if [ -d venv ]; then
        echo "Replacing venv (interpreter is missing or outside Python 3.10-3.12)."
        rm -rf venv
    else
        echo "Creating virtual environment..."
    fi
    "$PYTHON_CMD" -m venv venv
fi

venv/bin/python -m pip install --upgrade pip
venv/bin/python -m pip install -r requirements.txt

EXTRAS=()
[ "$WITH_INTERP" -eq 1 ] && EXTRAS+=("interp")
[ "$WITH_LORA" -eq 1 ] && EXTRAS+=("lora")
if [ ${#EXTRAS[@]} -gt 0 ]; then
    IFS=,
    EXTRA_SPEC="${EXTRAS[*]}"
    unset IFS
    echo "Installing extras: $EXTRA_SPEC"
    venv/bin/python -m pip install -e ".[$EXTRA_SPEC]"
else
    venv/bin/python -m pip install -e .
fi

if [ "$WITH_DEV" -eq 1 ]; then
    venv/bin/python -m pip install -r requirements-dev.txt
fi
if [ "$WITH_POSTGRES" -eq 1 ]; then
    venv/bin/python -m pip install -r requirements-postgres.txt
fi

if [ ! -f .env ]; then
    echo "Creating .env from .env.example (secrets generated; provider keys left blank)."
    cp .env.example .env
    gen() { "$PYTHON_CMD" -c 'import secrets; print(secrets.token_urlsafe(32))'; }
    # Replace only the known placeholders, one key at a time.
    python_replace() {
        local key="$1" value="$2"
        venv/bin/python - "$key" "$value" <<'PY'
import pathlib, sys
key, value = sys.argv[1], sys.argv[2]
path = pathlib.Path(".env")
lines = []
for line in path.read_text().splitlines():
    if line.startswith(key + "=") and "change-me-in-production" in line:
        line = f"{key}={value}"
    lines.append(line)
path.write_text("\n".join(lines) + "\n")
PY
    }
    python_replace API_SECRET_KEY "$(gen)"
    python_replace JWT_SECRET_KEY "$(gen)"
    python_replace API_KEY_HMAC_SECRET "$(gen)"
else
    echo ".env already exists; leaving it unchanged."
fi

mkdir -p data
echo "Applying database migrations..."
venv/bin/python -m alembic upgrade head
echo "Installing prompt presets (missing rows only)..."
venv/bin/python scripts/seed_prompt_presets.py --apply

if [ "$WITH_FRONTEND" -eq 1 ]; then
    if command -v node >/dev/null 2>&1 && command -v npm >/dev/null 2>&1 \
        && node -e 'process.exit(0 if int(process.versions.node.split(".")[0]) >= 18 else 1)'; then
        echo "Installing frontend dependencies (npm ci)..."
        (cd frontend && npm ci)
    else
        echo "Skipping frontend: Node.js 18+ and npm were not found."
        echo "  Install Node 18 or newer, then re-run ./install.sh"
    fi
fi

OLLAMA_URL="$(venv/bin/python - <<'PY'
from pathlib import Path
url = "http://localhost:11434"
env = Path(".env")
if env.exists():
    for line in env.read_text().splitlines():
        if line.startswith("OLLAMA_BASE_URL="):
            value = line.split("=", 1)[1].strip()
            if value:
                url = value
print(url)
PY
)"
if curl -fsS --max-time 2 "${OLLAMA_URL%/}/api/tags" >/dev/null 2>&1; then
    echo "Ollama is reachable at $OLLAMA_URL"
else
    echo "Ollama is not reachable at $OLLAMA_URL (optional; cloud keys in .env work without it)."
fi

touch venv/.installed

cat <<EOF

Install complete.

  Start the API and web UI:   ./start.sh
  CLI:                         venv/bin/aiasylum --help
  API (after start):           http://127.0.0.1:8000
  Web UI (after start):        http://127.0.0.1:3000
  API docs:                    http://127.0.0.1:8000/docs

Add provider keys to .env (OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY),
or run a local model with Ollama (see docs/OLLAMA.md).
The API listens on 127.0.0.1. For a non-loopback host, set REQUIRE_AUTH=true
and use scripts/remote_session.sh.
EOF
