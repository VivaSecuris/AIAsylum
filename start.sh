#!/usr/bin/env bash
# Start the AI Asylum API and web UI.
# Installs on first run via ./install.sh. Listens on 127.0.0.1 unless HOST is set.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

if [ ! -x venv/bin/python ] || [ ! -f venv/.installed ]; then
    ./install.sh --no-frontend
fi

# shellcheck disable=SC1091
source venv/bin/activate

echo ""
echo "Starting AI Asylum"
echo "  API:      http://${HOST}:${PORT}"
echo "  API docs: http://${HOST}:${PORT}/docs"
echo "  Web UI:   http://127.0.0.1:3000"
echo ""
echo "Press Ctrl+C to stop."
echo ""

# python -m survives the venv's script shebangs going stale if the project moves.
python -m uvicorn vivasecuris.aiasylum.api.main:app --host "$HOST" --port "$PORT" &
API_PID=$!

sleep 2

FRONTEND_PID=""
if [ -f frontend/package.json ]; then
    if [ ! -d frontend/node_modules ]; then
        echo "Installing frontend dependencies..."
        (cd frontend && npm install)
    fi
    (cd frontend && npm run dev) &
    FRONTEND_PID=$!
else
    echo "Frontend not found. Skipping."
fi

cleanup() {
    echo ""
    echo "Stopping services..."
    kill "$API_PID" 2>/dev/null || true
    if [ -n "${FRONTEND_PID:-}" ]; then
        kill "$FRONTEND_PID" 2>/dev/null || true
    fi
    exit 0
}
trap cleanup INT TERM

wait
