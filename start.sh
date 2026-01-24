#!/bin/bash

# AI Asylum Startup Script
# Checks/creates venv, installs deps, starts API and frontend

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🚀 Starting AI Asylum..."

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d'.' -f1,2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
    echo "❌ Python 3.10+ required. Found Python $PYTHON_VERSION"
    exit 1
fi

if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 14 ]; then
    echo "⚠️  WARNING: Python 3.14+ detected. This version has compatibility issues."
    echo "   pydantic-core will fail to build. Please use Python 3.11 or 3.12."
    echo ""
    echo "   To switch:"
    echo "   brew install python@3.11"
    echo "   python3.11 -m venv venv"
    echo "   source venv/bin/activate"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Try to find Python 3.11 or 3.12 if available
if command -v python3.11 &> /dev/null; then
    PYTHON_CMD=python3.11
    echo "✓ Using Python 3.11 (recommended)"
elif command -v python3.12 &> /dev/null; then
    PYTHON_CMD=python3.12
    echo "✓ Using Python 3.12 (recommended)"
elif command -v python3.10 &> /dev/null; then
    PYTHON_CMD=python3.10
    echo "✓ Using Python 3.10"
else
    PYTHON_CMD=python3
    echo "⚠️  Using system Python ($PYTHON_VERSION)"
fi

# Check for virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment with $PYTHON_CMD..."
    $PYTHON_CMD -m venv venv
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Verify Python version in venv
VENV_PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}' | cut -d'.' -f1,2)
VENV_PYTHON_MAJOR=$(echo $VENV_PYTHON_VERSION | cut -d'.' -f1)
VENV_PYTHON_MINOR=$(echo $VENV_PYTHON_VERSION | cut -d'.' -f2)

if [ "$VENV_PYTHON_MAJOR" -eq 3 ] && [ "$VENV_PYTHON_MINOR" -ge 14 ]; then
    echo "⚠️  WARNING: Virtual environment uses Python $VENV_PYTHON_VERSION"
    echo "   This may cause pydantic-core build failures."
    echo "   Consider recreating venv with: python3.11 -m venv venv"
fi

# Install dependencies if needed
if [ ! -f "venv/.installed" ]; then
    echo "📥 Installing dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
    touch venv/.installed
fi

# Check for .env file
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found. Copying from .env.example..."
    cp .env.example .env
    echo "📝 Please edit .env with your API keys before continuing"
fi

# Initialize database if needed
if [ ! -f "data/aiasylum.db" ]; then
    echo "🗄️  Initializing database..."
    mkdir -p data
    alembic upgrade head || echo "⚠️  Database initialization skipped (run 'make init' manually)"
fi

# Start services
echo ""
echo "✅ Starting services..."
echo "   API: http://localhost:8000"
echo "   Frontend: http://localhost:3000"
echo "   API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Start API in background
uvicorn vivasecuris.aiasylum.api.main:app --host 0.0.0.0 --port 8000 &
API_PID=$!

# Wait a moment for API to start
sleep 2

# Start frontend if package.json exists
if [ -f "frontend/package.json" ]; then
    cd frontend
    if [ ! -d "node_modules" ]; then
        echo "📦 Installing frontend dependencies..."
        npm install
    fi
    npm run dev &
    FRONTEND_PID=$!
    cd ..
else
    echo "⚠️  Frontend not found. Skipping..."
    FRONTEND_PID=""
fi

# Wait for interrupt
trap "echo ''; echo '🛑 Stopping services...'; kill $API_PID 2>/dev/null; [ -n '$FRONTEND_PID' ] && kill $FRONTEND_PID 2>/dev/null; exit" INT TERM

# Wait for processes
wait
