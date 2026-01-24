#!/bin/bash

# AI Asylum Startup Script
# Checks/creates venv, installs deps, starts API and frontend

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🚀 Starting AI Asylum..."

# Check for virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

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
