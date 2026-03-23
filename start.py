#!/usr/bin/env python3
"""
AI Asylum Startup Script (Cross-platform)
Checks/creates venv, installs deps, starts API and frontend
"""

import os
import sys
import subprocess
import signal
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.absolute()


def check_python_version():
    """Check Python version and warn if incompatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print("❌ Python 3.10+ required. Found Python {}.{}".format(version.major, version.minor))
        sys.exit(1)
    
    if version.major == 3 and version.minor >= 14:
        print("⚠️  WARNING: Python 3.14+ detected. This version has compatibility issues.")
        print("   pydantic-core will fail to build. Please use Python 3.11 or 3.12.")
        print("")
        print("   To switch:")
        print("   brew install python@3.11")
        print("   python3.11 -m venv venv")
        print("   source venv/bin/activate")
        print("")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)


def find_best_python():
    """Find the best Python version available."""
    # Try Python 3.11 first (recommended)
    for version in ['3.11', '3.12', '3.10']:
        python_cmd = f'python{version}'
        try:
            result = subprocess.run(
                [python_cmd, '--version'],
                capture_output=True,
                check=True
            )
            print(f"✓ Using {python_cmd} (recommended)")
            return python_cmd
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    
    # Fall back to system python3
    print(f"⚠️  Using system Python ({sys.version_info.major}.{sys.version_info.minor})")
    return sys.executable


def run_command(cmd, cwd=None, check=True):
    """Run a shell command."""
    print(f"▶️  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, check=check)
    return result.returncode == 0


def main():
    print("🚀 Starting AI Asylum...")
    
    # Check Python version
    check_python_version()
    
    # Find best Python version
    python_cmd = find_best_python()
    
    venv_path = SCRIPT_DIR / "venv"
    venv_python = venv_path / "bin" / "python" if sys.platform != "win32" else venv_path / "Scripts" / "python.exe"
    
    # Create virtual environment if needed
    if not venv_path.exists():
        print(f"📦 Creating virtual environment with {python_cmd}...")
        run_command([python_cmd, "-m", "venv", str(venv_path)])
    else:
        # Verify Python version in existing venv
        try:
            result = subprocess.run(
                [str(venv_python), "--version"],
                capture_output=True,
                text=True,
                check=True
            )
            venv_version_str = result.stdout.strip()
            
            # Check if it's Python 3.14+
            if "3.14" in venv_version_str or "3.15" in venv_version_str:
                print("⚠️  WARNING: Existing virtual environment uses Python 3.14+")
                print("   This will cause pydantic-core build failures.")
                print("")
                response = input(f"Recreate venv with {python_cmd}? (Y/n): ")
                if response.lower() != 'n':
                    print("🗑️  Removing old virtual environment...")
                    import shutil
                    shutil.rmtree(venv_path)
                    print(f"📦 Creating new virtual environment with {python_cmd}...")
                    run_command([python_cmd, "-m", "venv", str(venv_path)])
                else:
                    print("⚠️  Continuing with Python 3.14 venv (installation may fail)")
        except Exception:
            pass
    
    # Install dependencies if needed
    installed_marker = venv_path / ".installed"
    if not installed_marker.exists():
        print("📥 Installing dependencies...")
        pip_cmd = str(venv_python)
        run_command([pip_cmd, "install", "--upgrade", "pip"])
        run_command([pip_cmd, "install", "-r", "requirements.txt"])
        installed_marker.touch()
        print("📥 Warming Hugging Face benchmark dataset caches (one row each; first time may take several minutes)...")
        run_command([pip_cmd, "-m", "vivasecuris.aiasylum.benchmarks.prefetch"], check=False)
    
    # Check for .env file
    env_file = SCRIPT_DIR / ".env"
    if not env_file.exists():
        env_example = SCRIPT_DIR / ".env.example"
        if env_example.exists():
            print("⚠️  .env file not found. Copying from .env.example...")
            env_file.write_text(env_example.read_text())
            print("📝 Please edit .env with your API keys before continuing")
        else:
            print("⚠️  .env file not found. Creating from template...")
            env_content = """# API Keys (at least one provider required, or use Ollama)
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GOOGLE_API_KEY=

# Ollama Configuration (no API keys needed)
OLLAMA_BASE_URL=http://localhost:11434

# Database
DATABASE_URL=sqlite:///./data/aiasylum.db

# Security
API_SECRET_KEY=change-me-in-production
JWT_SECRET_KEY=change-me-in-production
API_KEY_HMAC_SECRET=change-me-in-production
API_KEYS=

# CORS
CORS_ORIGINS=http://localhost:3000,http://localhost:8000

# Rate Limiting
RATE_LIMIT_PER_MINUTE=60

# Request Limits
MAX_REQUEST_SIZE_MB=10

# Concurrency Control
MAX_CONCURRENT_WORKERS=5

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# Deep Analysis (disabled by default)
ENABLE_PROMPT_DIFFERENTIAL_ANALYSIS=false
ENABLE_ACTIVATION_PATCHING=false
ENABLE_COT_DETECTION=false

# RAG (optional)
ENABLE_RAG=false
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
"""
            env_file.write_text(env_content)
            print("📝 Created .env file. Please edit it with your API keys (or use Ollama)")
    
    # Initialize database if needed
    data_dir = SCRIPT_DIR / "data"
    data_dir.mkdir(exist_ok=True)
    db_file = data_dir / "aiasylum.db"
    if not db_file.exists():
        print("🗄️  Initializing database...")
        try:
            run_command([str(venv_python), "-m", "alembic", "upgrade", "head"], check=False)
        except Exception:
            print("⚠️  Database initialization skipped (run 'make init' manually)")
    
    # Start services
    print("")
    print("✅ Starting services...")
    print("   API: http://localhost:8000")
    print("   Frontend: http://localhost:3000")
    print("   API Docs: http://localhost:8000/docs")
    print("")
    print("Press Ctrl+C to stop all services")
    print("")
    
    processes = []
    
    # Start API
    api_cmd = [str(venv_python), "-m", "uvicorn", "vivasecuris.aiasylum.api.main:app", 
               "--host", "0.0.0.0", "--port", "8000"]
    api_process = subprocess.Popen(api_cmd, cwd=SCRIPT_DIR)
    processes.append(api_process)
    time.sleep(2)
    
    # Start frontend if package.json exists
    frontend_dir = SCRIPT_DIR / "frontend"
    if (frontend_dir / "package.json").exists():
        if not (frontend_dir / "node_modules").exists():
            print("📦 Installing frontend dependencies...")
            run_command(["npm", "install"], cwd=frontend_dir)
        frontend_process = subprocess.Popen(["npm", "run", "dev"], cwd=frontend_dir)
        processes.append(frontend_process)
    else:
        print("⚠️  Frontend not found. Skipping...")
    
    # Handle shutdown
    def signal_handler(sig, frame):
        print("\n🛑 Stopping services...")
        for proc in processes:
            try:
                proc.terminate()
            except Exception:
                pass
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Wait for processes
    try:
        for proc in processes:
            proc.wait()
    except KeyboardInterrupt:
        signal_handler(None, None)


if __name__ == "__main__":
    main()
