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


def run_command(cmd, cwd=None, check=True):
    """Run a shell command."""
    print(f"▶️  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, check=check)
    return result.returncode == 0


def main():
    print("🚀 Starting AI Asylum...")
    
    venv_path = SCRIPT_DIR / "venv"
    venv_python = venv_path / "bin" / "python" if sys.platform != "win32" else venv_path / "Scripts" / "python.exe"
    
    # Create virtual environment if needed
    if not venv_path.exists():
        print("📦 Creating virtual environment...")
        run_command([sys.executable, "-m", "venv", str(venv_path)])
    
    # Install dependencies if needed
    installed_marker = venv_path / ".installed"
    if not installed_marker.exists():
        print("📥 Installing dependencies...")
        pip_cmd = str(venv_python)
        run_command([pip_cmd, "install", "--upgrade", "pip"])
        run_command([pip_cmd, "install", "-r", "requirements.txt"])
        installed_marker.touch()
    
    # Check for .env file
    env_file = SCRIPT_DIR / ".env"
    if not env_file.exists():
        print("⚠️  .env file not found. Copying from .env.example...")
        env_example = SCRIPT_DIR / ".env.example"
        if env_example.exists():
            env_file.write_text(env_example.read_text())
            print("📝 Please edit .env with your API keys before continuing")
    
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
