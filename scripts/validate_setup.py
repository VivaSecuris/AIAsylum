#!/usr/bin/env python3
"""Validate setup and environment."""

import os
import sys
from pathlib import Path


def check_python_version():
    """Check Python version."""
    if sys.version_info < (3, 10):
        print("❌ Python 3.10+ required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")
    return True


def check_dependencies():
    """Check if dependencies are installed."""
    required = [
        "fastapi",
        "sqlalchemy",
        "alembic",
        "openai",
        "anthropic",
        "click",
    ]
    
    missing = []
    for dep in required:
        try:
            __import__(dep.replace("-", "_"))
            print(f"✅ {dep}")
        except ImportError:
            print(f"❌ {dep} not installed")
            missing.append(dep)
    
    return len(missing) == 0


def check_env_file():
    """Check if .env file exists."""
    env_file = Path(".env")
    if env_file.exists():
        print("✅ .env file exists")
        return True
    else:
        print("⚠️  .env file not found (copy from .env.example)")
        return False


def check_api_keys():
    """Check if at least one API key is set."""
    keys = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
    }
    
    has_key = any(keys.values())
    if has_key:
        print("✅ At least one API key is set")
    else:
        print("⚠️  No API keys found (or use Ollama for local models)")
    
    return True  # Not required if using Ollama


def check_database():
    """Check database setup."""
    data_dir = Path("data")
    db_file = data_dir / "aiasylum.db"
    
    if db_file.exists():
        print("✅ Database file exists")
        return True
    else:
        print("⚠️  Database not initialized (run 'make init')")
        return False


def main():
    """Run all checks."""
    print("🔍 Validating AI Asylum setup...\n")
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Environment File", check_env_file),
        ("API Keys", check_api_keys),
        ("Database", check_database),
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\n{name}:")
        results.append(check_func())
    
    print("\n" + "=" * 50)
    if all(results):
        print("✅ All checks passed!")
        return 0
    else:
        print("⚠️  Some checks failed. Review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
