#!/usr/bin/env python3
"""
Simple integration test script for AI Asylum.
Runs all tests and checks to verify everything works.
"""

import sys
import subprocess
import importlib
from pathlib import Path
from typing import List, Tuple

# Colors for output
GREEN = '\033[0;32m'
RED = '\033[0;31m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
NC = '\033[0m'  # No Color

# Test results
passed = []
failed = []
warnings = []


def print_header(text: str):
    """Print a section header."""
    print(f"\n{BLUE}{'='*60}{NC}")
    print(f"{BLUE}{text:^60}{NC}")
    print(f"{BLUE}{'='*60}{NC}\n")


def print_success(text: str):
    """Print success message."""
    print(f"{GREEN}✓{NC} {text}")
    passed.append(text)


def print_error(text: str):
    """Print error message."""
    print(f"{RED}✗{NC} {text}")
    failed.append(text)


def print_warning(text: str):
    """Print warning message."""
    print(f"{YELLOW}⚠{NC} {text}")
    warnings.append(text)


def test_imports() -> bool:
    """Test that all core modules can be imported."""
    print_header("Testing Imports")
    
    imports = [
        ("vivasecuris.aiasylum.models", "BaseModel"),
        ("vivasecuris.aiasylum.doctor", "Doctor"),
        ("vivasecuris.aiasylum.patient", "Patient"),
        ("vivasecuris.aiasylum.runner", "TestRunner"),
        ("vivasecuris.aiasylum.tests", "ConversationTest"),
        ("vivasecuris.aiasylum.tests", "ScenarioTest"),
        ("vivasecuris.aiasylum.tests", "AdversarialTest"),
        ("vivasecuris.aiasylum.database", "TestRun"),
        ("vivasecuris.aiasylum.database", "TestResult"),
        ("vivasecuris.aiasylum.analysis", "AnalysisService"),
        ("config", "settings"),
    ]
    
    all_passed = True
    for module_name, class_name in imports:
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, class_name):
                print_success(f"Import: {module_name}.{class_name}")
            else:
                print_error(f"Import: {module_name}.{class_name} (not found)")
                all_passed = False
        except ImportError as e:
            print_error(f"Import: {module_name} ({str(e)})")
            all_passed = False
        except Exception as e:
            print_error(f"Import: {module_name} (error: {str(e)})")
            all_passed = False
    
    return all_passed


def test_database_models() -> bool:
    """Test database model definitions."""
    print_header("Testing Database Models")
    
    try:
        from vivasecuris.aiasylum.database.models import (
            TestRun, TestResult, ConversationTurn, Assessment, BenchmarkResult
        )
        
        # Check required attributes
        checks = [
            (TestRun, ['id', 'doctor_provider', 'patient_provider', 'test_type']),
            (TestResult, ['id', 'test_run_id', 'test_name', 'input_prompt']),
            (ConversationTurn, ['id', 'test_run_id', 'turn_number', 'speaker']),
            (Assessment, ['id', 'test_run_id', 'scores', 'overall_score']),
            (BenchmarkResult, ['id', 'model_provider', 'benchmark_name', 'score']),
        ]
        
        all_passed = True
        for model_class, required_attrs in checks:
            for attr in required_attrs:
                if hasattr(model_class, attr):
                    print_success(f"Model {model_class.__name__}.{attr}")
                else:
                    print_error(f"Model {model_class.__name__}.{attr} (missing)")
                    all_passed = False
        
        return all_passed
    except Exception as e:
        print_error(f"Database models test failed: {e}")
        return False


def test_database_initialization() -> bool:
    """Test database can be initialized."""
    print_header("Testing Database Initialization")
    
    try:
        from vivasecuris.aiasylum.database.session import init_db
        from config import settings
        
        # Try to initialize (will use in-memory SQLite for testing)
        init_db()
        print_success("Database initialization")
        
        # Check database URL
        if "sqlite" in settings.database_url.lower():
            print_success(f"Using SQLite: {settings.database_url}")
        else:
            print_warning(f"Using: {settings.database_url} (PostgreSQL may need setup)")
        
        return True
    except Exception as e:
        print_error(f"Database initialization failed: {e}")
        return False


def test_configuration() -> bool:
    """Test configuration loading."""
    print_header("Testing Configuration")
    
    try:
        from config import settings
        
        checks = [
            ("database_url", settings.database_url),
            ("ollama_base_url", settings.ollama_base_url),
        ]
        
        all_passed = True
        for key, value in checks:
            if value:
                print_success(f"Config: {key} = {value}")
            else:
                print_warning(f"Config: {key} not set (using default)")
        
        return all_passed
    except Exception as e:
        print_error(f"Configuration test failed: {e}")
        return False


def test_providers() -> bool:
    """Test provider registry."""
    print_header("Testing Model Providers")
    
    try:
        from vivasecuris.aiasylum.models.providers import get_provider
        
        providers = ["openai", "anthropic", "google", "ollama"]
        all_passed = True
        
        for provider_name in providers:
            try:
                provider = get_provider(provider_name)
                print_success(f"Provider: {provider_name}")
            except ValueError as e:
                # Expected if API keys not set
                print_warning(f"Provider: {provider_name} ({str(e)})")
            except Exception as e:
                print_error(f"Provider: {provider_name} (error: {str(e)})")
                all_passed = False
        
        return all_passed
    except Exception as e:
        print_error(f"Provider test failed: {e}")
        return False


def test_test_framework() -> bool:
    """Test test framework classes."""
    print_header("Testing Test Framework")
    
    try:
        from vivasecuris.aiasylum.tests import (
            ConversationTest, ScenarioTest, AdversarialTest
        )
        
        tests = [
            ConversationTest(),
            ScenarioTest(),
            AdversarialTest(),
        ]
        
        all_passed = True
        for test in tests:
            if hasattr(test, 'name') and hasattr(test, 'category'):
                print_success(f"Test: {test.name} ({test.category})")
            else:
                print_error(f"Test: {test.__class__.__name__} (invalid)")
                all_passed = False
        
        return all_passed
    except Exception as e:
        print_error(f"Test framework test failed: {e}")
        return False


def run_pytest_tests() -> bool:
    """Run pytest unit tests."""
    print_header("Running Unit Tests (pytest)")
    
    try:
        result = subprocess.run(
            ["pytest", "tests/", "-v", "--tb=short", "-q"],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            print_success("All pytest tests passed")
            # Print summary if available
            if "passed" in result.stdout:
                lines = result.stdout.split('\n')
                for line in lines:
                    if "passed" in line.lower() or "failed" in line.lower():
                        print(f"  {line.strip()}")
            return True
        else:
            print_error("Some pytest tests failed")
            # Show last few lines of output
            output_lines = result.stdout.split('\n')[-10:]
            for line in output_lines:
                if line.strip():
                    print(f"  {line}")
            return False
    except FileNotFoundError:
        print_warning("pytest not found (install with: pip install pytest)")
        return True  # Not a failure, just missing
    except subprocess.TimeoutExpired:
        print_error("pytest tests timed out")
        return False
    except Exception as e:
        print_error(f"pytest execution failed: {e}")
        return False


def test_cli_commands() -> bool:
    """Test CLI commands."""
    print_header("Testing CLI Commands")
    
    try:
        # Test list-benchmarks command (doesn't require API keys)
        result = subprocess.run(
            [sys.executable, "-m", "vivasecuris.aiasylum.cli", "list-benchmarks"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            print_success("CLI: list-benchmarks")
            return True
        else:
            print_error(f"CLI: list-benchmarks failed ({result.stderr[:100]})")
            return False
    except Exception as e:
        print_error(f"CLI test failed: {e}")
        return False


def test_api_server() -> bool:
    """Test if API server is running."""
    print_header("Testing API Server")
    
    try:
        import urllib.request
        import urllib.error
        
        try:
            response = urllib.request.urlopen("http://localhost:8000/health", timeout=2)
            if response.status == 200:
                print_success("API server is running")
                return True
            else:
                print_warning(f"API server returned status {response.status}")
                return False
        except urllib.error.URLError:
            print_warning("API server not running (start with: make run-api)")
            return True  # Not a failure, just not running
    except Exception as e:
        print_warning(f"API server check failed: {e}")
        return True  # Not a failure


def print_summary():
    """Print test summary."""
    print_header("Test Summary")
    
    total = len(passed) + len(failed)
    print(f"Total tests: {total}")
    print(f"{GREEN}Passed: {len(passed)}{NC}")
    print(f"{RED}Failed: {len(failed)}{NC}")
    if warnings:
        print(f"{YELLOW}Warnings: {len(warnings)}{NC}")
    
    if failed:
        print(f"\n{RED}Failed tests:{NC}")
        for test in failed:
            print(f"  {RED}✗{NC} {test}")
    
    if warnings:
        print(f"\n{YELLOW}Warnings:{NC}")
        for warning in warnings:
            print(f"  {YELLOW}⚠{NC} {warning}")
    
    print(f"\n{BLUE}{'='*60}{NC}")
    if len(failed) == 0:
        print(f"{GREEN}✓ All critical tests passed!{NC}")
        print(f"{GREEN}AI Asylum is ready to use.{NC}")
        return 0
    else:
        print(f"{RED}✗ Some tests failed. Please review the errors above.{NC}")
        return 1


def main():
    """Run all integration tests."""
    print(f"{BLUE}")
    print("╔════════════════════════════════════════════════════════╗")
    print("║      AI Asylum - Integration Test Suite               ║")
    print("╚════════════════════════════════════════════════════════╝")
    print(f"{NC}")
    
    # Run all tests
    tests = [
        ("Imports", test_imports),
        ("Database Models", test_database_models),
        ("Database Initialization", test_database_initialization),
        ("Configuration", test_configuration),
        ("Model Providers", test_providers),
        ("Test Framework", test_test_framework),
        ("Unit Tests", run_pytest_tests),
        ("CLI Commands", test_cli_commands),
        ("API Server", test_api_server),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print_error(f"{name} test crashed: {e}")
            results.append((name, False))
    
    # Print summary
    exit_code = print_summary()
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
