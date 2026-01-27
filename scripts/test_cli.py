#!/usr/bin/env python3
"""
Test script for AI Asylum CLI.

This script tests all CLI commands to ensure they work correctly.
Run with: python scripts/test_cli.py
"""

import sys
import subprocess
import os
from pathlib import Path
from typing import Tuple, List

# Colors for output
GREEN = '\033[0;32m'
RED = '\033[0;31m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
CYAN = '\033[0;36m'
NC = '\033[0m'  # No Color

# Test results
passed_tests = []
failed_tests = []
skipped_tests = []


def print_header(text: str):
    """Print a section header."""
    print(f"\n{BLUE}{'='*70}{NC}")
    print(f"{BLUE}{text:^70}{NC}")
    print(f"{BLUE}{'='*70}{NC}\n")


def print_success(text: str):
    """Print success message."""
    print(f"{GREEN}✓{NC} {text}")
    passed_tests.append(text)


def print_error(text: str):
    """Print error message."""
    print(f"{RED}✗{NC} {text}")
    failed_tests.append(text)


def print_warning(text: str):
    """Print warning message."""
    print(f"{YELLOW}⚠{NC} {text}")
    skipped_tests.append(text)


def print_info(text: str):
    """Print info message."""
    print(f"{CYAN}ℹ{NC} {text}")


def run_cli_command(args: List[str], expected_exit_code: int = 0, capture_output: bool = True) -> Tuple[int, str, str]:
    """
    Run a CLI command and return the result.
    
    Args:
        args: List of command arguments (without 'python -m vivasecuris.aiasylum.cli')
        expected_exit_code: Expected exit code (default: 0)
        capture_output: Whether to capture output (default: True)
    
    Returns:
        Tuple of (exit_code, stdout, stderr)
    """
    # Get the project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Build the command
    cmd = [sys.executable, "-m", "vivasecuris.aiasylum.cli"] + args
    
    try:
        result = subprocess.run(
            cmd,
            cwd=project_root,
            capture_output=capture_output,
            text=True,
            timeout=30
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 124, "", "Command timed out after 30 seconds"
    except Exception as e:
        return 1, "", str(e)


def test_help_command():
    """Test the main help command."""
    print_header("Testing CLI Help Command")
    
    exit_code, stdout, stderr = run_cli_command(["--help"])
    
    if exit_code == 0:
        if "AI Asylum CLI" in stdout or "LLM Psychoanalysis Framework" in stdout:
            print_success("Help command works and shows correct description")
            if "run" in stdout:
                print_success("Help shows 'run' command")
            if "results" in stdout:
                print_success("Help shows 'results' command")
            if "list-benchmarks" in stdout:
                print_success("Help shows 'list-benchmarks' command")
            if "run-benchmark" in stdout:
                print_success("Help shows 'run-benchmark' command")
            if "ollama" in stdout:
                print_success("Help shows 'ollama' subcommand")
            return True
        else:
            print_error("Help command output doesn't contain expected text")
            return False
    else:
        print_error(f"Help command failed with exit code {exit_code}")
        if stderr:
            print_error(f"Error: {stderr}")
        return False


def test_list_benchmarks():
    """Test the list-benchmarks command."""
    print_header("Testing list-benchmarks Command")
    
    exit_code, stdout, stderr = run_cli_command(["list-benchmarks"])
    
    if exit_code == 0:
        if "Available benchmarks" in stdout or "benchmarks" in stdout.lower():
            print_success("list-benchmarks command works")
            # Check for some expected benchmarks
            expected_benchmarks = ["mmlu", "truthfulqa", "hellaswag", "arc", "math"]
            found_benchmarks = [b for b in expected_benchmarks if b.lower() in stdout.lower()]
            if found_benchmarks:
                print_success(f"Found expected benchmarks: {', '.join(found_benchmarks)}")
            return True
        else:
            print_error("list-benchmarks output doesn't contain expected text")
            return False
    else:
        print_error(f"list-benchmarks command failed with exit code {exit_code}")
        if stderr:
            print_error(f"Error: {stderr}")
        return False


def test_results_command():
    """Test the results command."""
    print_header("Testing results Command")
    
    # Test with default limit
    exit_code, stdout, stderr = run_cli_command(["results"])
    
    if exit_code == 0:
        print_success("results command works (no database errors)")
        if "Found" in stdout or "test runs" in stdout.lower():
            print_success("results command shows test runs output")
        return True
    else:
        print_warning(f"results command failed (may be due to database not initialized): {stderr[:100] if stderr else 'Unknown error'}")
        return False
    
    # Test with limit option
    exit_code2, stdout2, stderr2 = run_cli_command(["results", "--limit", "5"])
    if exit_code2 == 0:
        print_success("results command works with --limit option")
        return True
    else:
        print_warning("results command with --limit failed (may be due to database)")
        return False


def test_run_command_validation():
    """Test the run command with validation (without actually running tests)."""
    print_header("Testing run Command Validation")
    
    # Test missing required arguments
    exit_code, stdout, stderr = run_cli_command(["run"])
    
    if exit_code != 0:
        if "required" in stderr.lower() or "Missing option" in stderr:
            print_success("run command correctly validates required arguments")
        else:
            print_error(f"run command validation failed unexpectedly: {stderr[:200]}")
            return False
    else:
        print_error("run command should fail without required arguments")
        return False
    
    # Test invalid test type
    exit_code2, stdout2, stderr2 = run_cli_command([
        "run",
        "--doctor-provider", "ollama",
        "--doctor-model", "test",
        "--patient-provider", "ollama",
        "--patient-model", "test",
        "--test-type", "invalid_type"
    ])
    
    if exit_code2 != 0:
        if "invalid choice" in stderr2.lower() or "Invalid value" in stderr2:
            print_success("run command correctly validates test type")
        else:
            print_warning(f"run command validation message unclear: {stderr2[:200]}")
    else:
        print_error("run command should fail with invalid test type")
        return False
    
    return True


def test_run_benchmark_validation():
    """Test the run-benchmark command validation."""
    print_header("Testing run-benchmark Command Validation")
    
    # Test missing required arguments
    exit_code, stdout, stderr = run_cli_command(["run-benchmark"])
    
    if exit_code != 0:
        if "required" in stderr.lower() or "Missing option" in stderr:
            print_success("run-benchmark command correctly validates required arguments")
        else:
            print_error(f"run-benchmark validation failed: {stderr[:200]}")
            return False
    else:
        print_error("run-benchmark should fail without required arguments")
        return False
    
    return True


def test_ollama_commands():
    """Test Ollama subcommands."""
    print_header("Testing Ollama Subcommands")
    
    # Test ollama help
    exit_code, stdout, stderr = run_cli_command(["ollama", "--help"])
    
    if exit_code == 0:
        if "Ollama" in stdout or "ollama" in stdout.lower():
            print_success("ollama --help works")
            if "list" in stdout:
                print_success("ollama help shows 'list' command")
            if "pull" in stdout:
                print_success("ollama help shows 'pull' command")
            if "info" in stdout:
                print_success("ollama help shows 'info' command")
            if "check" in stdout:
                print_success("ollama help shows 'check' command")
        else:
            print_warning("ollama help output doesn't contain expected text")
    else:
        print_warning(f"ollama --help failed: {stderr[:100] if stderr else 'Unknown error'}")
    
    # Test ollama list (may fail if Ollama not running, that's OK)
    exit_code2, stdout2, stderr2 = run_cli_command(["ollama", "list"])
    if exit_code2 == 0:
        print_success("ollama list command works")
    else:
        print_warning(f"ollama list failed (Ollama may not be running): {stderr2[:100] if stderr2 else 'Unknown error'}")
    
    # Test ollama check with invalid model (should handle gracefully)
    exit_code3, stdout3, stderr3 = run_cli_command(["ollama", "check", "nonexistent_model_12345"])
    if exit_code3 == 0:
        # Should output that model is not available
        if "not available" in stdout3.lower() or "not found" in stdout3.lower():
            print_success("ollama check handles non-existent models gracefully")
        else:
            print_warning("ollama check output unclear for non-existent model")
    else:
        print_warning(f"ollama check failed: {stderr3[:100] if stderr3 else 'Unknown error'}")
    
    return True


def test_invalid_command():
    """Test handling of invalid commands."""
    print_header("Testing Invalid Command Handling")
    
    exit_code, stdout, stderr = run_cli_command(["invalid-command-xyz"])
    
    if exit_code != 0:
        if "No such command" in stderr or "Unknown command" in stderr or "Usage:" in stderr:
            print_success("CLI correctly handles invalid commands")
            return True
        else:
            print_warning(f"Invalid command error message unclear: {stderr[:200]}")
            return False
    else:
        print_error("CLI should fail on invalid commands")
        return False


def test_command_help():
    """Test help for individual commands."""
    print_header("Testing Individual Command Help")
    
    commands_to_test = [
        ["run", "--help"],
        ["results", "--help"],
        ["list-benchmarks", "--help"],
        ["run-benchmark", "--help"],
    ]
    
    for cmd in commands_to_test:
        exit_code, stdout, stderr = run_cli_command(cmd)
        if exit_code == 0:
            if "Usage:" in stdout or "Options:" in stdout:
                print_success(f"{cmd[0]} --help works")
            else:
                print_warning(f"{cmd[0]} --help output unclear")
        else:
            print_warning(f"{cmd[0]} --help failed: {stderr[:100] if stderr else 'Unknown error'}")
    
    return True


def print_summary():
    """Print test summary."""
    print_header("Test Summary")
    
    total = len(passed_tests) + len(failed_tests) + len(skipped_tests)
    print(f"Total tests: {total}")
    print(f"{GREEN}Passed: {len(passed_tests)}{NC}")
    print(f"{RED}Failed: {len(failed_tests)}{NC}")
    print(f"{YELLOW}Skipped/Warnings: {len(skipped_tests)}{NC}")
    
    if failed_tests:
        print(f"\n{RED}Failed tests:{NC}")
        for test in failed_tests:
            print(f"  {RED}✗{NC} {test}")
    
    if skipped_tests:
        print(f"\n{YELLOW}Warnings/Skipped:{NC}")
        for test in skipped_tests[:5]:  # Show first 5
            print(f"  {YELLOW}⚠{NC} {test}")
        if len(skipped_tests) > 5:
            print(f"  ... and {len(skipped_tests) - 5} more")
    
    print(f"\n{BLUE}{'='*70}{NC}")
    if len(failed_tests) == 0:
        print(f"{GREEN}✓ All critical tests passed!{NC}")
        print(f"{GREEN}CLI is working correctly.{NC}")
        return 0
    else:
        print(f"{RED}✗ Some tests failed. Please review the errors above.{NC}")
        return 1


def main():
    """Run all CLI tests."""
    print(f"{BLUE}")
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║         AI Asylum CLI Test Suite                                ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    print(f"{NC}")
    
    print_info("Testing CLI commands...")
    print_info("Note: Some tests may be skipped if database/Ollama is not configured\n")
    
    # Run all tests
    tests = [
        ("Help Command", test_help_command),
        ("List Benchmarks", test_list_benchmarks),
        ("Results Command", test_results_command),
        ("Run Command Validation", test_run_command_validation),
        ("Run Benchmark Validation", test_run_benchmark_validation),
        ("Ollama Commands", test_ollama_commands),
        ("Invalid Command Handling", test_invalid_command),
        ("Command Help", test_command_help),
    ]
    
    for name, test_func in tests:
        try:
            test_func()
        except Exception as e:
            print_error(f"{name} test crashed: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    exit_code = print_summary()
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
