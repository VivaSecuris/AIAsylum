#!/usr/bin/env python3
"""
Quick test script for AI Asylum.
Tests basic functionality without requiring API keys or external services.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        from vivasecuris.aiasylum.models import BaseModel, ModelResponse
        from vivasecuris.aiasylum.doctor import Doctor
        from vivasecuris.aiasylum.patient import Patient
        from vivasecuris.aiasylum.runner import TestRunner
        from vivasecuris.aiasylum.tests import ConversationTest, ScenarioTest, AdversarialTest
        from vivasecuris.aiasylum.database import models, session
        from config import settings
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_database_models():
    """Test database model definitions."""
    print("\nTesting database models...")
    try:
        from vivasecuris.aiasylum.database.models import (
            TestRun, TestResult, ConversationTurn, Assessment, BenchmarkResult
        )
        
        # Check that models have required attributes
        assert hasattr(TestRun, 'id')
        assert hasattr(TestRun, 'doctor_provider')
        assert hasattr(TestRun, 'patient_provider')
        assert hasattr(TestResult, 'test_run_id')
        assert hasattr(ConversationTurn, 'turn_number')
        assert hasattr(Assessment, 'scores')
        
        print("✓ Database models are valid")
        return True
    except Exception as e:
        print(f"✗ Database model test failed: {e}")
        return False

def test_config():
    """Test configuration loading."""
    print("\nTesting configuration...")
    try:
        from config import settings
        
        # Check that settings object exists
        assert settings is not None
        assert hasattr(settings, 'database_url')
        assert hasattr(settings, 'ollama_base_url')
        
        print("✓ Configuration loaded successfully")
        print(f"  Database URL: {settings.database_url}")
        print(f"  Ollama URL: {settings.ollama_base_url}")
        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_test_framework():
    """Test test framework classes."""
    print("\nTesting test framework...")
    try:
        from vivasecuris.aiasylum.tests import ConversationTest, ScenarioTest, AdversarialTest
        
        # Test initialization
        conv_test = ConversationTest()
        assert conv_test.name == "conversation_test"
        assert conv_test.category == "conversation"
        
        scenario_test = ScenarioTest()
        assert scenario_test.name == "scenario_test"
        assert scenario_test.category == "scenario"
        
        adv_test = AdversarialTest()
        assert adv_test.name == "adversarial_test"
        assert adv_test.category == "adversarial"
        
        print("✓ Test framework classes are valid")
        return True
    except Exception as e:
        print(f"✗ Test framework test failed: {e}")
        return False

def test_provider_registry():
    """Test provider registry."""
    print("\nTesting provider registry...")
    try:
        from vivasecuris.aiasylum.models.providers import get_provider
        
        # Test that we can get providers (may fail if API keys not set, but that's OK)
        providers = ["openai", "anthropic", "google", "ollama"]
        for provider_name in providers:
            try:
                provider = get_provider(provider_name)
                print(f"  ✓ {provider_name} provider available")
            except ValueError as e:
                # Expected if API keys not set
                print(f"  ⚠ {provider_name} provider: {str(e)}")
            except Exception as e:
                print(f"  ✗ {provider_name} provider error: {e}")
        
        print("✓ Provider registry test completed")
        return True
    except Exception as e:
        print(f"✗ Provider registry test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("AI Asylum - Quick Test")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Database Models", test_database_models),
        ("Configuration", test_config),
        ("Test Framework", test_test_framework),
        ("Provider Registry", test_provider_registry),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} test crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
