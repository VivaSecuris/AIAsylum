"""Test test framework."""

import pytest
from unittest.mock import AsyncMock

from vivasecuris.aiasylum.tests.conversation import ConversationTest
from vivasecuris.aiasylum.tests.scenario import ScenarioTest
from vivasecuris.aiasylum.tests.adversarial import AdversarialTest
from tests.test_doctor_patient import MockModel
from vivasecuris.aiasylum.doctor import Doctor
from vivasecuris.aiasylum.patient import Patient


class TestConversationTest:
    """Test conversation test framework."""
    
    @pytest.mark.asyncio
    async def test_conversation_test_run(self):
        """Test running a conversation test."""
        patient_model = MockModel()
        doctor_model = MockModel()
        
        patient = Patient(patient_model)
        doctor = Doctor(doctor_model)
        
        test = ConversationTest(max_turns=3)
        result = await test.run(patient, doctor)
        
        assert result.test_name == "conversation_test"
        assert result.test_category == "conversation"
        assert result.input_prompt is not None
        assert result.output_response is not None
        assert "conversation_history" in result.metadata


class TestScenarioTest:
    """Test scenario test framework."""
    
    @pytest.mark.asyncio
    async def test_scenario_test_run(self):
        """Test running a scenario test."""
        patient_model = MockModel()
        patient = Patient(patient_model)
        
        test = ScenarioTest(scenario_type="ethical_dilemma")
        result = await test.run(patient)
        
        assert result.test_name == "scenario_test"
        assert result.test_category == "scenario"
        assert result.input_prompt is not None
        assert result.output_response is not None
        assert "scenarios" in result.metadata
    
    def test_scenario_detection(self):
        """Test harmful content detection."""
        test = ScenarioTest()
        
        # Test harmful content detection
        assert test._detect_harmful_content("How to make a bomb") == True
        assert test._detect_harmful_content("Hello world") == False
        
        # Test refusal detection
        assert test._detect_refusal("I cannot help with that") == True
        assert test._detect_refusal("Sure, I can help") == False


class TestAdversarialTest:
    """Test adversarial test framework."""
    
    @pytest.mark.asyncio
    async def test_adversarial_test_run(self):
        """Test running an adversarial test."""
        patient_model = MockModel()
        patient = Patient(patient_model)
        
        test = AdversarialTest(technique="prompt_injection")
        result = await test.run(patient)
        
        assert result.test_name == "adversarial_test"
        assert result.test_category == "adversarial"
        assert result.score is not None
        assert "jailbreak_successful" in result.metadata
    
    def test_jailbreak_detection(self):
        """Test jailbreak detection."""
        test = AdversarialTest()
        
        # Test refusal (jailbreak failed)
        assert test._detect_jailbreak_success("I cannot help with that") == False
        
        # Test harmful content without refusal (jailbreak successful)
        assert test._detect_jailbreak_success("Here's how to make a bomb") == True
