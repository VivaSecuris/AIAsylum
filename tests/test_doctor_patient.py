"""Test doctor and patient systems."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from vivasecuris.aiasylum.doctor import Doctor
from vivasecuris.aiasylum.patient import Patient
from vivasecuris.aiasylum.models.base import BaseModel, ModelResponse


class MockModel(BaseModel):
    """Mock model for testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__("mock-model", "mock", **kwargs)
        self.responses = []
    
    async def generate(self, prompt="", system_prompt=None, messages=None, **kwargs):
        if messages:
            # Simulate conversation
            last_message = messages[-1]["content"] if messages else ""
            response_text = f"Response to: {last_message}"
        else:
            response_text = "Mock response"
        
        return ModelResponse(
            content=response_text,
            model=self.model_name,
            provider=self.provider,
        )
    
    async def stream_generate(self, prompt="", system_prompt=None, messages=None, **kwargs):
        yield "Mock stream response"


class TestDoctor:
    """Test doctor model system."""
    
    def test_doctor_initialization(self):
        """Test doctor initialization."""
        model = MockModel()
        doctor = Doctor(model)
        
        assert doctor.model == model
        assert doctor.system_prompt is not None
        assert len(doctor.conversation_history) == 0
    
    def test_doctor_custom_prompt(self):
        """Test doctor with custom system prompt."""
        model = MockModel()
        custom_prompt = "Custom system prompt"
        doctor = Doctor(model, system_prompt=custom_prompt)
        
        assert doctor.system_prompt == custom_prompt
    
    @pytest.mark.asyncio
    async def test_doctor_conduct_interview(self):
        """Test doctor conducting interview."""
        model = MockModel()
        doctor = Doctor(model)
        
        response = await doctor.conduct_interview("Patient response", turn_number=1)
        
        assert response.content is not None
        assert len(doctor.conversation_history) > 0
    
    @pytest.mark.asyncio
    async def test_doctor_generate_assessment(self):
        """Test doctor generating assessment."""
        model = MockModel()
        doctor = Doctor(model)
        
        assessment = await doctor.generate_assessment(
            "Test conversation summary",
            test_results={"score": 0.8}
        )
        
        assert assessment.content is not None
    
    def test_doctor_reset(self):
        """Test doctor reset."""
        model = MockModel()
        doctor = Doctor(model)
        doctor.conversation_history = [{"role": "user", "content": "test"}]
        
        doctor.reset()
        
        assert len(doctor.conversation_history) == 0


class TestPatient:
    """Test patient model system."""
    
    def test_patient_initialization(self):
        """Test patient initialization."""
        model = MockModel()
        patient = Patient(model)
        
        assert patient.model == model
        assert len(patient.conversation_history) == 0
    
    @pytest.mark.asyncio
    async def test_patient_respond(self):
        """Test patient responding to prompt."""
        model = MockModel()
        patient = Patient(model)
        
        response = await patient.respond("Test question")
        
        assert response.content is not None
        assert len(patient.conversation_history) == 2  # User prompt + assistant response
    
    def test_patient_reset(self):
        """Test patient reset."""
        model = MockModel()
        patient = Patient(model)
        patient.conversation_history = [{"role": "user", "content": "test"}]
        
        patient.reset()
        
        assert len(patient.conversation_history) == 0
