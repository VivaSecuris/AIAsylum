"""Test database models and operations."""

import pytest
from datetime import datetime

from vivasecuris.aiasylum.database.models import (
    TestRun,
    TestResult,
    ConversationTurn,
    Assessment,
    BenchmarkResult,
)


class TestDatabaseModels:
    """Test database models."""
    
    def test_test_run_creation(self, db_session):
        """Test creating a test run."""
        test_run = TestRun(
            doctor_provider="ollama",
            doctor_model="llama3.2",
            patient_provider="ollama",
            patient_model="llama2",
            test_type="conversation",
            status="pending",
        )
        
        db_session.add(test_run)
        db_session.commit()
        
        assert test_run.id is not None
        assert test_run.created_at is not None
        assert test_run.doctor_model == "llama3.2"
    
    def test_test_result_creation(self, db_session):
        """Test creating a test result."""
        test_run = TestRun(
            doctor_provider="ollama",
            doctor_model="llama3.2",
            patient_provider="ollama",
            patient_model="llama2",
            test_type="conversation",
        )
        db_session.add(test_run)
        db_session.commit()
        
        result = TestResult(
            test_run_id=test_run.id,
            test_name="test_conversation",
            test_category="conversation",
            input_prompt="Test prompt",
            output_response="Test response",
            score=0.8,
        )
        
        db_session.add(result)
        db_session.commit()
        
        assert result.id is not None
        assert result.test_run_id == test_run.id
        assert result.score == 0.8
    
    def test_conversation_turn_creation(self, db_session):
        """Test creating a conversation turn."""
        test_run = TestRun(
            doctor_provider="ollama",
            doctor_model="llama3.2",
            patient_provider="ollama",
            patient_model="llama2",
            test_type="conversation",
        )
        db_session.add(test_run)
        db_session.commit()
        
        turn = ConversationTurn(
            test_run_id=test_run.id,
            turn_number=1,
            speaker="doctor",
            prompt="Hello",
            response="Hi there",
        )
        
        db_session.add(turn)
        db_session.commit()
        
        assert turn.id is not None
        assert turn.turn_number == 1
        assert turn.speaker == "doctor"
    
    def test_assessment_creation(self, db_session):
        """Test creating an assessment."""
        test_run = TestRun(
            doctor_provider="ollama",
            doctor_model="llama3.2",
            patient_provider="ollama",
            patient_model="llama2",
            test_type="conversation",
        )
        db_session.add(test_run)
        db_session.commit()
        
        assessment = Assessment(
            test_run_id=test_run.id,
            assessment_text="Test assessment",
            scores={"alignment": 0.8, "safety": 0.7},
            overall_score=0.75,
        )
        
        db_session.add(assessment)
        db_session.commit()
        
        assert assessment.id is not None
        assert assessment.overall_score == 0.75
        assert "alignment" in assessment.scores
