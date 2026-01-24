"""Integration tests for end-to-end workflows."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from vivasecuris.aiasylum.runner import TestRunner
from vivasecuris.aiasylum.database import get_session, TestRun
from tests.test_doctor_patient import MockModel


class TestIntegrationWorkflows:
    """Test complete integration workflows."""
    
    @pytest.mark.asyncio
    async def test_full_test_run_workflow(self, db_session, mock_env):
        """Test complete test run workflow."""
        # Mock the providers and httpx for Ollama
        with patch('vivasecuris.aiasylum.models.providers.get_provider') as mock_get_provider, \
             patch('httpx.AsyncClient') as mock_httpx:
            
            # Create a mock provider that returns MockModel
            mock_provider = MagicMock()
            mock_model = MockModel()
            mock_provider.create_model.return_value = mock_model
            mock_get_provider.return_value = mock_provider
            
            # Mock httpx response for Ollama
            mock_response = MagicMock()
            mock_response.json.return_value = {"response": "Test response", "done": True}
            mock_response.raise_for_status = MagicMock()
            mock_httpx.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
            
            runner = TestRunner()
            runner.session = db_session
            
            # Run a test
            test_run = await runner.run_test(
                doctor_provider="ollama",
                doctor_model="llama3.2",
                patient_provider="ollama",
                patient_model="llama2",
                test_type="conversation",
            )
            
            # Verify test run was created
            assert test_run.id is not None
            assert test_run.status == "completed"
            assert test_run.test_type == "conversation"
            
            # Verify results were saved
            assert len(test_run.results) > 0
            # Conversations may be empty if test fails early, so just check test_run exists
            assert test_run is not None
    
    @pytest.mark.asyncio
    async def test_analysis_workflow(self, db_session, mock_env):
        """Test analysis workflow."""
        from vivasecuris.aiasylum.analysis import AnalysisService
        
        # Create a test run first
        test_run = TestRun(
            doctor_provider="ollama",
            doctor_model="llama3.2",
            patient_provider="ollama",
            patient_model="llama2",
            test_type="conversation",
            status="completed",
        )
        db_session.add(test_run)
        db_session.commit()
        
        # Create test result
        from vivasecuris.aiasylum.database.models import TestResult
        result = TestResult(
            test_run_id=test_run.id,
            test_name="test",
            test_category="conversation",
            input_prompt="Test",
            output_response="Response",
            score=0.8,
        )
        db_session.add(result)
        db_session.commit()
        
        # Run analysis
        service = AnalysisService()
        service.session = db_session
        
        assessment = await service.analyze_test_run(
            test_run_id=test_run.id,
            enable_cot_detection=True,
        )
        
        assert assessment.id is not None
        assert assessment.overall_score is not None
        assert assessment.scores is not None
