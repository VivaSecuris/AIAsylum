"""Integration tests for end-to-end workflows."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from vivasecuris.aiasylum.runner import TestRunner
from vivasecuris.aiasylum.database import get_session, TestRun, Assessment
from tests.test_doctor_patient import MockModel


class TestIntegrationWorkflows:
    """Test complete integration workflows."""
    
    @pytest.mark.asyncio
    async def test_full_test_run_workflow(self, db_session, mock_env):
        """Test complete test run workflow."""
        # Use the test session so runner and test share the same DB
        with patch('vivasecuris.aiasylum.runner.runner.get_session', return_value=db_session), \
             patch.object(db_session, 'close'):
            # Patch the name the runner imported, not its source module
            with patch('vivasecuris.aiasylum.runner.runner.get_provider') as mock_get_provider:
                mock_provider = MagicMock()
                doctor_mock_model = MockModel()
                patient_mock_model = MockModel()
                def create_model_side_effect(model_name, **kwargs):
                    if "doctor" in str(model_name) or "llama3" in str(model_name):
                        return doctor_mock_model
                    return patient_mock_model
                mock_provider.create_model.side_effect = create_model_side_effect
                mock_get_provider.return_value = mock_provider

                runner = TestRunner()
                test_run = await runner.run_test(
                    doctor_provider="ollama",
                    doctor_model="llama3.2",
                    patient_provider="ollama",
                    patient_model="llama2",
                    test_type="conversation",
                )

            db_session.refresh(test_run)
            assert test_run.id is not None
            assert test_run.status == "completed"
            assert test_run.test_type == "conversation"
            assert len(test_run.results) > 0
    
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
        db_session.refresh(test_run)
        test_run_id = test_run.id
        
        # Create test result
        from vivasecuris.aiasylum.database.models import TestResult
        result = TestResult(
            test_run_id=test_run_id,
            test_name="test",
            test_category="conversation",
            input_prompt="Test",
            output_response="Response",
            score=0.8,
        )
        db_session.add(result)
        db_session.commit()
        
        # Run analysis using the same session so it sees our test run
        with patch('vivasecuris.aiasylum.analysis.analyzer.get_session', return_value=db_session), \
             patch.object(db_session, 'close'):
            service = AnalysisService()
            assessment = await service.analyze_test_run(
                test_run_id=test_run_id,
                enable_cot_detection=True,
            )
        
        assert assessment.id is not None
        assert assessment.overall_score is not None
        assert assessment.scores is not None

        # Analysis metadata must survive the save (it was silently dropped via a metadata= kwarg)
        db_session.expire_all()
        persisted = db_session.get(Assessment, assessment.id)
        assert "cot_analysis" in (persisted.meta_data or {})
