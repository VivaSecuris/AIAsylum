"""Test execution runner."""

from typing import Dict, List, Optional

from typing import Dict, List, Optional

from vivasecuris.aiasylum.models import get_provider
from vivasecuris.aiasylum.tests import ConversationTest, ScenarioTest, AdversarialTest
from vivasecuris.aiasylum.database import get_session, TestRun, TestResult, ConversationTurn
from vivasecuris.aiasylum.tests.base import TestResult as TestResultType
from vivasecuris.aiasylum.constants import (
    TEST_TYPE_CONVERSATION,
    TEST_TYPE_SCENARIO,
    TEST_TYPE_ADVERSARIAL,
    STATUS_PENDING,
    STATUS_RUNNING,
    STATUS_COMPLETED,
    STATUS_FAILED,
)
from vivasecuris.aiasylum.exceptions import TestExecutionError


class TestRunner:
    """Runs tests and saves results to database."""
    
    def __init__(self):
        self.session = get_session()
    
    async def run_test(
        self,
        doctor_provider: str,
        doctor_model: str,
        patient_provider: str,
        patient_model: str,
        test_type: str,
        test_config: Optional[Dict] = None,
    ) -> TestRun:
        """
        Run a test and save results.
        
        Args:
            doctor_provider: Provider for doctor model
            doctor_model: Doctor model name
            patient_provider: Provider for patient model
            patient_model: Patient model name
            test_type: Type of test (conversation, scenario, adversarial)
            test_config: Optional test configuration
        
        Returns:
            TestRun database record
        """
        # Create test run record
        test_run = TestRun(
            doctor_provider=doctor_provider,
            doctor_model=doctor_model,
            patient_provider=patient_provider,
            patient_model=patient_model,
            test_type=test_type,
            status=STATUS_RUNNING,
        )
        self.session.add(test_run)
        self.session.commit()
        self.session.refresh(test_run)
        
        try:
            # Get providers and create models
            doctor_provider_instance = get_provider(doctor_provider)
            patient_provider_instance = get_provider(patient_provider)
            
            doctor_model_instance = doctor_provider_instance.create_model(doctor_model)
            patient_model_instance = patient_provider_instance.create_model(patient_model)
            
            # Run appropriate test (tests expect models, not Patient/Doctor objects)
            test_result: TestResultType
            if test_type == TEST_TYPE_CONVERSATION:
                test = ConversationTest()
                test_result = await test.run(patient_model_instance, doctor_model_instance, context=test_config)
            elif test_type == TEST_TYPE_SCENARIO:
                test = ScenarioTest()
                test_result = await test.run(patient_model_instance, doctor_model_instance, context=test_config)
            elif test_type == TEST_TYPE_ADVERSARIAL:
                test = AdversarialTest()
                test_result = await test.run(patient_model_instance, doctor_model_instance, context=test_config)
            else:
                raise TestExecutionError(f"Unknown test type: {test_type}")
            
            # Save test result
            db_result = TestResult(
                test_run_id=test_run.id,
                test_name=test_result.test_name,
                test_category=test_result.test_category,
                input_prompt=test_result.input_prompt,
                output_response=test_result.output_response,
                score=test_result.score,
                scores=test_result.scores,
                analysis=test_result.analysis,
                flags=test_result.flags,
                metadata=test_result.metadata,
            )
            self.session.add(db_result)
            
            # Save conversation turns if available
            if test_result.metadata and "conversation_history" in test_result.metadata:
                for i, turn in enumerate(test_result.metadata["conversation_history"]):
                    turn_record = ConversationTurn(
                        test_run_id=test_run.id,
                        turn_number=i,
                        speaker=turn["speaker"],
                        prompt=turn.get("prompt", ""),
                        response=turn.get("response", ""),
                    )
                    self.session.add(turn_record)
            
            # Update test run status
            test_run.status = "completed"
            self.session.commit()
            
            return test_run
        
        except Exception as e:
            test_run.status = "failed"
            test_run.metadata = {"error": str(e)}
            self.session.commit()
            raise
    
    def get_test_run(self, test_run_id: int) -> Optional[TestRun]:
        """Get a test run by ID."""
        return self.session.query(TestRun).filter(TestRun.id == test_run_id).first()
    
    def list_test_runs(
        self,
        limit: int = 100,
        offset: int = 0,
        test_type: Optional[str] = None,
    ) -> List[TestRun]:
        """List test runs."""
        query = self.session.query(TestRun)
        if test_type:
            query = query.filter(TestRun.test_type == test_type)
        return query.order_by(TestRun.created_at.desc()).limit(limit).offset(offset).all()
