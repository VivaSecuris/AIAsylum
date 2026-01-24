"""Test execution runner."""

from typing import Dict, List, Optional

from typing import Dict, List, Optional

from vivasecuris.aiasylum.models import get_provider
from vivasecuris.aiasylum.tests import ConversationTest, ScenarioTest, AdversarialTest
from vivasecuris.aiasylum.database import get_session, TestRun, TestResult, ConversationTurn, PromptLibrary
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
        pass  # Don't create session here - create per operation
    
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
        session = get_session()
        try:
            # Create test run record
            test_run = TestRun(
                doctor_provider=doctor_provider,
                doctor_model=doctor_model,
                patient_provider=patient_provider,
                patient_model=patient_model,
                test_type=test_type,
                status=STATUS_RUNNING,
            )
            session.add(test_run)
            session.commit()
            session.refresh(test_run)
            
            try:
                # Get providers and create models
                doctor_provider_instance = get_provider(doctor_provider)
                patient_provider_instance = get_provider(patient_provider)
                
                doctor_model_instance = doctor_provider_instance.create_model(doctor_model)
                patient_model_instance = patient_provider_instance.create_model(patient_model)
                
                # Load system prompts for doctor and patient
                doctor_system_prompt = None
                patient_system_prompt = None
                
                if test_config and test_config.get("doctor_system_prompt_id"):
                    doctor_prompt = session.query(PromptLibrary).filter(
                        PromptLibrary.id == test_config["doctor_system_prompt_id"],
                        PromptLibrary.prompt_type == "system_prompt",
                        PromptLibrary.target == "doctor"
                    ).first()
                    if doctor_prompt:
                        doctor_system_prompt = doctor_prompt.prompt_text
                        doctor_prompt.usage_count = (doctor_prompt.usage_count or 0) + 1
                        session.commit()
                
                if test_config and test_config.get("patient_system_prompt_id"):
                    patient_prompt = session.query(PromptLibrary).filter(
                        PromptLibrary.id == test_config["patient_system_prompt_id"],
                        PromptLibrary.prompt_type == "system_prompt",
                        PromptLibrary.target == "patient"
                    ).first()
                    if patient_prompt:
                        patient_system_prompt = patient_prompt.prompt_text
                        patient_prompt.usage_count = (patient_prompt.usage_count or 0) + 1
                        session.commit()
                
                # Load test prompt from library if prompt_id is specified
                prompt_text = None
                if test_config and test_config.get("prompt_id"):
                    prompt = session.query(PromptLibrary).filter(
                        PromptLibrary.id == test_config["prompt_id"],
                        PromptLibrary.prompt_type == "test_prompt"
                    ).first()
                    if prompt:
                        prompt_text = prompt.prompt_text
                        # Increment usage count
                        prompt.usage_count = (prompt.usage_count or 0) + 1
                        session.commit()
                
                # Update test_config with system prompts
                if not test_config:
                    test_config = {}
                if doctor_system_prompt:
                    test_config["doctor_system_prompt"] = doctor_system_prompt
                if patient_system_prompt:
                    test_config["patient_system_prompt"] = patient_system_prompt
                
                # Run appropriate test (tests expect models, not Patient/Doctor objects)
                test_result: TestResultType
                if test_type == TEST_TYPE_CONVERSATION:
                    # Use prompt from library if available, otherwise use doctor_prompt from config
                    doctor_prompt = prompt_text or test_config.get("doctor_prompt")
                    test = ConversationTest(doctor_prompt=doctor_prompt)
                    test_result = await test.run(patient_model_instance, doctor_model_instance, context=test_config)
                elif test_type == TEST_TYPE_SCENARIO:
                    # Use prompt from library if available, otherwise use scenarios from config
                    if prompt_text:
                        test = ScenarioTest(scenarios=[prompt_text])
                    else:
                        scenario_type = test_config.get("scenario_type", "ethical_dilemma") if test_config else "ethical_dilemma"
                        test = ScenarioTest(scenario_type=scenario_type)
                    test_result = await test.run(patient_model_instance, doctor_model_instance, context=test_config)
                elif test_type == TEST_TYPE_ADVERSARIAL:
                    # Use prompt from library if available, otherwise use technique from config
                    if prompt_text:
                        test = AdversarialTest(prompts=[prompt_text])
                    else:
                        technique = test_config.get("technique", "prompt_injection") if test_config else "prompt_injection"
                        test = AdversarialTest(technique=technique)
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
                    meta_data=test_result.metadata,
                )
                session.add(db_result)
                
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
                        session.add(turn_record)
                
                # Update test run status
                test_run.status = "completed"
                session.commit()
                
                return test_run
            
            except Exception as e:
                test_run.status = "failed"
                if not test_run.meta_data:
                    test_run.meta_data = {}
                test_run.meta_data["error"] = str(e)
                session.commit()
                raise
        finally:
            session.close()
    
    def get_test_run(self, test_run_id: int) -> Optional[TestRun]:
        """Get a test run by ID."""
        session = get_session()
        try:
            return session.query(TestRun).filter(TestRun.id == test_run_id).first()
        finally:
            session.close()
    
    async def execute_test_run(self, test_run_id: int) -> TestRun:
        """
        Execute a test for an existing test run.
        
        Args:
            test_run_id: ID of the test run to execute
        
        Returns:
            TestRun database record
        """
        session = get_session()
        try:
            test_run = session.query(TestRun).filter(TestRun.id == test_run_id).first()
            if not test_run:
                raise TestExecutionError(f"Test run {test_run_id} not found")
            
            if test_run.status not in (STATUS_PENDING, STATUS_FAILED):
                raise TestExecutionError(f"Test run {test_run_id} is not in a startable state (current: {test_run.status})")
            
            # Update status to running
            test_run.status = STATUS_RUNNING
            session.commit()
            session.refresh(test_run)
            
            try:
                # Get providers and create models
                doctor_provider_instance = get_provider(test_run.doctor_provider)
                patient_provider_instance = get_provider(test_run.patient_provider)
                
                doctor_model_instance = doctor_provider_instance.create_model(test_run.doctor_model)
                patient_model_instance = patient_provider_instance.create_model(test_run.patient_model)
                
                # Get test config from metadata if available
                test_config = test_run.meta_data.get("test_config") if test_run.meta_data else {}
                
                # Load system prompts for doctor and patient
                doctor_system_prompt = None
                patient_system_prompt = None
                
                if test_config.get("doctor_system_prompt_id"):
                    doctor_prompt = session.query(PromptLibrary).filter(
                        PromptLibrary.id == test_config["doctor_system_prompt_id"],
                        PromptLibrary.prompt_type == "system_prompt",
                        PromptLibrary.target == "doctor"
                    ).first()
                    if doctor_prompt:
                        doctor_system_prompt = doctor_prompt.prompt_text
                        doctor_prompt.usage_count = (doctor_prompt.usage_count or 0) + 1
                        session.commit()
                
                if test_config.get("patient_system_prompt_id"):
                    patient_prompt = session.query(PromptLibrary).filter(
                        PromptLibrary.id == test_config["patient_system_prompt_id"],
                        PromptLibrary.prompt_type == "system_prompt",
                        PromptLibrary.target == "patient"
                    ).first()
                    if patient_prompt:
                        patient_system_prompt = patient_prompt.prompt_text
                        patient_prompt.usage_count = (patient_prompt.usage_count or 0) + 1
                        session.commit()
                
                # Load test prompt from library if prompt_id is specified
                prompt_text = None
                if test_config.get("prompt_id"):
                    prompt = session.query(PromptLibrary).filter(
                        PromptLibrary.id == test_config["prompt_id"],
                        PromptLibrary.prompt_type == "test_prompt"
                    ).first()
                    if prompt:
                        prompt_text = prompt.prompt_text
                        # Increment usage count
                        prompt.usage_count = (prompt.usage_count or 0) + 1
                        session.commit()
                
                # Update test_config with system prompts
                if doctor_system_prompt:
                    test_config["doctor_system_prompt"] = doctor_system_prompt
                if patient_system_prompt:
                    test_config["patient_system_prompt"] = patient_system_prompt
                
                # Run appropriate test
                test_result: TestResultType
                if test_run.test_type == TEST_TYPE_CONVERSATION:
                    # Use prompt from library if available, otherwise use doctor_prompt from config
                    doctor_prompt = prompt_text or test_config.get("doctor_prompt")
                    test = ConversationTest(doctor_prompt=doctor_prompt)
                    test_result = await test.run(patient_model_instance, doctor_model_instance, context=test_config)
                elif test_run.test_type == TEST_TYPE_SCENARIO:
                    # Use prompt from library if available, otherwise use scenarios from config
                    if prompt_text:
                        test = ScenarioTest(scenarios=[prompt_text])
                    else:
                        scenario_type = test_config.get("scenario_type", "ethical_dilemma")
                        test = ScenarioTest(scenario_type=scenario_type)
                    test_result = await test.run(patient_model_instance, doctor_model_instance, context=test_config)
                elif test_run.test_type == TEST_TYPE_ADVERSARIAL:
                    # Use prompt from library if available, otherwise use technique from config
                    if prompt_text:
                        test = AdversarialTest(prompts=[prompt_text])
                    else:
                        technique = test_config.get("technique", "prompt_injection")
                        test = AdversarialTest(technique=technique)
                    test_result = await test.run(patient_model_instance, doctor_model_instance, context=test_config)
                else:
                    raise TestExecutionError(f"Unknown test type: {test_run.test_type}")
                
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
                    meta_data=test_result.metadata,
                )
                session.add(db_result)
                
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
                        session.add(turn_record)
                
                # Update test run status
                test_run.status = "completed"
                session.commit()
                
                return test_run
            
            except Exception as e:
                test_run.status = "failed"
                if not test_run.meta_data:
                    test_run.meta_data = {}
                test_run.meta_data["error"] = str(e)
                session.commit()
                raise
        finally:
            session.close()
    
    def list_test_runs(
        self,
        limit: int = 100,
        offset: int = 0,
        test_type: Optional[str] = None,
    ) -> List[TestRun]:
        """List test runs."""
        session = get_session()
        try:
            query = session.query(TestRun)
            if test_type:
                query = query.filter(TestRun.test_type == test_type)
            return query.order_by(TestRun.created_at.desc()).limit(limit).offset(offset).all()
        finally:
            session.close()
