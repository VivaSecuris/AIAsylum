"""Test execution runner."""

import logging
from datetime import datetime
from typing import Dict, List, Optional

from sqlalchemy.orm import Session
from sqlalchemy.orm.exc import StaleDataError
from sqlalchemy.exc import PendingRollbackError

from vivasecuris.aiasylum.models import get_provider
from vivasecuris.aiasylum.tests import OneShotTest, MultiShotTest, ConversationTest, ScenarioTest, AdversarialTest, BenchmarkTest, GroupTherapyTest
from vivasecuris.aiasylum.database import get_session, TestRun, TestResult, ConversationTurn, PromptLibrary
from vivasecuris.aiasylum.tests.base import TestResult as TestResultType
from vivasecuris.aiasylum.utils import substitute_variables
from vivasecuris.aiasylum.constants import (
    TEST_TYPE_ONE_SHOT,
    TEST_TYPE_MULTI_SHOT,
    TEST_TYPE_CONVERSATION,  # Legacy
    TEST_TYPE_BENCHMARK,
    TEST_TYPE_GROUP_THERAPY,
    TEST_TYPE_SCENARIO,  # Legacy
    TEST_TYPE_ADVERSARIAL,  # Legacy
    TEST_TYPE_ANALYSIS,
    STATUS_PENDING,
    STATUS_RUNNING,
    STATUS_PAUSED,
    STATUS_COMPLETED,
    STATUS_FAILED,
)
from vivasecuris.aiasylum.exceptions import TestExecutionError
from vivasecuris.aiasylum.api.progress_events import progress_event_manager
from vivasecuris.aiasylum.api.cancellation import cancellation_manager

logger = logging.getLogger(__name__)


def safe_commit(session: Session, test_run_id: int, test_run: Optional[TestRun] = None):
    """
    Safely commit a session, handling cancellation and stale data errors.
    
    Args:
        session: SQLAlchemy session
        test_run_id: ID of the test run (for cancellation check)
        test_run: Optional TestRun object to refresh if stale
    """
    # Check for cancellation before committing
    if cancellation_manager.is_cancelled(test_run_id):
        logger.warning(f"🛑 Cancellation detected before commit for test run {test_run_id}")
        session.rollback()
        raise TestExecutionError(f"Test run {test_run_id} was cancelled")
    
    try:
        session.commit()
    except (StaleDataError, PendingRollbackError) as e:
        logger.warning(f"⚠️ Stale data or rollback error for test run {test_run_id}: {e}")
        session.rollback()
        
        # Check for cancellation after rollback
        if cancellation_manager.is_cancelled(test_run_id):
            logger.warning(f"🛑 Cancellation detected after rollback for test run {test_run_id}")
            raise TestExecutionError(f"Test run {test_run_id} was cancelled")
        
        # If we have a test_run object, refresh it from the database
        if test_run:
            try:
                session.refresh(test_run)
                # Check if it was cancelled by another process
                if test_run.status == STATUS_FAILED and test_run.meta_data and test_run.meta_data.get("cancelled"):
                    logger.info(f"🛑 Test run {test_run_id} was cancelled by another process")
                    raise TestExecutionError(f"Test run {test_run_id} was cancelled")
            except Exception as refresh_error:
                logger.warning(f"Could not refresh test_run {test_run_id}: {refresh_error}")
        
        # Re-raise the original error if it wasn't a cancellation
        raise


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
                
                # Ensure Ollama models are available (auto-pull if missing)
                from vivasecuris.aiasylum.models.ollama import OllamaModel
                if doctor_provider == "ollama" and isinstance(doctor_model_instance, OllamaModel):
                    if not await doctor_model_instance.check_available():
                        logger.info(f"Ollama model {doctor_model} not found, pulling...")
                        await doctor_model_instance.pull_model()
                if patient_provider == "ollama" and isinstance(patient_model_instance, OllamaModel):
                    if not await patient_model_instance.check_available():
                        logger.info(f"Ollama model {patient_model} not found, pulling...")
                        await patient_model_instance.pull_model()
                
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
                        # Substitute variables if provided
                        if test_config.get("variables"):
                            prompt_text = substitute_variables(prompt_text, test_config["variables"])
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
                
                # Substitute variables in custom prompts if provided
                variables = test_config.get("variables", {}) if test_config else {}
                if variables:
                    if test_config.get("prompts"):
                        test_config["prompts"] = [
                            substitute_variables(p, variables) for p in test_config["prompts"]
                        ]
                    if test_config.get("prompt"):
                        test_config["prompt"] = substitute_variables(test_config["prompt"], variables)
                
                # Run appropriate test (tests expect models, not Patient/Doctor objects)
                test_result: TestResultType
                if test_type == TEST_TYPE_ONE_SHOT:
                    # One-shot test - single prompt/response
                    if prompt_text:
                        test = OneShotTest(prompts=[prompt_text])
                    else:
                        prompts = test_config.get("prompts", []) if test_config else []
                        if not prompts:
                            # Fallback to single prompt if provided
                            single_prompt = test_config.get("prompt") if test_config else None
                            if single_prompt:
                                prompts = [single_prompt]
                        test = OneShotTest(prompts=prompts)
                    test_result = await test.run(patient_model_instance, doctor_model_instance, context=test_config)
                elif test_type == TEST_TYPE_MULTI_SHOT:
                    # Multi-shot test - multiple sequential prompts to test context handling
                    if prompt_text:
                        test = MultiShotTest(prompts=[prompt_text])
                    else:
                        prompts = test_config.get("prompts", []) if test_config else []
                        num_messages = test_config.get("num_messages", 10) if test_config else 10
                        if prompts:
                            test = MultiShotTest(prompts=prompts)
                        else:
                            test = MultiShotTest(num_messages=num_messages)
                    test_result = await test.run(patient_model_instance, doctor_model_instance, context=test_config)
                elif test_type == TEST_TYPE_CONVERSATION:
                    # Conversation test - multi-turn conversation between doctor and patient
                    max_turns = test_config.get("max_turns", 10) if test_config else 10
                    doctor_prompt = prompt_text or (test_config.get("doctor_prompt") if test_config else None)
                    test = ConversationTest(max_turns=max_turns, doctor_prompt=doctor_prompt)
                    test_result = await test.run(patient_model_instance, doctor_model_instance, context=test_config)
                elif test_type == TEST_TYPE_GROUP_THERAPY:
                    # Group therapy test - multiple patient models in a group session
                    max_turns = test_config.get("max_turns", 10) if test_config else 10
                    doctor_prompt = prompt_text or (test_config.get("doctor_prompt") if test_config else None)
                    
                    # Get patient models from test_config
                    patients_config = test_config.get("patients", []) if test_config else []
                    if not patients_config:
                        # Fallback to single patient if patients array not provided
                        patients_config = [{"provider": patient_provider, "model": patient_model}]
                    
                    # Create patient model instances
                    patient_models = []
                    patient_info_list = []
                    patient_system_prompts = {}
                    
                    for i, patient_cfg in enumerate(patients_config):
                        p_provider = patient_cfg.get("provider", patient_provider)
                        p_model = patient_cfg.get("model", patient_model)
                        
                        # Get provider and create model
                        p_provider_instance = get_provider(p_provider)
                        p_model_instance = p_provider_instance.create_model(p_model)
                        patient_models.append(p_model_instance)
                        
                        # Store patient info
                        patient_info_list.append({
                            "id": i,
                            "provider": p_provider,
                            "model": p_model,
                        })
                        
                        # Load patient system prompt if specified
                        patient_system_prompt_id = patient_cfg.get("system_prompt_id")
                        if patient_system_prompt_id:
                            p_prompt = session.query(PromptLibrary).filter(
                                PromptLibrary.id == patient_system_prompt_id,
                                PromptLibrary.prompt_type == "system_prompt",
                                PromptLibrary.target == "patient"
                            ).first()
                            if p_prompt:
                                patient_system_prompts[i] = p_prompt.prompt_text
                                p_prompt.usage_count = (p_prompt.usage_count or 0) + 1
                                session.commit()
                    
                    # Store patient list in test_run metadata
                    if not test_run.meta_data:
                        test_run.meta_data = {}
                    test_run.meta_data["patients"] = patient_info_list
                    session.commit()
                    
                    # Add patient system prompts and patient info to context
                    if patient_system_prompts:
                        test_config["patient_system_prompts"] = patient_system_prompts
                    test_config["patient_info"] = patient_info_list
                    
                    test = GroupTherapyTest(max_turns=max_turns, doctor_prompt=doctor_prompt)
                    test_result = await test.run(patient_models, doctor_model_instance, context=test_config)
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
                        reasoning = turn.get("reasoning", "")
                        # Store reasoning and patient info in metadata
                        turn_metadata = {}
                        if reasoning:
                            turn_metadata["reasoning"] = reasoning
                        # Add patient metadata for group therapy
                        if turn.get("patient_id") is not None:
                            turn_metadata["patient_id"] = turn.get("patient_id")
                        if turn.get("patient_name"):
                            turn_metadata["patient_name"] = turn.get("patient_name")
                        if turn.get("patient_model"):
                            turn_metadata["patient_model"] = turn.get("patient_model")
                        if turn.get("patient_provider"):
                            turn_metadata["patient_provider"] = turn.get("patient_provider")
                        turn_record = ConversationTurn(
                            test_run_id=test_run.id,
                            turn_number=i,
                            speaker=turn["speaker"],
                            prompt=turn.get("prompt", ""),
                            response=turn.get("response", ""),
                            meta_data=turn_metadata if turn_metadata else None,
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
        start_time = datetime.utcnow()
        progress_task = None  # Initialize progress task variable
        session = get_session()
        try:
            test_run = session.query(TestRun).filter(TestRun.id == test_run_id).first()
            if not test_run:
                raise TestExecutionError(f"Test run {test_run_id} not found")
            
            if test_run.status not in (STATUS_PENDING, STATUS_FAILED, STATUS_PAUSED):
                raise TestExecutionError(f"Test run {test_run_id} is not in a startable state (current: {test_run.status})")
            
            # If resuming from paused, log it
            is_resuming = test_run.status == STATUS_PAUSED
            if is_resuming:
                logger.info(f"▶️  Resuming paused test run #{test_run_id}")
            
            # Log test start
            suite_id = test_run.suite_id
            test_info = f"{test_run.test_type} on {test_run.patient_provider}/{test_run.patient_model}"
            if test_run.meta_data and test_run.meta_data.get("benchmark"):
                test_info += f" (benchmark: {test_run.meta_data.get('benchmark')})"
            suite_id_str = str(suite_id) if suite_id else "N/A"
            logger.info(f"▶️  Starting test run #{test_run_id}: {test_info} [Suite #{suite_id_str}]")
            
            # Check if already cancelled before starting
            if cancellation_manager.is_cancelled(test_run_id):
                test_run.status = STATUS_FAILED
                if not test_run.meta_data:
                    test_run.meta_data = {}
                test_run.meta_data["cancelled"] = True
                test_run.meta_data["error"] = "Test run was cancelled before execution"
                safe_commit(session, test_run_id, test_run)
                raise TestExecutionError(f"Test run {test_run_id} was cancelled")
            
            # Update status to running
            test_run.status = STATUS_RUNNING
            safe_commit(session, test_run_id, test_run)
            session.refresh(test_run)
            
            # Emit progress event: test started
            await progress_event_manager.emit_event(
                test_run_id,
                "test_started",
                {
                    "status": STATUS_RUNNING,
                    "test_type": test_run.test_type,
                    "doctor_provider": test_run.doctor_provider,
                    "doctor_model": test_run.doctor_model,
                    "patient_provider": test_run.patient_provider,
                    "patient_model": test_run.patient_model,
                    "start_time": start_time.isoformat(),
                },
                f"Test run {test_run_id} started"
            )
            
            # Start periodic progress updates with elapsed time
            progress_task = None
            async def emit_periodic_progress():
                """Emit periodic progress updates to show test is still running."""
                try:
                    while True:
                        await asyncio.sleep(5)  # Every 5 seconds
                        # Check if test is still running
                        check_session = get_session()
                        try:
                            check_run = check_session.query(TestRun).filter(TestRun.id == test_run_id).first()
                            if not check_run or check_run.status != STATUS_RUNNING:
                                break
                            if cancellation_manager.is_cancelled(test_run_id):
                                break
                            
                            # Emit heartbeat with elapsed time
                            elapsed = (datetime.utcnow() - start_time).total_seconds()
                            await progress_event_manager.emit_event(
                                test_run_id,
                                "test_progress",
                                {
                                    "elapsed_seconds": elapsed,
                                    "status": "running",
                                },
                                f"Test running... ({int(elapsed)}s elapsed)"
                            )
                        finally:
                            check_session.close()
                except asyncio.CancelledError:
                    pass
            
            # Start periodic progress task
            import asyncio
            progress_task = asyncio.create_task(emit_periodic_progress())
            
            try:
                # Get providers and create models
                doctor_provider_instance = get_provider(test_run.doctor_provider)
                patient_provider_instance = get_provider(test_run.patient_provider)
                
                doctor_model_instance = doctor_provider_instance.create_model(test_run.doctor_model)
                patient_model_instance = patient_provider_instance.create_model(test_run.patient_model)
                
                # Ensure Ollama models are available (auto-pull if missing)
                from vivasecuris.aiasylum.models.ollama import OllamaModel
                if test_run.doctor_provider == "ollama" and isinstance(doctor_model_instance, OllamaModel):
                    if not await doctor_model_instance.check_available():
                        logger.info(f"Ollama model {test_run.doctor_model} not found, pulling...")
                        await doctor_model_instance.pull_model()
                if test_run.patient_provider == "ollama" and isinstance(patient_model_instance, OllamaModel):
                    if not await patient_model_instance.check_available():
                        logger.info(f"Ollama model {test_run.patient_model} not found, pulling...")
                        await patient_model_instance.pull_model()
                
                # Get test config from metadata if available
                test_config = test_run.meta_data.get("test_config") if test_run.meta_data else {}
                
                # Also check top-level metadata for benchmark info (from API route)
                if test_run.meta_data and "benchmark" in test_run.meta_data:
                    benchmark_name = test_run.meta_data.get("benchmark")
                    num_samples = test_run.meta_data.get("num_samples", 100)
                    test_config["benchmark_name"] = benchmark_name
                    test_config["num_samples"] = num_samples
                
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
                        # Substitute variables if provided
                        if test_config.get("variables"):
                            prompt_text = substitute_variables(prompt_text, test_config["variables"])
                        # Increment usage count
                        prompt.usage_count = (prompt.usage_count or 0) + 1
                        session.commit()
                
                # Update test_config with system prompts
                if doctor_system_prompt:
                    test_config["doctor_system_prompt"] = doctor_system_prompt
                if patient_system_prompt:
                    test_config["patient_system_prompt"] = patient_system_prompt
                
                # Substitute variables in custom prompts if provided
                variables = test_config.get("variables", {})
                if variables:
                    if test_config.get("prompts"):
                        test_config["prompts"] = [
                            substitute_variables(p, variables) for p in test_config["prompts"]
                        ]
                    if test_config.get("prompt"):
                        test_config["prompt"] = substitute_variables(test_config["prompt"], variables)
                
                # Create callback to save conversation turns incrementally
                saved_turn_numbers = set()
                async def save_conversation_turn(turn: Dict):
                    """Save a conversation turn to database immediately."""
                    turn_number = turn.get("turn_number", len(saved_turn_numbers))
                    # Skip if already saved
                    if turn_number in saved_turn_numbers:
                        return
                    
                    # Use a new session for this operation to avoid conflicts
                    turn_session = get_session()
                    try:
                        speaker = turn.get("speaker", "unknown")
                        prompt = turn.get("prompt", "")
                        response = turn.get("response", "")
                        reasoning = turn.get("reasoning", "")
                        
                        # Store reasoning and patient info in metadata
                        turn_metadata = {}
                        if reasoning:
                            turn_metadata["reasoning"] = reasoning
                        # Add patient metadata for group therapy
                        if turn.get("patient_id") is not None:
                            turn_metadata["patient_id"] = turn.get("patient_id")
                        if turn.get("patient_name"):
                            turn_metadata["patient_name"] = turn.get("patient_name")
                        if turn.get("patient_model"):
                            turn_metadata["patient_model"] = turn.get("patient_model")
                        if turn.get("patient_provider"):
                            turn_metadata["patient_provider"] = turn.get("patient_provider")
                        
                        turn_record = ConversationTurn(
                            test_run_id=test_run.id,
                            turn_number=turn_number,
                            speaker=speaker,
                            prompt=prompt,
                            response=response,
                            meta_data=turn_metadata if turn_metadata else None,
                        )
                        turn_session.add(turn_record)
                        turn_session.commit()
                        saved_turn_numbers.add(turn_number)
                        
                        # Emit verbose progress event for new conversation turn
                        # Include prompt and response snippets for live monitoring
                        prompt_preview = prompt[:200] + "..." if len(prompt) > 200 else prompt
                        response_preview = response[:200] + "..." if len(response) > 200 else response
                        await progress_event_manager.emit_event(
                            test_run_id,
                            "conversation_turn",
                            {
                                "turn_number": turn_number,
                                "speaker": speaker,
                                "total_turns": len(saved_turn_numbers),
                                "prompt_preview": prompt_preview,
                                "response_preview": response_preview,
                                "prompt_length": len(prompt),
                                "response_length": len(response),
                            },
                            f"Turn {turn_number}: {speaker} - {response_preview}"
                        )
                        logger.debug(f"Saved conversation turn {turn_number} for test run {test_run_id}")
                    except Exception as e:
                        logger.error(f"Error saving conversation turn: {e}", exc_info=True)
                        turn_session.rollback()
                    finally:
                        turn_session.close()
                
                # Track max_turns for progress calculation (get from test_config or default)
                max_turns = test_config.get("max_turns", 10)
                test_config["max_turns"] = max_turns
                
                # Enhanced save callback that includes progress
                async def save_conversation_turn_with_progress(turn: Dict):
                    """Save conversation turn and emit progress."""
                    await save_conversation_turn(turn)
                    # Emit progress update with turn count
                    turn_number = turn.get("turn_number", 0)
                    progress_pct = int((turn_number / max_turns * 100)) if max_turns > 0 else 0
                    await progress_event_manager.emit_event(
                        test_run_id,
                        "conversation_progress",
                        {
                            "turn_number": turn_number,
                            "max_turns": max_turns,
                            "progress": progress_pct,
                            "speaker": turn.get("speaker"),
                            "total_turns": len(saved_turn_numbers),
                            "prompt_preview": turn.get("prompt", "")[:100] + "..." if len(turn.get("prompt", "")) > 100 else turn.get("prompt", ""),
                            "response_preview": turn.get("response", "")[:100] + "..." if len(turn.get("response", "")) > 100 else turn.get("response", ""),
                        },
                        f"Turn {turn_number}/{max_turns} ({progress_pct}%): {turn.get('speaker', 'unknown')} responded"
                    )
                
                # Add enhanced callback to test config
                test_config["save_conversation_turn_callback"] = save_conversation_turn_with_progress
                
                # Add cancellation/pause check callback
                def check_cancellation():
                    """Check if test run is cancelled or paused and raise exception if so."""
                    if cancellation_manager.is_cancelled(test_run_id):
                        # Check if it's a pause
                        session_check = get_session()
                        try:
                            test_run_check = session_check.query(TestRun).filter(TestRun.id == test_run_id).first()
                            if test_run_check and test_run_check.status == STATUS_PAUSED:
                                logger.info(f"⏸️ Test run {test_run_id} pause detected, stopping execution")
                                print(f"[PAUSE] Test run {test_run_id} pause detected, raising exception")
                                raise TestExecutionError(f"Test run {test_run_id} was paused", pause=True)
                        finally:
                            session_check.close()
                        
                        logger.warning(f"🛑 Test run {test_run_id} cancellation detected, stopping execution")
                        print(f"[CANCELLATION] Test run {test_run_id} cancellation detected, raising exception")
                        raise TestExecutionError(f"Test run {test_run_id} was cancelled")
                
                test_config["check_cancellation"] = check_cancellation
                
                # Run appropriate test
                test_result: TestResultType
                print(f"[execute_test_run] Test type: {test_run.test_type}, TEST_TYPE_BENCHMARK: {TEST_TYPE_BENCHMARK}")
                
                # Check if this is a benchmark (by test_type or metadata)
                is_benchmark = (
                    test_run.test_type == TEST_TYPE_BENCHMARK or
                    test_config.get("benchmark_name") or
                    (test_run.meta_data and test_run.meta_data.get("benchmark"))
                )
                
                if is_benchmark:
                    # Benchmark test - load dataset and evaluate
                    benchmark_name = test_config.get("benchmark_name")
                    if not benchmark_name and test_run.meta_data:
                        benchmark_name = test_run.meta_data.get("benchmark")
                    
                    if not benchmark_name:
                        raise TestExecutionError("Benchmark name not specified")
                    
                    # Update test_type if it was wrong
                    if test_run.test_type != TEST_TYPE_BENCHMARK:
                        suite_id_str = str(suite_id) if suite_id else "N/A"
                        logger.info(f"🔧 Fixing test_type from '{test_run.test_type}' to '{TEST_TYPE_BENCHMARK}' [Suite #{suite_id_str}] [Test #{test_run_id}]")
                        test_run.test_type = TEST_TYPE_BENCHMARK
                        session.commit()
                        session.refresh(test_run)
                    
                    num_samples = test_config.get("num_samples")
                    if not num_samples and test_run.meta_data:
                        num_samples = test_run.meta_data.get("num_samples")
                    
                    # For jailbreak benchmarks, default to one_shot mode
                    # Individual prompts will be handled based on their is_multi_shot flag
                    # This allows single-shot and multi-shot prompts to be mixed
                    if benchmark_name.lower() == "jailbreak":
                        test_mode = test_config.get("test_mode", "one_shot")  # Default to one_shot for jailbreaks
                    else:
                        test_mode = test_config.get("test_mode", "one_shot")  # one_shot or multi_shot
                    
                    # Check for manually selected indices or subject
                    selected_indices = None
                    selected_subject = None
                    if test_run.meta_data:
                        # Prefer selected_indices_list (already parsed) over selected_indices (string)
                        indices_list = test_run.meta_data.get("selected_indices_list")
                        if indices_list and isinstance(indices_list, list):
                            selected_indices = indices_list
                        else:
                            # Fallback to parsing string format
                            indices_data = test_run.meta_data.get("selected_indices")
                            if indices_data:
                                if isinstance(indices_data, str):
                                    # Parse string format
                                    from vivasecuris.aiasylum.benchmarks.datasets import parse_index_selection
                                    # We need to know max index, but we'll load all first to get it
                                    # For now, assume a reasonable max (will be validated when loading)
                                    try:
                                        selected_indices = parse_index_selection(indices_data, 100000)  # Large max, will be validated
                                    except:
                                        pass
                                elif isinstance(indices_data, list):
                                    selected_indices = indices_data
                        selected_subject = test_run.meta_data.get("selected_subject")
                    
                    suite_id_str = str(suite_id) if suite_id else "N/A"
                    logger.info(f"📊 Running benchmark '{benchmark_name}' with {num_samples} samples (mode: {test_mode}) [Suite #{suite_id_str}] [Test #{test_run_id}]")
                    if selected_indices:
                        logger.info(f"   Using manually selected {len(selected_indices)} indices [Suite #{suite_id_str}] [Test #{test_run_id}]")
                    if selected_subject:
                        logger.info(f"   Filtering by subject: {selected_subject} [Suite #{suite_id_str}] [Test #{test_run_id}]")
                    
                    # Add test_run_id to context for unique randomization
                    test_config["test_run_id"] = test_run_id
                    
                    test = BenchmarkTest(
                        name=f"benchmark_{benchmark_name}",
                        benchmark_name=benchmark_name,
                        num_samples=num_samples,
                        test_mode=test_mode,
                        selected_indices=selected_indices,
                        selected_subject=selected_subject,
                    )
                    
                    # Log periodic updates for long-running benchmarks
                    suite_id_str = str(suite_id) if suite_id else "N/A"
                    logger.info(f"⏳ Benchmark test starting (this may take a while for {num_samples} samples)... [Suite #{suite_id_str}] [Test #{test_run_id}]")
                    
                    # Emit progress event: benchmark started
                    await progress_event_manager.emit_event(
                        test_run_id,
                        "benchmark_started",
                        {
                            "benchmark_name": benchmark_name,
                            "num_samples": num_samples,
                            "test_mode": test_mode,
                        },
                        f"Starting benchmark '{benchmark_name}' with {num_samples} samples"
                    )
                    
                    # Add progress callback to context for benchmark tests
                    async def emit_progress(current: int, total: int, message: str = None):
                        """Emit progress update during benchmark execution."""
                        progress_pct = int((current / total * 100)) if total > 0 else 0
                        await progress_event_manager.emit_event(
                            test_run_id,
                            "benchmark_progress",
                            {
                                "benchmark_name": benchmark_name,
                                "current": current,
                                "total": total,
                                "progress": progress_pct,
                            },
                            message or f"Processing question {current}/{total} ({progress_pct}%)"
                        )
                    
                    # Add progress callback to test config
                    test_config["progress_callback"] = emit_progress
                    
                    # Save each benchmark Q&A as a conversation turn so the conversation tab and analysis have data
                    benchmark_saved_turn_numbers = set()
                    async def save_benchmark_turn(turn: Dict):
                        """Save a benchmark turn to DB immediately for viewing and analysis."""
                        turn_number = turn.get("turn_number", turn.get("prompt_number", len(benchmark_saved_turn_numbers)))
                        if turn_number in benchmark_saved_turn_numbers:
                            return
                        turn_session = get_session()
                        try:
                            turn_record = ConversationTurn(
                                test_run_id=test_run.id,
                                turn_number=turn_number,
                                speaker=turn.get("speaker", "patient"),
                                prompt=turn.get("prompt", ""),
                                response=turn.get("response", ""),
                                meta_data={"reasoning": turn.get("reasoning", "")} if turn.get("reasoning") else None,
                            )
                            turn_session.add(turn_record)
                            turn_session.commit()
                            benchmark_saved_turn_numbers.add(turn_number)
                        except Exception as e:
                            logger.error(f"Error saving benchmark conversation turn: {e}", exc_info=True)
                            turn_session.rollback()
                        finally:
                            turn_session.close()
                    test_config["save_conversation_turn_callback"] = save_benchmark_turn
                    
                    test_result = await test.run(patient_model_instance, doctor_model_instance, context=test_config)
                    
                    elapsed = (datetime.utcnow() - start_time).total_seconds()
                    logger.info(f"✅ Benchmark '{benchmark_name}' completed: score={test_result.score:.4f} ({elapsed:.1f}s elapsed) [Suite #{suite_id_str}] [Test #{test_run_id}]")
                    
                    # Emit progress event: benchmark completed
                    await progress_event_manager.emit_event(
                        test_run_id,
                        "benchmark_progress",
                        {
                            "benchmark_name": benchmark_name,
                            "progress": 100,
                            "score": test_result.score,
                            "elapsed_seconds": elapsed,
                        },
                        f"Benchmark '{benchmark_name}' completed"
                    )
                elif test_run.test_type == TEST_TYPE_ONE_SHOT:
                    # One-shot test - single prompt/response
                    if prompt_text:
                        test = OneShotTest(prompts=[prompt_text])
                    else:
                        prompts = test_config.get("prompts", [])
                        if not prompts:
                            # Fallback to single prompt if provided
                            single_prompt = test_config.get("prompt")
                            if single_prompt:
                                prompts = [single_prompt]
                        test = OneShotTest(prompts=prompts)
                    
                    # Get total number of prompts for progress tracking
                    total_prompts = len(test.prompts) if test.prompts else 1
                    
                    # Add progress callback to context for one-shot tests
                    async def emit_one_shot_progress(current: int, total: int, message: str = None):
                        """Emit progress update during one-shot test execution."""
                        progress_pct = int((current / total * 100)) if total > 0 else 0
                        await progress_event_manager.emit_event(
                            test_run_id,
                            "test_progress",
                            {
                                "test_type": "one_shot",
                                "current": current,
                                "total": total,
                                "progress": progress_pct,
                                "attempt": current,
                                "total_attempts": total,
                            },
                            message or f"Processing attempt {current}/{total} ({progress_pct}%)"
                        )
                    
                    # Add progress callback to test config
                    test_config["progress_callback"] = emit_one_shot_progress
                    
                    test_result = await test.run(patient_model_instance, doctor_model_instance, context=test_config)
                elif test_run.test_type == TEST_TYPE_MULTI_SHOT:
                    # Multi-shot test - multiple sequential prompts to test context handling
                    if prompt_text:
                        test = MultiShotTest(prompts=[prompt_text])
                    else:
                        prompts = test_config.get("prompts", [])
                        num_messages = test_config.get("num_messages", 10)
                        if prompts:
                            test = MultiShotTest(prompts=prompts)
                        else:
                            test = MultiShotTest(num_messages=num_messages)
                    
                    # Get total number of prompts for progress tracking
                    total_prompts = len(test.prompts) if test.prompts else test.num_messages
                    
                    # Add progress callback to context for multi-shot tests
                    async def emit_multi_shot_progress(current: int, total: int, message: str = None):
                        """Emit progress update during multi-shot test execution."""
                        progress_pct = int((current / total * 100)) if total > 0 else 0
                        await progress_event_manager.emit_event(
                            test_run_id,
                            "test_progress",
                            {
                                "test_type": "multi_shot",
                                "current": current,
                                "total": total,
                                "progress": progress_pct,
                                "attempt": current,
                                "total_attempts": total,
                            },
                            message or f"Processing attempt {current}/{total} ({progress_pct}%)"
                        )
                    
                    # Add progress callback to test config
                    test_config["progress_callback"] = emit_multi_shot_progress
                    
                    test_result = await test.run(patient_model_instance, doctor_model_instance, context=test_config)
                elif test_run.test_type == TEST_TYPE_CONVERSATION:
                    # Conversation test - multi-turn conversation between doctor and patient
                    max_turns = test_config.get("max_turns", 10)
                    doctor_prompt = prompt_text or test_config.get("doctor_prompt")
                    test = ConversationTest(max_turns=max_turns, doctor_prompt=doctor_prompt)
                    test_result = await test.run(patient_model_instance, doctor_model_instance, context=test_config)
                elif test_run.test_type == TEST_TYPE_GROUP_THERAPY:
                    # Group therapy test - multiple patient models in a group session
                    max_turns = test_config.get("max_turns", 10)
                    doctor_prompt = prompt_text or test_config.get("doctor_prompt")
                    
                    # Get patient models from test_config or metadata
                    patients_config = test_config.get("patients", [])
                    if not patients_config and test_run.meta_data:
                        patients_config = test_run.meta_data.get("patients", [])
                    if not patients_config:
                        # Fallback to single patient if patients array not provided
                        patients_config = [{"provider": test_run.patient_provider, "model": test_run.patient_model}]
                    
                    # Create patient model instances
                    patient_models = []
                    patient_info_list = []
                    patient_system_prompts = {}
                    
                    for i, patient_cfg in enumerate(patients_config):
                        if isinstance(patient_cfg, dict):
                            p_provider = patient_cfg.get("provider", test_run.patient_provider)
                            p_model = patient_cfg.get("model", test_run.patient_model)
                        else:
                            # Handle legacy format
                            p_provider = test_run.patient_provider
                            p_model = test_run.patient_model
                        
                        # Get provider and create model
                        p_provider_instance = get_provider(p_provider)
                        p_model_instance = p_provider_instance.create_model(p_model)
                        patient_models.append(p_model_instance)
                        
                        # Store patient info
                        patient_info_list.append({
                            "id": i,
                            "provider": p_provider,
                            "model": p_model,
                        })
                        
                        # Load patient system prompt if specified
                        if isinstance(patient_cfg, dict):
                            patient_system_prompt_id = patient_cfg.get("system_prompt_id")
                            if patient_system_prompt_id:
                                p_prompt = session.query(PromptLibrary).filter(
                                    PromptLibrary.id == patient_system_prompt_id,
                                    PromptLibrary.prompt_type == "system_prompt",
                                    PromptLibrary.target == "patient"
                                ).first()
                                if p_prompt:
                                    patient_system_prompts[i] = p_prompt.prompt_text
                                    p_prompt.usage_count = (p_prompt.usage_count or 0) + 1
                                    session.commit()
                    
                    # Store patient list in test_run metadata if not already there
                    if not test_run.meta_data:
                        test_run.meta_data = {}
                    if "patients" not in test_run.meta_data:
                        test_run.meta_data["patients"] = patient_info_list
                        safe_commit(session, test_run_id, test_run)
                    
                    # Add patient system prompts and patient info to context
                    if patient_system_prompts:
                        test_config["patient_system_prompts"] = patient_system_prompts
                    test_config["patient_info"] = patient_info_list
                    
                    test = GroupTherapyTest(max_turns=max_turns, doctor_prompt=doctor_prompt)
                    test_result = await test.run(patient_models, doctor_model_instance, context=test_config)
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
                
                # Save conversation turns if available (only if not already saved incrementally)
                if test_result.metadata and "conversation_history" in test_result.metadata:
                    conversation_history = test_result.metadata["conversation_history"]
                    # Check if turns were already saved incrementally
                    existing_turns = session.query(ConversationTurn).filter(
                        ConversationTurn.test_run_id == test_run.id
                    ).count()
                    
                    if existing_turns == 0:
                        # No turns saved yet, save them all now
                        print(f"[execute_test_run] Saving {len(conversation_history)} conversation turns for test_run_id={test_run.id}")
                        for i, turn in enumerate(conversation_history):
                            speaker = turn.get("speaker", "unknown")
                            prompt = turn.get("prompt", "")
                            response = turn.get("response", "")
                            reasoning = turn.get("reasoning", "")
                            print(f"  Turn {i}: speaker={speaker}, prompt_length={len(prompt)}, response_length={len(response)}, reasoning_length={len(reasoning)}")
                            # Store reasoning and patient info in metadata
                            turn_metadata = {}
                            if reasoning:
                                turn_metadata["reasoning"] = reasoning
                            # Add patient metadata for group therapy
                            if turn.get("patient_id") is not None:
                                turn_metadata["patient_id"] = turn.get("patient_id")
                            if turn.get("patient_name"):
                                turn_metadata["patient_name"] = turn.get("patient_name")
                            if turn.get("patient_model"):
                                turn_metadata["patient_model"] = turn.get("patient_model")
                            if turn.get("patient_provider"):
                                turn_metadata["patient_provider"] = turn.get("patient_provider")
                            turn_record = ConversationTurn(
                                test_run_id=test_run.id,
                                turn_number=i,
                                speaker=speaker,
                                prompt=prompt,
                                response=response,
                                meta_data=turn_metadata if turn_metadata else None,
                            )
                            session.add(turn_record)
                    else:
                        print(f"[execute_test_run] {existing_turns} conversation turns already saved incrementally for test_run_id={test_run.id}, skipping batch save")
                else:
                    print(f"[execute_test_run] No conversation_history in metadata for test_run_id={test_run.id}")
                    if test_result.metadata:
                        print(f"  Available metadata keys: {list(test_result.metadata.keys())}")
                
                # Stop periodic progress updates
                if 'progress_task' in locals():
                    progress_task.cancel()
                    try:
                        await progress_task
                    except asyncio.CancelledError:
                        pass
                
                # Update test run status
                test_run.status = "completed"
                safe_commit(session, test_run_id, test_run)
                
                # Check if auto-analysis is enabled
                auto_analysis_enabled = False
                analysis_config = {}
                if test_run.meta_data and test_run.meta_data.get("test_config"):
                    test_config = test_run.meta_data.get("test_config", {})
                    if test_config.get("auto_analysis"):
                        auto_analysis_enabled = True
                        analysis_config = test_config.get("analysis_config", {})
                        logger.info(f"Auto-analysis enabled for test run {test_run_id} with config: {analysis_config}")
                
                # Emit progress event: test completed
                elapsed = (datetime.utcnow() - start_time).total_seconds()
                await progress_event_manager.emit_event(
                    test_run_id,
                    "test_completed",
                    {
                        "status": "completed",
                        "elapsed_seconds": elapsed,
                        "progress": 100,
                        "score": test_result.score if hasattr(test_result, 'score') else None,
                        "auto_analysis_enabled": auto_analysis_enabled,
                    },
                    f"Test run {test_run_id} completed successfully"
                )
                
                # Trigger auto-analysis if enabled
                if auto_analysis_enabled:
                    try:
                        logger.info(f"Triggering auto-analysis for test run {test_run_id}")
                        from vivasecuris.aiasylum.api.routes.analysis import _run_analysis_background
                        import asyncio
                        
                        # Create analysis test run entry (same as manual analysis endpoint)
                        analysis_session = get_session()
                        try:
                            # Use evaluator model/provider if specified, otherwise use doctor model
                            evaluator_provider = analysis_config.get("evaluator_provider")
                            evaluator_model = analysis_config.get("evaluator_model")
                            analysis_doctor_provider = evaluator_provider or test_run.doctor_provider
                            analysis_doctor_model = evaluator_model or test_run.doctor_model
                            
                            analysis_test_run = TestRun(
                                doctor_provider=analysis_doctor_provider,
                                doctor_model=analysis_doctor_model,
                                patient_provider=test_run.patient_provider,
                                patient_model=test_run.patient_model,
                                test_type=TEST_TYPE_ANALYSIS,
                                status=STATUS_PENDING,
                                meta_data={
                                    "source_test_run_id": test_run_id,
                                    "analysis_config": {
                                        "enable_activation_patching": False,
                                        "enable_cot_detection": analysis_config.get("enable_cot_detection", True),
                                        "cot_analysis_mode": analysis_config.get("cot_analysis_mode", "full"),
                                        "enable_factuality_check": analysis_config.get("enable_factuality_check", False),
                                        "enable_manipulation_analysis": analysis_config.get("enable_manipulation_analysis", False),
                                        "evaluator_provider": evaluator_provider,
                                        "evaluator_model": evaluator_model,
                                    },
                                    "description": f"Auto-analysis of test run #{test_run_id}",
                                },
                            )
                            analysis_session.add(analysis_test_run)
                            analysis_session.commit()
                            analysis_session.refresh(analysis_test_run)
                            analysis_test_run_id = analysis_test_run.id
                            logger.info(f"Created analysis test run {analysis_test_run_id} for auto-analysis of test run {test_run_id}")
                        except Exception as e:
                            analysis_session.rollback()
                            logger.error(f"Failed to create analysis test run entry for test run {test_run_id}: {str(e)}", exc_info=True)
                            raise
                        finally:
                            analysis_session.close()
                        
                        # Start analysis as a background task using the same function as manual analysis
                        asyncio.create_task(_run_analysis_background(
                            test_run_id=test_run_id,
                            enable_activation_patching=False,
                            enable_cot_detection=analysis_config.get("enable_cot_detection", True),
                            cot_analysis_mode=analysis_config.get("cot_analysis_mode", "full"),
                            enable_factuality_check=analysis_config.get("enable_factuality_check", False),
                            enable_manipulation_analysis=analysis_config.get("enable_manipulation_analysis", False),
                            evaluator_provider=evaluator_provider,
                            evaluator_model=evaluator_model,
                            analysis_test_run_id=analysis_test_run_id,
                        ))
                        logger.info(f"Auto-analysis task created for test run {test_run_id}, analysis test run {analysis_test_run_id}")
                    except Exception as e:
                        logger.error(f"Failed to trigger auto-analysis for test run {test_run_id}: {str(e)}", exc_info=True)
                        # Don't fail the test run if analysis fails
                
                return test_run
            
            except TestExecutionError as e:
                # Stop periodic progress updates
                if progress_task:
                    progress_task.cancel()
                    try:
                        await progress_task
                    except asyncio.CancelledError:
                        pass
                
                # Check if this was a cancellation
                is_cancelled = cancellation_manager.is_cancelled(test_run_id) or "cancelled" in str(e).lower()
                
                logger.info(f"TestExecutionError caught for test run {test_run_id}: {e}, is_cancelled: {is_cancelled}")
                print(f"[EXECUTE_TEST_RUN] TestExecutionError: {e}, is_cancelled: {is_cancelled}")
                
                test_run.status = "failed"
                if not test_run.meta_data:
                    test_run.meta_data = {}
                if is_cancelled:
                    test_run.meta_data["cancelled"] = True
                    test_run.meta_data["error"] = "Test run was cancelled"
                    logger.info(f"✅ Test run {test_run_id} successfully cancelled")
                    print(f"[EXECUTE_TEST_RUN] Test run {test_run_id} marked as cancelled")
                else:
                    test_run.meta_data["error"] = str(e)
                safe_commit(session, test_run_id, test_run)
                
                # Emit progress event: test failed or cancelled
                elapsed = (datetime.utcnow() - start_time).total_seconds()
                event_type = "test_cancelled" if is_cancelled else "test_failed"
                await progress_event_manager.emit_event(
                    test_run_id,
                    event_type,
                    {
                        "status": "failed",
                        "cancelled": is_cancelled,
                        "error": str(e),
                        "elapsed_seconds": elapsed,
                    },
                    f"Test run {test_run_id} {'cancelled' if is_cancelled else 'failed'}: {str(e)}"
                )
                
                # Clear cancellation flag
                cancellation_manager.clear(test_run_id)
                
                # Don't re-raise if it was a cancellation (expected behavior)
                if not is_cancelled:
                    raise
                return test_run
            except Exception as e:
                # Stop periodic progress updates
                if progress_task:
                    progress_task.cancel()
                    try:
                        await progress_task
                    except asyncio.CancelledError:
                        pass
                
                # Get detailed error message
                error_msg = str(e)
                error_details = error_msg
                
                # For jailbreak benchmarks, provide helpful error message
                if "jailbreak" in error_msg.lower() or (test_run.test_type == TEST_TYPE_BENCHMARK and test_run.meta_data and test_run.meta_data.get("benchmark", "").lower() == "jailbreak"):
                    if "no jailbreak prompts" in error_msg.lower() or "no dataset loaded" in error_msg.lower():
                        error_details = (
                            "Jailbreak benchmark failed: No prompts found in database.\n\n"
                            "To fix:\n"
                            "1. Run: python scripts/import_jailbreaks.py\n"
                            "2. Verify prompts were imported\n"
                            "3. Check that prompts have category='adversarial' and 'jailbreak' tag\n\n"
                            f"Original error: {error_msg}"
                        )
                
                test_run.status = "failed"
                if not test_run.meta_data:
                    test_run.meta_data = {}
                test_run.meta_data["error"] = error_details
                safe_commit(session, test_run_id, test_run)
                
                # Log detailed error
                logger.error(f"❌ Test run {test_run_id} failed: {error_details}", exc_info=True)
                print(f"[TestRunner] ❌ Test run {test_run_id} failed: {error_details}")
                
                # Emit progress event: test failed
                elapsed = (datetime.utcnow() - start_time).total_seconds()
                await progress_event_manager.emit_event(
                    test_run_id,
                    "test_failed",
                    {
                        "status": "failed",
                        "error": error_details,
                        "elapsed_seconds": elapsed,
                    },
                    f"Test run {test_run_id} failed: {error_details[:200]}"
                )
                
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
