"""Test execution runner."""

from typing import Dict, List, Optional

from typing import Dict, List, Optional

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
                # Legacy test types (for backward compatibility)
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
                    print(f"[execute_test_run] Running benchmark test (detected from test_type or metadata)")
                    benchmark_name = test_config.get("benchmark_name")
                    if not benchmark_name and test_run.meta_data:
                        benchmark_name = test_run.meta_data.get("benchmark")
                    
                    if not benchmark_name:
                        raise TestExecutionError("Benchmark name not specified")
                    
                    # Update test_type if it was wrong
                    if test_run.test_type != TEST_TYPE_BENCHMARK:
                        print(f"[execute_test_run] Fixing test_type from '{test_run.test_type}' to '{TEST_TYPE_BENCHMARK}'")
                        test_run.test_type = TEST_TYPE_BENCHMARK
                        session.commit()
                        session.refresh(test_run)
                    
                    num_samples = test_config.get("num_samples")
                    if not num_samples and test_run.meta_data:
                        num_samples = test_run.meta_data.get("num_samples")
                    
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
                    
                    print(f"[execute_test_run] Benchmark: {benchmark_name}, samples: {num_samples}, mode: {test_mode}")
                    if selected_indices:
                        print(f"[execute_test_run] Using manually selected indices: {selected_indices}")
                    if selected_subject:
                        print(f"[execute_test_run] Filtering by subject: {selected_subject}")
                    
                    test = BenchmarkTest(
                        name=f"benchmark_{benchmark_name}",
                        benchmark_name=benchmark_name,
                        num_samples=num_samples,
                        test_mode=test_mode,
                        selected_indices=selected_indices,
                        selected_subject=selected_subject,
                    )
                    test_result = await test.run(patient_model_instance, doctor_model_instance, context=test_config)
                    print(f"[execute_test_run] Benchmark test completed: {test_result.test_name}, score: {test_result.score}")
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
                        session.commit()
                    
                    # Add patient system prompts and patient info to context
                    if patient_system_prompts:
                        test_config["patient_system_prompts"] = patient_system_prompts
                    test_config["patient_info"] = patient_info_list
                    
                    test = GroupTherapyTest(max_turns=max_turns, doctor_prompt=doctor_prompt)
                    test_result = await test.run(patient_models, doctor_model_instance, context=test_config)
                # Legacy test types (for backward compatibility)
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
                    conversation_history = test_result.metadata["conversation_history"]
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
                    print(f"[execute_test_run] No conversation_history in metadata for test_run_id={test_run.id}")
                    if test_result.metadata:
                        print(f"  Available metadata keys: {list(test_result.metadata.keys())}")
                
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
