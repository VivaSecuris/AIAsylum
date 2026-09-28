"""Test suite runner for creating and managing test suites."""

from datetime import datetime
from typing import Dict, List, Optional

from vivasecuris.aiasylum.database import get_session, TestRun, TestSuite
from vivasecuris.aiasylum.constants import STATUS_PENDING, TEST_TYPE_BENCHMARK, TEST_TYPE_GROUP_THERAPY
from vivasecuris.aiasylum.suites.progress_tracker import ProgressTracker


class SuiteRunner:
    """Create and manage test suites."""

    def create_suite(
        self,
        name: Optional[str],
        test_types: List[str],
        benchmarks: List[str],
        models: List[Dict[str, str]],
        test_config: Optional[Dict] = None,
        num_samples: Optional[int] = None,
        doctor: Optional[Dict[str, str]] = None,
    ) -> TestSuite:
        """Create a test suite with all test run combinations.

        ``doctor`` interviews and assesses every model; without one each model is
        its own doctor (the original behaviour). Group therapy is one session with
        every model as a patient, sharing the suite's patient system prompt.
        """
        session = get_session()
        try:
            from vivasecuris.aiasylum.runner.run_config import validate_patient_prompt_selection
            for test_type in test_types:
                validate_patient_prompt_selection(session, test_type, test_config)
            # Create suite record
            suite = TestSuite(
                name=name,
                status="pending",
                total_runs=0,
                completed_runs=0,
                failed_runs=0,
                running_runs=0,
                pending_runs=0,
                meta_data={
                    "test_types": test_types,
                    "benchmarks": benchmarks,
                    "models": models,
                    "test_config": test_config or {},
                    "num_samples": num_samples,
                },
            )
            session.add(suite)
            session.commit()
            session.refresh(suite)

            # Create test runs for all combinations
            runs: List[TestRun] = []

            def add_run(doctor_model: Dict[str, str], patient: Dict[str, str], test_type: str, config: Dict) -> None:
                run = TestRun(
                    doctor_provider=doctor_model["provider"],
                    doctor_model=doctor_model["model"],
                    patient_provider=patient["provider"],
                    patient_model=patient["model"],
                    test_type=test_type,
                    status=STATUS_PENDING,
                    suite_id=suite.id,
                    meta_data={"test_config": config, "suite_id": suite.id},
                )
                session.add(run)
                runs.append(run)

            # For each test type + model combination
            for test_type in test_types:
                # Skip "benchmark" test type here - benchmarks are handled separately
                if test_type == TEST_TYPE_BENCHMARK:
                    continue
                base_config = {**(test_config or {}), "suite_id": suite.id}
                if test_type == TEST_TYPE_GROUP_THERAPY:
                    shared_prompt = {
                        key: value for key, value in (
                            ("system_prompt_id", base_config.get("patient_system_prompt_id")),
                            ("system_prompt", base_config.get("patient_system_prompt")),
                        ) if value
                    }
                    patients = [{"provider": m["provider"], "model": m["model"], **shared_prompt} for m in models]
                    add_run(doctor or models[0], models[0], test_type, {**base_config, "patients": patients})
                    continue
                for model in models:
                    add_run(doctor or model, model, test_type, dict(base_config))

            # For each benchmark + model combination
            for benchmark in benchmarks:
                for model in models:
                    # Jailbreak benchmarks default to one_shot mode
                    # Individual prompts will be handled based on their is_multi_shot flag
                    # This allows single-shot and multi-shot prompts to be mixed properly
                    test_mode = "one_shot"  # Default for all benchmarks, including jailbreak
                    test_run = TestRun(
                        doctor_provider=model["provider"],
                        doctor_model=model["model"],
                        patient_provider=model["provider"],
                        patient_model=model["model"],
                        test_type=TEST_TYPE_BENCHMARK,
                        status=STATUS_PENDING,
                        suite_id=suite.id,
                        meta_data={
                            "benchmark": benchmark,
                            "num_samples": num_samples,
                            "test_config": {
                                **(test_config or {}),
                                "benchmark_name": benchmark,
                                "num_samples": num_samples or 100,
                                "test_mode": test_mode,
                            },
                            "suite_id": suite.id,
                        },
                    )
                    session.add(test_run)
                    runs.append(test_run)

            session.flush()  # assigns the IDs
            test_run_ids = [run.id for run in runs]
            session.commit()

            # Update suite total_runs
            suite.total_runs = len(test_run_ids)
            suite.pending_runs = len(test_run_ids)
            suite.meta_data["test_run_ids"] = test_run_ids
            session.commit()
            session.refresh(suite)

            return suite
        finally:
            session.close()

    def get_suite(self, suite_id: int) -> Optional[TestSuite]:
        """Get a test suite by ID."""
        session = get_session()
        try:
            return session.query(TestSuite).filter(TestSuite.id == suite_id).first()
        finally:
            session.close()

    def list_suites(self, limit: int = 100, offset: int = 0) -> List[TestSuite]:
        """List test suites."""
        session = get_session()
        try:
            return (
                session.query(TestSuite)
                .order_by(TestSuite.created_at.desc())
                .limit(limit)
                .offset(offset)
                .all()
            )
        finally:
            session.close()

    def delete_suite(self, suite_id: int) -> bool:
        """Delete a test suite and all its test runs (cascade deletes test runs and their results)."""
        session = get_session()
        try:
            suite = session.query(TestSuite).filter(TestSuite.id == suite_id).first()
            if not suite:
                return False
            # Cascade deletes test_runs and each run's results, conversation_turns, assessments
            session.delete(suite)
            session.commit()
            return True
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_suite_runs(self, suite_id: int) -> List[TestRun]:
        """Get all test runs in a suite."""
        session = get_session()
        try:
            return (
                session.query(TestRun)
                .filter(TestRun.suite_id == suite_id)
                .order_by(TestRun.created_at)
                .all()
            )
        finally:
            session.close()
