"""Test suite runner for creating and managing test suites."""

from datetime import datetime
from typing import Dict, List, Optional

from vivasecuris.aiasylum.database import get_session, TestRun, TestSuite
from vivasecuris.aiasylum.constants import STATUS_PENDING, TEST_TYPE_BENCHMARK
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
    ) -> TestSuite:
        """Create a test suite with all test run combinations."""
        session = get_session()
        try:
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
            test_run_ids = []

            # For each test type + model combination
            for test_type in test_types:
                # Skip "benchmark" test type here - benchmarks are handled separately
                if test_type == TEST_TYPE_BENCHMARK:
                    continue
                    
                for model in models:
                    # Merge test_config with suite-level config
                    run_test_config = {
                        **(test_config or {}),
                        "suite_id": suite.id,
                    }
                    test_run = TestRun(
                        doctor_provider=model["provider"],
                        doctor_model=model["model"],
                        patient_provider=model["provider"],
                        patient_model=model["model"],
                        test_type=test_type,
                        status=STATUS_PENDING,
                        suite_id=suite.id,
                        meta_data={
                            "test_config": run_test_config,
                            "suite_id": suite.id,
                        },
                    )
                    session.add(test_run)
                    test_run_ids.append(test_run.id)

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
                                "benchmark_name": benchmark,
                                "num_samples": num_samples or 100,
                                "test_mode": test_mode,
                            },
                            "suite_id": suite.id,
                        },
                    )
                    session.add(test_run)
                    test_run_ids.append(test_run.id)

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
        """Delete a test suite and all its test runs."""
        session = get_session()
        try:
            suite = session.query(TestSuite).filter(TestSuite.id == suite_id).first()
            if not suite:
                return False

            # Delete all test runs (cascade should handle this, but explicit is better)
            test_runs = session.query(TestRun).filter(TestRun.suite_id == suite_id).all()
            for tr in test_runs:
                session.delete(tr)

            session.delete(suite)
            session.commit()
            return True
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
