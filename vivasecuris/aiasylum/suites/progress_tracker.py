"""Progress tracking for test suites."""

from datetime import datetime
from typing import Dict, List, Optional

from vivasecuris.aiasylum.database import get_session, TestRun, TestSuite
from vivasecuris.aiasylum.constants import (
    STATUS_PENDING,
    STATUS_RUNNING,
    STATUS_COMPLETED,
    STATUS_FAILED,
)


class ProgressTracker:
    """Track and calculate progress for test suites."""

    @staticmethod
    def update_suite_progress(suite_id: int) -> None:
        """Update progress counters for a test suite."""
        session = get_session()
        try:
            suite = session.query(TestSuite).filter(TestSuite.id == suite_id).first()
            if not suite:
                return

            # Get all test runs in this suite
            test_runs = session.query(TestRun).filter(TestRun.suite_id == suite_id).all()

            # Count by status
            total = len(test_runs)
            pending = sum(1 for tr in test_runs if tr.status == STATUS_PENDING)
            running = sum(1 for tr in test_runs if tr.status == STATUS_RUNNING)
            completed = sum(1 for tr in test_runs if tr.status == STATUS_COMPLETED)
            failed = sum(1 for tr in test_runs if tr.status == STATUS_FAILED)

            # Update suite counters
            suite.total_runs = total
            suite.pending_runs = pending
            suite.running_runs = running
            suite.completed_runs = completed
            suite.failed_runs = failed

            # Update started_at if first test run started
            if running > 0 or completed > 0 or failed > 0:
                if not suite.started_at:
                    suite.started_at = datetime.utcnow()

            # Update suite status
            if total == 0:
                suite.status = "pending"
            elif running > 0:
                suite.status = "running"
            elif completed == total:
                suite.status = "completed"
                if not suite.completed_at:
                    suite.completed_at = datetime.utcnow()
            elif failed == total:
                suite.status = "failed"
                if not suite.completed_at:
                    suite.completed_at = datetime.utcnow()
            elif failed > 0:
                suite.status = "partially_failed"
                if completed + failed == total and not suite.completed_at:
                    suite.completed_at = datetime.utcnow()
            else:
                suite.status = "pending"

            suite.updated_at = datetime.utcnow()
            session.commit()
        finally:
            session.close()

    @staticmethod
    def get_progress_breakdown(suite_id: int) -> Dict:
        """Get detailed progress breakdown for a suite."""
        session = get_session()
        try:
            suite = session.query(TestSuite).filter(TestSuite.id == suite_id).first()
            if not suite:
                return {}

            test_runs = session.query(TestRun).filter(TestRun.suite_id == suite_id).all()

            # Calculate progress percentage
            progress_percentage = (
                (suite.completed_runs / suite.total_runs * 100)
                if suite.total_runs > 0
                else 0
            )

            # Get currently running test runs
            running_tests = [
                {
                    "id": tr.id,
                    "model": f"{tr.patient_provider}/{tr.patient_model}",
                    "test_type": tr.test_type,
                    "benchmark": tr.meta_data.get("benchmark") if tr.test_type == "benchmark" else None,
                    "started_at": tr.created_at.isoformat() if tr.created_at else None,
                }
                for tr in test_runs
                if tr.status == STATUS_RUNNING
            ]

            # Calculate time estimates
            elapsed_time = None
            estimated_remaining = None
            average_time_per_run = None

            if suite.started_at:
                elapsed_time = (datetime.utcnow() - suite.started_at).total_seconds()

            # Calculate average time from completed runs
            completed_runs = [tr for tr in test_runs if tr.status == STATUS_COMPLETED]
            if completed_runs and suite.started_at:
                # Estimate based on time since start and number completed
                if len(completed_runs) > 0:
                    time_per_completed = elapsed_time / len(completed_runs) if elapsed_time else None
                    if time_per_completed:
                        average_time_per_run = time_per_completed
                        remaining_runs = suite.total_runs - suite.completed_runs - suite.failed_runs
                        estimated_remaining = time_per_completed * remaining_runs if remaining_runs > 0 else 0

            # Progress by model
            progress_by_model: Dict[str, Dict] = {}
            for tr in test_runs:
                model_key = f"{tr.patient_provider}/{tr.patient_model}"
                if model_key not in progress_by_model:
                    progress_by_model[model_key] = {
                        "total": 0,
                        "completed": 0,
                        "failed": 0,
                        "running": 0,
                        "pending": 0,
                    }
                progress_by_model[model_key]["total"] += 1
                if tr.status == STATUS_COMPLETED:
                    progress_by_model[model_key]["completed"] += 1
                elif tr.status == STATUS_FAILED:
                    progress_by_model[model_key]["failed"] += 1
                elif tr.status == STATUS_RUNNING:
                    progress_by_model[model_key]["running"] += 1
                else:
                    progress_by_model[model_key]["pending"] += 1

            # Progress by test type
            progress_by_test_type: Dict[str, Dict] = {}
            for tr in test_runs:
                test_type = tr.meta_data.get("benchmark", tr.test_type) if tr.test_type == "benchmark" else tr.test_type
                if test_type not in progress_by_test_type:
                    progress_by_test_type[test_type] = {
                        "total": 0,
                        "completed": 0,
                        "failed": 0,
                        "running": 0,
                        "pending": 0,
                    }
                progress_by_test_type[test_type]["total"] += 1
                if tr.status == STATUS_COMPLETED:
                    progress_by_test_type[test_type]["completed"] += 1
                elif tr.status == STATUS_FAILED:
                    progress_by_test_type[test_type]["failed"] += 1
                elif tr.status == STATUS_RUNNING:
                    progress_by_test_type[test_type]["running"] += 1
                else:
                    progress_by_test_type[test_type]["pending"] += 1

            return {
                "progress_percentage": round(progress_percentage, 2),
                "total_runs": suite.total_runs,
                "completed_runs": suite.completed_runs,
                "failed_runs": suite.failed_runs,
                "running_runs": suite.running_runs,
                "pending_runs": suite.pending_runs,
                "running_tests": running_tests,
                "elapsed_time_seconds": elapsed_time,
                "estimated_remaining_seconds": estimated_remaining,
                "average_time_per_run_seconds": average_time_per_run,
                "progress_by_model": progress_by_model,
                "progress_by_test_type": progress_by_test_type,
            }
        finally:
            session.close()
