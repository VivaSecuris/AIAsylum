"""Test suite routes."""

import logging
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

from vivasecuris.aiasylum.suites import SuiteRunner, ProgressTracker
from vivasecuris.aiasylum.database import get_session, TestRun, TestSuite
from vivasecuris.aiasylum.runner import TestRunner
from vivasecuris.aiasylum.constants import STATUS_PENDING

# Set up logging
logger = logging.getLogger(__name__)


router = APIRouter()


class SuiteRequest(BaseModel):
    """Test suite creation request."""
    name: Optional[str] = None
    test_types: List[str] = []
    benchmarks: List[str] = []
    models: List[dict]  # [{"provider": "ollama", "model": "llama3.2"}, ...]
    test_config: Optional[dict] = None
    num_samples: Optional[int] = None


class SuiteResponse(BaseModel):
    """Test suite response."""
    id: int
    name: Optional[str]
    status: str
    total_runs: int
    completed_runs: int
    failed_runs: int
    running_runs: int
    pending_runs: int
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime
    meta_data: dict

    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }


class SuiteProgressResponse(BaseModel):
    """Detailed progress response for a suite."""
    progress_percentage: float
    total_runs: int
    completed_runs: int
    failed_runs: int
    running_runs: int
    pending_runs: int
    running_tests: List[dict]
    elapsed_time_seconds: Optional[float]
    estimated_remaining_seconds: Optional[float]
    average_time_per_run_seconds: Optional[float]
    progress_by_model: dict
    progress_by_test_type: dict


async def _run_suite_test_background(test_run_id: int, suite_id: int):
    """Background task to run a test and update suite progress."""
    # Get test run details for logging
    session = get_session()
    try:
        test_run = session.query(TestRun).filter(TestRun.id == test_run_id).first()
        if test_run:
            test_info = f"{test_run.test_type} on {test_run.patient_provider}/{test_run.patient_model}"
            if test_run.meta_data and test_run.meta_data.get("benchmark"):
                test_info += f" (benchmark: {test_run.meta_data.get('benchmark')})"
            suite_id_str = str(suite_id) if suite_id else "N/A"
            logger.info(f"🚀 Starting test run #{test_run_id}: {test_info} [Suite #{suite_id_str}]")
        else:
            suite_id_str = str(suite_id) if suite_id else "N/A"
            logger.warning(f"⚠️  Test run #{test_run_id} not found [Suite #{suite_id_str}]")
    finally:
        session.close()
    
    runner = TestRunner()
    start_time = datetime.utcnow()
    try:
        suite_id_str = str(suite_id) if suite_id else "N/A"
        logger.info(f"▶️  Executing test run #{test_run_id}... [Suite #{suite_id_str}]")
        await runner.execute_test_run(test_run_id)
        elapsed = (datetime.utcnow() - start_time).total_seconds()
        suite_id_str = str(suite_id) if suite_id else "N/A"
        logger.info(f"✅ Test run #{test_run_id} completed successfully in {elapsed:.1f}s [Suite #{suite_id_str}]")
    except Exception as e:
        elapsed = (datetime.utcnow() - start_time).total_seconds()
        suite_id_str = str(suite_id) if suite_id else "N/A"
        logger.error(f"❌ Test run #{test_run_id} failed after {elapsed:.1f}s: {e} [Suite #{suite_id_str}]", exc_info=True)
    finally:
        # Update suite progress after test run completes
        ProgressTracker.update_suite_progress(suite_id)
        
        # Log suite progress update
        session = get_session()
        try:
            suite = session.query(TestSuite).filter(TestSuite.id == suite_id).first()
            if suite:
                progress_pct = (suite.completed_runs / suite.total_runs * 100) if suite.total_runs > 0 else 0
                suite_id_str = str(suite_id) if suite_id else "N/A"
                logger.info(
                    f"📊 Suite progress: {suite.completed_runs}/{suite.total_runs} completed "
                    f"({progress_pct:.1f}%), {suite.running_runs} running, {suite.pending_runs} pending, "
                    f"{suite.failed_runs} failed [Suite #{suite_id_str}]"
                )
        finally:
            session.close()


@router.post("/", response_model=SuiteResponse)
async def create_suite(request: SuiteRequest, background_tasks: BackgroundTasks):
    """Create a test suite and trigger all test runs."""
    if not request.test_types and not request.benchmarks:
        raise HTTPException(
            status_code=400,
            detail="At least one test type or benchmark must be specified"
        )
    
    if not request.models:
        raise HTTPException(
            status_code=400,
            detail="At least one model must be specified"
        )

    # Validate models
    for i, model in enumerate(request.models):
        if not isinstance(model, dict):
            raise HTTPException(
                status_code=400,
                detail=f"Model {i+1} must be an object with 'provider' and 'model' fields"
            )
        if not model.get("provider") or not model.get("model"):
            raise HTTPException(
                status_code=400,
                detail=f"Model {i+1} must have both 'provider' and 'model' specified"
            )

    runner = SuiteRunner()
    suite = runner.create_suite(
        name=request.name,
        test_types=request.test_types,
        benchmarks=request.benchmarks,
        models=request.models,
        test_config=request.test_config,
        num_samples=request.num_samples,
    )

    # Get suite ID before session closes
    suite_id = suite.id

    # Get all test runs and start them in background
    test_runs = runner.get_suite_runs(suite_id)
    suite_id_str = str(suite_id) if suite_id else "N/A"
    logger.info(f"🚀 Starting suite #{suite_id} with {len(test_runs)} test runs [Suite #{suite_id_str}]")
    for test_run in test_runs:
        background_tasks.add_task(_run_suite_test_background, test_run.id, suite_id)
    logger.info(f"📋 All {len(test_runs)} test runs queued for execution [Suite #{suite_id_str}]")

    # Re-fetch suite to ensure it's attached to a session for serialization
    # This ensures FastAPI can properly serialize the response
    session = get_session()
    try:
        suite = session.query(TestSuite).filter(TestSuite.id == suite_id).first()
        if not suite:
            raise HTTPException(status_code=500, detail="Failed to retrieve created suite")
        return suite
    finally:
        session.close()


@router.get("/", response_model=List[SuiteResponse])
async def list_suites(limit: int = 100, offset: int = 0):
    """List test suites."""
    runner = SuiteRunner()
    suites = runner.list_suites(limit=limit, offset=offset)
    return suites


@router.get("/{suite_id}", response_model=SuiteResponse)
async def get_suite(suite_id: int):
    """Get a test suite by ID."""
    # Update progress before returning
    ProgressTracker.update_suite_progress(suite_id)
    
    # Re-fetch suite in our own session to ensure it's properly attached
    session = get_session()
    try:
        suite = session.query(TestSuite).filter(TestSuite.id == suite_id).first()
        if not suite:
            raise HTTPException(status_code=404, detail="Test suite not found")
        return suite
    finally:
        session.close()


@router.get("/{suite_id}/runs", response_model=List[dict])
async def get_suite_runs(suite_id: int):
    """Get all test runs in a suite with detailed status."""
    runner = SuiteRunner()
    suite = runner.get_suite(suite_id)
    if not suite:
        raise HTTPException(status_code=404, detail="Test suite not found")
    
    test_runs = runner.get_suite_runs(suite_id)
    
    return [
        {
            "id": tr.id,
            "doctor_provider": tr.doctor_provider,
            "doctor_model": tr.doctor_model,
            "patient_provider": tr.patient_provider,
            "patient_model": tr.patient_model,
            "test_type": tr.test_type,
            "status": tr.status,
            "created_at": tr.created_at.isoformat() if tr.created_at else None,
            "updated_at": tr.updated_at.isoformat() if tr.updated_at else None,
            "benchmark": tr.meta_data.get("benchmark") if tr.test_type == "benchmark" else None,
        }
        for tr in test_runs
    ]


@router.get("/{suite_id}/progress", response_model=SuiteProgressResponse)
async def get_suite_progress(suite_id: int):
    """Get detailed progress breakdown for a suite."""
    runner = SuiteRunner()
    suite = runner.get_suite(suite_id)
    if not suite:
        raise HTTPException(status_code=404, detail="Test suite not found")
    
    # Update progress before calculating breakdown
    ProgressTracker.update_suite_progress(suite_id)
    
    progress = ProgressTracker.get_progress_breakdown(suite_id)
    return SuiteProgressResponse(**progress)


@router.delete("/{suite_id}")
async def delete_suite(suite_id: int):
    """Delete a test suite and all its test runs."""
    runner = SuiteRunner()
    success = runner.delete_suite(suite_id)
    if not success:
        raise HTTPException(status_code=404, detail="Test suite not found")
    
    return {"message": "Test suite deleted successfully", "id": suite_id}
