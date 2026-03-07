"""Test suite routes."""

import asyncio
import logging
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

from vivasecuris.aiasylum.suites import SuiteRunner, ProgressTracker
from vivasecuris.aiasylum.database import get_session, TestRun, TestSuite
from vivasecuris.aiasylum.runner import TestRunner
from vivasecuris.aiasylum.constants import STATUS_PENDING, STATUS_COMPLETED, TEST_TYPE_ANALYSIS

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


class SuiteUpdate(BaseModel):
    """Test suite update request."""
    name: Optional[str] = None  # Custom name for the suite


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


async def _trigger_suite_analysis(completed_test_run_ids: List[int]):
    """Trigger analysis for each completed test run (used when suite finishes). Runs analyses in background."""
    from vivasecuris.aiasylum.api.routes.analysis import _run_analysis_background
    default_cot = True
    default_cot_mode = "full"
    default_factuality = False
    default_manipulation = False
    for test_run_id in completed_test_run_ids:
        try:
            session = get_session()
            try:
                test_run = session.query(TestRun).filter(TestRun.id == test_run_id).first()
                if not test_run or test_run.status != STATUS_COMPLETED:
                    continue
                analysis_test_run = TestRun(
                    doctor_provider=test_run.doctor_provider,
                    doctor_model=test_run.doctor_model,
                    patient_provider=test_run.patient_provider,
                    patient_model=test_run.patient_model,
                    test_type=TEST_TYPE_ANALYSIS,
                    status=STATUS_PENDING,
                    meta_data={
                        "source_test_run_id": test_run_id,
                        "analysis_config": {
                            "enable_activation_patching": False,
                            "enable_cot_detection": default_cot,
                            "cot_analysis_mode": default_cot_mode,
                            "enable_factuality_check": default_factuality,
                            "enable_manipulation_analysis": default_manipulation,
                            "evaluator_provider": None,
                            "evaluator_model": None,
                        },
                        "description": f"Suite auto-analysis of test run #{test_run_id}",
                    },
                )
                session.add(analysis_test_run)
                session.commit()
                session.refresh(analysis_test_run)
                analysis_test_run_id = analysis_test_run.id
                logger.info(f"Created analysis run {analysis_test_run_id} for suite completion (source test run {test_run_id})")
            finally:
                session.close()
            # Run each analysis in background (don't await so they run in parallel)
            asyncio.create_task(
                _run_analysis_background(
                    test_run_id,
                    False,
                    default_cot,
                    default_cot_mode,
                    default_factuality,
                    default_manipulation,
                    None,
                    None,
                    analysis_test_run_id,
                )
            )
        except Exception as e:
            logger.error(f"Suite auto-analysis failed for test run {test_run_id}: {e}", exc_info=True)


async def _run_suite_test_background(test_run_id: int, suite_id: int):
    """Background task to run a test and update suite progress with worker pool limiting."""
    from vivasecuris.aiasylum.api.worker_pool import worker_pool
    
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
    
    # Run with worker pool limit
    async def _execute():
        suite_id_str = str(suite_id) if suite_id else "N/A"
        logger.info(f"▶️  Executing test run #{test_run_id}... [Suite #{suite_id_str}]")
        await runner.execute_test_run(test_run_id)
    
    try:
        await worker_pool.run_with_limit(test_run_id, _execute())
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
        
        # Log suite progress and trigger auto-analysis when suite finishes
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
                # When suite just finished (all runs done), run analysis on each completed test run
                if suite.status in ("completed", "partially_failed") and not (suite.meta_data or {}).get("suite_analysis_triggered"):
                    suite.meta_data = suite.meta_data or {}
                    suite.meta_data["suite_analysis_triggered"] = True
                    session.commit()
                    # Get completed test run ids (need fresh query after commit)
                    completed_runs = session.query(TestRun.id).filter(
                        TestRun.suite_id == suite_id,
                        TestRun.status == STATUS_COMPLETED,
                    ).all()
                    completed_ids = [r.id for r in completed_runs]
                    if completed_ids:
                        logger.info(f"📋 Suite #{suite_id} finished: triggering analysis for {len(completed_ids)} completed run(s)")
                        asyncio.create_task(_trigger_suite_analysis(completed_ids))
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
            "meta_data": tr.meta_data or {},
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


@router.put("/{suite_id}", response_model=SuiteResponse)
async def update_suite(suite_id: int, update: SuiteUpdate):
    """Update a test suite (e.g., rename it)."""
    session = get_session()
    try:
        suite = session.query(TestSuite).filter(TestSuite.id == suite_id).first()
        if not suite:
            raise HTTPException(status_code=404, detail="Test suite not found")
        
        # Update name
        if update.name is not None:
            if update.name.strip():
                suite.name = update.name.strip()
            else:
                suite.name = None  # Remove name if empty string
        
        session.commit()
        session.refresh(suite)
        return suite
    finally:
        session.close()


@router.delete("/{suite_id}")
async def delete_suite(suite_id: int):
    """Delete a test suite and all its test runs."""
    runner = SuiteRunner()
    try:
        success = runner.delete_suite(suite_id)
        if not success:
            raise HTTPException(status_code=404, detail="Test suite not found")
        return {"message": "Test suite deleted successfully", "id": suite_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete suite: {str(e)}")
