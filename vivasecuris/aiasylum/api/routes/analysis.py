"""Analysis routes."""

import asyncio
from typing import Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

from vivasecuris.aiasylum.analysis import AnalysisService
from vivasecuris.aiasylum.database import Assessment, get_session, TestRun

router = APIRouter()


class AnalysisRequest(BaseModel):
    """Analysis request."""
    enable_activation_patching: bool = False
    enable_cot_detection: bool = False
    cot_analysis_mode: str = "full"
    enable_factuality_check: bool = False
    enable_manipulation_analysis: bool = False
    evaluator_provider: Optional[str] = None
    evaluator_model: Optional[str] = None


async def _run_analysis_background(
    test_run_id: int,
    enable_activation_patching: bool,
    enable_cot_detection: bool,
    cot_analysis_mode: str,
    enable_factuality_check: bool,
    enable_manipulation_analysis: bool,
    evaluator_provider: Optional[str],
    evaluator_model: Optional[str],
):
    """Run analysis in background task."""
    service = AnalysisService()
    try:
        await service.analyze_test_run(
            test_run_id=test_run_id,
            enable_activation_patching=enable_activation_patching,
            enable_cot_detection=enable_cot_detection,
            cot_analysis_mode=cot_analysis_mode,
            enable_factuality_check=enable_factuality_check,
            enable_manipulation_analysis=enable_manipulation_analysis,
            evaluator_provider=evaluator_provider,
            evaluator_model=evaluator_model,
        )
    except Exception as e:
        # Log error - assessment creation will handle status
        print(f"Error running analysis for test run {test_run_id}: {e}")


@router.post("/test-run/{test_run_id}")
async def analyze_test_run(
    test_run_id: int,
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
):
    """Run analysis on a test run (runs in background)."""
    # Verify test run exists
    session = get_session()
    try:
        test_run = session.query(TestRun).filter(TestRun.id == test_run_id).first()
        if not test_run:
            raise HTTPException(status_code=404, detail=f"Test run {test_run_id} not found")
        if test_run.status != "completed":
            raise HTTPException(
                status_code=400,
                detail=f"Test run {test_run_id} is not completed (status: {test_run.status})"
            )
    finally:
        session.close()
    
    # Start analysis in background
    background_tasks.add_task(
        _run_analysis_background,
        test_run_id,
        request.enable_activation_patching,
        request.enable_cot_detection,
        request.cot_analysis_mode,
        request.enable_factuality_check,
        request.enable_manipulation_analysis,
        request.evaluator_provider,
        request.evaluator_model,
    )
    
    return {"message": "Analysis started", "test_run_id": test_run_id}


@router.get("/test-run/{test_run_id}/assessments")
async def get_assessments(test_run_id: int):
    """Get assessments for a test run."""
    from vivasecuris.aiasylum.database import get_session
    session = get_session()
    try:
        assessments = (
            session.query(Assessment)
            .filter(Assessment.test_run_id == test_run_id)
            .order_by(Assessment.created_at.desc())
            .all()
        )
        return assessments
    finally:
        session.close()
