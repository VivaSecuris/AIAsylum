"""Analysis routes."""

import asyncio
import logging
import traceback
from typing import Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

from vivasecuris.aiasylum.analysis import AnalysisService
from vivasecuris.aiasylum.database import Assessment, get_session, TestRun

logger = logging.getLogger(__name__)
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
    logger.info(f"Starting background analysis for test run {test_run_id}")
    logger.info(f"Analysis config: COT={enable_cot_detection}, Factuality={enable_factuality_check}, Manipulation={enable_manipulation_analysis}")
    
    service = AnalysisService()
    try:
        logger.info(f"Calling analyze_test_run for test run {test_run_id}")
        assessment = await service.analyze_test_run(
            test_run_id=test_run_id,
            enable_activation_patching=enable_activation_patching,
            enable_cot_detection=enable_cot_detection,
            cot_analysis_mode=cot_analysis_mode,
            enable_factuality_check=enable_factuality_check,
            enable_manipulation_analysis=enable_manipulation_analysis,
            evaluator_provider=evaluator_provider,
            evaluator_model=evaluator_model,
        )
        logger.info(f"Analysis completed successfully for test run {test_run_id}, assessment ID: {assessment.id}")
    except Exception as e:
        # Log error with full traceback
        logger.error(f"Error running analysis for test run {test_run_id}: {str(e)}", exc_info=True)
        logger.error(f"Traceback: {traceback.format_exc()}")
        print(f"ERROR: Analysis failed for test run {test_run_id}: {e}")
        print(traceback.format_exc())


@router.post("/test-run/{test_run_id}")
async def analyze_test_run(
    test_run_id: int,
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
):
    """Run analysis on a test run (runs in background)."""
    logger.info(f"Received analysis request for test run {test_run_id}")
    logger.info(f"Request config: {request.dict()}")
    
    # Verify test run exists
    session = get_session()
    try:
        test_run = session.query(TestRun).filter(TestRun.id == test_run_id).first()
        if not test_run:
            logger.error(f"Test run {test_run_id} not found")
            raise HTTPException(status_code=404, detail=f"Test run {test_run_id} not found")
        if test_run.status != "completed":
            logger.error(f"Test run {test_run_id} is not completed (status: {test_run.status})")
            raise HTTPException(
                status_code=400,
                detail=f"Test run {test_run_id} is not completed (status: {test_run.status})"
            )
        logger.info(f"Test run {test_run_id} validated, starting background analysis")
    finally:
        session.close()
    
    # Start analysis in background
    try:
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
        logger.info(f"Background task added for test run {test_run_id}")
    except Exception as e:
        logger.error(f"Failed to add background task for test run {test_run_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to start analysis: {str(e)}")
    
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
