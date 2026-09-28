"""Analysis routes."""

import asyncio
import logging
import traceback
from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from vivasecuris.aiasylum.analysis import AnalysisService
from vivasecuris.aiasylum.database import Assessment, get_session, TestRun
from vivasecuris.aiasylum.constants import TEST_TYPE_ANALYSIS, TEST_TYPE_BENCHMARK, STATUS_PENDING, STATUS_RUNNING, STATUS_COMPLETED, STATUS_FAILED

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
    evaluator_system_prompt_id: Optional[int] = None
    # Extra evaluator instructions, added to the built-in scoring prompt (an ID wins)
    evaluator_system_prompt: Optional[str] = Field(None, max_length=20000)
    evaluator_temperature: Optional[float] = Field(None, ge=0, le=2)
    evaluator_max_tokens: Optional[int] = Field(None, ge=1, le=32768)
    evaluator_top_p: Optional[float] = Field(None, gt=0, le=1)
    evaluator_enable_cot: bool = False


class AssessmentResponse(BaseModel):
    """Assessment response."""
    id: int
    test_run_id: int
    created_at: Optional[datetime] = None
    assessment_text: str
    scores: dict
    overall_score: float
    analysis_type: Optional[str] = None
    flags: List[str]
    concerns: Optional[str] = None
    recommendations: Optional[str] = None
    metadata: dict

    @classmethod
    def from_orm(cls, obj: Assessment):
        """Create response from SQLAlchemy model, handling metadata conflict."""
        return cls(
            id=obj.id,
            test_run_id=obj.test_run_id,
            created_at=obj.created_at,
            assessment_text=obj.assessment_text,
            scores=obj.scores or {},
            overall_score=obj.overall_score,
            analysis_type=obj.analysis_type,
            flags=obj.flags or [],
            concerns=obj.concerns,
            recommendations=obj.recommendations,
            metadata=obj.meta_data or {},
        )


async def _run_analysis_background(
    test_run_id: int,
    enable_activation_patching: bool,
    enable_cot_detection: bool,
    cot_analysis_mode: str,
    enable_factuality_check: bool,
    enable_manipulation_analysis: bool,
    evaluator_provider: Optional[str],
    evaluator_model: Optional[str],
    analysis_test_run_id: Optional[int] = None,
    evaluator_system_prompt_id: Optional[int] = None,
    evaluator_system_prompt: Optional[str] = None,
    evaluator_temperature: Optional[float] = None,
    evaluator_max_tokens: Optional[int] = None,
    evaluator_top_p: Optional[float] = None,
    evaluator_enable_cot: bool = False,
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
            analysis_test_run_id=analysis_test_run_id,
            evaluator_system_prompt_id=evaluator_system_prompt_id,
            evaluator_system_prompt=evaluator_system_prompt,
            evaluator_temperature=evaluator_temperature,
            evaluator_max_tokens=evaluator_max_tokens,
            evaluator_top_p=evaluator_top_p,
            evaluator_enable_cot=evaluator_enable_cot,
        )
        logger.info(f"Analysis completed successfully for test run {test_run_id}, assessment ID: {assessment.id}")
    except Exception as e:
        # Log error with full traceback
        logger.error(f"Error running analysis for test run {test_run_id}: {str(e)}", exc_info=True)
        logger.error(f"Traceback: {traceback.format_exc()}")
        print(f"ERROR: Analysis failed for test run {test_run_id}: {e}")
        print(traceback.format_exc())
        
        # Update analysis test run status to failed if it exists
        if analysis_test_run_id:
            session = get_session()
            try:
                analysis_run = session.query(TestRun).filter(TestRun.id == analysis_test_run_id).first()
                if analysis_run:
                    analysis_run.status = "failed"
                    if analysis_run.meta_data is None:
                        analysis_run.meta_data = {}
                    analysis_run.meta_data["error"] = str(e)
                    session.commit()
            finally:
                session.close()


@router.post("/test-run/{test_run_id}")
async def analyze_test_run(
    test_run_id: int,
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
):
    """Run analysis on a test run (runs in background). Creates a new analysis test run entry."""
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
        logger.info(f"Test run {test_run_id} validated, creating analysis test run")
        
        # Create a new test run entry for the analysis
        # Use the evaluator model/provider if specified, otherwise use the doctor model
        analysis_doctor_provider = request.evaluator_provider or test_run.doctor_provider
        analysis_doctor_model = request.evaluator_model or test_run.doctor_model
        
        analysis_test_run = TestRun(
            doctor_provider=analysis_doctor_provider,
            doctor_model=analysis_doctor_model,
            patient_provider=test_run.patient_provider,  # Same patient model being analyzed
            patient_model=test_run.patient_model,
            test_type=TEST_TYPE_ANALYSIS,
            status=STATUS_PENDING,
            meta_data={
                "source_test_run_id": test_run_id,
                "analysis_config": {
                    "enable_activation_patching": request.enable_activation_patching,
                    "enable_cot_detection": request.enable_cot_detection,
                    "cot_analysis_mode": request.cot_analysis_mode,
                    "enable_factuality_check": request.enable_factuality_check,
                    "enable_manipulation_analysis": request.enable_manipulation_analysis,
                    "evaluator_provider": request.evaluator_provider,
                    "evaluator_model": request.evaluator_model,
                    "evaluator_system_prompt_id": request.evaluator_system_prompt_id,
                    "evaluator_system_prompt": request.evaluator_system_prompt,
                    "evaluator_temperature": request.evaluator_temperature,
                    "evaluator_max_tokens": request.evaluator_max_tokens,
                    "evaluator_top_p": request.evaluator_top_p,
                    "evaluator_enable_cot": request.evaluator_enable_cot,
                },
                "description": f"Analysis of test run #{test_run_id}",
            },
        )
        session.add(analysis_test_run)
        session.commit()
        session.refresh(analysis_test_run)
        
        logger.info(f"Created analysis test run {analysis_test_run.id} for source test run {test_run_id}")
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
            analysis_test_run.id,  # Pass the analysis test run ID
            request.evaluator_system_prompt_id,
            evaluator_system_prompt=request.evaluator_system_prompt,
            evaluator_temperature=request.evaluator_temperature,
            evaluator_max_tokens=request.evaluator_max_tokens,
            evaluator_top_p=request.evaluator_top_p,
            evaluator_enable_cot=request.evaluator_enable_cot,
        )
        logger.info(f"Background task added for test run {test_run_id}, analysis test run {analysis_test_run.id}")
    except Exception as e:
        logger.error(f"Failed to add background task for test run {test_run_id}: {str(e)}", exc_info=True)
        # Update analysis test run status to failed
        session = get_session()
        try:
            analysis_run = session.query(TestRun).filter(TestRun.id == analysis_test_run.id).first()
            if analysis_run:
                analysis_run.status = STATUS_FAILED
                if analysis_run.meta_data is None:
                    analysis_run.meta_data = {}
                analysis_run.meta_data["error"] = str(e)
                session.commit()
        finally:
            session.close()
        raise HTTPException(status_code=500, detail=f"Failed to start analysis: {str(e)}")
    
    return {
        "message": "Analysis started",
        "test_run_id": test_run_id,
        "analysis_test_run_id": analysis_test_run.id,
    }


@router.post("/analyze-unanalyzed")
async def analyze_unanalyzed(
    background_tasks: BackgroundTasks,
    request: AnalysisRequest = AnalysisRequest(
        enable_activation_patching=False,
        enable_cot_detection=True,
        cot_analysis_mode="full",
        enable_factuality_check=False,
        enable_manipulation_analysis=False,
    ),
):
    """Start analysis for completed behavioral tests that have no assessments.

    Benchmarks already have objective scores and are excluded from this bulk
    default. An explicit analysis of an individual benchmark remains available.

    Defaults match the suite auto-analysis config: COT detection on, factuality and
    manipulation analysis off, no custom evaluator.
    """

    session = get_session()
    try:
        unanalyzed_runs = (
            session.query(TestRun)
            .filter(
                TestRun.status == STATUS_COMPLETED,
                TestRun.test_type != TEST_TYPE_ANALYSIS,
                TestRun.test_type != TEST_TYPE_BENCHMARK,
                ~TestRun.id.in_(session.query(Assessment.test_run_id).distinct()),
            )
            .order_by(TestRun.id)
            .all()
        )

        if not unanalyzed_runs:
            return {"started": 0, "test_run_ids": []}

        analysis_doctor_provider = request.evaluator_provider
        analysis_doctor_model = request.evaluator_model

        created = []
        for run in unanalyzed_runs:
            doc_provider = analysis_doctor_provider or run.doctor_provider
            doc_model = analysis_doctor_model or run.doctor_model
            analysis_run = TestRun(
                doctor_provider=doc_provider,
                doctor_model=doc_model,
                patient_provider=run.patient_provider,
                patient_model=run.patient_model,
                test_type=TEST_TYPE_ANALYSIS,
                status=STATUS_PENDING,
                meta_data={
                    "source_test_run_id": run.id,
                    "analysis_config": {
                        "enable_activation_patching": request.enable_activation_patching,
                        "enable_cot_detection": request.enable_cot_detection,
                        "cot_analysis_mode": request.cot_analysis_mode,
                        "enable_factuality_check": request.enable_factuality_check,
                        "enable_manipulation_analysis": request.enable_manipulation_analysis,
                        "evaluator_provider": request.evaluator_provider,
                        "evaluator_model": request.evaluator_model,
                        "evaluator_system_prompt_id": request.evaluator_system_prompt_id,
                        "evaluator_system_prompt": request.evaluator_system_prompt,
                        "evaluator_temperature": request.evaluator_temperature,
                        "evaluator_max_tokens": request.evaluator_max_tokens,
                        "evaluator_top_p": request.evaluator_top_p,
                        "evaluator_enable_cot": request.evaluator_enable_cot,
                    },
                    "description": f"Batch analysis of test run #{run.id}",
                },
            )
            session.add(analysis_run)
            session.flush()
            created.append((run.id, analysis_run.id))

        session.commit()
        logger.info(f"Created {len(created)} analysis test runs for unanalyzed runs")
    finally:
        session.close()

    for source_id, analysis_id in created:
        background_tasks.add_task(
            _run_analysis_background,
            source_id,
            request.enable_activation_patching,
            request.enable_cot_detection,
            request.cot_analysis_mode,
            request.enable_factuality_check,
            request.enable_manipulation_analysis,
            request.evaluator_provider,
            request.evaluator_model,
            analysis_id,
            request.evaluator_system_prompt_id,
            evaluator_system_prompt=request.evaluator_system_prompt,
            evaluator_temperature=request.evaluator_temperature,
            evaluator_max_tokens=request.evaluator_max_tokens,
            evaluator_top_p=request.evaluator_top_p,
            evaluator_enable_cot=request.evaluator_enable_cot,
        )

    started_ids = [source_id for source_id, _ in created]
    logger.info(f"Queued background analysis for test run ids: {started_ids}")
    return {"started": len(created), "test_run_ids": started_ids}


@router.get("/test-run/{test_run_id}/assessments", response_model=List[AssessmentResponse])
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
        return [AssessmentResponse.from_orm(a) for a in assessments]
    finally:
        session.close()
