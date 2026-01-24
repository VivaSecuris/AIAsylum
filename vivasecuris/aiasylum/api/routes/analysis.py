"""Analysis routes."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from vivasecuris.aiasylum.analysis import AnalysisService
from vivasecuris.aiasylum.database import Assessment

router = APIRouter()


class AnalysisRequest(BaseModel):
    """Analysis request."""
    enable_activation_patching: bool = False
    enable_cot_detection: bool = False
    cot_analysis_mode: str = "full"


@router.post("/test-run/{test_run_id}")
async def analyze_test_run(test_run_id: int, request: AnalysisRequest):
    """Run analysis on a test run."""
    service = AnalysisService()
    
    try:
        assessment = await service.analyze_test_run(
            test_run_id=test_run_id,
            enable_activation_patching=request.enable_activation_patching,
            enable_cot_detection=request.enable_cot_detection,
            cot_analysis_mode=request.cot_analysis_mode,
        )
        return assessment
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/test-run/{test_run_id}/assessments")
async def get_assessments(test_run_id: int):
    """Get assessments for a test run."""
    from vivasecuris.aiasylum.database import get_session
    session = get_session()
    assessments = (
        session.query(Assessment)
        .filter(Assessment.test_run_id == test_run_id)
        .order_by(Assessment.created_at.desc())
        .all()
    )
    return assessments
