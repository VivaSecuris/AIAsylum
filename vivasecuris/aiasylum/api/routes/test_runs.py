"""Test run routes."""

from typing import List, Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

from vivasecuris.aiasylum.runner import TestRunner
from vivasecuris.aiasylum.database import get_session, TestRun, TestResult, ConversationTurn

router = APIRouter()


class TestRunRequest(BaseModel):
    """Test run creation request."""
    doctor_provider: str
    doctor_model: str
    patient_provider: str
    patient_model: str
    test_type: str
    test_config: Optional[dict] = None


class TestRunResponse(BaseModel):
    """Test run response."""
    id: int
    doctor_provider: str
    doctor_model: str
    patient_provider: str
    patient_model: str
    test_type: str
    status: str
    
    class Config:
        from_attributes = True


@router.post("/", response_model=TestRunResponse)
async def create_test_run(request: TestRunRequest, background_tasks: BackgroundTasks):
    """Create and run a test."""
    runner = TestRunner()
    
    # Run test in background
    test_run = await runner.run_test(
        doctor_provider=request.doctor_provider,
        doctor_model=request.doctor_model,
        patient_provider=request.patient_provider,
        patient_model=request.patient_model,
        test_type=request.test_type,
        test_config=request.test_config,
    )
    
    return test_run


@router.get("/", response_model=List[TestRunResponse])
async def list_test_runs(
    limit: int = 100,
    offset: int = 0,
    test_type: Optional[str] = None,
):
    """List test runs."""
    runner = TestRunner()
    test_runs = runner.list_test_runs(limit=limit, offset=offset, test_type=test_type)
    return test_runs


@router.get("/{test_run_id}", response_model=TestRunResponse)
async def get_test_run(test_run_id: int):
    """Get a test run by ID."""
    runner = TestRunner()
    test_run = runner.get_test_run(test_run_id)
    if not test_run:
        raise HTTPException(status_code=404, detail="Test run not found")
    return test_run


@router.get("/{test_run_id}/results")
async def get_test_results(test_run_id: int):
    """Get results for a test run."""
    session = get_session()
    results = session.query(TestResult).filter(TestResult.test_run_id == test_run_id).all()
    return results


@router.get("/{test_run_id}/conversation")
async def get_conversation(test_run_id: int):
    """Get conversation turns for a test run."""
    session = get_session()
    turns = (
        session.query(ConversationTurn)
        .filter(ConversationTurn.test_run_id == test_run_id)
        .order_by(ConversationTurn.turn_number)
        .all()
    )
    return turns
