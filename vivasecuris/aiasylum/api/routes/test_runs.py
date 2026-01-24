"""Test run routes."""

from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

from vivasecuris.aiasylum.runner import TestRunner
from vivasecuris.aiasylum.database import get_session, TestRun, TestResult, ConversationTurn
from vivasecuris.aiasylum.constants import STATUS_PENDING, STATUS_RUNNING, STATUS_FAILED

router = APIRouter()


class TestRunRequest(BaseModel):
    """Test run creation request."""
    doctor_provider: str
    doctor_model: str
    patient_provider: str
    patient_model: str
    test_type: str
    test_config: Optional[dict] = None
    prompt_id: Optional[int] = None  # Optional prompt from library
    variables: Optional[dict] = None  # Variable values for prompt substitution (e.g., {"country": "France"})


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


class ConversationTurnResponse(BaseModel):
    """Conversation turn response."""
    id: int
    test_run_id: int
    turn_number: int
    speaker: str
    prompt: str
    response: str
    created_at: Optional[datetime] = None
    metadata: Optional[dict] = None
    
    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }
    
    @classmethod
    def from_orm(cls, obj):
        """Create response from SQLAlchemy model, handling metadata conflict."""
        return cls(
            id=obj.id,
            test_run_id=obj.test_run_id,
            turn_number=obj.turn_number,
            speaker=obj.speaker,
            prompt=obj.prompt,
            response=obj.response,
            created_at=obj.created_at,
            metadata=obj.meta_data or {},
        )


async def _run_test_background(test_run_id: int):
    """Background task to run a test."""
    runner = TestRunner()
    try:
        await runner.execute_test_run(test_run_id)
    except Exception as e:
        # Error is already handled in execute_test_run (sets status to failed)
        print(f"Error running test {test_run_id}: {e}")


@router.post("/", response_model=TestRunResponse)
async def create_test_run(request: TestRunRequest, background_tasks: BackgroundTasks):
    """Create and run a test."""
    runner = TestRunner()
    
    # Create test run record first
    session = get_session()
    try:
        test_config = request.test_config or {}
        if request.prompt_id:
            test_config["prompt_id"] = request.prompt_id
        if request.variables:
            test_config["variables"] = request.variables
        
        test_run = TestRun(
            doctor_provider=request.doctor_provider,
            doctor_model=request.doctor_model,
            patient_provider=request.patient_provider,
            patient_model=request.patient_model,
            test_type=request.test_type,
            status=STATUS_PENDING,
            meta_data={"test_config": test_config},
        )
        session.add(test_run)
        session.commit()
        session.refresh(test_run)
        
        # Run test in background
        background_tasks.add_task(_run_test_background, test_run.id)
        
        return test_run
    finally:
        session.close()


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


@router.delete("/{test_run_id}")
async def delete_test_run(test_run_id: int):
    """Delete a test run and all its associated data."""
    print(f"DELETE /api/v1/test-runs/{test_run_id} - Starting deletion")
    session = None
    try:
        session = get_session()
        test_run = session.query(TestRun).filter(TestRun.id == test_run_id).first()
        if not test_run:
            print(f"DELETE /api/v1/test-runs/{test_run_id} - Test run not found")
            raise HTTPException(status_code=404, detail="Test run not found")
        
        print(f"DELETE /api/v1/test-runs/{test_run_id} - Found test run, status: {test_run.status}")
        
        # Check if test run is currently running
        if test_run.status == STATUS_RUNNING:
            print(f"DELETE /api/v1/test-runs/{test_run_id} - Cannot delete running test run")
            raise HTTPException(
                status_code=400,
                detail="Cannot delete a test run that is currently running. Please wait for it to complete or fail."
            )
        
        # Delete the test run (cascade will handle related records)
        print(f"DELETE /api/v1/test-runs/{test_run_id} - Deleting test run from database")
        session.delete(test_run)
        session.commit()
        print(f"DELETE /api/v1/test-runs/{test_run_id} - Deletion committed successfully")
        session.close()
        
        return {"message": "Test run deleted successfully", "id": test_run_id}
    except HTTPException as e:
        print(f"DELETE /api/v1/test-runs/{test_run_id} - HTTPException: {e.status_code} - {e.detail}")
        if session:
            session.rollback()
            session.close()
        raise
    except Exception as e:
        print(f"DELETE /api/v1/test-runs/{test_run_id} - Exception: {str(e)}")
        import traceback
        traceback.print_exc()
        if session:
            session.rollback()
            session.close()
        raise HTTPException(status_code=500, detail=f"Failed to delete test run: {str(e)}")


@router.get("/{test_run_id}/results")
async def get_test_results(test_run_id: int):
    """Get results for a test run."""
    session = get_session()
    try:
        results = session.query(TestResult).filter(TestResult.test_run_id == test_run_id).all()
        return results
    finally:
        session.close()


@router.get("/{test_run_id}/conversation", response_model=List[ConversationTurnResponse])
async def get_conversation(test_run_id: int):
    """Get conversation turns for a test run."""
    session = get_session()
    try:
        turns = (
            session.query(ConversationTurn)
            .filter(ConversationTurn.test_run_id == test_run_id)
            .order_by(ConversationTurn.turn_number)
            .all()
        )
        # Debug logging
        print(f"Found {len(turns)} conversation turns for test_run_id={test_run_id}")
        if turns:
            print(f"First turn: speaker={turns[0].speaker}, response_length={len(turns[0].response)}")
        # Convert to response models to handle metadata properly
        return [ConversationTurnResponse.from_orm(turn) for turn in turns]
    finally:
        session.close()


@router.post("/{test_run_id}/start", response_model=TestRunResponse)
async def start_test_run(test_run_id: int, background_tasks: BackgroundTasks):
    """Start/execute a pending test run."""
    runner = TestRunner()
    test_run = runner.get_test_run(test_run_id)
    if not test_run:
        raise HTTPException(status_code=404, detail="Test run not found")
    
    if test_run.status not in (STATUS_PENDING, STATUS_FAILED):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot start test run in status: {test_run.status}. Only pending or failed test runs can be started."
        )
    
    # Run test in background
    background_tasks.add_task(_run_test_background, test_run_id)
    
    # Return the test run (status will be updated by background task)
    return test_run
