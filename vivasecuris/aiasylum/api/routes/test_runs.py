"""Test run routes."""

from datetime import datetime
from typing import List, Optional
import json
import logging
from uuid import uuid4

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from vivasecuris.aiasylum.runner import TestRunner
from vivasecuris.aiasylum.database import get_session, TestRun, TestResult, ConversationTurn
from vivasecuris.aiasylum.constants import STATUS_PENDING, STATUS_RUNNING, STATUS_PAUSED, STATUS_FAILED, STATUS_COMPLETED, TEST_TYPE_GROUP_THERAPY
from vivasecuris.aiasylum.api.progress_events import progress_event_manager
from vivasecuris.aiasylum.api.cancellation import cancellation_manager
from vivasecuris.aiasylum.exceptions import TestExecutionError
from vivasecuris.aiasylum.runner.run_config import normalize_test_config, validate_test_config, validate_patient_prompt_selection

router = APIRouter()


class TestRunRequest(BaseModel):
    """Test run creation request."""
    doctor_provider: str
    doctor_model: str
    patient_provider: str
    patient_model: str
    test_type: str
    test_config: Optional[dict] = None
    lineage_parent: Optional[str] = Field(None, max_length=1024)
    prompt_id: Optional[int] = None  # Optional prompt from library
    variables: Optional[dict] = None  # Variable values for prompt substitution (e.g., {"country": "France"})
    suite_id: Optional[int] = None  # Optional suite ID to link test run to a suite
    name: Optional[str] = Field(None, max_length=200)  # Optional display name


class TestRunUpdate(BaseModel):
    """Test run update request."""
    name: Optional[str] = None  # Custom name for the test run


class TestRunResponse(BaseModel):
    """Test run response."""
    id: int
    doctor_provider: str
    doctor_model: str
    patient_provider: str
    patient_model: str
    test_type: str
    status: str
    suite_id: Optional[int] = None
    suite_name: Optional[str] = None
    meta_data: Optional[dict] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


def _visible_response_metadata(value, *, expose_reasoning: bool):
    """Apply the transcript's reasoning policy to copied nested evidence too."""
    if isinstance(value, list):
        return [_visible_response_metadata(item, expose_reasoning=expose_reasoning) for item in value]
    if not isinstance(value, dict):
        return value
    visible = {
        key: _visible_response_metadata(item, expose_reasoning=expose_reasoning)
        for key, item in value.items()
    }
    if not expose_reasoning and value.get("reasoning_source") not in ("inline", "provider"):
        visible.pop("reasoning", None)
        visible.pop("reasoning_source", None)
    return visible


class ConversationTurnResponse(BaseModel):
    """Conversation turn response."""
    id: int
    test_run_id: int
    turn_number: int
    speaker: str
    prompt: str
    response: str
    model_name: Optional[str] = None
    model_provider: Optional[str] = None
    usage: Optional[dict] = None
    created_at: Optional[datetime] = None
    metadata: Optional[dict] = None
    
    class Config:
        from_attributes = True
        protected_namespaces = ()
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }
    
    @classmethod
    def from_orm(cls, obj, *, expose_reasoning: bool = False):
        """Create response from SQLAlchemy model, handling metadata conflict."""
        # A ReACT thought is framework-internal chain-of-thought and is only
        # returned when the test was configured with CoT enabled (see
        # get_conversation). A model's *own* trace -- an inline <think> block or
        # a field the provider returned -- is part of what the model produced
        # and is always returned, labelled by its source.
        safe_metadata = _visible_response_metadata(
            obj.meta_data or {}, expose_reasoning=expose_reasoning,
        )
        return cls(
            id=obj.id,
            test_run_id=obj.test_run_id,
            turn_number=obj.turn_number,
            speaker=obj.speaker,
            prompt=obj.prompt,
            response=obj.response,
            model_name=getattr(obj, "model_name", None),
            model_provider=getattr(obj, "model_provider", None),
            usage=getattr(obj, "usage", None),
            created_at=obj.created_at,
            metadata=safe_metadata,
        )


async def _run_test_background(test_run_id: int):
    """Background task to run a test with worker pool limiting."""
    import asyncio
    from vivasecuris.aiasylum.api.worker_pool import worker_pool
    
    runner = TestRunner()
    
    # Run with worker pool limit
    async def _execute():
        await runner.execute_test_run(test_run_id)
    
    try:
        await worker_pool.run_with_limit(test_run_id, _execute())
    except asyncio.CancelledError:
        logger = logging.getLogger(__name__)
        logger.warning(f"🛑 Test run {test_run_id} task was cancelled")
        print(f"[BACKGROUND] Test run {test_run_id} task was cancelled")
        # Update status in database
        session = get_session()
        try:
            test_run = session.query(TestRun).filter(TestRun.id == test_run_id).first()
            if test_run and test_run.status == STATUS_RUNNING:
                test_run.status = STATUS_FAILED
                if not test_run.meta_data:
                    test_run.meta_data = {}
                test_run.meta_data["cancelled"] = True
                test_run.meta_data["error"] = "Test run was cancelled"
                session.commit()
                # Emit cancellation event
                await progress_event_manager.emit_event(
                    test_run_id,
                    "test_cancelled",
                    {
                        "status": "failed",
                        "cancelled": True,
                    },
                    "Test run was cancelled"
                )
        finally:
            session.close()
        raise
    except TestExecutionError as e:
        # Check if this is a pause (not a cancellation)
        if hasattr(e, 'pause') and e.pause:
            logger = logging.getLogger(__name__)
            logger.info(f"⏸️ Test run {test_run_id} was paused")
            print(f"[BACKGROUND] Test run {test_run_id} was paused")
            # Status should already be set to paused by the pause endpoint
            # Just make sure it's set correctly
            session = get_session()
            try:
                test_run = session.query(TestRun).filter(TestRun.id == test_run_id).first()
                if test_run and test_run.status != STATUS_PAUSED:
                    test_run.status = STATUS_PAUSED
                    if not test_run.meta_data:
                        test_run.meta_data = {}
                    test_run.meta_data["paused"] = True
                    session.commit()
            finally:
                session.close()
        else:
            # Error is already handled in execute_test_run (sets status to failed)
            logger = logging.getLogger(__name__)
            logger.error(f"Error running test {test_run_id}: {e}")
            print(f"Error running test {test_run_id}: {e}")
    except Exception as e:
        # Error is already handled in execute_test_run (sets status to failed)
        logger = logging.getLogger(__name__)
        logger.error(f"Error running test {test_run_id}: {e}")
        print(f"Error running test {test_run_id}: {e}")
    finally:
        # Unregister task when done (unless paused)
        session = get_session()
        try:
            test_run = session.query(TestRun).filter(TestRun.id == test_run_id).first()
            if test_run and test_run.status != STATUS_PAUSED:
                cancellation_manager.unregister_task(test_run_id)
        finally:
            session.close()


@router.post("/", response_model=TestRunResponse)
async def create_test_run(request: TestRunRequest, background_tasks: BackgroundTasks):
    """Create and run a test."""
    # Reject empty provider/model so we fail fast with a clear error
    if not (request.doctor_provider and request.doctor_provider.strip()):
        raise HTTPException(status_code=400, detail="doctor_provider is required")
    if not (request.doctor_model and request.doctor_model.strip()):
        raise HTTPException(status_code=400, detail="doctor_model is required")
    if not (request.patient_provider and request.patient_provider.strip()):
        raise HTTPException(status_code=400, detail="patient_provider is required")
    if not (request.patient_model and request.patient_model.strip()):
        raise HTTPException(status_code=400, detail="patient_model is required")

    runner = TestRunner()
    
    # Create test run record first
    session = get_session()
    try:
        test_config = request.test_config or {}
        if request.prompt_id:
            test_config["prompt_id"] = request.prompt_id
        if request.variables:
            test_config["variables"] = request.variables
        errors = validate_test_config(test_config)
        if errors:
            raise HTTPException(status_code=422, detail="; ".join(errors))
        normalize_test_config(test_config)
        try:
            validate_patient_prompt_selection(session, request.test_type, test_config)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        
        # Validate group_therapy test type
        if request.test_type == TEST_TYPE_GROUP_THERAPY:
            patients = test_config.get("patients", [])
            if not patients or len(patients) == 0:
                raise HTTPException(
                    status_code=400,
                    detail="Group therapy test requires at least one patient model. Please add patient models in test_config.patients"
                )
            # Validate each patient has provider and model
            for i, patient in enumerate(patients):
                if not isinstance(patient, dict):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Patient {i+1} must be an object with 'provider' and 'model' fields"
                    )
                if not patient.get("provider") or not patient.get("model"):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Patient {i+1} must have both 'provider' and 'model' specified"
                    )
        
        test_run = TestRun(
            doctor_provider=request.doctor_provider,
            doctor_model=request.doctor_model,
            patient_provider=request.patient_provider,
            patient_model=request.patient_model,
            test_type=request.test_type,
            status=STATUS_PENDING,
            suite_id=request.suite_id,
            meta_data={"test_config": test_config, "lineage_parent": request.lineage_parent,
                       "lineage_id": uuid4().hex,
                       **({"name": request.name.strip()} if request.name and request.name.strip() else {})},
        )
        session.add(test_run)
        session.commit()
        session.refresh(test_run)
        
        # Create and register the asyncio task so we can cancel it later
        # Use create_task to get a reference to the task - it will run in the background
        import asyncio
        task = asyncio.create_task(_run_test_background(test_run.id))
        cancellation_manager.register_task(test_run.id, task)
        
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
    from vivasecuris.aiasylum.database import TestSuite
    runner = TestRunner()
    test_runs = runner.list_test_runs(limit=limit, offset=offset, test_type=test_type)
    
    # Get suite information for test runs that belong to suites
    session = get_session()
    try:
        suite_ids = {tr.suite_id for tr in test_runs if tr.suite_id}
        suites = {}
        if suite_ids:
            suite_list = session.query(TestSuite).filter(TestSuite.id.in_(suite_ids)).all()
            suites = {s.id: s for s in suite_list}
        
        # Build response with suite information
        result = []
        for tr in test_runs:
            suite_name = None
            if tr.suite_id and tr.suite_id in suites:
                suite_name = suites[tr.suite_id].name or "Unnamed Suite"
            result.append(TestRunResponse(
                id=tr.id,
                doctor_provider=tr.doctor_provider,
                doctor_model=tr.doctor_model,
                patient_provider=tr.patient_provider,
                patient_model=tr.patient_model,
                test_type=tr.test_type,
                status=tr.status,
                suite_id=tr.suite_id,
                suite_name=suite_name,
                meta_data=tr.meta_data or {},
                created_at=tr.created_at,
                updated_at=tr.updated_at,
            ))
        return result
    finally:
        session.close()


@router.get("/{test_run_id}", response_model=TestRunResponse)
async def get_test_run(test_run_id: int):
    """Get a test run by ID."""
    from vivasecuris.aiasylum.database import TestSuite
    runner = TestRunner()
    test_run = runner.get_test_run(test_run_id)
    if not test_run:
        raise HTTPException(status_code=404, detail="Test run not found")
    
    # Get suite information if test run belongs to a suite
    suite_name = None
    if test_run.suite_id:
        session = get_session()
        try:
            suite = session.query(TestSuite).filter(TestSuite.id == test_run.suite_id).first()
            if suite:
                suite_name = suite.name or "Unnamed Suite"
        finally:
            session.close()
    
    return TestRunResponse(
        id=test_run.id,
        doctor_provider=test_run.doctor_provider,
        doctor_model=test_run.doctor_model,
        patient_provider=test_run.patient_provider,
        patient_model=test_run.patient_model,
        test_type=test_run.test_type,
        status=test_run.status,
        suite_id=test_run.suite_id,
        suite_name=suite_name,
        meta_data=test_run.meta_data or {},
        created_at=test_run.created_at,
        updated_at=test_run.updated_at,
    )


@router.put("/{test_run_id}", response_model=TestRunResponse)
async def update_test_run(test_run_id: int, update: TestRunUpdate):
    """Update a test run (e.g., rename it)."""
    session = get_session()
    try:
        test_run = session.query(TestRun).filter(TestRun.id == test_run_id).first()
        if not test_run:
            raise HTTPException(status_code=404, detail="Test run not found")
        
        # Update name in meta_data
        if update.name is not None:
            if not test_run.meta_data:
                test_run.meta_data = {}
            if update.name.strip():
                test_run.meta_data["name"] = update.name.strip()
            elif "name" in test_run.meta_data:
                # Remove name if empty string
                del test_run.meta_data["name"]
        
        session.commit()
        session.refresh(test_run)
        return test_run
    finally:
        session.close()


@router.delete("/{test_run_id}")
async def delete_test_run(test_run_id: int):
    """Delete a test run and all its associated data (results, conversation turns, assessments)."""
    print(f"DELETE /api/v1/test-runs/{test_run_id} - Starting deletion")
    session = None
    try:
        session = get_session()
        test_run = session.query(TestRun).filter(TestRun.id == test_run_id).first()
        if not test_run:
            print(f"DELETE /api/v1/test-runs/{test_run_id} - Test run not found")
            raise HTTPException(status_code=404, detail="Test run not found")

        _require_inactive_campaign(test_run, session)

        print(f"DELETE /api/v1/test-runs/{test_run_id} - Found test run, status: {test_run.status}")
        
        # If test run is running, stop it first (mark as cancelled)
        if test_run.status == STATUS_RUNNING:
            print(f"DELETE /api/v1/test-runs/{test_run_id} - Stopping running test run before deletion")
            cancellation_manager.cancel(test_run_id)
            test_run.status = STATUS_FAILED
            if not test_run.meta_data:
                test_run.meta_data = {}
            test_run.meta_data["cancelled"] = True
            test_run.meta_data["error"] = "Test run was cancelled during deletion"
            test_run.meta_data["stopped_at"] = datetime.utcnow().isoformat()
            session.commit()
            # Emit cancellation event
            await progress_event_manager.emit_event(
                test_run_id,
                "test_cancelled",
                {
                    "status": "failed",
                    "cancelled": True,
                },
                "Test run was cancelled during deletion"
            )
            # Continue with deletion - don't return here
        
        # Delete the test run (cascade will handle results, conversation_turns, assessments)
        print(f"DELETE /api/v1/test-runs/{test_run_id} - Deleting test run from database")
        from vivasecuris.aiasylum.api.model_history import archive_run
        archive_run(test_run)
        session.delete(test_run)
        session.commit()
        print(f"DELETE /api/v1/test-runs/{test_run_id} - Deletion committed successfully")

        # Clear cancellation flag and unregister task to prevent issues if ID is reused
        cancellation_manager.clear(test_run_id)

        return {"message": "Test run deleted successfully", "id": test_run_id}
    except HTTPException as e:
        print(f"DELETE /api/v1/test-runs/{test_run_id} - HTTPException: {e.status_code} - {e.detail}")
        if session:
            session.rollback()
        raise
    except Exception as e:
        print(f"DELETE /api/v1/test-runs/{test_run_id} - Exception: {str(e)}")
        import traceback
        traceback.print_exc()
        if session:
            session.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete test run: {str(e)}")
    finally:
        if session:
            session.close()


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
        test_run = session.query(TestRun).filter(TestRun.id == test_run_id).first()
        if not test_run:
            raise HTTPException(status_code=404, detail="Test run not found")
        test_cfg = (test_run.meta_data or {}).get("test_config") or {}
        expose_reasoning = bool(
            test_cfg.get("enable_doctor_cot") or test_cfg.get("enable_patient_cot")
        )

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
        return [
            ConversationTurnResponse.from_orm(turn, expose_reasoning=expose_reasoning)
            for turn in turns
        ]
    finally:
        session.close()


def _require_standalone_run(test_run: TestRun) -> None:
    """Campaign workers must keep their scheduler and cancellation ownership."""
    if (test_run.meta_data or {}).get("benchmark_campaign"):
        raise HTTPException(
            status_code=409,
            detail=(
                "This run belongs to a benchmark comparison and is managed by its queue. "
                "Open the benchmark comparison to cancel it or start a new comparison to retry."
            ),
        )


def _require_inactive_campaign(test_run: TestRun, session) -> None:
    """Keep comparison rows and cancellation handles until all workers drain."""
    campaign = (test_run.meta_data or {}).get("benchmark_campaign")
    if not campaign:
        return
    campaign_id = campaign.get("id") if isinstance(campaign, dict) else None
    rows = [test_run]
    if campaign_id:
        rows.extend(
            row for row in session.query(TestRun).filter(TestRun.test_type == "benchmark")
            if isinstance((row.meta_data or {}).get("benchmark_campaign"), dict)
            and row.meta_data["benchmark_campaign"].get("id") == campaign_id
            and row.id != test_run.id
        )
    if any(
        row.status in (STATUS_PENDING, STATUS_RUNNING, STATUS_PAUSED)
        or cancellation_manager.has_active_task(row.id)
        for row in rows
    ):
        raise HTTPException(
            status_code=409,
            detail=(
                "This benchmark comparison still has queued or active work. "
                "Use Cancel comparison and wait for its workers to stop before deleting a run."
            ),
        )


@router.post("/{test_run_id}/start", response_model=TestRunResponse)
async def start_test_run(test_run_id: int, background_tasks: BackgroundTasks):
    """Start/execute a pending test run."""
    import asyncio
    
    runner = TestRunner()
    test_run = runner.get_test_run(test_run_id)
    if not test_run:
        raise HTTPException(status_code=404, detail="Test run not found")

    _require_standalone_run(test_run)

    if test_run.status not in (STATUS_PENDING, STATUS_FAILED, STATUS_PAUSED):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot start test run in status: {test_run.status}. Only pending, failed, or paused test runs can be started."
        )
    
    # Clear any previous cancellation flag
    cancellation_manager.clear(test_run_id)
    
    # Create and register the asyncio task so we can cancel it later
    # Use create_task to get a reference to the task - it will run in the background
    task = asyncio.create_task(_run_test_background(test_run_id))
    cancellation_manager.register_task(test_run_id, task)
    
    # Return the test run (status will be updated by background task)
    return test_run


@router.post("/{test_run_id}/pause")
async def pause_test_run(test_run_id: int):
    """Pause a running test run."""
    import logging
    logger = logging.getLogger(__name__)
    
    session = get_session()
    try:
        test_run = session.query(TestRun).filter(TestRun.id == test_run_id).first()
        if not test_run:
            raise HTTPException(status_code=404, detail="Test run not found")

        _require_standalone_run(test_run)

        if test_run.status != STATUS_RUNNING:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot pause test run in status: {test_run.status}. Only running test runs can be paused."
            )
        
        # Mark for cancellation to stop execution at next checkpoint
        cancellation_manager.cancel(test_run_id)
        logger.info(f"⏸️ Pause request for test run {test_run_id}")
        
        # Update status to paused
        test_run.status = STATUS_PAUSED
        if not test_run.meta_data:
            test_run.meta_data = {}
        test_run.meta_data["paused_at"] = datetime.utcnow().isoformat()
        test_run.meta_data["paused"] = True
        session.commit()
        logger.info(f"✅ Test run {test_run_id} paused")
        
        # Emit pause event
        await progress_event_manager.emit_event(
            test_run_id,
            "test_paused",
            {
                "status": "paused",
                "paused_at": test_run.meta_data["paused_at"],
            },
            "Test run paused. You can resume it later."
        )
        
        return {
            "message": "Test run paused. Execution will stop at next checkpoint.",
            "id": test_run_id,
            "status": "paused"
        }
    finally:
        session.close()


@router.post("/{test_run_id}/resume")
async def resume_test_run(test_run_id: int, background_tasks: BackgroundTasks):
    """Resume a paused test run."""
    import asyncio
    import logging
    logger = logging.getLogger(__name__)
    
    session = get_session()
    try:
        test_run = session.query(TestRun).filter(TestRun.id == test_run_id).first()
        if not test_run:
            raise HTTPException(status_code=404, detail="Test run not found")

        _require_standalone_run(test_run)

        if test_run.status != STATUS_PAUSED:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot resume test run in status: {test_run.status}. Only paused test runs can be resumed."
            )
        
        # Clear cancellation flag and update status
        cancellation_manager.clear(test_run_id)
        test_run.status = STATUS_RUNNING
        if not test_run.meta_data:
            test_run.meta_data = {}
        test_run.meta_data["resumed_at"] = datetime.utcnow().isoformat()
        test_run.meta_data["paused"] = False
        session.commit()
        logger.info(f"▶️ Resume request for test run {test_run_id}")
        
        # Create and register the asyncio task to continue execution
        task = asyncio.create_task(_run_test_background(test_run_id))
        cancellation_manager.register_task(test_run_id, task)
        
        # Emit resume event
        await progress_event_manager.emit_event(
            test_run_id,
            "test_resumed",
            {
                "status": "running",
                "resumed_at": test_run.meta_data["resumed_at"],
            },
            "Test run resumed. Execution will continue from where it left off."
        )
        
        return test_run
    finally:
        session.close()


@router.post("/{test_run_id}/stop")
async def stop_test_run(test_run_id: int):
    """Stop/cancel a running test run."""
    import logging
    logger = logging.getLogger(__name__)
    
    session = get_session()
    try:
        test_run = session.query(TestRun).filter(TestRun.id == test_run_id).first()
        if not test_run:
            raise HTTPException(status_code=404, detail="Test run not found")

        _require_standalone_run(test_run)

        if test_run.status not in (STATUS_RUNNING, STATUS_PAUSED):
            raise HTTPException(
                status_code=400,
                detail=f"Cannot stop test run in status: {test_run.status}. Only running or paused test runs can be stopped."
            )
        
        # Mark for cancellation immediately and cancel the asyncio task
        # This will both set the cancellation flag and cancel the running asyncio task
        cancelled = cancellation_manager.cancel(test_run_id)
        logger.info(f"🛑 Stop request for test run {test_run_id}, cancellation flag set: {cancelled}, task cancelled")
        print(f"[STOP] Test run {test_run_id} marked for cancellation and asyncio task cancelled")
        
        # Immediately update status to failed (cancelled) - this makes it deletable right away
        test_run.status = STATUS_FAILED
        if not test_run.meta_data:
            test_run.meta_data = {}
        test_run.meta_data["cancelled"] = True
        test_run.meta_data["error"] = "Test run was cancelled by user"
        test_run.meta_data["stopped_at"] = datetime.utcnow().isoformat()
        session.commit()
        logger.info(f"✅ Test run {test_run_id} status immediately updated to failed (cancelled) - ready for deletion")
        print(f"[STOP] Test run {test_run_id} status immediately updated to failed - can be deleted now")
        
        # Emit cancellation event immediately
        await progress_event_manager.emit_event(
            test_run_id,
            "test_cancelled",
            {
                "status": "failed",
                "cancelled": True,
                "immediate": True,
            },
            "Test run was cancelled and stopped immediately"
        )
        
        return {
            "message": "Test run stopped immediately. Background execution will terminate at next checkpoint.",
            "id": test_run_id,
            "status": "failed",
            "cancelled": True
        }
    finally:
        session.close()


@router.get("/{test_run_id}/progress")
async def stream_test_run_progress(test_run_id: int):
    """Stream real-time progress updates for a test run using Server-Sent Events."""
    import logging
    logger = logging.getLogger(__name__)
    
    # Verify test run exists and get current status
    session = get_session()
    current_status = None
    try:
        test_run = session.query(TestRun).filter(TestRun.id == test_run_id).first()
        if not test_run:
            logger.warning(f"Test run {test_run_id} not found for progress stream")
            raise HTTPException(status_code=404, detail="Test run not found")
        current_status = test_run.status
        logger.info(f"Starting progress stream for test run {test_run_id} (status: {current_status})")
    finally:
        session.close()
    
    # Stream events as Server-Sent Events
    async def stream_with_status():
        # Send current status immediately when client connects
        if current_status:
            status_event = {
                "test_run_id": test_run_id,
                "event_type": "status_update",
                "timestamp": datetime.utcnow().isoformat(),
                "data": {"status": current_status},
                "message": f"Current status: {current_status}"
            }
            yield f"data: {json.dumps(status_event)}\n\n"
        
        # Then stream live events
        async for event in progress_event_manager.stream_events(test_run_id):
            yield event
    
    return StreamingResponse(
        stream_with_status(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable buffering in nginx
        }
    )
