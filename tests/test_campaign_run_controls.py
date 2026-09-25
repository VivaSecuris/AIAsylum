"""Campaign workers retain ownership when generic test-run controls are used."""

import asyncio
from copy import deepcopy

import pytest
from fastapi import BackgroundTasks, HTTPException

from vivasecuris.aiasylum.api import benchmark_campaigns as campaigns
from vivasecuris.aiasylum.api.cancellation import cancellation_manager
from vivasecuris.aiasylum.api.routes import test_runs
from vivasecuris.aiasylum.database import TestRun, get_session


@pytest.fixture(autouse=True)
def isolate_cancellation(monkeypatch):
    monkeypatch.setattr(cancellation_manager, "_tasks", {})
    monkeypatch.setattr(cancellation_manager, "_cancelled", set())


def create_run(status="pending", *, campaign=True):
    config = {
        "id": "controls-audit", "name": "Comparison", "models": ["model"],
        "benchmarks": ["mmlu"], "num_samples": 1, "seed": 0,
        "max_new_tokens": 256,
    }
    with get_session() as session:
        row = TestRun(
            doctor_provider="transformers", doctor_model="model",
            patient_provider="transformers", patient_model="model",
            test_type="benchmark", status=status,
            meta_data={"benchmark_campaign": config, "benchmark": "mmlu",
                       "campaign_model": "model"} if campaign else {},
        )
        session.add(row)
        session.commit()
        return row.id


async def blocked_worker(*_args):
    await asyncio.Event().wait()


async def invoke_control(action, run_id):
    if action == "pause":
        return await test_runs.pause_test_run(run_id)
    if action == "stop":
        return await test_runs.stop_test_run(run_id)
    route = test_runs.start_test_run if action == "start" else test_runs.resume_test_run
    return await route(run_id, BackgroundTasks())


@pytest.mark.parametrize("action,status", [
    ("start", "pending"), ("start", "failed"), ("start", "paused"),
    ("resume", "paused"), ("pause", "running"),
    ("stop", "running"), ("stop", "paused"),
])
@pytest.mark.parametrize("already_cancelled", [False, True])
@pytest.mark.asyncio
async def test_campaign_controls_reject_without_mutating_worker_or_run(
    monkeypatch, action, status, already_cancelled,
):
    run_id = create_run(status)
    monkeypatch.setattr(test_runs, "_run_test_background", blocked_worker)
    if already_cancelled:
        cancellation_manager.cancel(run_id)
    original = asyncio.create_task(blocked_worker())
    cancellation_manager.register_task(run_id, original)
    with get_session() as session:
        metadata = deepcopy(session.get(TestRun, run_id).meta_data)
    try:
        with pytest.raises(HTTPException) as rejected:
            await invoke_control(action, run_id)
        assert rejected.value.status_code == 409
        assert cancellation_manager._tasks[run_id] is original
        assert cancellation_manager.is_cancelled(run_id) is already_cancelled
        assert original.cancelling() == 0
        with get_session() as session:
            row = session.get(TestRun, run_id)
            assert row.status == status
            assert row.meta_data == metadata
    finally:
        workers = {original, *cancellation_manager._tasks.values()}
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)


@pytest.mark.asyncio
async def test_campaign_cancel_still_reaches_original_worker_after_start_rejected(monkeypatch):
    run_id = create_run()
    monkeypatch.setattr(test_runs, "_run_test_background", blocked_worker)
    original = asyncio.create_task(blocked_worker())
    cancellation_manager.register_task(run_id, original)
    try:
        with pytest.raises(HTTPException) as rejected:
            await test_runs.start_test_run(run_id, BackgroundTasks())
        assert rejected.value.status_code == 409
        assert cancellation_manager._tasks[run_id] is original
        result = campaigns.cancel_campaign("controls-audit")
        assert original.cancelling() == 1
        await asyncio.gather(original, return_exceptions=True)
        assert original.cancelled()
        assert result["status"] == "failed"
        with get_session() as session:
            row = session.get(TestRun, run_id)
            assert row.status == "failed"
            assert row.meta_data["cancelled"] is True
    finally:
        workers = {original, *cancellation_manager._tasks.values()}
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)


@pytest.mark.parametrize("status", ["pending", "failed", "paused"])
@pytest.mark.asyncio
async def test_noncampaign_start_still_schedules_worker(monkeypatch, status):
    run_id = create_run(status, campaign=False)
    started = asyncio.Event()
    calls = []

    async def execute(row_id):
        calls.append(row_id)
        started.set()
        await blocked_worker()

    monkeypatch.setattr(test_runs, "_run_test_background", execute)
    cancellation_manager.cancel(run_id)
    returned = await test_runs.start_test_run(run_id, BackgroundTasks())
    worker = cancellation_manager._tasks[run_id]
    try:
        assert returned.id == run_id
        assert not cancellation_manager.is_cancelled(run_id)
        await asyncio.wait_for(started.wait(), timeout=1)
        assert calls == [run_id]
    finally:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)


@pytest.mark.parametrize("status", ["pending", "running", "paused"])
@pytest.mark.asyncio
async def test_active_campaign_run_cannot_be_deleted(status):
    run_id = create_run(status)
    with get_session() as session:
        metadata = deepcopy(session.get(TestRun, run_id).meta_data)
    with pytest.raises(HTTPException) as rejected:
        await test_runs.delete_test_run(run_id)
    assert rejected.value.status_code == 409
    assert not cancellation_manager.is_cancelled(run_id)
    assert cancellation_manager._tasks == {}
    with get_session() as session:
        row = session.get(TestRun, run_id)
        assert row.status == status
        assert row.meta_data == metadata


@pytest.mark.parametrize("sibling_status", ["pending", "running", "paused"])
@pytest.mark.asyncio
async def test_terminal_campaign_run_cannot_be_deleted_with_active_sibling(sibling_status):
    run_id = create_run("completed")
    sibling_id = create_run(sibling_status)
    with pytest.raises(HTTPException) as rejected:
        await test_runs.delete_test_run(run_id)
    assert rejected.value.status_code == 409
    assert not cancellation_manager.is_cancelled(run_id)
    assert not cancellation_manager.is_cancelled(sibling_id)
    assert cancellation_manager._tasks == {}
    with get_session() as session:
        assert session.get(TestRun, run_id).status == "completed"
        assert session.get(TestRun, sibling_id).status == sibling_status


@pytest.mark.parametrize("draining_sibling", [False, True])
@pytest.mark.asyncio
async def test_terminal_campaign_run_cannot_be_deleted_while_worker_drains(draining_sibling):
    run_id = create_run("failed")
    worker_id = create_run("failed") if draining_sibling else run_id
    cancellation_manager.cancel(worker_id)
    original = asyncio.create_task(blocked_worker())
    cancellation_manager.register_task(worker_id, original)
    with get_session() as session:
        metadata = deepcopy(session.get(TestRun, run_id).meta_data)
    try:
        with pytest.raises(HTTPException) as rejected:
            await test_runs.delete_test_run(run_id)
        assert rejected.value.status_code == 409
        assert cancellation_manager._tasks[worker_id] is original
        assert cancellation_manager.is_cancelled(worker_id)
        assert original.cancelling() == 0
        with get_session() as session:
            row = session.get(TestRun, run_id)
            assert row.status == "failed"
            assert row.meta_data == metadata
            assert session.get(TestRun, worker_id) is not None
    finally:
        original.cancel()
        await asyncio.gather(original, return_exceptions=True)


@pytest.mark.parametrize("status", ["completed", "failed"])
@pytest.mark.parametrize("finished_registration", [False, True])
@pytest.mark.asyncio
async def test_idle_terminal_campaign_run_can_be_deleted(monkeypatch, status, finished_registration):
    from vivasecuris.aiasylum.api import model_history

    run_id = create_run(status)
    archived = []
    monkeypatch.setattr(model_history, "archive_run", lambda row: archived.append(row.id))
    cancellation_manager.cancel(run_id)
    if finished_registration:
        finished = asyncio.create_task(asyncio.sleep(0))
        await finished
        cancellation_manager.register_task(run_id, finished)
    result = await test_runs.delete_test_run(run_id)
    assert result["id"] == run_id
    assert archived == [run_id]
    assert not cancellation_manager.is_cancelled(run_id)
    assert run_id not in cancellation_manager._tasks
    with get_session() as session:
        assert session.get(TestRun, run_id) is None
