"""Restart recovery must distinguish abandoned work from live model jobs."""

import asyncio
from contextlib import asynccontextmanager
from copy import deepcopy
from datetime import datetime, timezone
import sys

import pytest
from sqlalchemy import inspect

from vivasecuris.aiasylum.api import model_jobs
from vivasecuris.aiasylum.database import get_session
from vivasecuris.aiasylum.database.models import InterpRun, WeightRun


@pytest.fixture(autouse=True)
def isolated_model_slot(tmp_path, monkeypatch):
    """Never contend with a developer's actual model worker."""
    lock_path = tmp_path / "model-job.lock"
    monkeypatch.setenv("AIASYLUM_MODEL_LOCK", str(lock_path))
    monkeypatch.setattr(model_jobs, "_slot", None)
    monkeypatch.setattr(model_jobs, "_held_by", None)
    monkeypatch.setattr(model_jobs, "_waiting", [])
    return lock_path


def _row(model, row_id, status, **kwargs):
    if model is InterpRun:
        return model(id=row_id, mode="single", model_a="test/model", status=status, **kwargs)
    return model(id=row_id, kind="surgery", source_model="test/model", status=status, **kwargs)


def _state(model, row_id):
    with get_session() as session:
        row = session.get(model, row_id)
        if row is None:
            return None
        return {
            attr.key: deepcopy(getattr(row, attr.key))
            for attr in inspect(model).column_attrs
        }


def _utc(value):
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


@asynccontextmanager
async def _external_lock(path):
    """An independent process owns flock until its stdin closes."""
    pytest.importorskip("fcntl")
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "import fcntl, sys; "
        "handle = open(sys.argv[1], 'a'); "
        "fcntl.flock(handle, fcntl.LOCK_EX); "
        "print('locked', flush=True); "
        "sys.stdin.read()",
        str(path),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        assert await asyncio.wait_for(process.stdout.readline(), timeout=5) == b"locked\n"
        yield process
    finally:
        process.stdin.close()
        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
        assert process.returncode == 0


async def _wait_for_recovery_queue(task):
    async def queued():
        while "server restart recovery" not in model_jobs.slot_status()["waiting"]:
            if task.done():
                await task
                pytest.fail("Recovery finished before waiting for the occupied model slot")
            await asyncio.sleep(0.01)

    await asyncio.wait_for(queued(), timeout=2)


def test_capture_is_read_only_and_keeps_table_id_spaces_separate(test_db):
    from vivasecuris.aiasylum.api.job_recovery import capture_interrupted_model_jobs

    statuses = {
        InterpRun: {1: "running", 2: "completed", 3: "failed", 4: "pending"},
        WeightRun: {1: "completed", 2: "pending", 3: "failed", 4: "running"},
    }
    for model, rows in statuses.items():
        test_db.add_all(_row(model, row_id, status) for row_id, status in rows.items())
    test_db.commit()
    before = {(model, row_id): _state(model, row_id) for model, rows in statuses.items() for row_id in rows}

    snapshot = capture_interrupted_model_jobs()

    assert set(snapshot) == {"interp", "weights"}
    assert sorted(snapshot["interp"]) == [1, 4]
    assert sorted(snapshot["weights"]) == [2, 4]
    for (model, row_id), original in before.items():
        assert _state(model, row_id) == original


@pytest.mark.asyncio
async def test_recovery_preserves_artifacts_config_and_lineage_and_is_idempotent(test_db, tmp_path):
    from vivasecuris.aiasylum.api.job_recovery import (
        capture_interrupted_model_jobs,
        recover_interrupted_model_jobs,
    )

    started_at = datetime(2025, 1, 2, 3, 4, 5)
    terminal_at = datetime(2025, 1, 2, 4, 5, 6)
    artifact_dir = tmp_path / "partial-artifacts"
    artifact_dir.mkdir()
    artifact = artifact_dir / "manifest.json"
    artifact.write_text('{"existing": "partial artifact"}\n')
    metadata = {
        "config": {"layers": [3, 7], "scale": 0.5},
        "source_direction": {"run_id": 28, "layer_scores": {"3": 0.9}},
        "provenance": {"parent_model": "test/original", "sha256": "abc123"},
        "progress": {"completed_steps": 4},
    }
    active = ((InterpRun, 1, "running"), (InterpRun, 4, "pending"),
              (WeightRun, 2, "pending"), (WeightRun, 4, "running"))
    for model, row_id, status in active:
        extra = {"source_run_id": 28, "artifact_bytes": 1234} if model is WeightRun else {"prompt_a": "Preserve this prompt"}
        test_db.add(_row(model, row_id, status, out_dir=str(artifact_dir),
                         started_at=started_at, meta_data=deepcopy(metadata), **extra))
    terminal = ((InterpRun, 2, "completed"), (InterpRun, 3, "failed"),
                (WeightRun, 1, "completed"), (WeightRun, 3, "failed"))
    for model, row_id, status in terminal:
        test_db.add(_row(model, row_id, status, completed_at=terminal_at,
                         error="Original diagnostic", meta_data={"original": True}))
    test_db.commit()
    tracked = active + terminal
    before = {(model, row_id): _state(model, row_id) for model, row_id, _ in tracked}
    snapshot = capture_interrupted_model_jobs()
    # Jobs accepted after the startup snapshot belong to this server lifetime.
    for model in (InterpRun, WeightRun):
        test_db.add(_row(model, 20, "pending", meta_data={"new": True}))
    test_db.commit()
    new_before = {model: _state(model, 20) for model in (InterpRun, WeightRun)}

    earliest = datetime.now(timezone.utc)
    assert await recover_interrupted_model_jobs(snapshot) == {"interp": 2, "weights": 2}
    latest = datetime.now(timezone.utc)

    changed_fields = {"status", "completed_at", "updated_at", "error", "meta_data"}
    for model, row_id, previous_status in active:
        recovered = _state(model, row_id)
        assert recovered["status"] == "failed"
        assert earliest <= _utc(recovered["completed_at"]) <= latest
        assert "restart" in recovered["error"].lower()
        assert "retry" in recovered["error"].lower()
        assert recovered["meta_data"]["interrupted"] is True
        recovery = recovered["meta_data"]["recovery"]
        assert recovery["previous_status"] == previous_status
        assert earliest <= _utc(recovery["recovered_at"]) <= latest
        for key, value in metadata.items():
            assert recovered["meta_data"][key] == value
        for key, value in before[model, row_id].items():
            if key not in changed_fields:
                assert recovered[key] == value
    for model, row_id, _ in terminal:
        assert _state(model, row_id) == before[model, row_id]
    for model in (InterpRun, WeightRun):
        assert _state(model, 20) == new_before[model]
    assert artifact.read_text() == '{"existing": "partial artifact"}\n'
    after = {(model, row_id): _state(model, row_id) for model, row_id, _ in tracked}

    assert await recover_interrupted_model_jobs(snapshot) == {"interp": 0, "weights": 0}
    assert capture_interrupted_model_jobs() == {"interp": [20], "weights": [20]}
    for (model, row_id), original in after.items():
        assert _state(model, row_id) == original


@pytest.mark.asyncio
async def test_recovery_rechecks_captured_rows_before_updating(test_db):
    from vivasecuris.aiasylum.api.job_recovery import (
        capture_interrupted_model_jobs,
        recover_interrupted_model_jobs,
    )

    for model in (InterpRun, WeightRun):
        test_db.add_all(_row(model, row_id, "running") for row_id in (1, 2, 3, 4))
    test_db.commit()
    snapshot = capture_interrupted_model_jobs()
    for model in (InterpRun, WeightRun):
        test_db.get(model, 1).status = "completed"
        test_db.get(model, 1).completed_at = datetime(2025, 1, 2)
        test_db.get(model, 1).meta_data = {"result": "finished during restart"}
        test_db.get(model, 2).status = "failed"
        test_db.get(model, 2).error = "Original worker failure"
        test_db.delete(test_db.get(model, 3))
        # JSON null should be handled just like an empty metadata dictionary.
        test_db.get(model, 4).meta_data = None
    test_db.commit()
    terminal_before = {(model, row_id): _state(model, row_id) for model in (InterpRun, WeightRun) for row_id in (1, 2, 3)}

    assert await recover_interrupted_model_jobs(snapshot) == {"interp": 1, "weights": 1}

    for (model, row_id), original in terminal_before.items():
        assert _state(model, row_id) == original
    for model in (InterpRun, WeightRun):
        recovered = _state(model, 4)
        assert recovered["status"] == "failed"
        assert recovered["meta_data"]["interrupted"] is True
        assert recovered["meta_data"]["recovery"]["previous_status"] == "running"


@pytest.mark.asyncio
async def test_recovery_waits_for_external_flock_and_rechecks_live_worker_completion(test_db, isolated_model_slot):
    from vivasecuris.aiasylum.api.job_recovery import (
        capture_interrupted_model_jobs,
        recover_interrupted_model_jobs,
    )

    test_db.add(_row(InterpRun, 1, "running"))
    test_db.add(_row(WeightRun, 1, "running"))
    test_db.commit()
    snapshot = capture_interrupted_model_jobs()
    before = {model: _state(model, 1) for model in (InterpRun, WeightRun)}
    task = None
    try:
        async with _external_lock(isolated_model_slot):
            task = asyncio.create_task(recover_interrupted_model_jobs(snapshot))
            await _wait_for_recovery_queue(task)
            assert not task.done()
            for model in (InterpRun, WeightRun):
                assert _state(model, 1) == before[model]
            # A standalone worker can finish while recovery waits for its lock.
            with get_session() as session:
                row = session.get(WeightRun, 1)
                row.status = "completed"
                row.completed_at = datetime(2025, 2, 3)
                row.meta_data = {"result": "external worker finished"}
                session.commit()
            completed = _state(WeightRun, 1)
        assert await asyncio.wait_for(task, timeout=3) == {"interp": 1, "weights": 0}
    finally:
        if task is not None and not task.done():
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    assert _state(InterpRun, 1)["status"] == "failed"
    assert _state(WeightRun, 1) == completed
    assert model_jobs.slot_status() == {"held_by": None, "waiting": []}
    handle = model_jobs.try_process_lock()
    assert handle is not None
    handle.close()


@pytest.mark.asyncio
async def test_cancelled_recovery_wait_leaves_rows_recoverable_and_releases_slot(test_db, isolated_model_slot):
    from vivasecuris.aiasylum.api.job_recovery import (
        capture_interrupted_model_jobs,
        recover_interrupted_model_jobs,
    )

    for model in (InterpRun, WeightRun):
        test_db.add(_row(model, 1, "running", meta_data={"checkpoint": 4}))
    test_db.commit()
    snapshot = capture_interrupted_model_jobs()
    before = {model: _state(model, 1) for model in (InterpRun, WeightRun)}

    async with _external_lock(isolated_model_slot):
        task = asyncio.create_task(recover_interrupted_model_jobs(snapshot))
        try:
            await _wait_for_recovery_queue(task)
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        for model in (InterpRun, WeightRun):
            assert _state(model, 1) == before[model]
        assert capture_interrupted_model_jobs() == snapshot
        assert model_jobs.slot_status() == {"held_by": None, "waiting": []}
        assert not model_jobs.model_slot().locked()

    handle = model_jobs.try_process_lock()
    assert handle is not None
    handle.close()
    assert await asyncio.wait_for(recover_interrupted_model_jobs(snapshot), timeout=3) == {"interp": 1, "weights": 1}
    assert model_jobs.slot_status() == {"held_by": None, "waiting": []}


@pytest.mark.asyncio
async def test_app_startup_schedules_recovery_without_blocking_and_shutdown_cancels_it(test_db, monkeypatch):
    from vivasecuris.aiasylum.api import benchmark_campaigns, main

    monkeypatch.setattr(benchmark_campaigns, "recover_interrupted_campaigns", lambda: None)
    monkeypatch.setattr(main.app.state, "model_job_recovery", None, raising=False)
    test_db.add(_row(InterpRun, 1, "running", meta_data={"checkpoint": 3}))
    test_db.commit()
    before = _state(InterpRun, 1)

    async with model_jobs.hold("live model worker"):
        # Startup must return while the slot remains occupied so health/routes
        # remain available. Shutdown must also finish without waiting for it.
        await asyncio.wait_for(main.startup_event(), timeout=1)
        task = main.app.state.model_job_recovery
        try:
            await _wait_for_recovery_queue(task)
            assert not task.done()
            assert _state(InterpRun, 1) == before
        finally:
            await asyncio.wait_for(main.shutdown_event(), timeout=1)
        assert task.cancelled()
        assert _state(InterpRun, 1) == before
        assert model_jobs.slot_status() == {"held_by": "live model worker", "waiting": []}

    assert model_jobs.slot_status() == {"held_by": None, "waiting": []}
