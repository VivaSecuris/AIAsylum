"""The isolated benchmark process must fail and cancel without orphaning a GPU job."""

import asyncio
import os
from pathlib import Path
import sys

import pytest

from config.settings import Settings
from vivasecuris.aiasylum.api import benchmark_runtime as runtime
from vivasecuris.aiasylum.database import TestRun


def saved_run(session):
    row = TestRun(doctor_provider="transformers", doctor_model="m", patient_provider="transformers",
                  patient_model="m", test_type="benchmark", status="pending")
    session.add(row)
    session.commit()
    return row.id


def test_missing_worker_marks_failure(test_db, monkeypatch):
    run_id = saved_run(test_db)
    monkeypatch.setattr(runtime.settings, "benchmark_python", "/missing/benchmark-python")
    with pytest.raises(RuntimeError, match="unavailable"):
        asyncio.run(runtime.execute_benchmark_job(run_id))
    test_db.expire_all()
    row = test_db.get(TestRun, run_id)
    assert row.status == "failed"
    assert "unavailable" in row.meta_data["error"]


def test_failed_child_is_visible_and_logs_are_private(test_db, monkeypatch, tmp_path):
    run_id = saved_run(test_db)
    monkeypatch.setattr(runtime.settings, "benchmark_python", sys.executable)
    monkeypatch.setattr(Settings, "project_root", property(lambda _: tmp_path))
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/run_benchmark_job.py").write_text("print('load failed')\nraise SystemExit(7)\n")
    with pytest.raises(RuntimeError, match="status 7"):
        asyncio.run(runtime.execute_benchmark_job(run_id))
    test_db.expire_all()
    row = test_db.get(TestRun, run_id)
    assert row.status == "failed"
    log = tmp_path / f"runs/benchmark-jobs/{run_id}.log"
    assert "load failed" in log.read_text()
    assert log.stat().st_mode & 0o777 == 0o600


@pytest.mark.skipif(os.name != "posix", reason="GPU workers use POSIX process groups")
@pytest.mark.parametrize("delay_spawn_return", [False, True])
def test_cancellation_reaps_child_before_returning(test_db, monkeypatch, tmp_path, delay_spawn_return):
    run_id = saved_run(test_db)
    monkeypatch.setattr(runtime.settings, "benchmark_python", sys.executable)
    monkeypatch.setattr(Settings, "project_root", property(lambda _: tmp_path))
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/run_benchmark_job.py").write_text(
        "from pathlib import Path\nimport os,time\nPath('child.pid').write_text(str(os.getpid()))\ntime.sleep(60)\n"
    )
    if delay_spawn_return:
        real_spawn = asyncio.create_subprocess_exec

        async def delayed_spawn(*args, **kwargs):
            process = await real_spawn(*args, **kwargs)
            await asyncio.sleep(0.15)
            return process

        monkeypatch.setattr(runtime.asyncio, "create_subprocess_exec", delayed_spawn)

    async def cancel():
        task = asyncio.create_task(runtime.execute_benchmark_job(run_id))
        try:
            async with asyncio.timeout(5):
                while not (tmp_path / "child.pid").exists():
                    await asyncio.sleep(0.02)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    asyncio.run(cancel())
    pid = int((tmp_path / "child.pid").read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    test_db.expire_all()
    row = test_db.get(TestRun, run_id)
    assert row.status == "failed"
    assert row.meta_data["cancelled"] is True
