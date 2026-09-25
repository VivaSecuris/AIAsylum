"""Execute a saved benchmark in the configured, isolated model runtime."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import signal

from config import settings
from vivasecuris.aiasylum.database import TestRun, get_session


def _record_failure(test_run_id: int, message: str, *, cancelled: bool = False) -> None:
    with get_session() as session:
        run = session.get(TestRun, test_run_id)
        if run is None or run.status == "completed":
            return
        run.status = "failed"
        run.meta_data = {**(run.meta_data or {}), "error": message, **({"cancelled": True} if cancelled else {})}
        session.commit()


async def _stop_process(process) -> None:
    """Wait for the process group to exit before letting another GPU job start."""
    if process.returncode is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        await asyncio.wait_for(process.wait(), timeout=10)
    except asyncio.TimeoutError:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await process.wait()


async def _finish_despite_cancellation(task):
    """Repeated cancel requests must not interrupt child discovery or reaping."""
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
    return task.result()


async def execute_benchmark_job(test_run_id: int) -> None:
    """Run an existing DB row; child processes share the model-job file lock.

    Settings are server-controlled, never supplied in an HTTP request. The
    child inherits the configured database, model cache and provider settings.
    Child stdout goes to a private per-run log, not the HTTP response.
    """
    if not settings.benchmark_python:
        from vivasecuris.aiasylum.runner import TestRunner
        await TestRunner().execute_test_run(test_run_id)
        return

    root = settings.project_root
    executable = Path(settings.benchmark_python).expanduser()
    if not executable.is_absolute():
        executable = root / executable
    if not executable.is_file() or not os.access(executable, os.X_OK):
        message = "The configured benchmark Python executable is unavailable"
        _record_failure(test_run_id, message)
        raise RuntimeError(message)
    log_dir = root / "runs" / "benchmark-jobs"
    log_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    log_path = log_dir / f"{int(test_run_id)}.log"
    env = dict(os.environ)
    env.update(DATABASE_URL=settings.database_url, AIASYLUM_BENCHMARK_RUNTIME="1", PYTHONUNBUFFERED="1")
    # The same resolved path is used even if the API was started elsewhere.
    lock_path = Path(env.get("AIASYLUM_MODEL_LOCK", "runs/.model-job.lock"))
    env["AIASYLUM_MODEL_LOCK"] = str(lock_path.resolve())
    with get_session() as session:
        run = session.get(TestRun, test_run_id)
        if run is None:
            raise ValueError(f"Test run {test_run_id} does not exist")
        if run.test_type != "benchmark":
            raise ValueError("The benchmark runtime only executes benchmark runs")
        run.meta_data = {**(run.meta_data or {}), "benchmark_runtime": {
            "python": str(executable), "log": str(log_path.relative_to(root)), "isolated": True,
        }}
        session.commit()
    process = None
    spawn_task = None
    log = log_path.open("a")
    try:
        log_path.chmod(0o600)
        spawn_task = asyncio.create_task(asyncio.create_subprocess_exec(
                str(executable), str(root / "scripts" / "run_benchmark_job.py"), str(test_run_id),
                cwd=root, env=env, stdout=log, stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
        ))
        process = await asyncio.shield(spawn_task)
        returncode = await process.wait()
        if returncode:
            message = f"Benchmark worker exited with status {returncode}; see {log_path.relative_to(root)}"
            # Keep the more useful runner error when it has already been saved.
            with get_session() as session:
                run = session.get(TestRun, test_run_id)
                saved_error = (run.meta_data or {}).get("error") if run else None
            _record_failure(test_run_id, saved_error or message)
            raise RuntimeError(saved_error or message)
    except asyncio.CancelledError:
        if process is None and spawn_task is not None:
            try:
                process = await _finish_despite_cancellation(spawn_task)
            except Exception:
                pass
        if process is not None:
            await _finish_despite_cancellation(asyncio.create_task(_stop_process(process)))
        _record_failure(test_run_id, "Benchmark cancelled; worker stopped", cancelled=True)
        raise
    except Exception as exc:
        if process is not None:
            await _stop_process(process)
        _record_failure(test_run_id, str(exc))
        raise
    finally:
        log.close()
