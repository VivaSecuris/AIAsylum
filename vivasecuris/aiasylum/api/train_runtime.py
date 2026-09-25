"""Drive one training worker process and relay its events to a run's reporter.

Training (LoRA, distillation) runs in ``weights.train_worker`` as a separate
process rather than a thread in the API, for three reasons that a direction
edit does not have. Its memory is an order of magnitude larger and fails
late: an allocator that grows past unified memory takes down the process that
owns it, and that must not be the API serving the progress stream. The MPS
fallback switch has to be set before torch is imported, and the API may have
imported torch already. And stopping a backward pass mid-flight is not
something a Python thread can be asked to do; a SIGTERM to the worker's own
process group is.

The worker writes one JSON event per stdout line. This module reads them,
turns ``note``/``step``/``eval``/``count`` into reporter calls, and returns
the ``done`` result. Cancellation is polled between events: the worker gets
SIGTERM, a grace period to save a partial adapter, then SIGKILL.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

WORKER_MODULE = "vivasecuris.aiasylum.weights.train_worker"
EXIT_STOPPED = 3


class TrainingWorkerError(RuntimeError):
    """The worker exited without a ``done`` event."""


def _project_root() -> Path:
    from config import settings

    return Path(settings.project_root)


def stop_process_group(pid: int, grace: float = 30.0, exited: Optional[Callable[[], bool]] = None) -> None:
    """SIGTERM the group, wait up to ``grace`` seconds, then SIGKILL what is left.

    ``exited`` says whether the leader has gone; a Popen owner passes its own
    poll so the child is reaped in one place. Without it (recovery after a
    restart, where the process is not our child) existence is probed directly.
    The worker was started in its own session, so its pid is its group id.
    """
    if exited is None:
        def exited() -> bool:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return True
            return False

    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.time() + grace
    while time.time() < deadline:
        if exited():
            return
        time.sleep(0.2)
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def run_training_worker(
    run_id: int,
    job: Dict[str, Any],
    log_path: Path,
    reporter,
    is_cancelled: Callable[[], bool],
    *,
    on_pid: Optional[Callable[[int], None]] = None,
    python: Optional[str] = None,
    module: str = WORKER_MODULE,
    stop_grace: float = 30.0,
    poll: float = 0.5,
) -> Dict[str, Any]:
    """Run ``job`` in a worker process; return its ``done`` result.

    The returned dict carries ``stopped: True`` when the worker was asked to
    stop (by ``is_cancelled`` here, or a SIGTERM from elsewhere) and saved a
    partial adapter. Raises :class:`TrainingWorkerError` when the worker fails,
    with the worker's own error message when it produced one. Any exception the
    reporter raises (the SSE reporter raises on cancellation) terminates the
    worker before propagating.
    """
    run_dir = Path(job["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    job_path = run_dir / "job.json"
    job_path.write_text(json.dumps(job, indent=1, default=str))

    env = dict(os.environ)
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    env["PYTHONUNBUFFERED"] = "1"

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "a")
    try:
        log.write(f"--- run {run_id}: {module} {job_path}\n")
        log.flush()
        proc = subprocess.Popen(
            [python or sys.executable, "-m", module, str(job_path)],
            stdout=subprocess.PIPE, stderr=log, stdin=subprocess.DEVNULL,
            start_new_session=True, env=env, text=True, cwd=str(_project_root()),
        )
    except Exception:
        log.close()
        raise
    if on_pid is not None:
        on_pid(proc.pid)

    lines: "queue.Queue[Optional[str]]" = queue.Queue()

    def read_stdout() -> None:
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                lines.put(line)
        finally:
            lines.put(None)

    threading.Thread(target=read_stdout, name=f"train-worker-{run_id}", daemon=True).start()

    result: Optional[Dict[str, Any]] = None
    last_error: Optional[Dict[str, Any]] = None
    stopping = False

    def request_stop() -> None:
        nonlocal stopping
        if not stopping:
            stopping = True
            logger.info("Stopping training worker %s for run %s", proc.pid, run_id)
            stop_process_group(proc.pid, grace=stop_grace, exited=lambda: proc.poll() is not None)

    try:
        while True:
            try:
                line = lines.get(timeout=poll)
            except queue.Empty:
                if is_cancelled():
                    request_stop()
                continue
            if line is None:
                break
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except ValueError:
                log.write(f"[non-json stdout] {line}\n")
                continue
            kind = event.get("event")
            if kind == "note":
                reporter.note(str(event.get("message", "")))
            elif kind == "step":
                reporter.metrics({k: v for k, v in event.items() if k != "event"})
            elif kind == "eval":
                reporter.metrics({"phase": "eval", "step": event.get("step"), "eval_loss": event.get("eval_loss")})
            elif kind == "count":
                reporter.count(int(event.get("done", 0)), int(event.get("total", 0)),
                               f"{str(event.get('phase', 'teacher')).replace('_', ' ')} ")
            elif kind == "done":
                result = event.get("result") or {}
            elif kind == "error":
                last_error = event
            if is_cancelled():
                request_stop()
        proc.wait()
    except BaseException:
        # The reporter raised (cancellation), or the API thread is going away:
        # never leave a trainer holding memory behind.
        request_stop()
        proc.wait()
        raise
    finally:
        log.close()

    code = proc.returncode
    if result is not None and (code == 0 or code == EXIT_STOPPED or result.get("stopped")):
        if stopping:
            result["stopped"] = True
        return result
    if stopping or code == EXIT_STOPPED:
        return {"stopped": True, "train": None, "history": [], "output_path": None, "merged": False}
    if last_error is not None:
        message = str(last_error.get("message") or "training worker failed")
        logger.error("Training worker for run %s failed: %s\n%s", run_id, message, last_error.get("traceback", ""))
        raise TrainingWorkerError(message)
    raise TrainingWorkerError(
        f"Training worker exited with status {code} without a result; see {log_path}"
    )
