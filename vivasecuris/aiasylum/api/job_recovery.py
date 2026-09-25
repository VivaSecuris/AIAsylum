"""Recover abandoned model jobs when the single-process API starts.

These jobs run in API-owned threads, not a persistent queue. Capture their IDs
before accepting requests so delayed recovery cannot invalidate newly submitted
work. The shared model lock also lets an exiting worker finish before deciding
whether its recorded result was interrupted.
"""

from __future__ import annotations

from datetime import datetime
import logging

from vivasecuris.aiasylum.api.model_jobs import hold
from vivasecuris.aiasylum.constants import STATUS_FAILED, STATUS_PENDING, STATUS_RUNNING
from vivasecuris.aiasylum.database import InterpRun, WeightRun, get_session

logger = logging.getLogger(__name__)
_MODELS = {"interp": InterpRun, "weights": WeightRun}
_ACTIVE = (STATUS_PENDING, STATUS_RUNNING)


def capture_interrupted_model_jobs() -> dict[str, list[int]]:
    """Snapshot old jobs before this API can accept new submissions."""
    with get_session() as session:
        return {
            name: [row.id for row in session.query(model.id).filter(model.status.in_(_ACTIVE))]
            for name, model in _MODELS.items()
        }


async def recover_interrupted_model_jobs(snapshot: dict[str, list[int]]) -> dict[str, int]:
    """Record interruption without retrying work or modifying any artifacts.

    Recheck status under the GPU lock: an old worker may have completed while we
    waited. This runs as a background startup task so an external validation job
    holding the GPU does not prevent the API and saved results from opening.
    """
    counts = {name: 0 for name in _MODELS}
    if not any(snapshot.values()):
        return counts
    async with hold("server restart recovery"):
        now = datetime.utcnow()
        with get_session() as session:
            for name, model in _MODELS.items():
                ids = snapshot.get(name, [])
                if not ids:
                    continue
                rows = session.query(model).filter(model.id.in_(ids), model.status.in_(_ACTIVE))
                for row in rows:
                    previous_status = row.status
                    # A training run's worker process outlives the API thread that
                    # spawned it. Left alone it keeps training into a row nobody
                    # will read, holding unified memory the whole time.
                    _stop_orphaned_worker(name, row)
                    row.status = STATUS_FAILED
                    row.completed_at = now
                    row.error = (
                        "Server restarted before this run finished. "
                        "Its settings and any saved artifacts were preserved. "
                        "Create a new run to retry."
                    )
                    row.meta_data = {
                        **(row.meta_data or {}),
                        "interrupted": True,
                        "recovery": {
                            "previous_status": previous_status,
                            "recovered_at": now.isoformat() + "Z",
                        },
                    }
                    counts[name] += 1
            session.commit()
    logger.info("Recovered interrupted model jobs: %s", counts)
    return counts


def _stop_orphaned_worker(name: str, row) -> None:
    pid = (row.meta_data or {}).get("worker_pid") if name == "weights" else None
    if not pid:
        return
    try:
        from vivasecuris.aiasylum.api.train_runtime import stop_process_group

        stop_process_group(int(pid), grace=10.0)
        logger.info("Stopped orphaned training worker %s for weight run %s", pid, row.id)
    except Exception:
        logger.warning("Could not stop training worker %s for weight run %s", pid, row.id, exc_info=True)


def log_recovery_result(task) -> None:
    """Consume a background exception and make failed recovery visible in logs."""
    if task.cancelled():
        return
    error = task.exception()
    if error is not None:
        logger.error(
            "Could not recover interrupted model jobs",
            exc_info=(type(error), error, error.__traceback__),
        )
