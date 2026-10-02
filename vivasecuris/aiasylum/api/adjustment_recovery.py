"""Reconcile interrupted adjustment application without executing model work.

Startup recovery runs before requests are accepted. The ordinary weight-job
recovery owns interrupted training; this module only restores its parent link.
"""

from datetime import datetime

from sqlalchemy.orm import Session

from vivasecuris.aiasylum.constants import STATUS_FAILED, STATUS_PENDING
from vivasecuris.aiasylum.database import get_session
from vivasecuris.aiasylum.database.models import ModelAdjustment, WeightRun


def find_adjustment_weight_run(session: Session, adjustment_id: str) -> WeightRun | None:
    """Find the latest explicitly correlated LoRA job, never infer from a path."""
    if not adjustment_id:
        return None
    return session.query(WeightRun).filter(
        WeightRun.kind == "lora",
        WeightRun.meta_data["adjustment_id"].as_string() == adjustment_id,
    ).order_by(WeightRun.id.desc()).first()


def recover_interrupted_adjustments() -> dict[str, int]:
    """Resolve old applying states once at startup; never retry an operation."""
    counts = {"reattached": 0, "failed": 0}
    now = datetime.utcnow().isoformat() + "Z"
    with get_session() as session:
        for row in session.query(ModelAdjustment).filter_by(status="applying"):
            job = find_adjustment_weight_run(session, row.id) if row.mode == "weights" else None
            if job is not None:
                row.weight_run_id = job.id
                row.status = "training"
                counts["reattached"] += 1
            else:
                row.status = "failed"
                row.meta_data = {**(row.meta_data or {}), "error": (
                    "Server restarted before this adjustment finished applying. "
                    "No completed application was recorded. Create a new proposal to retry."
                )}
                counts["failed"] += 1
            row.meta_data = {**(row.meta_data or {}), "application_recovery": {
                "previous_status": "applying", "recovered_at": now,
                "weight_run_id": job.id if job is not None else None,
            }}
        session.commit()
    return counts


def reconcile_failed_application(
    session: Session, row: ModelAdjustment, error: Exception,
) -> WeightRun | None:
    """Reconcile a failed apply after rollback; the caller commits these changes.

    A preflight rejection can be retried. Once a job was created, retain its
    identity and failure instead of presenting the operation as unstarted. Only
    pending jobs are marked failed here; a running or terminal job owns its state.
    """
    if row.status != "applying":
        return None
    job = find_adjustment_weight_run(session, row.id) if row.mode == "weights" else None
    detail = str(getattr(error, "detail", None) or error)
    if job is None:
        row.status = "proposed"
    else:
        message = f"Adjustment application failed after creating training run {job.id}: {detail}"
        if job.status == STATUS_PENDING:
            job.status = STATUS_FAILED
            job.completed_at = datetime.utcnow()
            job.error = message
        row.weight_run_id = job.id
        row.status = "failed"
        row.meta_data = {**(row.meta_data or {}), "error": message + " Create a new proposal to retry."}
    session.flush()
    return job
