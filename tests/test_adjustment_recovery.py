"""Interrupted adjustment application retains real job identity without retries."""

from copy import deepcopy
from datetime import datetime
from types import SimpleNamespace

import pytest
from sqlalchemy import inspect

from vivasecuris.aiasylum.api.adjustment_recovery import (
    find_adjustment_weight_run,
    reconcile_failed_application,
    recover_interrupted_adjustments,
)
from vivasecuris.aiasylum.database import get_session
from vivasecuris.aiasylum.database.models import ModelAdjustment, WeightRun


def _adjustment(session, name, *, mode="weights", status="applying"):
    row = ModelAdjustment(id=name, provider="transformers", model="original/model",
                          mode=mode, status=status, meta_data={
                              "settings": {"system_prompt": " Exact original\n", "seed": 0},
                              "proposal": {"summary": "Keep this proposal"},
                          })
    session.add(row)
    session.flush()
    return row


def _job(session, adjustment_id, *, status="pending", kind="lora", **kw):
    row = WeightRun(kind=kind, source_model="original/model", status=status,
                    meta_data={"adjustment_id": adjustment_id, "dataset": {"sha256": "keep"}}, **kw)
    session.add(row)
    session.flush()
    return row


def _snapshot(row):
    return {attr.key: deepcopy(getattr(row, attr.key)) for attr in inspect(type(row)).column_attrs}


def test_correlation_requires_explicit_metadata_and_lora_kind(test_db):
    assert find_adjustment_weight_run(test_db, "revision") is None
    first = _job(test_db, "revision")
    latest = _job(test_db, "revision", status="completed")
    _job(test_db, "another-revision")
    _job(test_db, "revision", kind="surgery")
    inferred = WeightRun(kind="lora", source_model="original/model", status="pending",
                         out_dir="models/adjust-revision",
                         meta_data={"options": {"notes": "Natural-language adjustment revision"}})
    test_db.add(inferred)
    test_db.commit()
    assert latest.id > first.id
    assert find_adjustment_weight_run(test_db, "revision").id == latest.id
    assert find_adjustment_weight_run(test_db, "missing") is None
    assert find_adjustment_weight_run(test_db, "") is None


def test_startup_recovery_reattaches_jobs_without_changing_them_or_reexecuting(test_db):
    parents, jobs = [], []
    for status in ("pending", "running", "completed", "failed"):
        parent = _adjustment(test_db, f"with-{status}")
        job = _job(test_db, parent.id, status=status, error="Keep existing error" if status == "failed" else None)
        parents.append(parent)
        jobs.append(job)
    missing = _adjustment(test_db, "missing-job")
    profile = _adjustment(test_db, "profile", mode="profile")
    unrelated = [_adjustment(test_db, status, status=status) for status in ("proposed", "training", "applied", "failed")]
    test_db.commit()
    jobs_before = {job.id: _snapshot(job) for job in jobs}
    unrelated_before = {row.id: _snapshot(row) for row in unrelated}
    metadata_before = deepcopy(parents[0].meta_data)

    assert recover_interrupted_adjustments() == {"reattached": 4, "failed": 2}

    test_db.expire_all()
    for parent, job in zip(parents, jobs):
        assert parent.status == "training" and parent.weight_run_id == job.id
        assert parent.meta_data["settings"] == metadata_before["settings"]
        assert parent.meta_data["proposal"] == metadata_before["proposal"]
        assert parent.meta_data["application_recovery"]["weight_run_id"] == job.id
        assert _snapshot(job) == jobs_before[job.id]
    for row in (missing, profile):
        assert row.status == "failed" and row.weight_run_id is None
        assert "Create a new proposal to retry" in row.meta_data["error"]
    for row in unrelated:
        assert _snapshot(row) == unrelated_before[row.id]
    after = {row.id: _snapshot(row) for row in [*parents, missing, profile, *unrelated]}
    assert recover_interrupted_adjustments() == {"reattached": 0, "failed": 0}
    test_db.expire_all()
    assert {row.id: _snapshot(row) for row in [*parents, missing, profile, *unrelated]} == after
    assert test_db.query(WeightRun).count() == len(jobs)


def test_failed_preflight_returns_to_proposed_without_changing_saved_inputs(test_db):
    row = _adjustment(test_db, "preflight")
    test_db.commit()
    before = deepcopy(row.meta_data)
    assert reconcile_failed_application(test_db, row, RuntimeError("Insufficient resources")) is None
    test_db.commit()
    assert row.status == "proposed" and row.weight_run_id is None
    assert row.meta_data == before and test_db.query(WeightRun).count() == 0


def test_created_pending_job_failure_retains_link_and_terminates_pending_state(test_db):
    row = _adjustment(test_db, "partial-creation")
    job = _job(test_db, row.id)
    test_db.commit()
    before = deepcopy(job.meta_data)
    found = reconcile_failed_application(test_db, row, OSError("disk full writing train.jsonl"))
    test_db.commit()
    assert found.id == job.id and row.weight_run_id == job.id
    assert row.status == job.status == "failed"
    assert job.completed_at is not None
    assert "disk full" in job.error and "Create a new proposal to retry" in row.meta_data["error"]
    assert job.meta_data == before


@pytest.mark.parametrize("status", ["running", "completed", "failed"])
def test_failure_reconciliation_does_not_overwrite_running_or_terminal_job(test_db, status):
    row = _adjustment(test_db, "existing-job")
    job = _job(test_db, row.id, status=status, error="original evidence",
               completed_at=datetime(2026, 1, 2) if status != "running" else None)
    test_db.commit()
    before = _snapshot(job)
    reconcile_failed_application(test_db, row, RuntimeError("response failed"))
    test_db.commit()
    assert row.status == "failed" and row.weight_run_id == job.id
    assert _snapshot(job) == before


def test_reconciliation_does_not_reset_a_different_application_state(test_db):
    row = _adjustment(test_db, "already-applied", mode="profile", status="applied")
    test_db.commit()
    before = _snapshot(row)
    assert reconcile_failed_application(test_db, row, RuntimeError("late failure")) is None
    test_db.commit()
    assert _snapshot(row) == before


@pytest.mark.asyncio
async def test_real_weight_creation_persists_correlation_before_dataset_write_failure(test_db, tmp_path, monkeypatch):
    from vivasecuris.aiasylum.api.routes import weights
    from vivasecuris.aiasylum.weights import train_data

    row = _adjustment(test_db, "f1a1a1a1-a1a1-a1a1-a1a1-a1a1a1a1a1a1")
    test_db.commit()
    monkeypatch.setattr(weights, "_models_root", lambda: tmp_path / "models")
    monkeypatch.setattr(weights, "_runs_root", lambda: tmp_path / "runs")
    monkeypatch.setattr(weights, "_preflight_checks", lambda *a, **kw: SimpleNamespace(checks=[]))
    def failed_write(*args, **kwargs):
        # A separate session proves the correlation was committed already.
        with get_session() as session:
            saved = find_adjustment_weight_run(session, row.id)
            assert saved is not None and saved.status == "pending"
        raise OSError("Injected training data write failure")
    monkeypatch.setattr(train_data, "write_jsonl", failed_write)
    launched = []
    async def no_execute(run_id):
        launched.append(run_id)
    monkeypatch.setattr(weights, "_run_weights_background", no_execute)
    request = weights.WeightRunRequest(
        kind="lora", source_model="original/model", output_name=f"adjust-{row.id}",
        adjustment_id=row.id, eval_rows=1,
        dataset_rows=[{"prompt": f"Question {i}", "response": f"Answer {i}"} for i in range(4)],
    )
    with pytest.raises(OSError, match="Injected training data write failure"):
        await weights.create_weight_run(request)
    test_db.expire_all()
    assert test_db.query(WeightRun).count() == 1
    reconcile_failed_application(test_db, row, OSError("Injected training data write failure"))
    test_db.commit()
    assert row.status == "failed" and row.weight_run_id is not None
    assert test_db.get(WeightRun, row.weight_run_id).status == "failed"
    assert not launched


def test_weight_correlation_field_is_bounded_and_optional():
    from vivasecuris.aiasylum.api.routes.weights import WeightRunRequest

    assert WeightRunRequest(kind="lora", source_model="model").adjustment_id is None
    for invalid in ("", "a" * 37):
        with pytest.raises(ValueError):
            WeightRunRequest(kind="lora", source_model="model", adjustment_id=invalid)
