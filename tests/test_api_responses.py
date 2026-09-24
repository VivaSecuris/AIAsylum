"""Tests for API response shapes the frontend depends on, and the settings paths behind them."""

from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from config import settings
from vivasecuris.aiasylum.api.main import app
from vivasecuris.aiasylum.database import Assessment, TestRun


def _completed_run(db_session) -> TestRun:
    test_run = TestRun(
        doctor_provider="ollama",
        doctor_model="llama3.2",
        patient_provider="ollama",
        patient_model="llama2",
        test_type="conversation",
        status="completed",
    )
    db_session.add(test_run)
    db_session.commit()
    db_session.refresh(test_run)
    return test_run


def test_project_root_is_the_repo():
    # _persist_to_env writes API keys to project_root/.env, so this must not drift outside the repo
    assert (settings.project_root / "setup.py").exists()
    assert settings.config_dir == settings.project_root / "config"


def test_test_run_list_includes_timestamps(db_session):
    test_run = _completed_run(db_session)
    with patch("vivasecuris.aiasylum.api.routes.test_runs.get_session", return_value=db_session), \
         patch.object(db_session, "close"):
        response = TestClient(app).get("/api/v1/test-runs/")
    assert response.status_code == 200
    row = next(r for r in response.json() if r["id"] == test_run.id)
    assert row["created_at"] is not None
    assert "updated_at" in row


def test_assessments_expose_metadata_key(db_session):
    test_run = _completed_run(db_session)
    db_session.add(Assessment(
        test_run_id=test_run.id,
        assessment_text="Looks aligned.",
        scores={"safety": 0.7},
        overall_score=0.8,
        meta_data={"cot_analysis": {"cot_detected": True}},
    ))
    db_session.commit()

    with patch("vivasecuris.aiasylum.database.get_session", return_value=db_session), \
         patch.object(db_session, "close"):
        response = TestClient(app).get(f"/api/v1/analysis/test-run/{test_run.id}/assessments")
    assert response.status_code == 200
    assessment = response.json()[0]
    # The frontend reads `metadata`; the ORM attribute name `meta_data` must not leak
    assert assessment["metadata"]["cot_analysis"]["cot_detected"] is True
    assert "meta_data" not in assessment
    assert assessment["created_at"] is not None
    assert assessment["scores"] == {"safety": 0.7}
