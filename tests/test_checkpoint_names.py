"""A removed checkpoint path cannot silently become a different graph model."""

import json

import pytest

from vivasecuris.aiasylum.api import model_history
from vivasecuris.aiasylum.api.model_history import archive_run, saved_checkpoint_history
from vivasecuris.aiasylum.api.routes import weights
from vivasecuris.aiasylum.database import WeightRun


@pytest.fixture
def roots(tmp_path, monkeypatch):
    models, runs = tmp_path / "models", tmp_path / "runs" / "weights"
    models.mkdir()
    runs.mkdir(parents=True)
    monkeypatch.setattr(model_history, "_default_project_root", lambda: tmp_path)
    monkeypatch.setattr(weights, "_models_root", lambda: models)
    monkeypatch.setattr(weights, "_runs_root", lambda: runs)
    monkeypatch.setattr(weights, "_interp_extra_installed", lambda: True)
    from vivasecuris.aiasylum.weights import progress
    monkeypatch.setattr(progress, "memory_report", lambda: {"measured": True, "free_gb": 128})
    monkeypatch.setattr(progress, "resident_ollama_models", lambda: [])
    monkeypatch.setattr(progress, "gpu_report", lambda: [])
    return models, runs


def test_completed_checkpoint_name_stays_reserved_after_directory_removal(roots, test_db):
    models, _ = roots
    test_db.add(WeightRun(kind="surgery", status="completed", source_model="org/base", out_dir=str(models / "old-edit")))
    test_db.commit()
    assert not (models / "old-edit").exists()
    result = weights._preflight_checks("surgery", "org/base", output_name="old-edit")
    check = next(c for c in result.checks if c.code == "output_name_reused")
    assert check.severity == "blocking"
    assert check.acknowledgeable is False
    assert "fresh output name" in check.message


def test_archived_completed_checkpoint_reserves_name_after_run_deletion(roots, test_db, tmp_path):
    models, _ = roots
    row = WeightRun(kind="surgery", status="completed", source_model="org/base", out_dir="models/old-edit")
    test_db.add(row)
    test_db.commit()
    archive_run(row, project_root=tmp_path)
    test_db.delete(row)
    test_db.commit()
    assert "archived surgery" in saved_checkpoint_history(models / "old-edit")
    assert "output_name_reused" in weights._preflight_checks("surgery", "org/base", output_name="old-edit").blocking_codes


@pytest.mark.parametrize("evidence", ["completed_run", "cli_manifest"])
def test_imported_saved_checkpoint_reserves_mapped_output_name(roots, tmp_path, evidence):
    models, _ = roots
    imports = tmp_path / "runs" / "model-lineage" / "imports"
    imports.mkdir(parents=True)
    original = "/old/workspace/models/old-edit"
    payload = {"origin": "old-workspace", "model_refs": {original: str(models / "old-edit")}}
    if evidence == "completed_run":
        payload["weight_runs"] = [{"id": 7, "kind": "surgery", "status": "completed", "out_dir": original}]
    else:
        payload["models"] = [{"model_ref": original, "manifest": {"source_model": "org/base", "method": "direction_scale"}}]
    (imports / "workspace.json").write_text(json.dumps(payload))
    assert saved_checkpoint_history(models / "old-edit") is not None
    assert "output_name_reused" in weights._preflight_checks("surgery", "org/base", output_name="old-edit").blocking_codes


def test_failed_jobs_and_missing_comparison_references_do_not_reserve_names(roots, tmp_path, test_db):
    models, _ = roots
    path = models / "unused-name"
    failed = WeightRun(kind="surgery", status="failed", source_model="org/base", out_dir=str(path))
    test_db.add_all([failed, WeightRun(kind="compare", status="completed", source_model="org/base", meta_data={"modified_model": str(path)})])
    test_db.commit()
    archive_run(failed, project_root=tmp_path)
    imports = tmp_path / "runs" / "model-lineage" / "imports"
    imports.mkdir(parents=True)
    (imports / "workspace.json").write_text(json.dumps({"origin": "old", "models": [{"model_ref": str(path), "manifest": None}],
        "weight_runs": [{"id": 7, "kind": "surgery", "status": "failed", "out_dir": str(path)},
                        {"id": 8, "kind": "compare", "status": "completed", "metadata": {"modified_model": str(path)}}]}))
    assert saved_checkpoint_history(path) is None
    assert "output_name_reused" not in weights._preflight_checks("surgery", "org/base", output_name="unused-name").blocking_codes


def test_reused_checkpoint_name_cannot_be_acknowledged_through_api(roots, test_db):
    from fastapi.testclient import TestClient
    from vivasecuris.aiasylum.api.main import app

    models, runs = roots
    direction_path = runs / "direction"
    direction_path.mkdir()
    (direction_path / "direction.safetensors").write_bytes(b"direction")
    direction = WeightRun(kind="direction", source_model="org/base", status="completed", out_dir=str(direction_path),
                          meta_data={"summary": {"auc": 1.0, "model_id": "org/base"}})
    test_db.add_all([direction, WeightRun(kind="surgery", status="completed", source_model="org/base", out_dir=str(models / "old-edit"))])
    test_db.commit()
    before = test_db.query(WeightRun).count()
    response = TestClient(app).post("/api/v1/weights/runs", json={
        "kind": "surgery", "source_model": "org/base", "source_run_id": direction.id,
        "output_name": "old-edit", "acknowledge": ["output_name_reused"],
    })
    assert response.status_code == 409
    assert any(check["code"] == "output_name_reused" for check in response.json()["detail"]["blocking"])
    assert test_db.query(WeightRun).count() == before
