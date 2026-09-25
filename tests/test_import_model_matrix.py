"""Import saved validation evidence without models, GPUs, or downloads."""
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import import_model_matrix as importer
from vivasecuris.aiasylum.database import InterpRun


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def matrix(tmp_path):
    root = tmp_path / "matrix"
    out = root / "qwen3-0.6b" / "fingerprint-123"
    source = out / "interp"
    snapshot = "/srv/hf/models--Qwen--Qwen3-0.6B/snapshots/commit-123"
    write(out / "identity.json", {"repo_id": "Qwen/Qwen3-0.6B", "snapshot": snapshot, "implementation": "code-sha"})
    report = {"model": snapshot, "status": "passed", "device": "cuda:0", "dtype": "bfloat16", "layers": 2,
              "gpu": "Test GPU", "started_at": "2026-09-24T19:00:00+00:00", "steps": {}}
    for mode in ("single", "comparison"):
        mode_dir = source / mode
        mode_dir.mkdir(parents=True)
        prompt_a = "Explain autumn leaves." if mode == "single" else "The capital of France is"
        prompt_b = "The capital of Germany is" if mode == "comparison" else None
        meta = {"model": snapshot, "device": "cuda:0", "dtype": "bfloat16", "requested_device": "cuda:0",
                "window": 64, "window_len": 3, "num_layers": 3, "dim_reduction": "pca"}
        if mode == "single":
            meta.update(analysis_mode="single", prompt_preview=prompt_a)
        else:
            meta.update(prompt_a=prompt_a, prompt_b=prompt_b, spike_layer=1)
        write(mode_dir / "meta.json", meta)
        for name in ("pca_payload", "predictions", "attention_payload", "mlp_payload"):
            write(mode_dir / f"{name}.json", {"0": [1, 2]})
        (mode_dir / "dashboard.html").write_text("<!DOCTYPE html><html>Validated dashboard</html>")
        if mode == "single":
            np.save(mode_dir / "activation_norms.npy", np.ones((3, 3)))
            shapes = {"activation_norm_mat": [3, 3]}
        else:
            np.save(mode_dir / "cos_mat.npy", np.ones((3, 3)))
            np.save(mode_dir / "dn_mat.npy", np.zeros((3, 3)))
            write(mode_dir / "patching_results.json", {"experiments": [{"results": [{"recovered": 0.3}]}]})
            shapes = {"cos_mat": [3, 3], "dn_mat": [3, 3]}
        report["steps"][mode] = {
            "status": "passed", "matrix_shapes": shapes, "prompt_a": prompt_a, "prompt_b": prompt_b,
            "completed_at": "2026-09-24T19:01:00+00:00", "options": {
                "device": "cuda:0", "dtype": "bfloat16", "max_len": 128, "window": 64, "topk": 5,
                "dim_reduction": "pca", "enable_attention_capture": True, "enable_mlp_capture": True,
                "enable_patching": mode == "comparison", "enable_qkv_capture": False,
            },
        }
    write(source / "report.json", report)
    write(root / "matrix-report.json", {"action": "run", "status": "passed", "results": {
        "qwen3-0.6b": {"model": "Qwen/Qwen3-0.6B", "snapshot": snapshot, "status": "passed", "out": str(out),
                       "stages": {"interp": {"status": "passed", "report": str(source / "report.json")}}}
    }})
    return root, source, tmp_path / "runs/interp"


def test_import_preserves_provenance_options_and_copies_artifacts(matrix, test_db):
    root, source, destination = matrix
    result = importer.import_matrix(root, destination)
    assert len(result["imported"]) == 2 and not result["refused"]
    rows = test_db.query(InterpRun).order_by(InterpRun.id).all()
    assert [row.mode for row in rows] == ["single", "comparison"]
    for row in rows:
        assert row.status == "completed" and row.model_a == "Qwen/Qwen3-0.6B"
        assert Path(row.out_dir) == destination / str(row.id)
        assert (Path(row.out_dir) / "dashboard.html").read_text() == (source / row.mode / "dashboard.html").read_text()
        assert row.meta_data["options"]["max_len"] == 128
        assert row.meta_data["options"]["enable_patching"] == (row.mode == "comparison")
        assert row.meta_data["summary"]["preflight"]["device"] == "cuda:0"
        assert row.meta_data["matrix_import"]["commit"] == "commit-123"
        assert row.started_at.isoformat() == "2026-09-24T19:00:00"
    assert rows[0].prompt_a == "Explain autumn leaves."
    assert rows[1].prompt_b == "The capital of Germany is"


def test_reimport_is_idempotent_and_source_files_are_independent(matrix, test_db):
    root, source, destination = matrix
    first = importer.import_matrix(root, destination)
    second = importer.import_matrix(root, destination)
    assert [r["id"] for r in first["imported"]] == [r["id"] for r in second["imported"]]
    assert {r["status"] for r in second["imported"]} == {"already_imported"}
    assert test_db.query(InterpRun).count() == 2
    (source / "single/dashboard.html").unlink()
    assert (destination / str(first["imported"][0]["id"]) / "dashboard.html").exists()


@pytest.mark.parametrize("damage", ["missing_comparison", "failed_report", "missing_dashboard", "nan_matrix", "mismatched_model", "missing_options", "symlink"])
def test_incomplete_or_corrupt_outputs_never_create_completed_rows(matrix, test_db, damage):
    root, source, destination = matrix
    report = importer.read_json(source / "report.json")
    if damage == "missing_comparison":
        report["steps"].pop("comparison")
    elif damage == "failed_report":
        report["status"] = "failed"
    elif damage == "missing_dashboard":
        (source / "comparison/dashboard.html").unlink()
    elif damage == "nan_matrix":
        np.save(source / "comparison/cos_mat.npy", np.full((3, 3), np.nan))
    elif damage == "mismatched_model":
        report["model"] = "another model"
    elif damage == "missing_options":
        report["steps"]["single"].pop("options")
    elif damage == "symlink":
        (source / "single/other.json").symlink_to(source / "single/meta.json")
    write(source / "report.json", report)
    result = importer.import_matrix(root, destination)
    assert result["refused"] and not result["imported"]
    assert test_db.query(InterpRun).count() == 0
    assert not destination.exists()


def test_unpassed_entries_are_skipped_and_download_reports_rejected(matrix, test_db):
    root, _, destination = matrix
    manifest = importer.read_json(root / "matrix-report.json")
    manifest["results"]["qwen3-0.6b"]["status"] = "failed"
    write(root / "matrix-report.json", manifest)
    assert len(importer.import_matrix(root, destination)["skipped"]) == 1
    assert test_db.query(InterpRun).count() == 0
    manifest["action"] = "download"
    write(root / "matrix-report.json", manifest)
    with pytest.raises(ValueError, match="download reports"):
        importer.import_matrix(root, destination)


def test_dry_run_has_no_database_or_filesystem_writes(matrix, test_db):
    root, _, destination = matrix
    result = importer.import_matrix(root, destination, dry_run=True)
    assert result["would_import"] == [{"model": "qwen3-0.6b", "modes": ["single", "comparison"]}]
    assert not destination.exists() and test_db.query(InterpRun).count() == 0


def test_copy_failure_rolls_back_both_rows_and_cleans_only_new_directories(matrix, test_db, monkeypatch):
    root, _, destination = matrix
    original = importer.shutil.copy2

    def copy(source, target):
        if source.parent.name == "comparison":
            raise OSError("disk full")
        return original(source, target)

    monkeypatch.setattr(importer.shutil, "copy2", copy)
    result = importer.import_matrix(root, destination)
    assert "disk full" in result["refused"][0]["error"]
    assert test_db.query(InterpRun).count() == 0
    assert [p for p in destination.iterdir() if p.is_dir()] == []


def test_existing_destination_is_never_overwritten(matrix, test_db):
    root, _, destination = matrix
    destination.mkdir(parents=True)
    (destination / "1").mkdir()
    (destination / "1/keep.txt").write_text("existing artifact")
    result = importer.import_matrix(root, destination)
    assert "Refusing to overwrite" in result["refused"][0]["error"]
    assert (destination / "1/keep.txt").read_text() == "existing artifact"
    assert test_db.query(InterpRun).count() == 0


def test_imported_outputs_work_with_history_dashboard_artifact_and_delete_endpoints(matrix, test_db, monkeypatch):
    from fastapi.testclient import TestClient
    from vivasecuris.aiasylum.api.main import app
    from vivasecuris.aiasylum.api.routes import interp

    root, source, destination = matrix
    monkeypatch.setattr(interp, "RUNS_ROOT", destination)
    summary = importer.import_matrix(root, destination)
    client = TestClient(app)
    row = summary["imported"][1]
    base = f"/api/v1/interp/runs/{row['id']}"
    assert client.get(base).json()["status"] == "completed"
    assert "Validated dashboard" in client.get(base + "/dashboard").text
    assert client.get(base + "/artifacts/patching_results.json").status_code == 200
    assert client.delete(base).status_code == 200
    assert not Path(row["out_dir"]).exists()
    assert (source / "comparison/dashboard.html").exists()
