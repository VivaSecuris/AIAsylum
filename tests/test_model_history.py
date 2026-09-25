"""Deleting operational records must not erase the model creation graph."""

import json
from datetime import datetime
from urllib.parse import parse_qs, urlparse

import pytest

from vivasecuris.aiasylum.api import model_history
from vivasecuris.aiasylum.api.model_history import archive_run
from vivasecuris.aiasylum.api.model_lineage import build_model_lineage
from vivasecuris.aiasylum.database import Assessment, InterpRun, TestResult as Result, TestRun as Run, WeightRun


def graph(tmp_path):
    return build_model_lineage(catalog={"models": [], "location": "test server"}, project_root=tmp_path)


def test_archive_and_delete_retains_model_steps_dependencies_and_original_status(tmp_path, test_db):
    path = tmp_path / "models" / "removed-edit"
    direction = WeightRun(id=1, kind="direction", status="completed", source_model="org/base",
                          meta_data={"lineage_id": "direction-uuid", "options": {"seed": 17}, "summary": {"auc": 0.98}})
    surgery = WeightRun(id=2, kind="surgery", status="completed", source_model="org/base", source_run_id=1, out_dir=str(path),
                        meta_data={"lineage_id": "edit-uuid", "source_direction_lineage_id": "direction-uuid", "options": {"beta": 0}})
    test_db.add_all([direction, surgery])
    test_db.commit()
    for row in (direction, surgery):
        archive_run(row, project_root=tmp_path)
        test_db.delete(row)
    test_db.commit()
    result = graph(tmp_path)
    steps = {n["run_id"]: n for n in result["nodes"] if n["run_id"] is not None}
    assert len(steps) == 2
    assert steps[1]["status"] == steps[2]["status"] == "completed"
    assert steps[1]["summary"]["auc"] == 0.98
    assert steps[1]["settings"]["options"]["seed"] == 17
    assert all(n["settings"]["archived_at"] and "run" not in n["links"] for n in steps.values())
    assert any(e["source"] == steps[1]["id"] and e["target"] == steps[2]["id"] for e in result["edges"])
    assert next(n for n in result["nodes"] if n["model_ref"] == str(path))["status"] == "missing"
    assert parse_qs(urlparse(steps[2]["links"]["resume"]).query)["kind"] == ["direction"]


def test_current_database_record_wins_over_its_own_archive(tmp_path, test_db):
    row = InterpRun(mode="single", model_a="org/base", status="completed", meta_data={"lineage_id": "run-uuid"})
    test_db.add(row)
    test_db.commit()
    archive_run(row, project_root=tmp_path)
    result = graph(tmp_path)
    runs = [n for n in result["nodes"] if n["kind"] == "interp"]
    assert len(runs) == 1
    assert runs[0]["settings"]["archived_at"] is None
    assert runs[0]["links"]["run"] == f"/interp/{row.id}"


def test_deleted_legacy_parent_cannot_be_replaced_by_reused_live_integer(tmp_path, test_db):
    old = WeightRun(id=1, kind="direction", status="completed", source_model="org/base", created_at=datetime(2026, 9, 1))
    child = WeightRun(id=2, kind="surgery", status="completed", source_model="org/base", source_run_id=1,
                      created_at=datetime(2026, 9, 2), meta_data={"lineage_id": "child-uuid"})
    test_db.add_all([old, child])
    test_db.commit()
    archive_run(old, project_root=tmp_path)
    test_db.delete(old)
    test_db.commit()
    new = WeightRun(id=1, kind="direction", source_model="org/base", status="completed",
                    created_at=datetime(2026, 9, 3), meta_data={"lineage_id": "new-direction-uuid"})
    test_db.add(new)
    test_db.commit()
    result = graph(tmp_path)
    old_node = next(n for n in result["nodes"] if n["kind"] == "direction" and n["settings"]["archived_at"])
    new_node = next(n for n in result["nodes"] if n["id"].endswith(":new-direction-uuid"))
    child_node = next(n for n in result["nodes"] if n["id"].endswith(":child-uuid"))
    assert any(e["source"] == old_node["id"] and e["target"] == child_node["id"] for e in result["edges"])
    assert not any(e["source"] == new_node["id"] and e["target"] == child_node["id"] for e in result["edges"])
    assert "source_run_id" not in parse_qs(urlparse(child_node["links"]["resume"]).query)


def test_two_reused_legacy_generations_remain_distinct(tmp_path, test_db):
    old = WeightRun(id=1, kind="direction", status="completed", source_model="org/base", created_at=datetime(2026, 9, 1))
    test_db.add(old)
    test_db.commit()
    archive_run(old, project_root=tmp_path)
    test_db.delete(old)
    test_db.commit()
    new = WeightRun(id=1, kind="direction", status="completed", source_model="org/base", created_at=datetime(2026, 9, 3))
    test_db.add(new)
    test_db.commit()
    before = [n["id"] for n in graph(tmp_path)["nodes"] if n["kind"] == "direction"]
    assert len(set(before)) == 2
    archive_run(new, project_root=tmp_path)
    test_db.delete(new)
    test_db.commit()
    after = [n["id"] for n in graph(tmp_path)["nodes"] if n["kind"] == "direction"]
    assert before == after


def test_archive_write_failure_propagates_and_preserves_existing_snapshot(tmp_path, test_db, monkeypatch):
    row = WeightRun(kind="direction", source_model="org/base", meta_data={"lineage_id": "archive-uuid"})
    test_db.add(row)
    test_db.commit()
    path = archive_run(row, project_root=tmp_path)
    previous = path.read_bytes()

    def fail_replace(*args):
        raise OSError("disk failure")

    monkeypatch.setattr(model_history.os, "replace", fail_replace)
    with pytest.raises(OSError, match="disk failure"):
        archive_run(row, project_root=tmp_path)
    assert path.read_bytes() == previous
    assert not list(path.parent.glob(".archive-*.tmp"))
    assert test_db.query(WeightRun).count() == 1


def test_test_run_archive_keeps_compact_results_without_transcripts(tmp_path, test_db):
    row = Run(test_type="conversation", patient_provider="transformers", patient_model="org/base",
              doctor_provider="openai", doctor_model="doctor", meta_data={"test_config": {"max_turns": 5}})
    test_db.add(row)
    test_db.flush()
    test_db.add_all([
        Result(test_run_id=row.id, test_name="capability", input_prompt="private prompt", output_response="large transcript", score=0.75, scores={"accuracy": 0.75}),
        Assessment(test_run_id=row.id, assessment_text="long assessment", overall_score=0.8, scores={"safety": 0.8}, analysis_type="final"),
    ])
    test_db.commit()
    path = archive_run(row, project_root=tmp_path)
    payload = json.loads(path.read_text())
    assert "large transcript" not in path.read_text()
    assert "long assessment" not in path.read_text()
    assert payload["row"]["metadata"]["summary"]["assessments"][0]["overall_score"] == 0.8
    test_db.delete(row)
    test_db.commit()
    archived = next(n for n in graph(tmp_path)["nodes"] if n["kind"] == "test")
    assert archived["settings"]["archived_at"]
    assert archived["summary"]["results"][0]["score"] == 0.75
    assert archived["summary"]["outcomes_recorded"] is True
