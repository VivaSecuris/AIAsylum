"""Graph edges describe recorded provenance, including portable histories."""

import json
import socket
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from vivasecuris.aiasylum.api.model_lineage import build_model_lineage
from vivasecuris.aiasylum.database import InterpRun, TestRun as Run, WeightRun


def inventory(*items):
    return {"location": "test server", "models": [{
        "name": ref.rsplit("/", 1)[-1], "model_ref": ref,
        "kind": "custom" if Path(ref).is_absolute() else "base",
        "availability": "ready", "created_at": None, "source_model": (manifest or {}).get("source_model"),
        "reason": None, "manifest": manifest, "size_bytes": 10,
        "history": {"test_runs": 0, "interp_runs": 0, "weight_runs": 0},
    } for ref, manifest in items]}


def test_probe_remains_a_probe_in_history_and_retains_fork_options(tmp_path, test_db):
    options = {"pooling": "mean", "n_jailbreak": 0, "holdout_techniques": 0}
    row = WeightRun(kind="probe", source_model="org/base", status="completed", meta_data={"options": options})
    test_db.add(row)
    test_db.commit()
    graph = build_model_lineage(catalog=inventory(("org/base", None)), project_root=tmp_path)
    node = next(node for node in graph["nodes"] if node["kind"] == "probe")
    query = parse_qs(urlparse(node["links"]["resume"]).query)
    assert query["kind"] == ["probe"]
    assert json.loads(query["options"][0]) == options


def test_rescored_benchmark_keeps_original_edge_and_canonical_model(tmp_path, test_db):
    snapshot = str(tmp_path / "cache" / "snapshots" / "revision")
    common = dict(doctor_provider="transformers", doctor_model=snapshot,
                  patient_provider="transformers", patient_model=snapshot,
                  test_type="benchmark", status="completed")
    source = Run(**common, meta_data={"lineage_id": "original-uuid", "campaign_model": "org/base", "benchmark_campaign": {"id": "original"}})
    test_db.add(source)
    test_db.flush()
    corrected = Run(**common, meta_data={
        "lineage_id": "corrected-uuid", "lineage_parent": "inherited-grandparent", "campaign_model": "org/base", "benchmark_campaign": {"id": "corrected", "source_campaign_id": "original"},
        "model_provenance": {"path": snapshot},
        "rescoring": {"source_test_run_id": source.id, "source_result_id": 1, "source_lineage_id": "original-uuid"},
    })
    test_db.add(corrected)
    test_db.commit()
    graph = build_model_lineage(catalog=inventory(("org/base", None)), project_root=tmp_path)
    tests = {node["run_id"]: node for node in graph["nodes"] if node["kind"] == "test"}
    assert tests[corrected.id]["model_ref"] == "org/base"
    assert tests[corrected.id]["settings"]["model_provenance"] == {"path": snapshot}
    assert tests[corrected.id]["links"]["comparison"] == "/benchmarks?campaign=corrected"
    assert any(edge["source"] == tests[source.id]["id"] and edge["target"] == tests[corrected.id]["id"]
               and edge["label"] == "rescored answers" for edge in graph["edges"])
    assert not any(node["kind"] == "model" and node["model_ref"] == snapshot for node in graph["nodes"])
    assert not any(node["id"] == "inherited-grandparent" for node in graph["nodes"])


def export(tmp_path, payload):
    root = tmp_path / "imports"
    root.mkdir(exist_ok=True)
    (root / f"{payload['origin']}.json").write_text(json.dumps(payload))
    return root


def test_unused_download_suggestions_stay_out_of_history_but_owned_models_remain(tmp_path):
    custom = str(tmp_path / "models" / "custom")
    catalog = inventory(("org/suggested", None), ("org/cached", None), (custom, None), ("org/previously-used", None))
    catalog["models"][0]["availability"] = "download_required"
    catalog["models"][3]["availability"] = "download_required"
    catalog["models"][3]["history"]["interp_runs"] = 1
    graph = build_model_lineage(catalog=catalog, project_root=tmp_path)
    assert {node["model_ref"] for node in graph["nodes"]} == {"org/cached", custom, "org/previously-used"}
    assert graph["edges"] == []
    assert len(catalog["models"]) == 4


def test_undownloaded_baseline_with_recorded_edges_stays_in_history(tmp_path, test_db):
    catalog = inventory(("org/base", None), ("org/unused", None))
    for item in catalog["models"]:
        item["availability"] = "download_required"
    test_db.add(WeightRun(kind="surgery", source_model="org/base", source_run_id=99))
    test_db.commit()
    graph = build_model_lineage(catalog=catalog, project_root=tmp_path)
    base = next(node for node in graph["nodes"] if node["kind"] == "model")
    assert base["model_ref"] == "org/base"
    assert base["status"] == "download_required"
    assert any(edge["source"] == base["id"] for edge in graph["edges"])
    assert any(node["kind"] == "missing" for node in graph["nodes"])
    assert not any(node.get("model_ref") == "org/unused" for node in graph["nodes"])
    node_ids = {node["id"] for node in graph["nodes"]}
    assert all(edge["source"] in node_ids and edge["target"] in node_ids for edge in graph["edges"])


def test_direction_branches_never_invent_sweep_to_surgery_dependency(tmp_path, test_db):
    checkpoint = str(tmp_path / "models" / "edit")
    direction = WeightRun(id=1, kind="direction", status="completed", source_model="org/base")
    test_db.add_all([
        direction,
        WeightRun(id=2, kind="sweep", status="completed", source_model="org/base", source_run_id=1),
        WeightRun(id=3, kind="select", status="completed", source_model="org/base", source_run_id=1),
        WeightRun(id=4, kind="surgery", status="completed", source_model="org/base", source_run_id=1, out_dir=checkpoint),
        WeightRun(id=5, kind="compare", status="completed", source_model="org/base", meta_data={"modified_model": checkpoint}),
    ])
    test_db.commit()
    graph = build_model_lineage(catalog=inventory(("org/base", None), (checkpoint, {"source_model": "org/base"})), project_root=tmp_path)
    by_run = {n["run_id"]: n for n in graph["nodes"] if n["run_id"]}
    pairs = {(e["source"], e["target"]): e["label"] for e in graph["edges"]}
    for child in (2, 3, 4):
        assert pairs[by_run[1]["id"], by_run[child]["id"]] == "direction input"
    assert (by_run[2]["id"], by_run[4]["id"]) not in pairs
    assert (by_run[3]["id"], by_run[4]["id"]) not in pairs
    model = next(n for n in graph["nodes"] if n["kind"] == "model" and n["model_ref"] == checkpoint)
    assert pairs[by_run[4]["id"], model["id"]] == "saved checkpoint"
    assert pairs[model["id"], by_run[5]["id"]] == "modified model"
    assert not any(n["id"].startswith("manifest:") for n in graph["nodes"])


def test_imported_ids_are_namespaced_and_models_map_to_current_server(tmp_path, test_db):
    checkpoint = str(tmp_path / "models" / "edit")
    original = "/old/workspace/models/edit"
    test_db.add(WeightRun(id=1, kind="direction", source_model="org/base", status="completed"))
    test_db.commit()
    imports = export(tmp_path, {"origin": "old-workspace", "location": "Mac workspace", "model_refs": {original: checkpoint},
        "weight_runs": [
            {"id": 1, "kind": "direction", "status": "completed", "source_model": "org/base", "method": "rfm", "metadata": {"options": {"rfm_rank": 4}}},
            {"id": 2, "kind": "surgery", "status": "completed", "source_model": "org/base", "source_run_id": 1, "out_dir": original},
        ]})
    graph = build_model_lineage(catalog=inventory((checkpoint, {"source_model": "org/base"})), project_root=tmp_path, imports_root=imports)
    directions = [n for n in graph["nodes"] if n["kind"] == "direction"]
    assert len(directions) == 2
    assert len({n["id"] for n in directions}) == 2
    assert len([n for n in graph["nodes"] if n["model_ref"] == checkpoint and n["kind"] == "model"]) == 1
    imported_edit = next(n for n in graph["nodes"] if n["id"] == "weight:old-workspace:2")
    assert "run" not in imported_edit["links"]
    assert "Recreate the recorded direction" in imported_edit["settings"]["resume_note"]
    assert "Imported run IDs are not live jobs" in imported_edit["settings"]["resume_note"]
    query = parse_qs(urlparse(imported_edit["links"]["resume"]).query)
    assert query["kind"] == ["direction"]
    assert query["method"] == ["rfm"]
    assert json.loads(query["options"][0]) == {"rfm_rank": 4}
    assert "source_run_id" not in query
    assert query["lineage_parent"] == [imported_edit["id"]]


def test_missing_direction_is_explicit_with_saved_snapshot(tmp_path, test_db):
    test_db.add(WeightRun(id=3, kind="surgery", status="completed", source_model="org/base", source_run_id=99,
                         meta_data={"source_direction": {"layer": 12, "auc": 0.98}}))
    test_db.commit()
    graph = build_model_lineage(catalog=inventory(), project_root=tmp_path)
    missing = next(n for n in graph["nodes"] if n["kind"] == "missing")
    assert "99" in missing["missing_reason"]
    assert missing["summary"] == {"layer": 12, "auc": 0.98}
    assert any(e["source"] == missing["id"] and e["label"] == "direction input" for e in graph["edges"])
    assert graph["warnings"]


def test_unscoped_manifest_id_cannot_attach_to_unrelated_server_direction(tmp_path, test_db):
    test_db.add(WeightRun(id=1, kind="direction", source_model="org/other", status="completed"))
    test_db.commit()
    graph = build_model_lineage(catalog=inventory((str(tmp_path / "models" / "edit"), {
        "source_model": "org/base", "extra": {"direction_run_id": 1}, "beta": 0,
    })), project_root=tmp_path)
    actual = next(n for n in graph["nodes"] if n["kind"] == "direction")
    manifest = next(n for n in graph["nodes"] if n["origin"] == "checkpoint-manifest" and n["kind"] == "surgery")
    assert not any(e["source"] == actual["id"] and e["target"] == manifest["id"] for e in graph["edges"])
    assert any(n["kind"] == "missing" for n in graph["nodes"])


def test_local_resume_reuses_only_an_available_completed_direction(tmp_path, test_db):
    direction_path = tmp_path / "direction"
    direction_path.mkdir()
    (direction_path / "direction.safetensors").write_bytes(b"direction")
    test_db.add_all([
        WeightRun(id=1, kind="direction", source_model="org/base", status="completed", out_dir=str(direction_path)),
        WeightRun(id=2, kind="surgery", source_model="org/base", status="completed", source_run_id=1),
    ])
    test_db.commit()
    graph = build_model_lineage(catalog=inventory(), project_root=tmp_path)
    edit = next(n for n in graph["nodes"] if n["run_id"] == 2)
    query = parse_qs(urlparse(edit["links"]["resume"]).query)
    assert query["kind"] == ["surgery"]
    assert query["source_run_id"] == ["1"]
    (direction_path / "direction.safetensors").unlink()
    graph = build_model_lineage(catalog=inventory(), project_root=tmp_path)
    query = parse_qs(urlparse(next(n for n in graph["nodes"] if n["run_id"] == 2)["links"]["resume"]).query)
    assert query["kind"] == ["direction"]
    assert "source_run_id" not in query


def test_fork_from_interpretability_node_resolves_without_false_missing_warning(tmp_path, test_db):
    parent = f"interp:server-{socket.gethostname()}:7"
    test_db.add(InterpRun(id=7, mode="single", status="completed", model_a="org/base"))
    test_db.add(WeightRun(id=1, kind="direction", source_model="org/base", meta_data={"lineage_parent": parent}))
    test_db.commit()
    graph = build_model_lineage(catalog=inventory(), project_root=tmp_path)
    assert not graph["warnings"]
    assert next(n for n in graph["nodes"] if n["id"] == parent)["kind"] == "interp"
    assert any(e["source"] == parent and e["label"] == "forked from" for e in graph["edges"])


def test_invalid_import_does_not_hide_valid_history(tmp_path, test_db):
    root = tmp_path / "imports"
    root.mkdir()
    (root / "broken.json").write_text("not JSON")
    test_db.add(WeightRun(kind="direction", source_model="org/base"))
    test_db.commit()
    graph = build_model_lineage(catalog=inventory(), project_root=tmp_path, imports_root=root)
    assert graph["warnings"]
    assert any(n["kind"] == "direction" for n in graph["nodes"])


def test_durable_identity_prevents_deleted_parent_id_reuse_from_rewriting_history(tmp_path, test_db):
    test_db.add_all([
        WeightRun(id=1, kind="direction", source_model="org/base", status="completed", meta_data={"lineage_id": "new-direction-uuid"}),
        WeightRun(id=2, kind="surgery", source_model="org/base", source_run_id=1,
                  meta_data={"lineage_id": "edit-uuid", "source_direction_lineage_id": "deleted-direction-uuid"}),
        InterpRun(id=7, mode="single", model_a="org/base", meta_data={"lineage_id": "interp-uuid"}),
    ])
    test_db.commit()
    graph = build_model_lineage(catalog=inventory(), project_root=tmp_path)
    current = next(n for n in graph["nodes"] if n["id"].endswith(":new-direction-uuid"))
    edit = next(n for n in graph["nodes"] if n["id"].endswith(":edit-uuid"))
    missing = next(n for n in graph["nodes"] if n["id"].endswith(":deleted-direction-uuid"))
    assert missing["kind"] == "missing"
    assert any(n["id"].endswith(":interp-uuid") for n in graph["nodes"])
    assert not any(e["source"] == current["id"] and e["target"] == edit["id"] for e in graph["edges"])
    assert any(e["source"] == missing["id"] and e["target"] == edit["id"] for e in graph["edges"])


def test_lineage_endpoint_contract(monkeypatch):
    from fastapi.testclient import TestClient
    from vivasecuris.aiasylum.api.main import app
    from vivasecuris.aiasylum.api import model_lineage

    payload = {"nodes": [], "edges": [], "warnings": []}
    monkeypatch.setattr(model_lineage, "build_model_lineage", lambda: payload)
    response = TestClient(app).get("/api/v1/models/lineage")
    assert response.status_code == 200
    assert response.json() == payload


def test_interp_resume_preserves_prompt_pair_progression_and_options(tmp_path, test_db):
    options = {"device": "cuda", "dtype": "bfloat16", "max_len": 512, "enable_patching": True}
    prompts = ["First example\nQuestion", "First example\nSecond example\nQuestion"]
    test_db.add_all([
        InterpRun(id=1, mode="comparison", model_a="org/base", prompt_a="Question A?", prompt_b="Question B?",
                  meta_data={"options": options}),
        InterpRun(id=2, mode="progression", model_a="org/base", prompts=prompts, meta_data={"options": options}),
    ])
    test_db.commit()
    graph = build_model_lineage(catalog=inventory(), project_root=tmp_path)
    runs = {node["run_id"]: node for node in graph["nodes"] if node["kind"] == "interp"}
    pair_query = parse_qs(urlparse(runs[1]["links"]["resume"]).query)
    progression_query = parse_qs(urlparse(runs[2]["links"]["resume"]).query)
    assert pair_query["prompt"] == pair_query["prompt_a"] == ["Question A?"]
    assert pair_query["prompt_b"] == ["Question B?"]
    assert json.loads(pair_query["options"][0]) == options
    assert json.loads(progression_query["prompts"][0]) == prompts


def test_weights_resume_preserves_objective_config_and_rederived_parent_contrast(tmp_path, test_db):
    objective_config = {"positive_prompts": ["Target behavior"], "negative_prompts": ["Control behavior"]}
    test_db.add_all([
        WeightRun(id=1, kind="direction", source_model="org/base", objective="custom", method="diff_in_means",
                  meta_data={"objective_config": objective_config, "options": {"seed": 7}}),
        WeightRun(id=2, kind="surgery", source_model="org/base", source_run_id=1, objective="refusal",
                  meta_data={"objective_config": {"stale": "value"}}),
    ])
    test_db.commit()
    graph = build_model_lineage(catalog=inventory(), project_root=tmp_path)
    for node in graph["nodes"]:
        if node["kind"] not in ("direction", "surgery"):
            continue
        query = parse_qs(urlparse(node["links"]["resume"]).query)
        assert query["kind"] == ["direction"]
        assert query["objective"] == ["custom"]
        assert json.loads(query["objective_config"][0]) == objective_config


def test_test_resume_preserves_participants_and_configuration_when_recorded(tmp_path, test_db):
    config = {"max_turns": 9, "prompt": "Retain this question", "temperature": 0.0}
    test_db.add_all([
        Run(id=1, test_type="conversation", patient_provider="transformers", patient_model="org/base",
            doctor_provider="openai", doctor_model="doctor-model", meta_data={"test_config": config}),
        Run(id=2, test_type="scenario", patient_provider="transformers", patient_model="org/base",
            doctor_provider="openai", doctor_model="doctor-model"),
    ])
    test_db.commit()
    graph = build_model_lineage(catalog=inventory(), project_root=tmp_path)
    runs = {node["run_id"]: node for node in graph["nodes"] if node["kind"] == "test"}
    query = parse_qs(urlparse(runs[1]["links"]["resume"]).query)
    assert query["provider"] == ["transformers"]
    assert query["model"] == ["org/base"]
    assert query["type"] == ["conversation"]
    assert query["doctor_provider"] == ["openai"]
    assert query["doctor_model"] == ["doctor-model"]
    assert json.loads(query["test_config"][0]) == config
    assert runs[1]["settings"]["test_config"] == config
    unknown = parse_qs(urlparse(runs[2]["links"]["resume"]).query)
    assert "test_config" not in unknown
    assert runs[2]["summary"]["configuration_recorded"] is False


def test_imported_group_therapy_patients_use_current_server_model_paths(tmp_path):
    current = str(tmp_path / "models" / "edit")
    original = "/old/workspace/models/edit"
    imports = export(tmp_path, {"origin": "old", "model_refs": {original: current}, "test_runs": [
        {"id": 1, "test_type": "group_therapy", "patient_provider": "transformers", "patient_model": original,
         "doctor_provider": "openai", "doctor_model": "doctor", "metadata": {"test_config": {
             "patients": [{"provider": "transformers", "model": original}], "topic": "Recorded topic",
         }}},
    ]})
    graph = build_model_lineage(catalog=inventory(), project_root=tmp_path, imports_root=imports)
    node = next(n for n in graph["nodes"] if n["kind"] == "test")
    query = parse_qs(urlparse(node["links"]["resume"]).query)
    assert "run" not in node["links"]
    assert json.loads(query["test_config"][0])["patients"][0]["model"] == current


def test_imported_and_database_naive_times_match_aware_manifest_times(tmp_path, test_db):
    test_db.add(WeightRun(id=1, kind="direction", source_model="org/base",
                         created_at=datetime(2026, 9, 24, 2, 16)))
    test_db.commit()
    imports = export(tmp_path, {"origin": "old", "weight_runs": [
        {"id": 2, "kind": "direction", "source_model": "org/base", "created_at": "2026-09-24T02:18:00"},
        {"id": 3, "kind": "direction", "source_model": "org/base", "created_at": "2026-09-23T20:18:00-06:00"},
    ]})
    catalog = inventory((str(tmp_path / "models" / "edit"), {"source_model": "org/base"}))
    catalog["models"][0]["created_at"] = "2026-09-24T02:18:00+00:00"
    graph = build_model_lineage(catalog=catalog, project_root=tmp_path, imports_root=imports)
    imported = [n for n in graph["nodes"] if n["origin"] == "old"]
    checkpoint = next(n for n in graph["nodes"] if n["kind"] == "model" and n["model_ref"].endswith("/edit"))
    assert {n["created_at"] for n in imported} == {checkpoint["created_at"]} == {"2026-09-24T02:18:00+00:00"}
    dated = [n for n in graph["nodes"] if n["created_at"]]
    assert dated[0]["run_id"] == 1
    assert dated[0]["created_at"] == "2026-09-24T02:16:00+00:00"


def test_lineage_iso_normalizes_datetime_objects_and_rejects_invalid_strings():
    from vivasecuris.aiasylum.api.model_lineage import _iso

    assert _iso(datetime(2026, 9, 24, 2, 18)) == _iso(datetime(2026, 9, 24, 2, 18, tzinfo=timezone.utc))
    assert _iso("invalid date") is None
