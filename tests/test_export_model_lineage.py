"""Portable histories keep archived branches and resolve their fork edges."""

import hashlib
import json
from pathlib import Path
import socket
import sqlite3

import pytest

from scripts.export_model_lineage import export
from vivasecuris.aiasylum.api.model_lineage import build_model_lineage


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / "workspace"
    (root / "data").mkdir(parents=True)
    with sqlite3.connect(root / "data/aiasylum.db") as db:
        db.execute("CREATE TABLE weight_runs (id INTEGER, kind TEXT, status TEXT, created_at TEXT, source_model TEXT, source_run_id INTEGER, out_dir TEXT, metadata TEXT)")
        db.execute("CREATE TABLE interp_runs (id INTEGER, mode TEXT, status TEXT, created_at TEXT, model_a TEXT, model_b TEXT, prompt_a TEXT, prompt_b TEXT, prompts TEXT, metadata TEXT)")
        db.execute("CREATE TABLE test_runs (id INTEGER, test_type TEXT, status TEXT, created_at TEXT, patient_provider TEXT, patient_model TEXT, doctor_provider TEXT, doctor_model TEXT, metadata TEXT)")
    return root


def insert(workspace, table, **row):
    row.setdefault("metadata", {})
    row.setdefault("created_at", "2026-09-24T02:00:00")
    row = {key: json.dumps(value) if isinstance(value, (dict, list)) else value for key, value in row.items()}
    with sqlite3.connect(workspace / "data/aiasylum.db") as db:
        db.execute(f"INSERT INTO {table} ({','.join(row)}) VALUES ({','.join('?' for _ in row)})", list(row.values()))


def archive(workspace, name, table, row, origin=None):
    root = workspace / "runs/model-lineage/archive"
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{name}.json").write_text(json.dumps({"table": table, "row": row,
        "origin": origin or "server-" + socket.gethostname(), "archived_at": "2026-09-24T03:00:00+00:00"}))


def model_id(path):
    return "model:transformers:" + hashlib.sha256(str(path).encode()).hexdigest()[:24]


def imported_graph(workspace, remote):
    out = remote / "runs/model-lineage/imports/workspace.json"
    payload = export(workspace, str(remote), out)
    graph = build_model_lineage(catalog={"models": [], "location": "destination"}, project_root=remote)
    return payload, graph


def test_local_run_and_model_fork_ids_resolve_after_export_and_import(workspace, tmp_path):
    source_origin = "server-" + socket.gethostname()
    checkpoint = workspace / "models/edit"
    checkpoint.mkdir(parents=True)
    (checkpoint / "asylum_surgery.json").write_text(json.dumps({"source_model": "org/base", "method": "direction_scale"}))
    insert(workspace, "weight_runs", id=1, kind="direction", source_model=str(checkpoint), status="completed",
           metadata={"lineage_id": "direction-uuid", "lineage_parent": model_id(checkpoint)})
    insert(workspace, "weight_runs", id=2, kind="direction", source_model=str(checkpoint), status="completed",
           metadata={"lineage_id": "fork-uuid", "lineage_parent": f"weight:{source_origin}:direction-uuid"})
    insert(workspace, "interp_runs", id=3, mode="progression", model_a=str(checkpoint), status="completed", prompts=["one", "two"],
           metadata={"lineage_id": "interp-uuid", "lineage_parent": f"weight:{source_origin}:fork-uuid"})
    insert(workspace, "test_runs", id=4, test_type="conversation", patient_provider="transformers", patient_model=str(checkpoint),
           doctor_provider="openai", doctor_model="doctor", metadata={"test_config": {"lineage_parent": f"interp:{source_origin}:interp-uuid"}})
    payload, graph = imported_graph(workspace, tmp_path / "remote")
    assert payload["weight_runs"][0]["metadata"]["lineage_parent"] == model_id(tmp_path / "remote/models/edit")
    assert payload["weight_runs"][1]["metadata"]["lineage_parent"] == f"weight:{payload['origin']}:direction-uuid"
    assert payload["interp_runs"][0]["prompts"] == ["one", "two"]
    assert not graph["warnings"]
    nodes = {node["id"] for node in graph["nodes"]}
    forks = [edge for edge in graph["edges"] if edge["label"] == "forked from"]
    assert len(forks) == 4
    assert all(edge["source"] in nodes and edge["target"] in nodes for edge in forks)


def test_archived_steps_survive_export_and_live_record_wins_duplicates(workspace, tmp_path):
    source_origin = "server-" + socket.gethostname()
    archived = {"id": 1, "kind": "direction", "status": "completed", "source_model": "org/base",
                "created_at": "2026-09-23T00:00:00+00:00", "metadata": {"lineage_id": "deleted-uuid", "options": {"seed": 3}}}
    archive(workspace, "deleted", "weight_runs", archived)
    archive(workspace, "duplicate", "weight_runs", {**archived, "id": 2, "status": "failed", "metadata": {"lineage_id": "live-uuid"}})
    insert(workspace, "weight_runs", id=2, kind="direction", source_model="org/base", status="completed",
           metadata={"lineage_id": "live-uuid", "lineage_parent": f"weight:{source_origin}:deleted-uuid"})
    payload, graph = imported_graph(workspace, tmp_path / "remote")
    assert len(payload["weight_runs"]) == 2
    live = next(row for row in payload["weight_runs"] if row["id"] == 2)
    deleted = next(row for row in payload["weight_runs"] if row["id"] == 1)
    assert live["status"] == "completed" and not live.get("archived")
    assert deleted["archived"] is True
    archived_node = next(node for node in graph["nodes"] if node["id"].endswith(":deleted-uuid"))
    assert archived_node["settings"]["archived_at"]
    assert archived_node["settings"]["options"] == {"seed": 3}
    assert "run" not in archived_node["links"]
    assert not graph["warnings"]


def test_removed_checkpoint_refs_and_model_forks_are_mapped_from_archived_rows(workspace, tmp_path):
    checkpoint = workspace / "models/deleted-edit"
    archive(workspace, "surgery", "weight_runs", {"id": 1, "kind": "surgery", "status": "completed", "source_model": "org/base",
        "out_dir": str(checkpoint), "created_at": "2026-09-23T00:00:00+00:00", "metadata": {
            "summary": {"manifest": {"source_model": "org/base", "method": "direction_scale"}}}})
    insert(workspace, "weight_runs", id=2, kind="direction", source_model=str(checkpoint), status="completed",
           metadata={"lineage_parent": model_id(checkpoint)})
    remote = tmp_path / "remote"
    payload = export(workspace, str(remote), tmp_path / "export.json")
    assert payload["model_refs"][str(checkpoint)] == str(remote / "models/deleted-edit")
    assert payload["models"][0]["manifest"]["method"] == "direction_scale"
    current = next(row for row in payload["weight_runs"] if row["id"] == 2)
    assert current["metadata"]["lineage_parent"] == model_id(remote / "models/deleted-edit")


def test_unrelated_import_origins_are_not_rewritten_or_flattened(workspace, tmp_path):
    foreign = "weight:workspace-another-origin:1"
    insert(workspace, "weight_runs", id=1, kind="direction", source_model="org/base", metadata={"lineage_parent": foreign})
    imports = workspace / "runs/model-lineage/imports"
    imports.mkdir(parents=True)
    original = {"origin": "workspace-another-origin", "weight_runs": [{"id": 1, "kind": "direction"}]}
    file = imports / "other.json"
    file.write_text(json.dumps(original))
    payload = export(workspace, str(tmp_path / "remote"), tmp_path / "export.json")
    assert len(payload["weight_runs"]) == 1
    assert payload["weight_runs"][0]["metadata"]["lineage_parent"] == foreign
    assert json.loads(file.read_text()) == original
    assert any("not flattened" in warning for warning in payload["warnings"])


def test_legacy_reused_ids_keep_distinct_identities_when_exported(workspace, tmp_path):
    archive(workspace, "old", "weight_runs", {"id": 1, "kind": "direction", "status": "completed", "source_model": "org/base",
        "created_at": "2026-09-01T00:00:00+00:00", "metadata": {}})
    insert(workspace, "weight_runs", id=1, kind="direction", source_model="org/base", status="completed", created_at="2026-09-02T00:00:00")
    payload, graph = imported_graph(workspace, tmp_path / "remote")
    directions = [node for node in graph["nodes"] if node["kind"] == "direction"]
    assert len({node["id"] for node in directions}) == 2
    assert any(row.get("_lineage_identity", "").startswith("legacy-1-") for row in payload["weight_runs"])
    assert not graph["warnings"]
