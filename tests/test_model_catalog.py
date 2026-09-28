"""Inventory uses real artifacts and run provenance, never evaluation presence."""

import json
from datetime import datetime
from pathlib import Path

import pytest

from vivasecuris.aiasylum.api import model_catalog
from vivasecuris.aiasylum.database import InterpRun, TestRun as Run, WeightRun


def checkpoint(path: Path, *, manifest=None, sharded=False):
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text('{"model_type": "qwen2"}')
    (path / "tokenizer.json").write_text('{"model": {"type": "BPE"}}')
    if sharded:
        (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
            "layer.0": "model-00001-of-00002.safetensors", "layer.1": "model-00002-of-00002.safetensors",
        }}))
        for number in (1, 2):
            (path / f"model-{number:05}-of-00002.safetensors").write_bytes(b"weights")
    else:
        (path / "model.safetensors").write_bytes(b"weights")
    if manifest is not None:
        (path / "asylum_surgery.json").write_text(json.dumps(manifest))
    return path


@pytest.fixture
def catalog(tmp_path):
    def build():
        return model_catalog.build_model_catalog(
            models_root=tmp_path / "models", project_root=tmp_path,
            cache_roots=[tmp_path / "hub"], presets_path=tmp_path / "matrix.json",
        )
    return build


def test_unregistered_custom_weights_are_visible_without_assessments(tmp_path, catalog, test_db):
    path = checkpoint(tmp_path / "models" / "my-edit")
    models = catalog()["models"]
    assert len(models) == 1
    assert models[0]["model_ref"] == str(path)
    assert models[0]["kind"] == "custom"
    assert models[0]["availability"] == "ready"
    assert models[0]["manifest"] is None
    assert models[0]["history"] == {"test_runs": 0, "interp_runs": 0, "weight_runs": 0}
    assert test_db.query(WeightRun).count() == 0
    assert catalog()["models"][0]["id"] == models[0]["id"]


def test_manifest_preserves_provenance_and_adds_undownloaded_baseline(tmp_path, catalog):
    manifest = {"source_model": "org/base", "method": "direction_scale", "beta": 0,
                "created_at": "2026-09-24T00:00:00+00:00", "extra": {"objective": "refusal"}}
    checkpoint(tmp_path / "models" / "edit", manifest=manifest)
    custom, base = catalog()["models"]
    assert custom["manifest"] == manifest
    assert custom["source_model"] == "org/base"
    assert custom["created_at"] == manifest["created_at"]
    assert custom["run_id"] is None
    assert base["model_ref"] == "org/base"
    assert base["availability"] == "download_required"


@pytest.mark.parametrize("missing", ["config.json", "tokenizer.json", "model.safetensors"])
def test_missing_or_empty_required_file_is_incomplete(tmp_path, missing):
    path = checkpoint(tmp_path / "edit")
    (path / missing).write_bytes(b"")
    assert model_catalog.checkpoint_readiness(path)[0] == "incomplete"
    (path / missing).unlink()
    assert model_catalog.checkpoint_readiness(path)[0] == "incomplete"


def test_shard_index_requires_every_nonempty_shard(tmp_path):
    path = checkpoint(tmp_path / "edit", sharded=True)
    assert model_catalog.checkpoint_readiness(path) == ("ready", None)
    (path / "model-00002-of-00002.safetensors").unlink()
    status, reason = model_catalog.checkpoint_readiness(path)
    assert status == "incomplete"
    assert "model-00002" in reason


@pytest.mark.parametrize("unsafe", ["../outside.safetensors", "/tmp/outside.safetensors", "..\\outside.safetensors"])
def test_shard_index_rejects_unsafe_paths(tmp_path, unsafe):
    path = checkpoint(tmp_path / "edit", sharded=True)
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"w": unsafe}}))
    status, reason = model_catalog.checkpoint_readiness(path)
    assert status == "incomplete"
    assert "unsafe" in reason


def test_shard_symlink_cannot_escape_custom_checkpoint(tmp_path):
    path = checkpoint(tmp_path / "edit", sharded=True)
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"weights")
    shard = path / "model-00001-of-00002.safetensors"
    shard.unlink()
    shard.symlink_to(outside)
    assert "unsafe" in model_catalog.checkpoint_readiness(path)[1]


def test_cached_baseline_allows_hf_blob_links_and_deduplicates_presets(tmp_path, catalog):
    repo = tmp_path / "hub" / "models--org--base"
    snapshot = checkpoint(repo / "snapshots" / "commit")
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("commit")
    (repo / "blobs").mkdir()
    blob = repo / "blobs" / "hash"
    blob.write_bytes(b"weights")
    (snapshot / "model.safetensors").unlink()
    (snapshot / "model.safetensors").symlink_to(blob)
    (tmp_path / "matrix.json").write_text(json.dumps({"models": [{"repo_id": "org/base"}, {"repo_id": "org/base"}]}))
    checkpoint(tmp_path / "models" / "edit", manifest={"source_model": "org/base"})
    models = catalog()["models"]
    assert len(models) == 2
    base = next(m for m in models if m["kind"] == "base")
    assert base["model_ref"] == "org/base"
    assert base["availability"] == "ready"
    assert base["size_bytes"] >= len(b"weights")
    assert model_catalog.checkpoint_status(snapshot)["availability"] == "ready"


def test_partial_main_snapshot_does_not_borrow_an_old_revisions_weights(tmp_path, catalog):
    repo = tmp_path / "hub" / "models--org--base"
    checkpoint(repo / "snapshots" / "old")
    current = checkpoint(repo / "snapshots" / "new")
    (current / "model.safetensors").unlink()
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("new")
    assert catalog()["models"][0]["availability"] == "incomplete"


def test_cached_pinned_snapshot_without_main_has_exact_load_path(tmp_path, catalog):
    snapshot = checkpoint(tmp_path / "hub" / "models--org--base" / "snapshots" / "commit")
    model = catalog()["models"][0]
    assert model["name"] == "org/base"
    assert model["model_ref"] == str(snapshot)
    assert model["availability"] == "ready"


def test_inventory_needs_neither_model_runtime_nor_network(tmp_path, catalog, monkeypatch):
    import builtins
    import socket

    checkpoint(tmp_path / "models" / "edit", manifest={"source_model": "org/base"})
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert name.split(".")[0] not in {"torch", "transformers", "huggingface_hub"}
        return real_import(name, *args, **kwargs)

    def network_forbidden(*args, **kwargs):
        pytest.fail("Inventory must not open a network connection")

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(socket, "create_connection", network_forbidden)
    assert len(catalog()["models"]) == 2


def test_missing_surgery_output_is_visible_and_has_creation_provenance(tmp_path, catalog, test_db):
    row = WeightRun(kind="surgery", source_model="org/base", status="completed", out_dir="models/deleted")
    test_db.add(row)
    test_db.commit()
    custom = next(m for m in catalog()["models"] if m["kind"] == "custom")
    assert custom["availability"] == "missing"
    assert custom["model_ref"] == str(tmp_path / "models" / "deleted")
    assert custom["source_model"] == "org/base"
    assert custom["run_id"] == row.id
    assert custom["history"]["weight_runs"] == 1


def test_history_deduplicates_relative_and_absolute_refs_and_model_roles(tmp_path, catalog, test_db):
    path = checkpoint(tmp_path / "models" / "edit", manifest={"source_model": "org/base"})
    test_db.add_all([
        WeightRun(kind="surgery", status="completed", source_model="org/base", out_dir=str(path)),
        WeightRun(kind="compare", status="completed", source_model="org/base", meta_data={"modified_model": "models/edit"}),
        Run(doctor_provider="transformers", doctor_model=str(path), patient_provider="local", patient_model="models/edit", test_type="conversation"),
        Run(doctor_provider="openai", doctor_model=str(path), patient_provider="openai", patient_model="models/edit", test_type="conversation"),
        InterpRun(mode="model_diff", model_a="models/edit", model_b=str(path), provider="transformers"),
    ])
    test_db.commit()
    models = catalog()["models"]
    assert len(models) == 2
    custom = next(m for m in models if m["kind"] == "custom")
    assert custom["history"] == {"test_runs": 1, "interp_runs": 1, "weight_runs": 2}
    assert custom["run_id"] is not None
    assert custom["aliases"] == ["models/edit"]


def test_catalog_endpoint_returns_server_location_and_exact_contract(tmp_path, catalog, monkeypatch):
    from fastapi.testclient import TestClient
    from vivasecuris.aiasylum.api.main import app

    checkpoint(tmp_path / "models" / "edit")
    payload = catalog()
    monkeypatch.setattr(model_catalog, "build_model_catalog", lambda: payload)
    response = TestClient(app).get("/api/v1/models/catalog")
    assert response.status_code == 200
    assert set(response.json()) == {"models", "location"}
    assert response.json()["location"]
    assert set(response.json()["models"][0]) == {
        "id", "model_ref", "name", "kind", "provider", "source_model", "availability",
        "reason", "size_bytes", "created_at", "run_id", "manifest", "history", "aliases",
    }


def test_imported_missing_models_keep_provenance_and_namespaced_history(tmp_path, catalog, test_db):
    remote = str(tmp_path / "models" / "not-transferred")
    original = "/old/workspace/models/not-transferred"
    imports = tmp_path / "runs" / "model-lineage" / "imports"
    imports.mkdir(parents=True)
    manifest = {"source_model": "org/base", "method": "direction_scale", "beta": 0}
    (imports / "workspace.json").write_text(json.dumps({
        "origin": "workspace-1", "location": "Mac workspace", "model_refs": {original: remote},
        "models": [{"model_ref": original, "name": "not-transferred", "manifest": manifest, "size_bytes": 10}],
        "weight_runs": [{"id": 1, "kind": "surgery", "source_model": "org/base", "out_dir": original}],
    }))
    test_db.add(WeightRun(id=1, kind="compare", source_model="org/base", meta_data={"modified_model": remote}))
    test_db.commit()
    custom = next(m for m in catalog()["models"] if m["kind"] == "custom")
    assert custom["availability"] == "missing"
    assert "Mac workspace" in custom["reason"]
    assert custom["model_ref"] == remote
    assert custom["manifest"] == manifest
    assert custom["run_id"] is None
    assert custom["history"]["weight_runs"] == 2
    assert custom["size_bytes"] == 0


def test_catalog_serializes_naive_database_creation_as_utc(tmp_path, catalog, test_db):
    path = checkpoint(tmp_path / "models" / "edit")
    test_db.add(WeightRun(kind="surgery", status="completed", source_model="org/base", out_dir=str(path),
                         created_at=datetime(2026, 9, 24, 2, 18)))
    test_db.commit()
    custom = next(m for m in catalog()["models"] if m["kind"] == "custom")
    assert custom["created_at"] == "2026-09-24T02:18:00+00:00"


@pytest.mark.parametrize("value", ["not a timestamp", "2026-99-24T02:18:00", "2026-09-24", None, 42])
def test_invalid_or_incomplete_timestamp_is_not_invented(value):
    assert model_catalog.normalize_timestamp(value) is None


def test_aliases_preserve_observed_relative_paths_spaces_and_hash_without_merging_names(tmp_path, catalog, test_db):
    first = checkpoint(tmp_path / "models" / "model #1")
    second = checkpoint(tmp_path / "other" / "model #1")
    test_db.add_all([
        Run(doctor_provider="transformers", doctor_model="./models/model #1",
            patient_provider="local", patient_model="models/model #1", test_type="one_shot"),
        Run(doctor_provider="openai", doctor_model="api-only-alias",
            patient_provider="transformers", patient_model=str(second), test_type="one_shot"),
    ])
    test_db.commit()
    models = {row["model_ref"]: row for row in catalog()["models"]}
    assert set(models) == {str(first), str(second)}
    assert models[str(first)]["aliases"] == ["./models/model #1", "models/model #1"]
    assert models[str(second)]["aliases"] == []
    assert models[str(first)]["availability"] == models[str(second)]["availability"] == "ready"


def test_aliases_link_only_selected_main_snapshot_not_an_older_revision(tmp_path, catalog, test_db):
    repo = tmp_path / "hub" / "models--org--base"
    current = checkpoint(repo / "snapshots" / "new")
    old = checkpoint(repo / "snapshots" / "old")
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("new")
    test_db.add(Run(doctor_provider="transformers", doctor_model=str(old),
                    patient_provider="transformers", patient_model="hub/models--org--base/snapshots/new",
                    test_type="one_shot"))
    test_db.commit()
    models = {row["model_ref"]: row for row in catalog()["models"]}
    assert set(models) == {"org/base", str(old)}
    assert models["org/base"]["aliases"] == sorted([str(current), "hub/models--org--base/snapshots/new"])
    assert str(old) not in models["org/base"]["aliases"]
    assert models[str(old)]["aliases"] == []


def test_pinned_cache_without_main_does_not_claim_repo_id_is_an_equivalent_request(tmp_path, catalog, test_db):
    snapshot = checkpoint(tmp_path / "hub" / "models--org--base" / "snapshots" / "pinned")
    test_db.add(Run(doctor_provider="transformers", doctor_model="org/base",
                    patient_provider="local", patient_model="hub/models--org--base/snapshots/pinned",
                    test_type="one_shot"))
    test_db.commit()
    model = catalog()["models"][0]
    assert model["model_ref"] == str(snapshot)
    assert model["aliases"] == ["hub/models--org--base/snapshots/pinned"]
    assert model["availability"] == "ready"


def test_current_snapshot_alias_is_available_without_any_recorded_run(tmp_path, catalog):
    repo = tmp_path / "hub" / "models--org--base"
    snapshot = checkpoint(repo / "snapshots" / "current")
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("current")
    model = catalog()["models"][0]
    assert model["model_ref"] == "org/base"
    assert model["aliases"] == [str(snapshot)]
