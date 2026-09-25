import json
from pathlib import Path

import pytest

from vivasecuris.aiasylum.api import custom_checkpoints as deletion
from vivasecuris.aiasylum.api.model_catalog import build_model_catalog
from vivasecuris.aiasylum.api.model_history import saved_checkpoint_history
from vivasecuris.aiasylum.database import WeightRun


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("AIASYLUM_MODEL_LOCK", str(tmp_path / "model.lock"))
    root = tmp_path / "models"
    root.mkdir()
    for name in ("edit-a", "edit-b"):
        path = root / name
        path.mkdir()
        (path / "asylum_surgery.json").write_text(json.dumps({"source_model": "org/base", "method": "direction_scale"}))
        (path / "config.json").write_text('{"model_type":"qwen2"}')
        (path / "model.safetensors").write_bytes(b"custom weights")
        (path / "tokenizer.json").write_text('{}')
    return tmp_path, root


def remove(workspace, names):
    project, root = workspace
    return deletion.delete_custom_checkpoints(names, models_root=root, project_root=project)


def test_deletion_retains_results_manifest_identity_and_base_files(workspace, test_db):
    project, root = workspace
    base = project / "hub" / "base"
    base.mkdir(parents=True)
    (base / "weights").write_bytes(b"base weights")
    row = WeightRun(kind="surgery", status="completed", source_model="org/base", out_dir=str(root / "edit-a"))
    test_db.add(row)
    test_db.commit()
    result = remove(workspace, ["edit-a"])
    assert result["freed_bytes"] > 0
    assert not (root / "edit-a").exists()
    assert (root / "edit-b" / "model.safetensors").read_bytes() == b"custom weights"
    assert (base / "weights").read_bytes() == b"base weights"
    assert test_db.get(WeightRun, row.id).status == "completed"
    records = deletion.deletion_records(project)
    assert records[str(root / "edit-a")]["manifest"]["source_model"] == "org/base"
    assert saved_checkpoint_history(root / "edit-a", project_root=project)
    catalog = build_model_catalog(models_root=root, project_root=project, cache_roots=[])
    deleted = next(m for m in catalog["models"] if m["name"] == "edit-a")
    assert deleted["availability"] == "deleted"
    assert deleted["history"]["weight_runs"] == 1
    assert deleted["size_bytes"] == 0


def test_orphan_manifest_stays_in_catalog_after_deletion(workspace):
    project, root = workspace
    before = build_model_catalog(models_root=root, project_root=project, cache_roots=[])
    remove(workspace, ["edit-b"])
    after = build_model_catalog(models_root=root, project_root=project, cache_roots=[])
    old = next(m for m in before["models"] if m["name"] == "edit-b")
    new = next(m for m in after["models"] if m["name"] == "edit-b")
    assert old["id"] == new["id"]
    assert new["manifest"] == old["manifest"]
    assert new["availability"] == "deleted"
    assert any(model["model_ref"] == "org/base" for model in after["models"])
    from vivasecuris.aiasylum.api.model_lineage import build_model_lineage
    graph = build_model_lineage(catalog=after, project_root=project)
    node = next(node for node in graph["nodes"] if node.get("model_ref") == str(root / "edit-b") and node["kind"] == "model")
    assert node["status"] == "deleted"
    assert any(edge["target"] == node["id"] for edge in graph["edges"])


@pytest.mark.parametrize("name", ["../edit-b", "/tmp/weights", ".private", "edit/b", "missing"])
def test_entire_selection_validated_before_deleting_anything(workspace, name):
    with pytest.raises((ValueError, LookupError)):
        remove(workspace, ["edit-a", name])
    assert (workspace[1] / "edit-a" / "model.safetensors").is_file()


def test_refuses_checkpoint_and_file_symlinks(workspace):
    project, root = workspace
    (root / "linked").symlink_to(root / "edit-a", target_is_directory=True)
    with pytest.raises(ValueError, match="linked"):
        remove(workspace, ["linked"])
    (root / "edit-b" / "link").symlink_to(root / "edit-a" / "model.safetensors")
    with pytest.raises(ValueError, match="linked files"):
        remove(workspace, ["edit-b"])


def test_archive_failure_preserves_all_weights(workspace, monkeypatch):
    def failed(*args):
        raise OSError("full disk")
    monkeypatch.setattr(deletion, "_write_record", failed)
    with pytest.raises(OSError, match="full disk"):
        remove(workspace, ["edit-a"])
    assert (workspace[1] / "edit-a" / "model.safetensors").is_file()


def test_busy_model_lease_and_queued_runs_block_deletion(workspace, test_db):
    lock = deletion.try_process_lock()
    try:
        with pytest.raises(RuntimeError, match="model job"):
            remove(workspace, ["edit-a"])
    finally:
        lock.close()
    test_db.add(WeightRun(kind="direction", source_model="org/base", status="pending"))
    test_db.commit()
    with pytest.raises(RuntimeError, match="queued"):
        remove(workspace, ["edit-a"])
    assert (workspace[1] / "edit-a").is_dir()


def test_no_provenance_is_not_deleted(workspace):
    (workspace[1] / "edit-a" / "asylum_surgery.json").unlink()
    with pytest.raises(ValueError, match="provenance"):
        remove(workspace, ["edit-a"])
    assert (workspace[1] / "edit-a" / "model.safetensors").is_file()
