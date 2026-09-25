"""Editable organization persists without changing model or run provenance."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from uuid import UUID

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from vivasecuris.aiasylum.api import model_organization as organization
from vivasecuris.aiasylum.api.routes import model_organization as routes


@pytest.fixture
def store(tmp_path):
    return organization.OrganizationStore(tmp_path / "runs" / "model-organization.json")


@pytest.fixture
def client(store, monkeypatch):
    monkeypatch.setattr(routes, "get_store", lambda: store)
    app = FastAPI()
    app.include_router(routes.router, prefix="/api/v1/model-organization")
    return TestClient(app)


URL = "/api/v1/model-organization"


def test_experiments_and_model_step_membership_survive_reopening(store):
    assert store.read() == {"experiments": [], "items": {}}
    experiment = store.create_experiment(" Refusal repair ", " Compare independent methods. ")
    assert str(UUID(experiment["id"])) == experiment["id"]
    assert experiment["name"] == "Refusal repair"
    assert experiment["notes"] == "Compare independent methods."
    assert experiment["created_at"].endswith("+00:00")
    key = "model:/home/ubuntu/models/custom"
    store.put_item(key, "Control", "Keep the original checkpoint.", [experiment["id"]])
    store.put_item("step:weight:origin:run-uuid", "Try method two", "Review if loss increases.", [experiment["id"]])

    reopened = organization.OrganizationStore(store.path)
    updated = reopened.update_experiment(experiment["id"], name="Repair methods")
    assert updated["created_at"] == experiment["created_at"]
    assert updated["notes"] == experiment["notes"]
    snapshot = reopened.read()
    assert snapshot["experiments"][0]["name"] == "Repair methods"
    assert snapshot["items"][key] == {
        "label": "Control", "notes": "Keep the original checkpoint.", "experiment_ids": [experiment["id"]],
    }
    assert len(snapshot["items"]) == 2
    assert json.loads(store.path.read_text()) == snapshot
    assert store.path.stat().st_mode & 0o777 == 0o600


def test_blank_item_metadata_removes_only_that_item(store):
    store.put_item("model:org/base", label="Baseline")
    store.put_item("step:interp:1", notes="Investigate this run.")
    snapshot = store.put_item("model:org/base", label=" ", notes="\n")
    assert snapshot["items"] == {
        "step:interp:1": {"label": "", "notes": "Investigate this run.", "experiment_ids": []},
    }
    assert store.put_item("model:missing", "", "", []) == snapshot


def test_multiple_store_instances_do_not_overwrite_other_items(store):
    first = store.create_experiment("First experiment")
    second_store = organization.OrganizationStore(store.path)
    second = second_store.create_experiment("Second experiment")
    store.put_item("model:org/a", label="A", experiment_ids=[first["id"]])
    second_store.put_item("model:org/b", label="B", experiment_ids=[second["id"]])
    snapshot = store.read()
    assert len(snapshot["experiments"]) == 2
    assert set(snapshot["items"]) == {"model:org/a", "model:org/b"}


def test_concurrent_updates_preserve_every_independent_item(store):
    def save(number):
        other = organization.OrganizationStore(store.path)
        other.put_item(f"step:interp:{number}", label=f"Run {number}")

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(save, range(24)))
    assert len(store.read()["items"]) == 24
    assert not list(store.path.parent.glob(".model-organization-*.json"))


@pytest.mark.parametrize("document", [
    b"{broken", b"[]", b'{}', b'{"experiments": [], "items": {}, "unknown": true}',
    b'{"experiments": [], "items": {}, "items": {}}',
    b'{"experiments": [], "items": {"model:x": {"experiment_ids": ["unknown"]}}}',
])
def test_corrupt_existing_document_is_preserved_and_returns_503(store, client, document):
    store.path.parent.mkdir(parents=True)
    store.path.write_bytes(document)
    assert client.get(URL).status_code == 503
    response = client.post(f"{URL}/experiments", json={"name": "Repair"})
    assert response.status_code == 503
    assert "existing file was preserved" in response.json()["detail"]
    assert store.path.read_bytes() == document


def test_duplicate_experiment_ids_in_storage_are_corrupt(store):
    experiment = store.create_experiment("One")
    raw = json.dumps({"experiments": [experiment, experiment], "items": {}})
    store.path.write_text(raw)
    with pytest.raises(organization.OrganizationUnavailable, match="existing file was preserved"):
        store.put_item("model:x", label="Cannot save")
    assert store.path.read_text() == raw


def test_failed_atomic_replace_preserves_document_and_cleans_temporary_file(store, monkeypatch):
    store.put_item("model:org/base", label="Original")
    previous = store.path.read_bytes()

    def fail_replace(source, destination):
        raise OSError("filesystem unavailable")

    monkeypatch.setattr(organization.os, "replace", fail_replace)
    with pytest.raises(organization.OrganizationUnavailable, match="no organization changes were saved"):
        store.put_item("model:org/base", label="Unwritten")
    assert store.path.read_bytes() == previous
    assert not list(store.path.parent.glob(".model-organization-*.json"))


def test_limits_allow_edits_and_removal_but_reject_new_entries(store, monkeypatch):
    monkeypatch.setattr(organization, "MAX_EXPERIMENTS", 2)
    monkeypatch.setattr(organization, "MAX_ITEMS", 2)
    first = store.create_experiment("First")
    store.create_experiment("Second")
    with pytest.raises(ValueError, match="at most 2 experiments"):
        store.create_experiment("Third")
    assert store.update_experiment(first["id"], name="Renamed")["name"] == "Renamed"
    store.put_item("model:a", label="A")
    store.put_item("model:b", label="B")
    with pytest.raises(ValueError, match="at most 2 organized"):
        store.put_item("model:c", label="C")
    assert store.put_item("model:a", label="Renamed")["items"]["model:a"]["label"] == "Renamed"
    store.put_item("model:a")
    assert "model:c" in store.put_item("model:c", label="C")["items"]


def test_api_contract_and_unknown_experiment(client):
    assert client.get(URL).json() == {"experiments": [], "items": {}}
    response = client.post(f"{URL}/experiments", json={"name": "Methods"})
    assert response.status_code == 201
    experiment = response.json()
    assert set(experiment) == {"id", "name", "notes", "created_at"}
    updated = client.patch(f"{URL}/experiments/{experiment['id']}", json={"notes": "Retain baselines."})
    assert updated.status_code == 200
    assert updated.json()["name"] == experiment["name"]
    assert updated.json()["notes"] == "Retain baselines."
    response = client.put(f"{URL}/items", json={
        "key": "model:Qwen/Qwen3-0.6B", "label": "Small baseline", "notes": "", "experiment_ids": [experiment["id"]],
    })
    assert response.status_code == 200
    assert set(response.json()) == {"experiments", "items"}
    assert client.patch(f"{URL}/experiments/missing", json={"name": "Missing"}).status_code == 404
    assert client.put(f"{URL}/items", json={"key": "model:x", "experiment_ids": ["missing"]}).status_code == 422


@pytest.mark.parametrize("payload", [
    {"key": "unknown:x"}, {"key": "model:"}, {"key": "step: "}, {"key": "model:x\n"},
    {"key": "model:" + "x" * 1024}, {"key": "model:x", "label": "x" * 121},
    {"key": "model:x", "notes": "x" * 4001}, {"key": "model:x", "experiment_ids": ["same", "same"]},
    {"key": "model:x", "experiment_ids": "unknown"}, {"key": "model:x", "unknown": True},
])
def test_invalid_item_payloads_return_422_without_writes(client, store, payload):
    assert client.put(f"{URL}/items", json=payload).status_code == 422
    assert not store.path.exists()


@pytest.mark.parametrize("payload", [{"name": " "}, {"name": "x" * 121}, {"name": 42}, {"name": "ok", "notes": "x" * 4001}])
def test_invalid_experiment_payloads_return_422(client, store, payload):
    assert client.post(f"{URL}/experiments", json=payload).status_code == 422
    assert not store.path.exists()


def test_organization_updates_do_not_touch_model_or_lineage_files(store):
    root = store.path.parent.parent
    checkpoint = root / "models" / "custom" / "asylum_surgery.json"
    history = root / "runs" / "model-lineage" / "archive" / "run.json"
    for path in (checkpoint, history):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"original": true}\n')
    before = {path: path.read_bytes() for path in (checkpoint, history)}
    experiment = store.create_experiment("Repair attempt")
    store.put_item(f"model:{checkpoint.parent}", label="Renamed in UI", experiment_ids=[experiment["id"]])
    store.put_item("step:archived-run", notes="Try another method next.")
    assert all(path.read_bytes() == content for path, content in before.items())
