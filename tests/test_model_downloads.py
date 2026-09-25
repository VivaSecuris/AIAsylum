"""Downloads are supervised worker processes; nothing here touches the network.

The worker is replaced by a fake process that the test drives: it emits the
same JSON event lines the real worker prints, writes bytes into the cache the
way the Hub client does (into ``blobs/``), and exits with whatever status the
test chooses. That exercises the manager's state machine, the on-disk progress
measurement and the routes without a model or a connection.
"""

from __future__ import annotations

import json
import queue
import threading
import time

import pytest

from tests.test_model_catalog import checkpoint
from vivasecuris.aiasylum.api import model_downloads
from vivasecuris.aiasylum.api.model_downloads import DownloadManager, repo_dir
from vivasecuris.aiasylum.api.routes import models as models_route


class FakeProcess:
    """Stands in for ``subprocess.Popen``: a stdout the test feeds, an exit code it sets."""

    def __init__(self):
        self._lines: "queue.Queue[str | None]" = queue.Queue()
        self._done = threading.Event()
        self.returncode = None
        self.terminated = False

    @property
    def stdout(self):
        while True:
            line = self._lines.get()
            if line is None:
                return
            yield line

    def emit(self, **event):
        self._lines.put(json.dumps(event) + "\n")

    def exit(self, code: int):
        self.returncode = code
        self._lines.put(None)
        self._done.set()

    def terminate(self):
        self.terminated = True
        self.exit(-15)

    def wait(self):
        self._done.wait(5)
        return self.returncode


@pytest.fixture
def cache(tmp_path):
    return tmp_path / "hub"


@pytest.fixture
def spawned():
    return []


@pytest.fixture
def manager(cache, tmp_path, spawned):
    def spawn(repo_id, revision, log_path):
        proc = FakeProcess()
        spawned.append((repo_id, revision, proc))
        return proc

    return DownloadManager(cache_dir=cache, log_dir=tmp_path / "logs", spawn=spawn)


def settle(manager, download_id, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        record = manager.get(download_id)
        if not record["active"]:
            return record
        time.sleep(0.02)
    raise AssertionError("download did not settle")


def land_snapshot(cache, repo_id, commit="abc123"):
    """Lay out a complete snapshot the way the Hub client leaves one."""
    repo = repo_dir(cache, repo_id)
    snapshot = checkpoint(repo / "snapshots" / commit)
    (repo / "refs").mkdir(parents=True, exist_ok=True)
    (repo / "refs" / "main").write_text(commit)
    (repo / "blobs").mkdir(exist_ok=True)
    (repo / "blobs" / "weights").write_bytes(b"w" * 1000)
    return snapshot


# -- validation ---------------------------------------------------------------

@pytest.mark.parametrize("bad", ["", "qwen", "../etc", "org/../x", "org/name/extra", "/abs/path", "org/na me"])
def test_repo_ids_must_be_namespace_slash_name(manager, bad):
    with pytest.raises(ValueError):
        manager.start(bad)


def test_revision_cannot_escape(manager):
    with pytest.raises(ValueError):
        manager.start("org/name", "../main")


# -- state machine ------------------------------------------------------------

def test_progress_is_measured_from_bytes_on_disk(manager, cache, spawned):
    record = manager.start("org/name")
    assert record["status"] == "resolving" and record["active"]
    _, _, proc = spawned[0]

    proc.emit(event="plan", bytes_total=4000, files_total=3, commit="abc123")
    time.sleep(0.05)
    assert manager.get(record["id"])["status"] == "downloading"
    assert manager.get(record["id"])["bytes_total"] == 4000

    blobs = repo_dir(cache, "org/name") / "blobs"
    blobs.mkdir(parents=True)
    (blobs / "shard.incomplete").write_bytes(b"x" * 1500)
    assert manager.get(record["id"])["bytes_done"] == 1500

    snapshot = land_snapshot(cache, "org/name")
    proc.emit(event="done", snapshot=str(snapshot))
    proc.exit(0)
    final = settle(manager, record["id"])
    assert final["status"] == "completed"
    assert final["snapshot"] == str(snapshot)
    assert final["error"] is None


def test_worker_error_message_is_surfaced(manager, spawned):
    record = manager.start("org/gated")
    _, _, proc = spawned[0]
    proc.emit(event="error", message="org/gated is gated. Accept its terms on huggingface.co.")
    proc.exit(1)
    final = settle(manager, record["id"])
    assert final["status"] == "failed"
    assert "gated" in final["error"]


def test_exit_without_snapshot_is_a_failure(manager, spawned):
    record = manager.start("org/name")
    spawned[0][2].exit(0)
    assert settle(manager, record["id"])["status"] == "failed"


def test_finished_but_unloadable_snapshot_is_a_failure(manager, cache, spawned):
    record = manager.start("org/name")
    _, _, proc = spawned[0]
    snapshot = land_snapshot(cache, "org/name")
    (snapshot / "tokenizer.json").unlink()
    proc.emit(event="done", snapshot=str(snapshot))
    proc.exit(0)
    final = settle(manager, record["id"])
    assert final["status"] == "failed"
    assert "tokenizer" in final["error"].lower()


def test_cancel_terminates_the_worker_and_keeps_bytes(manager, cache, spawned):
    record = manager.start("org/name")
    _, _, proc = spawned[0]
    blobs = repo_dir(cache, "org/name") / "blobs"
    blobs.mkdir(parents=True)
    (blobs / "shard.incomplete").write_bytes(b"x" * 10)
    manager.cancel(record["id"])
    final = settle(manager, record["id"])
    assert proc.terminated
    assert final["status"] == "cancelled"
    assert (blobs / "shard.incomplete").exists()


def test_second_start_for_an_active_repo_returns_the_same_download(manager, spawned):
    first = manager.start("org/name")
    second = manager.start("org/name", "main")
    assert second["id"] == first["id"]
    assert len(spawned) == 1


def test_already_cached_model_completes_without_a_worker(manager, cache, spawned):
    land_snapshot(cache, "org/name")
    record = manager.start("org/name")
    assert record["status"] == "completed"
    assert record["already_present"] is True
    assert spawned == []


def test_incomplete_cached_snapshot_is_downloaded_again(manager, cache, spawned):
    snapshot = land_snapshot(cache, "org/name")
    (snapshot / "model.safetensors").unlink()
    record = manager.start("org/name")
    assert record["status"] == "resolving"
    assert len(spawned) == 1


def test_spawn_failure_is_reported_not_raised(cache, tmp_path):
    def broken(*_):
        raise OSError("no python")

    manager = DownloadManager(cache_dir=cache, log_dir=tmp_path / "logs", spawn=broken)
    record = manager.start("org/name")
    assert record["status"] == "failed"
    assert "no python" in record["error"]


# -- removal ------------------------------------------------------------------

def test_delete_removes_only_a_repo_inside_the_cache(manager, cache):
    land_snapshot(cache, "org/name")
    other = land_snapshot(cache, "org/other")
    result = manager.delete_cached("org/name")
    assert result["freed_bytes"] == 1000
    assert not repo_dir(cache, "org/name").exists()
    assert other.exists()
    with pytest.raises(LookupError):
        manager.delete_cached("org/name")


def test_delete_refuses_while_downloading(manager, cache, spawned):
    manager.start("org/name")
    with pytest.raises(RuntimeError):
        manager.delete_cached("org/name")
    spawned[0][2].exit(1)


def test_delete_refuses_a_symlinked_repo(manager, cache, tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    cache.mkdir(parents=True)
    repo_dir(cache, "org/link").symlink_to(elsewhere)
    with pytest.raises(LookupError):
        manager.delete_cached("org/link")
    assert elsewhere.exists()


# -- routes -------------------------------------------------------------------

@pytest.fixture
def client(manager, monkeypatch):
    from fastapi.testclient import TestClient

    from vivasecuris.aiasylum.api.main import app

    monkeypatch.setattr(models_route, "get_manager", lambda: manager)
    monkeypatch.setattr(model_downloads, "_manager", manager)
    return TestClient(app)


def test_routes_start_watch_cancel_and_list(client, manager, spawned, cache):
    r = client.post("/api/v1/models/downloads", json={"repo_id": "org/name"})
    assert r.status_code == 202, r.text
    download = r.json()
    assert download["status"] == "resolving"

    listing = client.get("/api/v1/models/downloads").json()
    assert listing["cache_dir"] == str(cache)
    assert listing["free_bytes"] is None or listing["free_bytes"] > 0
    assert [d["id"] for d in listing["downloads"]] == [download["id"]]

    assert client.get(f"/api/v1/models/downloads/{download['id']}").json()["repo_id"] == "org/name"
    assert client.get("/api/v1/models/downloads/nope").status_code == 404

    r = client.post(f"/api/v1/models/downloads/{download['id']}/cancel")
    assert r.status_code == 200
    assert spawned[0][2].terminated
    assert settle(manager, download["id"])["status"] == "cancelled"


def test_routes_reject_bad_ids_and_report_conflicts(client, spawned, cache):
    assert client.post("/api/v1/models/downloads", json={"repo_id": "llama3"}).status_code == 400
    assert client.post("/api/v1/models/downloads", json={"repo_id": "org/name", "revision": "../x"}).status_code == 400

    client.post("/api/v1/models/downloads", json={"repo_id": "org/name"})
    assert client.delete("/api/v1/models/cache/org/name").status_code == 409
    spawned[0][2].exit(1)
    settle(models_route.get_manager(), client.get("/api/v1/models/downloads").json()["downloads"][0]["id"])

    assert client.delete("/api/v1/models/cache/org/name").status_code == 404
    land_snapshot(cache, "org/name")
    r = client.delete("/api/v1/models/cache/org/name")
    assert r.status_code == 200
    assert r.json()["freed_bytes"] == 1000
    assert client.delete("/api/v1/models/cache/etc").status_code == 400


# -- the worker's own decisions ----------------------------------------------

def test_friendly_messages_name_the_fix():
    class GatedRepoError(Exception):
        pass

    class RepositoryNotFoundError(Exception):
        pass

    assert "HF_TOKEN" in model_downloads._friendly(GatedRepoError("401"), "org/x")
    assert "not found" in model_downloads._friendly(RepositoryNotFoundError("404"), "org/x")
    assert "ValueError: boom" == model_downloads._friendly(ValueError("boom"), "org/x")


def test_blob_bytes_ignores_snapshot_symlinks(cache):
    snapshot = land_snapshot(cache, "org/name")
    repo = repo_dir(cache, "org/name")
    (snapshot / "linked.safetensors").symlink_to(repo / "blobs" / "weights")
    assert model_downloads.blob_bytes(repo) == 1000
