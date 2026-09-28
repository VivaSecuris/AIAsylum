"""Download Hugging Face checkpoints onto the API server, with progress.

The catalog (``model_catalog.py``) reports a base model as ``download_required``
when its weights are not in the server's Hub cache. Until now the only way to
change that from the UI was to start a run and wait for ``from_pretrained`` to
fetch the weights as a side effect: no progress, no cancellation, and on a
rented GPU box no way to see what a multi-gigabyte pull was doing.

Each download runs in a separate worker process rather than a thread, for two
reasons: ``snapshot_download`` cannot be interrupted from Python, so cancelling
means terminating a process; and the Hub client's retries and progress bars
should never block the API's event loop. Progress is measured from the bytes
that have actually landed in the repository's ``blobs`` directory, so it is
correct across a restart, a resumed download and a cancelled one, and it never
depends on parsing the Hub client's output.

Both sides of the process boundary live here: the manager the routes call, and
the worker entry point (``python -m ... --worker REPO REVISION``) it launches.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from vivasecuris.aiasylum.api.model_catalog import checkpoint_readiness
from vivasecuris.aiasylum.api.model_jobs import slot_status, try_process_lock
from vivasecuris.aiasylum.database import InterpRun, TestRun, WeightRun, get_session

# A Hub repository is ``namespace/name``; both halves are filesystem-safe by
# Hub policy, and the check here keeps them that way before they become a
# directory name under the cache.
REPO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
REVISION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,127}$")

# What a checkpoint needs to load: config, tokenizer, weights and the chat
# template. Never README images, training logs or a second weight format.
ALLOW_PATTERNS = ["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"]

# Free space that must remain after a download: a surgery run writes another
# full copy of the model next to it.
DISK_RESERVE_BYTES = 5 * 2**30

ACTIVE = frozenset({"queued", "resolving", "downloading"})
KEEP_FINISHED = 50


def hub_cache_dir() -> Path:
    """Where ``from_pretrained`` will look, so a download here is a download for the loader."""
    try:
        from huggingface_hub import constants

        return Path(constants.HF_HUB_CACHE).expanduser()
    except ImportError:
        home = Path(os.environ.get("HF_HOME") or Path("~/.cache/huggingface").expanduser())
        return Path(os.environ.get("HF_HUB_CACHE") or home / "hub").expanduser()


def repo_dir(cache: Path, repo_id: str) -> Path:
    return cache / ("models--" + repo_id.replace("/", "--"))


def blob_bytes(repo: Path) -> int:
    """Bytes downloaded so far, partial files included; symlinked snapshots are not counted twice."""
    total = 0
    blobs = repo / "blobs"
    if not blobs.is_dir():
        return 0
    for item in blobs.iterdir():
        try:
            if item.is_file() and not item.is_symlink():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def cached_snapshot(cache: Path, repo_id: str, revision: str = "main") -> Optional[Path]:
    """The snapshot ``revision`` resolves to, if that revision is in the cache."""
    repo = repo_dir(cache, repo_id)
    ref = repo / "refs" / revision
    commit = revision
    if ref.is_file():
        try:
            commit = ref.read_text().strip()
        except (OSError, UnicodeError):
            return None
    if not commit or "/" in commit or commit in (".", ".."):
        return None
    snapshot = repo / "snapshots" / commit
    return snapshot if snapshot.is_dir() else None


def validate(repo_id: str, revision: str) -> None:
    if not REPO_ID.match(repo_id or ""):
        raise ValueError("repo_id must be a Hugging Face id of the form namespace/name.")
    if not REVISION.match(revision or "") or ".." in revision.split("/"):
        raise ValueError("revision must be a branch, tag or commit hash.")


@dataclass
class Download:
    id: str
    repo_id: str
    revision: str
    status: str = "queued"
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    bytes_total: Optional[int] = None
    files_total: Optional[int] = None
    commit: Optional[str] = None
    snapshot: Optional[str] = None
    error: Optional[str] = None
    log_path: Optional[str] = None
    already_present: bool = False
    cancel_requested: bool = False

    def as_dict(self, cache: Path) -> Dict[str, Any]:
        done = blob_bytes(repo_dir(cache, self.repo_id))
        now = time.time()
        elapsed = (self.completed_at or now) - (self.started_at or self.created_at)
        return {
            "id": self.id,
            "repo_id": self.repo_id,
            "revision": self.revision,
            "status": self.status,
            "active": self.status in ACTIVE,
            "bytes_done": done,
            "bytes_total": self.bytes_total,
            "files_total": self.files_total,
            "commit": self.commit,
            "snapshot": self.snapshot,
            "error": self.error,
            "already_present": self.already_present,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "elapsed_seconds": round(elapsed, 1),
            "log_path": self.log_path,
        }


def _default_spawn(repo_id: str, revision: str, log_path: Path) -> subprocess.Popen:
    from config import settings

    env = dict(os.environ, HF_HUB_DISABLE_PROGRESS_BARS="1", PYTHONUNBUFFERED="1")
    log = log_path.open("ab")
    try:
        return subprocess.Popen(
            [sys.executable, "-m", __name__, "--worker", repo_id, revision],
            stdout=subprocess.PIPE, stderr=log, stdin=subprocess.DEVNULL,
            cwd=str(settings.project_root), env=env, text=True, bufsize=1,
            start_new_session=True,
        )
    finally:
        log.close()


class DownloadManager:
    """Start, watch, cancel and remove downloads. One per process."""

    def __init__(self, cache_dir: Optional[Path] = None, log_dir: Optional[Path] = None,
                 spawn: Optional[Callable[..., Any]] = None):
        self._cache = cache_dir
        self._log_dir = log_dir
        self._spawn = spawn or _default_spawn
        self._lock = threading.Lock()
        self._cache_mutation_lock = threading.Lock()
        self._records: Dict[str, Download] = {}
        self._procs: Dict[str, Any] = {}

    @property
    def cache(self) -> Path:
        return self._cache or hub_cache_dir()

    @property
    def log_dir(self) -> Path:
        if self._log_dir is None:
            from config import settings

            self._log_dir = Path(settings.project_root) / "runs" / "downloads"
        return self._log_dir

    def disk(self) -> Dict[str, Any]:
        cache = self.cache
        probe = cache
        while not probe.exists() and probe.parent != probe:
            probe = probe.parent
        try:
            usage = shutil.disk_usage(probe)
            free, total = usage.free, usage.total
        except OSError:
            free = total = None
        return {"cache_dir": str(cache), "free_bytes": free, "total_bytes": total}

    def list(self) -> List[Dict[str, Any]]:
        with self._lock:
            records = sorted(self._records.values(), key=lambda r: r.created_at, reverse=True)
        return [r.as_dict(self.cache) for r in records]

    def get(self, download_id: str) -> Dict[str, Any]:
        with self._lock:
            record = self._records.get(download_id)
        if record is None:
            raise LookupError(f"No download {download_id}.")
        return record.as_dict(self.cache)

    def active_for(self, repo_id: str) -> Optional[Download]:
        with self._lock:
            for record in self._records.values():
                if record.repo_id == repo_id and record.status in ACTIVE:
                    return record
        return None

    def start(self, repo_id: str, revision: str = "main") -> Dict[str, Any]:
        # Starting a download and deleting its repository must not pass their
        # respective active-download checks concurrently.
        with self._cache_mutation_lock:
            return self._start(repo_id, revision)

    def _start(self, repo_id: str, revision: str) -> Dict[str, Any]:
        repo_id = (repo_id or "").strip()
        revision = (revision or "main").strip() or "main"
        validate(repo_id, revision)

        existing = self.active_for(repo_id)
        if existing is not None:
            return existing.as_dict(self.cache)

        record = Download(id=uuid.uuid4().hex[:12], repo_id=repo_id, revision=revision)

        # Already complete in the cache: report it without touching the network.
        snapshot = cached_snapshot(self.cache, repo_id, revision)
        if snapshot is not None and checkpoint_readiness(snapshot, allowed_root=repo_dir(self.cache, repo_id))[0] == "ready":
            record.status = "completed"
            record.already_present = True
            record.snapshot = str(snapshot)
            record.commit = snapshot.name
            record.bytes_total = blob_bytes(repo_dir(self.cache, repo_id))
            record.completed_at = record.started_at = time.time()
            self._remember(record)
            return record.as_dict(self.cache)

        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.log_dir / f"{record.id}.log"
        record.log_path = str(log_path)
        record.started_at = time.time()
        record.status = "resolving"
        self._remember(record)
        try:
            proc = self._spawn(repo_id, revision, log_path)
        except Exception as exc:  # the worker could not even be launched
            self._finish(record, "failed", error=f"Could not start the download worker: {exc}")
            return record.as_dict(self.cache)
        with self._lock:
            self._procs[record.id] = proc
        threading.Thread(target=self._watch, args=(record, proc), name=f"download-{record.id}", daemon=True).start()
        return record.as_dict(self.cache)

    def cancel(self, download_id: str) -> Dict[str, Any]:
        with self._lock:
            record = self._records.get(download_id)
            proc = self._procs.get(download_id)
        if record is None:
            raise LookupError(f"No download {download_id}.")
        if record.status in ACTIVE and proc is not None:
            record.cancel_requested = True
            try:
                proc.terminate()
            except Exception:
                pass
        return record.as_dict(self.cache)

    def delete_cached(self, repo_id: str) -> Dict[str, Any]:
        """Remove cached weights only while no queued or active job can use them."""
        validate(repo_id, "main")
        with self._cache_mutation_lock:
            return self._delete_cached(repo_id)

    def _delete_cached(self, repo_id: str) -> Dict[str, Any]:
        if self.active_for(repo_id) is not None:
            raise RuntimeError(f"{repo_id} is being downloaded; cancel that first.")
        lease = try_process_lock()
        if lease is None:
            raise RuntimeError("A model job is using the server. Wait for it to finish before deleting checkpoints.")
        try:
            # The process lease covers loading and cleanup; queued rows cover
            # jobs that have pinned a checkpoint but have not acquired it yet.
            status = slot_status()
            if status["held_by"] or status["waiting"]:
                raise RuntimeError("Finish or cancel queued and active runs before deleting checkpoints.")
            session = get_session()
            try:
                for cls in (TestRun, InterpRun, WeightRun):
                    if session.query(cls.id).filter(cls.status.in_(["pending", "running", "queued", "paused"])).first():
                        raise RuntimeError("Finish or cancel queued and active runs before deleting checkpoints.")
            finally:
                session.close()
            cache = self.cache.resolve()
            repo = repo_dir(cache, repo_id)
            if repo.is_symlink() or not repo.is_dir():
                raise LookupError(f"{repo_id} is not in the cache at {cache}.")
            if repo.resolve().parent != cache:
                raise ValueError("Refusing to delete outside the cache directory.")
            freed = blob_bytes(repo)
            shutil.rmtree(repo)
            return {"repo_id": repo_id, "freed_bytes": freed}
        finally:
            lease.close()

    # -- internals -----------------------------------------------------------

    def _remember(self, record: Download) -> None:
        with self._lock:
            self._records[record.id] = record
            finished = [r for r in self._records.values() if r.status not in ACTIVE]
            for stale in sorted(finished, key=lambda r: r.created_at)[:-KEEP_FINISHED] if len(finished) > KEEP_FINISHED else []:
                self._records.pop(stale.id, None)
                self._procs.pop(stale.id, None)

    def _finish(self, record: Download, status: str, error: Optional[str] = None) -> None:
        record.status = status
        record.error = error
        record.completed_at = time.time()
        with self._lock:
            self._procs.pop(record.id, None)

    def _watch(self, record: Download, proc: Any) -> None:
        """Follow the worker's event lines, then settle the record on exit."""
        message = None
        try:
            for raw in proc.stdout:
                line = raw.strip() if isinstance(raw, str) else raw.decode(errors="replace").strip()
                if not line.startswith("{"):
                    continue
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                kind = event.get("event")
                if kind == "plan":
                    record.bytes_total = int(event.get("bytes_total") or 0)
                    record.files_total = int(event.get("files_total") or 0)
                    record.commit = event.get("commit")
                    record.status = "downloading"
                elif kind == "done":
                    record.snapshot = event.get("snapshot")
                elif kind == "error":
                    message = event.get("message")
        except Exception as exc:
            message = message or f"Lost the worker's output: {exc}"
        rc = proc.wait()

        if record.cancel_requested:
            self._finish(record, "cancelled", error="Cancelled. Downloaded bytes are kept and a new download resumes from them.")
            return
        if rc == 0 and record.snapshot:
            snapshot = Path(record.snapshot)
            availability, reason = checkpoint_readiness(snapshot, allowed_root=repo_dir(self.cache, record.repo_id))
            if availability == "ready":
                self._finish(record, "completed")
            else:
                self._finish(record, "failed", error=f"Download finished but the checkpoint is not loadable: {reason}")
            return
        if message is None:
            tail = ""
            if record.log_path:
                try:
                    tail = Path(record.log_path).read_text(errors="replace").strip().splitlines()[-1:]
                    tail = tail[0] if tail else ""
                except OSError:
                    tail = ""
            message = f"The download worker exited with status {rc}." + (f" Last log line: {tail}" if tail else "")
        self._finish(record, "failed", error=message)


_manager: Optional[DownloadManager] = None


def get_manager() -> DownloadManager:
    global _manager
    if _manager is None:
        _manager = DownloadManager()
    return _manager


# -- the worker process ------------------------------------------------------

def _emit(**event: Any) -> None:
    sys.stdout.write(json.dumps(event) + "\n")
    sys.stdout.flush()


def _friendly(exc: Exception, repo_id: str) -> str:
    name = type(exc).__name__
    text = str(exc).strip().splitlines()[0] if str(exc).strip() else name
    if name == "GatedRepoError":
        return (f"{repo_id} is gated. Accept its terms on huggingface.co, then set HF_TOKEN "
                f"on the server or run `huggingface-cli login` there.")
    if name == "RepositoryNotFoundError":
        return f"{repo_id} was not found on the Hub (or it is private and no token was given)."
    if name == "RevisionNotFoundError":
        return f"That revision does not exist for {repo_id}."
    if name in ("LocalEntryNotFoundError", "ConnectionError", "OfflineModeIsEnabled"):
        return f"The server could not reach huggingface.co: {text}"
    return f"{name}: {text}"


def worker(repo_id: str, revision: str) -> int:
    """Resolve, check disk, download. Progress events go to stdout as JSON lines."""
    cache = hub_cache_dir()
    try:
        from huggingface_hub import HfApi, snapshot_download
        from huggingface_hub.utils import disable_progress_bars
    except ImportError:
        _emit(event="error", message='huggingface_hub is not installed; install the interp extra: pip install -e ".[interp]"')
        return 1
    try:
        from vivasecuris.aiasylum.interp.core.loader import get_hf_token

        token = get_hf_token()
    except Exception:
        token = None

    disable_progress_bars()
    try:
        info = HfApi(token=token).model_info(repo_id, revision=revision, files_metadata=True)
        names = [s.rfilename for s in (info.siblings or [])]
        patterns = list(ALLOW_PATTERNS)
        if not any(n.endswith(".safetensors") for n in names):
            patterns.append("*.bin")
        wanted = [s for s in (info.siblings or []) if any(fnmatch(s.rfilename, p) for p in patterns)]
        total = sum(int(s.size or 0) for s in wanted)
        _emit(event="plan", bytes_total=total, files_total=len(wanted), commit=getattr(info, "sha", None))

        present = blob_bytes(repo_dir(cache, repo_id))
        probe = cache
        while not probe.exists() and probe.parent != probe:
            probe = probe.parent
        free = shutil.disk_usage(probe).free
        needed = max(total - present, 0) + DISK_RESERVE_BYTES
        if free < needed:
            _emit(event="error", message=(
                f"Not enough disk for {repo_id}: {free / 2**30:.1f} GiB free, need "
                f"{(total - present) / 2**30:.1f} GiB more plus a {DISK_RESERVE_BYTES / 2**30:.0f} GiB reserve."))
            return 3

        path = snapshot_download(repo_id, revision=revision, allow_patterns=patterns, token=token)
        _emit(event="done", snapshot=str(path))
        return 0
    except Exception as exc:  # any failure is reported, never swallowed
        _emit(event="error", message=_friendly(exc, repo_id))
        return 1


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        try:
            validate(sys.argv[2], sys.argv[3])
        except ValueError as exc:
            _emit(event="error", message=str(exc))
            sys.exit(2)
        sys.exit(worker(sys.argv[2], sys.argv[3]))
    sys.stderr.write("usage: python -m vivasecuris.aiasylum.api.model_downloads --worker REPO_ID REVISION\n")
    sys.exit(2)
