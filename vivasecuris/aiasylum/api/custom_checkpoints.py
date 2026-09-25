"""Remove saved custom weights while retaining their identity and provenance."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from config import settings
from vivasecuris.aiasylum.api.model_jobs import try_process_lock
from vivasecuris.aiasylum.database import InterpRun, TestRun, WeightRun, get_session


def deletion_root(project_root: Path) -> Path:
    return project_root / "runs" / "model-lineage" / "deleted-checkpoints"


def deletion_records(project_root: Path) -> dict[str, dict]:
    records = {}
    for path in deletion_root(project_root).glob("*.json"):
        try:
            row = json.loads(path.read_text())
            if isinstance(row, dict) and isinstance(row.get("model_ref"), str):
                records[row["model_ref"]] = row
        except (ValueError, OSError):
            continue
    return records


def _write_record(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
            tmp = Path(stream.name)
            json.dump(row, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def delete_custom_checkpoints(names: list[str], *, models_root: Path | None = None,
                              project_root: Path | None = None) -> dict:
    """Delete only explicit direct children carrying a saved-surgery manifest.

    The model lease prevents another worker loading files while they disappear.
    An atomic rename removes a checkpoint from discovery before disk cleanup.
    Failed cleanup leaves a named private directory recorded for recovery, never
    a ready checkpoint. Existing results, directions and base caches are retained.
    """
    root = (models_root or settings.weights_models_root).resolve()
    project = (project_root or settings.project_root).resolve()
    if not names or len(names) > 100 or len(names) != len(set(names)):
        raise ValueError("Choose between 1 and 100 distinct custom checkpoint names.")
    lock = try_process_lock()
    if lock is None:
        raise RuntimeError("A model job is using the server. Wait for it to finish before deleting checkpoints.")
    try:
        session = get_session()
        try:
            for cls in (TestRun, InterpRun, WeightRun):
                if session.query(cls.id).filter(cls.status.in_(["pending", "running", "queued", "paused"])).first():
                    raise RuntimeError("Finish or cancel queued and active runs before deleting checkpoints.")
        finally:
            session.close()

        plans = []
        for name in names:
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,199}", name):
                raise ValueError("Checkpoint names must be plain names, not paths.")
            path = root / name
            if path.is_symlink() or path.resolve().parent != root:
                raise ValueError(f"Refusing to delete a linked or external checkpoint: {name}")
            if not path.is_dir():
                raise LookupError(f"Custom checkpoint does not exist: {name}")
            try:
                manifest = json.loads((path / "asylum_surgery.json").read_text())
                if not isinstance(manifest, dict) or not manifest.get("source_model"):
                    raise ValueError("Missing source model")
            except (OSError, ValueError) as exc:
                raise ValueError(f"{name} has no valid saved-custom-model provenance; refusing to remove it.") from exc
            files = []
            for item in path.rglob("*"):
                if item.is_symlink():
                    raise ValueError(f"{name} contains linked files; refusing to remove it.")
                if item.is_file():
                    files.append({"path": str(item.relative_to(path)), "bytes": item.stat().st_size})
            record_path = deletion_root(project) / (hashlib.sha256(str(path).encode()).hexdigest() + ".json")
            private = root / (".deleted-" + uuid4().hex)
            row = {"model_ref": str(path), "name": name, "manifest": manifest,
                   "deleted_at": datetime.now(timezone.utc).isoformat(), "status": "planned",
                   "size_bytes": sum(item["bytes"] for item in files), "files": files,
                   "cleanup_path": str(private)}
            plans.append((path, private, record_path, row))

        # Every requested path is validated and provenance is durable before the
        # first destructive operation. Failure to archive never deletes weights.
        for _, _, record_path, row in plans:
            _write_record(record_path, row)
        deleted = []
        for path, private, record_path, row in plans:
            path.rename(private)
            row["status"] = "cleanup_pending"
            _write_record(record_path, row)
            shutil.rmtree(private)
            row["status"] = "deleted"
            row.pop("cleanup_path", None)
            _write_record(record_path, row)
            deleted.append({"name": row["name"], "model_ref": row["model_ref"],
                            "freed_bytes": row["size_bytes"], "deleted_at": row["deleted_at"]})
        return {"deleted": deleted, "freed_bytes": sum(row["freed_bytes"] for row in deleted)}
    finally:
        lock.close()
