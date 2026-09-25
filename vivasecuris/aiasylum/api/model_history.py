"""Atomic run snapshots retained when operational run records are deleted."""

from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import settings
from vivasecuris.aiasylum.api.model_catalog import normalize_timestamp
from vivasecuris.aiasylum.constants import WEIGHT_KINDS_WRITING_MODELS
from vivasecuris.aiasylum.database import InterpRun, TestRun, WeightRun, get_session


RUN_FIELDS = {
    "weight_runs": (WeightRun, ("id", "kind", "status", "created_at", "updated_at", "started_at", "completed_at", "source_model", "source_run_id", "out_dir", "method", "objective", "error")),
    "interp_runs": (InterpRun, ("id", "mode", "status", "created_at", "updated_at", "started_at", "completed_at", "model_a", "model_b", "provider", "prompt_a", "prompt_b", "prompts", "out_dir", "error")),
    "test_runs": (TestRun, ("id", "status", "created_at", "updated_at", "patient_provider", "patient_model", "doctor_provider", "doctor_model", "test_type", "suite_id")),
}


def _default_project_root() -> Path:
    """Separate default so API tests can redirect archival writes alone."""
    return settings.project_root


def archive_run(row: Any, *, project_root: Path | None = None) -> Path:
    """Persist a complete run snapshot before deleting its database row.

    Errors deliberately propagate: callers must leave the run intact when its
    history cannot be saved. Only the row's recorded fields are serialized;
    database connections and application/provider configuration are excluded.
    This saves provenance and settings, not model weights or deleted artifacts.
    """
    table = next((name for name, (cls, _) in RUN_FIELDS.items() if isinstance(row, cls)), None)
    if table is None or row.id is None:
        raise ValueError("Only persisted weight, interpretability, and test runs can be archived.")
    fields = RUN_FIELDS[table][1]
    record = {name: normalize_timestamp(getattr(row, name)) if name.endswith("_at") else getattr(row, name)
              for name in fields}
    record["metadata"] = dict(row.meta_data or {})
    if table == "test_runs":
        record["metadata"]["summary"] = {
            **(record["metadata"].get("summary") or {}),
            "results": [{"id": result.id, "test_name": result.test_name, "test_category": result.test_category,
                         "score": result.score, "scores": result.scores} for result in row.results],
            "assessments": [{"id": assessment.id, "analysis_type": assessment.analysis_type,
                             "overall_score": assessment.overall_score, "scores": assessment.scores}
                            for assessment in row.assessments],
            "outcomes_recorded": True,
        }
    identity = record["metadata"].get("lineage_id")
    if not isinstance(identity, str) or not identity:
        # Separate reused legacy integers on disk even if a pre-UUID database
        # created multiple generations of the same run ID.
        generation = hashlib.sha256(str(record["created_at"]).encode()).hexdigest()[:16]
        identity = f"legacy-{row.id}-{generation}"
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,200}", identity):
        identity = hashlib.sha256(identity.encode()).hexdigest()
    root = (project_root or _default_project_root()) / "runs" / "model-lineage" / "archive"
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"{table.removesuffix('_runs')}-{identity}.json"
    payload = json.dumps({"version": 1, "origin": "server-" + socket.gethostname(), "table": table,
                          "archived_at": datetime.now(timezone.utc).isoformat(),
                          "row": record}, indent=2, ensure_ascii=False, allow_nan=False)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=root, prefix=".archive-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target


def read_archived_runs(project_root: Path) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    """Read snapshots without deleting, registering, or replaying any run."""
    output: dict[str, list[dict[str, Any]]] = {name: [] for name in RUN_FIELDS}
    warnings = []
    root = project_root / "runs" / "model-lineage" / "archive"
    for path in sorted(root.glob("*.json")) if root.is_dir() else []:
        try:
            payload = json.loads(path.read_text())
            table, record = payload["table"], payload["row"]
            if table not in output or not isinstance(record, dict) or record.get("id") is None:
                raise ValueError("unrecognized archived run")
            output[table].append({**record, "archived": True, "archived_at": normalize_timestamp(payload.get("archived_at"))})
        except (OSError, ValueError, KeyError, TypeError) as exc:
            warnings.append(f"Could not read archived run {path.name}: {exc}")
    return output, warnings


def saved_checkpoint_history(path: Path, *, project_root: Path | None = None) -> str | None:
    """Return evidence that this output path already identified saved weights.

    A deleted checkpoint name stays reserved so a new edit cannot inherit the
    old model's path-based graph identity. Failed attempts and comparison-only
    references do not reserve names.
    """
    project_root = project_root or _default_project_root()
    target = path.resolve()
    from vivasecuris.aiasylum.api.custom_checkpoints import deletion_records
    if str(target) in deletion_records(project_root):
        return "a deleted custom checkpoint with preserved provenance"

    def matches(ref: Any, mappings: dict[str, Any] | None = None) -> bool:
        if not isinstance(ref, str) or not ref:
            return False
        ref = (mappings or {}).get(ref, ref)
        if not isinstance(ref, str):
            return False
        candidate = Path(ref).expanduser()
        try:
            return (candidate if candidate.is_absolute() else project_root / candidate).resolve() == target
        except (OSError, RuntimeError):
            return False

    session = get_session()
    try:
        completed = session.query(WeightRun.id, WeightRun.out_dir, WeightRun.kind).filter(
            WeightRun.kind.in_(WEIGHT_KINDS_WRITING_MODELS), WeightRun.status == "completed"
        ).all()
        for row in completed:
            if matches(row.out_dir):
                return f"completed {row.kind} #{row.id}"
    finally:
        session.close()

    archived, _ = read_archived_runs(project_root)
    for row in archived["weight_runs"]:
        if row.get("kind") in WEIGHT_KINDS_WRITING_MODELS and row.get("status") == "completed" and matches(row.get("out_dir")):
            return f"archived {row.get('kind')} #{row['id']}"

    imports = project_root / "runs" / "model-lineage" / "imports"
    for file in sorted(imports.glob("*.json")) if imports.is_dir() else []:
        try:
            imported = json.loads(file.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(imported, dict):
            continue
        mappings = imported.get("model_refs")
        mappings = mappings if isinstance(mappings, dict) else {}
        origin = imported.get("location") or imported.get("origin") or file.stem
        records = imported.get("weight_runs")
        for row in records if isinstance(records, list) else []:
            if isinstance(row, dict) and row.get("kind") in WEIGHT_KINDS_WRITING_MODELS and row.get("status") == "completed" and matches(row.get("out_dir"), mappings):
                return f"imported completed {row.get('kind')} #{row.get('id')} ({origin})"
        models = imported.get("models")
        for model in models if isinstance(models, list) else []:
            if not isinstance(model, dict) or not matches(model.get("model_ref"), mappings):
                continue
            manifest = model.get("manifest")
            if isinstance(manifest, dict) and manifest.get("source_model") and manifest.get("method"):
                return f"an imported saved-checkpoint manifest ({origin})"
    return None
