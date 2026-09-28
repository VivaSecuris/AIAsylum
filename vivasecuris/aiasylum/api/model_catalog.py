"""Read-only inventory of checkpoints on the API server.

The filesystem is the source of truth for availability; run records only add
provenance and history. Listing this catalog never downloads weights, imports a
model runtime, or creates database rows. ``ready`` means the required files are
present, not that the checkpoint fits the accelerator or passed an evaluation.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from config import settings
from vivasecuris.aiasylum.constants import WEIGHT_KINDS_WRITING_MODELS
from vivasecuris.aiasylum.database import InterpRun, TestRun, WeightRun, get_session


def normalize_timestamp(value: Any) -> str | None:
    """Serialize recorded timestamps uniformly; SQLite's naive dates are UTC."""
    if isinstance(value, str):
        if "T" not in value and " " not in value:
            return None
        try:
            value = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def _json_object(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else None
    except (OSError, UnicodeError, ValueError):
        return None


def _nonempty(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError, RuntimeError):
        return False


def checkpoint_readiness(path: Path, *, allowed_root: Path | None = None) -> tuple[str, str | None]:
    """Check complete Transformers files, including every indexed weight shard.

    HF snapshots link weights into the repository's ``blobs`` directory, so the
    caller may supply that repository as the containment root. A custom shard
    index may never reference outside its checkpoint directory.
    """
    if not path.is_dir():
        return "missing", "Checkpoint directory is missing on this server."
    config = _json_object(path / "config.json")
    if not config:
        return "incomplete", "Missing or unreadable model config.json."

    allowed_root = allowed_root or path
    index = next((path / name for name in (
        "model.safetensors.index.json", "pytorch_model.bin.index.json"
    ) if (path / name).exists()), None)
    if index is not None:
        payload = _json_object(index)
        weight_map = payload.get("weight_map") if payload else None
        if not isinstance(weight_map, dict) or not weight_map:
            return "incomplete", "Weight shard index is empty or unreadable."
        for name in weight_map.values():
            if not isinstance(name, str) or not name or "\\" in name:
                return "incomplete", "Weight shard index contains an unsafe path."
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts or not _within(path / name, allowed_root):
                return "incomplete", "Weight shard index contains an unsafe path."
            if not _nonempty(path / name):
                return "incomplete", f"Missing or empty weight shard: {name}."
    else:
        weights = [path / name for name in ("model.safetensors", "pytorch_model.bin")]
        if not any(_nonempty(item) and _within(item, allowed_root) for item in weights):
            return "incomplete", "Missing complete model weights or a weight shard index."

    # A tokenizer configuration alone has no vocabulary. These are the common
    # fast-tokenizer, SentencePiece, and BPE vocabulary representations.
    tokenizers = ("tokenizer.json", "tokenizer.model", "spiece.model", "vocab.txt")
    has_tokenizer = any(_nonempty(path / name) for name in tokenizers)
    has_tokenizer = has_tokenizer or all(_nonempty(path / name) for name in ("vocab.json", "merges.txt"))
    if not has_tokenizer:
        return "incomplete", "Missing tokenizer vocabulary files."
    return "ready", None


def _size_bytes(path: Path) -> int:
    total = 0
    try:
        for file in path.rglob("*"):
            try:
                if file.is_file():
                    total += file.stat().st_size
            except OSError:
                continue
    except OSError:
        pass
    return total


def checkpoint_status(path: Path) -> dict[str, Any]:
    """Shared local-checkpoint preflight for callers about to load weights."""
    allowed_root = None
    # Pinned cache snapshots are also usable directly as model paths. Their
    # ordinary HF blob links must not be mistaken for an incomplete checkpoint.
    if path.parent.name == "snapshots" and path.parent.parent.name.startswith("models--"):
        allowed_root = path.parent.parent
    availability, reason = checkpoint_readiness(path, allowed_root=allowed_root)
    return {"availability": availability, "reason": reason, "size_bytes": _size_bytes(path)}


def _cache_roots() -> list[Path]:
    hf_home = Path(os.environ.get("HF_HOME") or Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")).expanduser() / "huggingface")
    default = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE") or str(hf_home / "hub")
    roots = [os.environ.get("TRANSFORMERS_CACHE"), default]
    return list(dict.fromkeys(Path(value).expanduser().resolve() for value in roots if value))


def _cached_models(roots: list[Path]) -> dict[str, tuple[Path, Path, bool]]:
    """repo ID -> (snapshot directory, repository directory, default revision)."""
    out = {}
    for root in roots:
        if not root.is_dir():
            continue
        for repo in sorted(root.glob("models--*")):
            if not repo.is_dir():
                continue
            model_id = repo.name[len("models--"):].replace("--", "/")
            if model_id in out:
                continue
            try:
                revision = (repo / "refs" / "main").read_text().strip()
            except (OSError, UnicodeError):
                revision = ""
            if revision and "/" not in revision and "\\" not in revision and revision not in (".", ".."):
                # A partial current snapshot must not borrow readiness from an
                # older complete revision the default loader would not select.
                out[model_id] = (repo / "snapshots" / revision, repo, True)
                continue
            snapshots = sorted((repo / "snapshots").glob("*"))
            snapshots = [p for p in snapshots if p.is_dir()]
            if snapshots:
                snapshot = next((p for p in snapshots if checkpoint_readiness(p, allowed_root=repo)[0] == "ready"), snapshots[0])
                out[model_id] = (snapshot, repo, False)
    return out


def build_model_catalog(
    *, models_root: Path | None = None, project_root: Path | None = None,
    cache_roots: list[Path] | None = None, presets_path: Path | None = None,
    imports_root: Path | None = None,
) -> dict[str, Any]:
    project_root = (project_root or settings.project_root).resolve()
    models_root = (models_root or settings.weights_models_root).resolve()
    # An explicitly supplied preset file is a caller-owned inventory. The
    # ordinary UI includes common models independently of the validation matrix.
    common = (_json_object(project_root / "config" / "common_models.json") or {}) if presets_path is None else {}
    common_metadata = {
        item["repo_id"]: {"family": item["family"], "gated": item.get("gated", False)}
        for item in common.get("models", [])
        if isinstance(item, dict) and isinstance(item.get("repo_id"), str) and isinstance(item.get("family"), str)
    }
    presets_path = presets_path or project_root / "config" / "interp_model_matrix.json"
    imports_root = imports_root or project_root / "runs" / "model-lineage" / "imports"
    cached = _cached_models(_cache_roots() if cache_roots is None else cache_roots)
    snapshot_aliases = {str(snapshot.resolve()): repo for repo, (snapshot, _, _) in cached.items()}
    observed_refs: dict[str, set[str]] = {}
    for snapshot, repo in snapshot_aliases.items():
        observed_refs.setdefault(repo, set()).add(snapshot)

    def canonical(ref: str | None) -> str | None:
        if not isinstance(ref, str) or not ref.strip():
            return None
        original_ref = ref
        ref = ref.strip()
        path = Path(ref).expanduser()
        looks_local = path.is_absolute() or ref.startswith((".", "~")) or ref.split("/")[0] == models_root.name
        if looks_local or (project_root / path).exists():
            path = path if path.is_absolute() else project_root / path
            key = snapshot_aliases.get(str(path.resolve()), str(path.resolve()))
        else:
            key = ref
        # Consumers can merge only identities the server resolved to the same
        # checkpoint; names or path suffixes alone do not establish identity.
        observed_refs.setdefault(key, set()).update((original_ref, ref))
        return key

    session = get_session()
    try:
        # Query only model identity and provenance. Assessment scores are not
        # part of inventory, and do not determine whether a model is visible.
        tests = session.query(TestRun.id, TestRun.patient_provider, TestRun.patient_model,
                              TestRun.doctor_provider, TestRun.doctor_model).all()
        interps = session.query(InterpRun.id, InterpRun.model_a, InterpRun.model_b, InterpRun.provider).all()
        weights = session.query(WeightRun.id, WeightRun.kind, WeightRun.source_model,
                                WeightRun.out_dir, WeightRun.created_at, WeightRun.meta_data).all()
    finally:
        session.close()

    entries: dict[str, dict[str, Any]] = {}
    history: dict[str, dict[str, set[Any]]] = {}

    def remember(ref: str | None, kind: str, run_id: Any) -> None:
        key = canonical(ref)
        if key:
            history.setdefault(key, {"test_runs": set(), "interp_runs": set(), "weight_runs": set()})[kind].add(run_id)

    def entry(key: str, **values: Any) -> dict[str, Any]:
        row = entries.setdefault(key, {
            "id": "transformers:" + hashlib.sha256(key.encode()).hexdigest()[:24],
            "model_ref": key, "name": key.rsplit("/", 1)[-1], "kind": "base",
            "provider": "transformers", "source_model": None,
            "availability": "download_required", "reason": "Weights are not downloaded on this server.",
            "size_bytes": 0, "created_at": None, "run_id": None, "manifest": None,
            "history": {"test_runs": 0, "interp_runs": 0, "weight_runs": 0},
        })
        row.update(values)
        return row

    def local(path: Path, *, source: str | None = None, run_id: int | None = None, created_at: Any = None) -> dict[str, Any]:
        key = canonical(str(path))
        assert key is not None
        manifest = _json_object(path / "asylum_surgery.json")
        availability, reason = checkpoint_readiness(path)
        row = entry(key, name=path.name, kind="custom", availability=availability, reason=reason,
                    size_bytes=_size_bytes(path), manifest=manifest)
        row["source_model"] = canonical((manifest or {}).get("source_model") or source)
        manifest_created = (manifest or {}).get("created_at")
        row["created_at"] = normalize_timestamp(manifest_created) or normalize_timestamp(created_at) or row["created_at"]
        if run_id is not None:
            row["run_id"] = run_id
        return row

    if models_root.is_dir():
        for path in sorted(models_root.iterdir()):
            if path.is_dir() and not path.name.startswith(".") and any((path / name).exists() for name in ("config.json", "asylum_surgery.json")):
                local(path)

    for row in tests:
        for provider, ref in ((row.patient_provider, row.patient_model), (row.doctor_provider, row.doctor_model)):
            if provider in ("transformers", "local"):
                remember(ref, "test_runs", row.id)
    for row in interps:
        if row.provider in ("transformers", "local", None):
            for ref in (row.model_a, row.model_b):
                remember(ref, "interp_runs", row.id)
    for row in weights:
        for ref in (row.source_model, (row.meta_data or {}).get("modified_model")):
            remember(ref, "weight_runs", row.id)
        if row.kind in WEIGHT_KINDS_WRITING_MODELS and row.out_dir:
            remember(row.out_dir, "weight_runs", row.id)
            path = Path(row.out_dir).expanduser()
            path = path if path.is_absolute() else project_root / path
            local(path, source=row.source_model, run_id=row.id, created_at=row.created_at)

    # Portable provenance can advertise checkpoints awaiting transfer. Their
    # availability still comes exclusively from this server's filesystem.
    seen_origins = set()
    for file in sorted(imports_root.glob("*.json")) if imports_root.is_dir() else []:
        imported = _json_object(file) or {}
        origin = imported.get("origin")
        if not isinstance(origin, str) or not origin or origin in seen_origins:
            continue
        seen_origins.add(origin)
        mappings = imported.get("model_refs") or {}
        mappings = mappings if isinstance(mappings, dict) else {}

        def imported_ref(ref: Any) -> str | None:
            return canonical(mappings.get(ref, ref)) if isinstance(ref, str) else None

        for item in imported.get("models", []):
            if not isinstance(item, dict):
                continue
            key = imported_ref(item.get("model_ref"))
            if not key or not Path(key).is_absolute():
                continue
            row = entries.get(key) or local(Path(key))
            provenance = item.get("manifest")
            if row["manifest"] is None and isinstance(provenance, dict):
                row["manifest"] = provenance
                row["source_model"] = imported_ref(provenance.get("source_model"))
                created = provenance.get("created_at")
                row["created_at"] = normalize_timestamp(created)
            if row["availability"] == "missing":
                row["reason"] = f"Checkpoint is recorded in {imported.get('location') or origin}; its weights are not on this server."

        for table, refs in (
            ("weight_runs", ("source_model",)),
            ("interp_runs", ("model_a", "model_b")),
            ("test_runs", ("patient_model", "doctor_model")),
        ):
            records = imported.get(table, [])
            for record in records if isinstance(records, list) else []:
                if not isinstance(record, dict) or record.get("id") is None:
                    continue
                run_key = f"{origin}:{record['id']}"
                for field in refs:
                    if table == "test_runs" and record.get(field.replace("_model", "_provider")) not in ("transformers", "local"):
                        continue
                    remember(imported_ref(record.get(field)), table, run_key)
                if table == "weight_runs":
                    metadata = record.get("metadata", record.get("meta_data", {}))
                    if isinstance(metadata, str):
                        try:
                            metadata = json.loads(metadata)
                        except ValueError:
                            metadata = {}
                    if isinstance(metadata, dict):
                        remember(imported_ref(metadata.get("modified_model")), table, run_key)
                    if record.get("kind") in WEIGHT_KINDS_WRITING_MODELS:
                        remember(imported_ref(record.get("out_dir")), table, run_key)

    # Deleted checkpoint identities remain visible in history even when they
    # had no run row. Include their originals in the base-model inventory too.
    from vivasecuris.aiasylum.api.custom_checkpoints import deletion_records
    for ref, record in deletion_records(project_root).items():
        if Path(ref).exists():
            continue
        row = entries.get(ref) or local(Path(ref))
        row.update(availability="deleted", reason="Custom weights were deleted. Saved results and history are preserved.",
                   manifest=record.get("manifest"), size_bytes=0)
        row["source_model"] = canonical((record.get("manifest") or {}).get("source_model"))

    presets = _json_object(presets_path) or {}
    base_refs = set(cached) | set(history)
    base_refs.update(common_metadata)
    base_refs.update(item["repo_id"] for item in presets.get("models", []) if isinstance(item, dict) and isinstance(item.get("repo_id"), str))
    base_refs.update(row["source_model"] for row in entries.values() if row["source_model"])
    for ref in sorted(base_refs):
        key = canonical(ref)
        if not key or key in entries:
            continue
        if Path(key).is_absolute():
            local(Path(key))
        elif key in cached:
            snapshot, repo, is_default = cached[key]
            availability, reason = checkpoint_readiness(snapshot, allowed_root=repo)
            entry(key, name=key, model_ref=key if is_default else str(snapshot.resolve()),
                  availability=availability, reason=reason, size_bytes=_size_bytes(snapshot))
        else:
            entry(key, name=key)

    for key, row in entries.items():
        if key in common_metadata:
            row.update(common_metadata[key])
        row["history"] = {kind: len(ids) for kind, ids in history.get(key, {
            "test_runs": set(), "interp_runs": set(), "weight_runs": set(),
        }).items()}
        references = observed_refs.get(key, set()) - {row["model_ref"]}
        if key in cached and not cached[key][2]:
            # No main ref means the explicit snapshot is usable, but a bare
            # repo ID could fetch different weights. Do not equate those load
            # requests merely because they share an inventory provenance key.
            references = {ref for ref in references if ref.strip() != key}
        row["aliases"] = sorted(references)

    return {"models": sorted(entries.values(), key=lambda row: (row["kind"] != "custom", row["name"].lower())),
            "location": socket.gethostname()}
