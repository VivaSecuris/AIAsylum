#!/usr/bin/env python3
"""Export read-only experiment history without merging database run IDs.

The origin identifier persists independently of the destination server. Model
references are translated explicitly; imported run IDs never become live IDs.

This bundle includes this workspace's live and archived records. It does not
flatten previously imported bundles: copy those bundles separately, preserving
their origins and updating their explicit model_refs for the new destination.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket
import sqlite3
import uuid


TABLES = ("weight_runs", "interp_runs", "test_runs")


def _object(value):
    if isinstance(value, str):
        value = json.loads(value or "{}")
    return value if isinstance(value, dict) else {}


def _timestamp(value):
    if not isinstance(value, str):
        return None
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return (stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)).astimezone(timezone.utc).isoformat()
    except ValueError:
        return None


def _identity(row):
    durable = row["metadata"].get("lineage_id")
    return ("uuid", durable) if isinstance(durable, str) and durable else ("legacy", str(row["id"]), _timestamp(row.get("created_at")) or "")


def _read_history(workspace, warnings):
    tables = {}
    local_origins = {"server-" + socket.gethostname()}
    with sqlite3.connect(f"file:{workspace / 'data/aiasylum.db'}?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        for table in TABLES:
            tables[table] = []
            for record in conn.execute(f"SELECT * FROM {table} ORDER BY id"):
                row = dict(record)
                row["metadata"] = _object(row.get("metadata"))
                if isinstance(row.get("prompts"), str):
                    row["prompts"] = json.loads(row["prompts"])
                tables[table].append(row)

    known = {table: {_identity(row) for row in rows} for table, rows in tables.items()}
    archive = workspace / "runs/model-lineage/archive"
    for file in sorted(archive.glob("*.json")) if archive.is_dir() else []:
        try:
            payload = json.loads(file.read_text())
            table, row = payload["table"], payload["row"]
            if table not in tables or not isinstance(row, dict) or row.get("id") is None:
                raise ValueError("unrecognized archived run")
            row["metadata"] = _object(row.get("metadata"))
            if isinstance(payload.get("origin"), str) and payload["origin"].startswith("server-"):
                local_origins.add(payload["origin"])
            key = _identity(row)
            if key not in known[table]:
                tables[table].append({**row, "archived": True, "archived_at": payload.get("archived_at")})
                known[table].add(key)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            warnings.append(f"Could not export archived run {file.name}: {exc}")

    # Match the live graph's identity for pre-UUID records whose database
    # integer was reused. Importing must not collapse these into one node.
    for rows in tables.values():
        versions = {}
        for row in rows:
            if _identity(row)[0] == "legacy":
                versions.setdefault(str(row["id"]), []).append(row)
        for run_id, records in versions.items():
            records.sort(key=lambda row: _timestamp(row.get("created_at")) or "")
            for row in records[1:]:
                generation = hashlib.sha256(str(_timestamp(row.get("created_at"))).encode()).hexdigest()[:16]
                row["_lineage_identity"] = f"legacy-{run_id}-{generation}"
    return tables, local_origins


def export(workspace: Path, remote_root: str, out: Path) -> dict:
    workspace = workspace.resolve()
    remote_root = remote_root.rstrip("/")
    identity = workspace / "data/model-lineage-origin"
    if identity.exists():
        origin = identity.read_text().strip()
    else:
        origin = "workspace-" + uuid.uuid4().hex
        identity.parent.mkdir(parents=True, exist_ok=True)
        identity.write_text(origin + "\n")
    payload = {
        "version": 1, "origin": origin, "location": "Imported Mac workspace",
        "model_refs": {},
        "artifact_root": f"{remote_root}/runs/model-lineage/artifacts/{origin}",
        "artifacts": {},
        "models": [],
        "warnings": [],
    }
    tables, local_origins = _read_history(workspace, payload["warnings"])
    payload.update(tables)
    model_root = workspace / "models"

    def remember_model(ref):
        if not isinstance(ref, str) or not ref:
            return None
        path = Path(ref).expanduser()
        path = (path if path.is_absolute() else workspace / path).resolve()
        if not path.is_relative_to(model_root) or path == model_root:
            return None
        relative = path.relative_to(model_root)
        remote = f"{remote_root}/models/{relative.as_posix()}"
        payload["model_refs"].update({ref: remote, str(path): remote, f"models/{relative.as_posix()}": remote})
        return path

    for path in sorted(model_root.iterdir()) if model_root.is_dir() else []:
        if path.is_dir() and not path.name.startswith("."):
            remember_model(str(path))
            manifest_path = path / 'asylum_surgery.json'
            manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
            payload['models'].append({
                'name': path.name, 'model_ref': str(path), 'manifest': manifest,
                'size_bytes': sum(p.stat().st_size for p in path.rglob('*') if p.is_file()),
            })
    for table, rows in tables.items():
        for row in rows:
            for field in ("source_model", "model_a", "model_b", "patient_model", "doctor_model"):
                remember_model(row.get(field))
            remember_model(row["metadata"].get("modified_model"))
            patients = _object(row["metadata"].get("test_config")).get("patients", [])
            for patient in patients if isinstance(patients, list) else []:
                if isinstance(patient, dict):
                    remember_model(patient.get("model"))
            if table == "weight_runs" and row.get("out_dir"):
                path = Path(row["out_dir"])
                path = (path if path.is_absolute() else workspace / path).resolve()
                # Mirrors constants.WEIGHT_KINDS_WRITING_MODELS; this script reads
                # sqlite directly and deliberately imports nothing from the package.
                if row.get("kind") in ("surgery", "expert_surgery", "lora", "distill"):
                    remember_model(row["out_dir"])
                    if row.get("status") == "completed" and not any(item["model_ref"] == str(path) for item in payload["models"]):
                        manifest = _object(row["metadata"].get("summary")).get("manifest")
                        payload["models"].append({"name": path.name, "model_ref": str(path),
                                                  "manifest": manifest if isinstance(manifest, dict) else None,
                                                  "size_bytes": 0})
                # Archived integer IDs may now refer to another run's files.
                # Preserve recorded summaries without advertising those files.
                if not row.get("archived") and path.is_dir() and path.is_relative_to(workspace / "runs/weights"):
                    payload["artifacts"][f"weights:{row['id']}"] = {
                        "path": f"{payload['artifact_root']}/weights/{row['id']}",
                        "files": [str(p.relative_to(path)) for p in sorted(path.rglob('*')) if p.is_file()],
                    }

    for item in payload["models"]:
        remember_model(_object(item.get("manifest")).get("source_model"))

    model_ids = {}
    for original, remote in payload["model_refs"].items():
        original = Path(original)
        original = str((original if original.is_absolute() else workspace / original).resolve())
        old_id = "model:transformers:" + hashlib.sha256(original.encode()).hexdigest()[:24]
        model_ids[old_id] = "model:transformers:" + hashlib.sha256(remote.encode()).hexdigest()[:24]

    def portable_parent(parent):
        if not isinstance(parent, str):
            return parent
        for kind in ("weight", "interp", "test"):
            for local_origin in local_origins:
                prefix = f"{kind}:{local_origin}:"
                if parent.startswith(prefix):
                    return f"{kind}:{origin}:" + parent[len(prefix):]
        for old_id, new_id in model_ids.items():
            if parent == old_id:
                return new_id
            if parent == "manifest:" + old_id or parent.startswith("missing:manifest:" + old_id + ":"):
                return parent.replace(old_id, new_id, 1)
        return parent

    for rows in tables.values():
        for row in rows:
            metadata = row["metadata"]
            if "lineage_parent" in metadata:
                metadata["lineage_parent"] = portable_parent(metadata["lineage_parent"])
            test_config = metadata.get("test_config")
            if isinstance(test_config, dict) and "lineage_parent" in test_config:
                test_config["lineage_parent"] = portable_parent(test_config["lineage_parent"])

    imports = workspace / "runs/model-lineage/imports"
    if imports.is_dir() and any(imports.glob("*.json")):
        payload["warnings"].append(
            "Previously imported histories are not flattened into this workspace origin. "
            "Copy their original bundles separately and update their model_refs for the destination; "
            "fork references to those origins remain unchanged."
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = export(args.workspace, args.remote_root.rstrip('/'), args.out)
    print(json.dumps({"origin": result["origin"], "out": str(args.out),
                      "counts": {k: len(result[k]) for k in TABLES}, "warnings": result["warnings"]}))
