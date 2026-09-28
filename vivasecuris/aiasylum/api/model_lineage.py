"""Recorded model lineage across this server and portable history exports.

Edges describe recorded inputs and outputs. Chronological neighbors are never
assumed to depend on each other; in particular a sweep and a surgery that used
the same direction remain siblings unless a fork was explicitly recorded.
"""

from __future__ import annotations

import hashlib
import json
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from config import settings
from vivasecuris.aiasylum.api.model_catalog import build_model_catalog, normalize_timestamp
from vivasecuris.aiasylum.constants import (
    WEIGHT_KINDS,
    WEIGHT_KINDS_CONSUMING_DIRECTION,
    WEIGHT_KINDS_WRITING_MODELS,
)
from vivasecuris.aiasylum.database import InterpRun, TestRun, WeightRun, get_session


def _object(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return {}
    return value if isinstance(value, dict) else {}


def _iso(value: Any) -> str | None:
    return normalize_timestamp(value)


def _run_rows(project_root: Path | None = None) -> dict[str, Any]:
    from vivasecuris.aiasylum.api.model_history import read_archived_runs

    session = get_session()
    fields = {
        WeightRun: ("id", "kind", "status", "created_at", "source_model", "source_run_id", "out_dir", "method", "objective", "error"),
        InterpRun: ("id", "mode", "status", "created_at", "model_a", "model_b", "provider", "prompt_a", "prompt_b", "prompts", "error"),
        TestRun: ("id", "status", "created_at", "patient_provider", "patient_model", "doctor_provider", "doctor_model", "test_type"),
    }
    try:
        output = {}
        for cls, name in ((WeightRun, "weight_runs"), (InterpRun, "interp_runs"), (TestRun, "test_runs")):
            output[name] = [
                {**{key: _iso(getattr(row, key)) if key == "created_at" else getattr(row, key) for key in fields[cls]},
                 "metadata": row.meta_data or {}}
                for row in session.query(cls).order_by(cls.id).all()
            ]
    finally:
        session.close()
    archived, warnings = read_archived_runs(project_root or settings.project_root)
    for table, records in archived.items():
        def identity(record: dict[str, Any]) -> tuple[str, ...]:
            durable = _object(record.get("metadata")).get("lineage_id")
            return ("uuid", durable) if isinstance(durable, str) and durable else ("legacy", str(record["id"]), _iso(record.get("created_at")) or "")

        live_keys = {identity(record) for record in output[table]}
        output[table].extend(record for record in records if identity(record) not in live_keys)
        # Extremely old databases may reuse integers without UUIDs. Preserve
        # each creation generation instead of letting one overwrite another.
        legacy: dict[str, list[dict[str, Any]]] = {}
        for record in output[table]:
            if identity(record)[0] == "legacy":
                legacy.setdefault(str(record["id"]), []).append(record)
        for run_id, versions in legacy.items():
            versions.sort(key=lambda record: _iso(record.get("created_at")) or "")
            for record in versions[1:]:
                generation = hashlib.sha256(str(_iso(record.get("created_at"))).encode()).hexdigest()[:16]
                record["_lineage_identity"] = f"legacy-{run_id}-{generation}"
    output["archive_warnings"] = warnings
    return output


def build_model_lineage(*, catalog: dict[str, Any] | None = None, project_root: Path | None = None,
                        imports_root: Path | None = None) -> dict[str, Any]:
    project_root = (project_root or settings.project_root).resolve()
    imports_root = imports_root or project_root / "runs" / "model-lineage" / "imports"
    catalog = catalog if catalog is not None else build_model_catalog()
    server_origin = "server-" + socket.gethostname()
    groups = [{"origin": server_origin, "location": catalog.get("location", socket.gethostname()),
               "model_refs": {}, "local": True, **_run_rows(project_root)}]
    warnings: list[str] = list(groups[0].get("archive_warnings", []))
    seen_origins = {server_origin}
    if imports_root.is_dir():
        for file in sorted(imports_root.glob("*.json")):
            try:
                payload = json.loads(file.read_text())
                origin = payload.get("origin")
                if not isinstance(origin, str) or not origin or len(origin) > 200 or origin in seen_origins:
                    raise ValueError("missing or duplicate origin")
                seen_origins.add(origin)
                groups.append({**payload, "local": False})
            except (OSError, ValueError, AttributeError) as exc:
                warnings.append(f"Could not read lineage import {file.name}: {exc}")

    nodes: dict[str, dict[str, Any]] = {}
    edges: set[tuple[str, str, str]] = set()
    fork_edges: list[tuple[str, str, str]] = []
    rescore_edges: list[tuple[str, str, str]] = []
    by_model: dict[tuple[str, str], str] = {}

    def normalize(ref: Any, group: dict[str, Any] | None = None) -> str | None:
        if not isinstance(ref, str) or not ref.strip():
            return None
        ref = ref.strip()
        mappings = _object((group or {}).get("model_refs"))
        ref = mappings.get(ref, ref)
        if not isinstance(ref, str):
            return None
        path = Path(ref).expanduser()
        if path.is_absolute() or ref.startswith((".", "~", "models/")):
            ref = str((path if path.is_absolute() else project_root / path).resolve())
        return ref

    def add_node(node_id: str, label: str, kind: str, **values: Any) -> str:
        nodes.setdefault(node_id, {
            "id": node_id, "label": label, "kind": kind, "status": "unknown", "created_at": None,
            "model_ref": None, "source_model": None, "run_id": None, "origin": server_origin,
            "settings": {}, "summary": {}, "links": {}, "missing_reason": None,
        }).update(label=label, kind=kind, **values)
        return node_id

    def edge(source: str | None, target: str | None, label: str) -> None:
        if source and target and source != target:
            edges.add((source, target, label))

    def model(ref: Any, group: dict[str, Any] | None = None, provider: str = "transformers") -> str | None:
        ref = normalize(ref, group)
        if ref is None:
            return None
        provider = "transformers" if provider in (None, "local") else provider
        key = (provider, ref)
        if key not in by_model:
            node_id = "model:" + provider + ":" + hashlib.sha256(ref.encode()).hexdigest()[:24]
            by_model[key] = node_id
            missing = provider == "transformers" and Path(ref).is_absolute() and not Path(ref).is_dir()
            add_node(node_id, Path(ref).name if Path(ref).is_absolute() else ref, "model",
                     model_ref=ref, status="missing" if missing else "unverified",
                     missing_reason="Model directory is missing on this server." if missing else None,
                     settings={"provider": provider},
                     links={"resume": "/weights?" + urlencode({"kind": "direction", "source_model": ref, "lineage_parent": node_id})} if provider == "transformers" else {})
        return by_model[key]

    for item in catalog.get("models", []):
        node_id = model(item["model_ref"])
        assert node_id is not None
        node = nodes[node_id]
        node.update(label=item["name"], status=item["availability"], created_at=_iso(item.get("created_at")),
                    source_model=item.get("source_model"), missing_reason=item.get("reason"),
                    settings={"provider": "transformers", "kind": item["kind"], "manifest": item.get("manifest")},
                    summary={"size_bytes": item["size_bytes"], "history": item["history"]})
        # A cache with only a pinned revision is loaded by path but referred to
        # by repo ID in historic records. It is still one model node.
        if item["kind"] == "base" and item["name"] != item["model_ref"]:
            by_model[("transformers", item["name"])] = node_id

    def rows(group: dict[str, Any], name: str) -> list[dict[str, Any]]:
        value = group.get(name, [])
        return [row for row in value if isinstance(row, dict) and row.get("id") is not None] if isinstance(value, list) else []

    def lineage_identity(row: dict[str, Any]) -> str:
        metadata = _object(row.get("metadata", row.get("meta_data")))
        durable = metadata.get("lineage_id")
        return durable if isinstance(durable, str) and durable else row.get("_lineage_identity", str(row["id"]))

    weight_identities = {(group["origin"], str(row["id"])): lineage_identity(row)
                         for group in groups for row in rows(group, "weight_runs")}

    def weight_id(group: dict[str, Any], run_id: Any, identity: str | None = None) -> str:
        suffix = identity or weight_identities.get((group["origin"], str(run_id)), str(run_id))
        return f"weight:{group['origin']}:{suffix}"

    produced: set[str] = set()
    for group in groups:
        for row in rows(group, "weight_runs"):
            metadata = _object(row.get("metadata", row.get("meta_data")))
            source_ref = normalize(row.get("source_model"), group)
            kind = row.get("kind", "direction")
            node_id = weight_id(group, row["id"], lineage_identity(row))
            run_kind = kind if kind in WEIGHT_KINDS else "test"
            summary = _object(metadata.get("summary"))
            links = {"run": f"/weights/{row['id']}"} if group.get("local") and not row.get("archived") else {}
            add_node(node_id, f"{str(kind).capitalize()} #{row['id']}", run_kind,
                     status=row.get("status", "unknown"), created_at=_iso(row.get("created_at")),
                     model_ref=source_ref, source_model=source_ref, run_id=row["id"], origin=group["origin"],
                     settings={"stage": kind, "method": row.get("method"), "objective": row.get("objective"),
                               "objective_config": metadata.get("objective_config"),
                               "options": _object(metadata.get("options")), "source_run_id": row.get("source_run_id"),
                               "modified_model": normalize(metadata.get("modified_model"), group),
                               "source_direction": _object(metadata.get("source_direction")),
                               "archived_at": row.get("archived_at"),
                               "origin_location": group.get("location"), "artifact_root": group.get("artifact_root")},
                     summary=summary, links=links, missing_reason=row.get("error"))

    def missing_parent(node_id: str, label: str, origin: str, reason: str, summary: Any = None) -> str:
        if node_id not in nodes:
            add_node(node_id, label, "missing", status="missing", origin=origin,
                     missing_reason=reason, summary=_object(summary))
            warnings.append(reason)
        return node_id

    for group in groups:
        weight_rows = rows(group, "weight_runs")
        rows_by_id: dict[str, list[dict[str, Any]]] = {}
        for candidate in weight_rows:
            rows_by_id.setdefault(str(candidate["id"]), []).append(candidate)
        row_by_identity = {lineage_identity(row): row for row in weight_rows}
        by_output = {row.get("out_dir"): row for row in weight_rows if row.get("out_dir")}
        for row in weight_rows:
            node_id = weight_id(group, row["id"], lineage_identity(row))
            node = nodes[node_id]
            metadata = _object(row.get("metadata", row.get("meta_data")))
            kind = row.get("kind")
            edge(model(row.get("source_model"), group), node_id, "baseline" if kind == "compare" else "model input")
            if kind == "compare":
                edge(model(metadata.get("modified_model"), group), node_id, "modified model")
            parent_id = row.get("source_run_id")
            parent_identity = metadata.get("source_direction_lineage_id")
            parent_identity = parent_identity if isinstance(parent_identity, str) and parent_identity else None
            if parent_id is None and metadata.get("direction_dir") in by_output:
                parent_id = by_output[metadata["direction_dir"]]["id"]
            parent_row = row_by_identity.get(parent_identity) if parent_identity else None
            if not parent_identity:
                candidates = rows_by_id.get(str(parent_id), [])
                if len(candidates) == 1:
                    parent_row = candidates[0]
                elif candidates:
                    child_time = _iso(row.get("created_at"))
                    candidates = [candidate for candidate in candidates if child_time and _iso(candidate.get("created_at")) and _iso(candidate.get("created_at")) <= child_time]
                    parent_row = max(candidates, key=lambda candidate: _iso(candidate.get("created_at"))) if candidates else None
            if parent_id is not None or parent_identity:
                parent = weight_id(group, parent_id, parent_identity or (lineage_identity(parent_row) if parent_row else f"unresolved-{parent_id}"))
                missing_parent(parent, f"Missing direction #{parent_id}", group["origin"],
                               f"Direction #{parent_id} referenced by {node['label']} from {group['origin']} is not in the recorded history.",
                               metadata.get("source_direction"))
                edge(parent, node_id, "direction input")
            elif kind in WEIGHT_KINDS_CONSUMING_DIRECTION or (
                kind == "expert_surgery" and row.get("method") == "expert_direction_scale"
            ):
                parent = f"missing:{node_id}:direction"
                missing_parent(parent, "Unrecorded direction", group["origin"],
                               f"{node['label']} has a direction snapshot but no resolvable parent run.", metadata.get("source_direction"))
                edge(parent, node_id, "direction input")

            fork_parent = metadata.get("lineage_parent")
            if isinstance(fork_parent, str) and fork_parent:
                fork_edges.append((fork_parent, node_id, group["origin"]))

            if kind in WEIGHT_KINDS_WRITING_MODELS and row.get("out_dir") and row.get("status") == "completed":
                checkpoint_id = model(row["out_dir"], group)
                edge(node_id, checkpoint_id, "saved checkpoint")
                if checkpoint_id:
                    produced.add(checkpoint_id)

            resume_options = dict(_object(metadata.get("options")))
            # Earlier runs stored these dependencies only as artifact paths.
            # Resolve against this history, never reinterpret an imported ID as
            # an unrelated live run on the current server.
            dependency = {
                "induce": ("probe_run_id", "probe_dir", "probe"),
                "hneuron_bake": ("hneurons_run_id", "hneurons_dir", "hneurons"),
            }.get(kind)
            if dependency:
                field, path_field, expected_kind = dependency
                dependency_row = by_output.get(metadata.get(path_field))
                if dependency_row is None and resume_options.get(field) is not None:
                    candidates = rows_by_id.get(str(resume_options[field]), [])
                    dependency_row = candidates[0] if len(candidates) == 1 else None
                if (group.get("local") and dependency_row and not dependency_row.get("archived")
                        and dependency_row.get("kind") == expected_kind
                        and dependency_row.get("status") == "completed"
                        and dependency_row.get("out_dir") and Path(dependency_row["out_dir"]).is_dir()):
                    resume_options[field] = dependency_row["id"]
                else:
                    resume_options.pop(field, None)
                    node["settings"]["resume_note"] = (
                        f"Choose a completed {expected_kind} run on this server before launching this branch; "
                        "the original dependency is unavailable here."
                    )
            resume = {"kind": kind, "source_model": node["source_model"], "method": row.get("method"),
                      "objective": row.get("objective"), "options": json.dumps(resume_options, separators=(",", ":")),
                      "lineage_parent": node_id}
            if isinstance(metadata.get("objective_config"), dict):
                resume["objective_config"] = json.dumps(metadata["objective_config"], separators=(",", ":"))
            if metadata.get("modified_model"):
                resume["modified_model"] = normalize(metadata["modified_model"], group)
            if kind in WEIGHT_KINDS_CONSUMING_DIRECTION or (
                kind == "expert_surgery" and row.get("method") == "expert_direction_scale"
            ):
                direction_path = Path(parent_row["out_dir"]) if parent_row and parent_row.get("out_dir") else None
                if group.get("local") and parent_row and not parent_row.get("archived") and parent_row.get("kind") == "direction" and parent_row.get("status") == "completed" and direction_path and (direction_path / "direction.safetensors").is_file():
                    resume["source_run_id"] = parent_row["id"]
                else:
                    # A remote DB integer never identifies an imported run.
                    # Recreate its direction first, retaining the fork origin.
                    node["settings"]["resume_note"] = (
                        "Recreate the recorded direction on this server first; then choose a new sweep or edit. "
                        "Imported run IDs are not live jobs."
                        if not group.get("local") else
                        "The original direction or its artifacts are unavailable. Recreate the direction first; "
                        "then choose a new sweep or edit."
                    )
                    parent_meta = _object((parent_row or {}).get("metadata", (parent_row or {}).get("meta_data")))
                    resume.update(kind="direction", method=(parent_row or {}).get("method") or "diff_in_means",
                                  options=json.dumps(_object(parent_meta.get("options")), separators=(",", ":")))
                    if parent_row and parent_row.get("objective"):
                        resume["objective"] = parent_row["objective"]
                    if isinstance(parent_meta.get("objective_config"), dict):
                        resume["objective_config"] = json.dumps(parent_meta["objective_config"], separators=(",", ":"))
            if node["source_model"]:
                node["links"]["resume"] = "/weights?" + urlencode({k: v for k, v in resume.items() if v is not None})

        for row in rows(group, "interp_runs"):
            node_id = f"interp:{group['origin']}:{lineage_identity(row)}"
            metadata = _object(row.get("metadata", row.get("meta_data")))
            source = normalize(row.get("model_a"), group)
            links = {"run": f"/interp/{row['id']}"} if group.get("local") and not row.get("archived") else {}
            links["resume"] = "/interp?" + urlencode({k: v for k, v in {
                "mode": row.get("mode"), "model_a": source, "model_b": normalize(row.get("model_b"), group),
                "prompt": row.get("prompt_a"), "prompt_a": row.get("prompt_a"), "prompt_b": row.get("prompt_b"),
                "prompts": json.dumps(row["prompts"], separators=(",", ":")) if isinstance(row.get("prompts"), list) else None,
                "options": json.dumps(_object(metadata.get("options")), separators=(",", ":")),
                "lineage_parent": node_id,
            }.items() if v is not None})
            add_node(node_id, f"{row.get('mode', 'Analysis').replace('_', ' ').capitalize()} #{row['id']}", "interp",
                     status=row.get("status", "unknown"), created_at=_iso(row.get("created_at")),
                     model_ref=source, source_model=source, run_id=row["id"], origin=group["origin"],
                     settings={"mode": row.get("mode"), "prompt_a": row.get("prompt_a"), "prompt_b": row.get("prompt_b"),
                               "prompts": row.get("prompts"), "options": _object(metadata.get("options")),
                               "archived_at": row.get("archived_at"),
                               "origin_location": group.get("location")},
                     summary=_object(metadata.get("summary")), links=links, missing_reason=row.get("error"))
            edge(model(row.get("model_a"), group, row.get("provider", "transformers")), node_id, "model A")
            edge(model(row.get("model_b"), group, row.get("provider", "transformers")), node_id, "model B")
            if isinstance(metadata.get("lineage_parent"), str) and metadata["lineage_parent"]:
                fork_edges.append((metadata["lineage_parent"], node_id, group["origin"]))

        for row in rows(group, "test_runs"):
            node_id = f"test:{group['origin']}:{lineage_identity(row)}"
            metadata = _object(row.get("metadata", row.get("meta_data")))
            test_config = metadata.get("test_config")
            if isinstance(test_config, dict):
                test_config = dict(test_config)
                # Group-therapy patients are model references as well. Preserve
                # all recorded settings while translating only portable paths.
                if isinstance(test_config.get("patients"), list):
                    test_config["patients"] = [
                        {**patient, "model": normalize(patient.get("model"), group)} if isinstance(patient, dict) else patient
                        for patient in test_config["patients"]
                    ]
            else:
                test_config = None
            campaign = _object(metadata.get("benchmark_campaign"))
            campaign_model = metadata.get("campaign_model") if row.get("test_type") == "benchmark" else None
            patient = normalize(campaign_model or row.get("patient_model"), group)
            doctor = normalize(row.get("doctor_model"), group)
            links = {"run": f"/test-runs/{row['id']}"} if group.get("local") and not row.get("archived") else {}
            links["resume"] = "/create-test?" + urlencode({key: value for key, value in {
                "provider": row.get("patient_provider"), "model": patient, "type": row.get("test_type"),
                "doctor_provider": row.get("doctor_provider"), "doctor_model": doctor,
                "test_config": json.dumps(test_config, separators=(",", ":")) if test_config is not None else None,
                "lineage_parent": node_id,
            }.items() if value is not None})
            if campaign.get("id") and group.get("local"):
                links["comparison"] = "/benchmarks?" + urlencode({"campaign": campaign["id"]})
                links["resume"] = links["comparison"]
            add_node(node_id, f"{row.get('test_type', 'Test').capitalize()} #{row['id']}", "test",
                     status=row.get("status", "unknown"), created_at=_iso(row.get("created_at")),
                     model_ref=patient, run_id=row["id"], origin=group["origin"],
                     settings={"test_type": row.get("test_type"), "patient_provider": row.get("patient_provider"),
                               "patient_model": patient, "doctor_provider": row.get("doctor_provider"), "doctor_model": doctor,
                               "test_config": test_config, "archived_at": row.get("archived_at"), "origin_location": group.get("location"),
                               "model_provenance": metadata.get("model_provenance"), "rescoring": metadata.get("rescoring"),
                               "benchmark_campaign": campaign or None},
                     summary={**_object(metadata.get("summary")), "configuration_recorded": test_config is not None}, links=links)
            for role in ("patient", "doctor"):
                if campaign and role == "doctor":
                    continue
                edge(model(patient if role == "patient" else row.get(f"{role}_model"), group, row.get(f"{role}_provider", "transformers")), node_id, role)
            rescoring = _object(metadata.get("rescoring"))
            source_id = rescoring.get("source_test_run_id")
            if type(source_id) is int and source_id > 0:
                candidates = [source for source in rows(group, "test_runs") if source.get("id") == source_id]
                durable_source = rescoring.get("source_lineage_id")
                source_identity = durable_source if isinstance(durable_source, str) and durable_source else (
                    lineage_identity(candidates[0]) if len(candidates) == 1 else str(source_id))
                rescore_edges.append((f"test:{group['origin']}:{source_identity}", node_id, group["origin"]))
            parent = metadata.get("lineage_parent") or (test_config or {}).get("lineage_parent")
            if not rescoring and isinstance(parent, str) and parent:
                fork_edges.append((parent, node_id, group["origin"]))

    # CLI checkpoints may have no run row at all. Their on-disk manifest still
    # records a real edit; do not silently attach unscoped direction_run_id to
    # this server's unrelated integer ID space.
    for item in catalog.get("models", []):
        checkpoint_id = model(item["model_ref"])
        manifest = _object(item.get("manifest"))
        if item["kind"] != "custom" or not manifest or checkpoint_id in produced:
            continue
        edit_id = f"manifest:{checkpoint_id}"
        add_node(edit_id, f"Saved edit: {item['name']}", "surgery", status="recorded",
                 created_at=_iso(item.get("created_at")), source_model=item.get("source_model"),
                 model_ref=item.get("source_model"), origin="checkpoint-manifest", settings=manifest,
                 summary={"provenance": "Checkpoint manifest; original run record is unavailable."})
        edge(model(item.get("source_model")), edit_id, "original model")
        direction_id = _object(manifest.get("extra")).get("direction_run_id")
        if direction_id is not None:
            missing_id = f"missing:{edit_id}:direction:{direction_id}"
            missing_parent(missing_id, f"Unresolved direction #{direction_id}", "checkpoint-manifest",
                           f"The manifest for {item['name']} records direction #{direction_id}, but its original run registry is unavailable.")
            edge(missing_id, edit_id, "direction input")
        edge(edit_id, checkpoint_id, "saved checkpoint")

    for parent, child, origin in fork_edges:
        missing_parent(parent, "Missing fork origin", origin,
                       f"Fork origin {parent} is not in the recorded history.")
        edge(parent, child, "forked from")
    for parent, child, origin in rescore_edges:
        missing_parent(parent, "Missing original benchmark", origin,
                       f"The saved-answer source {parent} is not in the recorded history.")
        edge(parent, child, "rescored answers")

    def order(node: dict[str, Any]) -> tuple[float, str]:
        try:
            stamp = datetime.fromisoformat(node["created_at"].replace("Z", "+00:00"))
            if stamp.tzinfo is None:
                stamp = stamp.replace(tzinfo=timezone.utc)
            return stamp.timestamp(), node["id"]
        except (ValueError, TypeError, AttributeError):
            return float("-inf"), node["id"]

    # Discovery suggestions belong in the model picker. Show them in history
    # only after a run or recorded relationship makes them part of the graph.
    connected = {node_id for source, target, _ in edges for node_id in (source, target)}

    def unused_suggestion(node: dict[str, Any]) -> bool:
        history = _object(_object(node.get("summary")).get("history"))
        has_history = any(isinstance(count, (int, float)) and count > 0 for count in history.values())
        return (node["kind"] == "model" and _object(node.get("settings")).get("kind") == "base"
                and node["status"] == "download_required" and node["id"] not in connected and not has_history)

    return {"nodes": sorted((node for node in nodes.values() if not unused_suggestion(node)), key=order),
            "edges": [{"source": a, "target": b, "label": label} for a, b, label in sorted(edges)],
            "warnings": list(dict.fromkeys(warnings))}
