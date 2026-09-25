#!/usr/bin/env python3
"""Import verified CUDA smoke outputs into this server's interpretability history.

Run after migrations, on the same server as the matrix outputs and API:
    venv/bin/python scripts/import_model_matrix.py --matrix runs/model-matrix
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.validate_interp_smoke import check_result


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Validation timestamps must include a timezone")
    return parsed.astimezone(timezone.utc).replace(tzinfo=None)


def inspect_model(matrix_dir: Path, key: str, entry: dict) -> list[dict]:
    """Recheck saved evidence before creating any completed rows."""
    import numpy as np

    stage = entry.get("stages", {}).get("interp", {})
    if entry.get("status") != "passed" or stage.get("status") != "passed":
        return []
    out = Path(entry["out"]).resolve()
    report_path = Path(stage["report"]).resolve()
    if matrix_dir not in out.parents or report_path != out / "interp/report.json":
        raise ValueError("Run/report path is outside its matrix output directory")
    identity = read_json(out / "identity.json")
    report = read_json(report_path)
    if report.get("status") != "passed" or report.get("error"):
        raise ValueError("Engine smoke report is failed or incomplete")
    if identity.get("repo_id") != entry.get("model") or not identity.get("implementation"):
        raise ValueError("Model identity or implementation fingerprint is missing/mismatched")
    snapshot = identity.get("snapshot")
    if not snapshot or snapshot != entry.get("snapshot") or snapshot != report.get("model"):
        raise ValueError("Snapshot provenance does not match the engine report")
    if not all(report.get("steps", {}).get(mode, {}).get("status") == "passed" for mode in ("single", "comparison")):
        raise ValueError("Both single and comparison must have passed before import")
    started = timestamp(report["started_at"])
    candidates = []
    for mode in ("single", "comparison"):
        step = report["steps"][mode]
        if step.get("error"):
            raise ValueError(f"{mode}: failed step cannot be imported")
        source = report_path.parent / mode
        if source.is_symlink() or not source.is_dir():
            raise ValueError(f"{mode}: expected a regular artifact directory")
        files = sorted(source.iterdir())
        if any(file.is_symlink() or not file.is_file() or file.suffix not in (".json", ".npy", ".html") for file in files):
            raise ValueError(f"{mode}: artifact directory contains links, subdirectories or unexpected files")
        for file in files:
            if file.suffix == ".json":
                json.loads(file.read_text())  # optional artifacts must be usable by the API too
        meta = read_json(source / "meta.json")
        if meta.get("model") != snapshot or meta.get("analysis_mode", mode) != mode:
            raise ValueError(f"{mode}: artifact metadata does not match the run")
        options = dict(step["options"])
        required_options = {"device", "dtype", "max_len", "window", "topk", "dim_reduction",
                            "enable_attention_capture", "enable_mlp_capture", "enable_patching"}
        if not required_options <= options.keys():
            raise ValueError(f"{mode}: exact execution options are missing; rerun the smoke validator")
        device, dtype = meta["device"], meta["dtype"]
        if not device.startswith("cuda") or dtype != "bfloat16":
            raise ValueError(f"{mode}: expected recorded CUDA/BF16 execution")
        if dtype != report.get("dtype") or dtype != options["dtype"]:
            raise ValueError(f"{mode}: recorded precision differs between outputs")
        if meta.get("requested_device", options["device"]) != options["device"]:
            raise ValueError(f"{mode}: requested device differs between outputs")
        if not options["enable_attention_capture"] or not options["enable_mlp_capture"] or options["enable_patching"] != (mode == "comparison"):
            raise ValueError(f"{mode}: capture options do not match the smoke outputs")
        if options["window"] != meta["window"] or options["dim_reduction"] != meta["dim_reduction"]:
            raise ValueError(f"{mode}: analysis options differ between outputs")
        prompt_a, prompt_b = step["prompt_a"], step.get("prompt_b")
        if not isinstance(prompt_a, str) or not prompt_a.strip():
            raise ValueError(f"{mode}: actual prompt is missing")
        if mode == "comparison" and (not prompt_b or prompt_a != meta.get("prompt_a") or prompt_b != meta.get("prompt_b")):
            raise ValueError("comparison: actual prompts differ from artifact metadata")
        if mode == "single" and meta.get("prompt_preview") != prompt_a[:200] + ("..." if len(prompt_a) > 200 else ""):
            raise ValueError("single: actual prompt differs from artifact metadata")
        payloads = {name: read_json(source / filename) for name, filename in (
            ("pca_payload", "pca_payload.json"), ("predictions_payload", "predictions.json"),
            ("attention_payload", "attention_payload.json"), ("mlp_payload", "mlp_payload.json"),
        )}
        if mode == "single":
            payloads["activation_norm_mat"] = np.load(source / "activation_norms.npy", allow_pickle=False)
        else:
            payloads.update(cos_mat=np.load(source / "cos_mat.npy", allow_pickle=False),
                            dn_mat=np.load(source / "dn_mat.npy", allow_pickle=False),
                            patching_results=read_json(source / "patching_results.json"))
        html = (source / "dashboard.html").read_text()
        checked = check_result(SimpleNamespace(**payloads), html, mode, report["layers"])
        if checked["matrix_shapes"] != step["matrix_shapes"] or meta.get("num_layers") != report["layers"] + 1:
            raise ValueError(f"{mode}: saved matrix dimensions differ from the validated dimensions")
        completed = timestamp(step["completed_at"])
        if completed < started:
            raise ValueError(f"{mode}: completion predates execution")
        provenance = {"matrix_key": key, "source_report": str(report_path), "source_artifacts": str(source),
                      "source_fingerprint": out.name, "implementation": identity["implementation"],
                      "repo_id": identity["repo_id"], "snapshot": snapshot, "commit": Path(snapshot).name,
                      "imported_at": datetime.now(timezone.utc).isoformat()}
        provenance["source_key"] = hashlib.sha256(json.dumps(
            [str(report_path), out.name, identity["implementation"], snapshot, mode]
        ).encode()).hexdigest()
        candidates.append({"mode": mode, "model_a": identity["repo_id"], "prompt_a": prompt_a,
                           "prompt_b": prompt_b, "started_at": started, "completed_at": completed,
                           "source": source, "files": files, "meta_data": {
                               "options": options, "matrix_import": provenance,
                               "summary": {**{name: meta.get(name) for name in ("spike_layer", "num_layers", "window_len")},
                                           **checked, "device": device, "dtype": dtype,
                                           "preflight": {"device": device, "dtype": dtype, "recorded_execution": True},
                                           "gpu": report.get("gpu"), "peak_allocated_gib": report.get("peak_allocated_gib")},
                           }})
    return candidates


def import_model(candidates: list[dict], runs_root: Path, session) -> list[dict]:
    """Commit each model's two rows only after both artifact copies are ready."""
    from vivasecuris.aiasylum.database import InterpRun

    staged = []
    created = []
    results = []
    try:
        existing = {row.meta_data.get("matrix_import", {}).get("source_key"): row
                    for row in session.query(InterpRun).all() if row.meta_data}
        for candidate in candidates:
            previous = existing.get(candidate["meta_data"]["matrix_import"]["source_key"])
            if previous:
                if previous.status != "completed" or not previous.out_dir or not (Path(previous.out_dir) / "dashboard.html").is_file():
                    raise ValueError(f"Existing imported run {previous.id} is incomplete; repair or delete it before reimporting")
                results.append({"id": previous.id, "mode": candidate["mode"], "status": "already_imported"})
                continue
            row = InterpRun(**{k: v for k, v in candidate.items() if k not in ("source", "files")}, status="pending")
            session.add(row)
            session.flush()
            destination = runs_root / str(row.id)
            if destination.exists():
                raise ValueError(f"Refusing to overwrite existing artifact directory {destination}")
            temporary = Path(tempfile.mkdtemp(prefix=".matrix-import-", dir=runs_root))
            staged.append(temporary)
            for file in candidate["files"]:
                if file.is_symlink():
                    raise ValueError(f"Artifact changed to a symlink during import: {file}")
                shutil.copy2(file, temporary / file.name)
            temporary.rename(destination)
            staged.remove(temporary)
            created.append(destination)
            row.out_dir = str(destination)
            row.status = "completed"
            results.append({"id": row.id, "mode": row.mode, "status": "imported", "out_dir": row.out_dir})
        session.commit()
        return results
    except BaseException:
        session.rollback()
        for directory in staged + created:
            shutil.rmtree(directory, ignore_errors=True)
        raise


def import_matrix(matrix_dir: Path, runs_root: Path = ROOT / "runs/interp", dry_run: bool = False) -> dict:
    from vivasecuris.aiasylum.database import get_session

    matrix_dir, runs_root = matrix_dir.resolve(), runs_root.resolve()
    manifest = read_json(matrix_dir / "matrix-report.json")
    if manifest.get("action") != "run":
        raise ValueError("Only a validation run report can be imported; download reports are not evidence")
    summary = {"imported": [], "skipped": [], "refused": []}
    inspections = []
    for key, entry in manifest.get("results", {}).items():
        try:
            candidates = inspect_model(matrix_dir, key, entry)
            if candidates:
                inspections.append((key, candidates))
            else:
                summary["skipped"].append({"model": key, "reason": "validation did not pass"})
        except (OSError, ValueError, KeyError, TypeError) as exc:
            summary["refused"].append({"model": key, "error": str(exc)})
    if dry_run:
        summary["would_import"] = [{"model": key, "modes": [c["mode"] for c in candidates]} for key, candidates in inspections]
        return summary
    if not inspections:
        return summary
    import fcntl
    runs_root.mkdir(parents=True, exist_ok=True)
    with (runs_root / ".matrix-import.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another matrix import is in progress; retry after it finishes") from exc
        session = get_session()
        try:
            for key, candidates in inspections:
                try:
                    summary["imported"].extend({"model": key, **result} for result in import_model(candidates, runs_root, session))
                except Exception as exc:
                    summary["refused"].append({"model": key, "error": str(exc)})
        finally:
            session.close()
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=ROOT / "runs/model-matrix")
    parser.add_argument("--dry-run", action="store_true", help="Validate artifacts without writing database rows or copies")
    args = parser.parse_args(argv)
    try:
        summary = import_matrix(args.matrix, dry_run=args.dry_run)
        print(json.dumps(summary, indent=2))
        return 1 if summary["refused"] else 0
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
