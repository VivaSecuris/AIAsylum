"""Persisted, paired benchmark comparisons built from ordinary test runs.

Every model receives the same pinned dataset, seed and generation budget. Runs
stay in the existing test/history tables, so responses remain inspectable.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
from collections import Counter, defaultdict
from uuid import uuid4

from sqlalchemy.orm import selectinload

from vivasecuris.aiasylum.database import TestRun, get_session
from vivasecuris.aiasylum.api.model_catalog import build_model_catalog, normalize_timestamp

logger = logging.getLogger(__name__)
SUPPORTED = ("mmlu", "gsm8k", "hellaswag", "arc")
_tasks: dict[str, asyncio.Task] = {}


def _rows(session):
    return session.query(TestRun).options(selectinload(TestRun.results), selectinload(TestRun.conversations)).filter(TestRun.test_type == "benchmark").order_by(TestRun.id).all()


def _result_error(result, detail, provenance, requested):
    """Reject incomplete or internally inconsistent evidence without changing it."""
    if result is None:
        return "Worker marked this run completed without saving a benchmark result."
    count, correct, accuracy = detail.get("num_samples"), detail.get("correct"), detail.get("accuracy")
    if type(count) is not int or not 0 < count <= requested or type(correct) is not int or not 0 <= correct <= count:
        return "Saved benchmark sample or correct-answer counts are invalid."
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value)
           for value in (result.score, accuracy)):
        return "Saved benchmark accuracy is missing or invalid."
    if not math.isclose(result.score, correct / count, abs_tol=1e-9) or not math.isclose(accuracy, result.score, abs_tol=1e-9):
        return "Saved benchmark accuracy does not match the scored sample counts."
    answers = detail.get("results")
    if (not isinstance(answers, list) or len(answers) != count
            or any(not isinstance(answer, dict) or type(answer.get("correct")) is not bool for answer in answers)
            or sum(answer["correct"] for answer in answers) != correct):
        return "Saved individual answers do not match the benchmark totals."
    if provenance and not isinstance(provenance, dict):
        return "Saved sample provenance is malformed."
    if provenance:
        available = provenance.get("available_count")
        if type(available) is int and count != min(requested, available):
            return "Benchmark scored fewer samples than the requested available dataset."
        ids = provenance.get("sample_ids")
        if (provenance.get("actual_count") != count or provenance.get("requested_count") != requested
                or not isinstance(ids, list) or len(ids) != count
                or any(not isinstance(sample_id, str) or not sample_id for sample_id in ids)
                or len(set(ids)) != count or [answer.get("sample_id") for answer in answers] != ids):
            return "Saved sample provenance does not match the scored answers."
    return None


def _protocol_verified(meta, detail, config, benchmark):
    from vivasecuris.aiasylum.benchmarks.datasets import BENCHMARK_DATASETS
    from vivasecuris.aiasylum.benchmarks.simple import SCORING_VERSION
    expected = BENCHMARK_DATASETS.get(benchmark, {})
    provenance = detail.get("dataset_provenance") or {}
    generation = detail.get("generation") or {}
    scoring = detail.get("scoring") or {}
    if not all(isinstance(value, dict) for value in (provenance, generation, scoring)):
        return False
    revision = (meta.get("test_config") or {}).get("dataset_revision")
    return bool(
        provenance.get("ordered_sample_hash") and revision
        and provenance.get("resolved_revision") == revision
        and provenance.get("dataset") == expected.get("dataset")
        and provenance.get("config") == expected.get("config")
        and provenance.get("split") == expected.get("split")
        and generation.get("seed") == config["seed"]
        and generation.get("max_new_tokens") == config["max_new_tokens"]
        and generation.get("temperature") == 0.0
        and generation.get("independent_questions") is True
        and generation.get("prompt_protocol")
        and scoring.get("version") == SCORING_VERSION
        and scoring.get("method") == ("numeric_exact_match" if benchmark == "gsm8k" else "mcq_final_answer")
    )


def summarize(rows: list[TestRun]) -> dict:
    config = rows[0].meta_data["benchmark_campaign"]
    runs = []
    signatures = defaultdict(set)
    warnings = ["Small-sample screening with generated answers; these are not official leaderboard scores."]
    verified = True
    older_scoring = False
    fewer_samples = False
    expected_pairs = Counter((model, benchmark) for model in config["models"] for benchmark in config["benchmarks"])
    actual_pairs = Counter()
    for row in rows:
        meta = row.meta_data or {}
        # Relationship row order is not a contract. Always inspect the latest saved result.
        result = max(row.results, key=lambda item: item.id) if row.results else None
        detail = (result.meta_data or {}) if result else {}
        provenance = detail.get("dataset_provenance")
        benchmark = meta.get("benchmark")
        model = meta.get("campaign_model", row.patient_model)
        actual_pairs[(model, benchmark)] += 1
        error = None
        if row.status == "completed":
            from vivasecuris.aiasylum.benchmarks.simple import SCORING_VERSION
            saved_scoring = detail.get("scoring") or {}
            older_scoring |= not isinstance(saved_scoring, dict) or saved_scoring.get("version") != SCORING_VERSION
            error = _result_error(result, detail, provenance, config["num_samples"])
            if error or not _protocol_verified(meta, detail, config, benchmark):
                verified = False
            else:
                signatures[benchmark].add(json.dumps({
                    "hash": provenance["ordered_sample_hash"], "count": provenance["actual_count"],
                    "dataset": provenance.get("dataset"), "config": provenance.get("config"), "split": provenance.get("split"),
                    "revision": provenance.get("resolved_revision"),
                    "generation": detail.get("generation"), "scoring": detail.get("scoring"),
                }, sort_keys=True))
            fewer_samples |= not error and detail["num_samples"] < config["num_samples"]
        valid_complete = row.status == "completed" and not error
        actual_total = detail.get("num_samples") if valid_complete else config["num_samples"]
        runs.append({
            "id": row.id, "model": model, "benchmark": benchmark,
            "status": "failed" if error else row.status, "stored_status": row.status,
            "score": result.score if valid_complete else None,
            "num_correct": detail.get("correct") if valid_complete else None,
            "num_samples": detail.get("num_samples") if valid_complete else None,
            "truncated_count": detail.get("truncated_count") if valid_complete else None,
            "invalid_answer_count": detail.get("invalid_answer_count") if valid_complete else None,
            "requested_samples": config["num_samples"],
            "progress": {"current": actual_total if valid_complete else len(row.conversations), "total": actual_total},
            "error": error or meta.get("error"), "provenance": provenance,
            "generation": detail.get("generation"), "scoring": detail.get("scoring"),
            "model_provenance": meta.get("model_provenance"),
            "rescoring": meta.get("rescoring"),
        })
    completed = sum(r["status"] == "completed" for r in runs)
    expected = sum(expected_pairs.values())
    complete_pairs = actual_pairs == expected_pairs
    active = any(r["status"] == "running" for r in runs)
    pending = any(r["status"] == "pending" for r in runs)
    status = "running" if active else "pending" if pending else "completed" if completed == expected and complete_pairs else "partial" if completed else "failed"
    matching = verified and all(len(s) == 1 for s in signatures.values())
    if not matching:
        warnings.append("Sample or scoring provenance differs or is missing. Do not rank these results as a paired comparison.")
    if older_scoring:
        warnings.append("Saved results use an older or unknown answer scorer. Re-score complete saved answers with the current scorer, or run a new comparison. Original results remain intact.")
    if not complete_pairs:
        warnings.append("Some model/benchmark pairs are missing or duplicated. This comparison is incomplete.")
    if fewer_samples:
        warnings.append("The dataset contained fewer matching samples than requested; scores use the actual sample counts shown.")
    if status in ("partial", "failed"):
        warnings.append("Some runs did not complete with valid results. Open their errors before drawing conclusions.")
    return {**config, "created_at": normalize_timestamp(rows[0].created_at), "status": status,
            "completed": completed, "total": expected, "runs": runs,
            "comparable": matching and complete_pairs and completed == expected,
            "scoring_outdated": older_scoring, "warnings": warnings}


def list_campaigns() -> list[dict]:
    with get_session() as session:
        groups = defaultdict(list)
        for row in _rows(session):
            campaign = (row.meta_data or {}).get("benchmark_campaign")
            if campaign:
                groups[campaign["id"]].append(row)
        return [summarize(rows) for rows in reversed(list(groups.values()))]


def get_campaign(campaign_id: str) -> dict | None:
    return next((c for c in list_campaigns() if c["id"] == campaign_id), None)


def pin_inputs(models: list[str], benchmarks: list[str]) -> tuple[dict, dict]:
    """Resolve only already-downloaded weights; pin dataset commits before queueing."""
    from pathlib import Path
    from huggingface_hub import HfApi
    from vivasecuris.aiasylum.api.model_catalog import _cache_roots, _cached_models
    from vivasecuris.aiasylum.benchmarks.datasets import BENCHMARK_DATASETS

    catalog = {m["model_ref"]: m for m in build_model_catalog()["models"]}
    cached = _cached_models(_cache_roots())
    resolved = {}
    for ref in models:
        model = catalog.get(ref)
        if not model or model.get("availability") != "ready":
            raise ValueError(f"Model is not ready on this server: {ref}. Download it in Find models first.")
        if model["kind"] == "custom":
            path = Path(ref)
            identity = hashlib.sha256(json.dumps({
                "manifest": model.get("manifest"),
                "files": [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(path.iterdir()) if p.is_file()],
            }, sort_keys=True).encode()).hexdigest()
            resolved[ref] = {"path": ref, "kind": "custom", "manifest": model.get("manifest"), "file_metadata_hash": identity}
        else:
            if ref not in cached:
                raise ValueError(f"Cached snapshot disappeared for {ref}; refresh the model library.")
            path = cached[ref][0]
            resolved[ref] = {"path": str(path), "kind": "base", "repo_id": ref, "revision": path.name}
    api = HfApi()
    revisions = {}
    for name in benchmarks:
        info = BENCHMARK_DATASETS[name]
        repo = info.get("path") or info.get("dataset") or info.get("dataset_name")
        if not repo:
            raise ValueError(f"Dataset identity is missing for {name}")
        revisions[name] = api.dataset_info(repo, timeout=20).sha
    return resolved, revisions


def create_campaign(data: dict, resolved: dict, revisions: dict) -> dict:
    config = {"id": uuid4().hex, **data}
    with get_session() as session:
        for model in data["models"]:
            for benchmark in data["benchmarks"]:
                session.add(TestRun(
                    doctor_provider="transformers", doctor_model=resolved[model]["path"],
                    patient_provider="transformers", patient_model=resolved[model]["path"],
                    test_type="benchmark", status="pending",
                    meta_data={"benchmark_campaign": config, "campaign_model": model,
                               "model_provenance": resolved[model], "benchmark": benchmark,
                               "num_samples": data["num_samples"],
                               "test_config": {"benchmark_name": benchmark, "test_mode": "one_shot",
                                               "num_samples": data["num_samples"], "seed": data["seed"],
                                               "max_new_tokens": data["max_new_tokens"],
                                               "dataset_revision": revisions[benchmark]}},
                ))
        session.commit()
    return get_campaign(config["id"])


def _fail(run_id: int, error: str, *, cancelled=False):
    with get_session() as session:
        row = session.get(TestRun, run_id)
        if row and row.status in ("pending", "running"):
            row.status = "failed"
            row.meta_data = {**(row.meta_data or {}), "error": error, "cancelled": cancelled}
            session.commit()


async def run_campaign(campaign_id: str):
    from vivasecuris.aiasylum.api.benchmark_runtime import execute_benchmark_job
    from vivasecuris.aiasylum.api.cancellation import cancellation_manager
    campaign = get_campaign(campaign_id)
    for run in campaign["runs"]:
        with get_session() as session:
            row = session.get(TestRun, run["id"])
            if not row or row.status != "pending":
                continue
        task = asyncio.create_task(execute_benchmark_job(run["id"]))
        cancellation_manager.register_task(run["id"], task)
        try:
            await task
        except asyncio.CancelledError:
            _fail(run["id"], "Benchmark comparison was cancelled.", cancelled=True)
            if asyncio.current_task().cancelling():
                raise
        except Exception as exc:
            logger.exception("Benchmark campaign run %s failed", run["id"])
            _fail(run["id"], str(exc))
        finally:
            cancellation_manager.unregister_task(run["id"])
        _fail(run["id"], "Benchmark worker exited without saving a completed result.")


def schedule(campaign_id: str):
    task = asyncio.create_task(run_campaign(campaign_id))
    _tasks[campaign_id] = task
    def done(finished):
        _tasks.pop(campaign_id, None)
        if not finished.cancelled() and finished.exception():
            logger.error("Campaign %s failed: %s", campaign_id, finished.exception())
    task.add_done_callback(done)


def cancel_campaign(campaign_id: str):
    from vivasecuris.aiasylum.api.cancellation import cancellation_manager
    campaign = get_campaign(campaign_id)
    if campaign is None:
        return None
    for run in campaign["runs"]:
        if run["status"] == "pending":
            _fail(run["id"], "Benchmark comparison was cancelled before this run started.", cancelled=True)
        if run["status"] in ("pending", "running"):
            cancellation_manager.cancel(run["id"])
    return get_campaign(campaign_id)


def recover_interrupted_campaigns():
    """A restarted server must never show a dead queue as still running."""
    with get_session() as session:
        for row in session.query(TestRun).filter(TestRun.status.in_(["pending", "running"])):
            if (row.meta_data or {}).get("benchmark_campaign"):
                row.status = "failed"
                row.meta_data = {**row.meta_data, "error": "Server restarted before this comparison finished. Start a new comparison to retry.", "interrupted": True}
        session.commit()
