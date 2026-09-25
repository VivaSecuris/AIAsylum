"""Create a derived comparison from saved answers without running any models.

Raw dataset rows are not stored in full, so their original sample hash cannot
be recomputed here. We verify that hash and the saved questions agree across
models, then separately hash the exact saved evidence used for rescoring.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import re
from uuid import uuid4

from sqlalchemy import text

from vivasecuris.aiasylum.api import benchmark_campaigns as campaigns
from vivasecuris.aiasylum.api.cancellation import cancellation_manager
from vivasecuris.aiasylum.benchmarks import simple as scoring
from vivasecuris.aiasylum.database import TestRun, TestResult, get_session


class RescoreConflict(ValueError):
    """Saved evidence is not complete enough to derive a valid comparison."""


def _hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def _conflict(message):
    raise RescoreConflict(message)


def _active(source_id: str, rows) -> bool:
    task = campaigns._tasks.get(source_id)
    return bool(task is not None and not task.done()) or any(
        cancellation_manager.has_active_task(row.id) for row in rows
    )


def _validate_answers(detail: dict, benchmark: str, run_id: int) -> str:
    provenance = detail["dataset_provenance"]
    sample_hash = provenance.get("ordered_sample_hash")
    if not isinstance(sample_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", sample_hash):
        _conflict(f"Run {run_id} has no valid saved sample hash")
    if not isinstance(detail.get("runtime"), dict) or not detail["runtime"]:
        _conflict(f"Run {run_id} is missing generation runtime provenance")
    sample_indices = provenance.get("sample_indices")
    if not isinstance(sample_indices, list) or len(sample_indices) != len(detail["results"]):
        _conflict(f"Run {run_id} is missing ordered dataset indices")
    evidence = []
    for i, answer in enumerate(detail["results"]):
        required = ("question", "response", "ground_truth", "choices", "sample_id", "dataset_index", "truncated")
        missing = [key for key in required if key not in answer]
        if missing:
            _conflict(f"Run {run_id}, answer {i + 1} is missing saved fields: {', '.join(missing)}")
        if (not isinstance(answer["question"], str) or not answer["question"].strip()
                or not isinstance(answer["response"], str)
                or type(answer["truncated"]) is not bool
                or type(answer["dataset_index"]) is not int
                or answer["dataset_index"] < 0
                or answer["dataset_index"] != sample_indices[i]):
            _conflict(f"Run {run_id}, answer {i + 1} has invalid saved fields")
        choices, gold = answer["choices"], answer["ground_truth"]
        if not isinstance(choices, list):
            _conflict(f"Run {run_id}, answer {i + 1} has no saved choice list")
        if benchmark == "gsm8k":
            if choices or not isinstance(gold, (str, int, float)) or isinstance(gold, bool) or scoring.numeric_answer(str(gold)) is None:
                _conflict(f"Run {run_id}, answer {i + 1} has an invalid saved numerical answer")
        else:
            if not 2 <= len(choices) <= 26 or any(not isinstance(c, str) or not c.strip() for c in choices):
                _conflict(f"Run {run_id}, answer {i + 1} is missing valid saved choices; they cannot be reconstructed")
            if isinstance(gold, str) and len(gold.strip()) == 1 and gold.strip().isalpha():
                index = ord(gold.strip().upper()) - ord("A")
            elif type(gold) is int or isinstance(gold, str) and gold.isdigit():
                index = int(gold)
            else:
                _conflict(f"Run {run_id}, answer {i + 1} has an invalid saved choice answer")
            if not 0 <= index < len(choices):
                _conflict(f"Run {run_id}, answer {i + 1} has an out-of-range saved choice answer")
        evidence.append({key: deepcopy(answer[key]) for key in (
            "question", "ground_truth", "choices", "sample_id", "dataset_index"
        )})
    return _hash(evidence)


def _validate_source(source_id: str, rows):
    if not rows:
        raise KeyError(source_id)
    config = rows[0].meta_data["benchmark_campaign"]
    if _active(source_id, rows) or any(row.status != "completed" for row in rows):
        _conflict("Wait until every comparison run has completed and its workers have stopped")
    if any((row.meta_data or {}).get("benchmark_campaign") != config for row in rows):
        _conflict("Source comparison configuration differs between runs")
    try:
        expected = Counter((model, benchmark) for model in config["models"] for benchmark in config["benchmarks"])
        actual = Counter((row.meta_data.get("campaign_model", row.patient_model), row.meta_data.get("benchmark")) for row in rows)
        if not expected or expected != actual or any(count != 1 for count in expected.values()):
            _conflict("Source comparison has missing or duplicate model/benchmark pairs")
        if any(benchmark not in campaigns.SUPPORTED for benchmark in config["benchmarks"]):
            _conflict("This comparison contains a benchmark that cannot be rescored from saved answers")
        requested = config["num_samples"]
        if type(requested) is not int or requested < 1:
            _conflict("Source comparison has an invalid requested sample count")
    except (KeyError, TypeError) as exc:
        raise RescoreConflict("Source comparison configuration is incomplete") from exc
    selected = []
    signatures = defaultdict(set)
    for row in rows:
        result = max(row.results, key=lambda r: r.id) if row.results else None
        detail = result.meta_data if result else None
        if not isinstance(detail, dict):
            _conflict(f"Run {row.id} has no saved benchmark result")
        provenance = detail.get("dataset_provenance")
        error = campaigns._result_error(result, detail, provenance, requested)
        if error:
            _conflict(f"Run {row.id}: {error}")
        old_scoring = detail.get("scoring")
        if not isinstance(old_scoring, dict) or not isinstance(old_scoring.get("version"), str) or not old_scoring["version"]:
            _conflict(f"Run {row.id} is missing its original scoring protocol")
        # Only the scorer version may change; every other recorded protocol
        # field must already match the source campaign before it is rescored.
        protocol_detail = {**detail, "scoring": {**old_scoring, "version": scoring.SCORING_VERSION}}
        benchmark = row.meta_data["benchmark"]
        try:
            valid_protocol = campaigns._protocol_verified(row.meta_data, protocol_detail, config, benchmark)
        except (KeyError, TypeError, AttributeError):
            valid_protocol = False
        if not valid_protocol:
            _conflict(f"Run {row.id} has incomplete or inconsistent dataset/generation provenance")
        evidence_hash = _validate_answers(detail, benchmark, row.id)
        signatures[benchmark].add(_hash({
            "provenance": provenance, "generation": detail["generation"],
            "saved_evidence_hash": evidence_hash,
        }))
        selected.append((row, result, detail, evidence_hash))
    if any(len(values) != 1 for values in signatures.values()):
        _conflict("Source models did not receive the same saved questions, gold answers, sample hashes and generation settings")
    return config, selected


def _rescore_detail(detail: dict, benchmark: str, provenance: dict) -> dict:
    output = deepcopy(detail)
    for answer in output["results"]:
        response, choices = answer["response"], answer["choices"]
        answer["correct"] = scoring.exact_response_matches(
            response, answer["ground_truth"], choices, numeric=benchmark == "gsm8k"
        )
        answer["answer_valid"] = (scoring.choice_answer(response, choices) is not None if choices
                                  else scoring.numeric_answer(response) is not None)
        # A token-limit flag and final-answer validity are separate measurements.
        # Keep the recorded truncated flag exactly as generated.
    output["correct"] = sum(answer["correct"] for answer in output["results"])
    output["accuracy"] = output["correct"] / output["num_samples"]
    output["invalid_answer_count"] = sum(not answer["answer_valid"] for answer in output["results"])
    output["truncated_count"] = sum(answer["truncated"] for answer in output["results"])
    output["scoring"] = {**output["scoring"], "version": scoring.SCORING_VERSION}
    output["rescoring"] = deepcopy(provenance)
    return output


def rescore_campaign(source_campaign_id: str) -> dict:
    """Atomically derive a completed campaign, or return its existing equivalent.

    No network, model-provider, model-cache or GPU operation occurs here. SQLite
    acquires its write reservation before reading, so concurrent API/CLI clicks
    cannot both create a derivative from the same source results and scorer.
    """
    with get_session() as session:
        if session.get_bind().dialect.name == "sqlite":
            session.execute(text("BEGIN IMMEDIATE"))
        else:
            # Lock existing campaign rows before taking a fresh evidence snapshot.
            session.query(TestRun.id).filter(TestRun.test_type == "benchmark").with_for_update().all()
        all_rows = campaigns._rows(session)
        rows = [row for row in all_rows if (row.meta_data or {}).get("benchmark_campaign", {}).get("id") == source_campaign_id]
        config, selected = _validate_source(source_campaign_id, rows)
        key = _hash({"source_campaign_id": source_campaign_id, "scoring_version": scoring.SCORING_VERSION,
                     "source_results": [(row.id, result.id) for row, result, _, _ in selected]})
        source_hash = _hash([{"run_id": row.id, "result_id": result.id,
                              "run_metadata": row.meta_data, "result_metadata": detail,
                              "score": result.score, "scores": result.scores,
                              "input_prompt": result.input_prompt, "output_response": result.output_response}
                             for row, result, detail, _ in selected])
        existing = [row for row in all_rows if (row.meta_data or {}).get("benchmark_campaign", {}).get("rescoring_key") == key]
        if existing:
            saved = existing[0].meta_data["benchmark_campaign"]
            if saved.get("rescoring_source_hash") != source_hash:
                _conflict("Source answers changed after their previous rescoring; review the stored evidence")
            summary = campaigns.summarize(existing)
            if summary["status"] != "completed" or not summary["comparable"]:
                _conflict("The existing rescored comparison is incomplete; review it before creating another")
            return summary
        suffix = " · rescored"
        name = str(config.get("name") or "Model comparison")
        derived = {**deepcopy(config), "id": uuid4().hex, "name": name[:120 - len(suffix)] + suffix,
                   "source_campaign_id": source_campaign_id, "rescoring_version": scoring.SCORING_VERSION,
                   "rescoring_key": key, "rescoring_source_hash": source_hash}
        now = datetime.now(timezone.utc).isoformat()
        new_rows = []
        for source, result, detail, evidence_hash in selected:
            proof = {"source_campaign_id": source_campaign_id,
                     "source_test_run_id": source.id, "source_result_id": result.id,
                     "source_scoring": deepcopy(detail["scoring"]), "scored_at": now,
                     "version": scoring.SCORING_VERSION, "saved_evidence_hash": evidence_hash,
                     "operation": "saved_answers_cpu_only"}
            if source.meta_data.get("lineage_id"):
                proof["source_lineage_id"] = source.meta_data["lineage_id"]
            rescored = _rescore_detail(detail, source.meta_data["benchmark"], proof)
            metadata = {**deepcopy(source.meta_data), "benchmark_campaign": deepcopy(derived),
                        "rescoring": proof, "lineage_id": uuid4().hex}
            # This step descends from the saved source run, not from its parent.
            # Keep the original test_config intact as generation evidence; the
            # graph uses rescoring provenance rather than its old fork fields.
            metadata.pop("lineage_parent", None)
            row = TestRun(doctor_provider=source.doctor_provider, doctor_model=source.doctor_model,
                          patient_provider=source.patient_provider, patient_model=source.patient_model,
                          test_type="benchmark", status="completed", meta_data=metadata)
            session.add(row)
            session.flush()
            scores = {**deepcopy(result.scores or {}), "accuracy": rescored["accuracy"],
                      "correct": rescored["correct"], "total": rescored["num_samples"]}
            summary = f"Rescored saved answers: {rescored['accuracy']:.2%} ({rescored['correct']}/{rescored['num_samples']})"
            summary += f"\nSource test run: {source.id}; source result: {result.id}; scorer: {scoring.SCORING_VERSION}. No model generation was performed."
            row.results.append(TestResult(test_name=result.test_name, test_category=result.test_category,
                input_prompt=result.input_prompt, output_response=summary, score=rescored["accuracy"],
                scores=scores, analysis=None, flags=deepcopy(result.flags), meta_data=rescored))
            new_rows.append(row)
        session.flush()
        response = campaigns.summarize(new_rows)
        if not response["comparable"] or response["status"] != "completed":
            _conflict("Derived scoring did not produce a complete paired comparison")
        session.commit()
        return response
