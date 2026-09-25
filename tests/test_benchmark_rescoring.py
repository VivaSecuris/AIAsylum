"""Rescoring saved evidence is atomic, repeatable and never regenerates answers."""

from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from vivasecuris.aiasylum.api import benchmark_campaigns as campaigns
from vivasecuris.aiasylum.api import benchmark_rescoring as rescoring
from vivasecuris.aiasylum.api.cancellation import cancellation_manager
from vivasecuris.aiasylum.benchmarks import simple as scoring
from vivasecuris.aiasylum.database import TestRun, TestResult, get_session


@pytest.fixture(autouse=True)
def no_workers(monkeypatch):
    monkeypatch.setattr(campaigns, "_tasks", {})
    monkeypatch.setattr(cancellation_manager, "_tasks", {})


def source_campaign():
    data = {"name": "A" * 120, "models": ["base", "custom"], "benchmarks": ["mmlu", "gsm8k"],
            "num_samples": 2, "seed": 7, "max_new_tokens": 256}
    c = campaigns.create_campaign(data, {m: {"path": m, "revision": "model-revision"} for m in data["models"]},
                                  {"mmlu": "a" * 40, "gsm8k": "b" * 40})
    with get_session() as session:
        for entry in c["runs"]:
            row = session.get(TestRun, entry["id"])
            benchmark = row.meta_data["benchmark"]
            numeric = benchmark == "gsm8k"
            answers = [{"question": "Question\nFinal answer:", "response": "Final answer: 4" if numeric else "A",
                        "ground_truth": "steps\n#### 4" if numeric else "A", "choices": [] if numeric else ["four", "five"],
                        "sample_id": f"{benchmark}:{i}", "dataset_index": i, "question_number": i + 1,
                        "correct": False, "answer_valid": False, "truncated": i == 1,
                        "finish_reason": "length" if i == 1 else "stop", "usage": {"completion_tokens": 5}}
                       for i in range(2)]
            detail = {"num_samples": 2, "correct": 0, "accuracy": 0.0, "results": answers,
                "benchmark_name": benchmark, "test_mode": "one_shot", "invalid_answer_count": 2, "truncated_count": 1,
                "runtime": {"transformers": "5.17.0", "device": "cuda:0", "dtype": "bfloat16"},
                "dataset_provenance": {"dataset": "openai/gsm8k" if numeric else "cais/mmlu", "config": "main" if numeric else "all",
                    "split": "test", "resolved_revision": ("b" if numeric else "a") * 40,
                    "ordered_sample_hash": ("b" if numeric else "a") * 64,
                    "requested_count": 2, "actual_count": 2, "available_count": 100,
                    "sample_ids": [a["sample_id"] for a in answers], "sample_indices": [0, 1]},
                "generation": {"seed": 7, "max_new_tokens": 256, "temperature": 0.0,
                    "independent_questions": True, "prompt_protocol": "zero-shot-direct-answer-v1", "thinking": False},
                "scoring": {"version": "older-scorer", "method": "numeric_exact_match" if numeric else "mcq_final_answer"}}
            row.status = "completed"
            row.results.append(TestResult(test_name=f"benchmark_{benchmark}", test_category="benchmark",
                input_prompt="Original exam", output_response="Original score: 0", score=0.0,
                scores={"accuracy": 0.0, "correct": 0, "total": 2}, meta_data=detail))
        session.commit()
    return campaigns.get_campaign(c["id"])


def snapshot():
    with get_session() as session:
        return {row.id: {"meta": deepcopy(row.meta_data), "status": row.status,
                       "results": [{"id": r.id, "meta": deepcopy(r.meta_data), "score": r.score,
                                    "scores": deepcopy(r.scores), "output": r.output_response} for r in row.results]}
                for row in session.query(TestRun).all()}


def mutate(run_id, fn):
    with get_session() as session:
        row = session.get(TestRun, run_id)
        detail = deepcopy(row.results[0].meta_data)
        fn(detail)
        row.results[0].meta_data = detail
        session.commit()


def test_rescore_preserves_originals_and_uses_current_scoring_without_generation(monkeypatch):
    source = source_campaign()
    before = snapshot()
    monkeypatch.setattr(campaigns, "pin_inputs", lambda *a, **kw: pytest.fail("No dataset or model refetch"))
    monkeypatch.setattr(campaigns, "schedule", lambda *a, **kw: pytest.fail("No model execution"))
    result = rescoring.rescore_campaign(source["id"])
    assert result["id"] != source["id"]
    assert result["source_campaign_id"] == source["id"]
    assert len(result["name"]) == 120 and result["name"].endswith(" · rescored")
    assert result["status"] == "completed" and result["comparable"]
    assert all(run["score"] == 1.0 for run in result["runs"])
    after = snapshot()
    assert {k: after[k] for k in before} == before
    with get_session() as session:
        for run in result["runs"]:
            row = session.get(TestRun, run["id"])
            proof = row.meta_data["rescoring"]
            original = before[proof["source_test_run_id"]]["results"][0]
            detail = row.results[0].meta_data
            assert proof["source_result_id"] == original["id"]
            assert proof["source_scoring"]["version"] == "older-scorer"
            assert proof["version"] == scoring.SCORING_VERSION
            assert len(proof["saved_evidence_hash"]) == 64
            assert detail["scoring"]["version"] == scoring.SCORING_VERSION
            for key in ("runtime", "generation", "dataset_provenance"):
                assert detail[key] == original["meta"][key]
            assert detail["truncated_count"] == 1 and detail["invalid_answer_count"] == 0
            for new, old in zip(detail["results"], original["meta"]["results"]):
                assert {k: v for k, v in new.items() if k not in {"correct", "answer_valid"}} == {
                    k: v for k, v in old.items() if k not in {"correct", "answer_valid"}}
                assert new["correct"] and new["answer_valid"]


def test_repeat_and_concurrent_clicks_return_one_derivative():
    source = source_campaign()
    with ThreadPoolExecutor(max_workers=2) as pool:
        values = list(pool.map(rescoring.rescore_campaign, [source["id"], source["id"]]))
    assert values[0]["id"] == values[1]["id"]
    assert rescoring.rescore_campaign(source["id"])["id"] == values[0]["id"]
    assert len(snapshot()) == 8


def test_rescored_runs_have_distinct_lineage_and_preserve_source_identity():
    source = source_campaign()
    source_lineages = {}
    with get_session() as session:
        for run in source["runs"]:
            row = session.get(TestRun, run["id"])
            identity = f"{row.id:032x}"
            source_lineages[row.id] = identity
            metadata = deepcopy(row.meta_data)
            metadata.update(lineage_id=identity, lineage_parent={"kind": "test", "id": 999})
            metadata["test_config"]["lineage_parent"] = {"kind": "test", "id": 999}
            row.meta_data = metadata
        session.commit()
    original = snapshot()
    derived = rescoring.rescore_campaign(source["id"])
    after = snapshot()
    assert {key: after[key] for key in original} == original
    identities = set()
    for run in derived["runs"]:
        metadata = after[run["id"]]["meta"]
        proof = metadata["rescoring"]
        identities.add(metadata["lineage_id"])
        assert len(metadata["lineage_id"]) == 32
        assert metadata["lineage_id"] not in source_lineages.values()
        assert proof["source_lineage_id"] == source_lineages[proof["source_test_run_id"]]
        assert "lineage_parent" not in metadata
        assert metadata["test_config"] == original[proof["source_test_run_id"]]["meta"]["test_config"]
        assert after[run["id"]]["results"][0]["meta"]["rescoring"] == proof
    assert len(identities) == len(derived["runs"])
    assert rescoring.rescore_campaign(source["id"])["id"] == derived["id"]
    assert snapshot() == after


@pytest.mark.parametrize("mutation", [
    lambda detail: detail["results"][0].pop("choices"),
    lambda detail: detail["results"][0].pop("response"),
    lambda detail: detail["results"][0].pop("ground_truth"),
    lambda detail: detail["results"][0].update(question="different question"),
    lambda detail: detail["results"][0].update(choices=["different", "choices"]),
    lambda detail: detail["dataset_provenance"].update(ordered_sample_hash="c" * 64),
    lambda detail: detail["dataset_provenance"].update(ordered_sample_hash="not-a-hash"),
    lambda detail: detail["generation"].update(seed=99),
    lambda detail: detail.pop("runtime"),
])
def test_missing_or_mismatched_evidence_blocks_without_writes(mutation):
    source = source_campaign()
    mutate(source["runs"][0]["id"], mutation)
    before = snapshot()
    with pytest.raises(rescoring.RescoreConflict):
        rescoring.rescore_campaign(source["id"])
    assert snapshot() == before


@pytest.mark.parametrize("problem", ["running", "failed", "missing", "duplicate", "active_worker"])
def test_incomplete_source_or_active_worker_cannot_rescore(problem, monkeypatch):
    source = source_campaign()
    with get_session() as session:
        row = session.get(TestRun, source["runs"][0]["id"])
        if problem in {"running", "failed"}:
            row.status = problem
        elif problem == "missing":
            session.delete(row)
        elif problem == "duplicate":
            row.meta_data = {**row.meta_data, "campaign_model": "custom"}
        elif problem == "active_worker":
            monkeypatch.setattr(cancellation_manager, "has_active_task", lambda _: True)
        session.commit()
    before = snapshot()
    with pytest.raises(rescoring.RescoreConflict):
        rescoring.rescore_campaign(source["id"])
    assert snapshot() == before


def test_failure_mid_derivation_rolls_back_every_new_row(monkeypatch):
    source = source_campaign()
    before = snapshot()
    real = rescoring._rescore_detail
    calls = []
    def fail_second(*args):
        calls.append(True)
        if len(calls) == 2:
            raise RuntimeError("injected scoring failure")
        return real(*args)
    monkeypatch.setattr(rescoring, "_rescore_detail", fail_second)
    with pytest.raises(RuntimeError, match="injected"):
        rescoring.rescore_campaign(source["id"])
    assert snapshot() == before


def test_source_evidence_change_after_rescore_is_not_silently_reused():
    source = source_campaign()
    rescoring.rescore_campaign(source["id"])
    mutate(source["runs"][0]["id"], lambda detail: detail["results"][0].update(response="B"))
    with pytest.raises(rescoring.RescoreConflict, match="changed"):
        rescoring.rescore_campaign(source["id"])
    assert len(snapshot()) == 8


def test_api_returns_derived_comparison_or_actionable_409():
    from vivasecuris.aiasylum.api.main import app
    client = TestClient(app)
    assert client.post("/api/v1/benchmark-campaigns/missing/rescore").status_code == 404
    source = source_campaign()
    response = client.post(f"/api/v1/benchmark-campaigns/{source['id']}/rescore")
    assert response.status_code == 201
    assert response.json()["source_campaign_id"] == source["id"]
    mutate(source["runs"][0]["id"], lambda detail: detail["results"][0].pop("choices"))
    response = client.post(f"/api/v1/benchmark-campaigns/{source['id']}/rescore")
    assert response.status_code == 409 and "choices" in response.json()["detail"]
