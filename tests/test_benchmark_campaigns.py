"""Paired comparison persistence, provenance guards and queue lifecycle."""
import asyncio
import pytest
from fastapi.testclient import TestClient
from vivasecuris.aiasylum.api import benchmark_campaigns as campaigns
from vivasecuris.aiasylum.database import TestRun, TestResult, get_session


@pytest.fixture(autouse=True)
def isolate_cancellation(monkeypatch):
    from vivasecuris.aiasylum.api.cancellation import cancellation_manager
    monkeypatch.setattr(cancellation_manager, "_cancelled", set())
    monkeypatch.setattr(cancellation_manager, "_tasks", {})
    monkeypatch.setattr(campaigns, "_tasks", {})


def create(models=None):
    models = models or ["base", "edit"]
    return campaigns.create_campaign(
        {"name": "Comparison", "models": models, "benchmarks": ["mmlu"],
         "num_samples": 25, "seed": 7, "max_new_tokens": 256},
        {m: {"path": m, "revision": "pinned"} for m in models}, {"mmlu": "dataset-commit"})


def complete(run_id, sample_hash="same", score=0.0):
    with get_session() as session:
        row = session.get(TestRun, run_id)
        row.status = "completed"
        session.add(TestResult(test_run_id=run_id, test_name="benchmark_mmlu", input_prompt="exam",
                               output_response="answers", score=score,
                               meta_data={"correct": int(score * 25), "num_samples": 25,
                                          "accuracy": score,
                                          "results": [{"sample_id": f"sample-{i}", "correct": i < int(score * 25)} for i in range(25)],
                                          "dataset_provenance": {"ordered_sample_hash": sample_hash,
                                              "dataset": "cais/mmlu", "config": "all", "split": "test",
                                              "requested_count": 25, "actual_count": 25, "available_count": 100,
                                              "sample_ids": [f"sample-{i}" for i in range(25)],
                                              "resolved_revision": "dataset-commit"},
                                          "generation": {"seed": 7, "max_new_tokens": 256, "temperature": 0.0,
                                              "independent_questions": True, "prompt_protocol": "zero-shot-direct-answer-v1"},
                                          "scoring": {"version": "final-answer-v4", "method": "mcq_final_answer"}}))
        config = row.meta_data["benchmark_campaign"]
        if config.get("protocol_version", 1) >= 2:
            session.flush()
            record = session.query(TestResult).filter_by(test_run_id=run_id).one()
            detail = dict(record.meta_data)
            system = config.get("patient_system_prompt") or ("Answer the question accurately. Follow the requested answer format." if config.get("patient_prompt_framing", True) else None)
            enabled = config.get("enable_cot", False)
            detail["generation"] = {**detail["generation"], "enable_cot": enabled,
                "system_prompt": system, "temperature": config.get("temperature", 0), "top_p": config.get("top_p", 0.9),
                "prompt_protocol": "zero-shot-react-v1" if enabled else "zero-shot-direct-answer-v1",
                "requests": [{"request_system_prompts": [system] if system else [], "cot_enabled": enabled} for _ in range(25)]}
            record.meta_data = detail
        session.commit()


def test_comparison_pins_all_inputs_and_preserves_zero_scores():
    c = create()
    with get_session() as session:
        rows = session.query(TestRun).all()
        assert {r.meta_data["test_config"]["dataset_revision"] for r in rows} == {"dataset-commit"}
        assert {r.meta_data["test_config"]["seed"] for r in rows} == {7}
    for run in c["runs"]:
        complete(run["id"])
    result = campaigns.get_campaign(c["id"])
    assert result["status"] == "completed" and result["comparable"]
    assert [r["score"] for r in result["runs"]] == [0.0, 0.0]


def test_mismatched_question_sets_are_not_comparable():
    c = create()
    for index, run in enumerate(c["runs"]):
        complete(run["id"], str(index))
    result = campaigns.get_campaign(c["id"])
    assert result["status"] == "completed" and not result["comparable"]
    assert any("provenance" in warning for warning in result["warnings"])


def test_removed_run_cannot_produce_complete_comparison():
    c = create()
    complete(c["runs"][0]["id"])
    with get_session() as session:
        session.delete(session.get(TestRun, c["runs"][1]["id"]))
        session.commit()
    result = campaigns.get_campaign(c["id"])
    assert result["status"] == "partial" and not result["comparable"]
    assert result["total"] == 2


def test_restart_marks_only_campaign_jobs_interrupted():
    c = create()
    with get_session() as session:
        other = TestRun(doctor_provider="x", doctor_model="x", patient_provider="x", patient_model="x", test_type="one_shot", status="pending")
        session.add(other)
        session.commit()
        other_id = other.id
    campaigns.recover_interrupted_campaigns()
    assert campaigns.get_campaign(c["id"])["status"] == "failed"
    with get_session() as session:
        assert session.get(TestRun, other_id).status == "pending"


def test_cancel_preserves_finished_results_and_cancels_queue():
    c = create()
    complete(c["runs"][0]["id"], score=0.8)
    result = campaigns.cancel_campaign(c["id"])
    assert result["status"] == "partial"
    assert result["runs"][0]["score"] == 0.8
    assert result["runs"][1]["status"] == "failed"


@pytest.mark.asyncio
async def test_worker_runs_pairs_in_order_and_continues_after_failure(monkeypatch):
    import sys
    from types import SimpleNamespace
    c = create(["first", "broken", "last"])
    calls = []
    active = 0
    async def execute(run_id):
        nonlocal active
        active += 1
        assert active == 1
        calls.append(run_id)
        await asyncio.sleep(0)
        active -= 1
        if run_id == c["runs"][1]["id"]:
            raise RuntimeError("loader failed")
        complete(run_id)
    monkeypatch.setitem(sys.modules, "vivasecuris.aiasylum.api.benchmark_runtime", SimpleNamespace(execute_benchmark_job=execute))
    await campaigns.run_campaign(c["id"])
    assert calls == [r["id"] for r in c["runs"]]
    result = campaigns.get_campaign(c["id"])
    assert result["completed"] == 2 and result["status"] == "partial"
    assert result["runs"][1]["error"] == "loader failed"


def test_api_rejects_bad_or_unready_selection_without_creating_jobs(monkeypatch):
    from vivasecuris.aiasylum.api.main import app
    client = TestClient(app)
    body = {"models": ["missing"], "benchmarks": ["mmlu"]}
    assert client.post("/api/v1/benchmark-campaigns", json={**body, "num_samples": 0}).status_code == 422
    assert client.post("/api/v1/benchmark-campaigns", json={**body, "models": ["x", "x"]}).status_code == 422
    monkeypatch.setattr(campaigns, "build_model_catalog", lambda: {"models": []})
    result = client.post("/api/v1/benchmark-campaigns", json=body)
    assert result.status_code == 400
    assert campaigns.list_campaigns() == []



def mutate_result(run_id, change):
    with get_session() as session:
        result = session.query(TestResult).filter(TestResult.test_run_id == run_id).first()
        metadata = dict(result.meta_data)
        change(metadata)
        result.meta_data = metadata
        session.commit()


@pytest.mark.parametrize("missing", ["generation", "scoring", "dataset_provenance"])
def test_missing_protocol_proof_prevents_comparability(missing):
    c = create()
    for run in c["runs"]:
        complete(run["id"])
        mutate_result(run["id"], lambda meta: meta.pop(missing))
    assert not campaigns.get_campaign(c["id"])["comparable"]


@pytest.mark.parametrize("field,value", [("num_samples", 24), ("correct", 26), ("accuracy", 0.7)])
def test_inconsistent_scoring_counts_hide_invalid_scores(field, value):
    c = create()
    for run in c["runs"]:
        complete(run["id"])
    mutate_result(c["runs"][1]["id"], lambda meta: meta.update({field: value}))
    result = campaigns.get_campaign(c["id"])
    assert not result["comparable"]
    assert result["runs"][1]["score"] is None
    assert result["runs"][1]["error"]
    # Validation must not destroy the stored evidence.
    with get_session() as session:
        assert session.get(TestRun, c["runs"][1]["id"]).results[0].score == 0.0


def test_completed_worker_without_result_is_not_a_completed_comparison():
    c = create()
    for run in c["runs"]:
        with get_session() as session:
            session.get(TestRun, run["id"]).status = "completed"
            session.commit()
    result = campaigns.get_campaign(c["id"])
    assert result["status"] != "completed" and result["completed"] == 0
    assert not result["comparable"]
    assert all(run["error"] for run in result["runs"])


def test_pair_coverage_is_checked_not_only_row_count():
    c = create()
    for run in c["runs"]:
        complete(run["id"])
    with get_session() as session:
        row = session.get(TestRun, c["runs"][1]["id"])
        row.meta_data = {**row.meta_data, "campaign_model": "base"}
        session.commit()
    result = campaigns.get_campaign(c["id"])
    assert not result["comparable"] and result["status"] != "completed"
    assert any("pair" in warning.lower() for warning in result["warnings"])


def test_fewer_available_samples_reports_actual_denominator():
    c = create()
    for run in c["runs"]:
        complete(run["id"])
        def reduce(meta):
            meta["num_samples"] = 2
            meta["results"] = meta["results"][:2]
            meta["dataset_provenance"] = {**meta["dataset_provenance"], "actual_count": 2,
                "sample_ids": ["sample-0", "sample-1"], "available_count": 2}
        mutate_result(run["id"], reduce)
    result = campaigns.get_campaign(c["id"])
    assert result["comparable"]
    assert all(run["num_samples"] == 2 and run["progress"]["total"] == 2 for run in result["runs"])
    assert any("fewer" in warning.lower() for warning in result["warnings"])


def test_matching_but_unrequested_generation_protocol_is_not_comparable():
    c = create()
    for run in c["runs"]:
        complete(run["id"])
        def mutate(meta):
            meta["generation"] = {**meta["generation"], "seed": 99, "temperature": 0.7}
        mutate_result(run["id"], mutate)
    assert not campaigns.get_campaign(c["id"])["comparable"]


@pytest.mark.parametrize("mismatch", [None, "enable_cot", "temperature", "top_p"])
def test_campaign_cot_and_sampling_are_persisted_and_verified(mismatch):
    settings = {"enable_cot": True, "temperature": 0.2, "top_p": 0.7}
    c = campaigns.create_campaign(
        {"name": "Reasoned comparison", "models": ["base", "edit"], "benchmarks": ["mmlu"],
         "num_samples": 25, "seed": 7, "max_new_tokens": 256, **settings},
        {m: {"path": m, "revision": "pinned"} for m in ("base", "edit")}, {"mmlu": "dataset-commit"})
    with get_session() as session:
        for row in session.query(TestRun).all():
            assert all(row.meta_data["test_config"][key] == value for key, value in settings.items())
    for run in c["runs"]:
        complete(run["id"])
        def mutate(meta):
            meta["generation"] = {**meta["generation"], **settings, "prompt_protocol": "zero-shot-react-v1"}
            if mismatch:
                meta["generation"][mismatch] = False if mismatch == "enable_cot" else 0.9
        mutate_result(run["id"], mutate)
    assert campaigns.get_campaign(c["id"])["comparable"] is (mismatch is None)


@pytest.mark.parametrize("target", ["doctor", "patient"])
def test_campaign_freezes_selected_patient_prompt_before_queue(target):
    from vivasecuris.aiasylum.database import PromptLibrary
    with get_session() as session:
        prompt = PromptLibrary(name="Selected persona", prompt_type="system_prompt", target=target,
                               prompt_text="Exact original persona.")
        session.add(prompt)
        session.commit()
        prompt_id = prompt.id
    data = {"name": "Prompt comparison", "models": ["base", "edit"], "benchmarks": ["mmlu"],
            "num_samples": 25, "seed": 7, "max_new_tokens": 256, "patient_system_prompt_id": prompt_id}
    if target == "doctor":
        with pytest.raises(ValueError, match="another role"):
            campaigns.create_campaign(data, {}, {})
        assert campaigns.list_campaigns() == []
        return
    c = campaigns.create_campaign(data, {m: {"path": m} for m in data["models"]}, {"mmlu": "dataset-commit"})
    with get_session() as session:
        prompt = session.get(PromptLibrary, prompt_id)
        prompt.prompt_text = "Edited later."
        session.commit()
        for row in session.query(TestRun).all():
            cfg = row.meta_data["test_config"]
            assert cfg["patient_system_prompt"] == "Exact original persona."
            assert "patient_system_prompt_id" not in cfg
    for run in c["runs"]:
        complete(run["id"])
        mutate_result(run["id"], lambda meta: meta.update(generation={**meta["generation"], "system_prompt": "Exact original persona."}))
    assert campaigns.get_campaign(c["id"])["comparable"]
    mutate_result(c["runs"][-1]["id"], lambda meta: meta.update(generation={**meta["generation"], "system_prompt": "Edited later."}))
    assert not campaigns.get_campaign(c["id"])["comparable"]


@pytest.mark.parametrize("missing", ["requests", "prompt_protocol"])
def test_new_campaigns_require_prompt_protocol_and_per_question_evidence(missing):
    c = campaigns.create_campaign(
        {"name": "Verified protocol", "models": ["base"], "benchmarks": ["mmlu"],
         "num_samples": 25, "seed": 7, "max_new_tokens": 256, "enable_cot": True},
        {"base": {"path": "base"}}, {"mmlu": "dataset-commit"})
    complete(c["runs"][0]["id"])
    assert campaigns.get_campaign(c["id"])["comparable"]
    def remove(meta):
        generation = dict(meta["generation"])
        generation.pop(missing)
        meta["generation"] = generation
    mutate_result(c["runs"][0]["id"], remove)
    assert not campaigns.get_campaign(c["id"])["comparable"]


def test_changed_checkpoint_is_rejected_even_after_worker_completed(tmp_path):
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"first")
    fingerprint = campaigns._checkpoint_metadata_hash(tmp_path, None)
    c = campaigns.create_campaign(
        {"name": "Pinned weights", "models": ["custom"], "benchmarks": ["mmlu"],
         "num_samples": 25, "seed": 7, "max_new_tokens": 256},
        {"custom": {"path": str(tmp_path), "kind": "custom", "file_metadata_hash": fingerprint}},
        {"mmlu": "dataset-commit"})
    run_id = c["runs"][0]["id"]
    assert campaigns._verify_checkpoint(run_id)
    complete(run_id)
    weights.write_bytes(b"different weights")
    assert not campaigns._verify_checkpoint(run_id)
    result = campaigns.get_campaign(c["id"])
    assert result["status"] == "failed" and not result["comparable"]
    assert "Checkpoint files changed" in result["runs"][0]["error"]



@pytest.mark.asyncio
async def test_cancel_running_campaign_drains_active_job_and_skips_remaining(monkeypatch):
    import sys
    from types import SimpleNamespace
    c = create(["done", "active", "queued"])
    complete(c["runs"][0]["id"], score=0.8)
    started = asyncio.Event()
    drained = asyncio.Event()
    calls = []
    async def execute(run_id):
        calls.append(run_id)
        with get_session() as session:
            session.get(TestRun, run_id).status = "running"
            session.commit()
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await asyncio.sleep(0)
            drained.set()
            raise
    monkeypatch.setitem(sys.modules, "vivasecuris.aiasylum.api.benchmark_runtime", SimpleNamespace(execute_benchmark_job=execute))
    worker = asyncio.create_task(campaigns.run_campaign(c["id"]))
    await asyncio.wait_for(started.wait(), 3)
    campaigns.cancel_campaign(c["id"])
    await asyncio.wait_for(worker, 3)
    assert drained.is_set()
    assert calls == [c["runs"][1]["id"]]
    result = campaigns.get_campaign(c["id"])
    assert [run["status"] for run in result["runs"]] == ["completed", "failed", "failed"]
    assert result["runs"][0]["score"] == 0.8



def test_partial_scoring_cannot_masquerade_as_smaller_dataset():
    c = create()
    for run in c["runs"]:
        complete(run["id"])
        def truncate(meta):
            meta["num_samples"] = 2
            meta["results"] = meta["results"][:2]
            meta["dataset_provenance"] = {**meta["dataset_provenance"], "actual_count": 2,
                "sample_ids": ["sample-0", "sample-1"]}
        mutate_result(run["id"], truncate)
    result = campaigns.get_campaign(c["id"])
    assert not result["comparable"]
    assert all(run["score"] is None for run in result["runs"])


def test_old_scoring_results_stay_saved_but_cannot_be_verified():
    c = create()
    for run in c['runs']:
        complete(run['id'], score=0.8)
        mutate_result(run['id'], lambda meta: meta.update(scoring={**meta['scoring'], 'version': 'final-answer-v2'}))
    result = campaigns.get_campaign(c['id'])
    assert result['status'] == 'completed' and not result['comparable']
    assert all(run['score'] == 0.8 for run in result['runs'])
    assert any('older' in warning and 'scor' in warning for warning in result['warnings'])
    with get_session() as session:
        assert all(row.results[0].meta_data['scoring']['version'] == 'final-answer-v2'
                   for row in session.query(TestRun).all())
