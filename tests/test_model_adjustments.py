"""Natural-language proposals cannot silently change weights, settings or chat context."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from vivasecuris.aiasylum.api.routes import model_adjustments as routes
from vivasecuris.aiasylum.database.models import ModelAdjustment, WeightRun
from tests.test_model_catalog import checkpoint


def proposal():
    return {"summary": "Use short answers without inventing personal experiences.",
            "system_prompt": "You are an AI assistant. Answer briefly and do not invent a human biography.",
            "training_examples": [{"prompt": f"Training question {i}", "response": f"Brief AI answer {i}"} for i in range(6)],
            "test_prompts": ["Fresh question A", "Fresh question B"]}


def request(mode="profile", **updates):
    return routes.AdjustmentRequest.model_validate({
        "provider": "ollama" if mode == "profile" else "transformers", "model": "selected-model",
        "mode": mode, "instruction": "Stop inventing an age. Keep answers short.",
        "messages": [{"role": "user", "content": "Who are you?"}, {"role": "assistant", "content": "I am a 62-year-old man."}],
        "settings": {"system_prompt": " Exact original system\n", "temperature": 0, "enable_cot": False, "seed": 0},
        "coach": {"provider": "ollama", "model": "chosen-coach", "temperature": 0, "enable_cot": False},
        **updates,
    })


@pytest.fixture
def chat(monkeypatch):
    calls = []
    async def generate(body):
        calls.append(body.model_dump())
        return {"content": json.dumps(proposal()) if body.model == "chosen-coach" else "I am an AI assistant.",
                "provider": body.provider, "model": body.model, "finish_reason": "stop", "metadata": {}, "usage": {}}
    monkeypatch.setattr(routes, "chat_with_model", generate)
    return calls


@pytest.mark.asyncio
async def test_profile_propose_apply_retest_reopen_preserves_original(test_db, chat):
    body = request()
    original = deepcopy(body.model_dump())
    result = await routes.propose_adjustment(body)
    assert result["status"] == "proposed" and result["active_target"] is None
    assert test_db.query(WeightRun).count() == 0
    assert len(chat) == 1 and chat[0]["model"] == "chosen-coach"
    assert chat[0]["system_prompt"] == routes.COACH_SYSTEM
    supplied = json.loads(chat[0]["messages"][0]["content"])
    assert supplied["current_system_prompt"] == body.settings.system_prompt
    assert supplied["conversation"] == original["messages"]
    with pytest.raises(HTTPException) as error:
        await routes.test_adjustment(result["id"])
    assert error.value.status_code == 409 and len(chat) == 1
    applied = await routes.apply_adjustment(result["id"], routes.ApplyAdjustment())
    assert applied["active_target"]["settings"]["system_prompt"] == proposal()["system_prompt"]
    assert applied["settings"]["system_prompt"] == original["settings"]["system_prompt"]
    assert await routes.apply_adjustment(result["id"], routes.ApplyAdjustment()) == applied
    answer = await routes.test_adjustment(result["id"])
    assert answer["content"] == "I am an AI assistant."
    assert chat[-1]["messages"] == original["messages"][:-1]
    assert chat[-1]["system_prompt"] == proposal()["system_prompt"]
    assert chat[-1]["temperature"] == 0 and chat[-1]["seed"] == 0 and chat[-1]["enable_cot"] is False
    assert "change_request" not in str(chat[-1]) and "Training question" not in str(chat[-1])
    reopened = routes.list_adjustments("ollama", "selected-model")[0]
    assert reopened["id"] == result["id"] and reopened["tests"][0]["response"] == answer
    assert reopened["baseline"]["response"] == "I am a 62-year-old man."
    assert test_db.query(WeightRun).count() == 0 and body.model_dump() == original


@pytest.mark.asyncio
async def test_multi_turn_retest_uses_identical_prefix_not_generated_examples(test_db, chat):
    body = request(messages=[{"role": "user", "content": "Remember copper."},
        {"role": "assistant", "content": "Copper remembered."},
        {"role": "user", "content": "What word?"}, {"role": "assistant", "content": "Silver."}])
    result = await routes.propose_adjustment(body)
    await routes.apply_adjustment(result["id"], routes.ApplyAdjustment())
    await routes.test_adjustment(result["id"])
    assert chat[-1]["messages"] == [m.model_dump() for m in body.messages[:-1]]


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", ["not json", "{}", json.dumps({**proposal(), "command": "touch anything"}),
                                    json.dumps({**proposal(), "training_examples": []})])
async def test_malformed_or_executable_coach_result_creates_nothing(test_db, monkeypatch, bad):
    async def generate(body):
        return {"content": bad, "finish_reason": "stop"}
    monkeypatch.setattr(routes, "chat_with_model", generate)
    with pytest.raises(HTTPException) as error:
        await routes.propose_adjustment(request())
    assert error.value.status_code == 502
    assert test_db.query(ModelAdjustment).count() == test_db.query(WeightRun).count() == 0


def test_duplicate_or_overlapping_examples_rejected():
    duplicate = proposal()
    duplicate["training_examples"][1]["prompt"] = " Training   QUESTION 0 "
    with pytest.raises(HTTPException):
        routes.parse_proposal(json.dumps(duplicate))
    overlap = proposal()
    overlap["test_prompts"][0] = "Training question 0"
    with pytest.raises(HTTPException):
        routes.parse_proposal(json.dumps(overlap))
    assert routes.parse_proposal("```json\n" + json.dumps(proposal()) + "\n```") == proposal()


@pytest.mark.asyncio
async def test_truncated_coach_output_never_applied(test_db, monkeypatch):
    async def generate(body):
        return {"content": json.dumps(proposal()), "finish_reason": "length"}
    monkeypatch.setattr(routes, "chat_with_model", generate)
    with pytest.raises(HTTPException, match="token limit"):
        await routes.propose_adjustment(request())
    assert test_db.query(ModelAdjustment).count() == 0


@pytest.mark.asyncio
async def test_remote_weights_rejected_before_any_coach_call(test_db, chat):
    with pytest.raises(HTTPException) as error:
        await routes.propose_adjustment(request(mode="weights", provider="ollama"))
    assert error.value.status_code == 409 and not chat


@pytest.mark.asyncio
async def test_training_is_bounded_and_only_apply_launches_it_once(test_db, chat, monkeypatch, tmp_path):
    from vivasecuris.aiasylum.api.routes import weights
    source = str(checkpoint(tmp_path / "original"))
    monkeypatch.setattr(routes, "local_training_source", lambda provider, model: source)
    launched = []
    async def create(body):
        launched.append(body.model_dump())
        row = WeightRun(kind="lora", source_model=body.source_model, status="pending", out_dir="models/new")
        test_db.add(row)
        test_db.commit()
        return SimpleNamespace(id=row.id)
    monkeypatch.setattr(weights, "create_weight_run", create)
    result = await routes.propose_adjustment(request("weights"))
    assert not launched and result["weight_request"]["kind"] == "lora"
    assert result["weight_request"]["source_model"] == source
    assert result["weight_request"]["adjustment_id"] == result["id"]
    assert result["weight_request"]["dataset_rows"][0]["system"] == " Exact original system\n"
    assert result["weight_request"]["max_steps"] == 20 and result["weight_request"]["merge"] is True
    applied = await routes.apply_adjustment(result["id"], routes.ApplyAdjustment(acknowledge=["memory_pressure"]))
    assert applied["status"] == "training" and applied["active_target"] is None
    await routes.apply_adjustment(result["id"], routes.ApplyAdjustment())
    assert len(launched) == 1 and launched[0]["acknowledge"] == ["memory_pressure"]
    with pytest.raises(HTTPException) as error:
        await routes.test_adjustment(result["id"])
    assert error.value.status_code == 409
    assert len(chat) == 1


@pytest.mark.asyncio
async def test_training_preflight_failure_allows_reviewed_retry(test_db, chat, monkeypatch, tmp_path):
    from vivasecuris.aiasylum.api.routes import weights
    source = str(checkpoint(tmp_path / "original"))
    monkeypatch.setattr(routes, "local_training_source", lambda provider, model: source)
    async def rejected(body):
        raise HTTPException(409, "insufficient resources")
    monkeypatch.setattr(weights, "create_weight_run", rejected)
    result = await routes.propose_adjustment(request("weights"))
    with pytest.raises(HTTPException):
        await routes.apply_adjustment(result["id"], routes.ApplyAdjustment())
    assert routes.get_adjustment(result["id"])["status"] == "proposed"
    assert test_db.query(WeightRun).count() == 0


@pytest.mark.asyncio
async def test_parent_cannot_silently_select_another_model(test_db, chat):
    first = await routes.propose_adjustment(request())
    await routes.apply_adjustment(first["id"], routes.ApplyAdjustment())
    with pytest.raises(HTTPException) as error:
        await routes.propose_adjustment(request(model="other-model", parent_id=first["id"]))
    assert error.value.status_code == 409 and len(chat) == 1


@pytest.mark.parametrize("messages", [[], [{"role": "user", "content": "hello"}],
    [{"role": "assistant", "content": "a"}, {"role": "assistant", "content": "b"}]])
def test_only_completed_input_answer_pairs_are_adjustable(messages):
    with pytest.raises(ValueError):
        request(messages=messages)


def test_catalog_resolves_only_ready_verified_local_aliases(test_db, monkeypatch, tmp_path):
    from vivasecuris.aiasylum.api import model_catalog
    source = str(checkpoint(tmp_path / "cached" / "model"))
    monkeypatch.setattr(model_catalog, "build_model_catalog", lambda: {"models": [
        {"model_ref": source, "aliases": ["alias"], "availability": "ready"},
        {"model_ref": "missing", "availability": "missing"}]})
    assert routes.local_training_source("local", "alias") == source
    for provider, model in [("ollama", "alias"), ("transformers", "missing"), ("transformers", "not-installed")]:
        with pytest.raises(HTTPException):
            routes.local_training_source(provider, model)


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["x" * 18000, "\\" * 9000])
async def test_serialized_coach_limit_rejects_before_generation_or_persistence(test_db, chat, content):
    body = request(messages=[{"role": "user", "content": content}, {"role": "assistant", "content": content}])
    with pytest.raises(HTTPException) as error:
        await routes.propose_adjustment(body)
    assert error.value.status_code == 413
    assert "32000" in error.value.detail and "Shorten the conversation" in error.value.detail
    assert chat == [] and test_db.query(ModelAdjustment).count() == test_db.query(WeightRun).count() == 0


@pytest.mark.asyncio
async def test_exact_serialized_coach_limit_is_accepted(test_db, chat):
    body = request(messages=[{"role": "user", "content": "x"}, {"role": "assistant", "content": "answer"}])
    coach_data = {"change_request": body.instruction, "mode": body.mode,
                  "current_system_prompt": body.settings.system_prompt, "conversation": [m.model_dump() for m in body.messages]}
    extra = 32000 - len(json.dumps(coach_data, ensure_ascii=False))
    body.messages[0].content += "x" * extra
    result = await routes.propose_adjustment(body)
    assert result["status"] == "proposed" and len(chat[0]["messages"][0]["content"]) == 32000


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["servus", "agentic_a2a", "agentic", " AGENTIC "])
@pytest.mark.parametrize("role", ["target", "assistant"])
async def test_service_provider_and_alias_cannot_target_or_coach_profiles(test_db, chat, provider, role):
    updates = {"provider": provider} if role == "target" else {"coach": {"provider": provider, "model": "chosen-coach"}}
    with pytest.raises(HTTPException) as error:
        await routes.propose_adjustment(request(**updates))
    assert error.value.status_code == 422
    assert role in error.value.detail and "system instructions and conversation history" in error.value.detail
    assert chat == [] and test_db.query(ModelAdjustment).count() == 0


@pytest.mark.asyncio
async def test_old_service_profile_cannot_be_applied_or_advertised_active(test_db, chat):
    result = await routes.propose_adjustment(request())
    row = test_db.get(ModelAdjustment, result["id"])
    row.provider = "agentic"
    test_db.commit()
    with pytest.raises(HTTPException) as error:
        await routes.apply_adjustment(row.id, routes.ApplyAdjustment())
    assert error.value.status_code == 422
    row.status = "applied"
    test_db.commit()
    reopened = routes.get_adjustment(row.id)
    assert reopened["status"] == "failed" and reopened["active_target"] is None
    with pytest.raises(HTTPException):
        await routes.test_adjustment(row.id)
    assert len(chat) == 1


def install_cached_catalog(tmp_path, monkeypatch):
    from vivasecuris.aiasylum.api import model_catalog
    repo = tmp_path / "hub" / "models--org--base"
    snapshot = checkpoint(repo / "snapshots" / "revision-a")
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("revision-a")
    monkeypatch.setattr(model_catalog, "_cache_roots", lambda: [tmp_path / "hub"])
    original = model_catalog.build_model_catalog
    monkeypatch.setattr(model_catalog, "build_model_catalog", lambda: original(
        models_root=tmp_path / "models", project_root=tmp_path, cache_roots=[tmp_path / "hub"],
        presets_path=tmp_path / "matrix.json"))
    return repo, snapshot


def test_cached_repo_is_pinned_and_explicit_snapshot_never_becomes_mutable_repo(tmp_path, monkeypatch):
    from vivasecuris.aiasylum.api import model_catalog
    repo, snapshot = install_cached_catalog(tmp_path, monkeypatch)
    assert routes.local_training_source("local", "org/base") == str(snapshot)
    assert routes.local_training_source("transformers", str(snapshot)) == str(snapshot)
    next_snapshot = checkpoint(repo / "snapshots" / "revision-b")
    (repo / "refs" / "main").write_text("revision-b")
    assert routes.local_training_source("transformers", "org/base") == str(next_snapshot)
    assert routes.local_training_source("transformers", str(snapshot)) == str(snapshot)
    original = model_catalog.build_model_catalog
    monkeypatch.setattr(model_catalog, "build_model_catalog", lambda: {"models": [
        {**original()["models"][0], "aliases": ["server-verified-alias"]}]})
    assert routes.local_training_source("local", "server-verified-alias") == str(next_snapshot)
    with pytest.raises(HTTPException):
        routes.local_training_source("local", "unverified-alias")


@pytest.mark.asyncio
@pytest.mark.parametrize("remove_pin", [False, True])
async def test_apply_uses_saved_snapshot_when_main_changes(test_db, chat, monkeypatch, tmp_path, remove_pin):
    from vivasecuris.aiasylum.api.routes import weights
    repo, snapshot = install_cached_catalog(tmp_path, monkeypatch)
    result = await routes.propose_adjustment(request("weights", model="org/base"))
    assert result["training_source"] == str(snapshot)
    checkpoint(repo / "snapshots" / "revision-b")
    (repo / "refs" / "main").write_text("revision-b")
    monkeypatch.setattr(routes, "local_training_source", lambda *args: pytest.fail("Apply must not resolve mutable model refs"))
    calls = []
    async def create(body):
        calls.append(body)
        row = WeightRun(kind="lora", source_model=body.source_model, status="pending", out_dir=str(tmp_path / "new"))
        test_db.add(row)
        test_db.commit()
        return SimpleNamespace(id=row.id)
    monkeypatch.setattr(weights, "create_weight_run", create)
    if remove_pin:
        (snapshot / "model.safetensors").unlink()
        with pytest.raises(HTTPException) as error:
            await routes.apply_adjustment(result["id"], routes.ApplyAdjustment())
        assert error.value.status_code == 409 and "saved training checkpoint" in error.value.detail
        assert calls == [] and routes.get_adjustment(result["id"])["status"] == "proposed"
    else:
        await routes.apply_adjustment(result["id"], routes.ApplyAdjustment())
        assert len(calls) == 1 and calls[0].source_model == str(snapshot)


def test_existing_relative_local_checkpoint_is_resolved_without_hub_alias(tmp_path, monkeypatch):
    path = checkpoint(tmp_path / "models" / "edit")
    monkeypatch.chdir(tmp_path)
    assert routes.local_training_source("transformers", "models/edit") == str(path)
    with pytest.raises(HTTPException, match="pinned local training checkpoint"):
        routes.ready_training_source("org/base")


def test_missing_explicit_snapshot_does_not_fall_back_to_ready_repo_alias(tmp_path, monkeypatch):
    from vivasecuris.aiasylum.api import model_catalog
    repo, snapshot = install_cached_catalog(tmp_path, monkeypatch)
    missing = repo / "snapshots" / "deleted-old-revision"
    monkeypatch.setattr(model_catalog, "build_model_catalog", lambda: {"models": [
        {"model_ref": "org/base", "aliases": [str(missing)], "availability": "ready"}]})
    with pytest.raises(HTTPException, match="saved training checkpoint"):
        routes.local_training_source("transformers", str(missing))
    assert routes.local_training_source("transformers", str(snapshot)) == str(snapshot)


@pytest.mark.asyncio
async def test_apply_failure_after_job_commit_retains_failed_correlation(test_db, chat, monkeypatch, tmp_path):
    from vivasecuris.aiasylum.api.routes import weights
    source = str(checkpoint(tmp_path / "source"))
    result = await routes.propose_adjustment(request("weights", model=source))
    calls = []
    async def create(body):
        calls.append(body)
        test_db.add(WeightRun(kind="lora", source_model=body.source_model, status="pending", out_dir="models/new",
                             meta_data={"adjustment_id": body.adjustment_id}))
        test_db.commit()
        raise OSError("Dataset write failed")
    monkeypatch.setattr(weights, "create_weight_run", create)
    with pytest.raises(OSError, match="Dataset write failed"):
        await routes.apply_adjustment(result["id"], routes.ApplyAdjustment())
    test_db.expire_all()
    job = test_db.query(WeightRun).one()
    parent = test_db.get(ModelAdjustment, result["id"])
    assert parent.status == "failed" and parent.weight_run_id == job.id
    assert job.status == "failed" and "Dataset write failed" in job.error
    assert routes.get_adjustment(parent.id)["active_target"] is None
    with pytest.raises(HTTPException, match="previous application failed") as error:
        await routes.apply_adjustment(parent.id, routes.ApplyAdjustment())
    assert error.value.status_code == 409 and len(calls) == 1


@pytest.mark.asyncio
async def test_client_reported_baseline_evidence_is_stored_without_entering_prompts(test_db, chat):
    evidence = {"content": "I am a 62-year-old man.", "provider": "ollama", "model": "selected-model",
                "finish_reason": "stop", "usage": {"completion_tokens": 12},
                "metadata": {"reasoning": "PRIVATE_BASELINE_TRACE", "request_system_prompts": ["PRIVATE_BASELINE_SYSTEM"]}}
    body = request(baseline_evidence=evidence)
    result = await routes.propose_adjustment(body)
    assert result["baseline"]["evidence"] == evidence
    assert result["baseline"]["evidence_source"] == "client_reported"
    await routes.apply_adjustment(result["id"], routes.ApplyAdjustment())
    await routes.test_adjustment(result["id"])
    assert "PRIVATE_BASELINE_TRACE" not in str(chat) and "PRIVATE_BASELINE_SYSTEM" not in str(chat)
    assert "baseline_evidence" not in str(chat)
    assert routes.get_adjustment(result["id"])["baseline"]["evidence"] == evidence


@pytest.mark.parametrize("evidence,reason", [
    ({"content": "Different answer"}, "exact last assistant"),
    ({"content": "I am a 62-year-old man.", "metadata": {"reasoning": "x" * 128000}}, "128000"),
    ({"content": "I am a 62-year-old man.", "metadata": {"value": float("nan")}}, "JSON-safe"),
    ({"content": "I am a 62-year-old man.", "metadata": {"value": object()}}, "JSON-safe"),
])
def test_mismatched_oversized_or_nonjson_baseline_evidence_is_rejected(evidence, reason):
    with pytest.raises(ValueError, match=reason):
        request(baseline_evidence=evidence)
