"""Exercise user-selected options through Patient and persisted run execution."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from vivasecuris.aiasylum.models.base import ModelResponse
from vivasecuris.aiasylum.models.providers import AnthropicModel, GoogleModel, OpenAIModel
from vivasecuris.aiasylum.patient import Patient


def openai_response():
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="answer"), finish_reason="stop")], usage=None)


@pytest.mark.asyncio
async def test_patient_openai_honors_context_and_keeps_history():
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=openai_response())
    model = OpenAIModel("custom-account-model", SimpleNamespace(client=client), temperature=0.7)
    patient = Patient(model, system_prompt="Answer questions.")
    await patient.respond("First question", context={"temperature": 0.0, "seed": 17})
    await patient.respond("Follow up", context={"temperature": 0.2, "seed": 18})
    first, second = [call.kwargs for call in client.chat.completions.create.call_args_list]
    assert first["temperature"] == 0.0 and first["seed"] == 17
    assert second["temperature"] == 0.2 and second["seed"] == 18
    assert [message["role"] for message in second["messages"]] == ["system", "user", "assistant", "user"]
    assert model.temperature == 0.7


@pytest.mark.asyncio
async def test_patient_anthropic_overrides_and_rejects_unavailable_seed():
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=SimpleNamespace(content=[SimpleNamespace(text="answer")], stop_reason="end_turn", usage=None))
    model = AnthropicModel("custom-claude-id", SimpleNamespace(client=client))
    await Patient(model, system_prompt="Be concise.").respond("Question", context={"temperature": 0.0})
    request = client.messages.create.call_args.kwargs
    assert request["temperature"] == 0.0 and request["system"] == "Be concise."
    assert [message["role"] for message in request["messages"]] == ["user"]
    with pytest.raises(ValueError, match="does not support a generation seed"):
        await Patient(model).respond("Question", context={"seed": 3})
    assert client.messages.create.await_count == 1


@pytest.mark.asyncio
async def test_google_honors_overrides_for_single_and_history_paths():
    client = MagicMock()
    google = client.GenerativeModel.return_value
    response = SimpleNamespace(text="answer", finish_reason="stop", usage_metadata=None)
    google.generate_content_async = AsyncMock(return_value=response)
    google.start_chat.return_value.send_message_async = AsyncMock(return_value=response)
    model = GoogleModel("custom-gemini-id", SimpleNamespace(client=client), max_tokens=31)
    patient = Patient(model, system_prompt="Be concise.")
    await patient.respond("Question", context={"temperature": 0.0})
    await patient.respond("Follow up", context={"temperature": 0.2})
    assert google.generate_content_async.call_args.kwargs["generation_config"] == {"temperature": 0.0, "max_output_tokens": 31}
    assert google.start_chat.return_value.send_message_async.call_args.kwargs["generation_config"]["temperature"] == 0.2
    await model.generate("Question", max_tokens=12, top_p=0.8)
    assert google.generate_content_async.call_args.kwargs["generation_config"]["max_output_tokens"] == 12
    assert google.generate_content_async.call_args.kwargs["generation_config"]["top_p"] == 0.8
    with pytest.raises(ValueError, match="does not support a generation seed"):
        await model.generate("Question", seed=3)
    with pytest.raises(ValueError, match="Unsupported Google generation options"):
        await model.generate("Question", unsupported_option=True)


async def chunks(*values):
    for value in values:
        yield value


@pytest.mark.asyncio
async def test_stream_generation_uses_same_overrides():
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=chunks(SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="openai"))])))
    openai = OpenAIModel("model", SimpleNamespace(client=client))
    assert [part async for part in openai.stream_generate("q", temperature=0.0, max_tokens=11, seed=4)] == ["openai"]
    assert client.chat.completions.create.call_args.kwargs["max_tokens"] == 11
    assert client.chat.completions.create.call_args.kwargs["temperature"] == 0.0

    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=SimpleNamespace(text_stream=chunks("anthropic")))
    context.__aexit__ = AsyncMock(return_value=None)
    client.messages.stream.return_value = context
    anthropic = AnthropicModel("model", SimpleNamespace(client=client))
    assert [part async for part in anthropic.stream_generate("q", temperature=0.2, max_tokens=12)] == ["anthropic"]
    assert client.messages.stream.call_args.kwargs["max_tokens"] == 12
    assert client.messages.stream.call_args.kwargs["temperature"] == 0.2

    client.GenerativeModel.return_value.generate_content_async = AsyncMock(return_value=chunks(SimpleNamespace(text="google")))
    google = GoogleModel("model", SimpleNamespace(client=client))
    assert [part async for part in google.stream_generate("q", temperature=0.3, max_tokens=13)] == ["google"]
    assert client.GenerativeModel.return_value.generate_content_async.call_args.kwargs["generation_config"] == {"temperature": 0.3, "max_output_tokens": 13}


@pytest.mark.asyncio
async def test_benchmark_distinguishes_sample_seed_from_unsupported_generation_seed(monkeypatch):
    from vivasecuris.aiasylum.tests.benchmark import BenchmarkTest
    from vivasecuris.aiasylum.tests import benchmark

    async def dataset(*args, **kwargs):
        assert kwargs["seed"] == 17
        return [{"question": "Choose A", "choices": ["First", "Second"], "answer": "A", "dataset_index": 0}]

    monkeypatch.setattr(benchmark, "load_benchmark_dataset", dataset)
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=SimpleNamespace(content=[SimpleNamespace(text="A")], stop_reason="end_turn", usage=None))
    model = AnthropicModel("model", SimpleNamespace(client=client))
    result = await BenchmarkTest(benchmark_name="mmlu", num_samples=1, seed=17).run(model)
    assert result.score == 1.0
    assert "seed" not in client.messages.create.call_args.kwargs
    assert result.metadata["generation"]["seed"] is None
    assert result.metadata["generation"]["sample_seed"] == 17
    assert result.metadata["generation"]["seed_supported"] is False
    assert result.metadata["runtime"]["provider"] == "anthropic"
    import json
    json.dumps(result.metadata)


@pytest.mark.asyncio
@pytest.mark.parametrize("entrypoint", ["direct", "queued"])
async def test_multi_shot_library_selection_runs_all_prompts_in_order(monkeypatch, test_db, entrypoint):
    from vivasecuris.aiasylum.database import PromptLibrary, TestResult, TestRun
    from vivasecuris.aiasylum.runner import runner
    from tests.test_doctor_patient import MockModel

    records = [PromptLibrary(name=str(index), prompt_type="test_prompt", prompt_text=text) for index, text in enumerate(["Remember $word", "Repeat $word"])]
    test_db.add_all(records)
    test_db.commit()
    ids = [records[1].id, records[0].id]
    model = MockModel()
    calls = []

    async def generate(*args, **kwargs):
        calls.append(deepcopy(kwargs))
        return ModelResponse(content="Acknowledged", model="mock", provider="mock")

    model.generate = generate
    monkeypatch.setattr(runner, "get_provider", lambda _: SimpleNamespace(create_model=lambda _: model))
    config = {"prompt_ids": ids, "variables": {"word": "orchid"}, "temperature": 0.0}
    if entrypoint == "direct":
        await runner.TestRunner().run_test("mock", "m", "mock", "m", "multi_shot", config)
    else:
        row = TestRun(doctor_provider="mock", doctor_model="m", patient_provider="mock", patient_model="m", test_type="multi_shot", status="pending", meta_data={"test_config": config})
        test_db.add(row)
        test_db.commit()
        await runner.TestRunner().execute_test_run(row.id)

    test_db.expire_all()
    result = test_db.query(TestResult).one()
    assert [turn["prompt"] for turn in result.meta_data["conversation_history"]] == ["Repeat orchid", "Remember orchid"]
    assert len(calls[0]["messages"]) == 1
    assert len(calls[1]["messages"]) == 3
    assert all(call["temperature"] == 0.0 for call in calls)
    assert [record.usage_count for record in records] == [1, 1]
    assert test_db.query(TestRun).one().status == "completed"


def test_multi_shot_missing_library_prompt_fails_before_generation(test_db):
    from vivasecuris.aiasylum.runner.runner import _resolve_multi_shot_prompts
    with pytest.raises(ValueError, match="Selected test prompts are unavailable"):
        _resolve_multi_shot_prompts(test_db, {"prompt_ids": [918237]})


@pytest.mark.asyncio
async def test_unimplemented_toxicity_evaluator_is_marked_unavailable():
    from vivasecuris.aiasylum.api.routes.benchmarks import list_benchmarks
    rows = (await list_benchmarks())["benchmarks"]
    toxicity = next(row for row in rows if row["name"] == "realtoxicityprompts")
    assert toxicity["runnable"] is False
    assert "toxicity evaluator" in toxicity["unavailable_reason"]
    assert next(row for row in rows if row["name"] == "mmlu")["runnable"] is True


@pytest.mark.parametrize("provider", ["openai", "anthropic", "google"])
def test_hosted_provider_accepts_account_specific_model_ids(provider):
    from vivasecuris.aiasylum.models.registry import get_provider_info
    assert get_provider_info(provider).model_input == "text"
