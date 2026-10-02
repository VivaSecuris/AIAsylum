"""Explicit Ollama context budgets preserve full history and are recorded honestly."""

import json

import httpx
import pytest

from vivasecuris.aiasylum.models.ollama import OllamaModel
from vivasecuris.aiasylum.runner.run_config import generation_record, validate_test_config
from vivasecuris.aiasylum.utils.model_context import model_gen_kwargs_from_context


@pytest.mark.asyncio
async def test_group_doctor_context_reaches_ollama_without_shortening_any_message():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={
            "done": True, "done_reason": "length", "message": {"content": "Question"},
            "prompt_eval_count": 8000, "eval_count": 128,
        })

    model = OllamaModel("test-doctor")
    model._client = httpx.AsyncClient(base_url="http://ollama.test", transport=httpx.MockTransport(handler))
    context = {"roles": {"doctor": {"num_ctx": 32768, "max_tokens": 128, "temperature": 0}}}
    messages = [{"role": "system", "content": "Lead the whole group."}] + [
        {"role": "user", "content": f"Patient {i}: " + "shared context " * 900} for i in range(5)
    ] + [{"role": "user", "content": "Ask a question about the earliest patient's answer."}]
    try:
        response = await model.generate("", messages=messages, **model_gen_kwargs_from_context(context, role="doctor"))
    finally:
        await model.close()
    assert seen[0]["messages"] == messages
    assert seen[0]["options"] == {"temperature": 0.0, "num_predict": 128, "num_ctx": 32768}
    assert response.metadata["request_num_ctx"] == 32768
    assert response.metadata["request_message_count"] == len(messages)
    assert response.usage["prompt_eval_count"] == 8000
    assert response.finish_reason == "length"
    assert generation_record(context, "doctor", model)["num_ctx"] == 32768


@pytest.mark.asyncio
async def test_generate_and_stream_honor_per_call_context_over_model_default():
    seen = []

    def handler(request):
        payload = json.loads(request.content)
        seen.append(payload)
        if payload["stream"]:
            return httpx.Response(200, text=json.dumps({"response": "answer", "done": True}) + "\n")
        return httpx.Response(200, json={"response": "answer", "done": True})

    model = OllamaModel("test-model", num_ctx=8192)
    model._client = httpx.AsyncClient(base_url="http://ollama.test", transport=httpx.MockTransport(handler))
    try:
        response = await model.generate("Complete input")
        assert response.metadata["request_num_ctx"] == 8192
        chunks = [chunk async for chunk in model.stream_generate("Complete input", num_ctx=32768, seed=0)]
    finally:
        await model.close()
    assert chunks == ["answer"]
    assert [request["options"]["num_ctx"] for request in seen] == [8192, 32768]
    assert all(request["prompt"] == "Complete input" for request in seen)
    assert seen[1]["options"]["seed"] == 0


def test_context_budget_is_optional_validated_and_role_scoped():
    model = OllamaModel("test-model")
    assert "num_ctx" not in model._generation_options({})
    context = {"roles": {"doctor": {"num_ctx": 32768}, "patient": {"num_ctx": 8192}}}
    assert validate_test_config(context) == []
    assert model_gen_kwargs_from_context(context, "doctor")["num_ctx"] == 32768
    assert model_gen_kwargs_from_context(context, "patient", {"num_ctx": 16384})["num_ctx"] == 16384
    for value in [0, -1, 2.5, "invalid", 1048577]:
        assert validate_test_config({"roles": {"doctor": {"num_ctx": value}}})
    for value in [0, -1, 2.5, True, "invalid"]:
        with pytest.raises(ValueError, match="num_ctx"):
            model._generation_options({"num_ctx": value})


def test_context_window_form_control_and_round_trip_are_ollama_only():
    from tests.test_conversation_frontend_identity import run_frontend

    run_frontend(r"""
const config = load(path.join(frontend, 'lib/create-test-config'));
const { GenerationSettings } = load(path.join(frontend, 'components/forms/create-test/GenerationSettings'));
const { RunConfigurationCard } = load(path.join(frontend, 'components/test-runs/RunConfigurationCard'));
let { state } = config.initialFormState({}, { type: 'conversation' });
state.doctor.provider = 'ollama'; state.doctor.model = 'test-doctor';
state.patient.provider = 'ollama'; state.patient.model = 'test-patient';
state.doctor.generation.num_ctx = '32768';
state.patient.generation.num_ctx = '8192';
const payload = config.buildTestRunRequest(state);
assert.equal(payload.test_config.roles.doctor.num_ctx, 32768);
assert.equal(payload.test_config.roles.patient.num_ctx, 8192);
const restored = config.initialFormState({}, { type: 'conversation',
  doctor_provider: 'ollama', doctor_model: 'test-doctor', provider: 'ollama', model: 'test-patient',
  test_config: JSON.stringify(payload.test_config) }).state;
assert.equal(restored.doctor.generation.num_ctx, '32768');
assert.equal(restored.patient.generation.num_ctx, '8192');
state.doctor.generation.num_ctx = '-1';
assert(config.validateForm(state).errors.some(e => e.includes('context window')));
state.doctor.provider = 'openai';
assert(!('num_ctx' in config.buildTestRunRequest(state).test_config.roles.doctor));
assert(!config.validateForm(state).errors.some(e => e.includes('context window')));
let updated;
const props = { value: restored.doctor.generation, onChange: v => { updated = v; } };
const hidden = renderToStaticMarkup(React.createElement(GenerationSettings, props));
assert(!hidden.includes('Context window (tokens)'));
const tree = GenerationSettings({ ...props, showContextWindow: true });
const html = renderToStaticMarkup(tree);
assert(html.includes('Context window (tokens)'));
assert(html.includes('context 32768 tokens'));
find(tree, n => n.type === 'input' && n.props.placeholder === 'Ollama default').props.onChange({ target: { value: '16384' } });
assert.equal(updated.num_ctx, '16384');
const recorded = renderToStaticMarkup(React.createElement(RunConfigurationCard, { testRun: {
  test_type: 'group_therapy', meta_data: { resolved_config: { generation: { doctor: { num_ctx: 32768 } } } }
} }));
assert(recorded.includes('context 32768 tokens'));
""")
