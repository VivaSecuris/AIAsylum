"""Doctor/patient roles stay separate at the provider request boundaries.

Only the network transport and tensor model are replaced. Factory wrappers,
Ollama request construction, and local chat formatting use their real paths.
No network connection, checkpoint load, or model-job lease is acquired.
"""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from vivasecuris.aiasylum.models.providers import get_provider


def test_real_causal_model_starts_each_role_with_empty_generation_cache(monkeypatch):
    """Exercise actual HF generation, not a tensor-model mock or answer heuristic."""
    torch = pytest.importorskip("torch")
    from tests._tiny_lm import build_model, build_tokenizer
    from vivasecuris.aiasylum.models import transformers_local

    tensor_model, tokenizer = build_model(), build_tokenizer()
    tokenizer.chat_template = "{% for m in messages %}{{m['role']}} {{m['content']}} {% endfor %}assistant "
    monkeypatch.delenv("AIASYLUM_BENCHMARK_RUNTIME", raising=False)
    monkeypatch.setattr(transformers_local, "_get_cached", lambda *args: (tensor_model, tokenizer))
    wrappers = [get_provider("transformers").create_model("same-weights") for _ in range(2)]
    histories = [[{"role": "system", "content": text}, {"role": "user", "content": "w1 w2"}]
                 for text in ("w10 w11", "w20 w21 w22")]
    prefill = []
    scores = []
    first = False

    def capture_input(module, args, kwargs):
        nonlocal first
        if first:
            return
        first = True
        cache = kwargs.get("past_key_values")
        assert cache is None or int(cache.get_seq_length()) == 0
        assert int(kwargs["cache_position"][0]) == 0
        prefill.append(kwargs["input_ids"][0].clone())

    def capture_output(module, args, kwargs, output):
        if len(scores) < len(prefill):
            scores.append(output.logits[0, -1].detach().clone())

    before = tensor_model.register_forward_pre_hook(capture_input, with_kwargs=True)
    after = tensor_model.register_forward_hook(capture_output, with_kwargs=True)
    responses = []
    try:
        for role in (0, 1, 0):
            first = False
            wrapper = wrappers[role]
            text, applied = wrapper._build_prompt(tokenizer, "", None, histories[role])
            expected = tokenizer(text, add_special_tokens=not applied)["input_ids"]
            responses.append(wrapper._generate_sync("", None, histories[role], temperature=0, max_tokens=4, seed=7))
            assert prefill[-1].tolist() == expected
    finally:
        before.remove()
        after.remove()
    assert torch.equal(prefill[0], prefill[2])
    assert not torch.equal(prefill[0], prefill[1])
    assert torch.equal(scores[0], scores[2])
    assert responses[0].content == responses[2].content


def _history(role):
    return [
        {"role": "system", "content": f"{role} SYSTEM ONLY"},
        {"role": "user", "content": f"{role} first input"},
        {"role": "assistant", "content": f"{role} previous reply"},
        {"role": "user", "content": f"{role} follow-up"},
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("same_model", [False, True])
async def test_ollama_sends_each_roles_system_and_own_history(same_model):
    provider = get_provider("ollama")
    doctor = provider.create_model("doctor-model", temperature=0.1, max_tokens=17)
    patient = provider.create_model("doctor-model" if same_model else "patient-model",
                                    temperature=0.9, max_tokens=31)
    assert doctor is not patient and doctor.kwargs is not patient.kwargs
    payloads = []

    async def post(path, *, json, timeout):
        assert path == "/api/chat"
        payloads.append(deepcopy(json))
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"message": {"content": "answer"}, "done": True},
        )

    # Even a shared stateless transport does not share either role's messages.
    client = SimpleNamespace(post=post)
    doctor._get_client = AsyncMock(return_value=client)
    patient._get_client = AsyncMock(return_value=client)
    histories = [_history("DOCTOR"), _history("PATIENT"), _history("DOCTOR")]
    originals = deepcopy(histories)
    responses = [
        await doctor.generate("", messages=histories[0], temperature=0.2, top_p=0.8, seed=11),
        await patient.generate("", messages=histories[1], top_p=0.95, seed=22),
        await doctor.generate("", messages=histories[2]),
    ]

    assert histories == originals
    assert [p["messages"] for p in payloads] == originals
    assert [p["model"] for p in payloads] == [doctor.model_name, patient.model_name, doctor.model_name]
    assert [p["options"] for p in payloads] == [
        {"temperature": 0.2, "num_predict": 17, "top_p": 0.8, "seed": 11},
        {"temperature": 0.9, "num_predict": 31, "top_p": 0.95, "seed": 22},
        {"temperature": 0.1, "num_predict": 17},
    ]
    for response, payload in zip(responses, payloads):
        assert response.metadata["request_system_prompts"] == [
            message["content"] for message in payload["messages"] if message["role"] == "system"
        ]
        assert response.metadata["request_system_prompts_source"] == "provider"
    assert doctor.temperature == 0.1 and patient.temperature == 0.9


@pytest.mark.asyncio
@pytest.mark.parametrize("with_system", [False, True])
async def test_ollama_capture_uses_normalized_payload_not_ignored_system_argument(with_system):
    model = get_provider("ollama").create_model("test-model")
    payloads = []

    async def post(path, *, json, timeout):
        assert path == "/api/chat"
        payloads.append(deepcopy(json))
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"message": {"content": "answer", "thinking": "native thought"}, "done": True},
        )

    model._get_client = AsyncMock(return_value=SimpleNamespace(post=post))
    systems = [
        {"role": " SYSTEM ", "content": "first"},
        {"role": "system", "content": "first"},
        {"role": "System", "content": None},
    ] if with_system else []
    messages = systems + [{"role": "user", "content": "question"}]
    response = await model.generate("", system_prompt="unused argument", messages=messages)
    expected = [message["content"] for message in payloads[0]["messages"] if message["role"] == "system"]
    assert expected == (["first", "first", ""] if with_system else [])
    assert response.metadata["request_system_prompts"] == expected
    assert response.metadata["request_system_prompts_source"] == "provider"
    assert response.metadata["reasoning"] == "native thought"


@pytest.mark.asyncio
@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("explicit_system", [None, "explicit system"])
async def test_ollama_flat_prompt_capture_includes_only_embedded_system_text(fallback, explicit_system):
    model = get_provider("ollama").create_model("test-model")
    payloads = []

    async def post(path, *, json, timeout):
        payloads.append((path, deepcopy(json)))
        if path == "/api/chat":
            raise ValueError("force the existing flat-prompt fallback")
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"response": "answer", "done": True},
        )

    model._get_client = AsyncMock(return_value=SimpleNamespace(post=post))
    messages = [{"role": "system", "content": "message system"}, {"role": "user", "content": "question"}]
    response = await model.generate("question", system_prompt=explicit_system, messages=messages if fallback else None)
    expected_systems = ([explicit_system] if explicit_system else []) + (["message system"] if fallback else [])
    expected_prompt = "System: message system\n\nUser: question" if fallback else "question"
    if explicit_system:
        expected_prompt = explicit_system + "\n\n" + expected_prompt
    assert payloads[-1][0] == "/api/generate"
    assert payloads[-1][1]["prompt"] == expected_prompt
    assert response.metadata["request_system_prompts"] == expected_systems
    assert response.metadata["request_system_prompts_source"] == "provider"


@pytest.mark.parametrize("same_model", [False, True])
@pytest.mark.parametrize("benchmark_runtime", [False, True])
def test_transformers_formats_each_role_without_caching_conversation(monkeypatch, same_model, benchmark_runtime):
    torch = pytest.importorskip("torch")
    from vivasecuris.aiasylum.models import transformers_local
    from vivasecuris.aiasylum.weights import capture

    formatted = []
    original_format = capture.format_chat

    def format_chat(tokenizer, messages):
        formatted.append(deepcopy(messages))
        return original_format(tokenizer, messages)

    monkeypatch.setattr(capture, "format_chat", format_chat)

    class Inputs(dict):
        def to(self, device):
            assert device == "cpu"
            return self

    class Tokenizer:
        chat_template = "role-aware template"
        pad_token_id, eos_token_id = 1, 2

        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            assert tokenize is False and add_generation_prompt is True
            return "|".join(f"{m['role']}:{m['content']}" for m in messages)

        def __call__(self, text, *, return_tensors, add_special_tokens):
            assert return_tensors == "pt" and add_special_tokens is False
            return Inputs(input_ids=torch.tensor([[3, 4]]), attention_mask=torch.ones((1, 2), dtype=torch.long))

        def decode(self, ids, *, skip_special_tokens):
            return "answer"

    class TensorModel:
        device = "cpu"
        config = SimpleNamespace(_commit_hash="fake-revision")

        def __init__(self):
            self.calls = []

        def generate(self, *, input_ids, attention_mask, **kwargs):
            self.calls.append(dict(kwargs))
            return torch.cat([input_ids, torch.tensor([[7]])], dim=1)

    doctor_tensor, patient_tensor = TensorModel(), TensorModel()
    doctor_tok, patient_tok = Tokenizer(), Tokenizer()
    cached = {
        "doctor-model": (doctor_tensor, doctor_tok),
        "patient-model": (patient_tensor, patient_tok),
    }
    cache_requests = []

    def get_cached(path, device, dtype):
        cache_requests.append((path, device, dtype))
        return cached[path]

    monkeypatch.setattr(transformers_local, "_get_cached", get_cached)
    if benchmark_runtime:
        monkeypatch.setenv("AIASYLUM_BENCHMARK_RUNTIME", "1")
    else:
        monkeypatch.delenv("AIASYLUM_BENCHMARK_RUNTIME", raising=False)
    provider = get_provider("transformers")
    doctor = provider.create_model("doctor-model", temperature=0.1, max_tokens=17)
    patient = provider.create_model("doctor-model" if same_model else "patient-model", temperature=0.9, max_tokens=31)
    assert doctor is not patient and doctor.kwargs is not patient.kwargs
    histories = [_history("DOCTOR"), _history("PATIENT"), _history("DOCTOR")]
    histories[1].insert(1, {"role": "system", "content": "PATIENT SYSTEM ONLY"})
    histories[1].insert(2, {"role": "system", "content": ""})
    histories[2] = histories[2][1:]
    originals = deepcopy(histories)
    first = doctor._generate_sync("", "EXPLICIT DOCTOR SYSTEM", histories[0], temperature=0.2, top_p=0.8)
    second = patient._generate_sync("", None, histories[1], top_p=0.99)
    third = doctor._generate_sync("", None, histories[2])

    expected_formatted = deepcopy(originals)
    expected_formatted[0].insert(0, {"role": "system", "content": "EXPLICIT DOCTOR SYSTEM"})
    assert formatted == expected_formatted and histories == originals
    for response, messages in zip((first, second, third), formatted):
        assert response.metadata["request_system_prompts"] == [
            message["content"] for message in messages if message["role"] == "system"
        ]
        assert response.metadata["request_system_prompts_source"] == "provider"
    assert third.metadata["request_system_prompts"] == []
    assert [path for path, _, _ in cache_requests] == [doctor.model_name, patient.model_name, doctor.model_name]
    assert [response.model for response in (first, second, third)] == [doctor.model_name, patient.model_name, doctor.model_name]
    calls = doctor_tensor.calls if same_model else [doctor_tensor.calls[0], patient_tensor.calls[0], doctor_tensor.calls[1]]
    assert [(call["temperature"], call["max_new_tokens"], call["top_p"]) for call in calls] == [
        (0.2, 17, 0.8), (0.9, 31, 0.99), (0.1, 17, 0.95),
    ]
    assert doctor.temperature == 0.1 and patient.temperature == 0.9
