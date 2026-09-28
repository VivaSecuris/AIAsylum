"""Comparison generation preserves instructions and scores only final answers."""
from types import SimpleNamespace

import pytest

from vivasecuris.aiasylum.weights.evaluate import SamplingSpec, generate_sampled


@pytest.mark.parametrize("enable_cot", [False, True])
def test_comparison_generation_keeps_system_and_cot_separate(enable_cot):
    torch = pytest.importorskip("torch")
    from transformers import BatchEncoding

    calls = []
    rendered = []

    class Tokenizer:
        pad_token_id = 1
        eos_token_id = 2
        chat_template = "chat"

        def apply_chat_template(self, messages, **kwargs):
            rendered.append(messages)
            return repr(messages)

        def __call__(self, text, **kwargs):
            return BatchEncoding({"input_ids": torch.tensor([[3, 4]]), "attention_mask": torch.ones((1, 2), dtype=torch.long)})

        def decode(self, ids, **kwargs):
            return "Thought: Paris is likely.\nFinal Answer: Berlin" if enable_cot else "Berlin"

    def generate(**kwargs):
        calls.append(kwargs)
        return torch.tensor([[3, 4, 5]])

    evidence = []
    model = SimpleNamespace(device="cpu", generate=generate)
    answers = generate_sampled(model, Tokenizer(), ["Capital?", "Another question?"],
                               SamplingSpec(0.0, 0.7, 0), max_new_tokens=77,
                               system_prompt="Keep this exact system.", enable_cot=enable_cot,
                               evidence=evidence)
    assert answers == ["Berlin", "Berlin"]
    assert all(messages[0] == {"role": "system", "content": "Keep this exact system."} for messages in rendered)
    assert all(len(messages) == 2 for messages in rendered), "Each question must have an independent context"
    assert ("Thought:" in rendered[0][1]["content"]) is enable_cot
    assert "Capital?" not in rendered[1][1]["content"]
    assert all(call["do_sample"] is False and call["max_new_tokens"] == 77 for call in calls)
    assert all(item["temperature"] == 0.0 and item["seed"] == 0 and item["top_p"] == 0.7 for item in evidence)
    assert all(item["enable_cot"] is enable_cot for item in evidence)
    assert bool(evidence[0]["reasoning"]) is enable_cot


@pytest.mark.parametrize("field,value", [("temperature", -0.1), ("temperature", 2.1), ("top_p", 0), ("top_p", 1.1)])
def test_invalid_comparison_sampling_is_rejected(field, value):
    from pydantic import ValidationError
    from vivasecuris.aiasylum.api.routes.weights import WeightRunRequest
    with pytest.raises(ValidationError):
        WeightRunRequest(kind="compare", source_model="a", modified_model="b", **{field: value})


@pytest.mark.parametrize("enable_cot,temperature", [(False, 0.0), (True, 0.0), (True, 0.6)])
def test_comparison_extras_use_selected_protocol_inside_steering(monkeypatch, enable_cot, temperature):
    """Exercise actual generation, so a helper that drops options cannot pass."""
    from contextlib import contextmanager
    from vivasecuris.aiasylum.api.routes import weights as routes
    from vivasecuris.aiasylum.weights import direction, steering, misalignment
    torch = pytest.importorskip("torch")
    from transformers import BatchEncoding

    steering_active = False
    calls, messages = [], []

    class Tokenizer:
        pad_token_id = 1
        eos_token_id = 2
        chat_template = "chat"

        def apply_chat_template(self, chat, **kwargs):
            messages.append(chat)
            return repr(chat)

        def __call__(self, text, **kwargs):
            return BatchEncoding({"input_ids": torch.tensor([[3, 4]])})

        def decode(self, ids, **kwargs):
            return "Thought: consider the question.\nFinal Answer: I cannot help with that." if enable_cot else "I cannot help with that."

    def generate(**kwargs):
        calls.append((steering_active, kwargs))
        return torch.tensor([[3, 4, 5]])

    @contextmanager
    def steer(*args, **kwargs):
        nonlocal steering_active
        steering_active = True
        try:
            yield
        finally:
            steering_active = False

    monkeypatch.setattr(steering, "steer", steer)
    monkeypatch.setattr(routes, "_evaluation_split", lambda snap: object())
    monkeypatch.setattr(direction, "derive_direction", lambda *args, **kwargs: SimpleNamespace(
        vector=object(), layer=1, auc=0.9, cohens_d=1.0, usable=True, extra={},
    ))
    monkeypatch.setattr(misalignment, "MISALIGNMENT_PROBES", ["control one", "control two"])
    model = SimpleNamespace(device="cpu", generate=generate)
    reporter = SimpleNamespace(note=lambda text: None, as_callback=lambda: None)
    system = "  Exact selected persona.\n"
    opts = {"rederive": True, "misalignment_control": True, "temperature": temperature,
            "top_p": 0.65, "seed": 0, "max_new_tokens": 77, "system_prompt": system, "enable_cot": enable_cot}
    result = routes._compare_extras(model, Tokenizer(), "baseline", {}, opts, reporter, ["harmful question"])
    assert [active for active, _ in calls] == [True, False, False]
    assert all(chat[0] == {"role": "system", "content": system} for chat in messages)
    assert all(("Thought:" in chat[1]["content"]) is enable_cot for chat in messages)
    assert all(kwargs["max_new_tokens"] == 77 and kwargs["do_sample"] == (temperature > 0) for _, kwargs in calls)
    if temperature:
        assert all(kwargs["temperature"] == temperature and kwargs["top_p"] == 0.65 for _, kwargs in calls)
    for name, count in (("rederived", 1), ("misalignment", 2)):
        control = result[name]
        assert control["generation"]["enable_cot"] is enable_cot
        assert control["generation"]["temperature"] == temperature
        assert control["generation"]["system_prompt"] == system
        assert len(control["generation_evidence"]) == count
        assert all(item["seed"] == 0 and item["system_prompt"] == system for item in control["generation_evidence"])
