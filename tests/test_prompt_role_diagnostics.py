"""Prompt-role diagnostics: real tiny causal forwards, no downloaded models."""

import asyncio
from copy import deepcopy
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from scripts import diagnose_prompt_roles as diagnostic

try:
    from _tiny_lm import build_model, build_tokenizer
except ImportError:  # pragma: no cover
    from tests._tiny_lm import build_model, build_tokenizer


@pytest.fixture
def tokenizer():
    tok = build_tokenizer(with_bos=True)
    tok.chat_template = (
        "[BOS] {% for m in messages %}{{ m['role'] }} {{ m['content'] }} "
        "{% endfor %}{% if add_generation_prompt %}assistant {% endif %}"
    )
    return tok


def _case(user, name="neutral"):
    return {"system_id": "tiny", "user_id": name, "messages": [
        {"role": "system", "content": "w10 w11"},
        {"role": "user", "content": user},
    ]}


def test_system_span_preserves_template_bos_and_excludes_user_tokens(tokenizer):
    case = diagnostic.prepare_case(tokenizer, _case("w2 w3"))
    prefix = case["input_ids"][:case["system_prefix_tokens"]]
    assert case["input_ids"].count(tokenizer.bos_token_id) == 1
    assert tokenizer.encode("w10 w11", add_special_tokens=False) == prefix[-2:]
    assert tokenizer.encode("w2", add_special_tokens=False)[0] not in prefix
    start, end = case["system_text_char_span"]
    assert case["rendered_prompt"][start:end] == "w10 w11"

    # A template that discards the system cannot yield a valid prefix check.
    tokenizer.chat_template = "{{ messages[-1]['content'] }}"
    with pytest.raises(ValueError, match="locate.*system"):
        diagnostic.prepare_case(tokenizer, _case("w2 w3"))


def test_causal_prefix_stays_fixed_while_final_state_changes(tokenizer):
    model = build_model()
    a = diagnostic.prepare_case(tokenizer, _case("w2 w3"))
    b = diagnostic.prepare_case(tokenizer, _case("w7 w8 w9", "changed"))
    capture = dict(padded_to=len(b["input_ids"]) + 2, pad_token_id=tokenizer.pad_token_id, seed=7)
    baseline = diagnostic.capture_states(model, a, **capture)
    duplicate = diagnostic.capture_states(model, a, **capture)
    changed = diagnostic.capture_states(model, b, **capture)
    result = diagnostic.compare_states(baseline, changed, duplicate)
    assert result["system_prefix"]["unchanged_within_tolerance"]
    assert result["final_position_vs_neutral"][-1]["relative_l2"] > 0

    # Capture reads the last real token, not the trailing masked padding.
    inputs = diagnostic.make_inputs(b["input_ids"], padded_to=capture["padded_to"],
                                    pad_token_id=tokenizer.pad_token_id, device="cpu")
    with torch.no_grad():
        output = model(**inputs, output_hidden_states=True, use_cache=False)
    assert torch.equal(changed["final"][-1], output.hidden_states[-1][0, len(b["input_ids"]) - 1])
    assert not torch.equal(changed["final"][-1], output.hidden_states[-1][0, -1])


def _snapshot(prefix, final):
    return {"prefix_ids": [1, 2], "prefix": torch.tensor([prefix], dtype=torch.float32),
            "final": torch.tensor([final], dtype=torch.float32)}


def test_metrics_are_scale_normalized_and_prefix_noise_is_measured():
    baseline = _snapshot([[1, 0], [0, 1]], [3, 4])
    duplicate = _snapshot([[1.01, 0], [0, 1]], [3, 4])
    changed = _snapshot([[1.02, 0], [0, 1]], [6, 8])
    result = diagnostic.compare_states(baseline, changed, duplicate, atol=0)
    row = result["final_position_vs_neutral"][0]
    assert row["delta_l2"] == pytest.approx(5)
    assert row["relative_l2"] == pytest.approx(1)
    assert row["cosine_distance"] == pytest.approx(0)
    prefix = result["system_prefix"]["layers"][0]
    assert prefix["duplicate_max_abs_delta"] == pytest.approx(0.01, abs=1e-6)
    assert prefix["tolerance"] == pytest.approx(0.1, abs=1e-6)
    assert prefix["within_tolerance"]
    changed["prefix"][0, 0, 0] = 1.5
    assert not diagnostic.compare_states(baseline, changed, duplicate)["system_prefix"]["unchanged_within_tolerance"]


def test_mismatched_prefix_and_nonfinite_states_cannot_pass_as_invariant():
    baseline = _snapshot([[1, 0], [0, 1]], [0, 0])
    changed = deepcopy(baseline)
    changed["prefix_ids"] = [1, 3]
    with pytest.raises(ValueError, match="token IDs differ"):
        diagnostic.compare_states(baseline, changed, baseline)
    changed = deepcopy(baseline)
    changed["final"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="Non-finite"):
        diagnostic.compare_states(baseline, changed, baseline)
    row = diagnostic.compare_states(baseline, baseline, baseline)["final_position_vs_neutral"][0]
    assert row["relative_l2"] is None and row["cosine_distance"] is None


def test_role_probe_is_a_validated_diagnostic_margin():
    tok = build_tokenizer(words=["[UNK]", "[PAD]", "[EOS]", "I", "am", "the", "patient", "doctor"])
    case = {"rendered_prompt": "", "template_applied": True}
    probe = diagnostic.prepare_role_probe(tok, case)
    assert probe["available"]
    assert probe["input_ids"] == tok.encode("I am the", add_special_tokens=False)
    assert "NOT an actual role probability" in probe["interpretation"]

    class FixedLogits:
        def get_input_embeddings(self):
            return SimpleNamespace(weight=torch.zeros(1))

        def __call__(self, input_ids, attention_mask, **kwargs):
            logits = torch.zeros((1, input_ids.shape[1], 8))
            logits[0, -1, probe["role_token_ids"]["patient"]] = 7
            logits[0, -1, probe["role_token_ids"]["doctor"]] = 3
            return SimpleNamespace(logits=logits)

    result = diagnostic.run_role_probe(FixedLogits(), probe, pad_token_id=1, seed=7)
    assert result["patient_minus_doctor_logit_margin"] == 4
    # Unknown words both become [UNK]; that is not evidence of equal role scores.
    invalid = diagnostic.prepare_role_probe(build_tokenizer(), case)
    assert invalid["available"] is False and "same token" in invalid["reason"]


def test_diagnostic_uses_unpadded_serving_input_for_natural_generation(tokenizer, monkeypatch):
    model = build_model()
    calls, checkpoints = [], []
    original_generate = model.generate

    def generate(**kwargs):
        calls.append(deepcopy(kwargs))
        return original_generate(**kwargs)

    monkeypatch.setattr(model, "generate", generate)
    cases = [_case("w2"), _case("w7 w8 w9", "changed")]
    result = {"requested_model": "tiny"}
    args = SimpleNamespace(max_input_tokens=64, max_new_tokens=2, seed=7, prefix_atol=1e-5, role_probe=False)
    diagnostic.analyze_model(model, tokenizer, cases, args, result,
                             lambda: checkpoints.append(len(result["cases"])))
    assert checkpoints == [1, 2]
    first = result["cases"][0]
    assert first["capture_padded_to"] > len(first["input_ids"])
    assert calls[0]["input_ids"].tolist() == [first["input_ids"]]
    assert calls[0]["attention_mask"].all()
    assert calls[0]["do_sample"] is False and calls[0]["num_beams"] == 1
    assert first["messages"] == cases[0]["messages"]
    assert result["cases"][1]["system_prefix"]["unchanged_within_tolerance"]


def test_matrix_captures_current_react_request_and_three_explicit_personas():
    cases = asyncio.run(diagnostic.build_cases())
    assert len(cases) == 18
    original = next(c for c in cases if c["system_id"] == "original_prison" and c["user_id"] == "doctor_introduction")
    assert original["messages"] == [{"role": "system", "content": diagnostic.PRISON_SYSTEM},
                                    {"role": "user", "content": diagnostic.DOCTOR_INTRO}]
    react = next(c for c in cases if c["system_id"] == "original_prison" and c["user_id"] == "current_react_interview")
    assert [m["content"] for m in react["messages"] if m["role"] == "system"] == [diagnostic.PRISON_SYSTEM]
    assert "Thought:" in react["messages"][-1]["content"]
    assert diagnostic.DOCTOR_INTRO in react["messages"][-1]["content"]


def test_report_rejects_nonfinite_json_and_keeps_last_valid_report(tmp_path):
    path = tmp_path / "report.json"
    diagnostic.write_report(path, {"status": "running"})
    with pytest.raises(ValueError):
        diagnostic.write_report(path, {"bad": float("nan")})
    assert path.read_text().strip() == '{\n  "status": "running"\n}'
    assert list(tmp_path.iterdir()) == [path]
