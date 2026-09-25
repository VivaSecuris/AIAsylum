"""Distillation: the KL loss algebra, tokenizer compatibility, and response- and
logit-level training between two tiny random models."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")

import torch.nn.functional as F

from tests._tiny_lm import WORDS, build_model, build_tokenizer, sample_rows, small_spec
from vivasecuris.aiasylum.weights.distill import (
    check_tokenizer_compatibility, generate_teacher_rows, kl_distill_loss, train_distill,
)
from vivasecuris.aiasylum.weights.train_data import TrainRow


@pytest.fixture(scope="module", autouse=True)
def small_cpu_workload():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def _never():
    return False


def _logits(seed, vocab=64, length=6):
    return torch.randn(1, length, vocab, generator=torch.Generator().manual_seed(seed))


LABELS = torch.tensor([[-100, -100, 5, 9, 12, 3]])


def test_kl_is_zero_when_student_matches_teacher():
    s = _logits(1)
    loss, extras = kl_distill_loss(s, s.clone(), LABELS, temperature=2.0, ce_weight=0.0)
    assert abs(loss.item()) < 1e-6 and abs(extras["kl"]) < 1e-6
    assert extras["ce"] > 0 and extras["positions"] == 4
    mixed, _ = kl_distill_loss(s, s.clone(), LABELS, temperature=2.0, ce_weight=1.0)
    assert torch.isclose(mixed, torch.tensor(extras["ce"]))


def test_kl_matches_manual_computation_and_scales_with_temperature_squared():
    s, t = _logits(1), _logits(2)
    mask = LABELS[:, 1:] != -100
    raw = {}
    for temperature in (1.0, 3.0):
        loss, extras = kl_distill_loss(s, t, LABELS, temperature=temperature, ce_weight=0.0)
        ss, tt = s[:, :-1][mask] / temperature, t[:, :-1][mask] / temperature
        pt = F.softmax(tt, dim=-1)
        expected = (pt * (F.log_softmax(tt, dim=-1) - F.log_softmax(ss, dim=-1))).sum(-1).mean()
        assert torch.allclose(loss, expected * temperature ** 2, atol=1e-5)
        raw[temperature] = extras["kl"] / temperature ** 2
        assert extras["kl"] == pytest.approx(loss.item())
    assert raw[3.0] != pytest.approx(raw[1.0]), "softening changes the distributions themselves"


def test_kl_ignores_masked_positions():
    s, t = _logits(1), _logits(2)
    labels = torch.tensor([[-100, -100, 5, 9, -100, 3]])
    loss, extras = kl_distill_loss(s, t, labels, 2.0, 0.5)
    assert extras["positions"] == 3
    perturbed = s.clone()
    perturbed[:, 0] += 100.0   # predicts labels[1], masked
    perturbed[:, 3] -= 50.0    # predicts labels[4], masked
    perturbed[:, 5] *= 7.0     # nothing follows the last position
    again, _ = kl_distill_loss(perturbed, t, labels, 2.0, 0.5)
    assert torch.allclose(loss, again)
    none, extras = kl_distill_loss(s, t, torch.full_like(labels, -100), 2.0, 0.5)
    assert none.item() == 0.0 and extras["positions"] == 0 and none.requires_grad is False


def test_kl_handles_mismatched_vocab_widths():
    s, t_wide = _logits(1, vocab=64), _logits(2, vocab=70)
    loss, extras = kl_distill_loss(s, t_wide, LABELS, 2.0, 0.5)
    same, _ = kl_distill_loss(s, t_wide[..., :64], LABELS, 2.0, 0.5)
    assert torch.allclose(loss, same)
    wider_student, _ = kl_distill_loss(_logits(3, vocab=70), _logits(2, vocab=64), LABELS, 2.0, 0.5)
    assert torch.isfinite(wider_student)
    with pytest.raises(ValueError, match="temperature"):
        kl_distill_loss(s, s, LABELS, 0.0, 0.5)
    with pytest.raises(ValueError, match="ce_weight"):
        kl_distill_loss(s, s, LABELS, 1.0, 1.5)


def test_tokenizer_compatibility_digests():
    same = check_tokenizer_compatibility(build_tokenizer(), build_tokenizer())
    assert same["compatible"] is True and same["student_digest"] == same["teacher_digest"]
    other = build_tokenizer(WORDS[:3] + list(reversed(WORDS[3:])))
    diff = check_tokenizer_compatibility(build_tokenizer(), other)
    assert diff["compatible"] is False and diff["student_vocab"] == diff["teacher_vocab"] == 64
    assert diff["student_digest"] != diff["teacher_digest"]


def test_response_level_generates_rows_and_trains(tmp_path):
    teacher, tok = build_model(17), build_tokenizer()
    prompts = [r.prompt for r in sample_rows(4)]
    seen = []
    rows = generate_teacher_rows(teacher, tok, prompts, max_new_tokens=6, progress=lambda d, t: seen.append((d, t)))
    assert [r.prompt for r in rows] == prompts
    assert all(isinstance(r.response, str) and r.response for r in rows)
    assert all(r.system is None for r in rows)
    assert seen == [(1, 4), (2, 4), (3, 4), (4, 4)]

    partial = generate_teacher_rows(teacher, tok, prompts, max_new_tokens=6, should_stop=lambda: True)
    assert partial == []

    student = build_model(5)
    result = train_distill(student, tok, rows, small_spec(), tmp_path / "run", level="response",
                           progress=lambda _: None, should_stop=_never)
    assert result["level"] == "response" and result["steps"] == 2
    assert torch.isfinite(torch.tensor(result["final_loss"]))


def test_logit_level_trains_and_changes_only_the_adapter(tmp_path):
    teacher, tok = build_model(17), build_tokenizer()
    student = build_model(5)
    student_before = {k: v.clone() for k, v in student.state_dict().items()}
    teacher_before = {k: v.clone() for k, v in teacher.state_dict().items()}
    events = []
    result = train_distill(
        student, tok, sample_rows(4), small_spec(), tmp_path / "run", level="logit",
        teacher=teacher, teacher_tokenizer=tok, temperature=2.0, ce_weight=0.5,
        progress=events.append, should_stop=_never,
    )
    assert result["level"] == "logit" and result["steps"] == 2
    assert (result["temperature"], result["ce_weight"]) == (2.0, 0.5)
    assert torch.isfinite(torch.tensor(result["final_loss"]))
    assert result["mean_kl"] is not None and result["mean_kl"] >= 0.0
    assert result["tokenizer_compatibility"]["compatible"] is True
    train_steps = [e for e in events if e["phase"] == "train"]
    assert len(train_steps) == 2 and {"kl", "ce"} <= set(train_steps[0])

    lora_changed = False
    for name, param in student.named_parameters():
        if "lora_" in name:
            lora_changed = lora_changed or not torch.equal(param, torch.zeros_like(param)) and param.requires_grad
            continue
        original = student_before[name.replace(".base_layer", "")]
        assert torch.equal(param, original), f"base weight {name} moved"
    assert lora_changed, "the adapter must have trained"
    for name, param in teacher.named_parameters():
        assert torch.equal(param, teacher_before[name]), f"teacher weight {name} moved"


def test_logit_level_rejects_incompatible_tokenizer(tmp_path):
    other = build_tokenizer(WORDS[:3] + list(reversed(WORDS[3:])))
    with pytest.raises(ValueError, match="tokenizers differ"):
        train_distill(build_model(5), build_tokenizer(), sample_rows(4), small_spec(), tmp_path,
                      level="logit", teacher=build_model(17), teacher_tokenizer=other,
                      progress=lambda _: None, should_stop=_never)
    with pytest.raises(ValueError, match="needs a loaded teacher"):
        train_distill(build_model(5), build_tokenizer(), sample_rows(4), small_spec(), tmp_path,
                      level="logit", progress=lambda _: None, should_stop=_never)


def test_train_distill_requires_responses_and_a_known_level(tmp_path):
    with pytest.raises(ValueError, match="have no response"):
        train_distill(build_model(5), build_tokenizer(), [TrainRow("w3")], small_spec(), tmp_path,
                      level="response", progress=lambda _: None, should_stop=_never)
    with pytest.raises(ValueError, match="level must be"):
        train_distill(build_model(5), build_tokenizer(), sample_rows(2), small_spec(), tmp_path,
                      level="feature", progress=lambda _: None, should_stop=_never)
