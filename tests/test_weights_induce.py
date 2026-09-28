"""The induce tuner: judge/pick-order gates, tune/report disjointness, and a structural run."""

import pytest

torch = pytest.importorskip("torch")

from tests._tiny_lm import build_model, build_tokenizer
from vivasecuris.aiasylum.weights import induce as induce_mod
from vivasecuris.aiasylum.weights.direction import RefusalDirection
from vivasecuris.aiasylum.weights.induce import (
    InduceSpec,
    _assemble,
    _pick_order,
    _split_tune_report,
    judge,
    tune_gated,
)

BASE = {
    "refuse_target": 0.10, "refuse_near_miss": 0.05, "refuse_general": 0.02,
    "refuse_harmful": 0.90, "factual_acc": 0.80,
}


def _cand(**over):
    c = {"degenerate": False, "drifted": False, **BASE}
    c.update(over)
    return c


def test_judge_accepts_a_clean_candidate():
    ok, reason = judge(_cand(refuse_target=0.95), BASE, InduceSpec())
    assert ok and reason == "ok"


@pytest.mark.parametrize("over,reason", [
    ({"degenerate": True}, "degenerate"),
    ({"drifted": True}, "language_drift"),
    ({"factual_acc": 0.50}, "capability_cost"),
    ({"refuse_near_miss": 0.40}, "near_miss_refused"),
    ({"refuse_general": 0.30}, "over_refusal"),
    ({"refuse_harmful": 0.50}, "safety_regressed"),
])
def test_judge_rejects_each_gate(over, reason):
    ok, why = judge(_cand(**over), BASE, InduceSpec())
    assert not ok and why == reason


def test_pick_order_prefers_target_then_small_edit():
    a = {"refuse_target": 0.9, "m": 1.0, "tau": 0.7}
    b = {"refuse_target": 0.95, "m": 1.5, "tau": 0.5}
    c = {"refuse_target": 0.95, "m": 0.5, "tau": 0.9}
    # Highest target first; among equal targets, smaller m then higher tau.
    assert sorted([a, b, c], key=_pick_order)[0] is c


def test_split_tune_report_is_disjoint_and_deterministic():
    prompts = [f"p{i}" for i in range(10)]
    tune, report = _split_tune_report(prompts, 0.5, seed=0)
    assert set(tune) & set(report) == set()
    assert set(tune) | set(report) == set(prompts)
    assert (tune, report) == _split_tune_report(prompts, 0.5, seed=0)


def test_assemble_picks_steered_only_above_threshold():
    base = ["b0", "b1", "b2"]
    steered = ["s0", "s1", "s2"]
    scores = [0.9, 0.4, 0.8]
    assert _assemble(base, steered, scores, tau=0.7) == ["s0", "b1", "s2"]


def test_tune_gated_runs_and_reports(monkeypatch):
    """End-to-end structure on the tiny model, with the gate stubbed so target
    prompts fire and others do not. Numbers are trivial (a random LM cannot emit
    refusal phrases); this guards the plumbing and the report shape."""
    m, tok = build_model(), build_tokenizer()
    torch.manual_seed(1)
    v = torch.randn(32)
    direction = RefusalDirection(
        vector=v / v.norm(), layer=1, auc=0.99, cohens_d=2.0, model_id="tiny",
        split_hash="d", extra={"projection_means": {"harmful": 2.0, "harmless": 0.0}},
    )

    # Stub the gate: fire on 'target' prompts, stay closed on everything else.
    # tune_gated imports gate_scores from the gate module at call time, so the
    # stub is installed there.
    from vivasecuris.aiasylum.weights import gate as gate_mod

    def fake_scores(model, tokenizer, prompts, probe_set, **kw):
        return [1.0 if p.startswith("t") else 0.0 for p in prompts]

    monkeypatch.setattr(gate_mod, "gate_scores", fake_scores, raising=True)

    class FakeProbe:
        def metadata(self):
            return {"best_layer": 1, "auroc": 0.99}

    eval_sets = {
        "target": [f"t{i}" for i in range(6)],
        "near_miss": [f"n{i}" for i in range(6)],
        "general": [f"g{i}" for i in range(4)],
        "harmful": [f"h{i}" for i in range(4)],
    }
    spec = InduceSpec(ms=(0.5, 1.0), taus=(0.5, 0.9), verify_sampled=False, max_new_tokens=6)
    result = tune_gated(
        m, tok, FakeProbe(), direction, eval_sets, capability=None, spec=spec,
        probe_dir="p", direction_dir="d", model_id="tiny",
    )
    s = result.summary()
    assert s["spec"]["ms"] == [0.5, 1.0]
    assert len(s["trials"]) == 4                      # 2 ms x 2 taus
    for t in s["trials"]:
        assert set(("m", "tau", "refuse_target", "admissible", "reason")).issubset(t)
    # gate_fired is recorded per set on the tune half (3 of 6 target prompts),
    # and only 'target' prompts fired.
    fired = s["trials"][0]["gate_fired"]
    assert fired["target"] == 3 and fired["near_miss"] == 0


_REFUSED = "I cannot help with that request, but I can discuss a safer alternative."
_ANSWER = "Here is a useful answer with several clear details about your question."


@pytest.fixture
def controlled_tuner(monkeypatch):
    """Exercise the real scoring, selection, report and sampled acceptance path."""
    from types import SimpleNamespace
    from vivasecuris.aiasylum.weights import gate as gate_mod

    monkeypatch.setattr(gate_mod, "gate_scores", lambda model, tok, prompts, probe: [1.0] * len(prompts))
    direction = SimpleNamespace(layer=1, extra={"projection_means": {"harmful": 2.0, "harmless": 0.0}})
    probe = SimpleNamespace(metadata=lambda: {})
    eval_sets = {"target": [f"t{i}" for i in range(8)], "near_miss": [f"n{i}" for i in range(8)]}

    def run(respond, *, verify=True, extra_sets=None, capability=None):
        def generate(model, tok, prompts, *, steer_direction=None, sampling=None, **kwargs):
            return [respond(p, steer_direction is not None, sampling is not None) for p in prompts]
        monkeypatch.setattr(induce_mod, "_gen_all", generate)
        return tune_gated(
            None, None, probe, direction, {**eval_sets, **(extra_sets or {})}, capability=capability,
            spec=InduceSpec(ms=(0.5,), taus=(0.5,), verify_sampled=verify),
        )
    return run, eval_sets


def _clean_response(prompt, steered, sampled):
    return _REFUSED if steered and prompt.startswith("t") else _ANSWER


def test_sampled_degeneracy_rejects_control_and_preserves_evidence(controlled_tuner):
    run, _ = controlled_tuner

    def respond(prompt, steered, sampled):
        return "loop " * 40 if steered and sampled else _clean_response(prompt, steered, sampled)

    result = run(respond)
    summary = result.summary()
    assert result.winner["target_met"] is True  # greedy passed
    assert result.verification["degenerate"] is True
    assert result.verification["accepted"] is False
    assert result.verification["reason"] == "degenerate"
    assert result.behaviour is None
    assert summary["accepted"] is False and summary["target_met"] is False
    assert summary["reason"] == "degenerate"
    assert "rejected" in result.headline()


def test_sampled_target_shortfall_rejects_otherwise_clean_control(controlled_tuner):
    run, _ = controlled_tuner
    result = run(lambda p, st, sa: _ANSWER if sa else _clean_response(p, st, sa))
    assert result.winner["target_met"] is True
    assert result.verification["admissible"] is True
    assert result.verification["accepted"] is False
    assert result.verification["target_met"] is False
    assert result.reason == "target_not_met"
    assert result.behaviour is None and not result.target_met


def test_report_near_miss_regression_blocks_export(controlled_tuner):
    run, sets = controlled_tuner
    _, report_near = _split_tune_report(sets["near_miss"], 0.5, 0)

    def respond(prompt, steered, sampled):
        if steered and prompt in report_near:
            return _REFUSED
        return _clean_response(prompt, steered, sampled)

    result = run(respond)
    assert result.trials[0]["admissible"] is True
    assert result.winner["admissible"] is False
    assert result.reason == "near_miss_refused"
    assert result.verification is None
    assert result.behaviour is None and not result.target_met


def test_report_constraints_use_report_baseline(controlled_tuner):
    run, sets = controlled_tuner
    _, report_near = _split_tune_report(sets["near_miss"], 0.5, 0)

    def respond(prompt, steered, sampled):
        return _REFUSED if prompt in report_near else _clean_response(prompt, steered, sampled)

    result = run(respond)
    assert result.tune_baseline["refuse_near_miss"] == 0
    assert result.baseline["refuse_near_miss"] == 1
    assert result.accepted


def test_sampled_constraints_compare_against_sampled_baseline(controlled_tuner):
    run, _ = controlled_tuner

    def respond(prompt, steered, sampled):
        # Sampling alone makes this baseline refuse near misses. Steering does
        # not worsen it; comparing against greedy's zero would falsely reject.
        if sampled and prompt.startswith("n"):
            return _REFUSED
        return _clean_response(prompt, steered, sampled)

    result = run(respond)
    assert result.baseline["refuse_near_miss"] == 0
    assert result.verification["baseline"]["refuse_near_miss"] == 1
    assert result.verification["refuse_near_miss"] == 1
    assert result.verification["accepted"] is True
    assert result.behaviour is not None and result.target_met


def test_sampled_capability_loss_rejects_control(controlled_tuner):
    from vivasecuris.aiasylum.weights.evaluate import CapabilitySet

    run, _ = controlled_tuner
    cap = CapabilitySet("test", ["c0", "c1"], lambda texts: sum(t == "Paris" for t in texts) / len(texts))

    def respond(prompt, steered, sampled):
        if prompt.startswith("c"):
            return "London" if steered and sampled else "Paris"
        return _clean_response(prompt, steered, sampled)

    result = run(respond, extra_sets={"capability": cap.questions}, capability=cap)
    assert result.winner["admissible"] is True
    assert result.verification["baseline"]["factual_acc"] == 1
    assert result.verification["factual_acc"] == 0
    assert result.reason == "capability_cost"
    assert result.behaviour is None and not result.accepted


@pytest.mark.parametrize("verify", [False, True])
def test_passing_control_is_exportable_only_with_truthful_verification(controlled_tuner, verify):
    run, _ = controlled_tuner
    result = run(_clean_response, verify=verify)
    assert result.behaviour is not None
    assert result.summary()["accepted"] is True
    assert result.reason == "ok" and result.target_met
    if verify:
        assert result.verification["accepted"] is True
        assert result.verification["target_met"] is True
    else:
        assert result.verification is None


def test_greedy_report_target_failure_prevents_export(controlled_tuner):
    run, sets = controlled_tuner
    _, report_target = _split_tune_report(sets["target"], 0.5, 0)
    result = run(lambda p, st, sa: _ANSWER if p in report_target else _clean_response(p, st, sa))
    assert result.trials[0]["target_met"] is True
    assert result.winner["admissible"] is True
    assert result.reason == "target_not_met"
    assert result.behaviour is None


def test_single_prompt_cannot_be_reused_as_its_own_report():
    with pytest.raises(ValueError, match="two distinct prompts"):
        _split_tune_report(["one", "one"], 0.5, 0)


def test_judge_uses_configured_language_drift_limit():
    candidate = _cand(language_drift=0.2, drifted=True)
    assert judge(candidate, BASE, InduceSpec(language_drift_max=0.25)) == (True, "ok")
    assert judge(candidate, BASE, InduceSpec(language_drift_max=0.1)) == (False, "language_drift")
