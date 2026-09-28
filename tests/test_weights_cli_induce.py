"""Induce CLI exports only an accepted control into a fresh output directory."""

import json
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

pytest.importorskip("torch")
pytest.importorskip("transformers")

from vivasecuris.aiasylum.weights import cli as cli_mod
from vivasecuris.aiasylum.weights import induce as induce_mod
from vivasecuris.aiasylum.weights.gate import GatedSteer
from vivasecuris.aiasylum.weights.induce import InduceResult, InduceSpec


def _result(rejection=None):
    winner = {"m": 0.5, "tau": 0.8, "accepted": True, "reason": "ok", "target_met": True,
              "refuse_target": 0.9, "refuse_near_miss": 0.0}
    sampled = {"accepted": True, "reason": "ok", "target_met": True,
               "refuse_target": 0.95, "refuse_near_miss": 0.0,
               "baseline": {"refuse_target": 0.2}}
    behaviour = GatedSteer("probe", "direction", 1, 0.5, 0.8, 1.0, model_id="model")
    if rejection == "report":
        winner.update(accepted=False, reason="near_miss_refused", target_met=False)
        sampled = None
    elif rejection == "sampled":
        sampled.update(accepted=False, reason="capability_cost", target_met=False)
    elif rejection == "no_winner":
        winner = sampled = None
    return InduceResult(
        baseline={"refuse_target": 0.1}, trials=[{"m": 0.5, "tau": 0.8}],
        winner=winner, behaviour=None if rejection else behaviour,
        gate={}, spec=InduceSpec(), verification=sampled,
    )


@pytest.fixture
def stubbed(monkeypatch, tmp_path):
    from vivasecuris.aiasylum.interp.probes.train import ProbeSet
    from vivasecuris.aiasylum.weights.categories import CategorySet
    from vivasecuris.aiasylum.weights.direction import RefusalDirection

    seen = {"loads": [], "distill_calls": 0, "result": _result()}
    monkeypatch.setattr(cli_mod, "_require_interp", lambda: None)
    monkeypatch.setattr(cli_mod, "_preflight", lambda: seen.update(preflight=True))

    def load(model, **kwargs):
        seen["loads"].append(model)
        return "model", "tokenizer"

    monkeypatch.setattr("vivasecuris.aiasylum.interp.core.loader.load", load)
    monkeypatch.setattr(CategorySet, "load", lambda path: CategorySet(
        "category", [f"target {i}" for i in range(8)], [f"near {i}" for i in range(8)],
    ))
    monkeypatch.setattr(ProbeSet, "load", lambda path: SimpleNamespace(model_id="model"))
    monkeypatch.setattr(RefusalDirection, "load", lambda path: SimpleNamespace(layer=1))
    monkeypatch.setattr("vivasecuris.aiasylum.weights.direction.check_direction_format", lambda *args: None)
    monkeypatch.setattr("vivasecuris.aiasylum.weights.corpus.load_harmless_prompts", lambda **kwargs: ["benign"])
    monkeypatch.setattr("vivasecuris.aiasylum.weights.corpus.build_split", lambda **kwargs: SimpleNamespace(harmful_test=["harmful"]))
    monkeypatch.setattr(cli_mod, "_capability_for", lambda *args: SimpleNamespace(questions=["fact"]))
    monkeypatch.setattr(induce_mod, "tune_gated", lambda *args, **kwargs: seen["result"])

    def distill(*args, **kwargs):
        seen["distill_calls"] += 1
        return [{"prompt": "target", "response": "I cannot help with that request."}]

    monkeypatch.setattr(induce_mod, "build_distill_rows", distill)
    return seen


def _invoke(out):
    return CliRunner().invoke(cli_mod.weights, [
        "induce", "--model", "model", "--direction", "direction", "--probe", "probe",
        "--category", "category.jsonl", "--out", str(out),
    ])


def test_nonempty_output_is_preserved_before_loading(stubbed, tmp_path):
    out = tmp_path / "existing"
    out.mkdir()
    previous = {"behaviour.json": '{"old": true}', "distill_rows.jsonl": "old rows\n", "induce.json": "old report"}
    for name, content in previous.items():
        (out / name).write_text(content)
    result = _invoke(out)
    assert result.exit_code == 1
    assert "already exists and is not empty" in result.output
    assert stubbed["loads"] == [] and not stubbed.get("preflight")
    assert {p.name: p.read_text() for p in out.iterdir()} == previous


@pytest.mark.parametrize("rejection", ["report", "sampled", "no_winner"])
def test_rejected_run_writes_diagnostics_without_exporting_control(stubbed, tmp_path, rejection):
    stubbed["result"] = _result(rejection)
    out = tmp_path / "rejected"
    result = _invoke(out)
    assert result.exit_code == 1, result.output
    assert f"Control rejected: {stubbed['result'].reason}" in result.output
    assert {p.name for p in out.iterdir()} == {"induce.json"}
    report = json.loads((out / "induce.json").read_text())
    assert report["accepted"] is False and report["reason"] == stubbed["result"].reason
    assert stubbed["distill_calls"] == 0


@pytest.mark.parametrize("existing_empty", [False, True])
def test_accepted_run_exports_control_and_uses_sampled_baseline(stubbed, tmp_path, existing_empty):
    out = tmp_path / "accepted"
    if existing_empty:
        out.mkdir()
    result = _invoke(out)
    assert result.exit_code == 0, result.output
    assert "refuse_target 95% (baseline 20%)" in result.output
    assert "Wrote behaviour.json and 1 distill rows" in result.output
    assert {p.name for p in out.iterdir()} == {"induce.json", "behaviour.json", "distill_rows.jsonl"}
    assert json.loads((out / "induce.json").read_text())["accepted"] is True
    assert json.loads((out / "behaviour.json").read_text())["tau"] == 0.8
    assert json.loads((out / "distill_rows.jsonl").read_text())["prompt"] == "target"
    assert stubbed["distill_calls"] == 1
