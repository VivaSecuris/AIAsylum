"""CLI detector claims require the null, answer-length baseline, and neurons."""

import json

import pytest
from click.testing import CliRunner

pytest.importorskip("torch")
pytest.importorskip("transformers")

from vivasecuris.aiasylum.weights import cli as cli_mod
from vivasecuris.aiasylum.weights import hneurons as hneurons_mod
from vivasecuris.aiasylum.weights.hneurons import HNeuronSet


@pytest.fixture
def stubbed(monkeypatch, tmp_path):
    seen = {}
    rows = [{"question": f"question {i}", "aliases": ["answer"]} for i in range(16)]
    source = tmp_path / "questions.jsonl"
    source.write_text("\n".join(json.dumps(row) for row in rows))
    monkeypatch.setattr(cli_mod, "_require_interp", lambda: None)
    monkeypatch.setattr(cli_mod, "_preflight", lambda: None)
    monkeypatch.setattr("vivasecuris.aiasylum.interp.core.loader.load", lambda *args, **kwargs: ("model", "tokenizer"))
    monkeypatch.setattr(hneurons_mod, "label_consistency", lambda *args, **kwargs: {
        "correct": rows[:8], "incorrect": rows[8:],
    })
    monkeypatch.setattr(hneurons_mod, "capture_cett", lambda *args, **kwargs: (
        "features", [(0, 0), (0, 1)], 4, list(range(16)),
    ))

    def select(*args, **kwargs):
        seen["selection"] = kwargs
        return seen["hset"]

    monkeypatch.setattr(hneurons_mod, "select_hneurons", select)
    return seen, source


@pytest.mark.parametrize("auroc,null,surface,n_selected,warning", [
    (0.9, 0.6, 0.65, 2, None),
    (0.75, 0.55, 0.9, 2, "does not clear its answer-length baseline"),
    (0.75, 0.55, float("nan"), 2, "answer-length baseline is unavailable"),
    (0.9, 0.6, 0.65, 0, "no neurons were selected"),
    (0.75, 0.74, 0.6, 2, "does not clear its shuffled-label null"),
])
def test_detector_summary_reports_both_baselines_and_usability(
    stubbed, tmp_path, auroc, null, surface, n_selected, warning,
):
    seen, source = stubbed
    hset = HNeuronSet(
        neurons={0: [0, 1]} if n_selected else {}, model_id="model", d_ff=4,
        auroc=auroc, null_auroc_p95=null, surface_auroc=surface, n_selected=n_selected, n_total=8,
    )
    seen["hset"] = hset
    out = tmp_path / "neurons"
    result = CliRunner().invoke(cli_mod.weights, [
        "hneurons", "--model", "model", "--questions-file", str(source), "--out", str(out),
    ])
    assert result.exit_code == 0, result.output
    assert f"beats_null {hset.beats_null}" in result.output
    assert f"beats_surface {hset.beats_surface}" in result.output
    assert f"usable {hset.usable}" in result.output
    if warning:
        assert "WARNING: this detector is not usable" in result.output
        assert warning in result.output
    else:
        assert "WARNING" not in result.output
    report = json.loads((out / "hneurons.json").read_text())
    assert report["usable"] is (warning is None)
    assert seen["selection"]["answer_lengths"] == list(range(16))
