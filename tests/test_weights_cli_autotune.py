"""The `weights autotune` and `weights verify` commands: option parsing and the
publish-or-remove flow, with the engine stubbed. The engine itself is covered
by test_weights_pipeline.py and test_weights_execute_new_kinds.py."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from vivasecuris.aiasylum.weights import autotune as autotune_mod
from vivasecuris.aiasylum.weights import cli as cli_mod
from vivasecuris.aiasylum.weights import verify as verify_mod
from vivasecuris.aiasylum.weights.cli import weights
from vivasecuris.aiasylum.weights.verify import VerifyReport


def _report(passed=True, reasons=()):
    metrics = {"refuse_harmful": 0.0, "factual_acc": 0.9, "degenerate": False,
               "language_drift": 0.0, "drifted": False, "decoding": "greedy", "responses": {}}
    return VerifyReport(passed=passed, reasons=list(reasons), greedy=metrics, sampled=metrics,
                        hashes={"model.safetensors": "ab" * 32}, hash_check=None, device="cpu",
                        dtype="float32", n_prompts=2, capability_set="builtin")


@pytest.fixture
def stubbed(monkeypatch, tmp_path):
    """Everything heavy replaced: the loader, the corpus, the search, the writer
    and the verifier. Records the spec the search received and what got written."""
    seen = {}
    direction_dir = tmp_path / "direction"
    direction_dir.mkdir()
    from vivasecuris.aiasylum.weights.direction import RefusalDirection

    torch.manual_seed(0)
    basis = torch.linalg.qr(torch.randn(8, 2))[0].T.contiguous()
    RefusalDirection(vector=basis[0], layer=1, auc=0.95, cohens_d=2.0, model_id="m", split_hash="x",
                     basis=basis, basis_layers=[1, 1]).save(direction_dir)

    monkeypatch.setattr(cli_mod, "_preflight", lambda *a, **k: None)
    monkeypatch.setattr(cli_mod, "_warn_if_low_disk", lambda *a, **k: None)
    monkeypatch.setattr("vivasecuris.aiasylum.interp.core.loader.load",
                        lambda model, device="auto", dtype="bfloat16", seed=0: ("model", "tok"))
    monkeypatch.setattr("vivasecuris.aiasylum.weights.corpus.build_split",
                        lambda seed=0: SimpleNamespace(harmful_test=["bad one", "bad two"],
                                                       harmless_test=["fine one", "fine two"]))
    monkeypatch.setattr("vivasecuris.aiasylum.models.transformers_local.clear_cache", lambda: None)

    winner = {"index": 1, "rank": 1, "k": 1.0, "include_embeddings": False, "refuse_harmful": 0.0,
              "factual_acc": 0.9, "language_drift": 0.0, "degenerate": False, "drifted": False,
              "compliance": 1.0, "factual_drop": 0.0, "accepted": True, "target_met": True,
              "reason": "ok", "sampled": None, "mean_relative_change": 0.01, "matrices_edited": 4,
              "elapsed_s": 1.0, "responses": {"harmful": ["Sure"], "factual": ["Paris"]}}

    def fake_autotune(model, tok, direction, prompts, spec, capability=None, progress=None,
                      keep_winner_applied=True):
        seen["spec"] = spec
        seen["prompts"] = list(prompts)
        no_winner = seen.get("no_winner", False)
        result = SimpleNamespace(
            snapshot=SimpleNamespace(release=lambda: seen.update(snapshot_released=True)),
            baseline={"greedy": {"refuse_harmful": 0.9, "factual_acc": 0.9, "language_drift": 0.0}, "sampled": None},
            trials=[winner], winner=None if no_winner else winner,
            winner_summary={"matrices_edited": 4, "weights": None},
            summary=lambda: {"trials": [winner], "winner": None if no_winner else winner, "spec": spec.as_dict(),
                             "candidates_tried": 1, "candidates_planned": 1, "target_met": not no_winner,
                             "embeddings_tied": True},
        )
        return result

    def fake_save(model, tok, out_dir, **kwargs):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "model.safetensors").write_bytes(b"weights")
        (Path(out_dir) / "asylum_surgery.json").write_text(json.dumps({"extra": kwargs.get("extra")}))
        seen["saved"] = kwargs
        return str(out_dir)

    monkeypatch.setattr(autotune_mod, "autotune_edit", fake_autotune)
    monkeypatch.setattr("vivasecuris.aiasylum.weights.surgery.save_edited_model", fake_save)
    monkeypatch.setattr(verify_mod, "stamp_manifest", lambda path, report: True)
    def verify_after_release(*args, **kwargs):
        if "spec" in seen:
            assert seen.get("snapshot_released"), "Snapshot must release live weights before reload"
        return seen.get("report", _report())

    monkeypatch.setattr(verify_mod, "verify_checkpoint", verify_after_release)
    return seen, direction_dir


def test_autotune_parses_options_and_publishes_a_verified_model(stubbed, tmp_path):
    seen, direction_dir = stubbed
    out = tmp_path / "models" / "auto"
    report_path = tmp_path / "report.json"
    r = CliRunner().invoke(weights, [
        "autotune", "--model", "m", "--direction", str(direction_dir), "--out", str(out),
        "--ranks", "1,2", "--ks", "1.0,1.5", "--embeddings", "both", "--max-candidates", "3",
        "--stop-at-first", "--max-refusal", "0.2", "--n-prompts", "1", "--temperature", "0.9",
        "--seed", "5", "--report", str(report_path),
    ])
    assert r.exit_code == 0, r.output
    spec = seen["spec"]
    assert spec.ranks == (1, 2) and spec.ks == (1.0, 1.5) and spec.embedding_modes == (False, True)
    assert spec.max_candidates == 3 and spec.stop_at_first_admissible is True and spec.max_refusal == 0.2
    assert spec.sampling.temperature == 0.9 and spec.sampling.seed == 5 and spec.verify_sampled is True
    assert seen["prompts"] == ["bad one"]
    assert out.is_dir() and not (out.parent / ".staging-auto").exists()
    assert seen["saved"]["extra"]["autotune"]["winner"]["rank"] == 1
    assert "Verified and saved" in r.output and "WINNER" in r.output
    payload = json.loads(report_path.read_text())
    assert payload["verification"]["passed"] is True and payload["winner"]["rank"] == 1


def test_autotune_removes_the_staging_directory_when_verification_fails(stubbed, tmp_path):
    seen, direction_dir = stubbed
    seen["report"] = _report(passed=False, reasons=["sampled: language drift 40% above 10%"])
    out = tmp_path / "models" / "auto-bad"
    r = CliRunner().invoke(weights, ["autotune", "--model", "m", "--direction", str(direction_dir),
                                     "--out", str(out), "--embeddings", "no"])
    assert r.exit_code != 0
    assert "failed verification" in r.output and "language drift" in r.output
    assert not out.exists() and not (out.parent / ".staging-auto-bad").exists()
    assert seen["spec"].embedding_modes == (False,)


def test_autotune_with_no_admissible_candidate_writes_nothing(stubbed, tmp_path):
    seen, direction_dir = stubbed
    seen["no_winner"] = True
    out = tmp_path / "models" / "auto-none"
    r = CliRunner().invoke(weights, ["autotune", "--model", "m", "--direction", str(direction_dir),
                                     "--out", str(out), "--embeddings", "auto", "--no-verify-sampled"])
    assert r.exit_code != 0 and "No candidate cleared every gate" in r.output
    assert not out.exists() and "saved" not in seen
    assert seen["spec"].embedding_modes is None and seen["spec"].verify_sampled is False
    assert seen["snapshot_released"]


def test_autotune_releases_snapshot_when_writing_fails(stubbed, monkeypatch, tmp_path):
    seen, direction_dir = stubbed

    def broken_save(*args, **kwargs):
        raise OSError("test disk failure")

    monkeypatch.setattr("vivasecuris.aiasylum.weights.surgery.save_edited_model", broken_save)
    result = CliRunner().invoke(weights, ["autotune", "--model", "m", "--direction", str(direction_dir),
                                          "--out", str(tmp_path / "failed")])
    assert result.exit_code != 0
    assert seen["snapshot_released"]


def test_autotune_refuses_a_non_empty_output_directory(stubbed, tmp_path):
    _, direction_dir = stubbed
    out = tmp_path / "taken"
    out.mkdir()
    (out / "x").write_text("x")
    r = CliRunner().invoke(weights, ["autotune", "--model", "m", "--direction", str(direction_dir), "--out", str(out)])
    assert r.exit_code != 0 and "already exists" in r.output


def test_verify_exits_by_the_gate(stubbed, tmp_path):
    seen, _ = stubbed
    model_dir = tmp_path / "edited"
    model_dir.mkdir()
    out = tmp_path / "verify.json"
    r = CliRunner().invoke(weights, ["verify", "--model", str(model_dir), "--out", str(out), "--no-sampled"])
    assert r.exit_code == 0, r.output
    assert "PASS" in r.output and json.loads(out.read_text())["passed"] is True

    seen["report"] = _report(passed=False, reasons=["greedy: degenerate output"])
    r = CliRunner().invoke(weights, ["verify", "--model", str(model_dir)])
    assert r.exit_code == 1 and "FAIL" in r.output and "degenerate" in r.output


def test_verify_check_hashes_needs_a_manifest_with_digests(stubbed, tmp_path):
    model_dir = tmp_path / "plain"
    model_dir.mkdir()
    r = CliRunner().invoke(weights, ["verify", "--model", str(model_dir), "--check-hashes"])
    assert r.exit_code != 0 and "weights_sha256" in r.output
