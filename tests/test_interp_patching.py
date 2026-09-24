"""Phase 0 credibility checks on a tiny locally-built model.

Three things the engine used to get wrong, each pinned by a test that fails
on the old behaviour:

- the logit lens skipped the final norm, so its reading at the last hidden
  state did not equal the model's own logits;
- activation patching never re-ran the forward pass, so patching the whole
  residual from run B into run A did not reproduce B's output;
- pre-MLP capture was a stub, so neuron-level anything was silently empty.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

D_MODEL, N_LAYERS, D_FFN = 32, 4, 64
TOKENIZER_SRC = "Qwen/Qwen2.5-3B-Instruct"


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory):
    from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Config

    out = tmp_path_factory.mktemp("tiny-qwen-patching")
    tok = AutoTokenizer.from_pretrained(TOKENIZER_SRC)
    cfg = Qwen2Config(
        vocab_size=len(tok), hidden_size=D_MODEL, intermediate_size=D_FFN,
        num_hidden_layers=N_LAYERS, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=128, tie_word_embeddings=False,
        bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
    )
    torch.manual_seed(0)
    AutoModelForCausalLM.from_config(cfg).save_pretrained(str(out))
    tok.save_pretrained(str(out))
    return out


@pytest.fixture(scope="module")
def loaded(tiny_model_dir):
    from vivasecuris.aiasylum.interp.core.loader import load

    return load(str(tiny_model_dir), device="cpu", dtype="float32", seed=0)


@pytest.fixture(scope="module")
def full_config(tiny_model_dir, tmp_path_factory):
    from vivasecuris.aiasylum.interp.core.config import Config

    return Config(
        model=str(tiny_model_dir), out_dir=str(tmp_path_factory.mktemp("patching-out")),
        enable_attention_capture=True, enable_mlp_capture=True,
        enable_attn_output_capture=True, enable_qkv_capture=True,
        enable_pre_mlp_capture=True, enable_patching=True, topk=5,
    )


def _runs(loaded, config, prompt_a="The capital of France is", prompt_b="The capital of Spain is"):
    from vivasecuris.aiasylum.interp.core.runner import ModelRunner

    model, tok = loaded
    runner = ModelRunner(model, tok, debug_mode=True, config=config)
    a, b = runner.run_once(prompt_a), runner.run_once(prompt_b)
    assert a.input_ids.shape == b.input_ids.shape, "test prompts must tokenise to the same length"
    return a, b


# ---------------------------------------------------------------------------
# Logit lens
# ---------------------------------------------------------------------------


def test_logit_lens_with_final_norm_reproduces_model_logits(loaded):
    from vivasecuris.aiasylum.interp.analysis.predictions import PredictionAnalyzer

    model, tok = loaded
    ids = tok("hello there general", return_tensors="pt")["input_ids"]
    with torch.no_grad():
        out = model(input_ids=ids, output_hidden_states=True, return_dict=True)
    last_hidden = out.hidden_states[-1][0, -1]

    lens = PredictionAnalyzer(model, tok, topk=5)
    assert lens.final_norm is not None
    # The final entry reads back as the model's own logits whichever convention
    # transformers uses for it.
    lens_logits = lens.lens_logits(last_hidden, is_final=True)
    assert torch.allclose(lens_logits, out.logits[0, -1].float(), atol=1e-4)

    # At an intermediate layer the norm is applied, and it changes the reading.
    mid = out.hidden_states[2][0, -1]
    raw = PredictionAnalyzer(model, tok, topk=5, apply_final_norm=False)
    assert not torch.allclose(lens.lens_logits(mid), raw.lens_logits(mid), atol=1e-3)
    # Applying the norm by hand reproduces the normed lens at that layer.
    with torch.no_grad():
        by_hand = (model.model.norm(mid) @ model.lm_head.weight.T).float()
    assert torch.allclose(lens.lens_logits(mid), by_hand, atol=1e-4)


# ---------------------------------------------------------------------------
# Real patching
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("layer", [0, N_LAYERS // 2, N_LAYERS])
def test_full_window_patch_reproduces_source_run(loaded, full_config, layer):
    """Patching every position at one layer makes the rest of the forward pass
    identical to the source run, so the logits must match. The old
    implementation could not pass this: it never ran the layers above."""
    from vivasecuris.aiasylum.interp.patching import patch_and_run

    model, _ = loaded
    a, b = _runs(loaded, full_config)
    seq = a.input_ids.shape[1]
    logits, final = patch_and_run(model, a, b, layer, [(i, i) for i in range(seq)])
    assert torch.allclose(logits, a.logits[0, -1].float(), atol=1e-4)
    assert torch.allclose(final, a.hidden_states[-1][0, -1].float(), atol=1e-4)


def test_layer_sweep_reports_recovery_and_null(loaded, full_config):
    from vivasecuris.aiasylum.interp.patching import run_patching_experiments

    model, tok = loaded
    a, b = _runs(loaded, full_config)
    seq = a.input_ids.shape[1]
    payload = run_patching_experiments(
        model, tok, full_config, a, b, cos_mat=None, dn_mat=None,
        spike_layer=2, start_a=0, start_b=0, window_len=seq,
    )
    assert payload["enabled"] and payload["claim"] == "causal"
    assert payload["layers"] == list(range(N_LAYERS + 1))
    # Two directions x every hidden index, plus one null pass per direction.
    assert payload["forward_passes"] == 2 * (N_LAYERS + 1) + 2
    assert set(payload["summary"]) == {"A_to_B", "B_to_A"}
    assert set(payload["baseline"]) == {"A_to_B", "B_to_A"}

    # Replacing the final residual at the last position with the source's
    # makes the last-position logits the source's: the gap is fully recovered.
    for exp in payload["experiments"]:
        if exp["layer"] == N_LAYERS:
            res = exp["results"][0]
            assert math.isclose(res["recovered"], 1.0, abs_tol=1e-3), res
            assert math.isclose(res["recovered_kl"], 1.0, abs_tol=1e-3), res
            assert res["kl_to_source_after"] < 1e-5
            assert res["top_target_after"] == (
                res["top_a_before"] if exp["direction"] == "A_to_B" else res["top_b_before"]
            )
    # Dashboard contract: the fields it plots are still there.
    first = payload["experiments"][0]["results"][0]
    for key in ("cos_before", "cos_after", "delta_before", "delta_after", "top_target_after"):
        assert key in first


def test_head_and_neuron_patching_run_real_passes(loaded, full_config):
    from dataclasses import replace

    from vivasecuris.aiasylum.interp.patching import run_patching_experiments

    model, tok = loaded
    a, b = _runs(loaded, full_config)
    seq = a.input_ids.shape[1]

    heads = replace(full_config, patch_components="head")
    payload = run_patching_experiments(
        model, tok, heads, a, b, None, None, spike_layer=1, start_a=0, start_b=0, window_len=seq,
    )
    assert payload["component"] == "head" and payload["enabled"]
    assert payload["forward_passes"] == 2 * 4, payload["notes"]   # 4 heads x 2 directions at one layer
    assert all(e["component"] == "head" for e in payload["experiments"])

    neurons = replace(full_config, patch_components="neuron", patch_neurons=[(1, 0), (1, 3)])
    payload = run_patching_experiments(
        model, tok, neurons, a, b, None, None, spike_layer=1, start_a=0, start_b=0, window_len=seq,
    )
    assert payload["component"] == "neuron" and payload["forward_passes"] == 2 * 2, payload["notes"]
    assert {e["component_index"] for e in payload["experiments"]} == {0, 3}


# ---------------------------------------------------------------------------
# Pre-MLP capture
# ---------------------------------------------------------------------------


def test_pre_mlp_capture_returns_neuron_activations(loaded, full_config):
    a, _ = _runs(loaded, full_config)
    assert a.pre_mlp_activations is not None
    assert set(a.pre_mlp_activations) == set(range(N_LAYERS))
    seq = a.input_ids.shape[1]
    assert a.pre_mlp_activations[0].shape == (1, seq, D_FFN)

    # Sum of per-neuron contributions reconstructs the MLP output (no bias on Qwen2).
    from vivasecuris.aiasylum.interp.patching import compute_per_neuron_mlp_outputs

    per = compute_per_neuron_mlp_outputs(a, None if False else _model_of(loaded), 0, 0, seq)
    assert per.shape == (D_FFN, seq, D_MODEL)
    assert torch.allclose(per.sum(dim=0), a.mlp_activations[0][0].float(), atol=1e-4)


def _model_of(loaded):
    return loaded[0]


# ---------------------------------------------------------------------------
# Labels and baselines
# ---------------------------------------------------------------------------


def test_head_roles_are_labelled_descriptive():
    from vivasecuris.aiasylum.interp.analysis.attribution import AttributionAnalyzer
    from vivasecuris.aiasylum.interp.data.models import RunResult

    empty = RunResult(input_ids=torch.zeros(1, 2, dtype=torch.long), token_strs=["a", "b"],
                      hidden_states=(torch.zeros(1, 2, 4),), logits=torch.zeros(1, 2, 8))
    out = AttributionAnalyzer.classify_head_roles_by_contribution(empty, empty, 0, 0, 0, 2)
    assert out["claim"] == "descriptive"
    assert out["method"] == "contribution_threshold"
    # The old name still resolves to the same heuristic.
    assert AttributionAnalyzer.classify_head_roles_causal_head_gating is AttributionAnalyzer.classify_head_roles_by_contribution


def test_baselines_helpers():
    from vivasecuris.aiasylum.interp.analysis.baselines import (
        principal_angles, random_dictionary, random_unit_directions, shuffled_label_auc, subspace_overlap,
    )

    dirs = random_unit_directions(16, 5, seed=1)
    assert dirs.shape == (5, 16)
    assert torch.allclose(dirs.norm(dim=1), torch.ones(5), atol=1e-6)
    assert random_dictionary(16, 3, seed=2).shape == (3, 16)

    B = torch.linalg.qr(torch.randn(16, 3)).Q.t()
    assert max(principal_angles(B, B)) < 1e-5
    assert math.isclose(subspace_overlap(B, B), 1.0, abs_tol=1e-5)
    C = torch.linalg.qr(torch.randn(16, 16)).Q.t()[3:6]
    # Orthogonal complement rows of a random basis are not the same subspace.
    assert subspace_overlap(B, C) < 0.9

    gen = torch.Generator().manual_seed(0)
    pos, neg = torch.randn(40, generator=gen), torch.randn(40, generator=gen)
    null = shuffled_label_auc(pos, neg, n_perm=30, seed=0)
    assert 0.5 <= null["mean"] <= 0.65
    assert null["p95"] >= null["mean"]
