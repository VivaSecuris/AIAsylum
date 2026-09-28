"""Conditional steering: the gate is closed => stock output; open => steered; and it round-trips."""

import pytest

torch = pytest.importorskip("torch")

from tests._tiny_lm import build_model, build_tokenizer
from vivasecuris.aiasylum.weights.direction import RefusalDirection
from vivasecuris.aiasylum.weights.evaluate import _generate, generate_greedy
from vivasecuris.aiasylum.weights.gate import (
    GatedSteer,
    alpha_for,
    class_gap,
    steer_context,
)


@pytest.fixture
def model_tok():
    return build_model(), build_tokenizer()


@pytest.fixture
def direction():
    torch.manual_seed(3)
    v = torch.randn(32)
    v = v / v.norm()
    return RefusalDirection(
        vector=v, layer=1, auc=0.99, cohens_d=2.0, model_id="tiny",
        split_hash="deadbeef",
        extra={"projection_means": {"harmful": 2.0, "harmless": 0.0, "layer": 1}},
    )


def test_class_gap_reads_projection_means(direction):
    assert class_gap(direction) == pytest.approx(2.0)
    assert alpha_for(direction, 0.5) == pytest.approx(1.0)


def test_closed_gate_is_bit_identical(model_tok):
    from contextlib import nullcontext

    m, tok = model_tok
    prompts = ["w3 w4", "w5 w6 w7"]
    base = generate_greedy(m, tok, prompts, max_new_tokens=8, apply_template=False)
    gated = _generate(m, tok, prompts, max_new_tokens=8, apply_template=False,
                      context_for=lambda i: nullcontext())
    assert base == gated


def test_open_gate_changes_output(model_tok, direction):
    m, tok = model_tok
    prompts = ["w3 w4 w5"]
    base = generate_greedy(m, tok, prompts, max_new_tokens=12, apply_template=False)
    steered = _generate(
        m, tok, prompts, max_new_tokens=12, apply_template=False,
        context_for=lambda i: steer_context(m, direction, alpha_for(direction, 3.0)),
    )
    # A large activation addition at the derived layer must perturb generation.
    assert steered != base


def test_zero_alpha_installs_no_hook(model_tok, direction):
    m, tok = model_tok
    prompts = ["w3 w4 w5"]
    base = generate_greedy(m, tok, prompts, max_new_tokens=12, apply_template=False)
    at_zero = _generate(
        m, tok, prompts, max_new_tokens=12, apply_template=False,
        context_for=lambda i: steer_context(m, direction, 0.0),
    )
    assert base == at_zero


def test_gatedsteer_roundtrip(tmp_path):
    g = GatedSteer(probe_dir="p", direction_dir="d", layer=7, m=1.25, tau=0.7,
                   gap=1.5, model_id="tiny", goal="category")
    assert g.alpha == pytest.approx(1.875)
    g.save(tmp_path)
    back = GatedSteer.load(tmp_path)
    assert (back.layer, back.m, back.tau, back.gap, back.goal) == (7, 1.25, 0.7, 1.5, "category")
    assert back.alpha == pytest.approx(g.alpha)
