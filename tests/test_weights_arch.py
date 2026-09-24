"""Architecture detection and residual-write enumeration.

Uses tiny randomly-initialized models built from config, so these run in
seconds with no network access.
"""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from vivasecuris.aiasylum.interp.core.arch import (
    ARCH_QWEN2,
    KIND_EMBED,
    KIND_OUT,
    detect_architecture,
    embeddings_are_tied,
    get_decoder_layers,
    residual_write_matrices,
)

N_LAYERS, D_MODEL, D_FFN, VOCAB = 4, 32, 64, 128


def _tiny_qwen(tie: bool):
    from transformers import AutoModelForCausalLM, Qwen2Config

    cfg = Qwen2Config(
        vocab_size=VOCAB,
        hidden_size=D_MODEL,
        intermediate_size=D_FFN,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=tie,
    )
    return AutoModelForCausalLM.from_config(cfg)


@pytest.fixture(scope="module")
def tied_model():
    return _tiny_qwen(tie=True)


@pytest.fixture(scope="module")
def untied_model():
    return _tiny_qwen(tie=False)


def test_detects_qwen2(tied_model):
    assert detect_architecture(tied_model) == ARCH_QWEN2


def test_finds_all_decoder_layers(tied_model):
    assert len(get_decoder_layers(tied_model, ARCH_QWEN2)) == N_LAYERS


def test_enumerates_embed_plus_two_matrices_per_layer(untied_model):
    mats = residual_write_matrices(untied_model)
    assert len(mats) == 1 + 2 * N_LAYERS

    kinds = [m.kind for m in mats]
    assert kinds.count(KIND_EMBED) == 1
    assert kinds.count(KIND_OUT) == 2 * N_LAYERS

    names = [m.name for m in mats]
    assert any("embed_tokens" in n for n in names)
    assert sum("attn_out" in n for n in names) == N_LAYERS
    assert sum("mlp_down" in n for n in names) == N_LAYERS


def test_matrix_shapes_match_their_kind(untied_model):
    """The direction indexes axis 1 for the embedding and axis 0 for out-projections."""
    for m in residual_write_matrices(untied_model):
        if m.kind == KIND_EMBED:
            assert m.param.shape == (VOCAB, D_MODEL)
        else:
            assert m.param.shape[0] == D_MODEL


def test_lm_head_is_never_a_target(untied_model):
    """lm_head reads from the residual stream; it is not a write matrix."""
    head_ptr = untied_model.lm_head.weight.data_ptr()
    assert all(m.param.data_ptr() != head_ptr for m in residual_write_matrices(untied_model))


def test_tied_embeddings_detected(tied_model, untied_model):
    assert embeddings_are_tied(tied_model) is True
    assert embeddings_are_tied(untied_model) is False


def test_tied_weights_are_enumerated_once(tied_model):
    """The critical dedup: applying the projection twice squares it."""
    mats = residual_write_matrices(tied_model)
    ptrs = [m.param.data_ptr() for m in mats]
    assert len(ptrs) == len(set(ptrs)), "a shared tensor was returned more than once"
    assert len(mats) == 1 + 2 * N_LAYERS


def test_exclude_embeddings_option(untied_model):
    mats = residual_write_matrices(untied_model, include_embeddings=False)
    assert len(mats) == 2 * N_LAYERS
    assert all(m.kind == KIND_OUT for m in mats)


def test_unsupported_architecture_raises():
    """gpt2 has no `.model.layers`; it must fail loudly rather than edit nothing."""
    from transformers import AutoModelForCausalLM, GPT2Config

    gpt2 = AutoModelForCausalLM.from_config(
        GPT2Config(n_layer=2, n_embd=32, n_head=2, vocab_size=64, n_positions=64)
    )
    assert detect_architecture(gpt2) is None
    with pytest.raises(ValueError, match="Unsupported architecture"):
        residual_write_matrices(gpt2)


def test_ablation_on_a_real_model_removes_the_direction(untied_model):
    """End-to-end on an actual transformer: after beta=0 the residual loses r."""
    from vivasecuris.aiasylum.weights.surgery import apply_to_model

    torch.manual_seed(0)
    r = torch.randn(D_MODEL)
    r = r / r.norm()
    ids = torch.randint(0, VOCAB, (2, 12))

    with torch.no_grad():
        before = untied_model(ids, output_hidden_states=True).hidden_states[-1]
    assert (before @ r).abs().max() > 1e-3, "direction absent before ablation; test is vacuous"

    summary = apply_to_model(untied_model, r, beta=0.0)
    assert summary["matrices_edited"] == 1 + 2 * N_LAYERS
    assert summary["architecture"] == ARCH_QWEN2

    with torch.no_grad():
        after = untied_model(ids, output_hidden_states=True).hidden_states[-1]

    # Final-layer norm rescales, so compare the direction's share of the signal.
    share = (after @ r).abs().mean() / after.norm(dim=-1).mean()
    assert share < 0.05, f"direction still carries {share:.3f} of the residual"
