"""Expert-selective surgery: edit chosen experts, leave everything else alone.

Tiny random MoE models, CPU, seconds. The full-coverage contract of
residual_write_plan must be untouched by any of this.
"""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from vivasecuris.aiasylum.interp.core.arch import (
    expert_down_matrices,
    moe_layout,
    normalize_expert_selection,
    residual_write_plan,
)
from vivasecuris.aiasylum.weights.surgery import (
    ablate_experts,
    apply_direction_to_experts,
    apply_subspace_to_experts,
    edit_and_save,
)

WIDTH, LAYERS, VOCAB = 32, 2, 64
_BASE = dict(
    vocab_size=VOCAB, hidden_size=WIDTH, intermediate_size=WIDTH * 2,
    num_hidden_layers=LAYERS, num_attention_heads=4, num_key_value_heads=2,
    max_position_embeddings=32, bos_token_id=2, eos_token_id=2, pad_token_id=1,
    tie_word_embeddings=False,
)


def build(family: str, **overrides):
    if family == "qwen2_moe":
        cfg = transformers.Qwen2MoeConfig(**_BASE, num_experts=4, num_experts_per_tok=2,
                                          moe_intermediate_size=WIDTH,
                                          shared_expert_intermediate_size=WIDTH, **overrides)
    elif family == "qwen3_moe":
        cfg = transformers.Qwen3MoeConfig(**_BASE, num_experts=4, num_experts_per_tok=1,
                                          moe_intermediate_size=WIDTH, head_dim=WIDTH // 4, **overrides)
    elif family == "deepseek_v3":
        cfg = transformers.DeepseekV3Config(
            **_BASE, n_routed_experts=4, n_shared_experts=1, num_experts_per_tok=2,
            moe_intermediate_size=WIDTH, first_k_dense_replace=1, n_group=1, topk_group=1,
            kv_lora_rank=8, q_lora_rank=None, qk_rope_head_dim=4, qk_nope_head_dim=4, v_head_dim=8,
            **overrides)
    else:
        raise ValueError(family)
    torch.manual_seed(17)
    return transformers.AutoModelForCausalLM.from_config(cfg).eval()


def _snapshot(model):
    return {name: p.detach().clone() for name, p in model.named_parameters()}


def _changed(before, model):
    return sorted(name for name, p in model.named_parameters() if not torch.equal(before[name], p))


def test_layout_reports_each_moe_layer():
    layout = moe_layout(build("qwen2_moe"))
    assert sorted(layout) == list(range(LAYERS))
    for info in layout.values():
        assert info.n_experts == 4 and info.top_k == 2
        assert info.has_shared is True and info.experts_are_modules is True
        assert info.gate_kind == "linear_logits"

    layout = moe_layout(build("deepseek_v3"))
    assert sorted(layout) == [1], "layer 0 is dense under first_k_dense_replace=1"
    assert layout[1].gate_kind == "router_tuple"
    assert layout[1].n_experts == 4


def test_selection_validation():
    model = build("qwen2_moe")
    layout = moe_layout(model)
    assert normalize_expert_selection({"1": "all", 0: [3, 1]}, layout) == {0: [1, 3], 1: [0, 1, 2, 3]}
    with pytest.raises(ValueError, match="out of range"):
        normalize_expert_selection({0: [4]}, layout)
    with pytest.raises(ValueError, match="repeats"):
        normalize_expert_selection({0: [1, 1]}, layout)
    with pytest.raises(ValueError, match="empty"):
        normalize_expert_selection({0: []}, layout)
    with pytest.raises(ValueError, match="non-empty mapping"):
        normalize_expert_selection({}, layout)
    with pytest.raises(ValueError, match="no routed experts"):
        normalize_expert_selection({0: [0]}, moe_layout(build("deepseek_v3")))


def test_plan_reaches_only_the_selected_experts():
    model = build("qwen2_moe")
    plan = expert_down_matrices(model, {0: [1]})
    assert [m.name for m in plan.matrices] == ["layers.0.mlp_experts.1.down.weight"]
    assert plan.experts_edited == 1 and plan.shared_edited == 0 and plan.layers_edited == 1
    assert plan.moe_layers == LAYERS

    plan = expert_down_matrices(model, {0: [1]}, include_shared=True)
    assert [m.name for m in plan.matrices] == [
        "layers.0.mlp_experts.1.down.weight", "layers.0.mlp_shared_expert.down.weight",
    ]
    assert plan.shared_edited == 1


def test_direction_edit_touches_only_the_selected_parameters():
    model = build("qwen2_moe")
    before = _snapshot(model)
    r = torch.randn(WIDTH)
    r = r / r.norm()

    summary = apply_direction_to_experts(model, r, beta=0.0, selection={0: [1], 1: [2, 3]})
    assert _changed(before, model) == [
        "model.layers.0.mlp.experts.1.down_proj.weight",
        "model.layers.1.mlp.experts.2.down_proj.weight",
        "model.layers.1.mlp.experts.3.down_proj.weight",
    ]
    assert summary["coverage_verified"] is False
    assert summary["expert_mode"] == "direction"
    assert summary["expert_selection"] == {"0": [1], "1": [2, 3]}
    assert summary["matrices_edited"] == 3 and summary["beta"] == 0.0
    # The edited matrices no longer write r.
    for name in _changed(before, model):
        W = dict(model.named_parameters())[name]
        assert (r @ W).abs().max() < 1e-5

    # The full-coverage contract is untouched by any of this.
    assert residual_write_plan(model).coverage_verified is True


def test_subspace_edit_on_experts():
    model = build("qwen2_moe")
    basis, _ = torch.linalg.qr(torch.randn(WIDTH, 2))
    basis = basis.T.contiguous()
    summary = apply_subspace_to_experts(model, basis, k=1.0, selection={0: "all"})
    assert summary["subspace_rank"] == 2 and summary["matrices_edited"] == 4
    assert summary["expert_mode"] == "subspace"
    for e in range(4):
        W = model.model.layers[0].mlp.experts[e].down_proj.weight
        assert (basis @ W).abs().max() < 1e-5


def test_ablate_zeroes_the_selected_experts_and_leaves_the_router():
    model = build("qwen2_moe")
    before = _snapshot(model)
    summary = ablate_experts(model, {1: "all"}, scale=0.0)
    assert summary["expert_mode"] == "ablate" and summary["expert_scale"] == 0.0
    changed = _changed(before, model)
    assert changed == [f"model.layers.1.mlp.experts.{e}.down_proj.weight" for e in range(4)]
    for e in range(4):
        assert torch.count_nonzero(model.model.layers[1].mlp.experts[e].down_proj.weight) == 0
    assert torch.equal(before["model.layers.1.mlp.gate.weight"], model.model.layers[1].mlp.gate.weight)
    assert torch.equal(before["model.layers.1.mlp.shared_expert.down_proj.weight"],
                       model.model.layers[1].mlp.shared_expert.down_proj.weight)


def test_ablation_only_affects_tokens_routed_to_that_expert():
    """With top_k=1 and the last layer edited, an unrouted position's output is bit-identical."""
    model = build("qwen3_moe")
    ids = torch.randint(0, VOCAB, (1, 16))
    with torch.no_grad():
        out = model(ids, output_hidden_states=True, output_router_logits=True)
    before = out.hidden_states[-1][0]
    top1 = out.router_logits[LAYERS - 1].argmax(dim=-1)
    target = int(torch.bincount(top1, minlength=4).argmax())
    routed = (top1 == target).nonzero().flatten()
    unrouted = (top1 != target).nonzero().flatten()
    assert len(routed) and len(unrouted), "need both kinds of position for this to mean anything"

    ablate_experts(model, {LAYERS - 1: [target]}, scale=0.0)
    with torch.no_grad():
        after = model(ids, output_hidden_states=True).hidden_states[-1][0]
    assert torch.allclose(before[unrouted], after[unrouted], atol=1e-6)
    assert not torch.allclose(before[routed], after[routed], atol=1e-4)


def _word_level_tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    words = ["[UNK]", "[PAD]", "[EOS]", "the", "cat", "dog", "sat", "on", "a", "mat", "rug"]
    backend = Tokenizer(WordLevel({w: i for i, w in enumerate(words)}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]",
        model_input_names=["input_ids", "attention_mask"],
    )


def test_edit_and_save_records_a_partial_manifest(tmp_path):
    from vivasecuris.aiasylum.weights.manifest import SurgeryManifest

    src = tmp_path / "src"
    build("qwen2_moe").save_pretrained(src)
    _word_level_tokenizer().save_pretrained(src)

    out = edit_and_save(
        str(src), None, str(tmp_path / "out"), device="cpu", dtype="float32",
        expert_selection={"0": [0, 2]}, expert_mode="ablate", expert_scale=0.0,
    )
    manifest = SurgeryManifest.load(out)
    assert manifest.method == "expert_ablate"
    assert manifest.coverage_verified is False
    assert manifest.beta is None and manifest.direction_layer is None
    assert manifest.model_type == "qwen2_moe"
    assert manifest.matrices_edited == 2 and manifest.expert_matrices == 2
    assert manifest.moe_layers == LAYERS
    assert manifest.extra["expert_selection"] == {"0": [0, 2]}
    assert manifest.extra["expert_mode"] == "ablate"
    assert manifest.extra["moe_layout"] == {"0": 4, "1": 4}
    assert manifest.as_metadata()["surgery"]["coverage_verified"] is False

    with pytest.raises(ValueError, match="need a direction"):
        edit_and_save(str(src), None, str(tmp_path / "out2"), device="cpu", dtype="float32",
                      expert_selection={"0": [0]}, expert_mode="direction")
