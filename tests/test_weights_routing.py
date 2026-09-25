"""Routing statistics: the replayed top-k must match what the experts actually saw."""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from vivasecuris.aiasylum.weights.routing import RoutingCapture, routing_statistics, routing_table

WIDTH, LAYERS, VOCAB = 32, 2, 64
_BASE = dict(
    vocab_size=VOCAB, hidden_size=WIDTH, intermediate_size=WIDTH * 2,
    num_hidden_layers=LAYERS, num_attention_heads=4, num_key_value_heads=2,
    max_position_embeddings=32, bos_token_id=2, eos_token_id=2, pad_token_id=1,
    tie_word_embeddings=False,
)
_DEEPSEEK = dict(
    n_routed_experts=4, n_shared_experts=1, num_experts_per_tok=2, moe_intermediate_size=WIDTH,
    first_k_dense_replace=1, n_group=1, topk_group=1, kv_lora_rank=8, q_lora_rank=None,
    qk_rope_head_dim=4, qk_nope_head_dim=4, v_head_dim=8,
)
FAMILIES = {
    "qwen2_moe": ("Qwen2MoeConfig", dict(num_experts=4, num_experts_per_tok=2, moe_intermediate_size=WIDTH,
                                         shared_expert_intermediate_size=WIDTH)),
    "qwen3_moe": ("Qwen3MoeConfig", dict(num_experts=4, num_experts_per_tok=2, moe_intermediate_size=WIDTH,
                                         head_dim=WIDTH // 4)),
    "mixtral": ("MixtralConfig", dict(num_local_experts=4, num_experts_per_tok=2)),
    "deepseek_v3": ("DeepseekV3Config", _DEEPSEEK),
    "glm4_moe": ("Glm4MoeConfig", dict(n_routed_experts=4, n_shared_experts=1, num_experts_per_tok=2,
                                       moe_intermediate_size=WIDTH, first_k_dense_replace=1, n_group=1,
                                       topk_group=1, head_dim=WIDTH // 4)),
    "ernie4_5_moe": ("Ernie4_5_MoeConfig", dict(moe_num_experts=4, moe_num_shared_experts=1, moe_k=2,
                                               moe_intermediate_size=WIDTH, moe_layer_start_index=0,
                                               moe_layer_end_index=LAYERS)),
    "olmoe": ("OlmoeConfig", dict(num_experts=4, num_experts_per_tok=2)),
}


def build(family):
    cls_name, extra = FAMILIES[family]
    cls = getattr(transformers, cls_name, None)
    if cls is None:
        pytest.skip(f"{family} is unavailable in this transformers version")
    torch.manual_seed(17)
    return transformers.AutoModelForCausalLM.from_config(cls(**_BASE, **extra)).eval()


@pytest.fixture(scope="module")
def tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    words = ["[UNK]", "[PAD]", "[EOS]", "the", "cat", "dog", "sat", "on", "a", "mat", "rug", "how", "make"]
    backend = Tokenizer(WordLevel({w: i for i, w in enumerate(words)}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]",
        model_input_names=["input_ids", "attention_mask"],
    )


HARMFUL = ["how make a mat", "the dog sat on the cat", "how make the rug"]
HARMLESS = ["the cat sat on a mat", "a dog on a rug", "the cat"]


@pytest.mark.parametrize("family", list(FAMILIES))
def test_replayed_selection_matches_expert_row_counts(family, tokenizer):
    model = build(family)
    result = routing_statistics(model, tokenizer, HARMFUL, HARMLESS, max_length=16)

    assert result["consistency"]["gate_vs_expert_counts_match"] is True, family
    assert result["model_type"] == family
    assert result["prompts"] == {"harmful": 3, "harmless": 3}
    assert result["moe_layers"] == len(result["layers"]) >= 1
    for layer in result["layers"]:
        k = layer["top_k"]
        assert k == 2
        for cls in ("harmful", "harmless"):
            stats = layer[cls]
            assert stats["prompts"] == 3 and stats["tokens"] > 0
            assert abs(sum(stats["selection_frac"]) - k) < 1e-6
            assert all(0.0 <= f <= 1.0 for f in stats["last_token_frac"])
            assert abs(sum(stats["last_token_frac"]) - k) < 1e-6
        assert len(layer["delta_frac"]) == layer["n_experts"]
    assert result["ranking"] and abs(result["ranking"][0]["delta"]) >= abs(result["ranking"][-1]["delta"])
    assert len(routing_table(result, top=3)) == 3

    # Hooks are gone afterwards.
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())


def test_capture_is_reset_between_prompts(tokenizer):
    model = build("qwen2_moe")
    ids = tokenizer("the cat sat", return_tensors="pt")["input_ids"]
    with RoutingCapture(model) as cap, torch.no_grad():
        model(ids, use_cache=False)
        first = {layer: list(rows) for layer, rows in cap.expert_rows.items()}
        assert sum(first[0]) == ids.shape[1] * 2
        cap.reset()
        assert sum(cap.expert_rows[0]) == 0 and not cap.selections
