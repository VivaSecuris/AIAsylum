"""MoE residual-write enumeration and surgery on tiny random models.

Every family here stores experts as an ``nn.ModuleList``, so each expert's
down-projection (and the shared expert's, where one exists) is a residual writer
the edit must reach. The regression pinned here: DeepSeek-V2/V3, GLM4-MoE and
Ernie4.5-MoE spell the shared expert ``shared_experts``, and were silently
under-edited while the manifest looked complete.
"""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from vivasecuris.aiasylum.interp.core import arch as arch_mod
from vivasecuris.aiasylum.interp.core.arch import KIND_EMBED, KIND_OUT, residual_write_plan

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

# family -> (config class, kwargs, expected matrices incl. embedding, expected shared-expert matrices)
FAMILIES = {
    "qwen2_moe": ("Qwen2MoeConfig", dict(
        num_experts=2, num_experts_per_tok=2, moe_intermediate_size=WIDTH,
        shared_expert_intermediate_size=WIDTH), 9, 2),
    "qwen3_moe": ("Qwen3MoeConfig", dict(
        num_experts=2, num_experts_per_tok=2, moe_intermediate_size=WIDTH, head_dim=WIDTH // 4), 7, 0),
    "mixtral": ("MixtralConfig", dict(num_local_experts=2, num_experts_per_tok=2), 7, 0),
    "deepseek_v3": ("DeepseekV3Config", _DEEPSEEK, 9, 1),
    "deepseek_v2": ("DeepseekV2Config", _DEEPSEEK, 9, 1),
    "glm4_moe": ("Glm4MoeConfig", dict(
        n_routed_experts=4, n_shared_experts=1, num_experts_per_tok=2, moe_intermediate_size=WIDTH,
        first_k_dense_replace=1, n_group=1, topk_group=1, head_dim=WIDTH // 4), 9, 1),
    "ernie4_5_moe": ("Ernie4_5_MoeConfig", dict(
        moe_num_experts=4, moe_num_shared_experts=1, moe_k=2, moe_intermediate_size=WIDTH,
        moe_layer_start_index=0, moe_layer_end_index=LAYERS), 13, 2),
    "olmoe": ("OlmoeConfig", dict(num_experts=4, num_experts_per_tok=2), 11, 0),
}
SHARED_FAMILIES = [f for f, (_, _, _, shared) in FAMILIES.items() if shared]


def build_moe(family: str):
    cls_name, extra, _, _ = FAMILIES[family]
    cls = getattr(transformers, cls_name, None)
    if cls is None:
        pytest.skip(f"{family} is unavailable in this transformers version")
    torch.manual_seed(17)
    return transformers.AutoModelForCausalLM.from_config(cls(**_BASE, **extra)).eval()


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.mark.parametrize("family", list(FAMILIES))
def test_plan_counts_per_family(family):
    _, _, expected, expected_shared = FAMILIES[family]
    plan = residual_write_plan(build_moe(family))

    assert len(plan.matrices) == expected
    assert plan.shared_expert_matrices == expected_shared
    assert plan.is_moe is True
    assert plan.model_type == family
    assert plan.coverage_verified is True
    assert plan.n_layers == LAYERS
    assert 0 < plan.moe_layers <= LAYERS
    assert plan.expert_matrices == sum(".mlp_experts." in m.name for m in plan.matrices)

    kinds = [m.kind for m in plan.matrices]
    assert kinds.count(KIND_EMBED) == 1
    # Every out-projection has the residual on axis 0; that is the axis the edit indexes.
    for m in plan.matrices:
        if m.kind == KIND_OUT:
            assert m.param.shape[0] == WIDTH, m.name
    assert len({m.param.data_ptr() for m in plan.matrices}) == len(plan.matrices)


def test_router_gate_is_never_a_target():
    """The gate reads the residual to choose experts; it is not a writer."""
    model = build_moe("qwen2_moe")
    gates = {block.mlp.gate.weight.data_ptr() for block in model.model.layers}
    shared_gates = {block.mlp.shared_expert_gate.weight.data_ptr() for block in model.model.layers}
    ptrs = {m.param.data_ptr() for m in residual_write_plan(model).matrices}
    assert not (ptrs & gates)
    assert not (ptrs & shared_gates)


def test_shared_experts_regression_is_caught_structurally(monkeypatch):
    """With the plural spelling un-probed, the audit must refuse rather than under-edit."""
    model = build_moe("deepseek_v3")
    monkeypatch.setattr(arch_mod, "_SHARED_EXPERT_ATTRS", ("shared_expert", "shared_mlp"))
    with pytest.raises(ValueError, match="did not reach"):
        residual_write_plan(model)


@pytest.mark.parametrize("family", SHARED_FAMILIES)
def test_ablation_removes_direction_on_moe(family):
    """After beta=0 the final residual has lost r, shared expert included.

    Before the fix the shared expert of every family here kept writing r back on
    every token, and this share sat well above the threshold.
    """
    from vivasecuris.aiasylum.weights.surgery import apply_to_model

    model = build_moe(family)
    torch.manual_seed(0)
    r = torch.randn(WIDTH)
    r = r / r.norm()
    ids = torch.randint(0, VOCAB, (2, 12))

    with torch.no_grad():
        before = model(ids, output_hidden_states=True).hidden_states[-1]
    assert (before @ r).abs().max() > 1e-3, "direction absent before ablation; test is vacuous"

    summary = apply_to_model(model, r, beta=0.0)
    assert summary["matrices_edited"] == FAMILIES[family][2]
    assert summary["shared_expert_matrices"] == FAMILIES[family][3]
    assert summary["coverage_verified"] is True

    with torch.no_grad():
        after = model(ids, output_hidden_states=True).hidden_states[-1]
    share = (after @ r).abs().mean() / after.norm(dim=-1).mean()
    assert share < 0.05, f"direction still carries {share:.3f} of the residual"


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


def test_manifest_records_moe_stats(tmp_path):
    from vivasecuris.aiasylum.weights.manifest import SurgeryManifest
    from vivasecuris.aiasylum.weights.surgery import edit_and_save

    src = tmp_path / "src"
    model = build_moe("qwen2_moe")
    model.save_pretrained(src)
    _word_level_tokenizer().save_pretrained(src)

    r = torch.zeros(WIDTH)
    r[0] = 1.0
    out = edit_and_save(str(src), r, str(tmp_path / "out"), beta=0.0, device="cpu", dtype="float32")

    manifest = SurgeryManifest.load(out)
    assert manifest is not None
    assert manifest.model_type == "qwen2_moe"
    assert manifest.coverage_verified is True
    assert manifest.moe_layers == LAYERS
    assert manifest.expert_matrices == 2 * LAYERS
    assert manifest.shared_expert_matrices == LAYERS
    assert manifest.matrices_edited == 9
    assert manifest.as_metadata()["surgery"]["coverage_verified"] is True
