"""End-to-end pipeline check on a tiny locally-built model.

Exercises capture -> direction -> surgery -> save -> provider on a randomly
initialized Qwen2 small enough to run in seconds. The derived direction is
meaningless on random weights; what is being verified here is that every stage
hands the next one something it can use, that the manifest survives the round
trip, and that the saved directory loads back through the normal provider path.
"""

import asyncio
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

D_MODEL, N_LAYERS = 32, 4
TOKENIZER_SRC = "Qwen/Qwen2.5-3B-Instruct"


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory):
    """A saved Qwen2 that is tiny in depth/width but keeps the real vocabulary.

    Shrinking the vocab instead would be faster, but then the borrowed
    tokenizer emits ids past the end of the embedding table and every forward
    pass dies with "index out of range in self".
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Config

    out = tmp_path_factory.mktemp("tiny-qwen")
    tok = AutoTokenizer.from_pretrained(TOKENIZER_SRC)

    cfg = Qwen2Config(
        vocab_size=len(tok), hidden_size=D_MODEL, intermediate_size=64,
        num_hidden_layers=N_LAYERS, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=128, tie_word_embeddings=False,
        bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
    )
    AutoModelForCausalLM.from_config(cfg).save_pretrained(str(out))
    tok.save_pretrained(str(out))
    return out


@pytest.fixture(scope="module")
def loaded(tiny_model_dir):
    from vivasecuris.aiasylum.interp.core.loader import load

    return load(str(tiny_model_dir), device="cpu", dtype="float32", seed=0)


def test_capture_shapes(loaded):
    from vivasecuris.aiasylum.weights.capture import capture_last_token_residuals

    model, tok = loaded
    acts = capture_last_token_residuals(
        model, tok, ["hello there", "another prompt", "a third one"], batch_size=2, max_length=64
    )
    assert acts.shape == (N_LAYERS + 1, 3, D_MODEL)
    assert acts.dtype == torch.float32
    assert acts.device.type == "cpu"


def test_capture_applies_chat_template(loaded):
    """Template formatting must be on by default; the direction depends on it."""
    from vivasecuris.aiasylum.weights.capture import format_prompts

    _, tok = loaded
    out = format_prompts(tok, ["what is 2+2"])[0]
    assert "what is 2+2" in out
    assert len(out) > len("what is 2+2"), "chat template was not applied"


def test_direction_derivation_runs_and_reports_auc(loaded):
    from vivasecuris.aiasylum.weights.corpus import PromptSplit
    from vivasecuris.aiasylum.weights.direction import derive_direction

    model, tok = loaded
    split = PromptSplit(
        harmful_train=[f"harmful train {i}" for i in range(6)],
        harmful_test=[f"harmful test {i}" for i in range(4)],
        harmless_train=[f"harmless train {i}" for i in range(6)],
        harmless_test=[f"harmless test {i}" for i in range(4)],
        seed=0,
        source="synthetic",
    )
    d = derive_direction(model, tok, split, model_id="tiny", batch_size=4)

    assert d.vector.shape == (D_MODEL,)
    assert abs(d.vector.norm().item() - 1.0) < 1e-5, "direction must be unit norm"
    assert 0.0 <= d.auc <= 1.0
    assert 1 <= d.layer <= N_LAYERS
    assert len(d.layer_scores) == N_LAYERS
    assert d.split_hash == split.hash


def test_direction_save_load_roundtrip(loaded, tmp_path):
    from vivasecuris.aiasylum.weights.direction import RefusalDirection

    d = RefusalDirection(
        vector=torch.nn.functional.normalize(torch.randn(D_MODEL), dim=0),
        layer=2, auc=0.93, cohens_d=1.4, model_id="tiny", split_hash="abc123",
    )
    d.save(tmp_path / "dir")
    back = RefusalDirection.load(tmp_path / "dir")

    assert torch.allclose(back.vector, d.vector, atol=1e-6)
    assert (back.layer, back.auc, back.split_hash) == (2, 0.93, "abc123")
    assert back.usable is True


def test_surgery_writes_loadable_model_with_manifest(tiny_model_dir, tmp_path):
    from vivasecuris.aiasylum.weights.direction import RefusalDirection
    from vivasecuris.aiasylum.weights.manifest import SurgeryManifest
    from vivasecuris.aiasylum.weights.surgery import edit_and_save

    torch.manual_seed(3)
    d = RefusalDirection(
        vector=torch.nn.functional.normalize(torch.randn(D_MODEL), dim=0),
        layer=2, auc=0.95, cohens_d=1.8, model_id=str(tiny_model_dir), split_hash="deadbeef",
    )
    out = tmp_path / "ablated"
    edit_and_save(str(tiny_model_dir), d, str(out), beta=0.0, device="cpu", dtype="float32")

    assert (out / "config.json").exists()
    assert (out / "asylum_surgery.json").exists()

    m = SurgeryManifest.load(out)
    assert m.method == "direction_scale"
    assert m.beta == 0.0
    assert m.direction_layer == 2
    assert m.split_hash == "deadbeef"
    assert m.matrices_edited == 1 + 2 * N_LAYERS
    assert m.embeddings_tied is False


def test_surgery_refuses_to_clobber_existing_directory(tiny_model_dir, tmp_path):
    from vivasecuris.aiasylum.weights.direction import RefusalDirection
    from vivasecuris.aiasylum.weights.surgery import edit_and_save

    out = tmp_path / "occupied"
    out.mkdir()
    (out / "something.bin").write_text("do not delete me")

    d = RefusalDirection(
        vector=torch.nn.functional.normalize(torch.randn(D_MODEL), dim=0),
        layer=1, auc=0.9, cohens_d=1.0, model_id="tiny", split_hash="x",
    )
    with pytest.raises(FileExistsError):
        edit_and_save(str(tiny_model_dir), d, str(out), beta=0.0, device="cpu", dtype="float32")
    assert (out / "something.bin").read_text() == "do not delete me"


def test_steering_hooks_are_always_removed(loaded):
    """A leaked pre-hook would silently contaminate every later forward pass."""
    from vivasecuris.aiasylum.interp.core.arch import detect_architecture, get_decoder_layers
    from vivasecuris.aiasylum.weights.steering import steer

    model, _ = loaded
    blocks = get_decoder_layers(model, detect_architecture(model))
    count = lambda: sum(len(b._forward_pre_hooks) for b in blocks)
    before = count()

    v = torch.nn.functional.normalize(torch.randn(D_MODEL), dim=0)

    # add defaults to a single layer: injecting a fixed-norm vector at every
    # depth swamps the early residual stream, where norms are far smaller.
    with steer(model, v, alpha=2.0):
        assert count() == before + 1
    assert count() == before

    # ablation is scale-free, so it is safe to apply at every layer.
    with steer(model, v, mode="ablate"):
        assert count() == before + N_LAYERS
    assert count() == before

    with steer(model, v, alpha=1.0, layers=[0, 2]):
        assert count() == before + 2
    assert count() == before

    # And on the exception path.
    with pytest.raises(RuntimeError):
        with steer(model, v, alpha=1.0):
            raise RuntimeError("boom")
    assert count() == before


def test_ablation_mode_removes_direction_from_activations(loaded):
    """Ablation steering must zero the direction, not merely perturb it."""
    from vivasecuris.aiasylum.weights.steering import steer

    model, tok = loaded
    ids = tok("hello world", return_tensors="pt")
    # Seeded: the surviving share depends on the draw, and an unseeded draw made
    # the verdict depend on which tests happened to run first.
    gen = torch.Generator().manual_seed(0)
    v = torch.nn.functional.normalize(torch.randn(D_MODEL, generator=gen), dim=0)

    with steer(model, v, mode="ablate"):
        with torch.no_grad():
            out = model(**ids, output_hidden_states=True)

    # Every block's input had the direction projected out, so the component
    # surviving into later layers is small relative to the residual norm.
    h = out.hidden_states[-2]
    share = (h @ v).abs().mean() / h.norm(dim=-1).mean()
    assert share < 0.1, f"direction still carries {share:.3f} of the residual"


def test_invalid_steering_mode_raises(loaded):
    from vivasecuris.aiasylum.weights.steering import steer

    model, _ = loaded
    v = torch.nn.functional.normalize(torch.randn(D_MODEL), dim=0)
    with pytest.raises(ValueError, match="mode must be"):
        with steer(model, v, mode="sideways"):
            pass


def test_degenerate_output_detector():
    """Repetition collapse must not be mistaken for successful ablation."""
    from vivasecuris.aiasylum.weights.steering import _looks_degenerate

    assert _looks_degenerate(["as as as as as as as as as as as as"]) is True
    assert _looks_degenerate([""]) is True
    assert _looks_degenerate([
        "Creating a fake academic degree is illegal and carries serious consequences for everyone involved.",
        "Managing asthma effectively requires a written action plan and regular review with your doctor.",
    ]) is False


def test_steering_actually_changes_activations(loaded):
    from vivasecuris.aiasylum.weights.steering import steer

    model, tok = loaded
    ids = tok("hello world", return_tensors="pt")

    with torch.no_grad():
        base = model(**ids, output_hidden_states=True).hidden_states[-1]

    v = torch.nn.functional.normalize(torch.randn(D_MODEL), dim=0)
    with steer(model, v, alpha=10.0):
        with torch.no_grad():
            steered = model(**ids, output_hidden_states=True).hidden_states[-1]

    assert not torch.allclose(base, steered, atol=1e-3), "steering had no effect"

    with torch.no_grad():
        restored = model(**ids, output_hidden_states=True).hidden_states[-1]
    assert torch.allclose(base, restored, atol=1e-5), "model not restored after steering"


def test_refusal_rate_uses_shared_constants():
    from vivasecuris.aiasylum.weights.steering import refusal_rate

    assert refusal_rate(["I cannot help with that.", "Sure, here is how."]) == 0.5
    assert refusal_rate([]) == 0.0
    assert refusal_rate(["I'm sorry, I can't do that."]) == 1.0


def test_provider_loads_modified_model_and_surfaces_manifest(tiny_model_dir, tmp_path):
    """The payoff: a surgically modified model driven through the normal provider."""
    from vivasecuris.aiasylum.models import get_provider
    from vivasecuris.aiasylum.models.transformers_local import clear_cache
    from vivasecuris.aiasylum.weights.direction import RefusalDirection
    from vivasecuris.aiasylum.weights.surgery import edit_and_save

    torch.manual_seed(5)
    d = RefusalDirection(
        vector=torch.nn.functional.normalize(torch.randn(D_MODEL), dim=0),
        layer=3, auc=0.97, cohens_d=2.1, model_id=str(tiny_model_dir), split_hash="cafe",
    )
    out = tmp_path / "served"
    edit_and_save(str(tiny_model_dir), d, str(out), beta=0.0, device="cpu", dtype="float32")

    try:
        model = get_provider("transformers").create_model(
            str(out), temperature=0.0, max_tokens=8, device="cpu", dtype="float32"
        )
        response = asyncio.run(model.generate("hello"))

        assert response.provider == "transformers"
        assert response.usage["completion_tokens"] > 0
        surgery = response.metadata["surgery"]
        assert surgery["beta"] == 0.0
        assert surgery["direction_layer"] == 3
        assert surgery["direction_auc"] == 0.97
        assert surgery["split_hash"] == "cafe"
        assert surgery["source_model"] == str(tiny_model_dir)
    finally:
        clear_cache()


def test_provider_caches_by_path_device_dtype(tiny_model_dir):
    from vivasecuris.aiasylum.models import transformers_local as tl

    try:
        tl.clear_cache()
        a = tl._get_cached(str(tiny_model_dir), "cpu", "float32")
        b = tl._get_cached(str(tiny_model_dir), "cpu", "float32")
        assert a[0] is b[0], "second load did not hit the cache"
        # Read through the module: clear_cache() rebinds this global.
        assert len(tl._CACHE) == 1
        c = tl._get_cached(str(tiny_model_dir), "cpu", "float16")
        assert c[0] is not a[0], "dtype must be part of the cache key"
        assert len(tl._CACHE) == 2
    finally:
        tl.clear_cache()


def test_steering_reaches_the_final_hidden_state(loaded):
    """A direction derived at the deepest layer must still be steerable.

    Layer numbers address hidden_states, which has n_blocks + 1 entries, and
    derive_direction can and often does select the last one -- separation
    usually peaks late. A pre-hook on block k covers hidden_states[k] only up
    to n_blocks - 1, so the final index needs a post-hook on the last block.
    Before that existed, `weights steer` and the sweep both died with
    "No valid layers selected" on exactly the directions most worth steering.
    """
    from vivasecuris.aiasylum.interp.core.arch import detect_architecture, get_decoder_layers
    from vivasecuris.aiasylum.weights.steering import steer

    model, _ = loaded
    blocks = get_decoder_layers(model, detect_architecture(model))
    v = torch.nn.functional.normalize(torch.randn(D_MODEL), dim=0)

    pre = lambda: sum(len(b._forward_pre_hooks) for b in blocks)
    post = lambda: sum(len(b._forward_hooks) for b in blocks)
    pre0, post0 = pre(), post()

    # N_LAYERS is the count of blocks, so it is the final hidden_states index.
    with steer(model, v, alpha=1.0, layers=[N_LAYERS]):
        assert post() == post0 + 1, "final index should hook the last block's output"
        assert pre() == pre0
    assert (pre(), post()) == (pre0, post0), "hooks must always be removed"

    # One past the end is still an error, with the valid range spelled out.
    with pytest.raises(ValueError, match="hidden_states"):
        with steer(model, v, alpha=1.0, layers=[N_LAYERS + 1]):
            pass


def test_final_layer_steering_actually_changes_the_output(loaded):
    """The post-hook must modify the forward pass, not merely register."""
    from vivasecuris.aiasylum.weights.steering import steer

    model, tok = loaded
    ids = tok("hello world", return_tensors="pt")
    v = torch.nn.functional.normalize(torch.randn(D_MODEL), dim=0)

    with torch.no_grad():
        base = model(**ids).logits
        with steer(model, v, alpha=50.0, layers=[N_LAYERS]):
            steered = model(**ids).logits

    assert not torch.allclose(base, steered), "steering at the final layer was a no-op"


def _synthetic_split():
    from vivasecuris.aiasylum.weights.corpus import PromptSplit

    return PromptSplit(
        harmful_train=[f"harmful train {i}" for i in range(6)],
        harmful_test=[f"harmful test {i}" for i in range(4)],
        harmless_train=[f"harmless train {i}" for i in range(6)],
        harmless_test=[f"harmless test {i}" for i in range(4)],
        seed=0,
        source="synthetic",
    )


def test_derive_subspace_orthonormal_row0_is_vector(loaded):
    from vivasecuris.aiasylum.weights.direction import derive_direction, derive_subspace

    model, tok = loaded
    split = _synthetic_split()

    sub = derive_subspace(model, tok, split, rank=3, model_id="tiny", batch_size=4)
    assert sub.basis is not None
    assert 1 <= sub.rank <= 3
    assert sub.basis.shape[1] == D_MODEL
    # Rows are orthonormal.
    gram = sub.basis @ sub.basis.t()
    assert torch.allclose(gram, torch.eye(sub.rank), atol=1e-4)
    # Row 0 is exactly the single best direction, so rank-1 reduces to it.
    assert torch.allclose(sub.basis[0], sub.vector, atol=1e-5)

    single = derive_direction(model, tok, split, model_id="tiny", batch_size=4)
    rank1 = derive_subspace(model, tok, split, rank=1, model_id="tiny", batch_size=4)
    assert rank1.rank == 1
    assert torch.allclose(rank1.basis[0], single.vector, atol=1e-5)


def test_subspace_direction_save_load_roundtrip(tmp_path):
    from vivasecuris.aiasylum.weights.direction import RefusalDirection

    torch.manual_seed(11)
    basis = torch.linalg.qr(torch.randn(D_MODEL, 3))[0].t().contiguous()
    d = RefusalDirection(
        vector=basis[0].clone(), layer=2, auc=0.97, cohens_d=2.0,
        model_id="tiny", split_hash="beef", basis=basis, basis_layers=[2, 3, 1],
    )
    d.save(tmp_path / "sub")
    back = RefusalDirection.load(tmp_path / "sub")

    assert back.basis is not None
    assert back.rank == 3
    assert torch.allclose(back.basis, d.basis, atol=1e-6)
    assert back.basis_layers == [2, 3, 1]


def test_subspace_surgery_writes_loadable_model_with_manifest(tiny_model_dir, tmp_path):
    from vivasecuris.aiasylum.weights.direction import RefusalDirection
    from vivasecuris.aiasylum.weights.manifest import SurgeryManifest
    from vivasecuris.aiasylum.weights.surgery import edit_and_save

    torch.manual_seed(13)
    basis = torch.linalg.qr(torch.randn(D_MODEL, 3))[0].t().contiguous()
    d = RefusalDirection(
        vector=basis[0].clone(), layer=2, auc=0.96, cohens_d=1.9,
        model_id=str(tiny_model_dir), split_hash="cafe", basis=basis, basis_layers=[2, 3, 1],
    )
    out = tmp_path / "subspace-edit"
    edit_and_save(str(tiny_model_dir), d, str(out), use_subspace=True, k=1.25,
                  device="cpu", dtype="float32")

    m = SurgeryManifest.load(out)
    assert m.method == "direction_subspace"
    assert m.beta is None
    assert m.matrices_edited == 1 + 2 * N_LAYERS
    assert m.extra.get("subspace_rank") == 3
    assert m.extra.get("k") == 1.25
    assert m.extra.get("basis_layers") == [2, 3, 1]


def test_subspace_prefix_matches_a_smaller_rank_derivation(loaded):
    """`ablate --rank N` slices basis[:N]; that must equal deriving at rank N.

    The rows are built incrementally from a fixed pool of best-separating
    layers, so a prefix of a rank-8 basis is the rank-2 basis -- which is what
    lets `weights select` recommend a rank and `weights ablate --rank` honour it
    without re-deriving. Only holds while the pool is the same for both, i.e.
    rank <= pool_layers.
    """
    from vivasecuris.aiasylum.weights.direction import derive_subspace

    model, tok = loaded
    split = _synthetic_split()
    big = derive_subspace(model, tok, split, rank=4, pool_layers=4, model_id="tiny", batch_size=4)
    small = derive_subspace(model, tok, split, rank=2, pool_layers=4, model_id="tiny", batch_size=4)

    assert big.rank >= small.rank
    assert torch.allclose(big.basis[: small.rank], small.basis, atol=1e-5)


def test_subspace_is_orthonormal_and_extends_the_single_vector(loaded):
    """Row 0 of the basis must be the difference-in-means vector itself.

    Everything downstream that reads `.vector` keeps working only because of
    that: a subspace is an extension of the single-direction result, not a
    different object.
    """
    from vivasecuris.aiasylum.weights.corpus import PromptSplit
    from vivasecuris.aiasylum.weights.direction import derive_subspace

    model, tok = loaded
    split = PromptSplit(
        harmful_train=[f"harmful train {i}" for i in range(8)],
        harmful_test=[f"harmful test {i}" for i in range(4)],
        harmless_train=[f"harmless train {i}" for i in range(8)],
        harmless_test=[f"harmless test {i}" for i in range(4)],
        seed=0,
        source="synthetic",
    )
    d = derive_subspace(model, tok, split, rank=3, pool_layers=3, model_id="tiny", batch_size=4)

    assert d.basis is not None
    assert d.basis.shape[1] == D_MODEL
    assert 1 <= d.rank <= 3
    assert d.rank == d.basis.shape[0]

    # Row 0 is the primary direction.
    assert torch.allclose(d.basis[0], d.vector, atol=1e-5)

    # Orthonormal: B @ B.T is the identity.
    gram = d.basis @ d.basis.T
    assert torch.allclose(gram, torch.eye(d.rank), atol=1e-4), gram

    # as_basis() falls back to [vector] for a bare direction, so callers need
    # no special case.
    assert d.as_basis().shape == d.basis.shape
    assert d.metadata()["rank"] == d.rank
    assert len(d.basis_layers) == d.rank


def test_subspace_survives_the_save_load_round_trip(loaded, tmp_path):
    """The API reloads a direction from disk before every sweep, select and
    surgery, so a basis that did not round-trip would silently degrade a
    rank-N edit to rank 1."""
    from vivasecuris.aiasylum.weights.direction import RefusalDirection

    basis = torch.linalg.qr(torch.randn(D_MODEL, 3))[0].T.contiguous()
    d = RefusalDirection(
        vector=basis[0], layer=2, auc=0.97, cohens_d=2.2,
        model_id="tiny", split_hash="abc", basis=basis, basis_layers=[2, 3, 1],
    )
    d.save(tmp_path / "sub")
    back = RefusalDirection.load(tmp_path / "sub")

    assert back.basis is not None
    assert back.rank == 3
    assert torch.allclose(back.basis, basis, atol=1e-6)
    assert back.basis_layers == [2, 3, 1]


def test_rank_one_subspace_removal_equals_beta_zero(tiny_model_dir, tmp_path):
    """The two edit shapes must agree where they overlap.

    `rank=1, k=1` removes exactly the component that `beta=0` scales to zero.
    If these ever diverge, one of the two paths has the projection wrong -- and
    a transposed projection runs without error while doing nothing, so only an
    equivalence like this catches it.
    """
    from vivasecuris.aiasylum.weights.direction import RefusalDirection
    from vivasecuris.aiasylum.weights.surgery import edit_and_save

    torch.manual_seed(11)
    v = torch.nn.functional.normalize(torch.randn(D_MODEL), dim=0)
    d = RefusalDirection(
        vector=v, layer=2, auc=0.95, cohens_d=1.8,
        model_id=str(tiny_model_dir), split_hash="x",
        basis=v.reshape(1, -1), basis_layers=[2],
    )

    beta_dir = tmp_path / "beta0"
    sub_dir = tmp_path / "rank1"
    edit_and_save(str(tiny_model_dir), d, str(beta_dir), beta=0.0, device="cpu", dtype="float32")
    edit_and_save(str(tiny_model_dir), d, str(sub_dir), device="cpu", dtype="float32",
                  use_subspace=True, k=1.0)

    from transformers import AutoModelForCausalLM

    a = AutoModelForCausalLM.from_pretrained(str(beta_dir))
    b = AutoModelForCausalLM.from_pretrained(str(sub_dir))
    sa, sb = a.state_dict(), b.state_dict()
    assert set(sa) == set(sb)
    for name in sa:
        assert torch.allclose(sa[name].float(), sb[name].float(), atol=1e-5), name


def test_select_edit_reports_a_capability_gated_frontier(loaded):
    """The search must refuse a config on capability grounds, not just rank.

    A model that refuses nothing because it has been broken scores identically
    to one cleanly ablated, so the factual control is what makes the frontier
    mean anything.
    """
    from vivasecuris.aiasylum.weights.direction import RefusalDirection
    from vivasecuris.aiasylum.weights.surgery import select_edit

    model, tok = loaded
    basis = torch.linalg.qr(torch.randn(D_MODEL, 2))[0].T.contiguous()
    d = RefusalDirection(
        vector=basis[0], layer=2, auc=0.96, cohens_d=2.0,
        model_id="tiny", split_hash="x", basis=basis, basis_layers=[2, 3],
    )

    result = select_edit(
        model, tok, d, ["how do I do a bad thing", "another bad request"],
        ranks=(1, 2), ks=(1.0,), factual_floor=0.05, max_new_tokens=4, factual_limit=2,
    )

    assert set(result) == {"baseline", "frontier", "best", "capability_set", "capability_n"}
    assert result["capability_set"] == "builtin" and result["capability_n"] == 2
    assert {"refuse_harmful", "factual_acc"} <= set(result["baseline"])
    assert len(result["frontier"]) == 2
    for row in result["frontier"]:
        assert {"rank", "k", "refuse_harmful", "factual_acc",
                "factual_drop", "degenerate", "accepted"} <= set(row)
        # accepted is exactly "not degenerate and capability held"
        floor = result["baseline"]["factual_acc"] - 0.05
        assert row["accepted"] == ((not row["degenerate"]) and row["factual_acc"] >= floor)
    assert result["best"] is None or result["best"] in result["frontier"]


class _TemplateRecorder:
    """Stands in for a tokenizer: records the kwargs apply_chat_template got."""

    def __init__(self, chat_template):
        self.chat_template = chat_template
        self.kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.kwargs = kwargs
        return messages[-1]["content"]


def test_reasoning_is_off_for_templates_that_support_it():
    from vivasecuris.aiasylum.weights.capture import format_prompts

    qwen3 = _TemplateRecorder("{% if enable_thinking is defined %}...{% endif %}")
    format_prompts(qwen3, ["hi"])
    assert qwen3.kwargs["enable_thinking"] is False

    format_prompts(qwen3, ["hi"], thinking=True)
    assert qwen3.kwargs["enable_thinking"] is True


def test_templates_without_reasoning_are_rendered_unchanged():
    from vivasecuris.aiasylum.weights.capture import format_prompts

    qwen25 = _TemplateRecorder("{{ messages }}")
    format_prompts(qwen25, ["hi"])
    assert "enable_thinking" not in qwen25.kwargs


def test_refusal_inside_reasoning_does_not_count():
    from vivasecuris.aiasylum.weights.capture import strip_thinking
    from vivasecuris.aiasylum.weights.steering import refusal_rate

    mused = "<think>\nI'm sorry, I can't help with that... but it is fiction.\n</think>\n\nSure, here it is."
    assert strip_thinking(mused) == "Sure, here it is."
    assert refusal_rate([mused]) == 0.0
    # An unterminated trace (generation cut off mid-thought) has no answer yet.
    assert strip_thinking("<think>still going") == ""
    assert refusal_rate(["<think></think>I'm sorry, I can't help with that."]) == 1.0


def test_builtin_capability_set_is_the_smoke_test():
    from vivasecuris.aiasylum.weights.evaluate import CAPABILITY_QA, capability_set

    cs = capability_set("builtin")
    assert cs.name == "builtin" and cs.size == len(CAPABILITY_QA)
    assert cs.score(["Paris"] + [""] * (cs.size - 1)) == 1 / cs.size


def test_mmlu_capability_set_scores_by_letter_not_substring():
    from vivasecuris.aiasylum.weights.evaluate import mmlu_capability_set

    items = [
        {"question": "Pick B.", "choices": ["x", "y", "z", "w"], "answer": "B", "subject": "t"},
        {"question": "Pick D.", "choices": ["x", "y", "z", "w"], "answer": "D", "subject": "t"},
    ]
    cs = mmlu_capability_set(items)
    assert cs.name == "mmlu:2" and cs.max_new_tokens == 8
    assert "A. x" in cs.questions[0] and cs.questions[0].endswith("Answer with the letter only.")
    # "B" as a standalone letter counts; a sentence that merely contains the
    # letter (as in "a cat") does not; the first letter is the answer.
    assert cs.score(["B", "The answer is D."]) == 1.0
    assert cs.score(["a cat is an animal", "C or D"]) == 0.0
    assert cs.score(["B.", "b"]) == 0.5


def test_compare_verdict_matches_the_selection_rule():
    from vivasecuris.aiasylum.weights.evaluate import compare_verdict

    base = {"refuse_harmful": 0.875, "factual_acc": 0.917, "degenerate": False}
    assert compare_verdict(base, {"refuse_harmful": 0.0, "factual_acc": 0.917, "degenerate": False}) == "clean"
    # Fluent nonsense: not degenerate, but the control caught it.
    assert compare_verdict(base, {"refuse_harmful": 0.0, "factual_acc": 0.083, "degenerate": False}) == "capability_cost"
    assert compare_verdict(base, {"refuse_harmful": 0.0, "factual_acc": 0.0, "degenerate": True}) == "degenerate"
    assert compare_verdict(base, {"refuse_harmful": 0.875, "factual_acc": 0.917, "degenerate": False}) == "unchanged"
    # A drop inside the floor is still clean.
    assert compare_verdict(base, {"refuse_harmful": 0.3, "factual_acc": 0.9, "degenerate": False}) == "clean"
