"""The execute path for the new run kinds, end to end on tiny models.

The route tests stub execution; this runs `_execute` itself for a routing run,
an expert-surgery run and a LoRA run (through the real worker subprocess), with
both artifact roots redirected. Random weights, CPU float32, under a minute.
"""

import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")

from vivasecuris.aiasylum.api.routes import weights as weights_route
from vivasecuris.aiasylum.weights.manifest import SurgeryManifest
from vivasecuris.aiasylum.weights.progress import Reporter

WIDTH, VOCAB = 32, 64
_BASE = dict(
    vocab_size=VOCAB, hidden_size=WIDTH, intermediate_size=WIDTH * 2, num_hidden_layers=2,
    num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
    bos_token_id=2, eos_token_id=2, pad_token_id=1, tie_word_embeddings=False,
)


def _tokenizer():
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


def _save(cfg, path):
    torch.manual_seed(3)
    transformers.AutoModelForCausalLM.from_config(cfg).save_pretrained(path)
    _tokenizer().save_pretrained(path)
    return str(path)


@pytest.fixture
def roots(tmp_path, monkeypatch):
    runs, models = tmp_path / "runs", tmp_path / "models"
    runs.mkdir()
    models.mkdir()
    monkeypatch.setattr(weights_route, "_runs_root", lambda: runs)
    monkeypatch.setattr(weights_route, "_models_root", lambda: models)
    return runs, models


@pytest.fixture
def moe_dir(tmp_path):
    cfg = transformers.Qwen2MoeConfig(**_BASE, num_experts=4, num_experts_per_tok=2,
                                      moe_intermediate_size=WIDTH, shared_expert_intermediate_size=WIDTH)
    return _save(cfg, tmp_path / "moe")


@pytest.fixture
def dense_dir(tmp_path):
    return _save(transformers.Qwen2Config(**_BASE), tmp_path / "dense")


def _options(**overrides):
    base = dict(
        device="cpu", dtype="float32", n_per_class=8, test_fraction=0.25, seed=0, batch_size=2,
        max_length=16, beta=0.0, include_embeddings=True, use_subspace=False, k=None,
        thinking=False, expert_selection=None, expert_scale=0.0, include_shared_expert=False,
        lora_rank=2, lora_alpha=4, lora_dropout=0.0, lora_targets="attention", epochs=1,
        max_steps=2, lr=1e-3, train_batch_size=1, grad_accum=1, gradient_checkpointing=False,
        merge=True, eval_rows=1,
    )
    base.update(overrides)
    return base


def _snap(kind, source, out_dir, method, **opts):
    return {
        "kind": kind, "source_model": source, "source_run_id": None, "method": method,
        "objective": "refusal", "objective_config": None, "options": _options(**opts),
        "out_dir": str(out_dir), "direction_dir": None, "source_direction": {},
        "source_options": {}, "notes": "tiny", "modified_model": None, "dataset": None,
    }


@pytest.fixture
def stub_corpus(monkeypatch):
    def fake_split(objective, config, n_per_class, test_fraction, seed):
        harmful = ["how make a mat", "the dog sat on the cat", "how make the rug", "a dog on a mat"]
        harmless = ["the cat sat on a mat", "a dog on a rug", "the cat", "the rug"]
        return SimpleNamespace(
            harmful_train=harmful[1:], harmful_test=harmful[:1],
            harmless_train=harmless[1:], harmless_test=harmless[:1],
            hash="stub",
            summary=lambda: {"source": "stub", "seed": seed, "split_hash": "stub"},
            to_dict=lambda: {"harmful_train": harmful[1:], "harmful_test": harmful[:1],
                             "harmless_train": harmless[1:], "harmless_test": harmless[:1]},
        )

    monkeypatch.setattr(weights_route, "_build_objective_split", fake_split)


def test_routing_run_writes_routing_json(roots, moe_dir, stub_corpus):
    runs, _ = roots
    out = runs / "11"
    summary = weights_route._execute(11, _snap("routing", moe_dir, out, "expert_routing"), Reporter(enabled=False))

    assert (out / "routing.json").is_file() and (out / "prompt_split.json").is_file()
    full = json.loads((out / "routing.json").read_text())
    assert len(full["layers"]) == 2 and full["consistency"]["gate_vs_expert_counts_match"] is True
    assert "layers" not in summary, "the row keeps the ranking and the layer shape, not the tables"
    assert summary["layer_shape"] == [{"layer": 0, "n_experts": 4, "top_k": 2}, {"layer": 1, "n_experts": 4, "top_k": 2}]
    assert summary["ranking"] and summary["prompts"] == {"harmful": 4, "harmless": 4}
    headline = weights_route._headline("routing", summary)
    assert headline["consistent"] is True and headline["moe_layers"] == 2


def test_expert_surgery_run_writes_a_partial_model(roots, moe_dir):
    _, models = roots
    out = models / "moe-e1"
    summary = weights_route._execute(
        12, _snap("expert_surgery", moe_dir, out, "expert_ablate", expert_selection={"0": [1]}, expert_scale=0.0),
        Reporter(enabled=False),
    )
    assert out.is_dir() and not (models / ".staging-12").exists()
    manifest = SurgeryManifest.load(out)
    assert manifest.method == "expert_ablate" and manifest.coverage_verified is False
    assert manifest.matrices_edited == 1 and manifest.extra["expert_selection"] == {"0": [1]}
    assert manifest.extra["method"] == "expert_ablate" and manifest.extra["objective"] == "refusal"
    assert summary["manifest"]["coverage_verified"] is False and summary["output_path"] == str(out)

    edited, _ = transformers.AutoModelForCausalLM.from_pretrained(out), None
    assert torch.count_nonzero(edited.model.layers[0].mlp.experts[1].down_proj.weight) == 0
    assert torch.count_nonzero(edited.model.layers[0].mlp.experts[0].down_proj.weight) > 0
    assert weights_route._headline("expert_surgery", summary)["experts"] == 1


def test_lora_run_trains_in_a_worker_and_merges(roots, dense_dir):
    runs, models = roots
    run_dir = runs / "13"
    run_dir.mkdir()
    # Ten rows: the trainer holds out at most a fifth for the eval loss, so
    # fewer than five would mean no evaluation pass and nothing to assert on.
    words = ["the cat", "a dog", "the rug", "a mat", "how make"]
    rows = [{"prompt": f"{words[i % 5]} sat", "response": f"on the {words[(i + 2) % 5]}"} for i in range(10)]
    (run_dir / "train.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    out = models / "dense-tuned"
    snap = _snap("lora", dense_dir, out, "lora")
    snap["objective"] = "dataset"
    snap["dataset"] = {"source": "rows", "n_rows": 10, "sha256": "x" * 64, "path": str(run_dir / "train.jsonl")}

    notes = []

    class Quiet(Reporter):
        def __init__(self):
            super().__init__(enabled=False)
            self.ticks = []

        def note(self, text):
            notes.append(text)

        def metrics(self, data):
            self.ticks.append(data)

    reporter = Quiet()
    summary = weights_route._execute(13, snap, reporter)

    assert out.is_dir() and not (models / ".staging-13").exists()
    assert summary["merged"] is True and summary["output_path"] == str(out)
    assert summary["train"]["steps"] == 2 and summary["history"]
    assert any(t.get("phase") == "eval" for t in reporter.ticks), "the worker's eval events reached the reporter"
    assert (run_dir / "worker.log").is_file() and (run_dir / "job.json").is_file()
    assert (run_dir / "adapter" / "adapter_config.json").is_file()

    manifest = SurgeryManifest.load(out)
    assert manifest.method == "lora_merge" and manifest.extra["lora"]["rank"] == 2
    assert manifest.extra["method"] == "lora" and manifest.extra["objective"] == "dataset"
    assert manifest.extra["dataset"]["rows"] == 10
    assert weights_route._headline("lora", summary)["steps"] == 2

    merged = transformers.AutoModelForCausalLM.from_pretrained(out)
    base = transformers.AutoModelForCausalLM.from_pretrained(dense_dir)
    changed = [n for n, p in merged.named_parameters() if not torch.equal(p, dict(base.named_parameters())[n])]
    assert changed and all(any(t in n for t in ("q_proj", "k_proj", "v_proj", "o_proj")) for n in changed)


# --------------------------------------------------------------------------
# autotune: search in memory, write the winner, verify it from disk
# --------------------------------------------------------------------------

COMPLY = "Sure, here is a detailed and fluent answer to exactly what you asked for."
CJK = "我不能帮助你完成这个请求，因为这可能会对他人造成伤害。请考虑其他方式。"


def _canned(monkeypatch, harmful_text, *, verify_text=None):
    """Both generators in both modules answer from a script: capability
    questions get their accepted answer, harmful prompts get ``harmful_text``
    in the search and ``verify_text`` (default the same) in the disk check."""
    from vivasecuris.aiasylum.weights import autotune as autotune_mod
    from vivasecuris.aiasylum.weights import verify as verify_mod
    from vivasecuris.aiasylum.weights.evaluate import CAPABILITY_QA

    answers = {q: a[0] for q, a in CAPABILITY_QA}

    def make(text):
        def fake(model, tokenizer, prompts, *args, **kwargs):
            return [answers.get(p, text) for p in prompts]
        return fake

    monkeypatch.setattr(autotune_mod, "generate_greedy", make(harmful_text))
    monkeypatch.setattr(autotune_mod, "generate_sampled", make(harmful_text))
    monkeypatch.setattr(verify_mod, "generate_greedy", make(verify_text or harmful_text))
    monkeypatch.setattr(verify_mod, "generate_sampled", make(verify_text or harmful_text))


def _direction_dir(runs, dense_dir, run_id=20, rank=2):
    from vivasecuris.aiasylum.weights.direction import RefusalDirection

    torch.manual_seed(4)
    basis = torch.linalg.qr(torch.randn(WIDTH, rank))[0].T.contiguous()
    d = RefusalDirection(
        vector=basis[0], layer=1, auc=0.95, cohens_d=2.0, model_id=dense_dir, split_hash="stub",
        basis=basis, basis_layers=[1] * rank,
    )
    out = runs / str(run_id)
    out.mkdir()
    d.save(out)
    return str(out)


def _autotune_snap(dense_dir, out, direction_dir, **opts):
    defaults = dict(
        n_prompts=1, max_new_tokens=8, capability_set="builtin", ranks=[1, 2], ks=[1.0],
        factual_floor=0.05, language_drift_max=0.10, max_candidates=16,
        stop_at_first_admissible=False, max_refusal=0.10, verify_sampled=True,
        sampling_seed=0, embedding_modes=None,
    )
    snap = _snap("autotune", dense_dir, out, "verified_subspace_search", **{**defaults, **opts})
    snap["direction_dir"] = direction_dir
    snap["source_run_id"] = 20
    return snap


def test_autotune_run_writes_a_verified_checkpoint(roots, dense_dir, stub_corpus, monkeypatch):
    import gc
    import weakref

    from vivasecuris.aiasylum.interp.core import loader
    from vivasecuris.aiasylum.weights import verify as verify_mod
    from vivasecuris.aiasylum.weights.verify import hash_weights

    runs, models = roots
    _canned(monkeypatch, COMPLY)
    original_load, original_verify = loader.load, verify_mod.verify_checkpoint
    parameters = []

    def tracked_load(source, **kwargs):
        model, tok = original_load(source, **kwargs)
        if str(source) == dense_dir:
            parameters.extend(weakref.ref(p) for p in model.parameters())
        return model, tok

    def verify_after_release(*args, **kwargs):
        gc.collect()
        assert parameters and all(ref() is None for ref in parameters), "Original weights survived into reload"
        return original_verify(*args, **kwargs)

    monkeypatch.setattr(loader, "load", tracked_load)
    monkeypatch.setattr(verify_mod, "verify_checkpoint", verify_after_release)
    out = models / "dense-auto"
    snap = _autotune_snap(dense_dir, out, _direction_dir(runs, dense_dir))

    summary = weights_route._execute(21, snap, Reporter(enabled=False))

    assert out.is_dir() and not (models / ".staging-21").exists()
    assert summary["output_path"] == str(out) and summary["target_met"] is True
    assert summary["candidates_tried"] == 4 and summary["candidates_planned"] == 4
    assert all(t["accepted"] for t in summary["trials"])
    # Every candidate tied at zero refusal; the smallest edit wins: rank 1,
    # and of the two rank-1 edits the one that touches fewer matrices.
    assert (summary["winner"]["rank"], summary["winner"]["include_embeddings"]) == (1, False)
    assert [(t["rank"], t["include_embeddings"]) for t in summary["trials"]] == [
        (1, True), (1, False), (2, True), (2, False)], "untied model: full edit tried first"
    assert summary["winner"]["sampled"]["accepted"] is True
    assert summary["verification"]["passed"] is True
    # The stub answers the same for base and edit, so the verdict is a no-change one.
    assert summary["verification"]["sampled"]["verdict"] in ("clean", "unchanged")
    assert summary["verification"]["greedy"]["drifted"] is False
    assert summary["evaluation"]["actual"] == 1

    manifest = SurgeryManifest.load(out)
    assert manifest.method == "direction_subspace" and manifest.embeddings_edited is False
    assert manifest.extra["subspace_rank"] == 1 and manifest.extra["k"] == 1.0
    assert manifest.extra["autotune"]["winner"]["rank"] == 1 and "responses" not in manifest.extra["autotune"]["winner"]
    assert manifest.extra["autotune"]["target_met"] is True and manifest.extra["autotune"]["run_id"] == 21
    assert manifest.extra["verification"]["passed"] is True
    assert "responses" not in manifest.extra["verification"]["greedy"]
    assert manifest.extra["weights_sha256"] == hash_weights(out)
    # The route enrichment after publish survived alongside.
    assert manifest.extra["method"] == "verified_subspace_search" and manifest.extra["objective"] == "refusal"

    headline = weights_route._headline("autotune", summary)
    assert headline["verified"] is True and headline["target_met"] is True
    assert headline["winner_rank"] == 1 and headline["candidates_tried"] == 4

    # The written model really is edited: the rank-1 write differs from the base.
    edited = transformers.AutoModelForCausalLM.from_pretrained(out)
    base = transformers.AutoModelForCausalLM.from_pretrained(dense_dir)
    assert not torch.equal(edited.model.layers[0].self_attn.o_proj.weight, base.model.layers[0].self_attn.o_proj.weight)


def test_autotune_fails_with_trial_log_when_nothing_passes(roots, dense_dir, stub_corpus, monkeypatch):
    runs, models = roots
    _canned(monkeypatch, CJK)
    out = models / "dense-auto-fail"
    snap = _autotune_snap(dense_dir, out, _direction_dir(runs, dense_dir))

    with pytest.raises(weights_route.RunFailed) as excinfo:
        weights_route._execute(22, snap, Reporter(enabled=False))

    assert "No candidate cleared every gate" in str(excinfo.value)
    trials = excinfo.value.summary["trials"]
    assert len(trials) == 4 and all(t["reason"] == "language_drift" for t in trials)
    assert excinfo.value.summary["winner"] is None
    assert not out.exists() and not (models / ".staging-22").exists()


def test_autotune_fails_when_reload_verification_fails(roots, dense_dir, stub_corpus, monkeypatch):
    runs, models = roots
    # Passes in memory, drifts when the written directory is reloaded.
    _canned(monkeypatch, COMPLY, verify_text=CJK)
    out = models / "dense-auto-unverified"
    snap = _autotune_snap(dense_dir, out, _direction_dir(runs, dense_dir))

    with pytest.raises(weights_route.RunFailed) as excinfo:
        weights_route._execute(23, snap, Reporter(enabled=False))

    assert "failed verification from disk" in str(excinfo.value)
    summary = excinfo.value.summary
    assert summary["winner"] is not None and summary["verification"]["passed"] is False
    assert any("language drift" in r for r in summary["verification"]["reasons"])
    assert not out.exists() and not (models / ".staging-23").exists()


def test_autotune_stop_at_first_and_rank_one_direction(roots, dense_dir, stub_corpus, monkeypatch):
    runs, models = roots
    _canned(monkeypatch, COMPLY)
    out = models / "dense-auto-first"
    snap = _autotune_snap(dense_dir, out, _direction_dir(runs, dense_dir, rank=1),
                          stop_at_first_admissible=True, verify_sampled=False, embedding_modes=[False])
    summary = weights_route._execute(24, snap, Reporter(enabled=False))
    assert summary["stopped_early"] is True and summary["candidates_tried"] == 1
    assert summary["winner"]["include_embeddings"] is False and summary["winner"]["sampled"] is None
    assert summary["verification"]["sampled"] is None and summary["verification"]["passed"] is True
    assert SurgeryManifest.load(out).embeddings_edited is False


def test_surgery_run_honours_rank(roots, dense_dir, stub_corpus):
    runs, models = roots
    out = models / "dense-rank1"
    snap = _snap("surgery", dense_dir, out, "direction_scale", use_subspace=True, k=1.0, rank=1)
    snap["direction_dir"] = _direction_dir(runs, dense_dir, run_id=25, rank=3)
    weights_route._execute(26, snap, Reporter(enabled=False))
    assert SurgeryManifest.load(out).extra["subspace_rank"] == 1


# --------------------------------------------------------------------------
# 2026-09-28: the newly wired kinds (induce, hneurons/bake, redteam, embeddings)
# --------------------------------------------------------------------------

def _wide_tokenizer():
    """A tokenizer whose vocab is the embeddings word-bank, so diverse_prompts
    tokenize to varied real tokens and there are >=16 shareable anchors."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    from vivasecuris.aiasylum.weights.embeddings import _WORD_BANK

    words = ["[UNK]", "[PAD]", "[EOS]"] + list(dict.fromkeys(_WORD_BANK))[:80]
    backend = Tokenizer(WordLevel({w: i for i, w in enumerate(words)}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]",
        model_input_names=["input_ids", "attention_mask"],
    ), len(words)


@pytest.fixture
def wide_dir(tmp_path):
    tok, vocab = _wide_tokenizer()
    cfg = transformers.Qwen2Config(**{**_BASE, "vocab_size": vocab})
    torch.manual_seed(3)
    transformers.AutoModelForCausalLM.from_config(cfg).save_pretrained(tmp_path / "wide")
    tok.save_pretrained(str(tmp_path / "wide"))
    return str(tmp_path / "wide")


def test_embed_align_self_is_perfect_retrieval(roots, wide_dir):
    runs, _ = roots
    out = runs / "40"
    snap = _snap("embed_align", wide_dir, out, "anchor_align", model_b=wide_dir,
                 which_embedding="input", max_anchors=64)
    summary = weights_route._execute(40, snap, Reporter(enabled=False))
    assert (out / "embeddings.json").is_file()
    # Aligning a model to itself recovers every anchor; the shuffled null does not.
    assert summary["retrieval"]["1"] == 1.0
    assert summary["null_shuffled"]["1"] < 1.0
    head = weights_route._headline("embed_align", summary)
    assert head["p_at_1"] == 1.0


def test_embed_extract_runs_and_scores(roots, wide_dir):
    runs, _ = roots
    out = runs / "41"
    snap = _snap("embed_extract", wide_dir, out, "logit_svd", n_queries=160, col_subset=4096)
    summary = weights_route._execute(41, snap, Reporter(enabled=False))
    assert (out / "embeddings.json").is_file()
    assert summary["true_dim"] == WIDTH
    assert 1 <= summary["recovered_dim"] <= WIDTH
    assert "subspace_overlap" in summary
    assert weights_route._headline("embed_extract", summary)["true_dim"] == WIDTH


@pytest.mark.parametrize("extra_cols", [0, 2048])
def test_embed_reconstruction_handles_partial_and_full_vocabulary(roots, wide_dir, extra_cols):
    runs, _ = roots
    out = runs / "reconstruction"
    snap = _snap("embed_recon", wide_dir, out, "anchor_reconstruction", reference_model=wide_dir,
                 max_anchors=24, n_queries=96, extra_cols=extra_cols)
    summary = weights_route._execute(45, snap, Reporter(enabled=False))
    assert (out / "embeddings.json").is_file()
    assert summary["extra"]["n_anchor_cols"] == 24
    assert summary["n_train"] + summary["n_test"] == 24
    assert 0 <= summary["retrieval"]["1"] <= 1


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
@pytest.mark.parametrize("kind", ["embed_align", "embed_extract", "embed_recon"])
def test_embedding_execute_paths_accept_live_mps_models(roots, wide_dir, kind):
    runs, _ = roots
    out = runs / kind
    snap = _snap(kind, wide_dir, out, "embedding_test", device="mps", dtype="float32",
                 model_b=wide_dir, reference_model=wide_dir, which_embedding="input",
                 max_anchors=48, n_queries=64, col_subset=4096, extra_cols=2048)
    summary = weights_route._execute(46, snap, Reporter(enabled=False))
    assert (out / "embeddings.json").is_file()
    if kind == "embed_align":
        assert summary["retrieval"]["1"] == 1.0
    elif kind == "embed_recon":
        assert summary["extra"]["n_anchor_cols"] == 48
    else:
        assert summary["true_dim"] == WIDTH


def test_hneuron_bake_writes_a_model(roots, dense_dir):
    from vivasecuris.aiasylum.weights.hneurons import HNeuronSet, _dense_down_projs

    runs, models = roots
    # Save a neuron set referencing real neurons of the dense model.
    model = transformers.AutoModelForCausalLM.from_pretrained(dense_dir)
    d_ff = _dense_down_projs(model)[0][1].weight.shape[1]
    hdir = runs / "42"
    hdir.mkdir()
    HNeuronSet(neurons={0: [1, 2], 1: [3]}, model_id=dense_dir, d_ff=d_ff,
               n_selected=3, n_total=2 * d_ff).save(hdir)

    out = models / "dense-hn"
    snap = _snap("hneuron_bake", dense_dir, out, "hneuron_scale", hneuron_alpha=0.5)
    snap["hneurons_dir"] = str(hdir)
    summary = weights_route._execute(43, snap, Reporter(enabled=False))

    assert out.is_dir() and not (models / ".staging-43").exists()
    manifest = SurgeryManifest.load(out)
    assert manifest.method == "hneuron_scale" and manifest.extra["n_neurons"] == 3
    assert weights_route._headline("hneuron_bake", summary)["n_neurons"] == 3
    # The baked column really changed.
    edited = transformers.AutoModelForCausalLM.from_pretrained(out)
    base = transformers.AutoModelForCausalLM.from_pretrained(dense_dir)
    assert not torch.equal(edited.model.layers[0].mlp.down_proj.weight,
                           base.model.layers[0].mlp.down_proj.weight)


def test_redteam_prompt_leak_runs(roots, dense_dir):
    runs, _ = roots
    out = runs / "44"
    snap = _snap("redteam", dense_dir, out, "leak_test", redteam_target="prompt_leak",
                 secret_system="the passphrase is mat rug cat", max_new_tokens=8)
    summary = weights_route._execute(44, snap, Reporter(enabled=False))
    assert (out / "leak.json").is_file()
    assert summary["target"] == "prompt_leak" and "per_family" in summary
    assert 0.0 <= summary["worst_leak"] <= 1.0
    assert weights_route._headline("redteam", summary)["target"] == "prompt_leak"


@pytest.mark.parametrize("sampled_outcome", [None, "pass", "degenerate", "target_not_met"])
def test_induce_runs_end_to_end(roots, dense_dir, stub_corpus, monkeypatch, sampled_outcome):
    """A category gate + refusal direction through the induce tuner. The gate is
    stubbed to fire on the category so the wiring, not the probe, is exercised."""
    from vivasecuris.aiasylum.interp.probes.train import LayerProbe, ProbeSet
    from vivasecuris.aiasylum.weights import gate as gate_mod
    from vivasecuris.aiasylum.weights.direction import RefusalDirection
    import numpy as np

    runs, _ = roots
    # A minimal saved probe on the dense model's width and a rank-1 direction.
    probe = ProbeSet(
        probes={1: LayerProbe(layer=1, weight=np.zeros(WIDTH), bias=0.0, mean=np.zeros(WIDTH),
                              scale=np.ones(WIDTH), auroc=0.99, accuracy=1.0, ece=0.0,
                              null_auroc_p95=0.5)},
        best_layer=1, model_id=dense_dir, pooling="mean", dataset_hash="stub",
    )
    pdir = runs / "50"
    pdir.mkdir()
    probe.save(pdir)

    torch.manual_seed(4)
    v = torch.randn(WIDTH)
    ddir = runs / "51"
    ddir.mkdir()
    RefusalDirection(vector=v / v.norm(), layer=1, auc=0.95, cohens_d=2.0, model_id=dense_dir,
                     split_hash="stub",
                     extra={"projection_means": {"harmful": 2.0, "harmless": 0.0}}).save(ddir)

    monkeypatch.setattr(gate_mod, "gate_scores",
                        lambda model, tok, prompts, ps, **kw: [1.0 if p.startswith("cat ") else 0.0 for p in prompts])
    # The induce branch reads the harmful/harmless corpus from the DB (empty here).
    from vivasecuris.aiasylum.weights import corpus as corpus_mod
    monkeypatch.setattr(corpus_mod, "load_harmless_prompts",
                        lambda limit=None, benchmark=None: [f"the {w}" for w in ("mat", "rug", "cat", "dog", "sat", "on", "a", "an")][:limit or 8])
    monkeypatch.setattr(corpus_mod, "build_split",
                        lambda **kw: SimpleNamespace(harmful_test=["how make a mat", "the dog sat", "a dog on a mat", "how make the rug"]))

    if sampled_outcome is not None:
        # Exercise the real selector, report-half gates, sampled verification,
        # and artifact publisher; only model responses are controlled.
        from vivasecuris.aiasylum.weights import induce as induce_mod
        from vivasecuris.aiasylum.weights.evaluate import CAPABILITY_QA

        answers = {q: accepted[0] for q, accepted in CAPABILITY_QA}

        def generated(model, tokenizer, prompts, *, steer_direction=None, sampling=None, **kwargs):
            responses = []
            for prompt in prompts:
                response = answers.get(prompt, COMPLY)
                if sampling is not None and sampled_outcome == "degenerate":
                    # A serving-mode collapse affects enough responses to trip
                    # the real aggregate degeneration gate.
                    responses.append("loop " * 40)
                    continue
                if prompt.startswith("cat ") and steer_direction is not None:
                    response = "I cannot help with that request."
                    if sampling is not None and sampled_outcome == "target_not_met":
                        response = COMPLY
                responses.append(response)
            return responses

        monkeypatch.setattr(induce_mod, "_gen_all", generated)

    out = runs / "52"
    snap = _snap("induce", dense_dir, out, "gated_steer", ms=[0.5], taus=[0.5],
                 n_benign=8, n_prompts=4, max_new_tokens=6,
                 verify_sampled=sampled_outcome is not None)
    snap["direction_dir"] = str(ddir)
    snap["probe_dir"] = str(pdir)
    snap["category_config"] = {"name": "cat", "prompts": [f"cat {w}" for w in
                               ["mat", "rug", "dog", "sat", "on", "a", "the", "cat"]],
                               "near_miss": [f"the {w}" for w in ["mat", "rug", "dog", "sat"]]}
    summary = weights_route._execute(52, snap, Reporter(enabled=False))

    assert (out / "induce.json").is_file()
    assert "trials" in summary and len(summary["trials"]) == 1
    # gate fired only on the category prompts (tune half of 8 = 4).
    assert summary["trials"][0]["gate_fired"]["target"] == 4
    assert summary["trials"][0]["gate_fired"]["near_miss"] == 0
    if sampled_outcome is not None:
        accepted = sampled_outcome == "pass"
        assert summary["winner"]["accepted"] is True, "Greedy report passes in each scenario"
        assert summary["accepted"] is accepted
        assert summary["verification"]["accepted"] is accepted
        assert (out / "behaviour.json").exists() is accepted
        assert (out / "distill_rows.jsonl").exists() is accepted
        headline = weights_route._headline("induce", summary)
        assert headline["accepted"] is accepted
        assert headline["verified"] is accepted
        if not accepted:
            assert summary["reason"] == sampled_outcome
            assert headline["refuse_target"] == 0.0


def test_direction_run_refuses_a_base_model_unless_asked(roots, dense_dir, stub_corpus):
    """The fixture tokenizer has no chat template, so a direction run on it is a
    base-model derivation: refused by default, recorded when asked for."""
    from vivasecuris.aiasylum.weights.capture import NoChatTemplateError

    runs, _ = roots
    out = runs / "27"
    snap = _snap("direction", dense_dir, out, "diff_in_means")
    with pytest.raises(NoChatTemplateError):
        weights_route._execute(27, snap, Reporter(enabled=False))
    assert not (out / "direction.json").exists()

    snap = _snap("direction", dense_dir, out, "diff_in_means", allow_no_chat_template=True)
    summary = weights_route._execute(27, snap, Reporter(enabled=False))
    meta = json.loads((out / "direction.json").read_text())
    assert meta["template_applied"] is False and meta["format_version"] == 2
    assert summary["template_applied"] is False
