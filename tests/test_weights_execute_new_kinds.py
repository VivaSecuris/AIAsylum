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
