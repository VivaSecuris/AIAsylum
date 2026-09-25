"""LoRA on a tiny random model: target resolution, two training steps, stop
handling, and merge + manifest. Random weights test mechanics, not meaning."""

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")

from tests._tiny_lm import build_model, build_tokenizer, sample_rows, small_spec
from vivasecuris.aiasylum.weights.lora import (
    LoraSpec, build_examples, merge_and_save, resolve_target_modules, train_lora,
)


@pytest.fixture(scope="module", autouse=True)
def small_cpu_workload():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def _never():
    return False


def test_spec_from_dict_ignores_unknown_keys_and_round_trips():
    spec = LoraSpec.from_dict({"rank": 4, "targets": "attention+mlp", "bogus": 1, "max_steps": 3})
    assert (spec.rank, spec.targets, spec.max_steps, spec.alpha) == (4, "attention+mlp", 3, 16)
    assert LoraSpec.from_dict(spec.to_dict()) == spec
    with pytest.raises(ValueError, match="rank"):
        LoraSpec(rank=0).validate()


def test_resolve_targets_presets_and_explicit_names():
    model = build_model()
    assert resolve_target_modules(model, LoraSpec(targets="attention")) == ["q_proj", "k_proj", "v_proj", "o_proj"]
    assert resolve_target_modules(model, LoraSpec(targets="attention+mlp")) == [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    ]
    assert resolve_target_modules(model, LoraSpec(targets="q_proj, v_proj")) == ["q_proj", "v_proj"]
    assert resolve_target_modules(model, LoraSpec(targets="model.layers.0.self_attn.q_proj")) == [
        "model.layers.0.self_attn.q_proj",
    ]
    with pytest.raises(ValueError, match=r"\['nonsense'\] do not exist.*q_proj"):
        resolve_target_modules(model, LoraSpec(targets="q_proj,nonsense"))
    with pytest.raises(ValueError, match="targets is empty"):
        resolve_target_modules(model, LoraSpec(targets=" , "))


def test_resolve_targets_never_touches_experts_unless_named():
    mixed = build_model(family="qwen2_moe", mlp_only_layers=[0])
    names = resolve_target_modules(mixed, LoraSpec(targets="attention+mlp"))
    assert names and all("experts" not in n for n in names)
    assert "model.layers.0.mlp.gate_proj" in names, "the dense block keeps its MLP targets"
    assert all(n.startswith("model.layers.") for n in names), "exact names when a short one would hit experts"
    assert not any(n.startswith("model.layers.1.mlp") for n in names)
    assert resolve_target_modules(mixed, LoraSpec(targets="attention")) == ["q_proj", "k_proj", "v_proj", "o_proj"]

    sparse = build_model(family="qwen2_moe")
    with pytest.raises(ValueError, match="no dense MLP blocks"):
        resolve_target_modules(sparse, LoraSpec(targets="attention+mlp"))
    explicit = resolve_target_modules(sparse, LoraSpec(targets="model.layers.0.mlp.experts.0.down_proj"))
    assert explicit == ["model.layers.0.mlp.experts.0.down_proj"]


def test_train_lora_two_steps_on_four_rows(tmp_path):
    model, tok = build_model(), build_tokenizer()
    events = []
    result = train_lora(model, tok, sample_rows(4), small_spec(), tmp_path / "run",
                        progress=events.append, should_stop=_never)

    assert result["steps"] == 2 and result["stopped"] is False
    assert result["final_loss"] is not None and torch.isfinite(torch.tensor(result["final_loss"]))
    assert result["trainable_params"] < 0.05 * result["total_params"]
    assert result["target_modules"] == ["q_proj", "k_proj", "v_proj", "o_proj"]
    assert result["dataset"]["kept"] == 4 and result["dataset"]["chat_template"] is False
    assert result["eval_loss_before"] is None, "4 rows hold nothing out (min(1, 4 // 5) == 0)"
    assert (tmp_path / "run" / "adapter" / "adapter_config.json").exists()
    assert result["adapter_path"].endswith("adapter")

    log_lines = [json.loads(l) for l in (tmp_path / "run" / "train_log.jsonl").read_text().splitlines()]
    steps = [l for l in log_lines if l["phase"] == "train"]
    assert [s["step"] for s in steps] == [1, 2] and steps[0]["total"] == 2
    assert {"loss", "lr", "tokens", "elapsed"} <= steps[0].keys()
    assert [e["step"] for e in events if e["phase"] == "train"] == [1, 2]
    assert [h["step"] for h in result["history"] if "loss" in h] == [1, 2]
    assert model.training is True, "training mode is left as train_lora set it, never eval"


def test_train_lora_holds_out_eval_rows_and_evaluates(tmp_path):
    model, tok = build_model(), build_tokenizer()
    events = []
    result = train_lora(model, tok, sample_rows(10), small_spec(), tmp_path / "run",
                        progress=events.append, should_stop=_never)
    assert result["dataset"]["train_rows"] == 9 and result["dataset"]["eval_rows"] == 1
    assert result["eval_loss_before"] is not None and result["eval_loss_after"] is not None
    assert all(torch.isfinite(torch.tensor(v)) for v in (result["eval_loss_before"], result["eval_loss_after"]))
    assert [e["step"] for e in events if e["phase"] == "eval"] == [0, 1, 2]
    assert [h for h in result["history"] if "eval_loss" in h][0]["step"] == 0


def test_train_lora_stops_on_request_and_saves_partial_adapter(tmp_path):
    model, tok = build_model(), build_tokenizer()
    seen = []

    def stop_after_first_step():
        return len(seen) >= 1

    result = train_lora(model, tok, sample_rows(4), small_spec(max_steps=4), tmp_path / "run",
                        progress=seen.append, should_stop=stop_after_first_step)
    assert result["stopped"] is True and result["steps"] == 1
    assert (tmp_path / "run" / "adapter-partial" / "adapter_config.json").exists()
    assert not (tmp_path / "run" / "adapter").exists()
    assert result["adapter_path"].endswith("adapter-partial")


def test_train_lora_requires_responses(tmp_path):
    from vivasecuris.aiasylum.weights.train_data import TrainRow

    with pytest.raises(ValueError, match="have no response"):
        train_lora(build_model(), build_tokenizer(), [TrainRow("w3")], small_spec(), tmp_path,
                   progress=lambda _: None, should_stop=_never)


def test_merge_and_save_writes_loadable_model_touching_only_targets(tmp_path):
    from vivasecuris.aiasylum.interp.core.loader import load
    from vivasecuris.aiasylum.weights.manifest import SurgeryManifest

    model, tok = build_model(), build_tokenizer()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    spec = small_spec()
    result = train_lora(model, tok, sample_rows(4), spec, tmp_path / "run",
                        progress=lambda _: None, should_stop=_never)

    out = merge_and_save(model, tok, tmp_path / "merged", source_model="tiny", spec=spec,
                         train_result=result, dataset_meta={"rows": 4, "digest": "abc"},
                         notes="test merge", extra={"job_id": 7})
    assert (out / "config.json").exists() and (out / "asylum_surgery.json").exists()

    manifest = SurgeryManifest.load(out)
    assert manifest.method == "lora_merge" and manifest.source_model == "tiny"
    assert manifest.architecture == "qwen2" and manifest.model_type == "qwen2"
    assert manifest.matrices_edited is None and manifest.coverage_verified is None
    assert manifest.extra["lora"]["rank"] == 2
    assert manifest.extra["train"]["steps"] == 2 and "history" not in manifest.extra["train"]
    assert manifest.extra["dataset"] == {"rows": 4, "digest": "abc"}
    assert manifest.extra["job_id"] == 7 and manifest.notes == "test merge"

    merged, merged_tok = load(str(out), device="cpu", dtype="float32")
    assert merged_tok("w3 w4")["input_ids"] == tok("w3 w4")["input_ids"]
    after = merged.state_dict()
    assert set(after) == set(before)
    changed = sorted(k for k in before if not torch.equal(before[k], after[k]))
    assert changed, "the merge must fold a non-zero delta into the targets"
    assert all(k.split(".")[-2] in {"q_proj", "k_proj", "v_proj", "o_proj"} for k in changed), changed


def test_merge_and_save_refuses_a_non_empty_directory(tmp_path):
    model, tok = build_model(), build_tokenizer()
    spec = small_spec()
    result = train_lora(model, tok, sample_rows(4), spec, tmp_path / "run",
                        progress=lambda _: None, should_stop=_never)
    target = tmp_path / "occupied"
    target.mkdir()
    (target / "config.json").write_text("{}")
    with pytest.raises(FileExistsError, match="not empty"):
        merge_and_save(model, tok, target, source_model="tiny", spec=spec,
                       train_result=result, dataset_meta={})


def test_build_examples_reports_chat_template_absent():
    _, stats = build_examples(build_tokenizer(), sample_rows(2), max_length=16)
    assert stats["chat_template"] is False and stats["kept"] == 2
