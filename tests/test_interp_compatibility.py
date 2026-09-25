"""Offline architecture/size regressions; random weights test mechanics, not meaning."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from vivasecuris.aiasylum.interp.analysis.ov_qk import compute_per_head_outputs, run_ov_qk_analysis
from vivasecuris.aiasylum.interp.analysis.predictions import PredictionAnalyzer
from vivasecuris.aiasylum.interp.core.config import Config
from vivasecuris.aiasylum.interp.core.runner import ModelRunner
from vivasecuris.aiasylum.interp.patching import patch_and_run, run_patching_experiments


@pytest.fixture(scope="module")
def tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    words = ["[UNK]", "[PAD]", "[EOS]", "the", "cat", "dog", "sat", "on", "a", "mat", "rug"]
    backend = Tokenizer(WordLevel({word: i for i, word in enumerate(words)}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]",
        model_input_names=["input_ids", "attention_mask"],
    )


@pytest.fixture(scope="module", autouse=True)
def small_cpu_workload():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def build_model(family="qwen2", width=32, layers=2):
    names = {
        "qwen2": "Qwen2Config", "qwen3": "Qwen3Config", "llama": "LlamaConfig",
        "gemma2": "Gemma2Config", "gpt_neox": "GPTNeoXConfig", "mixtral": "MixtralConfig",
        "qwen2_moe": "Qwen2MoeConfig",
    }
    cls = getattr(transformers, names[family], None)
    if cls is None:
        pytest.skip(f"{family} is unavailable in this transformers version")
    options = dict(
        vocab_size=64, hidden_size=width, intermediate_size=width * 2,
        num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2,
        head_dim=width // 4, max_position_embeddings=32,
        bos_token_id=2, eos_token_id=2, pad_token_id=1,
        attention_dropout=0.0, hidden_dropout=0.0,
        num_local_experts=2, num_experts=2, num_experts_per_tok=2,
        moe_intermediate_size=width, shared_expert_intermediate_size=width,
    )
    if family == "gemma2":
        # Make the transform large enough that omitting it fails the lens test.
        options.update(final_logit_softcapping=0.3, head_dim=width // 2,
                       query_pre_attn_scalar=width // 2)
    torch.manual_seed(17)
    return transformers.AutoModelForCausalLM.from_config(cls(**options)).eval()


def capture_config(tmp_path, **kwargs):
    base = dict(
        model="offline-test", out_dir=tmp_path, max_len=24, window=6,
        enable_pre_mlp_capture=True, enable_qkv_capture=True, topk=3,
    )
    base.update(kwargs)
    return Config(**base)


@pytest.mark.parametrize("family", ["qwen2", "qwen3", "llama", "gemma2", "gpt_neox", "mixtral", "qwen2_moe"])
@pytest.mark.parametrize("width,layers", [(32, 2), (64, 3)])
def test_capture_lens_and_real_patching_across_architectures(family, width, layers, tokenizer, tmp_path):
    model = build_model(family, width, layers)
    config = capture_config(tmp_path, enable_pre_mlp_capture=family not in {"mixtral", "qwen2_moe"})
    original_impl = model.config._attn_implementation
    runner = ModelRunner(model, tokenizer, debug_mode=True, config=config)
    source = runner.run_once("the cat sat on a mat")
    target = runner.run_once("the dog sat on a rug")

    assert model.config._attn_implementation == original_impl
    assert len(source.hidden_states) == layers + 1
    assert source.hidden_states[0].shape == (1, 6, width)
    assert len(source.attention_weights) == layers
    assert len(source.attn_outputs) == layers
    assert set(source.mlp_activations) == set(range(layers))
    assert set(source.qkv_outputs) == set(range(layers))
    assert all(t.device.type == "cpu" for t in (*source.hidden_states, *source.attention_weights))
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules())

    lens = PredictionAnalyzer(model, tokenizer, topk=128)
    assert lens.final_is_normed
    assert torch.allclose(lens.lens_logits(source.hidden_states[-1][0, -1], is_final=True),
                          source.logits[0, -1], atol=1e-5, rtol=1e-4)
    assert len(lens.get_predictions_at_position(source.hidden_states[-1][0, -1], True)[0]) == 64

    # The displayed tail still receives attention from keys earlier in the prompt.
    for layer in range(layers):
        per_head = compute_per_head_outputs(source, model, layer, start=4, window_len=2)
        assert per_head.shape == (4, 2, width)
        if family == "gpt_neox":
            bias = model.gpt_neox.layers[layer].attention.dense.bias.detach()
        else:
            bias = 0
        assert torch.allclose(per_head.sum(0) + bias, source.attn_outputs[layer][0, 4:6], atol=1e-5, rtol=1e-4)
        payload = run_ov_qk_analysis(source, model, layer, 4, 2)
        assert payload["qk_pattern_shape"] == [4, 2, 2]
        assert "not actual attention" in payload["qk_caveat"]
        assert "qk_vs_actual_max_diff" not in payload

    for layer in (0, layers):
        logits, final = patch_and_run(model, source, target, layer, [(i, i) for i in range(6)])
        assert torch.allclose(logits, source.logits[0, -1], atol=1e-5, rtol=1e-4)
        assert torch.allclose(final, source.hidden_states[-1][0, -1], atol=1e-5, rtol=1e-4)


def test_capture_flags_masks_context_and_cache(tokenizer, tmp_path, monkeypatch):
    model = build_model()
    config = capture_config(tmp_path, enable_attention_capture=False, enable_qkv_capture=False)
    calls = []
    original = model.forward

    def record(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "forward", record)
    result = ModelRunner(model, tokenizer, True, config).run_once("the " * 40, max_length=100)
    assert result.input_ids.shape[1] == 32  # configured context limit
    assert result.attention_weights is None
    assert result.mlp_activations
    assert calls[0]["output_attentions"] is False
    assert calls[0]["use_cache"] is False
    assert calls[0]["attention_mask"].shape == (1, 32)


def test_failed_forward_removes_hooks_and_restores_attention(tokenizer, tmp_path, monkeypatch):
    model = build_model()
    original_impl = model.config._attn_implementation
    runner = ModelRunner(model, tokenizer, True, capture_config(tmp_path))

    def fail(*args, **kwargs):
        raise RuntimeError("simulated CUDA out of memory")

    monkeypatch.setattr(model, "forward", fail)
    with pytest.raises(RuntimeError, match="out of memory"):
        runner.run_once("the cat")
    assert model.config._attn_implementation == original_impl
    assert runner._activation_hooks is None and runner._qkv_hooks is None
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules())


def test_sequence_length_equal_to_kv_heads(tokenizer, tmp_path):
    model = build_model()
    run = ModelRunner(model, tokenizer, True, capture_config(tmp_path)).run_once("the cat")
    outputs = compute_per_head_outputs(run, model, 0, 0, 2)
    assert torch.allclose(outputs.sum(0), run.attn_outputs[0][0], atol=1e-6)


def test_bfloat16_attention_analysis_and_resolved_metadata(tokenizer, tmp_path):
    from vivasecuris.aiasylum.interp.core.services.comparison_service import ComparisonService

    model = build_model().to(torch.bfloat16)
    config = capture_config(tmp_path, prompt_a="the cat sat", prompt_b="the dog sat",
                            device="auto", dtype="bf16")
    result = ComparisonService.run_comparison(model, tokenizer, config)
    assert result.attention_payload
    assert result.attribution_payload
    assert result.meta["device"] == "cpu"
    assert result.meta["dtype"] == "bfloat16"
    assert result.meta["requested_device"] == "auto"


@pytest.mark.parametrize("capture_neurons", [False, True])
def test_mlp_neurons_are_distinguished_from_residual_channels(capture_neurons, tokenizer, tmp_path):
    from vivasecuris.aiasylum.interp.analysis.circuit_analysis import CircuitAnalyzer
    from vivasecuris.aiasylum.interp.analysis.mlp_analysis import MLPAnalyzer

    model = build_model()
    config = capture_config(tmp_path, enable_pre_mlp_capture=capture_neurons)
    runner = ModelRunner(model, tokenizer, True, config)
    source, target = runner.run_once("the cat"), runner.run_once("the dog")
    scores = MLPAnalyzer.compute_neuron_contribution_scores(source, target, 0, 0, 0, 2)
    assert scores["activation_space"] == ("mlp_neurons" if capture_neurons else "residual_channels")
    assert scores["num_neurons"] == (64 if capture_neurons else 32)
    card = CircuitAnalyzer.build_circuit_card(source, target, 0, 1, 0, 0, 2)
    assert card["unit_label"] == scores["unit_label"]


def test_neuron_patching_avoids_full_reconstruction(tokenizer, tmp_path, monkeypatch):
    import vivasecuris.aiasylum.interp.patching as patching

    model = build_model()
    config = capture_config(tmp_path, enable_patching=True, patch_components="neuron", patch_neurons=[(1, 3)])
    runner = ModelRunner(model, tokenizer, True, config)
    source, target = runner.run_once("the cat"), runner.run_once("the dog")

    def forbidden(*args, **kwargs):
        raise AssertionError("allocated all-neuron by all-token by hidden-dimension tensor")

    monkeypatch.setattr(patching, "compute_per_neuron_mlp_outputs", forbidden)
    payload = run_patching_experiments(model, tokenizer, config, source, target, None, None, 1, 0, 0, 2)
    assert payload["enabled"]
    assert payload["forward_passes"] == 2
    assert payload["claim"] == "causal"


def test_api_config_honors_capture_budget(tmp_path):
    request = SimpleNamespace(model="offline", analysis_mode="single", max_len=80,
                              enable_attention_capture=False, enable_qkv_capture=True)
    config = Config.from_api_request(request, tmp_path)
    assert config.max_len == 80 and config.enable_attention_capture is True  # QKV analysis needs attention
    assert config.enable_qkv_capture is True


def test_cuda_request_fails_before_model_download(monkeypatch):
    from vivasecuris.aiasylum.interp.core.loader import load

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA was requested but is unavailable"):
        load("must-not-be-downloaded", device="cuda")


@pytest.mark.parametrize("width,layers", [(64, 2), (32, 3)])
def test_model_diff_rejects_incomparable_sizes(width, layers, tokenizer, tmp_path):
    from vivasecuris.aiasylum.interp.core.services.model_comparison_service import ModelComparisonService

    config = capture_config(tmp_path, prompt_a="the cat", enable_component_analysis=False)
    with pytest.raises(ValueError, match="Layer-by-layer comparison needs the same architecture"):
        ModelComparisonService.run_model_diff(
            config, lambda: (build_model(), tokenizer),
            lambda: (build_model(width=width, layers=layers), tokenizer), "small", "larger",
        )


def test_unembedding_fingerprint_detects_vocabulary_permutation(tokenizer, tmp_path):
    from vivasecuris.aiasylum.interp.core.services.model_comparison_service import ModelComparisonService

    model = build_model()
    config = capture_config(tmp_path, enable_component_analysis=False)
    _, before = ModelComparisonService._capture(model, tokenizer, "the cat", config)
    with torch.no_grad():
        model.get_output_embeddings().weight.copy_(model.get_output_embeddings().weight.flip(0))
    _, after = ModelComparisonService._capture(model, tokenizer, "the cat", config)
    assert before["lm_head_fingerprint"] != after["lm_head_fingerprint"]
