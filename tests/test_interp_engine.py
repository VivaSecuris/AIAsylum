"""The interpretability engine, driven the way a route will drive it.

Runs every mode on a tiny randomly-initialized Qwen2 so the whole thing
completes in seconds with no download. The numbers are meaningless on random
weights; what is verified is that each mode produces a result the dashboard
builder can render, and that model_diff enforces the invariants a
two-model comparison depends on.
"""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from vivasecuris.aiasylum.interp.core.config import Config
from vivasecuris.aiasylum.interp.core.orchestrator import AnalysisOrchestrator

D_MODEL, N_LAYERS = 32, 4
TOKENIZER_SRC = "Qwen/Qwen2.5-0.5B-Instruct"

# Small enough to stay far inside memory: attention capture scales as seq^2.
MAX_LEN = 64


def _build(tmp_path: Path, seed: int = 0) -> Path:
    from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Config

    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(TOKENIZER_SRC)
    cfg = Qwen2Config(
        vocab_size=len(tok), hidden_size=D_MODEL, intermediate_size=64,
        num_hidden_layers=N_LAYERS, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=128, tie_word_embeddings=False,
        bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
    )
    AutoModelForCausalLM.from_config(cfg).save_pretrained(str(tmp_path))
    tok.save_pretrained(str(tmp_path))
    return tmp_path


@pytest.fixture(scope="module")
def model_a(tmp_path_factory):
    return _build(tmp_path_factory.mktemp("interp-a"), seed=0)


@pytest.fixture(scope="module")
def model_b(tmp_path_factory):
    """A second model with the same architecture but different weights."""
    return _build(tmp_path_factory.mktemp("interp-b"), seed=99)


def _config(model: Path, out: Path, **kw) -> Config:
    """Build Config directly.

    Config.from_api_request() is deliberately avoided: it hardcodes
    max_len=2048 and forces every capture flag on, which is the path to
    multi-gigabyte activation tensors.
    """
    base = dict(
        model=str(model), out_dir=out, device="cpu", dtype="float32",
        max_len=MAX_LEN, window=16, topk=3, seed=0,
        enable_attention_capture=False, enable_mlp_capture=False,
        enable_component_analysis=False,
    )
    base.update(kw)
    return Config(**base)


def test_single_mode(model_a, tmp_path):
    cfg = _config(model_a, tmp_path / "single", analysis_mode="single", prompt="hello world")
    result, html = AnalysisOrchestrator.run_and_save(cfg)

    assert result.meta["model"] == str(model_a)
    assert len(result.tokens) > 0
    assert (tmp_path / "single" / "dashboard.html").exists()
    assert "<!DOCTYPE html>" in html


def test_comparison_mode(model_a, tmp_path):
    cfg = _config(
        model_a, tmp_path / "cmp", analysis_mode="comparison",
        prompt_a="how do I bake bread", prompt_b="how do I pick a lock",
    )
    result, html = AnalysisOrchestrator.run_and_save(cfg)

    assert result.cos_mat.shape == result.dn_mat.shape
    assert result.cos_mat.shape[0] == N_LAYERS + 1
    assert "spike_layer" in result.meta
    assert (tmp_path / "cmp" / "cos_mat.npy").exists()
    assert "<!DOCTYPE html>" in html


def test_progression_mode(model_a, tmp_path):
    cfg = _config(
        model_a, tmp_path / "prog", analysis_mode="progression",
        prompts=[
            "What is the capital of France?",
            "Q: What is 2+2? A: 4. What is the capital of France?",
            "Q: What is 2+2? A: 4. Q: Name a colour. A: blue. What is the capital of France?",
        ],
    )
    result, html = AnalysisOrchestrator.run_and_save(cfg)

    assert len(result.prompt_labels) == 3
    assert result.prompt_labels[0] == "zero-shot"
    assert "<!DOCTYPE html>" in html


def test_dashboards_have_no_external_urls(model_a, tmp_path):
    """Plotly must come from the local mount, not a CDN."""
    from vivasecuris.aiasylum.interp.visualization.assets import PLOTLY_URL_PATH

    cfg = _config(model_a, tmp_path / "offline", analysis_mode="single", prompt="hi")
    _, html = AnalysisOrchestrator.run_and_save(cfg)

    assert PLOTLY_URL_PATH in html
    assert "cdn.plot.ly" not in html
    assert 'src="http' not in html


# --- model_diff ------------------------------------------------------------

def _loader(path: Path):
    from vivasecuris.aiasylum.interp.core.loader import load

    return lambda: load(str(path), device="cpu", dtype="float32", seed=None)


def test_model_diff_compares_two_models(model_a, model_b, tmp_path):
    from vivasecuris.aiasylum.interp.core.services.model_comparison_service import (
        ModelComparisonService,
    )
    from vivasecuris.aiasylum.interp.visualization.dashboard_builder import DashboardBuilder

    cfg = _config(model_a, tmp_path / "diff", analysis_mode="model_diff", prompt_a="hello world")
    result = ModelComparisonService.run_model_diff(
        cfg, _loader(model_a), _loader(model_b), str(model_a), str(model_b),
    )

    assert result.meta["analysis_mode"] == "model_diff"
    assert result.meta["align"] == "identity"
    assert result.tokens_a == result.tokens_b, "same prompt must give the same tokens"
    assert result.cos_mat.shape[0] == N_LAYERS + 1
    # Different random weights must actually diverge.
    assert result.dn_mat.max() > 0, "two different models produced identical activations"

    html = DashboardBuilder().build_dashboard(result)
    assert "<!DOCTYPE html>" in html


def test_model_diff_detects_different_unembeddings(model_a, model_b, tmp_path):
    from vivasecuris.aiasylum.interp.core.services.model_comparison_service import (
        ModelComparisonService,
    )

    cfg = _config(model_a, tmp_path / "diff2", analysis_mode="model_diff", prompt_a="hello")
    result = ModelComparisonService.run_model_diff(
        cfg, _loader(model_a), _loader(model_b), "a", "b",
    )
    assert result.meta["shared_unembedding"] is False

    same = ModelComparisonService.run_model_diff(
        cfg, _loader(model_a), _loader(model_a), "a", "a",
    )
    assert same.meta["shared_unembedding"] is True
    # Identical weights must produce zero divergence.
    assert float(same.dn_mat.max()) == pytest.approx(0.0, abs=1e-5)


def test_model_diff_releases_between_loads(model_a, model_b, tmp_path):
    """The two models must never be resident at once."""
    from vivasecuris.aiasylum.interp.core.services.model_comparison_service import (
        ModelComparisonService,
    )

    calls = []
    cfg = _config(model_a, tmp_path / "diff3", analysis_mode="model_diff", prompt_a="hi")

    def la():
        calls.append("load_a")
        return _loader(model_a)()

    def lb():
        calls.append("load_b")
        return _loader(model_b)()

    ModelComparisonService.run_model_diff(
        cfg, la, lb, "a", "b", release=lambda: calls.append("release")
    )
    assert calls.index("release") < calls.index("load_b"), "B loaded before A was released"


def test_model_diff_rejects_mismatched_tokenization(model_a, tmp_path, monkeypatch):
    from vivasecuris.aiasylum.interp.core.services import model_comparison_service as mcs

    real = mcs.ModelComparisonService._capture
    state = {"n": 0}

    def fake(model, tokenizer, prompt, config):
        result, facts = real(model, tokenizer, prompt, config)
        state["n"] += 1
        if state["n"] == 2:  # make B disagree
            facts = dict(facts, tokens=facts["tokens"] + ["extra"])
        return result, facts

    monkeypatch.setattr(mcs.ModelComparisonService, "_capture", staticmethod(fake))
    cfg = _config(model_a, tmp_path / "diff4", analysis_mode="model_diff", prompt_a="hi")

    with pytest.raises(mcs.TokenizationMismatch, match="tokenized the prompt differently"):
        mcs.ModelComparisonService.run_model_diff(
            cfg, _loader(model_a), _loader(model_a), "a", "b"
        )


def test_model_diff_requires_a_prompt(model_a, tmp_path):
    from vivasecuris.aiasylum.interp.core.services.model_comparison_service import (
        ModelComparisonService,
    )

    cfg = _config(model_a, tmp_path / "diff5", analysis_mode="model_diff", prompt_a="")
    with pytest.raises(ValueError, match="requires a prompt"):
        ModelComparisonService.run_model_diff(
            cfg, _loader(model_a), _loader(model_a), "a", "b"
        )


def test_progression_survives_minimal_prompts(model_a, tmp_path):
    """Single-token prompts must degrade, not crash.

    Before the query-region floor and the component clamp, this path produced a
    zero-length window and died inside sklearn with a message about n_samples.
    """
    cfg = _config(
        model_a, tmp_path / "prog-min", analysis_mode="progression", prompts=["a", "b"]
    )
    result, html = AnalysisOrchestrator.run_and_save(cfg)

    assert result.query_window_len >= 1
    assert len(result.prompt_labels) == 2
    assert "<!DOCTYPE html>" in html


def test_progression_dashboard_keeps_js_template_literals(model_a, tmp_path):
    """JS placeholders must reach the browser, not be eaten by the f-string.

    The vendored builder wrote `${drName}` inside an f-string, so Python tried
    to evaluate drName and every progression dashboard raised NameError.
    """
    cfg = _config(
        model_a, tmp_path / "prog-js", analysis_mode="progression",
        prompts=["What is the capital of France?", "Q: 2+2? A: 4. What is the capital of France?"],
    )
    _, html = AnalysisOrchestrator.run_and_save(cfg)
    assert "${drName}" in html, "JS template literal was consumed by the f-string"
