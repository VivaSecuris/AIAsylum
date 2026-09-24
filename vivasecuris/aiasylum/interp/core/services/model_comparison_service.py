"""Compare two models on the same prompt.

labotomy compared two prompts through one model. This compares one prompt
through two sets of weights, which is what makes a weight edit legible: run a
prompt through the baseline and through the ablated model and see which layers
compute something different.

Two things differ from prompt-vs-prompt and are easy to get wrong:

**Memory.** Two 3B models is 12 GB of weights before a single activation is
captured. The loading discipline therefore lives in here rather than in
callers: load A, capture, release, load B, capture. The two models are never
resident at once.

**The unembedding.** Logit-lens numbers from two models are only in the same
space if both decode through the same ``lm_head``. Editing a model with tied
embeddings changes its unembedding too, so this records whether the two heads
still match and the payload says so rather than quietly implying they do.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from vivasecuris.aiasylum.interp.analysis.pca import PCAAnalyzer
from vivasecuris.aiasylum.interp.analysis.dim_reduction import DimensionReduction
from vivasecuris.aiasylum.interp.analysis.predictions import PredictionAnalyzer
from vivasecuris.aiasylum.interp.analysis.similarity import (
    compute_cosine_similarity,
    compute_delta_norm,
    find_spike_layer,
)
from vivasecuris.aiasylum.interp.core.config import Config
from vivasecuris.aiasylum.interp.core.loader import ModelLoader
from vivasecuris.aiasylum.interp.core.runner import ModelRunner
from vivasecuris.aiasylum.interp.data.models import ComparisonResult, RunResult

logger = logging.getLogger(__name__)

LoaderFn = Callable[[], Tuple[Any, Any]]


class TokenizationMismatch(ValueError):
    """The two models tokenized the prompt differently.

    Every per-token and per-layer comparison assumes position i means the same
    thing on both sides. If the tokenizations differ, the comparison is
    meaningless rather than merely noisy, so it fails instead of producing a
    plausible-looking dashboard.
    """


class ModelComparisonService:
    """Same prompt, two models."""

    @staticmethod
    def _capture(model, tokenizer, prompt: str, config: Config) -> Tuple[RunResult, Dict[str, Any]]:
        """One forward pass, plus the facts needed to validate the pairing."""
        debug_mode = config.debug_mode or (
            config.enable_component_analysis
            and (
                config.enable_attention_capture
                or config.enable_mlp_capture
                or config.enable_attn_output_capture
            )
        )
        runner = ModelRunner(model, tokenizer, debug_mode=debug_mode, config=config)
        result = runner.run_once(prompt, max_length=config.max_len)

        head = getattr(model, "lm_head", None)
        fingerprint = None
        if head is not None and hasattr(head, "weight"):
            with_no_grad = head.weight.detach().to("cpu", copy=False)
            # Cheap, stable summary of the unembedding; full equality on a
            # vocab-sized matrix is not worth the memory.
            fingerprint = [
                float(with_no_grad.float().sum().item()),
                float(with_no_grad.float().abs().sum().item()),
                list(with_no_grad.shape),
            ]

        facts = {
            "n_layers": len(result.hidden_states),
            "d_model": int(result.hidden_states[0].shape[-1]),
            "tokens": list(result.token_strs),
            "lm_head_fingerprint": fingerprint,
        }
        return result, facts

    @staticmethod
    def run_model_diff(
        config: Config,
        load_a: LoaderFn,
        load_b: LoaderFn,
        model_a_id: str,
        model_b_id: str,
        release: Optional[Callable[[], None]] = None,
        progress: Optional[Callable[[str], None]] = None,
    ) -> ComparisonResult:
        """Run ``config.prompt_a`` through both models and compare.

        ``load_a``/``load_b`` return ``(model, tokenizer)``. ``release`` is
        called between them to free the first model before the second loads.
        """
        prompt = config.prompt_a or config.prompt or ""
        if not prompt.strip():
            raise ValueError("model_diff requires a prompt (config.prompt_a)")

        def say(msg: str) -> None:
            logger.info(msg)
            if progress:
                progress(msg)

        ModelLoader.set_deterministic(config.seed)

        say(f"loading model A ({model_a_id})")
        model_a, tokenizer_a = load_a()
        try:
            say("capturing activations for model A")
            result_a, facts_a = ModelComparisonService._capture(model_a, tokenizer_a, prompt, config)
        finally:
            del model_a
            if release:
                say("releasing model A before loading B")
                release()

        say(f"loading model B ({model_b_id})")
        model_b, tokenizer_b = load_b()
        predictions_payload: Optional[Dict[str, Any]] = None
        try:
            say("capturing activations for model B")
            result_b, facts_b = ModelComparisonService._capture(model_b, tokenizer_b, prompt, config)

            # PredictionAnalyzer needs a live model for the unembedding, so the
            # logit lens runs here rather than after B is released.
            shared_unembedding = facts_a["lm_head_fingerprint"] == facts_b["lm_head_fingerprint"]
            window_len_early = min(
                len(facts_a["tokens"]), config.window or len(facts_a["tokens"])
            )
            if facts_a["tokens"] == facts_b["tokens"]:
                try:
                    say("comparing next-token predictions")
                    dn_probe = compute_delta_norm(result_a, result_b, 0, 0, window_len_early)
                    analyzer = PredictionAnalyzer(model_b, tokenizer_b, topk=config.topk)
                    predictions_payload = analyzer.compute_predictions_analysis(
                        result_a, result_b, find_spike_layer(dn_probe), 0, 0, window_len_early
                    )
                    if predictions_payload is not None:
                        predictions_payload["shared_unembedding"] = shared_unembedding
                        if not shared_unembedding:
                            predictions_payload["caveat"] = (
                                "The two models decode through different unembeddings, so "
                                "probabilities are not on a common scale. Read the overlap "
                                "of the top-k sets rather than the individual values."
                            )
                except Exception as exc:
                    logger.warning("Prediction comparison failed: %s", exc)
        finally:
            del model_b
            if release:
                say("releasing model B")
                release()

        # Position i must mean the same token on both sides.
        if facts_a["tokens"] != facts_b["tokens"]:
            raise TokenizationMismatch(
                f"'{model_a_id}' and '{model_b_id}' tokenized the prompt differently "
                f"({len(facts_a['tokens'])} vs {len(facts_b['tokens'])} tokens). "
                f"A per-position comparison would be meaningless. Compare models that "
                f"share a tokenizer, such as a model and its surgically edited copy."
            )
        if facts_a["n_layers"] != facts_b["n_layers"] or facts_a["d_model"] != facts_b["d_model"]:
            raise ValueError(
                f"Shape mismatch: '{model_a_id}' is {facts_a['n_layers']} layers x "
                f"{facts_a['d_model']} dims, '{model_b_id}' is {facts_b['n_layers']} x "
                f"{facts_b['d_model']}. Layer-by-layer comparison needs the same architecture."
            )

        if not shared_unembedding:
            logger.info(
                "The two models have different unembeddings; logit-lens values are "
                "not directly comparable."
            )

        # Same tokens on both sides, so the aligned window is the identity.
        window_len = min(len(facts_a["tokens"]), config.window or len(facts_a["tokens"]))
        start_a = start_b = 0
        tokens = facts_a["tokens"][:window_len]

        say("computing per-layer divergence")
        cos_mat = compute_cosine_similarity(result_a, result_b, start_a, start_b, window_len)
        dn_mat = compute_delta_norm(result_a, result_b, start_a, start_b, window_len)
        spike_layer = find_spike_layer(dn_mat)
        say(f"largest divergence at layer {spike_layer}")

        num_layers = len(result_a.hidden_states)
        say("computing trajectory embedding")
        pca_payload: Dict[str, Any] = {}
        for layer_idx in PCAAnalyzer.select_pca_layers(config.pca_layers, num_layers, spike_layer):
            for method in (config.dim_reduction, "pca"):
                try:
                    pca_payload[str(layer_idx)] = DimensionReduction.compute_for_layer(
                        result_a, result_b, layer_idx, start_a, start_b,
                        window_len, method, config,
                    )
                    break
                except Exception as exc:
                    logger.warning(
                        "%s failed for layer %d%s: %s",
                        method.upper(), layer_idx,
                        "; falling back to PCA" if method != "pca" else "", exc,
                    )

        meta = {
            "model": f"{model_a_id} vs {model_b_id}",
            "analysis_mode": "model_diff",
            "model_a": model_a_id,
            "model_b": model_b_id,
            "prompt_a": prompt,
            "prompt_b": prompt,
            "device": config.device,
            "dtype": config.dtype,
            "align": "identity",
            "marker": None,
            "window": config.window,
            "start_a": start_a,
            "start_b": start_b,
            "window_len": window_len,
            "spike_layer": spike_layer,
            "num_layers": num_layers,
            "dim_reduction": config.dim_reduction,
            "shared_unembedding": shared_unembedding,
        }

        return ComparisonResult(
            meta=meta,
            tokens_a=tokens,
            tokens_b=tokens,
            cos_mat=cos_mat,
            dn_mat=dn_mat,
            start_a=start_a,
            start_b=start_b,
            window_len=window_len,
            pca_payload=pca_payload or None,
            predictions_payload=predictions_payload,
        )

    @staticmethod
    def save_results(result: ComparisonResult, out_dir: Path) -> None:
        """Persist the artifacts, matching ComparisonService's layout."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        (out_dir / "meta.json").write_text(json.dumps(result.meta, indent=2))
        np.save(out_dir / "cos_mat.npy", result.cos_mat)
        np.save(out_dir / "dn_mat.npy", result.dn_mat)
        for name, payload in (
            ("pca_payload", result.pca_payload),
            ("predictions", result.predictions_payload),
        ):
            if payload is not None:
                (out_dir / f"{name}.json").write_text(json.dumps(payload, indent=2, default=str))
