"""Service for single-prompt analysis (no comparison)."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np

from vivasecuris.aiasylum.interp.core.loader import ModelLoader
from vivasecuris.aiasylum.interp.core.runner import ModelRunner
from vivasecuris.aiasylum.interp.core.config import Config
from vivasecuris.aiasylum.interp.analysis.activation_norms import compute_activation_norms
from vivasecuris.aiasylum.interp.analysis.pca import PCAAnalyzer
from vivasecuris.aiasylum.interp.analysis.dim_reduction import DimensionReduction
from vivasecuris.aiasylum.interp.analysis.predictions import PredictionAnalyzer
from vivasecuris.aiasylum.interp.data.models import RunResult, SinglePromptResult
from vivasecuris.aiasylum.interp.alignment.query_region import find_query_region, align_to_query_region
from vivasecuris.aiasylum.interp.patching import _get_lm_head_weight
from vivasecuris.aiasylum.interp.analysis.attribution import LogitAttributionAnalyzer

logger = logging.getLogger(__name__)


class SinglePromptService:
    """Orchestrates single-prompt analysis (one forward pass, no comparison)."""

    @staticmethod
    def run_single_analysis(
        model,
        tokenizer,
        config: Config,
    ) -> SinglePromptResult:
        """
        Run single-prompt analysis.

        Args:
            model: Loaded model
            tokenizer: Loaded tokenizer
            config: Configuration with analysis_mode="single" and config.prompt set

        Returns:
            SinglePromptResult with activation norms, PCA, predictions, optional attention/MLP/logit attribution.
        """
        logger.info("Starting single-prompt analysis")
        prompt = config.prompt or ""
        if not prompt.strip():
            raise ValueError("Single-prompt analysis requires config.prompt to be set")

        warnings = []
        ModelLoader.set_deterministic(config.seed)
        debug_mode = (
            config.debug_mode
            or (
                config.enable_component_analysis
                and (
                    config.enable_attention_capture
                    or config.enable_mlp_capture
                    or config.enable_attn_output_capture
                    or config.enable_pre_mlp_capture
                    or config.enable_qkv_capture
                )
            )
        )
        runner = ModelRunner(model, tokenizer, debug_mode=debug_mode, config=config)

        logger.info("Running forward pass")
        run_result = runner.run_once(prompt, max_length=config.max_len)
        num_layers = len(run_result.hidden_states)
        seq_len = len(run_result.token_strs)

        # Window: marker-based or full
        if config.query_marker and config.query_marker.strip():
            query_start, query_len = find_query_region(
                tokenizer, prompt, config.query_marker
            )
            start, window_len = align_to_query_region(
                tokenizer, run_result, query_start, query_len, config.window
            )
        else:
            start = 0
            window_len = min(seq_len, config.window) if config.window else seq_len

        tokens = run_result.token_strs[start : start + window_len]

        # Activation norms
        logger.info("Computing activation norms")
        activation_norm_mat = compute_activation_norms(run_result, start, window_len)

        # Layer selection for PCA (no spike layer; use middle as focus)
        focus_layer = num_layers // 2
        pca_layers = PCAAnalyzer.select_pca_layers(
            config.pca_layers, num_layers, focus_layer
        )

        # Dimensionality reduction
        logger.info(f"Computing {config.dim_reduction.upper()} (layers: {pca_layers})")
        pca_payload = {}
        for layer_idx in pca_layers:
            try:
                dr_data = DimensionReduction.compute_single_for_layer(
                    run_result,
                    layer_idx,
                    start,
                    window_len,
                    config.dim_reduction,
                    config,
                )
                pca_payload[str(layer_idx)] = dr_data
            except Exception as e:
                raise RuntimeError(f"{config.dim_reduction.upper()} projection failed at hidden-state index {layer_idx}: {e}") from e

        # Predictions (use last layer)
        predictions_payload = None
        try:
            pred_analyzer = PredictionAnalyzer(model, tokenizer, topk=config.topk)
            predictions_payload = pred_analyzer.compute_predictions_analysis_single(
                run_result, num_layers - 1, start, window_len
            )
        except Exception as e:
            raise RuntimeError(f"Predictions analysis failed: {e}") from e

        component_layers = sorted({min(i, num_layers - 2) for i in pca_layers})

        # Optional: attention / MLP summarization (simple serialization for dashboard)
        attention_payload = None
        mlp_payload = None
        if config.enable_component_analysis and run_result.attention_weights:
            attention_payload = {}
            for layer_idx, attn in enumerate(run_result.attention_weights):
                if attn is not None and layer_idx in component_layers:
                    # [1, n_heads, seq, seq] -> window slice then mean over heads -> [window_len, window_len]
                    a = attn[0].float().cpu().numpy()
                    a_window = np.mean(
                        a[:, start : start + window_len, start : start + window_len],
                        axis=0,
                    )
                    attention_payload[str(layer_idx)] = {
                        "mean_over_heads": a_window.tolist(),
                    }
        if config.enable_component_analysis and run_result.mlp_activations:
            mlp_payload = {}
            for layer_idx, mlp_t in run_result.mlp_activations.items():
                if layer_idx in component_layers and mlp_t is not None:
                    m = mlp_t[0, start : start + window_len, :].float().cpu().numpy()
                    norms = np.linalg.norm(m, axis=1).tolist()
                    mlp_payload[str(layer_idx)] = {"token_norms": norms}

        # Logit attribution at last token of window
        logit_attribution_payload = None
        if (
            config.enable_component_analysis
            and run_result.attn_outputs is not None
            and window_len > 0
        ):
            try:
                lm_head_weight = _get_lm_head_weight(model)
                if lm_head_weight is not None:
                    last_pos = start + window_len - 1
                    if last_pos < run_result.hidden_states[0].shape[1]:
                        payload = LogitAttributionAnalyzer.compute(
                            run_result,
                            position=last_pos,
                            lm_head_weight=lm_head_weight,
                            tokenizer=tokenizer,
                            topk=config.topk,
                        )
                        if payload.get("attribution_available"):
                            logit_attribution_payload = payload
            except Exception as e:
                logger.warning(f"Failed logit attribution: {e}")
                warnings.append(f"Supplementary analysis unavailable: Failed logit attribution: {e}")

        resolved_model_id = ModelLoader.resolve_model_id(config.model)
        if resolved_model_id != config.model:
            logger.info(f"Model resolved: '{config.model}' → '{resolved_model_id}'")

        meta = {
            "model": resolved_model_id,
            "original_model": config.model,
            "prompt_preview": prompt[:200] + ("..." if len(prompt) > 200 else ""),
            "analysis_mode": "single",
            "device": str(model.device),
            "dtype": str(model.dtype).removeprefix("torch."),
            "requested_device": config.device,
            "window": config.window,
            "start": start,
            "window_len": window_len,
            "num_layers": num_layers,
            "dim_reduction": config.dim_reduction,
            "warnings": warnings,
            "input_token_counts": [seq_len],
            "component_layers": component_layers,
        }

        return SinglePromptResult(
            meta=meta,
            run_result=run_result,
            tokens=tokens,
            start=start,
            window_len=window_len,
            activation_norm_mat=activation_norm_mat,
            pca_payload=pca_payload or None,
            predictions_payload=predictions_payload,
            attention_payload=attention_payload,
            mlp_payload=mlp_payload,
            logit_attribution_payload=logit_attribution_payload,
            cot_analysis=None,
        )

    @staticmethod
    def save_results(result: SinglePromptResult, out_dir: Path) -> None:
        """Save single-prompt results to files."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        meta = {**result.meta}
        meta_file = out_dir / "meta.json"
        with open(meta_file, "w") as f:
            json.dump(meta, f, indent=2)

        if result.activation_norm_mat is not None:
            np.save(out_dir / "activation_norms.npy", result.activation_norm_mat)

        if result.pca_payload:
            pca_file = out_dir / "pca_payload.json"
            # Convert any non-serializable values
            pca_serializable = {}
            for k, v in result.pca_payload.items():
                if isinstance(v, dict):
                    pca_serializable[k] = v
                else:
                    pca_serializable[k] = v
            with open(pca_file, "w") as f:
                json.dump(result.pca_payload, f, indent=2)

        if result.predictions_payload:
            pred_file = out_dir / "predictions.json"
            with open(pred_file, "w") as f:
                json.dump(result.predictions_payload, f, indent=2)

        if result.attention_payload:
            attn_file = out_dir / "attention_payload.json"
            with open(attn_file, "w") as f:
                json.dump(result.attention_payload, f, indent=2)

        if result.mlp_payload:
            mlp_file = out_dir / "mlp_payload.json"
            with open(mlp_file, "w") as f:
                json.dump(result.mlp_payload, f, indent=2)

        if result.logit_attribution_payload:
            la_file = out_dir / "logit_attribution_payload.json"
            with open(la_file, "w") as f:
                json.dump(result.logit_attribution_payload, f, indent=2)

        logger.info(f"Single-prompt results saved to {out_dir}")
