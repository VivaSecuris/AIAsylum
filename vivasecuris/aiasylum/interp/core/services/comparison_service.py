"""Main comparison service that orchestrates the comparison workflow.

Analyses are called directly rather than through a registry; labotomy's
core.analysis_registry was unused plumbing (empty at import, consulted by
nothing) and was dropped during vendoring
and merge payloads instead of hard-coding each analysis step here.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any
import numpy as np

from vivasecuris.aiasylum.interp.core.loader import ModelLoader
from vivasecuris.aiasylum.interp.core.runner import ModelRunner
from vivasecuris.aiasylum.interp.core.config import Config
from vivasecuris.aiasylum.interp.alignment.strategies import get_alignment_strategy
from vivasecuris.aiasylum.interp.analysis.similarity import (
    compute_cosine_similarity,
    compute_delta_norm,
    find_spike_layer,
)
from vivasecuris.aiasylum.interp.analysis.pca import PCAAnalyzer
from vivasecuris.aiasylum.interp.analysis.umap import UMAPAnalyzer
from vivasecuris.aiasylum.interp.analysis.tsne import TSNEAnalyzer
from vivasecuris.aiasylum.interp.analysis.dim_reduction import DimensionReduction
from vivasecuris.aiasylum.interp.analysis.predictions import PredictionAnalyzer
from vivasecuris.aiasylum.interp.analysis.attention_analysis import AttentionAnalyzer
from vivasecuris.aiasylum.interp.analysis.mlp_analysis import MLPAnalyzer
from vivasecuris.aiasylum.interp.analysis.circuit_analysis import CircuitAnalyzer
from vivasecuris.aiasylum.interp.analysis.temporal_analysis import TemporalAnalyzer
from vivasecuris.aiasylum.interp.analysis.attribution import AttributionAnalyzer, LogitAttributionAnalyzer
from vivasecuris.aiasylum.interp.analysis.ov_qk import run_ov_qk_analysis
from vivasecuris.aiasylum.interp.analysis.causal_scrub import (
    run_scrub_experiment,
    find_minimal_circuit_greedy,
)
from vivasecuris.aiasylum.interp.data.models import ComparisonResult, RunResult
from vivasecuris.aiasylum.interp.patching import run_patching_experiments
from vivasecuris.aiasylum.interp.patching import _get_lm_head_weight

logger = logging.getLogger(__name__)


class ComparisonService:
    """Orchestrates the comparison workflow."""

    @staticmethod
    def run_comparison(
        model,
        tokenizer,
        config: Config,
    ) -> ComparisonResult:
        """
        Run a full comparison between two prompts.
        
        Args:
            model: Loaded model
            tokenizer: Loaded tokenizer
            config: Configuration object
            
        Returns:
            ComparisonResult with all analysis data
        """
        if config.enable_scrub:
            raise ValueError("Causal scrubbing is unavailable: cached component reconstruction does not rerun downstream layers. Use activation patching for measured interventions.")
        logger.info("Starting comparison")
        
        # Set deterministic seed
        warnings = []
        ModelLoader.set_deterministic(config.seed)
        
        # Initialize runner with debug mode if component analysis is enabled
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
        
        # Run forward passes
        logger.info("Running forward pass for prompt A")
        result_a = runner.run_once(config.prompt_a, max_length=config.max_len)
        
        logger.info("Running forward pass for prompt B")
        result_b = runner.run_once(config.prompt_b, max_length=config.max_len)
        
        # Align sequences
        logger.info(f"Aligning sequences (strategy: {config.align})")
        alignment_strategy = get_alignment_strategy(config.align, config.marker)
        start_a, start_b, window_len = alignment_strategy.align(
            tokenizer, result_a, result_b, config.window
        )
        
        logger.info(f"Aligned window: start_a={start_a}, start_b={start_b}, length={window_len}")
        
        # Compute similarity matrices
        logger.info("Computing similarity matrices")
        cos_mat = compute_cosine_similarity(result_a, result_b, start_a, start_b, window_len)
        dn_mat = compute_delta_norm(result_a, result_b, start_a, start_b, window_len)
        
        # Find spike layer
        spike_layer = find_spike_layer(dn_mat)
        logger.info(f"Spike layer: {spike_layer}")
        
        # Get aligned tokens
        tokens_a = result_a.token_strs[start_a : start_a + window_len]
        tokens_b = result_b.token_strs[start_b : start_b + window_len]
        
        # Compute dimensionality reduction for selected layers
        logger.info(f"Computing {config.dim_reduction.upper()} (layers: {config.pca_layers})")
        num_layers = len(result_a.hidden_states)
        pca_layers = PCAAnalyzer.select_pca_layers(config.pca_layers, num_layers, spike_layer)
        pca_payload = {}
        
        for layer_idx in pca_layers:
            try:
                dr_data = DimensionReduction.compute_for_layer(
                    result_a,
                    result_b,
                    layer_idx,
                    start_a,
                    start_b,
                    window_len,
                    config.dim_reduction,
                    config,
                )
            except Exception as e:
                raise RuntimeError(f"{config.dim_reduction.upper()} projection failed at hidden-state index {layer_idx}: {e}") from e
            pca_payload[str(layer_idx)] = dr_data
        
        component_layers = sorted({min(i, num_layers - 2) for i in pca_layers})

        # Phase 2: Component localization analysis
        attention_payload = {}
        mlp_payload = {}
        circuit_payload = {}
        temporal_payload = {}
        attribution_payload = {}
        
        if config.enable_component_analysis:
            logger.info("Computing Phase 2 component localization analysis")
            
            # Compute attention and MLP analysis for selected layers
            for layer_idx in component_layers:
                try:
                    # Attention analysis (only if attention weights are available)
                    if result_a.attention_weights is not None and config.enable_attention_capture:
                        head_contributions = AttentionAnalyzer.compute_head_contribution_scores(
                            result_a, result_b, layer_idx, start_a, start_b, window_len
                        )
                        attention_payload[str(layer_idx)] = head_contributions
                        
                        # Attribution analysis
                        head_roles = AttributionAnalyzer.classify_head_roles_by_contribution(
                            result_a, result_b, layer_idx, start_a, start_b, window_len
                        )
                        attribution_payload[str(layer_idx)] = head_roles
                    
                    # MLP analysis (only if MLP activations are available)
                    if config.enable_mlp_capture or config.enable_pre_mlp_capture:
                        neuron_contributions = MLPAnalyzer.compute_neuron_contribution_scores(
                            result_a, result_b, layer_idx, start_a, start_b, window_len
                        )
                        mlp_payload[str(layer_idx)] = neuron_contributions
                        
                except Exception as e:
                    raise RuntimeError(f"Component analysis failed for block {layer_idx}: {e}") from e
        
            # Temporal localization (once for all layers)
            try:
                first_div_token = TemporalAnalyzer.detect_first_divergence_token(dn_mat)
                change_points = TemporalAnalyzer.detect_change_points(dn_mat)
                critical_points = TemporalAnalyzer.identify_critical_decision_points(dn_mat, cos_mat)
                divergence_timeline = TemporalAnalyzer.compute_divergence_timeline(dn_mat)
                
                temporal_payload = {
                    "first_divergence_token": int(first_div_token),
                    "change_points": change_points,
                    "critical_decision_points": critical_points,
                    "divergence_timeline": divergence_timeline,
                }
            except Exception as e:
                logger.warning(f"Failed to compute temporal analysis: {e}")
                warnings.append(f"Supplementary analysis unavailable: Failed to compute temporal analysis: {e}")
            
            # Circuit analysis for spike layer and first divergence token
            try:
                first_div_token = temporal_payload.get("first_divergence_token", 0)
                # Build circuit card for spike layer at last token
                circuit_card_last = CircuitAnalyzer.build_circuit_card(
                    result_a, result_b, min(spike_layer, num_layers - 2), window_len - 1,
                    start_a, start_b, window_len
                )
                # Build circuit card for spike layer at first divergence
                if first_div_token > 0:
                    circuit_card_first = CircuitAnalyzer.build_circuit_card(
                        result_a, result_b, min(spike_layer, num_layers - 2), first_div_token,
                        start_a, start_b, window_len
                    )
                else:
                    circuit_card_first = None
                
                # Safety neuron identification (only if MLP available)
                safety_neurons = None
                if config.enable_mlp_capture or config.enable_pre_mlp_capture:
                    safety_neurons = CircuitAnalyzer.identify_safety_neuron_clusters(
                        result_a, result_b, min(spike_layer, num_layers - 2), start_a, start_b, window_len
                    )
                
                circuit_payload = {
                    "spike_layer_last_token": circuit_card_last,
                    "spike_layer_first_divergence": circuit_card_first,
                    "safety_neuron_clusters": safety_neurons,
                }
            except Exception as e:
                logger.warning(f"Failed to compute circuit analysis: {e}")
                warnings.append(f"Supplementary analysis unavailable: Failed to compute circuit analysis: {e}")
        
        # OV/QK analysis when QKV capture is enabled
        ov_qk_payload = None
        if (
            config.enable_qkv_capture
            and result_a.qkv_outputs is not None
            and getattr(result_a, "attention_weights", None) is not None
        ):
            try:
                logger.info("Running OV/QK analysis for selected layers")
                ov_qk_payload = {}
                for layer_idx in component_layers:
                    try:
                        ov_qk_payload[str(layer_idx)] = run_ov_qk_analysis(
                            result_a, model, layer_idx, start_a, window_len
                        )
                    except Exception as e:
                        raise RuntimeError(f"OV/QK analysis failed for block {layer_idx}: {e}") from e
            except Exception as e:
                raise RuntimeError(f"OV/QK analysis failed: {e}") from e
        
        # Causal scrubbing and minimal circuit (when enabled)
        scrub_payload = None
        minimal_circuit_payload = None
        need_unembedding = (
            config.enable_scrub or config.enable_minimal_circuit
            or (result_a.attn_outputs is not None and config.enable_component_analysis)
        )
        lm_head_weight = _get_lm_head_weight(model) if need_unembedding else None
        if (config.enable_scrub or config.enable_minimal_circuit) and lm_head_weight is not None:
            last_pos = window_len - 1 if window_len > 0 else 0
            pos_a = start_a + last_pos
            if config.enable_scrub:
                try:
                    # Example: scrub all but a small set of components (e.g. spike layer heads)
                    scrub_payload = run_scrub_experiment(
                        result_a,
                        pos_a,
                        start_a,
                        window_len,
                        components_to_scrub=[],  # Can be extended via config
                        baseline="zero",
                        model=model,
                        lm_head_weight=lm_head_weight,
                    )
                except Exception as e:
                    logger.warning(f"Failed to run scrub experiment: {e}")
                    scrub_payload = {"scrub_available": False, "claim": "descriptive", "reason": str(e)}
            if config.enable_minimal_circuit:
                try:
                    minimal_circuit_payload = find_minimal_circuit_greedy(
                        result_a,
                        result_b,
                        start_a,
                        start_b,
                        window_len,
                        model=model,
                        lm_head_weight=lm_head_weight,
                        layer_range=(max(0, spike_layer - 2), min(num_layers - 1, spike_layer + 3)),
                        top_k_heads=4,
                        top_k_neurons=4,
                        metric="logit_l2",
                        max_components=10,
                    )
                except Exception as e:
                    logger.warning(f"Failed to run minimal circuit: {e}")
                    minimal_circuit_payload = {"available": False, "claim": "descriptive", "reason": str(e)}
        
        # Compute predictions analysis (for spike layer)
        logger.info("Computing predictions analysis")
        predictions_payload = None
        try:
            pred_analyzer = PredictionAnalyzer(model, tokenizer, topk=config.topk)
            predictions_payload = pred_analyzer.compute_predictions_analysis(
                result_a, result_b, spike_layer, start_a, start_b, window_len
            )
        except Exception as e:
            raise RuntimeError(f"Predictions analysis failed: {e}") from e

        # Phase 3: Activation patching experiments (counterfactuals)
        patching_results = None
        if config.enable_patching:
            logger.info("Running Phase 3 activation patching experiments")
            try:
                patching_results = run_patching_experiments(
                    model=model,
                    tokenizer=tokenizer,
                    config=config,
                    result_a=result_a,
                    result_b=result_b,
                    cos_mat=cos_mat,
                    dn_mat=dn_mat,
                    spike_layer=spike_layer,
                    start_a=start_a,
                    start_b=start_b,
                    window_len=window_len,
                )
            except Exception as e:
                raise RuntimeError(f"Activation patching failed: {e}") from e

        # Logit attribution (component-wise decomposition at last token)
        logit_attribution_payload = None
        if result_a.attn_outputs is not None and config.enable_component_analysis:
            try:
                if lm_head_weight is not None and window_len > 0:
                    last_pos = start_a + window_len - 1
                    if last_pos < result_a.hidden_states[0].shape[1]:
                        payload = LogitAttributionAnalyzer.compute(
                            result_a,
                            position=last_pos,
                            lm_head_weight=lm_head_weight,
                            tokenizer=tokenizer,
                            topk=config.topk,
                        )
                        if payload.get("attribution_available"):
                            logit_attribution_payload = payload
            except Exception as e:
                logger.warning(f"Failed to compute logit attribution: {e}")
                warnings.append(f"Supplementary analysis unavailable: Failed to compute logit attribution: {e}")

        # Get resolved model ID (the one actually used)
        resolved_model_id = ModelLoader.resolve_model_id(config.model)
        
        # Log which model is actually being used
        if resolved_model_id != config.model:
            logger.info(f"Model resolved: '{config.model}' → '{resolved_model_id}'")
        else:
            logger.info(f"Using model: '{resolved_model_id}'")
        
        # Build metadata
        meta = {
            "model": resolved_model_id,  # Store the resolved model ID (the one actually used)
            "original_model": config.model,  # Store original for reference
            "prompt_a": config.prompt_a,
            "prompt_b": config.prompt_b,
            "device": str(model.device),
            "dtype": str(model.dtype).removeprefix("torch."),
            "requested_device": config.device,
            "align": config.align,
            "marker": config.marker if config.align == "marker" else None,
            "window": config.window,
            "start_a": start_a,
            "start_b": start_b,
            "window_len": window_len,
            "spike_layer": spike_layer,
            "num_layers": num_layers,
            "dim_reduction": config.dim_reduction,
            "warnings": warnings,
            "input_token_counts": [len(result_a.token_strs), len(result_b.token_strs)],
            "component_layers": component_layers,
        }
        
        # Create comparison result
        result = ComparisonResult(
            meta=meta,
            tokens_a=tokens_a,
            tokens_b=tokens_b,
            cos_mat=cos_mat,
            dn_mat=dn_mat,
            start_a=start_a,
            start_b=start_b,
            window_len=window_len,
            pca_payload=pca_payload,
            predictions_payload=predictions_payload,
            patching_results=patching_results,
            attention_payload=attention_payload if attention_payload else None,
            mlp_payload=mlp_payload if mlp_payload else None,
            circuit_payload=circuit_payload if circuit_payload else None,
            temporal_payload=temporal_payload if temporal_payload else None,
            attribution_payload=attribution_payload if attribution_payload else None,
            logit_attribution_payload=logit_attribution_payload,
            ov_qk_payload=ov_qk_payload,
            scrub_payload=scrub_payload,
            minimal_circuit_payload=minimal_circuit_payload,
        )
        
        logger.info("Comparison completed successfully")
        return result

    @staticmethod
    def save_results(result: ComparisonResult, out_dir: Path):
        """
        Save comparison results to files.
        
        Args:
            result: ComparisonResult to save
            out_dir: Output directory
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        meta_file = out_dir / "meta.json"
        with open(meta_file, "w") as f:
            json.dump(result.meta, f, indent=2)
        
        # Save similarity matrices
        cos_file = out_dir / "cos_mat.npy"
        dn_file = out_dir / "dn_mat.npy"
        np.save(cos_file, result.cos_mat)
        np.save(dn_file, result.dn_mat)
        
        # Save PCA payload
        if result.pca_payload:
            pca_file = out_dir / "pca_payload.json"
            with open(pca_file, "w") as f:
                json.dump(result.pca_payload, f, indent=2)
        
        # Save Phase 2 payloads
        if result.attention_payload:
            attn_file = out_dir / "attention_payload.json"
            with open(attn_file, "w") as f:
                json.dump(result.attention_payload, f, indent=2)
        
        if result.mlp_payload:
            mlp_file = out_dir / "mlp_payload.json"
            with open(mlp_file, "w") as f:
                json.dump(result.mlp_payload, f, indent=2)
        
        if result.circuit_payload:
            circuit_file = out_dir / "circuit_payload.json"
            with open(circuit_file, "w") as f:
                json.dump(result.circuit_payload, f, indent=2)
        
        if result.temporal_payload:
            temporal_file = out_dir / "temporal_payload.json"
            with open(temporal_file, "w") as f:
                json.dump(result.temporal_payload, f, indent=2)
        
        if result.attribution_payload:
            attribution_file = out_dir / "attribution_payload.json"
            with open(attribution_file, "w") as f:
                json.dump(result.attribution_payload, f, indent=2)

        if result.logit_attribution_payload:
            logit_attr_file = out_dir / "logit_attribution_payload.json"
            with open(logit_attr_file, "w") as f:
                json.dump(result.logit_attribution_payload, f, indent=2)

        if result.ov_qk_payload:
            ov_qk_file = out_dir / "ov_qk_payload.json"
            with open(ov_qk_file, "w") as f:
                json.dump(result.ov_qk_payload, f, indent=2, default=str)

        if result.scrub_payload:
            scrub_file = out_dir / "scrub_payload.json"
            with open(scrub_file, "w") as f:
                json.dump(result.scrub_payload, f, indent=2, default=str)

        if result.minimal_circuit_payload:
            min_circuit_file = out_dir / "minimal_circuit_payload.json"
            with open(min_circuit_file, "w") as f:
                json.dump(result.minimal_circuit_payload, f, indent=2, default=str)

        # Save predictions payload
        if result.predictions_payload:
            pred_file = out_dir / "predictions.json"
            with open(pred_file, "w") as f:
                json.dump(result.predictions_payload, f, indent=2)
        
        # Save patching results (Phase 3)
        if result.patching_results:
            patch_file = out_dir / "patching_results.json"
            with open(patch_file, "w") as f:
                json.dump(result.patching_results, f, indent=2)
        
        logger.info(f"Results saved to {out_dir}")
