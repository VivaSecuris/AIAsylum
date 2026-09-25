"""Service for analyzing multiple prompts in progression (one-shot, multi-shot analysis)."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torch

from vivasecuris.aiasylum.interp.core.loader import ModelLoader
from vivasecuris.aiasylum.interp.core.runner import ModelRunner
from vivasecuris.aiasylum.interp.core.config import Config
from vivasecuris.aiasylum.interp.alignment.strategies import get_alignment_strategy
from vivasecuris.aiasylum.interp.alignment.query_region import find_query_region, align_to_query_region
from vivasecuris.aiasylum.interp.analysis.similarity import (
    compute_cosine_similarity,
    compute_delta_norm,
)
from vivasecuris.aiasylum.interp.analysis.pca import PCAAnalyzer
from vivasecuris.aiasylum.interp.analysis.umap import UMAPAnalyzer
from vivasecuris.aiasylum.interp.analysis.tsne import TSNEAnalyzer
from vivasecuris.aiasylum.interp.analysis.predictions import PredictionAnalyzer
from vivasecuris.aiasylum.interp.data.models import RunResult
from vivasecuris.aiasylum.interp.data.multi_prompt_models import ProgressionResult

logger = logging.getLogger(__name__)


class MultiPromptService:
    """Orchestrates multi-prompt progression analysis (zero-shot → one-shot → multi-shot)."""

    @staticmethod
    def run_progression_analysis(
        model,
        tokenizer,
        config: Config,
    ) -> ProgressionResult:
        """
        Run progression analysis across multiple prompts.
        
        Args:
            model: Loaded model
            tokenizer: Loaded tokenizer
            config: Configuration with prompts list
            
        Returns:
            ProgressionResult with all analysis data
        """
        logger.info("Starting multi-prompt progression analysis")
        
        if not config.prompts or len(config.prompts) < 2:
            raise ValueError("Need at least 2 prompts for progression analysis")
        
        # Set deterministic seed
        ModelLoader.set_deterministic(config.seed)
        
        # Initialize runner (same debug_mode logic as comparison for capture)
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
        
        # Run forward passes for all prompts
        run_results = []
        prompt_labels = []
        
        for i, prompt in enumerate(config.prompts):
            if i == 0:
                label = "zero-shot"
            elif i == 1:
                label = "one-shot"
            else:
                label = f"{i}-shot"
            
            prompt_labels.append(label)
            logger.info(f"Running forward pass for {label}")
            result = runner.run_once(prompt, max_length=config.max_len)
            run_results.append(result)
        
        # Find query regions in all prompts
        logger.info("Finding query regions")
        query_regions = []
        for prompt in config.prompts:
            start, length = find_query_region(
                tokenizer, prompt, config.query_marker
            )
            query_regions.append({"start": start, "length": length})
        
        # Use the first prompt's query region as reference
        ref_query_start = query_regions[0]["start"]
        ref_query_len = query_regions[0]["length"]
        
        # Align all prompts to their query regions
        logger.info("Aligning prompts to query regions")
        aligned_tokens = []
        alignment_info = []
        query_starts = []
        
        for i, (result, query_region) in enumerate(zip(run_results, query_regions)):
            aligned_start, aligned_len = align_to_query_region(
                tokenizer, result, query_region["start"], query_region["length"], config.window
            )
            
            tokens = result.token_strs[aligned_start : aligned_start + aligned_len]
            aligned_tokens.append(tokens)
            alignment_info.append({
                "start": aligned_start,
                "window_len": aligned_len,
            })
            query_starts.append(query_region["start"] - aligned_start)  # Relative to aligned window
        
        # Use minimum window length for consistent comparison
        min_window_len = min(info["window_len"] for info in alignment_info)
        query_window_len = min_window_len
        if query_window_len < 1:
            raise ValueError(
                "Progression analysis produced an empty comparison window. The "
                "prompts are too short to locate a query region; use prompts of "
                "at least a few tokens, or set query_marker to mark the query "
                "explicitly."
            )
        
        # Truncate all aligned tokens to same length
        aligned_tokens = [tokens[:query_window_len] for tokens in aligned_tokens]
        
        # Compute progression matrices: how each prompt differs from the previous one
        logger.info("Computing progression matrices")
        num_prompts = len(run_results)
        # Already L+1: hidden_states includes the embedding output.
        num_layers = len(run_results[0].hidden_states)
        
        progression_cos = np.zeros((num_prompts - 1, num_layers, query_window_len))
        progression_delta = np.zeros((num_prompts - 1, num_layers, query_window_len))
        
        for i in range(num_prompts - 1):
            result_a = run_results[i]
            result_b = run_results[i + 1]
            
            start_a = alignment_info[i]["start"]
            start_b = alignment_info[i + 1]["start"]
            
            # Compute similarity for this pair
            cos_mat = compute_cosine_similarity(
                result_a, result_b, start_a, start_b, query_window_len
            )
            dn_mat = compute_delta_norm(
                result_a, result_b, start_a, start_b, query_window_len
            )
            
            progression_cos[i] = cos_mat
            progression_delta[i] = dn_mat
        
        # Compute cumulative progression: how each prompt differs from zero-shot baseline
        logger.info("Computing cumulative progression (vs zero-shot)")
        cumulative_cos = np.zeros((num_prompts, num_layers, query_window_len))
        cumulative_delta = np.zeros((num_prompts, num_layers, query_window_len))
        
        zero_shot_result = run_results[0]
        zero_shot_start = alignment_info[0]["start"]
        
        for i in range(num_prompts):
            if i == 0:
                # Zero-shot vs itself = all ones for cos, all zeros for delta
                cumulative_cos[i] = np.ones((num_layers, query_window_len))
                cumulative_delta[i] = np.zeros((num_layers, query_window_len))
            else:
                result = run_results[i]
                start = alignment_info[i]["start"]
                
                cos_mat = compute_cosine_similarity(
                    zero_shot_result, result, zero_shot_start, start, query_window_len
                )
                dn_mat = compute_delta_norm(
                    zero_shot_result, result, zero_shot_start, start, query_window_len
                )
                
                cumulative_cos[i] = cos_mat
                cumulative_delta[i] = dn_mat
        
        # Compute PCA for selected layers
        logger.info(f"Computing {config.dim_reduction.upper()} (layers: {config.pca_layers})")
        pca_layers = PCAAnalyzer.select_pca_layers(
            config.pca_layers, num_layers, None  # No spike layer for progression
        )
        pca_payload = {}
        
        for layer_idx in pca_layers:
            # Fit a single reducer on concatenated activations so all prompts share a space.
            hiddens = []
            for i, result in enumerate(run_results):
                start = alignment_info[i]["start"]
                hidden = result.hidden_states[layer_idx][0, start : start + query_window_len, :].detach().cpu().numpy()
                hiddens.append(hidden)

            X = np.vstack(hiddens)

            layer_payload: Dict[str, Any] = {}
            explained_variance = None
            try:
                if config.dim_reduction == "umap":
                    embedding, _ = UMAPAnalyzer.fit_embedding(X, config)
                elif config.dim_reduction == "tsne":
                    embedding, _ = TSNEAnalyzer.fit_embedding(X, config)
                elif config.dim_reduction == "pca":
                    # One common basis, padded for a 3D plot even with short windows.
                    Xc = X - X.mean(axis=0)
                    _, singular, right = np.linalg.svd(Xc, full_matrices=False)
                    embedding = Xc @ right[:3].T
                    embedding = np.pad(embedding, ((0, 0), (0, 3 - embedding.shape[1])))
                    variance = singular ** 2
                    explained_variance = (variance[:3] / variance.sum()).tolist() if variance.sum() else [0.0] * min(3, len(variance))
                else:
                    raise ValueError(f"Unknown projection: {config.dim_reduction}")
            except Exception as exc:
                raise RuntimeError(f"{config.dim_reduction.upper()} projection failed at hidden-state index {layer_idx}: {exc}") from exc
            offset = 0
            for label, hidden in zip(prompt_labels, hiddens):
                layer_payload[label] = {
                    "pca": embedding[offset:offset + len(hidden)].tolist(),
                    "explained_variance": explained_variance,
                }
                offset += len(hidden)

            pca_payload[str(layer_idx)] = layer_payload
        
        # Keep captures inspectable for every progression step. These are
        # descriptive measurements, not evidence of a causal effect.
        component_layers = sorted({min(i, num_layers - 2) for i in pca_layers})
        attention_payload, mlp_payload, predictions_payload = {}, {}, {}
        predictor = PredictionAnalyzer(model, tokenizer, topk=config.topk)
        for i, (label, captured) in enumerate(zip(prompt_labels, run_results)):
            start = alignment_info[i]["start"]
            predictions_payload[label] = predictor.compute_predictions_analysis_single(
                captured, num_layers - 1, start, query_window_len,
            )
            if config.enable_attention_capture:
                attention_payload[label] = {}
                for block in component_layers:
                    attention = captured.attention_weights[block][0].float().numpy()
                    window = attention[:, start:start + query_window_len, start:start + query_window_len]
                    attention_payload[label][str(block)] = {
                        "mean_over_heads": window.mean(axis=0).tolist(),
                        "mean_window_mass_by_head": window.sum(axis=-1).mean(axis=-1).tolist(),
                    }
            if config.enable_mlp_capture:
                mlp_payload[label] = {}
                for block in component_layers:
                    activations = captured.mlp_activations[block][0, start:start + query_window_len].float().numpy()
                    mlp_payload[label][str(block)] = {
                        "token_norms": np.linalg.norm(activations, axis=-1).tolist(),
                        "activation_space": "residual_channels",
                    }

        # Compute example impact: which layers are most affected by adding examples
        logger.info("Computing example impact analysis")
        example_impact = {}
        
        # Average delta norm per layer across all progression steps
        layer_impact = np.mean(progression_delta, axis=(0, 2))  # Average over prompts and tokens
        most_affected_layers = np.argsort(layer_impact)[::-1][:10]  # Top 10 most affected
        
        example_impact = {
            "layer_impact": layer_impact.tolist(),
            "most_affected_layers": most_affected_layers.tolist(),
            "impact_by_step": [
                {
                    "step": f"{prompt_labels[i]} → {prompt_labels[i+1]}",
                    "layer_impact": np.mean(progression_delta[i], axis=1).tolist(),
                }
                for i in range(num_prompts - 1)
            ],
        }
        
        # Get resolved model ID
        resolved_model_id = ModelLoader.resolve_model_id(config.model)
        
        # Build metadata
        meta = {
            "model": resolved_model_id,
            "original_model": config.model,
            "prompts": config.prompts,
            "prompt_labels": prompt_labels,
            "analysis_mode": "progression",
            "query_marker": config.query_marker,
            "device": str(model.device),
            "dtype": str(model.dtype).removeprefix("torch."),
            "requested_device": config.device,
            "window": config.window,
            "query_window_len": query_window_len,
            "num_prompts": num_prompts,
            "num_layers": num_layers,
            "dim_reduction": config.dim_reduction,
            "input_token_counts": [len(captured.token_strs) for captured in run_results],
            "component_layers": component_layers,
        }
        
        # Create progression result
        result = ProgressionResult(
            meta=meta,
            run_results=run_results,
            prompt_labels=prompt_labels,
            aligned_tokens=aligned_tokens,
            alignment_info=alignment_info,
            progression_cos=progression_cos,
            progression_delta=progression_delta,
            cumulative_cos=cumulative_cos,
            cumulative_delta=cumulative_delta,
            query_starts=query_starts,
            query_window_len=query_window_len,
            pca_payload=pca_payload,
            example_impact=example_impact,
            attention_payload=attention_payload or None,
            mlp_payload=mlp_payload or None,
            predictions_payload=predictions_payload,
        )
        
        logger.info("Progression analysis completed successfully")
        return result

    @staticmethod
    def save_results(result: ProgressionResult, out_dir: Path):
        """
        Save progression results to files.
        
        Args:
            result: ProgressionResult to save
            out_dir: Output directory
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        meta_file = out_dir / "meta.json"
        with open(meta_file, "w") as f:
            json.dump(result.meta, f, indent=2)
        
        # Save progression matrices
        progression_cos_file = out_dir / "progression_cos.npy"
        progression_delta_file = out_dir / "progression_delta.npy"
        np.save(progression_cos_file, result.progression_cos)
        np.save(progression_delta_file, result.progression_delta)
        
        # Save cumulative matrices
        cumulative_cos_file = out_dir / "cumulative_cos.npy"
        cumulative_delta_file = out_dir / "cumulative_delta.npy"
        np.save(cumulative_cos_file, result.cumulative_cos)
        np.save(cumulative_delta_file, result.cumulative_delta)
        
        # Save PCA payload
        if result.pca_payload:
            pca_file = out_dir / "pca_payload.json"
            with open(pca_file, "w") as f:
                json.dump(result.pca_payload, f, indent=2)
        
        # Save example impact
        if result.example_impact:
            impact_file = out_dir / "example_impact.json"
            with open(impact_file, "w") as f:
                json.dump(result.example_impact, f, indent=2)
        
        for name, payload in (
            ("attention_payload", result.attention_payload),
            ("mlp_payload", result.mlp_payload),
            ("predictions", result.predictions_payload),
        ):
            if payload:
                (out_dir / f"{name}.json").write_text(json.dumps(payload, indent=2))

        logger.info(f"Progression results saved to {out_dir}")
