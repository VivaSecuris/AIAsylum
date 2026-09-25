"""Attribution methods for understanding component contributions."""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch

from vivasecuris.aiasylum.interp.data.models import RunResult
from vivasecuris.aiasylum.interp.analysis.residual_decomposition import decompose_residual_at_position
from vivasecuris.aiasylum.interp.analysis.attention_analysis import AttentionAnalyzer


class AttributionAnalyzer:
    """Attribution analysis using various methods."""

    @staticmethod
    def compute_direct_logit_attribution(
        result: RunResult,
        layer_idx: int,
        token_idx: int,
        start: int,
        window_len: int,
    ) -> Dict[str, Any]:
        """
        Compute direct logit attribution for attention heads.
        
        This attributes logit changes to individual source tokens via attention heads.
        
        Args:
            result: RunResult with attention weights
            layer_idx: Layer index
            token_idx: Target token index (relative to start)
            start: Start token position
            window_len: Window length
            
        Returns:
            Dictionary with logit attribution results
        """
        if result.attention_weights is None:
            return {
                "attribution_available": False,
                "head_attributions": [],
            }
        
        if layer_idx >= len(result.attention_weights):
            return {
                "attribution_available": False,
                "head_attributions": [],
            }
        
        attn = result.attention_weights[layer_idx]  # [1, num_heads, seq_len, seq_len]
        attn_np = attn[0, :, start + token_idx, start:start + window_len].detach().cpu().float().numpy()
        
        num_heads = attn_np.shape[0]
        head_attributions = []
        
        for head_idx in range(num_heads):
            head_attn = attn_np[head_idx, :]  # [window_len]
            
            # Get top attended tokens
            top_indices = np.argsort(head_attn)[::-1][:10]
            
            head_attributions.append({
                "head": head_idx,
                "attention_weights": head_attn.tolist(),
                "top_attended_tokens": [
                    {"token": int(idx), "weight": float(head_attn[idx])}
                    for idx in top_indices
                ],
                "total_attention": float(np.sum(head_attn)),
            })
        
        return {
            "attribution_available": True,
            "head_attributions": head_attributions,
            "num_heads": num_heads,
        }

    @staticmethod
    def classify_head_roles_by_contribution(
        result_a: RunResult,
        result_b: RunResult,
        layer_idx: int,
        start_a: int,
        start_b: int,
        window_len: int,
        task_performance_metric: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Label attention heads by how far their contribution score sits from
        the layer mean: above one standard deviation "facilitating", below
        "interfering", otherwise "irrelevant".

        This is a descriptive heuristic on captured activations. It trains no
        gates and measures no causal effect, so it must not be presented as
        Causal Head Gating; the payload says so in its ``claim`` field. For a
        causal head result use head-level activation patching.
        
        Args:
            result_a: First run result
            result_b: Second run result
            layer_idx: Layer index
            start_a: Start position in sequence A
            start_b: Start position in sequence B
            window_len: Window length
            task_performance_metric: Optional task performance metric [window_len]
            
        Returns:
            Dictionary with head role classifications
        """
        # This is a simplified implementation
        # Full CHG would require training gates and measuring causal impact
        
        head_contributions = AttentionAnalyzer.compute_head_contribution_scores(
            result_a, result_b, layer_idx, start_a, start_b, window_len
        )
        
        if not head_contributions.get("attention_available", False):
            return {
                "head_roles": [],
                "classification_available": False,
                "claim": "descriptive",
                "method": "contribution_threshold",
            }
        
        head_roles = []
        heads = head_contributions.get("heads", [])
        
        # Simplified classification based on contribution scores
        # In full CHG, this would use learned gates and causal measurements
        mean_contrib = np.mean([h["contribution_score"] for h in heads])
        std_contrib = np.std([h["contribution_score"] for h in heads])
        
        for head_info in heads:
            contrib = head_info["contribution_score"]
            
            # Classify based on contribution relative to mean
            if contrib > mean_contrib + std_contrib:
                role = "facilitating"
            elif contrib < mean_contrib - std_contrib:
                role = "interfering"
            else:
                role = "irrelevant"
            
            head_roles.append({
                "head": head_info["head"],
                "role": role,
                "contribution_score": contrib,
                "mean_attention_diff": head_info["mean_attention_diff"],
            })
        
        return {
            "head_roles": head_roles,
            "classification_available": True,
            "claim": "descriptive",
            "method": "contribution_threshold",
            "num_facilitating": sum(1 for h in head_roles if h["role"] == "facilitating"),
            "num_interfering": sum(1 for h in head_roles if h["role"] == "interfering"),
            "num_irrelevant": sum(1 for h in head_roles if h["role"] == "irrelevant"),
        }

    # Old name, kept so external callers fail loudly in review rather than at
    # import time. Same heuristic; the label was the problem.
    classify_head_roles_causal_head_gating = classify_head_roles_by_contribution

    @staticmethod
    def identify_mixed_heads(
        result: RunResult,
        layer_idx: int,
        start: int,
        window_len: int,
        variance_threshold: float = 0.1,
    ) -> List[Dict[str, Any]]:
        """
        Identify "mixed heads" that perform additive updates from different sources.
        
        Mixed heads show high variance in attention patterns across tokens.
        
        Args:
            result: RunResult
            layer_idx: Layer index
            start: Start token position
            window_len: Window length
            
        Returns:
            List of mixed head information
        """
        attn = AttentionAnalyzer.extract_attention_weights(result, layer_idx, start, window_len)
        if attn is None:
            return []
        
        num_heads = attn.shape[0]
        mixed_heads = []
        
        for head_idx in range(num_heads):
            head_attn = attn[head_idx, :, :]  # [window_len, window_len]
            
            # Compute variance in attention patterns across tokens
            # High variance indicates mixed behavior
            attention_variance = np.var(head_attn, axis=0)  # Variance across source tokens
            mean_variance = np.mean(attention_variance)
            
            if mean_variance > variance_threshold:
                # This head shows mixed behavior
                mixed_heads.append({
                    "head": head_idx,
                    "attention_variance": float(mean_variance),
                    "is_mixed": True,
                })
        
        return sorted(mixed_heads, key=lambda x: x["attention_variance"], reverse=True)


class LogitAttributionAnalyzer:
    """True component-wise logit attribution via residual decomposition and unembedding."""

    @staticmethod
    def compute(
        result: RunResult,
        position: int,
        lm_head_weight: torch.Tensor,
        tokenizer: Any,
        topk: int = 10,
    ) -> Dict[str, Any]:
        """
        Decompose logits at a position into contributions from embed and each layer's attn/MLP.

        Uses linearity: logits = unembed(embed) + sum_l unembed(attn_out_l) + sum_l unembed(mlp_out_l).

        Args:
            result: RunResult with attn_outputs and mlp_activations (from hook capture)
            position: Token position (absolute index in sequence)
            lm_head_weight: Unembedding matrix [vocab_size, hidden_dim]
            tokenizer: For decoding top token IDs
            topk: Number of top tokens to include per component

        Returns:
            Dict with by_component (list of {type, layer, logit_contribution, top_tokens, norm}),
            total_logits (optional, for sanity check), and logit_lens (optional list of cumulative logits per layer).
        """
        comps = decompose_residual_at_position(result, position)
        if not comps:
            return {
                "attribution_available": False,
                "reason": "No residual decomposition (missing attn_outputs/mlp_activations)",
            }

        # lm_head_weight: [vocab_size, d]
        vocab_size = lm_head_weight.shape[0]
        by_component: List[Dict[str, Any]] = []

        for name, vec in comps.items():
            # vec: [d], logit_contrib: [vocab_size]
            vec_2d = vec.unsqueeze(0)
            logit_contrib = (vec_2d @ lm_head_weight.t()).squeeze(0)
            logit_contrib_np = logit_contrib.cpu().float().numpy()

            top_vals, top_idx = torch.topk(logit_contrib, k=min(topk, vocab_size))
            top_tokens = [
                {"token": tokenizer.decode([int(i)]), "logit": float(logit_contrib[i])}
                for i in top_idx.tolist()
            ]

            comp_type = "embed" if name == "embed" else ("attn" if name.startswith("attn_") else "mlp")
            layer_idx: Optional[int] = None
            if comp_type != "embed":
                try:
                    layer_idx = int(name.split("_")[1])
                except (IndexError, ValueError):
                    pass

            by_component.append({
                "component": name,
                "type": comp_type,
                "layer": layer_idx,
                "logit_contribution_norm": float(torch.linalg.norm(logit_contrib).item()),
                "top_tokens": top_tokens,
                "norm": float(torch.linalg.norm(vec).item()),
            })

        # Total logits from model at this position (sanity check)
        model_logits = result.logits[0, position, :].detach().cpu().float()
        summed_logits = sum(
            (comps[name].unsqueeze(0) @ lm_head_weight.t()).squeeze(0)
            for name in comps
        )

        return {
            "attribution_available": True,
            "position": position,
            "by_component": by_component,
            "total_logits_shape": list(summed_logits.shape),
            "model_logits_match": bool(
                torch.allclose(summed_logits, model_logits, atol=1e-2, rtol=1e-2)
            ),
            "top_components": sorted(
                by_component,
                key=lambda x: x["logit_contribution_norm"],
                reverse=True,
            )[:10],
        }
