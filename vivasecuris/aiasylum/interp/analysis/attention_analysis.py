"""Attention head analysis for component localization."""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torch

from vivasecuris.aiasylum.interp.data.models import RunResult


class AttentionAnalyzer:
    """Analyzes attention patterns and head contributions."""

    @staticmethod
    def extract_attention_weights(
        result: RunResult,
        layer_idx: int,
        start: int,
        window_len: int,
    ) -> Optional[np.ndarray]:
        """
        Extract attention weights for a specific layer and token range.
        
        Args:
            result: RunResult with attention weights
            layer_idx: Layer index
            start: Start token position
            window_len: Window length
            
        Returns:
            Attention weights array [num_heads, window_len, window_len] or None
        """
        if result.attention_weights is None:
            return None
        
        if layer_idx >= len(result.attention_weights):
            return None
        
        attn = result.attention_weights[layer_idx]  # [1, num_heads, seq_len, seq_len]
        attn_np = attn[0, :, start:start + window_len, start:start + window_len].detach().cpu().float().numpy()
        return attn_np

    @staticmethod
    def compute_head_outputs(
        result: RunResult,
        layer_idx: int,
        start: int,
        window_len: int,
        model: Optional[Any] = None,
    ) -> Optional[np.ndarray]:
        """
        Compute per-head output contributions (OV circuit) when QKV and model available.

        Args:
            result: RunResult with attention_weights and optionally qkv_outputs
            layer_idx: Layer index
            start: Start token position
            window_len: Window length
            model: Model (for W_O); required when using qkv_outputs

        Returns:
            Head outputs array [num_heads, window_len, d_model] or None
        """
        if model is None:
            return None
        from vivasecuris.aiasylum.interp.analysis.ov_qk import compute_per_head_outputs
        per_head = compute_per_head_outputs(result, model, layer_idx, start, window_len)
        if per_head is None:
            return None
        return per_head.cpu().numpy()

    @staticmethod
    def compute_head_contribution_scores(
        result_a: RunResult,
        result_b: RunResult,
        layer_idx: int,
        start_a: int,
        start_b: int,
        window_len: int,
    ) -> Dict[str, Any]:
        """
        Compute contribution scores for each attention head.
        
        Args:
            result_a: First run result
            result_b: Second run result
            layer_idx: Layer index
            start_a: Start position in sequence A
            start_b: Start position in sequence B
            window_len: Window length
            
        Returns:
            Dictionary with head contribution analysis
        """
        attn_a = AttentionAnalyzer.extract_attention_weights(result_a, layer_idx, start_a, window_len)
        attn_b = AttentionAnalyzer.extract_attention_weights(result_b, layer_idx, start_b, window_len)
        
        if attn_a is None or attn_b is None:
            return {
                "heads": [],
                "num_heads": 0,
                "attention_available": False,
            }
        
        num_heads = attn_a.shape[0]
        
        # Compute attention pattern differences
        attn_diff = np.abs(attn_a - attn_b)  # [num_heads, window_len, window_len]
        
        # Compute per-head divergence scores
        head_scores = []
        for head_idx in range(num_heads):
            head_diff = attn_diff[head_idx]
            mean_diff = float(np.mean(head_diff))
            max_diff = float(np.max(head_diff))
            total_diff = float(np.sum(head_diff))
            
            head_scores.append({
                "head": head_idx,
                "mean_attention_diff": mean_diff,
                "max_attention_diff": max_diff,
                "total_attention_diff": total_diff,
                "contribution_score": mean_diff,  # Simplified - could use more sophisticated scoring
            })
        
        # Sort by contribution score
        head_scores.sort(key=lambda x: x["contribution_score"], reverse=True)
        
        return {
            "heads": head_scores,
            "num_heads": num_heads,
            "attention_available": True,
            "top_heads": head_scores[:10],  # Top 10 contributing heads
        }

    @staticmethod
    def compute_aggregated_attention(
        result: RunResult,
        layer_idx: int,
        start: int,
        window_len: int,
    ) -> Optional[np.ndarray]:
        """
        Compute mean attention across all heads.
        
        Args:
            result: RunResult
            layer_idx: Layer index
            start: Start token position
            window_len: Window length
            
        Returns:
            Aggregated attention [window_len, window_len] or None
        """
        attn = AttentionAnalyzer.extract_attention_weights(result, layer_idx, start, window_len)
        if attn is None:
            return None
        
        # Mean across heads
        aggregated = np.mean(attn, axis=0)  # [window_len, window_len]
        return aggregated

    @staticmethod
    def get_top_attended_tokens(
        result: RunResult,
        layer_idx: int,
        head_idx: int,
        token_idx: int,
        start: int,
        window_len: int,
        top_k: int = 10,
    ) -> List[Tuple[int, float]]:
        """
        Get top-k tokens that a specific head attends to at a specific position.
        
        Args:
            result: RunResult
            layer_idx: Layer index
            head_idx: Head index
            token_idx: Token position (relative to start)
            start: Start token position
            window_len: Window length
            
        Returns:
            List of (token_index, attention_weight) tuples
        """
        attn = AttentionAnalyzer.extract_attention_weights(result, layer_idx, start, window_len)
        if attn is None:
            return []
        
        if head_idx >= attn.shape[0] or token_idx >= window_len:
            return []
        
        # Get attention weights for this head and token
        attn_weights = attn[head_idx, token_idx, :]  # [window_len]
        
        # Get top-k
        top_indices = np.argsort(attn_weights)[::-1][:top_k]
        top_weights = [(int(idx), float(attn_weights[idx])) for idx in top_indices]
        
        return top_weights
