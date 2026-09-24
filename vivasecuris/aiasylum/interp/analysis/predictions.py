"""Prediction analysis for token predictions."""

import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple
import numpy as np

from vivasecuris.aiasylum.interp.data.models import RunResult


class PredictionAnalyzer:
    """Analyzes token predictions from activations."""

    def __init__(self, model, tokenizer, topk: int = 10):
        """
        Initialize prediction analyzer.
        
        Args:
            model: Model instance (needed for unembedding matrix)
            tokenizer: Tokenizer instance
            topk: Number of top predictions to consider
        """
        self.model = model
        self.tokenizer = tokenizer
        self.topk = topk
        
        # Get unembedding matrix (output projection)
        if hasattr(model, "lm_head"):
            self.unembedding = model.lm_head.weight  # [vocab_size, hidden_dim]
        elif hasattr(model, "embed_out"):
            self.unembedding = model.embed_out.weight
        else:
            # Fallback: try to find output embedding
            raise ValueError("Could not find unembedding matrix in model")

    def get_predictions_at_position(
        self,
        hidden_state: torch.Tensor,
    ) -> Tuple[List[int], List[float]]:
        """
        Get top-k predictions for a hidden state.
        
        Args:
            hidden_state: Hidden state tensor [hidden_dim]
            
        Returns:
            Tuple of (top_k_indices, top_k_probs)
        """
        # Project through unembedding
        logits = hidden_state @ self.unembedding.T  # [vocab_size]
        probs = F.softmax(logits, dim=-1)
        
        # Get top-k
        top_k_probs, top_k_indices = torch.topk(probs, k=self.topk)
        
        return top_k_indices.tolist(), top_k_probs.tolist()

    def compute_predictions_single(
        self,
        result: RunResult,
        layer_idx: int,
        start: int,
        window_len: int,
    ) -> Dict[str, Any]:
        """
        Compute prediction analysis for a single run over a window.

        Args:
            result: Single run result
            layer_idx: Layer to analyze
            start: Start position in sequence
            window_len: Length of window

        Returns:
            Dictionary with predictions (list of position, token, top tokens, probs),
            token_to_token_diff (list of norms), layer_idx.
        """
        predictions_data = []
        token_to_token_diff = []
        hs = result.hidden_states[layer_idx][0, start : start + window_len, :]
        for pos in range(window_len):
            top_indices, probs = self.get_predictions_at_position(hs[pos])
            token_str = result.token_strs[start + pos] if start + pos < len(result.token_strs) else ""
            predictions_data.append({
                "position": pos,
                "token": token_str,
                "top_tokens": [self.tokenizer.decode([idx]) for idx in top_indices],
                "probs": probs,
            })
            if pos > 0:
                diff = torch.linalg.vector_norm(hs[pos] - hs[pos - 1]).item()
                token_to_token_diff.append(diff)
        return {
            "predictions": predictions_data,
            "token_to_token_diff": token_to_token_diff,
            "layer_idx": layer_idx,
        }

    def compute_predictions_analysis_single(
        self,
        result: RunResult,
        layer_idx: int,
        start: int,
        window_len: int,
    ) -> Dict[str, Any]:
        """Alias for compute_predictions_single for plan/API consistency."""
        return self.compute_predictions_single(result, layer_idx, start, window_len)

    def compute_predictions_analysis(
        self,
        result_a: RunResult,
        result_b: RunResult,
        layer_idx: int,
        start_a: int,
        start_b: int,
        window_len: int,
    ) -> Dict[str, Any]:
        """
        Compute prediction analysis for aligned window.
        
        Args:
            result_a: First run result
            result_b: Second run result
            layer_idx: Layer to analyze
            start_a: Start position in sequence A
            start_b: Start position in sequence B
            window_len: Length of aligned window
            
        Returns:
            Dictionary with prediction analysis data
        """
        predictions_data = []
        token_to_token_diff_a = []
        token_to_token_diff_b = []
        
        # Get hidden states for this layer
        hs_a = result_a.hidden_states[layer_idx][0, start_a : start_a + window_len, :]
        hs_b = result_b.hidden_states[layer_idx][0, start_b : start_b + window_len, :]
        
        # Analyze each token position
        for pos in range(window_len):
            # Get predictions
            top_a, prob_a = self.get_predictions_at_position(hs_a[pos])
            top_b, prob_b = self.get_predictions_at_position(hs_b[pos])
            
            # Compute overlap
            overlap = len(set(top_a) & set(top_b)) / self.topk
            
            # Get token strings
            token_a = result_a.token_strs[start_a + pos] if start_a + pos < len(result_a.token_strs) else ""
            token_b = result_b.token_strs[start_b + pos] if start_b + pos < len(result_b.token_strs) else ""
            
            predictions_data.append({
                "position": pos,
                "token_a": token_a,
                "token_b": token_b,
                "top_a": [self.tokenizer.decode([idx]) for idx in top_a],
                "top_b": [self.tokenizer.decode([idx]) for idx in top_b],
                "prob_a": prob_a,
                "prob_b": prob_b,
                "overlap": float(overlap),
            })
            
            # Compute token-to-token differences
            if pos > 0:
                diff_a = torch.linalg.vector_norm(hs_a[pos] - hs_a[pos - 1]).item()
                diff_b = torch.linalg.vector_norm(hs_b[pos] - hs_b[pos - 1]).item()
                token_to_token_diff_a.append(diff_a)
                token_to_token_diff_b.append(diff_b)
        
        # Compute overlap ratio over all positions
        overlap_ratios = [p["overlap"] for p in predictions_data]
        
        return {
            "predictions": predictions_data,
            "token_to_token_diff_a": token_to_token_diff_a,
            "token_to_token_diff_b": token_to_token_diff_b,
            "overlap_ratios": overlap_ratios,
            "layer_idx": layer_idx,
        }
