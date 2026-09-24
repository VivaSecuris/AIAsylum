"""Prediction analysis for token predictions."""

import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple
import numpy as np

from vivasecuris.aiasylum.interp.core.arch import final_hidden_is_normed, get_final_norm
from vivasecuris.aiasylum.interp.data.models import RunResult


class PredictionAnalyzer:
    """Logit lens: read intermediate residuals through the model's own unembedding.

    The final normalisation is applied first (``apply_final_norm=True``), so the
    reading at the last hidden state is exactly the model's own logits and the
    readings at earlier layers are on the scale the unembedding was trained
    for. The un-normalised variant is kept behind the flag for comparison; it
    is not the default because it silently measures something different at
    every depth.
    """

    def __init__(self, model, tokenizer, topk: int = 10, apply_final_norm: bool = True):
        """
        Initialize prediction analyzer.
        
        Args:
            model: Model instance (needed for unembedding matrix)
            tokenizer: Tokenizer instance
            topk: Number of top predictions to consider
            apply_final_norm: Run hidden states through the final norm before
                the unembedding (the correct logit lens).
        """
        self.model = model
        self.tokenizer = tokenizer
        self.topk = topk
        self.apply_final_norm = apply_final_norm
        self.final_norm = get_final_norm(model) if apply_final_norm else None
        # transformers may hand back the last hidden state already normed; in
        # that case the norm is applied to intermediate layers only.
        self.final_is_normed = final_hidden_is_normed(model) if apply_final_norm else False
        
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
        is_final: bool = False,
    ) -> Tuple[List[int], List[float]]:
        """
        Get top-k predictions for a hidden state.
        
        Args:
            hidden_state: Hidden state tensor [hidden_dim]
            
        Returns:
            Tuple of (top_k_indices, top_k_probs)
        """
        logits = self.lens_logits(hidden_state, is_final=is_final)
        probs = F.softmax(logits, dim=-1)
        
        # Get top-k
        top_k_probs, top_k_indices = torch.topk(probs, k=self.topk)
        
        return top_k_indices.tolist(), top_k_probs.tolist()

    def lens_logits(self, hidden_state: torch.Tensor, is_final: bool = False) -> torch.Tensor:
        """Full-vocabulary lens logits for one hidden state, on CPU in float32.

        The hidden state is moved to the unembedding's device and dtype, so a
        CPU-cached residual can be read through a model resident on MPS or CUDA.
        ``is_final`` marks the last hidden-state entry, which is skipped by the
        norm when transformers has already normed it.
        """
        W = self.unembedding
        with torch.no_grad():
            h = hidden_state.to(device=W.device, dtype=W.dtype)
            if self.final_norm is not None and not (is_final and self.final_is_normed):
                h = self.final_norm(h)
            logits = h @ W.T  # [vocab_size]
        return logits.float().cpu()

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
        is_final = layer_idx == len(result.hidden_states) - 1
        for pos in range(window_len):
            top_indices, probs = self.get_predictions_at_position(hs[pos], is_final=is_final)
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
        is_final = layer_idx == len(result_a.hidden_states) - 1
        
        # Analyze each token position
        for pos in range(window_len):
            # Get predictions
            top_a, prob_a = self.get_predictions_at_position(hs_a[pos], is_final=is_final)
            top_b, prob_b = self.get_predictions_at_position(hs_b[pos], is_final=is_final)
            
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
