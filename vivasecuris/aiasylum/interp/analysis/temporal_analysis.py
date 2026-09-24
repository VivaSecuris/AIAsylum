"""Temporal localization analysis - detecting when divergence occurs."""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np


class TemporalAnalyzer:
    """Analyzes temporal patterns in divergence."""

    @staticmethod
    def detect_first_divergence_token(
        dn_mat: np.ndarray,
        threshold: float = 0.1,
    ) -> int:
        """
        Detect the first token where divergence exceeds threshold.
        
        Args:
            dn_mat: Delta norm matrix [num_layers, window_len]
            threshold: Divergence threshold
            
        Returns:
            Index of first divergence token, or 0 if none found
        """
        if dn_mat.size == 0:
            return 0
        
        num_tokens = dn_mat.shape[1]
        
        for token_idx in range(num_tokens):
            # Check if any layer has significant divergence at this token
            layer_divs = dn_mat[:, token_idx]
            if np.any(layer_divs > threshold):
                return token_idx
        
        return 0

    @staticmethod
    def detect_change_points(
        dn_mat: np.ndarray,
        window_size: int = 5,
        min_change: float = 0.05,
    ) -> List[Dict[str, Any]]:
        """
        Detect change points where divergence pattern shifts.
        
        Args:
            dn_mat: Delta norm matrix [num_layers, window_len]
            window_size: Window size for change detection
            min_change: Minimum change magnitude to consider
            
        Returns:
            List of change point dictionaries
        """
        if dn_mat.size == 0:
            return []
        
        num_tokens = dn_mat.shape[1]
        change_points = []
        
        # Compute mean divergence per token across layers
        mean_div_per_token = np.mean(dn_mat, axis=0)  # [window_len]
        
        for token_idx in range(window_size, num_tokens - window_size):
            # Compare windows before and after
            before_window = mean_div_per_token[token_idx - window_size:token_idx]
            after_window = mean_div_per_token[token_idx:token_idx + window_size]
            
            before_mean = np.mean(before_window)
            after_mean = np.mean(after_window)
            
            change_magnitude = abs(after_mean - before_mean)
            
            if change_magnitude > min_change:
                change_points.append({
                    "token": token_idx,
                    "change_magnitude": float(change_magnitude),
                    "before_mean": float(before_mean),
                    "after_mean": float(after_mean),
                    "direction": "increase" if after_mean > before_mean else "decrease",
                })
        
        return change_points

    @staticmethod
    def compute_divergence_timeline(
        dn_mat: np.ndarray,
        layer_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Compute divergence timeline for a specific layer or averaged.
        
        Args:
            dn_mat: Delta norm matrix [num_layers, window_len]
            layer_idx: Specific layer index, or None for average
            
        Returns:
            Dictionary with timeline data
        """
        if layer_idx is not None:
            divergence = dn_mat[layer_idx, :]
        else:
            divergence = np.mean(dn_mat, axis=0)
        
        num_tokens = len(divergence)
        tokens = list(range(num_tokens))
        
        return {
            "tokens": tokens,
            "divergence": divergence.tolist(),
            "mean_divergence": float(np.mean(divergence)),
            "max_divergence": float(np.max(divergence)),
            "max_divergence_token": int(np.argmax(divergence)),
        }

    @staticmethod
    def identify_critical_decision_points(
        dn_mat: np.ndarray,
        cos_mat: np.ndarray,
        min_divergence: float = 0.15,
        min_similarity_drop: float = 0.1,
    ) -> List[Dict[str, Any]]:
        """
        Identify tokens where model makes critical decisions (high divergence + similarity drop).
        
        Args:
            dn_mat: Delta norm matrix [num_layers, window_len]
            cos_mat: Cosine similarity matrix [num_layers, window_len]
            min_divergence: Minimum divergence threshold
            min_similarity_drop: Minimum similarity drop threshold
            
        Returns:
            List of critical decision point dictionaries
        """
        if dn_mat.size == 0 or cos_mat.size == 0:
            return []
        
        num_tokens = dn_mat.shape[1]
        decision_points = []
        
        # Compute mean across layers
        mean_div = np.mean(dn_mat, axis=0)
        mean_cos = np.mean(cos_mat, axis=0)
        
        for token_idx in range(1, num_tokens):
            divergence = mean_div[token_idx]
            similarity = mean_cos[token_idx]
            
            # Check if this is a critical point
            if divergence > min_divergence:
                # Check similarity drop from previous token
                if token_idx > 0:
                    prev_similarity = mean_cos[token_idx - 1]
                    similarity_drop = prev_similarity - similarity
                    
                    if similarity_drop > min_similarity_drop:
                        decision_points.append({
                            "token": token_idx,
                            "divergence": float(divergence),
                            "similarity": float(similarity),
                            "similarity_drop": float(similarity_drop),
                            "severity": "high" if divergence > 0.3 else "medium",
                        })
        
        return decision_points
