"""MLP activation analysis, distinguishing neurons from output residual channels."""

from typing import Dict, Any, List, Optional
import numpy as np
import torch

from vivasecuris.aiasylum.interp.data.models import RunResult


class MLPAnalyzer:
    """Analyzes MLP neuron activations and contributions."""

    @staticmethod
    def activation_metadata(result: RunResult, layer_idx: int) -> Dict[str, str]:
        neurons = layer_idx in (result.pre_mlp_activations or {})
        return {
            "activation_space": "mlp_neurons" if neurons else "residual_channels",
            "unit_label": "neuron" if neurons else "residual channel",
            "claim": "descriptive",
        }

    @staticmethod
    def extract_mlp_activations(
        result: RunResult,
        layer_idx: int,
        start: int,
        window_len: int,
    ) -> Optional[np.ndarray]:
        """
        Extract MLP activations for a specific layer and token range.
        
        Args:
            result: RunResult with MLP activations
            layer_idx: Layer index
            start: Start token position
            window_len: Window length
            
        Returns:
            MLP activations array [window_len, units] or None. Prefer the input
            to the down projection (neurons); otherwise report output channels.
        """
        mlp = (result.pre_mlp_activations or {}).get(layer_idx)
        if mlp is None:
            mlp = (result.mlp_activations or {}).get(layer_idx)
        if mlp is None:
            return None
        mlp_np = mlp[0, start:start + window_len, :].detach().cpu().float().numpy()
        return mlp_np

    @staticmethod
    def compute_neuron_contribution_scores(
        result_a: RunResult,
        result_b: RunResult,
        layer_idx: int,
        start_a: int,
        start_b: int,
        window_len: int,
    ) -> Dict[str, Any]:
        """
        Compute contribution scores for each MLP neuron.
        
        Args:
            result_a: First run result
            result_b: Second run result
            layer_idx: Layer index
            start_a: Start position in sequence A
            start_b: Start position in sequence B
            window_len: Window length
            
        Returns:
            Dictionary with neuron contribution analysis
        """
        mlp_a = MLPAnalyzer.extract_mlp_activations(result_a, layer_idx, start_a, window_len)
        mlp_b = MLPAnalyzer.extract_mlp_activations(result_b, layer_idx, start_b, window_len)
        
        if mlp_a is None or mlp_b is None:
            return {
                "neurons": [],
                "num_neurons": 0,
                "mlp_available": False,
            }
        
        num_neurons = mlp_a.shape[1]
        
        # Compute activation differences
        activation_diff = np.abs(mlp_a - mlp_b)  # [window_len, num_neurons]
        
        # Compute per-neuron divergence scores
        neuron_scores = []
        for neuron_idx in range(num_neurons):
            neuron_diff = activation_diff[:, neuron_idx]
            mean_diff = float(np.mean(neuron_diff))
            max_diff = float(np.max(neuron_diff))
            total_diff = float(np.sum(neuron_diff))
            
            # Also compute activation magnitude
            mean_activation_a = float(np.mean(np.abs(mlp_a[:, neuron_idx])))
            mean_activation_b = float(np.mean(np.abs(mlp_b[:, neuron_idx])))
            
            neuron_scores.append({
                "neuron": neuron_idx,
                "mean_activation_diff": mean_diff,
                "max_activation_diff": max_diff,
                "total_activation_diff": total_diff,
                "mean_activation_a": mean_activation_a,
                "mean_activation_b": mean_activation_b,
                "contribution_score": mean_diff,  # Simplified scoring
            })
        
        # Sort by contribution score
        neuron_scores.sort(key=lambda x: x["contribution_score"], reverse=True)
        
        return {
            **MLPAnalyzer.activation_metadata(result_a, layer_idx),
            "neurons": neuron_scores,
            "num_neurons": num_neurons,
            "mlp_available": True,
            "top_neurons": neuron_scores[:50],  # Top 50 contributing neurons
        }

    @staticmethod
    def compute_neuron_activation_distribution(
        result: RunResult,
        layer_idx: int,
        neuron_idx: int,
        start: int,
        window_len: int,
    ) -> Dict[str, float]:
        """
        Compute activation distribution statistics for a specific neuron.
        
        Args:
            result: RunResult
            layer_idx: Layer index
            neuron_idx: Neuron index
            start: Start token position
            window_len: Window length
            
        Returns:
            Dictionary with distribution statistics
        """
        mlp = MLPAnalyzer.extract_mlp_activations(result, layer_idx, start, window_len)
        if mlp is None or neuron_idx >= mlp.shape[1]:
            return {}
        
        activations = mlp[:, neuron_idx]
        
        return {
            "mean": float(np.mean(activations)),
            "std": float(np.std(activations)),
            "min": float(np.min(activations)),
            "max": float(np.max(activations)),
            "median": float(np.median(activations)),
            "q25": float(np.percentile(activations, 25)),
            "q75": float(np.percentile(activations, 75)),
        }
