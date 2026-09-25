"""Circuit analysis - identifying which components form circuits."""

from typing import Dict, Any, List, Optional, Set
import numpy as np

from vivasecuris.aiasylum.interp.analysis.attention_analysis import AttentionAnalyzer
from vivasecuris.aiasylum.interp.analysis.mlp_analysis import MLPAnalyzer
from vivasecuris.aiasylum.interp.data.models import RunResult


class CircuitAnalyzer:
    """Identifies circuits - groups of components that work together."""

    @staticmethod
    def build_circuit_card(
        result_a: RunResult,
        result_b: RunResult,
        layer_idx: int,
        token_idx: int,
        start_a: int,
        start_b: int,
        window_len: int,
        head_threshold: float = 0.1,
        neuron_threshold: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Build a circuit card for a specific (layer, token) location.
        
        Args:
            result_a: First run result
            result_b: Second run result
            layer_idx: Layer index
            token_idx: Token index (relative to start)
            start_a: Start position in sequence A
            start_b: Start position in sequence B
            window_len: Window length
            head_threshold: Threshold for head inclusion
            neuron_threshold: Threshold for neuron inclusion
            
        Returns:
            Dictionary with circuit information
        """
        # Get head contributions
        head_contributions = AttentionAnalyzer.compute_head_contribution_scores(
            result_a, result_b, layer_idx, start_a, start_b, window_len
        )
        
        # Get neuron contributions
        neuron_contributions = MLPAnalyzer.compute_neuron_contribution_scores(
            result_a, result_b, layer_idx, start_a, start_b, window_len
        )
        
        # Identify involved heads
        involved_heads = []
        if head_contributions.get("attention_available", False):
            for head_info in head_contributions.get("heads", []):
                if head_info["contribution_score"] > head_threshold:
                    involved_heads.append({
                        "head": head_info["head"],
                        "contribution": head_info["contribution_score"],
                        "contribution_pct": 0.0,  # Will compute below
                    })
        
        # Identify involved neurons
        involved_neurons = []
        if neuron_contributions.get("mlp_available", False):
            for neuron_info in neuron_contributions.get("neurons", []):
                if neuron_info["contribution_score"] > neuron_threshold:
                    involved_neurons.append({
                        "neuron": neuron_info["neuron"],
                        "contribution": neuron_info["contribution_score"],
                        "contribution_pct": 0.0,  # Will compute below
                    })
        
        # Compute contribution percentages
        total_head_contrib = sum(h["contribution"] for h in involved_heads)
        total_neuron_contrib = sum(n["contribution"] for n in involved_neurons)
        total_contrib = total_head_contrib + total_neuron_contrib
        
        if total_contrib > 0:
            for head in involved_heads:
                head["contribution_pct"] = (head["contribution"] / total_contrib) * 100
            for neuron in involved_neurons:
                neuron["contribution_pct"] = (neuron["contribution"] / total_contrib) * 100
        
        # Generate circuit summary
        summary_parts = []
        if involved_heads:
            summary_parts.append(f"{len(involved_heads)} attention heads")
        if involved_neurons:
            unit_label = neuron_contributions.get("unit_label", "component")
            summary_parts.append(f"{len(involved_neurons)} MLP {unit_label}s")
        
        circuit_summary = (
            f"Circuit at Layer {layer_idx}, Token {token_idx} involves "
            + " and ".join(summary_parts)
            if summary_parts
            else "No significant circuit components identified"
        )
        
        return {
            **MLPAnalyzer.activation_metadata(result_a, layer_idx),
            "layer": layer_idx,
            "token": token_idx,
            "involved_heads": involved_heads,
            "involved_neurons": involved_neurons,
            "total_components": len(involved_heads) + len(involved_neurons),
            "circuit_summary": circuit_summary,
            "head_available": head_contributions.get("attention_available", False),
            "neuron_available": neuron_contributions.get("mlp_available", False),
        }

    @staticmethod
    def identify_safety_neuron_clusters(
        result_a: RunResult,
        result_b: RunResult,
        layer_idx: int,
        start_a: int,
        start_b: int,
        window_len: int,
        activation_contrast_threshold: float = 0.2,
    ) -> Dict[str, Any]:
        """
        Identify neurons that show high activation contrast between safe/unsafe outputs.
        
        Args:
            result_a: First run result (typically safe)
            result_b: Second run result (typically unsafe)
            layer_idx: Layer index
            start_a: Start position in sequence A
            start_b: Start position in sequence B
            window_len: Window length
            activation_contrast_threshold: Minimum contrast threshold
            
        Returns:
            Dictionary with safety neuron information
        """
        mlp_a = MLPAnalyzer.extract_mlp_activations(result_a, layer_idx, start_a, window_len)
        mlp_b = MLPAnalyzer.extract_mlp_activations(result_b, layer_idx, start_b, window_len)
        
        if mlp_a is None or mlp_b is None:
            return {
                "safety_neurons": [],
                "num_safety_neurons": 0,
                "mlp_available": False,
            }
        
        num_neurons = mlp_a.shape[1]
        
        # Compute activation contrast
        mean_activation_a = np.mean(np.abs(mlp_a), axis=0)  # [num_neurons]
        mean_activation_b = np.mean(np.abs(mlp_b), axis=0)  # [num_neurons]
        
        # Contrast = difference in activation magnitudes
        activation_contrast = np.abs(mean_activation_a - mean_activation_b)
        
        # Identify safety neurons (high contrast)
        safety_neurons = []
        for neuron_idx in range(num_neurons):
            contrast = activation_contrast[neuron_idx]
            if contrast > activation_contrast_threshold:
                safety_neurons.append({
                    "neuron": neuron_idx,
                    "activation_contrast": float(contrast),
                    "mean_activation_a": float(mean_activation_a[neuron_idx]),
                    "mean_activation_b": float(mean_activation_b[neuron_idx]),
                })
        
        # Sort by contrast
        safety_neurons.sort(key=lambda x: x["activation_contrast"], reverse=True)
        
        return {
            **MLPAnalyzer.activation_metadata(result_a, layer_idx),
            "caveat": "Activation contrast in one prompt pair does not identify a safety function.",
            "safety_neurons": safety_neurons,
            "num_safety_neurons": len(safety_neurons),
            "total_neurons": num_neurons,
            "safety_neuron_percentage": (len(safety_neurons) / num_neurons) * 100 if num_neurons > 0 else 0,
            "mlp_available": True,
        }
