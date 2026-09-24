"""PCA analysis for dimensionality reduction."""

import numpy as np
from sklearn.decomposition import PCA
from typing import List, Dict, Any, Tuple, Optional
import torch

from vivasecuris.aiasylum.interp.data.models import RunResult


def compute_pca3_single(
    result: RunResult,
    layer_idx: int,
    start: int,
    window_len: int,
    random_state: int = 0,
) -> Dict[str, Any]:
    """
    Compute 3D PCA for a single run (one trajectory).

    Args:
        result: Single run result
        layer_idx: Layer index
        start: Start position in sequence
        window_len: Length of window
        random_state: Random state for PCA

    Returns:
        Dictionary with trajectory (list), explained_variance_ratio (list).
        Keys: "trajectory", "explained_variance_ratio" (single-run format for dashboard).
    """
    hs = result.hidden_states[layer_idx][0, start : start + window_len, :]
    X = hs.numpy()
    n_components = min(3, X.shape[0], X.shape[1])
    pca = PCA(n_components=n_components, random_state=random_state)
    trajectory = pca.fit_transform(X)
    if n_components < 3:
        pad_width = 3 - n_components
        trajectory = np.pad(trajectory, ((0, 0), (0, pad_width)), mode="constant")
        explained = np.pad(
            pca.explained_variance_ratio_,
            (0, pad_width),
            mode="constant",
            constant_values=0.0,
        )
    else:
        explained = pca.explained_variance_ratio_
    return {
        "trajectory": trajectory.tolist(),
        "explained_variance_ratio": explained.tolist(),
    }


class PCAAnalyzer:
    """Handles PCA computation for activation analysis."""

    @staticmethod
    def compute_pca3_for_layer(
        result_a: RunResult,
        result_b: RunResult,
        layer_idx: int,
        start_a: int,
        start_b: int,
        window_len: int,
    ) -> Dict[str, Any]:
        """
        Compute 3D PCA for a single layer.
        
        Args:
            result_a: First run result
            result_b: Second run result
            layer_idx: Layer index
            start_a: Start position in sequence A
            start_b: Start position in sequence B
            window_len: Length of aligned window
            
        Returns:
            Dictionary with PCA results including:
            - trajectory_a: [window_len, 3] array
            - trajectory_b: [window_len, 3] array
            - explained_variance_ratio: [3] array
        """
        # Extract aligned windows
        hs_a = result_a.hidden_states[layer_idx][0, start_a : start_a + window_len, :]
        hs_b = result_b.hidden_states[layer_idx][0, start_b : start_b + window_len, :]
        
        # Convert to numpy
        hs_a_np = hs_a.numpy()  # [window_len, hidden_dim]
        hs_b_np = hs_b.numpy()  # [window_len, hidden_dim]
        
        # Concatenate for fitting PCA
        X = np.vstack([hs_a_np, hs_b_np])  # [2*window_len, hidden_dim]
        
        # Fit PCA with 3 components
        n_components = min(3, X.shape[0], X.shape[1])
        pca = PCA(n_components=n_components)
        pca.fit(X)
        
        # Transform both sequences
        trajectory_a = pca.transform(hs_a_np)  # [window_len, n_components]
        trajectory_b = pca.transform(hs_b_np)  # [window_len, n_components]
        
        # Pad to 3 components if needed
        if n_components < 3:
            pad_width = 3 - n_components
            trajectory_a = np.pad(trajectory_a, ((0, 0), (0, pad_width)), mode='constant')
            trajectory_b = np.pad(trajectory_b, ((0, 0), (0, pad_width)), mode='constant')
            explained_variance_ratio = np.pad(
                pca.explained_variance_ratio_,
                (0, pad_width),
                mode='constant',
                constant_values=0.0
            )
        else:
            explained_variance_ratio = pca.explained_variance_ratio_
        
        return {
            "trajectory_a": trajectory_a.tolist(),
            "trajectory_b": trajectory_b.tolist(),
            "explained_variance_ratio": explained_variance_ratio.tolist(),
        }

    @staticmethod
    def select_pca_layers(
        pca_layers_str: str,
        num_layers: int,
        spike_layer: Optional[int] = None,
    ) -> List[int]:
        """
        Select which layers to compute PCA for.
        
        Args:
            pca_layers_str: Layer selection string ('auto', 'all', or comma list)
            num_layers: Total number of layers
            spike_layer: Layer of maximum divergence to centre 'auto' on. None
                when there is no pairwise divergence to speak of (progression
                mode passes None), in which case 'auto' centres on the middle
                layer instead.
            
        Returns:
            List of layer indices to compute PCA for
        """
        if pca_layers_str == "auto":
            # Spike layer ±2 neighbors, plus embeddings(0) and final(-1)
            layers = set([0, num_layers - 1])  # Embeddings and final layer
            # With no spike layer, centre on the middle of the stack.
            anchor = spike_layer if spike_layer is not None else num_layers // 2
            # Add anchor ±2 neighbors
            for offset in range(-2, 3):
                layer_idx = anchor + offset
                if 0 <= layer_idx < num_layers:
                    layers.add(layer_idx)
            return sorted(list(layers))
        
        elif pca_layers_str == "all":
            return list(range(num_layers))
        
        else:
            # Parse comma-separated list
            layer_indices = []
            for layer_str in pca_layers_str.split(","):
                layer_str = layer_str.strip()
                try:
                    layer_idx = int(layer_str)
                    # Handle negative indices
                    if layer_idx < 0:
                        layer_idx = num_layers + layer_idx
                    if 0 <= layer_idx < num_layers:
                        layer_indices.append(layer_idx)
                except ValueError:
                    continue
            return sorted(list(set(layer_indices)))
