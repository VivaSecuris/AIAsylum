"""UMAP analysis for dimensionality reduction."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from vivasecuris.aiasylum.interp.core.config import Config
from vivasecuris.aiasylum.interp.data.models import RunResult


class UMAPAnalyzer:
    """Handles UMAP computation for activation analysis."""

    @staticmethod
    def _get_umap():
        try:
            import umap  # type: ignore

            return umap
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "UMAP requested but 'umap-learn' is not installed. "
                "Install it with: pip install umap-learn"
            ) from e

    @staticmethod
    def compute_umap3_for_layer(
        result_a: RunResult,
        result_b: RunResult,
        layer_idx: int,
        start_a: int,
        start_b: int,
        window_len: int,
        config: Optional[Config] = None,
    ) -> Dict[str, Any]:
        """
        Compute 3D UMAP for a single layer.

        Returns:
            Dictionary with UMAP results including:
            - trajectory_a: [window_len, 3] array
            - trajectory_b: [window_len, 3] array
        """
        umap = UMAPAnalyzer._get_umap()

        # Extract aligned windows
        hs_a = result_a.hidden_states[layer_idx][0, start_a : start_a + window_len, :]
        hs_b = result_b.hidden_states[layer_idx][0, start_b : start_b + window_len, :]

        # Convert to numpy
        hs_a_np = hs_a.detach().cpu().numpy()
        hs_b_np = hs_b.detach().cpu().numpy()

        # Fit UMAP on concatenated data for a shared embedding space
        X = np.vstack([hs_a_np, hs_b_np])

        n_neighbors = config.umap_n_neighbors if config else 15
        min_dist = config.umap_min_dist if config else 0.1
        metric = config.umap_metric if config else "cosine"
        random_state = config.umap_random_state if config else 0

        reducer = umap.UMAP(
            n_components=3,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
        )
        reducer.fit(X)

        trajectory_a = reducer.transform(hs_a_np)
        trajectory_b = reducer.transform(hs_b_np)

        return {
            "trajectory_a": trajectory_a.tolist(),
            "trajectory_b": trajectory_b.tolist(),
            "umap_params": {
                "n_neighbors": n_neighbors,
                "min_dist": float(min_dist),
                "metric": metric,
                "random_state": random_state,
            },
        }

    @staticmethod
    def compute_umap3_for_single_run(
        result: RunResult,
        layer_idx: int,
        start: int,
        window_len: int,
        config: Optional[Config] = None,
    ) -> Dict[str, Any]:
        """
        Compute 3D UMAP for a single run (one trajectory).

        Returns:
            Dictionary with "trajectory", "umap_params".
        """
        umap = UMAPAnalyzer._get_umap()
        hs = result.hidden_states[layer_idx][0, start : start + window_len, :]
        X = hs.detach().cpu().numpy()
        n_neighbors = config.umap_n_neighbors if config else 15
        min_dist = config.umap_min_dist if config else 0.1
        metric = config.umap_metric if config else "cosine"
        random_state = config.umap_random_state if config else 0
        reducer = umap.UMAP(
            n_components=3,
            n_neighbors=min(n_neighbors, max(2, X.shape[0] - 1)),
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
        )
        trajectory = reducer.fit_transform(X)
        return {
            "trajectory": trajectory.tolist(),
            "umap_params": {
                "n_neighbors": n_neighbors,
                "min_dist": float(min_dist),
                "metric": metric,
                "random_state": random_state,
            },
        }

