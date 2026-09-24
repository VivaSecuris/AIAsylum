"""t-SNE analysis for dimensionality reduction."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from vivasecuris.aiasylum.interp.core.config import Config
from vivasecuris.aiasylum.interp.data.models import RunResult


class TSNEAnalyzer:
    """Handles t-SNE computation for activation analysis."""

    @staticmethod
    def _get_tsne():
        try:
            from sklearn.manifold import TSNE  # type: ignore

            return TSNE
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "t-SNE requested but 'scikit-learn' is not installed. "
                "Install it with: pip install scikit-learn"
            ) from e

    @staticmethod
    def compute_tsne3_for_layer(
        result_a: RunResult,
        result_b: RunResult,
        layer_idx: int,
        start_a: int,
        start_b: int,
        window_len: int,
        config: Optional[Config] = None,
    ) -> Dict[str, Any]:
        """
        Compute 3D t-SNE for a single layer.

        Args:
            result_a: First run result
            result_b: Second run result
            layer_idx: Layer index
            start_a: Start position in sequence A
            start_b: Start position in sequence B
            window_len: Length of aligned window
            config: Optional configuration object

        Returns:
            Dictionary with t-SNE results including:
            - trajectory_a: [window_len, 3] array
            - trajectory_b: [window_len, 3] array
            - tsne_params: Dictionary of parameters used
        """
        TSNE = TSNEAnalyzer._get_tsne()

        # Extract aligned windows
        hs_a = result_a.hidden_states[layer_idx][0, start_a : start_a + window_len, :]
        hs_b = result_b.hidden_states[layer_idx][0, start_b : start_b + window_len, :]

        # Convert to numpy
        hs_a_np = hs_a.detach().cpu().numpy()
        hs_b_np = hs_b.detach().cpu().numpy()

        # Fit t-SNE on concatenated data for a shared embedding space
        X = np.vstack([hs_a_np, hs_b_np])

        # Get parameters from config or use defaults
        perplexity = config.tsne_perplexity if config else 30.0
        learning_rate = config.tsne_learning_rate if config else "auto"
        n_iter = config.tsne_n_iter if config else 1000
        metric = config.tsne_metric if config else "euclidean"
        random_state = config.tsne_random_state if config else 0

        # Adjust perplexity if needed (must be less than n_samples)
        n_samples = X.shape[0]
        if perplexity >= n_samples:
            perplexity = max(1, n_samples - 1)

        reducer = TSNE(
            n_components=3,
            perplexity=perplexity,
            learning_rate=learning_rate,
            n_iter=n_iter,
            metric=metric,
            random_state=random_state,
        )
        embedding = reducer.fit_transform(X)

        # Split back into trajectories
        trajectory_a = embedding[:window_len]
        trajectory_b = embedding[window_len:]

        return {
            "trajectory_a": trajectory_a.tolist(),
            "trajectory_b": trajectory_b.tolist(),
            "tsne_params": {
                "perplexity": float(perplexity),
                "learning_rate": learning_rate if isinstance(learning_rate, str) else float(learning_rate),
                "n_iter": n_iter,
                "metric": metric,
                "random_state": random_state,
            },
        }

    @staticmethod
    def compute_tsne3_for_single_run(
        result: RunResult,
        layer_idx: int,
        start: int,
        window_len: int,
        config: Optional[Config] = None,
    ) -> Dict[str, Any]:
        """
        Compute 3D t-SNE for a single run (one trajectory).

        Returns:
            Dictionary with "trajectory", "tsne_params".
        """
        TSNE = TSNEAnalyzer._get_tsne()
        hs = result.hidden_states[layer_idx][0, start : start + window_len, :]
        X = hs.detach().cpu().numpy()
        perplexity = config.tsne_perplexity if config else 30.0
        n_samples = X.shape[0]
        if perplexity >= n_samples:
            perplexity = max(1, n_samples - 1)
        learning_rate = config.tsne_learning_rate if config else "auto"
        n_iter = config.tsne_n_iter if config else 1000
        metric = config.tsne_metric if config else "euclidean"
        random_state = config.tsne_random_state if config else 0
        reducer = TSNE(
            n_components=3,
            perplexity=perplexity,
            learning_rate=learning_rate,
            n_iter=n_iter,
            metric=metric,
            random_state=random_state,
        )
        trajectory = reducer.fit_transform(X)
        return {
            "trajectory": trajectory.tolist(),
            "tsne_params": {
                "perplexity": float(perplexity),
                "learning_rate": learning_rate if isinstance(learning_rate, str) else float(learning_rate),
                "n_iter": n_iter,
                "metric": metric,
                "random_state": random_state,
            },
        }
