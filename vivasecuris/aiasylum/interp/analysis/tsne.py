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
    def fit_embedding(X: np.ndarray, config: Optional[Config] = None):
        """Fit one shared embedding; never substitute a different algorithm."""
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[0] < 2:
            raise ValueError("t-SNE needs at least two token observations; increase the analysis window or use PCA")
        params = {
            "perplexity": min(float(config.tsne_perplexity if config else 30), X.shape[0] - 1),
            "learning_rate": config.tsne_learning_rate if config else "auto",
            "max_iter": config.tsne_n_iter if config else 1000,
            "metric": config.tsne_metric if config else "euclidean",
            "random_state": config.tsne_random_state if config else 0,
            # PCA initialization fails when a short window has fewer than 3 samples.
            "init": "random",
        }
        reducer = TSNEAnalyzer._get_tsne()(n_components=3, **params)
        embedding = reducer.fit_transform(X)
        if not np.isfinite(embedding).all():
            raise ValueError("t-SNE produced non-finite coordinates")
        return embedding, params

    @staticmethod
    def compute_tsne3_for_layer(
        result_a: RunResult, result_b: RunResult, layer_idx: int,
        start_a: int, start_b: int, window_len: int, config: Optional[Config] = None,
    ) -> Dict[str, Any]:
        a = result_a.hidden_states[layer_idx][0, start_a:start_a + window_len].detach().float().cpu().numpy()
        b = result_b.hidden_states[layer_idx][0, start_b:start_b + window_len].detach().float().cpu().numpy()
        embedding, params = TSNEAnalyzer.fit_embedding(np.vstack([a, b]), config)
        return {"trajectory_a": embedding[:len(a)].tolist(),
                "trajectory_b": embedding[len(a):].tolist(), "tsne_params": params}

    @staticmethod
    def compute_tsne3_for_single_run(
        result: RunResult, layer_idx: int, start: int, window_len: int,
        config: Optional[Config] = None,
    ) -> Dict[str, Any]:
        X = result.hidden_states[layer_idx][0, start:start + window_len].detach().float().cpu().numpy()
        embedding, params = TSNEAnalyzer.fit_embedding(X, config)
        return {"trajectory": embedding.tolist(), "tsne_params": params}
