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
    def fit_embedding(X: np.ndarray, config: Optional[Config] = None):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[0] < 3:
            raise ValueError("UMAP needs at least three token observations; increase the analysis window or use PCA")
        params = {
            "n_neighbors": min(config.umap_n_neighbors if config else 15, X.shape[0] - 1),
            "min_dist": config.umap_min_dist if config else 0.1,
            "metric": config.umap_metric if config else "cosine",
            "random_state": config.umap_random_state if config else 0,
            # Spectral initialization requires more samples than dimensions + 1.
            "init": "random",
        }
        reducer = UMAPAnalyzer._get_umap().UMAP(n_components=3, **params)
        embedding = reducer.fit_transform(X)
        if not np.isfinite(embedding).all():
            raise ValueError("UMAP produced non-finite coordinates")
        return embedding, params

    @staticmethod
    def compute_umap3_for_layer(
        result_a: RunResult, result_b: RunResult, layer_idx: int,
        start_a: int, start_b: int, window_len: int, config: Optional[Config] = None,
    ) -> Dict[str, Any]:
        a = result_a.hidden_states[layer_idx][0, start_a:start_a + window_len].detach().float().cpu().numpy()
        b = result_b.hidden_states[layer_idx][0, start_b:start_b + window_len].detach().float().cpu().numpy()
        embedding, params = UMAPAnalyzer.fit_embedding(np.vstack([a, b]), config)
        return {"trajectory_a": embedding[:len(a)].tolist(),
                "trajectory_b": embedding[len(a):].tolist(), "umap_params": params}

    @staticmethod
    def compute_umap3_for_single_run(
        result: RunResult, layer_idx: int, start: int, window_len: int,
        config: Optional[Config] = None,
    ) -> Dict[str, Any]:
        X = result.hidden_states[layer_idx][0, start:start + window_len].detach().float().cpu().numpy()
        embedding, params = UMAPAnalyzer.fit_embedding(X, config)
        return {"trajectory": embedding.tolist(), "umap_params": params}
