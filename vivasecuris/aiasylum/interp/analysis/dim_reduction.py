"""Unified interface for dimensionality reduction methods."""

from __future__ import annotations

from typing import Any, Dict, Optional

from vivasecuris.aiasylum.interp.analysis.pca import PCAAnalyzer, compute_pca3_single
from vivasecuris.aiasylum.interp.analysis.umap import UMAPAnalyzer
from vivasecuris.aiasylum.interp.analysis.tsne import TSNEAnalyzer
from vivasecuris.aiasylum.interp.core.config import Config
from vivasecuris.aiasylum.interp.data.models import RunResult


class DimensionReduction:
    """Unified interface for all dimensionality reduction methods."""

    @staticmethod
    def compute_for_layer(
        result_a: RunResult,
        result_b: RunResult,
        layer_idx: int,
        start_a: int,
        start_b: int,
        window_len: int,
        method: str,
        config: Optional[Config] = None,
    ) -> Dict[str, Any]:
        """
        Compute dimensionality reduction for a single layer using the specified method.

        Args:
            result_a: First run result
            result_b: Second run result
            layer_idx: Layer index
            start_a: Start position in sequence A
            start_b: Start position in sequence B
            window_len: Length of aligned window
            method: Method to use ("pca", "umap", or "tsne")
            config: Optional configuration object

        Returns:
            Dictionary with reduction results, including method-specific fields:
            - trajectory_a: [window_len, 3] array
            - trajectory_b: [window_len, 3] array
            - explained_variance: [3] array (PCA only) or None
            - method_params: Dictionary of method-specific parameters
        """
        method = method.lower()

        if method == "pca":
            result = PCAAnalyzer.compute_pca3_for_layer(
                result_a, result_b, layer_idx, start_a, start_b, window_len
            )
            # Standardize output format
            return {
                "trajectory_a": result["trajectory_a"],
                "trajectory_b": result["trajectory_b"],
                "explained_variance": result.get("explained_variance_ratio"),
                "pca_params": {
                    "n_components": 3,
                    "random_state": 0,
                },
            }

        elif method == "umap":
            result = UMAPAnalyzer.compute_umap3_for_layer(
                result_a, result_b, layer_idx, start_a, start_b, window_len, config
            )
            return {
                "trajectory_a": result["trajectory_a"],
                "trajectory_b": result["trajectory_b"],
                "explained_variance": None,
                "umap_params": result.get("umap_params", {}),
            }

        elif method == "tsne":
            result = TSNEAnalyzer.compute_tsne3_for_layer(
                result_a, result_b, layer_idx, start_a, start_b, window_len, config
            )
            return {
                "trajectory_a": result["trajectory_a"],
                "trajectory_b": result["trajectory_b"],
                "explained_variance": None,
                "tsne_params": result.get("tsne_params", {}),
            }

        else:
            raise ValueError(
                f"Unknown dimensionality reduction method: {method}. "
                "Must be one of 'pca', 'umap', or 'tsne'."
            )

    @staticmethod
    def compute_single_for_layer(
        result: RunResult,
        layer_idx: int,
        start: int,
        window_len: int,
        method: str,
        config: Optional[Config] = None,
    ) -> Dict[str, Any]:
        """
        Compute dimensionality reduction for a single run (one trajectory).

        Returns:
            Dictionary with "trajectory", "explained_variance" (PCA only), and method_params.
        """
        method = method.lower()
        if method == "pca":
            out = compute_pca3_single(
                result, layer_idx, start, window_len,
                random_state=config.pca_random_state if config else 0,
            )
            return {
                "trajectory": out["trajectory"],
                "explained_variance": out.get("explained_variance_ratio"),
                "pca_params": {"n_components": 3, "random_state": 0},
            }
        elif method == "umap":
            out = UMAPAnalyzer.compute_umap3_for_single_run(
                result, layer_idx, start, window_len, config
            )
            return {
                "trajectory": out["trajectory"],
                "explained_variance": None,
                "umap_params": out.get("umap_params", {}),
            }
        elif method == "tsne":
            out = TSNEAnalyzer.compute_tsne3_for_single_run(
                result, layer_idx, start, window_len, config
            )
            return {
                "trajectory": out["trajectory"],
                "explained_variance": None,
                "tsne_params": out.get("tsne_params", {}),
            }
        else:
            raise ValueError(
                f"Unknown dimensionality reduction method: {method}. "
                "Must be one of 'pca', 'umap', or 'tsne'."
            )

    @staticmethod
    def get_method_display_name(method: str) -> str:
        """Get display name for a method."""
        method = method.lower()
        if method == "pca":
            return "PCA"
        elif method == "umap":
            return "UMAP"
        elif method == "tsne":
            return "t-SNE"
        else:
            return method.upper()

    @staticmethod
    def get_axis_labels(method: str) -> tuple[str, str, str]:
        """Get axis labels for a method (x, y, z)."""
        method = method.lower()
        if method == "pca":
            return ("PC1", "PC2", "PC3")
        else:
            return ("Dim1", "Dim2", "Dim3")
