"""Activation magnitude (L2 norm) computation for single-run analysis."""

import numpy as np
import torch

from vivasecuris.aiasylum.interp.data.models import RunResult


def compute_activation_norms(
    result: RunResult,
    start: int,
    window_len: int,
) -> np.ndarray:
    """
    Compute per-layer, per-token L2 norm of hidden state for a single run.

    Args:
        result: Single run result
        start: Start position in sequence
        window_len: Length of window

    Returns:
        Array of shape [num_layers, window_len] with L2 norm per (layer, token).
    """
    num_layers = len(result.hidden_states)
    norms = np.zeros((num_layers, window_len), dtype=np.float32)
    for layer_idx in range(num_layers):
        hs = result.hidden_states[layer_idx][0, start : start + window_len, :]
        # L2 norm along last dimension (hidden_dim)
        norms[layer_idx, :] = torch.linalg.norm(hs, dim=-1).cpu().numpy()
    return norms
