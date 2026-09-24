"""Similarity computation between activations."""

import torch
import numpy as np
from typing import Tuple
import torch.nn.functional as F

from vivasecuris.aiasylum.interp.data.models import RunResult


def compute_cosine_similarity(
    result_a: RunResult,
    result_b: RunResult,
    start_a: int,
    start_b: int,
    window_len: int,
) -> np.ndarray:
    """
    Compute cosine similarity matrix between activations.
    
    Args:
        result_a: First run result
        result_b: Second run result
        start_a: Start position in sequence A
        start_b: Start position in sequence B
        window_len: Length of aligned window
        
    Returns:
        Cosine similarity matrix [L+1, window_len] where L+1 is number of layers
    """
    num_layers = len(result_a.hidden_states)
    cos_mat = np.zeros((num_layers, window_len))
    
    for layer_idx in range(num_layers):
        # Extract aligned window [window_len, hidden_dim]
        hs_a = result_a.hidden_states[layer_idx][0, start_a : start_a + window_len, :]
        hs_b = result_b.hidden_states[layer_idx][0, start_b : start_b + window_len, :]
        
        # Compute cosine similarity per token position
        cos_sim = F.cosine_similarity(hs_a, hs_b, dim=-1)
        cos_mat[layer_idx, :] = cos_sim.numpy()
    
    return cos_mat


def compute_delta_norm(
    result_a: RunResult,
    result_b: RunResult,
    start_a: int,
    start_b: int,
    window_len: int,
) -> np.ndarray:
    """
    Compute delta norm (Euclidean distance) matrix between activations.
    
    Args:
        result_a: First run result
        result_b: Second run result
        start_a: Start position in sequence A
        start_b: Start position in sequence B
        window_len: Length of aligned window
        
    Returns:
        Delta norm matrix [L+1, window_len] where L+1 is number of layers
    """
    num_layers = len(result_a.hidden_states)
    dn_mat = np.zeros((num_layers, window_len))
    
    for layer_idx in range(num_layers):
        # Extract aligned window [window_len, hidden_dim]
        hs_a = result_a.hidden_states[layer_idx][0, start_a : start_a + window_len, :]
        hs_b = result_b.hidden_states[layer_idx][0, start_b : start_b + window_len, :]
        
        # Compute delta and norm per token position
        delta = hs_b - hs_a  # [window_len, hidden_dim]
        norm = torch.linalg.vector_norm(delta, dim=-1)  # [window_len]
        dn_mat[layer_idx, :] = norm.numpy()
    
    return dn_mat


def find_spike_layer(dn_mat: np.ndarray) -> int:
    """
    Find the layer with maximum divergence at the last token.
    
    Args:
        dn_mat: Delta norm matrix [L+1, window_len]
        
    Returns:
        Layer index with maximum divergence
    """
    if dn_mat.shape[1] == 0:
        return 0
    
    # Use last token position
    last_token_delta = dn_mat[:, -1]
    spike_layer = int(np.argmax(last_token_delta))
    return spike_layer
