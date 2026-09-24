"""Data models for storing comparison results."""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import torch

# Per-layer Q/K/V: dict keys "q", "k", "v"; tensors [1, seq, n_heads, head_dim]
QKVOutputsType = Optional[Dict[int, Dict[str, torch.Tensor]]]


@dataclass
class RunResult:
    """Result from a single model forward pass."""
    
    input_ids: torch.Tensor  # [1, seq_len]
    token_strs: List[str]  # Token strings
    hidden_states: Tuple[torch.Tensor, ...]  # (L+1) tensors, each [1, seq_len, d]
    logits: torch.Tensor  # [1, seq_len, vocab_size]
    
    # Optional fields for advanced features
    attention_weights: Optional[Tuple[torch.Tensor, ...]] = None
    mlp_activations: Optional[Dict[int, torch.Tensor]] = None
    # Per-layer attention sublayer output (before residual add), each [1, seq_len, d_model]
    attn_outputs: Optional[Tuple[torch.Tensor, ...]] = None
    # Optional: intermediate MLP activations (e.g. after gate*up before down_proj), layer_idx -> [1, seq_len, intermediate_size]
    pre_mlp_activations: Optional[Dict[int, torch.Tensor]] = None
    # Optional: per-layer Q,K,V for OV/QK analysis; layer_idx -> {"q": [1,seq,n_heads,head_dim], "k": ..., "v": ...}
    qkv_outputs: QKVOutputsType = None


@dataclass
class ComparisonResult:
    """Result from comparing two prompts."""
    
    # Metadata
    meta: Dict[str, Any]
    
    # Aligned tokens
    tokens_a: List[str]
    tokens_b: List[str]
    
    # Similarity matrices [L+1, L] where L is aligned window length
    cos_mat: Any  # np.ndarray
    dn_mat: Any  # np.ndarray
    
    # Alignment info
    start_a: int
    start_b: int
    window_len: int
    
    # Optional analysis results
    pca_payload: Optional[Dict[str, Any]] = None
    # When Config.dim_reduction == "umap" we still populate pca_payload for backwards
    # compatibility with existing dashboard code paths. (See meta["dim_reduction"].)
    top_divergent_labels: Optional[Dict[str, Any]] = None
    predictions_payload: Optional[Dict[str, Any]] = None
    patching_results: Optional[Dict[str, Any]] = None
    cot_analysis: Optional[Dict[str, Any]] = None
    
    # Phase 2: Component localization
    attention_payload: Optional[Dict[str, Any]] = None  # Per-layer attention analysis
    mlp_payload: Optional[Dict[str, Any]] = None  # Per-layer MLP analysis
    circuit_payload: Optional[Dict[str, Any]] = None  # Circuit cards for selected locations
    temporal_payload: Optional[Dict[str, Any]] = None  # Temporal localization results
    attribution_payload: Optional[Dict[str, Any]] = None  # Attribution analysis results
    logit_attribution_payload: Optional[Dict[str, Any]] = None  # Component-wise logit decomposition
    ov_qk_payload: Optional[Dict[str, Any]] = None  # OV/QK per-head analysis
    scrub_payload: Optional[Dict[str, Any]] = None  # Causal scrubbing experiments
    minimal_circuit_payload: Optional[Dict[str, Any]] = None  # Minimal sufficient circuit (greedy)


@dataclass
class SinglePromptResult:
    """Result from analyzing a single prompt (no comparison)."""

    # Metadata
    meta: Dict[str, Any]

    # Single run
    run_result: RunResult

    # Tokens in the analysis window (for dashboard display)
    tokens: List[str]

    # Window used for analysis (start index and length in token space)
    start: int = 0
    window_len: int = 0

    # Per-layer per-token L2 norm of hidden state [num_layers, window_len]
    activation_norm_mat: Any = None  # np.ndarray

    # Optional analysis results
    pca_payload: Optional[Dict[str, Any]] = None
    predictions_payload: Optional[Dict[str, Any]] = None
    attention_payload: Optional[Dict[str, Any]] = None
    mlp_payload: Optional[Dict[str, Any]] = None
    logit_attribution_payload: Optional[Dict[str, Any]] = None
    cot_analysis: Optional[Dict[str, Any]] = None
    ov_qk_payload: Optional[Dict[str, Any]] = None
    scrub_payload: Optional[Dict[str, Any]] = None
