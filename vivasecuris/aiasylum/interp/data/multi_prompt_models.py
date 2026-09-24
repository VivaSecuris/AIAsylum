"""Data models for multi-prompt analysis (one-shot, multi-shot)."""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

from vivasecuris.aiasylum.interp.data.models import RunResult


@dataclass
class ProgressionResult:
    """Result from analyzing multiple prompts in progression (zero-shot → one-shot → multi-shot)."""
    
    # Metadata
    meta: Dict[str, Any]
    
    # All run results
    run_results: List[RunResult]  # One for each prompt
    
    # Prompt labels (e.g., ["zero-shot", "one-shot", "two-shot"])
    prompt_labels: List[str]
    
    # Aligned tokens for each prompt (after alignment to query region)
    aligned_tokens: List[List[str]]
    
    # Alignment info (start positions and window length for each prompt)
    alignment_info: List[Dict[str, int]]  # [{"start": int, "window_len": int}, ...]
    
    # Progression matrices: how activations change as we add examples
    # Shape: [num_prompts-1, num_layers+1, window_len]
    # progression_mat[i] = difference between prompt i+1 and prompt i
    progression_cos: np.ndarray  # Cosine similarity changes
    progression_delta: np.ndarray  # Delta norm changes
    
    # Cumulative progression: how each prompt differs from zero-shot baseline
    # Shape: [num_prompts, num_layers+1, window_len]
    cumulative_cos: np.ndarray  # Cosine similarity vs zero-shot
    cumulative_delta: np.ndarray  # Delta norm vs zero-shot
    
    # Query region info (where the actual query starts in each prompt)
    query_starts: List[int]  # Token position where query starts in each prompt
    query_window_len: int  # Length of query region to analyze
    
    # Optional analysis results
    pca_payload: Optional[Dict[str, Any]] = None
    predictions_payload: Optional[Dict[str, Any]] = None
    example_impact: Optional[Dict[str, Any]] = None  # Which layers/neurons are most affected by examples
