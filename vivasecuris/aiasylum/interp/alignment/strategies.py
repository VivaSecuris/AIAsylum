"""Alignment strategy implementations."""

from abc import ABC, abstractmethod
from typing import Tuple
from transformers import PreTrainedTokenizer

from vivasecuris.aiasylum.interp.data.models import RunResult


class AlignmentStrategy(ABC):
    """Base class for alignment strategies."""

    @abstractmethod
    def align(
        self,
        tokenizer: PreTrainedTokenizer,
        result_a: RunResult,
        result_b: RunResult,
        window: int = 0,
    ) -> Tuple[int, int, int]:
        """
        Align two token sequences.
        
        Args:
            tokenizer: Tokenizer instance
            result_a: First run result
            result_b: Second run result
            window: Maximum window size (0 = unlimited)
            
        Returns:
            Tuple of (start_a, start_b, length) for aligned region
        """
        pass


class PrefixAlignmentStrategy(AlignmentStrategy):
    """Align from shared token prefix."""

    def align(
        self,
        tokenizer: PreTrainedTokenizer,
        result_a: RunResult,
        result_b: RunResult,
        window: int = 0,
    ) -> Tuple[int, int, int]:
        """Find longest shared prefix and align from there."""
        tokens_a = result_a.input_ids[0].tolist()
        tokens_b = result_b.input_ids[0].tolist()
        
        # Find longest shared prefix
        min_len = min(len(tokens_a), len(tokens_b))
        prefix_len = 0
        for i in range(min_len):
            if tokens_a[i] == tokens_b[i]:
                prefix_len += 1
            else:
                break
        
        # Start from prefix end
        start_a = prefix_len
        start_b = prefix_len
        
        # Determine window length
        remaining_a = len(tokens_a) - start_a
        remaining_b = len(tokens_b) - start_b
        length = min(remaining_a, remaining_b)
        
        if window > 0:
            length = min(length, window)
        
        return (start_a, start_b, length)


class MarkerAlignmentStrategy(AlignmentStrategy):
    """Align from a marker token."""

    def __init__(self, marker: str = "<<<QUERY>>>"):
        """
        Initialize marker alignment strategy.
        
        Args:
            marker: Marker string to search for
        """
        self.marker = marker

    def align(
        self,
        tokenizer: PreTrainedTokenizer,
        result_a: RunResult,
        result_b: RunResult,
        window: int = 0,
    ) -> Tuple[int, int, int]:
        """Find marker in both sequences and align from there."""
        # Tokenize marker
        marker_ids = tokenizer.encode(self.marker, add_special_tokens=False)
        
        tokens_a = result_a.input_ids[0].tolist()
        tokens_b = result_b.input_ids[0].tolist()
        
        # Find marker in sequence A
        start_a = self._find_marker(tokens_a, marker_ids)
        if start_a is None:
            # Fallback to prefix alignment
            strategy = PrefixAlignmentStrategy()
            return strategy.align(tokenizer, result_a, result_b, window)
        
        # Find marker in sequence B
        start_b = self._find_marker(tokens_b, marker_ids)
        if start_b is None:
            # Fallback to prefix alignment
            strategy = PrefixAlignmentStrategy()
            return strategy.align(tokenizer, result_a, result_b, window)
        
        # Start after marker
        start_a += len(marker_ids)
        start_b += len(marker_ids)
        
        # Determine window length
        remaining_a = len(tokens_a) - start_a
        remaining_b = len(tokens_b) - start_b
        length = min(remaining_a, remaining_b)
        
        if window > 0:
            length = min(length, window)
        
        return (start_a, start_b, length)

    def _find_marker(self, tokens: list, marker_ids: list) -> int:
        """Find marker token sequence in token list."""
        if len(marker_ids) == 0:
            return 0
        
        for i in range(len(tokens) - len(marker_ids) + 1):
            if tokens[i : i + len(marker_ids)] == marker_ids:
                return i
        return None


class LastTokenAlignmentStrategy(AlignmentStrategy):
    """Compare only final token positions."""

    def align(
        self,
        tokenizer: PreTrainedTokenizer,
        result_a: RunResult,
        result_b: RunResult,
        window: int = 0,
    ) -> Tuple[int, int, int]:
        """Align only at the last token position."""
        len_a = result_a.input_ids.shape[1]
        len_b = result_b.input_ids.shape[1]
        
        start_a = len_a - 1
        start_b = len_b - 1
        length = 1
        
        return (start_a, start_b, length)


def get_alignment_strategy(align_type: str, marker: str = "<<<QUERY>>>") -> AlignmentStrategy:
    """
    Get alignment strategy by name.
    
    Args:
        align_type: Strategy type ('prefix', 'marker', 'last')
        marker: Marker string (for marker strategy)
        
    Returns:
        AlignmentStrategy instance
    """
    if align_type == "prefix":
        return PrefixAlignmentStrategy()
    elif align_type == "marker":
        return MarkerAlignmentStrategy(marker)
    elif align_type == "last":
        return LastTokenAlignmentStrategy()
    else:
        raise ValueError(f"Unknown alignment strategy: {align_type}")
