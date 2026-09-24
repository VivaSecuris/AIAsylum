"""Query region helpers for single-prompt and progression analysis."""

from typing import Tuple, Optional

from vivasecuris.aiasylum.interp.data.models import RunResult


def find_query_region(
    tokenizer,
    prompt: str,
    query_marker: Optional[str] = None,
) -> Tuple[int, int]:
    """
    Find the query region in a prompt.

    Args:
        tokenizer: Tokenizer to use
        prompt: Full prompt text
        query_marker: Optional marker string (e.g., "<<<QUERY>>>") to mark query start

    Returns:
        (start_pos, length) of query region
    """
    tokens = tokenizer.encode(prompt, add_special_tokens=False)

    if query_marker:
        # Find marker in tokens
        marker_tokens = tokenizer.encode(query_marker, add_special_tokens=False)

        # Find marker position
        prompt_lower = prompt.lower()
        marker_pos = prompt_lower.find(query_marker.lower())

        if marker_pos >= 0:
            # Find token position
            prompt_before_marker = prompt[:marker_pos]
            tokens_before = tokenizer.encode(prompt_before_marker, add_special_tokens=False)
            start_pos = len(tokens_before) + len(marker_tokens)

            # Query is everything after marker
            query_text = prompt[marker_pos + len(query_marker) :].strip()
            query_tokens = tokenizer.encode(query_text, add_special_tokens=False)
            length = len(query_tokens)

            return start_pos, length

    # No marker: assume query is the last part (last 50% or last 100 tokens,
    # whichever is smaller). Floored at one token: integer division sends a
    # single-token prompt to zero, which yields an empty analysis window and
    # fails much later, inside the dimensionality reduction.
    total_len = len(tokens)
    query_len = max(1, min(total_len // 2, 100)) if total_len else 0
    start_pos = max(0, total_len - query_len)

    return start_pos, query_len


def align_to_query_region(
    tokenizer,
    result: RunResult,
    query_start: int,
    query_len: int,
    window: int = 256,
) -> Tuple[int, int]:
    """
    Align a result to focus on the query region.

    Args:
        tokenizer: Tokenizer
        result: RunResult
        query_start: Start position of query region
        query_len: Length of query region
        window: Maximum window size

    Returns:
        (aligned_start, aligned_len)
    """
    total_len = len(result.token_strs)

    # Try to center on query region, but respect window size
    aligned_len = min(query_len, window)

    # Center the window on the query region
    query_center = query_start + query_len // 2
    aligned_start = max(0, query_center - aligned_len // 2)

    # Adjust if we go past the end
    if aligned_start + aligned_len > total_len:
        aligned_start = max(0, total_len - aligned_len)

    return aligned_start, aligned_len
