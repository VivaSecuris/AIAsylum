"""Residual stream decomposition for logit attribution and component analysis.

Uses captured attn_outputs and mlp_activations to decompose the residual stream
at a position into embed + sum over layers of (attn_out_l, mlp_out_l). When
capture was not enabled, components are omitted and logit attribution will be partial.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

from vivasecuris.aiasylum.interp.data.models import RunResult


def decompose_residual_at_position(
    result: RunResult,
    position: int,
) -> Dict[str, torch.Tensor]:
    """Decompose the final residual at a token position into components.

    Returns a dict mapping component name to vector [d]:
    - "embed": embedding at this position
    - "attn_0", "mlp_0", "attn_1", "mlp_1", ... for each layer

    If attn_outputs or mlp_activations are missing, those components are omitted.
    Caller can verify by summing values and comparing to result.hidden_states[-1][0, position].
    """
    out: Dict[str, torch.Tensor] = {}
    num_layers_plus_1 = len(result.hidden_states)
    if num_layers_plus_1 == 0:
        return out

    # Embedding
    embed = result.hidden_states[0][0, position, :].detach().cpu().float()
    out["embed"] = embed

    num_layers = num_layers_plus_1 - 1
    if num_layers <= 0:
        return out

    # Per-layer attn and MLP outputs
    attn_outputs = result.attn_outputs
    mlp_activations = result.mlp_activations

    for l in range(num_layers):
        if attn_outputs is not None and l < len(attn_outputs):
            t = attn_outputs[l][0, position, :].detach().cpu().float()
            out[f"attn_{l}"] = t
        if mlp_activations is not None and l in mlp_activations:
            t = mlp_activations[l][0, position, :].detach().cpu().float()
            out[f"mlp_{l}"] = t

    return out


def get_component_at_position(
    result: RunResult,
    layer_idx: int,
    position: int,
) -> Dict[str, torch.Tensor]:
    """Get residual components up to and including the given layer at a position.

    Returns dict with "embed", "attn_0", "mlp_0", ... "attn_{layer_idx}", "mlp_{layer_idx}".
    The residual at layer_idx (input to that layer) equals embed + attn_0 + mlp_0 + ...
    + attn_{layer_idx-1} + mlp_{layer_idx-1}. This dict also includes attn_{layer_idx}
    and mlp_{layer_idx} so the residual at layer_idx+1 = sum of all returned vectors.
    """
    out: Dict[str, torch.Tensor] = {}
    num_layers_plus_1 = len(result.hidden_states)
    if num_layers_plus_1 == 0 or layer_idx < 0:
        return out

    embed = result.hidden_states[0][0, position, :].detach().cpu().float()
    out["embed"] = embed

    attn_outputs = result.attn_outputs
    mlp_activations = result.mlp_activations

    for l in range(layer_idx + 1):
        if attn_outputs is not None and l < len(attn_outputs):
            t = attn_outputs[l][0, position, :].detach().cpu().float()
            out[f"attn_{l}"] = t
        if mlp_activations is not None and l in mlp_activations:
            t = mlp_activations[l][0, position, :].detach().cpu().float()
            out[f"mlp_{l}"] = t

    return out


def residual_sum_at_position(
    result: RunResult,
    position: int,
) -> Optional[torch.Tensor]:
    """Sum the decomposed residual components at a position.

    Returns a tensor equal to result.hidden_states[-1][0, position, :] when all
    components are captured; otherwise returns None if decomposition is partial.
    """
    comps = decompose_residual_at_position(result, position)
    if not comps:
        return None
    num_layers_plus_1 = len(result.hidden_states)
    expected_keys = {"embed"}
    num_layers = num_layers_plus_1 - 1
    for l in range(num_layers):
        expected_keys.add(f"attn_{l}")
        expected_keys.add(f"mlp_{l}")
    if comps.keys() != expected_keys:
        return None
    total = sum(comps.values())
    return total


class ResidualDecomposer:
    """Convenience wrapper for residual decomposition."""

    @staticmethod
    def decompose(
        result: RunResult,
        position: int,
    ) -> Dict[str, torch.Tensor]:
        """Full decomposition at position (same as decompose_residual_at_position)."""
        return decompose_residual_at_position(result, position)

    @staticmethod
    def get_components_up_to_layer(
        result: RunResult,
        layer_idx: int,
        position: int,
    ) -> Dict[str, torch.Tensor]:
        """Components up to and including layer_idx (same as get_component_at_position)."""
        return get_component_at_position(result, layer_idx, position)

    @staticmethod
    def check_decomposition_matches(
        result: RunResult,
        position: int,
        atol: float = 1e-4,
        rtol: float = 1e-3,
    ) -> Tuple[bool, Optional[str]]:
        """Verify that summed components match final hidden state at position.

        Returns (True, None) if they match, (False, error_message) otherwise.
        """
        total = residual_sum_at_position(result, position)
        if total is None:
            return False, "Decomposition incomplete (missing attn_outputs or mlp_activations)"
        final = result.hidden_states[-1][0, position, :].detach().cpu().float()
        if total.shape != final.shape:
            return False, f"Shape mismatch: {total.shape} vs {final.shape}"
        if not torch.allclose(total, final, atol=atol, rtol=rtol):
            diff = (total - final).abs().max().item()
            return False, f"Max absolute difference: {diff}"
        return True, None
