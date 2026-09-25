"""Cheap diagnostics on captured residuals.

Stable rank of the benign-centred refusal residuals, after "Refusal geometry
reflects refusal training" (arXiv 2608.25390): ``sr(M) = ||M||_F^2 / ||M||_2^2``
on ``dH = H_harmful - mean(H_benign)`` at a refusal-mediating layer. The paper
finds that higher stable rank corresponds to weaker single-vector ablation,
and that models trained with diverse refusal openers have higher stable rank.
So the number is a prediction, from the captures the direction stage already
makes, of whether one difference-in-means vector will be enough.

The bands below are heuristic, drawn from the range the paper reports (11 to
33 on a 1B model with N=60), and are labelled as such in the payload.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

RESISTANCE_BANDS = (
    (0.0, 8.0, "low", "a single direction should carry most of the refusal"),
    (8.0, 20.0, "moderate", "expect a rank-2 to rank-4 subspace to be needed"),
    (20.0, math.inf, "high", "refusal is spread out; single-vector ablation is likely to under-perform"),
)


def stable_rank(M) -> float:
    """``||M||_F^2 / ||M||_2^2``: the effective number of comparable singular values."""
    import torch

    M64 = M.double()
    if M64.dim() == 1:
        M64 = M64.unsqueeze(0)
    fro = (M64 * M64).sum()
    if fro.item() <= 0:
        return 0.0
    spec = torch.linalg.matrix_norm(M64, ord=2)
    return float((fro / (spec * spec).clamp_min(1e-300)).item())


def benign_centred_residuals(harmful, harmless):
    """``dH = H_harmful - mean(H_benign)`` for one layer's ``[n, d]`` captures."""
    return harmful.double() - harmless.double().mean(dim=0, keepdim=True)


def refusal_stable_ranks(harmful_acts, harmless_acts, layers=None) -> Dict[int, float]:
    """Stable rank of ``dH`` per layer over ``[n_layers+1, n, d]`` captures."""
    n_layers = harmful_acts.shape[0]
    chosen = list(layers) if layers is not None else list(range(n_layers))
    return {
        int(L): stable_rank(benign_centred_residuals(harmful_acts[L], harmless_acts[L]))
        for L in chosen
        if 0 <= L < n_layers
    }


def resistance_band(sr: Optional[float]) -> Optional[str]:
    if sr is None or sr != sr:
        return None
    for lo, hi, name, _ in RESISTANCE_BANDS:
        if lo <= sr < hi:
            return name
    return None


def stable_rank_summary(sr_by_layer: Dict[int, float], layer: Optional[int]) -> dict:
    """The number at the chosen layer, its band, and the range across layers."""
    clean = {int(k): float(v) for k, v in (sr_by_layer or {}).items() if v == v}
    at_layer = clean.get(int(layer)) if layer is not None else None
    band = resistance_band(at_layer)
    note = next((n for lo, hi, name, n in RESISTANCE_BANDS if name == band), None)
    return {
        "at_layer": at_layer,
        "band": band,
        "note": note,
        "min": min(clean.values()) if clean else None,
        "max": max(clean.values()) if clean else None,
        "argmax_layer": max(clean, key=clean.get) if clean else None,
        "heuristic_bands": [
            {"min": lo, "max": (None if hi == math.inf else hi), "band": name}
            for lo, hi, name, _ in RESISTANCE_BANDS
        ],
        "source": "arXiv 2608.25390; bands are heuristic",
    }
