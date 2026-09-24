"""Null baselines for causal claims.

Nothing in this engine may be labelled ``claim: causal`` without a number that
says what the same measurement gives when the intervention is meaningless.
Random dictionaries match trained sparse autoencoders on the standard metrics
(arXiv 2602.14111); a random direction of the right norm moves logits too;
a probe fit on shuffled labels still reports an AUC. These helpers produce
those controls so every payload can carry one.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple


def random_unit_directions(d: int, n: int = 1, seed: int = 0):
    """``[n, d]`` rows of unit norm, drawn isotropically."""
    import torch

    gen = torch.Generator().manual_seed(int(seed))
    x = torch.randn(n, d, generator=gen)
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)


def random_dictionary(d: int, n_features: int, seed: int = 0):
    """Random decoder ``[n_features, d]`` with unit rows: the SAE null."""
    return random_unit_directions(d, n_features, seed)


def shuffled_label_auc(pos, neg, n_perm: int = 20, seed: int = 0) -> dict:
    """AUC distribution of the projection score under shuffled class labels.

    ``pos``/``neg`` are 1-D score tensors. Returns the mean and 95th percentile
    of the permutation AUCs; a real AUC that does not clear the 95th percentile
    is not separating anything.
    """
    import torch

    from vivasecuris.aiasylum.weights.direction import _auc

    scores = torch.cat([pos, neg]).float()
    n_pos = len(pos)
    gen = torch.Generator().manual_seed(int(seed))
    aucs: List[float] = []
    for _ in range(int(n_perm)):
        perm = torch.randperm(len(scores), generator=gen)
        s = scores[perm]
        # Orient so the AUC is >= 0.5, as a fitted direction would be.
        a = _auc(s[:n_pos], s[n_pos:])
        aucs.append(max(a, 1.0 - a))
    aucs_sorted = sorted(aucs)
    q95 = aucs_sorted[min(len(aucs_sorted) - 1, int(math.ceil(0.95 * len(aucs_sorted))) - 1)]
    return {
        "n_perm": int(n_perm),
        "mean": float(sum(aucs) / max(1, len(aucs))),
        "p95": float(q95),
    }


def principal_angles(basis_a, basis_b) -> List[float]:
    """Principal angles (radians) between two row-orthonormal bases.

    Used to say whether two subspaces are the same object: all angles near
    zero means they span the same space. Rows are normalised defensively and
    re-orthonormalised via QR.
    """
    import torch

    A = torch.linalg.qr(basis_a.double().t()).Q     # [d, ka]
    B = torch.linalg.qr(basis_b.double().t()).Q     # [d, kb]
    s = torch.linalg.svdvals(A.t() @ B).clamp(-1.0, 1.0)
    return [float(math.acos(v)) for v in s.tolist()]


def subspace_overlap(basis_a, basis_b) -> float:
    """Mean squared cosine of the principal angles, in ``[0, 1]``."""
    angles = principal_angles(basis_a, basis_b)
    if not angles:
        return 0.0
    return float(sum(math.cos(t) ** 2 for t in angles) / len(angles))
