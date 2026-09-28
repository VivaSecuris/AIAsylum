"""Multi-dimensional refusal subspaces via RFM-AGOP.

Implements the Recursive Feature Machine with the Average Gradient Outer
Product as adapted to refusal by "Fast Multi-dimensional Refusal Subspaces via
RFM-AGOP" (arXiv 2607.02396, ICML 2026 MI workshop).

Why not difference-in-means. One mean-difference vector is the first axis of
refusal, and on models up to a few billion parameters removing it is enough.
On Qwen3-8B the paper needs three directions before ablation passes 50 percent
compliance, and the SVD-of-layer-means construction in ``derive_subspace``
finds *other layers' mean differences*, not the additional axes within a
layer. RFM learns a kernel classifier ``f`` on last-token residuals and reads
the discriminative subspace off the outer products of its input gradients:
directions the classifier does not use get zero weight, whatever their
variance.

Algorithm, per layer::

    M_0  = beta * w w^T + (1 - beta) * Cov_k(X)          # probe-informed init
    for t in 1..T:
        K    = exp(-||x_i - x_j||_M / L)                  # Mahalanobis Laplace kernel
        a    = (K + lambda I)^-1 y                        # ridge solve, y in {+1, -1}
        g_j  = grad_x f(x_j) = -(1/L) M sum_i a_i K_ji (x_j - x_i) / D_ji
        M_hat = (1/n) sum_j g_j g_j^T                     # AGOP
        M    = (1 - gamma) M + gamma * M_hat              # EMA, then unit trace
    eigendecompose M; orient each v_i by corr(X v_i, y); cone = top-k v_i
    weights = mu_i / mu_1                                  # for soft ablation

Everything runs on CPU in float64 on a few hundred residual vectors; the
expensive pieces are an ``n x n`` solve and a ``d x d`` eigendecomposition per
iteration, seconds on a laptop for ``d`` up to a few thousand.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# Default number of best difference-in-means layers on which RFM is run; the
# one whose leading eigenvector separates best on held-out data is kept.
DEFAULT_CANDIDATE_LAYERS = 4

# Below this held-out AUC the difference-in-means probe carries no signal, and
# warm-starting the kernel metric from it (and from the raw covariance) only
# imports whatever direction happens to have the most variance. Identity is
# the safer start there.
MIN_PROBE_AUC_FOR_INIT = 0.90

# Stop iterating once the feature matrix has settled, or once it has collapsed
# onto a single direction (which full-replacement RFM does after enough steps).
CONVERGENCE_TOL = 1e-6
COLLAPSE_SHARE = 0.9999


@dataclass
class RFMResult:
    eigenvalues: object          # torch [d], descending
    eigenvectors: object         # torch [d, d], columns aligned with eigenvalues, oriented toward y=+1
    feature_matrix: object       # torch [d, d]
    center: object               # torch [d] training mean
    bandwidth: float
    iterations: int
    train_auc: float
    ridge: float
    gamma: float
    init: str = "identity"
    iterations_run: int = 0

    def top(self, k: int):
        """Top-``k`` eigenvectors as ``[k, d]`` rows plus ``mu_i / mu_1`` weights."""
        k = max(1, min(int(k), self.eigenvectors.shape[1]))
        basis = self.eigenvectors[:, :k].t().contiguous()
        top = self.eigenvalues[:k]
        scale = float(top[0].item()) if top[0].item() > 0 else 1.0
        return basis, [float(v / scale) for v in top.tolist()]


def _sqrt_psd(M):
    import torch

    evals, evecs = torch.linalg.eigh(M)
    evals = evals.clamp_min(0.0)
    return (evecs * evals.sqrt().unsqueeze(0)) @ evecs.t()


def probe_informed_init(X, w, beta: float = 0.5, k: int = 10):
    """``beta * w w^T + (1 - beta) * Cov_k`` with both terms at unit trace.

    ``w`` is the difference-in-means direction: starting near it lets RFM find
    what the linear probe already knows and then grow the higher-dimensional
    structure around it. ``Cov_k`` is the rank-``k`` truncation of the data
    covariance, so the initial kernel already stretches along the directions
    the residuals actually vary in.
    """
    import torch

    X64 = X.double()
    Xc = X64 - X64.mean(dim=0, keepdim=True)
    _, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
    k = max(1, min(int(k), Vh.shape[0]))
    var = (S[:k] ** 2) / max(1, Xc.shape[0] - 1)
    cov_k = Vh[:k].t() @ torch.diag(var) @ Vh[:k]
    cov_k = cov_k / cov_k.trace().clamp_min(1e-12)
    w64 = w.double().flatten()
    w64 = w64 / w64.norm().clamp_min(1e-12)
    return float(beta) * torch.outer(w64, w64) + (1.0 - float(beta)) * cov_k


def choose_init(X, w, probe_auc: Optional[float], beta: float = 0.5, k: int = 10):
    """Probe-informed start when the linear probe is informative, else identity.

    On refusal data the difference-in-means direction is strong (held-out AUC
    near 1) and the paper's warm start converges in a few iterations. When the
    probe is at chance the same start is actively harmful: the covariance term
    hands the metric to whichever nuisance direction has the most variance and
    RFM never recovers. Returns ``None`` for identity.
    """
    if probe_auc is not None and probe_auc == probe_auc and probe_auc >= MIN_PROBE_AUC_FOR_INIT:
        return probe_informed_init(X, w, beta=beta, k=k)
    return None


def rfm_agop(
    X,
    y,
    M0=None,
    iterations: int = 5,
    gamma: float = 0.5,
    ridge: float = 1e-3,
    bandwidth: Optional[float] = None,
) -> RFMResult:
    """Run RFM-AGOP on ``X`` (``[n, d]``) with labels ``y`` (``[n]`` in {+1, -1}).

    ``bandwidth`` defaults to the median pairwise Mahalanobis distance at each
    iteration, so it tracks the unit-trace normalisation of ``M``. Iteration
    stops early once ``M`` settles or collapses onto one direction.
    """
    import torch

    from vivasecuris.aiasylum.weights.direction import _auc

    X64 = X.double()
    y64 = y.double().flatten()
    if X64.shape[0] != y64.shape[0]:
        raise ValueError(f"X has {X64.shape[0]} rows but y has {y64.shape[0]} labels")
    n, d = X64.shape
    if n < 4:
        raise ValueError("RFM needs at least four samples")

    center = X64.mean(dim=0)
    Xc = X64 - center
    eye_n = torch.eye(n, dtype=torch.float64)

    M = M0.double().clone() if M0 is not None else torch.eye(d, dtype=torch.float64)
    M = M / M.trace().clamp_min(1e-12)
    init_name = "probe_informed" if M0 is not None else "identity"

    L_used = float(bandwidth) if bandwidth else 1.0
    ran = 0
    for t in range(int(iterations)):
        R = _sqrt_psd(M)
        Z = Xc @ R
        D = torch.cdist(Z, Z)                                  # Mahalanobis distances
        if bandwidth is None:
            off = D[~torch.eye(n, dtype=torch.bool)]
            L_used = float(off.median().item()) or 1.0
        K = torch.exp(-D / L_used)
        alpha = torch.linalg.solve(K + ridge * eye_n, y64)     # [n]

        # grad f(x_j) = -(1/L) M sum_i alpha_i K_ji (x_j - x_i) / D_ji
        Dsafe = D.clone()
        Dsafe[Dsafe < 1e-12] = float("inf")                    # kills the i == j term
        W = (K / Dsafe) * alpha.unsqueeze(0)                   # W[j, i]
        G = W.sum(dim=1, keepdim=True) * Xc - W @ Xc           # [n, d] = sum_i W_ji (x_j - x_i)
        G = -(1.0 / L_used) * (G @ M)                          # M symmetric
        M_hat = (G.t() @ G) / n
        if M_hat.trace().item() <= 0:
            logger.warning("RFM iteration %d: zero gradient outer product; stopping", t + 1)
            break
        M_new = (1.0 - gamma) * M + gamma * M_hat
        M_new = M_new / M_new.trace().clamp_min(1e-12)
        delta = float((M_new - M).norm().item())
        M = M_new
        ran = t + 1
        top_share = float(torch.linalg.eigvalsh(M)[-1].item())
        logger.debug("RFM iteration %d: bandwidth %.4f, top eig share %.3f, delta %.2e",
                     ran, L_used, top_share, delta)
        if delta < CONVERGENCE_TOL or top_share > COLLAPSE_SHARE:
            break

    evals, evecs = torch.linalg.eigh(M)
    order = torch.argsort(evals, descending=True)
    evals, evecs = evals[order].clamp_min(0.0), evecs[:, order]

    # Orient every eigenvector toward the positive class.
    proj = Xc @ evecs                                          # [n, d]
    yc = y64 - y64.mean()
    corr = (proj * yc.unsqueeze(1)).sum(dim=0)
    signs = torch.where(corr < 0, -1.0, 1.0)
    evecs = evecs * signs.unsqueeze(0)

    pos = (Xc @ evecs[:, 0])[y64 > 0]
    neg = (Xc @ evecs[:, 0])[y64 <= 0]
    train_auc = _auc(pos.float(), neg.float()) if len(pos) and len(neg) else float("nan")

    return RFMResult(
        eigenvalues=evals.float(),
        eigenvectors=evecs.float(),
        feature_matrix=M.float(),
        center=center.float(),
        bandwidth=L_used,
        iterations=int(iterations),
        train_auc=float(train_auc),
        ridge=float(ridge),
        gamma=float(gamma),
        init=init_name,
        iterations_run=ran,
    )


def derive_rfm_subspace(
    model,
    tokenizer,
    split,
    rank: int = 4,
    iterations: int = 5,
    beta: float = 0.5,
    gamma: float = 0.5,
    ridge: float = 1e-3,
    bandwidth: Optional[float] = None,
    candidate_layers: int = DEFAULT_CANDIDATE_LAYERS,
    model_id: str = "unknown",
    batch_size: int = 8,
    max_length: int = 512,
    layer_range: Optional[tuple] = None,
    progress: Optional[callable] = None,
    thinking: bool = False,
    allow_no_chat_template: bool = False,
):
    """Derive an RFM-AGOP refusal cone as a :class:`RefusalDirection`.

    The captures and per-layer difference-in-means scores come from the same
    seeded split as :func:`derive_direction`. RFM runs on the
    ``candidate_layers`` best-separating layers; the layer whose leading
    eigenvector scores the best held-out AUC wins. ``basis`` holds the top
    ``rank`` eigenvectors (row 0 is the leading one and is also ``vector``),
    ``weights`` the eigenvalue ratios ``mu_i / mu_1`` that the soft ablation
    in ``steering`` and ``surgery`` applies.
    """
    import torch

    from vivasecuris.aiasylum.weights.capture import has_chat_template
    from vivasecuris.aiasylum.weights.direction import (
        RefusalDirection,
        _auc,
        _cohens_d,
        _rank_layers,
        _score_layers,
    )

    if rank < 1:
        raise ValueError(f"rank must be >= 1, got {rank}")

    directions, scores, caps = _score_layers(
        model, tokenizer, split, batch_size=batch_size, max_length=max_length,
        layer_range=layer_range, progress=progress, return_captures=True,
        thinking=thinking, allow_no_chat_template=allow_no_chat_template,
    )
    ranked = _rank_layers(scores)
    candidates = [s.layer for s in ranked[: max(1, int(candidate_layers))] if s.layer in directions]

    harmful_tr, harmless_tr = caps["harmful_train"], caps["harmless_train"]
    harmful_te, harmless_te = caps["harmful_test"], caps["harmless_test"]

    best = None
    for layer in candidates:
        if progress:
            progress(f"RFM-AGOP on layer {layer} ({int(iterations)} iterations)")
        X = torch.cat([harmful_tr[layer], harmless_tr[layer]], dim=0)
        y = torch.cat([
            torch.ones(harmful_tr.shape[1]), -torch.ones(harmless_tr.shape[1]),
        ])
        dim_auc = next((s.auc for s in scores if s.layer == layer), None)
        M0 = choose_init(X, directions[layer], dim_auc, beta=beta)
        res = rfm_agop(X, y, M0=M0, iterations=iterations, gamma=gamma, ridge=ridge, bandwidth=bandwidth)
        v1 = res.eigenvectors[:, 0]
        pos = (harmful_te[layer] - res.center) @ v1
        neg = (harmless_te[layer] - res.center) @ v1
        auc, d_eff = _auc(pos, neg), _cohens_d(pos, neg)
        logger.info("RFM layer %d: held-out AUC %.3f, d %.2f, train AUC %.3f",
                    layer, auc, d_eff, res.train_auc)
        key = (round(auc, 6), d_eff if d_eff == d_eff else -math.inf)
        if best is None or key > best[0]:
            best = (key, layer, res, auc, d_eff)

    if best is None:
        raise ValueError("RFM produced no usable layer")
    _, layer, res, auc, d_eff = best

    basis, weights = res.top(rank)
    v1 = basis[0]
    proj_h = ((harmful_tr[layer] - res.center) @ v1).mean().item()
    proj_b = ((harmless_tr[layer] - res.center) @ v1).mean().item()
    sr_by_layer = {s.layer: s.stable_rank for s in scores if s.stable_rank == s.stable_rank}

    from vivasecuris.aiasylum.weights.diagnostics import stable_rank_summary

    direction = RefusalDirection(
        vector=v1.contiguous(),
        layer=layer,
        auc=float(auc),
        cohens_d=float(d_eff),
        model_id=model_id,
        split_hash=split.hash,
        layer_scores=sorted(scores, key=lambda s: s.layer),
        basis=basis.contiguous(),
        basis_layers=[layer] * int(basis.shape[0]),
        weights=weights,
        method="rfm_agop",
        template_applied=has_chat_template(tokenizer),
        extra={
            "thinking": bool(thinking),
            "rfm": {
                "iterations": int(iterations),
                "gamma": float(gamma),
                "ridge": float(ridge),
                "beta": float(beta),
                "bandwidth": res.bandwidth,
                "init": res.init,
                "iterations_run": res.iterations_run,
                "train_auc": res.train_auc,
                "candidate_layers": candidates,
                "eigenvalues": [float(v) for v in res.eigenvalues[: max(2 * rank, 8)].tolist()],
                "dim_auc_at_layer": next((s.auc for s in scores if s.layer == layer), None),
            },
            # Offset of the projection: RFM centres the data, so the timeline
            # and any threshold must subtract the same mean.
            "center_projection": float((res.center @ v1).item()),
            "projection_means": {"harmful": proj_h, "harmless": proj_b, "layer": layer, "centered": True},
            "stable_rank": stable_rank_summary(sr_by_layer, layer),
        },
    )
    logger.info(
        "Derived rank-%d RFM-AGOP refusal cone at layer %d (held-out AUC %.3f); weights %s",
        direction.rank, layer, auc, [round(w, 3) for w in weights],
    )
    return direction
