"""Derive a refusal direction by difference-in-means over the residual stream.

Following Arditi et al. 2024 ("Refusal in Language Models Is Mediated by a
Single Direction"): at each layer, take the mean last-token residual for
harmful prompts minus the same for harmless prompts, normalize, then pick the
single layer whose direction best separates the two classes. That one vector is
then applied globally -- to every residual-writing matrix -- rather than
per-layer.

The layer is chosen on the *held-out* split. Choosing it on the fitting set
picks whichever layer overfit hardest, which looks excellent and generalizes
not at all.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Below this, the direction is not meaningfully separating the classes and
# nothing downstream (steering, surgery) will produce a real behavior change.
MIN_USABLE_AUC = 0.90


@dataclass
class LayerScore:
    layer: int
    auc: float
    cohens_d: float
    train_norm: float
    # Stable rank of the benign-centred refusal residuals at this layer; a
    # prediction of how much a single vector can remove (weights/diagnostics.py).
    stable_rank: float = float("nan")


@dataclass
class RefusalDirection:
    """A unit direction plus the evidence that it separates the classes.

    A single-direction result carries ``vector`` alone. A subspace result
    (``derive_subspace``) additionally carries ``basis`` -- an orthonormal
    ``[m, d]`` set whose first row *is* ``vector`` -- so that removing the whole
    subspace catches refusal components a single difference-in-means vector
    misses. Everything that reads ``vector`` keeps working unchanged; only
    subspace-aware surgery looks at ``basis``.
    """

    vector: object              # torch.Tensor [d_model], float32, unit norm
    layer: int
    auc: float
    cohens_d: float
    model_id: str
    split_hash: str
    layer_scores: List[LayerScore] = field(default_factory=list)
    basis: object = None        # torch.Tensor [m, d_model], orthonormal rows; row 0 == vector
    basis_layers: List[int] = field(default_factory=list)  # source layer of each basis row (best-effort)
    # Per-row removal weights for soft ablation (mu_i / mu_1 for an RFM cone).
    # None means every row is removed at full strength, the pre-2026 behaviour.
    weights: Optional[List[float]] = None
    method: str = "diff_in_means"
    # Method-specific provenance and the class projection means the timeline
    # and any threshold use. Free-form JSON.
    extra: dict = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        return self.auc >= MIN_USABLE_AUC

    @property
    def rank(self) -> int:
        """Number of directions removed: the basis size, or 1 for a bare vector."""
        return int(self.basis.shape[0]) if self.basis is not None else 1

    def as_basis(self):
        """Return the orthonormal ``[m, d]`` edit basis, falling back to ``[vector]``."""
        if self.basis is not None:
            return self.basis
        return self.vector.reshape(1, -1)

    def as_weights(self, rank: Optional[int] = None) -> Optional[List[float]]:
        """Removal weights for the first ``rank`` rows, or ``None`` for uniform."""
        if not self.weights:
            return None
        w = [float(x) for x in self.weights]
        return w[: int(rank)] if rank is not None else w

    def metadata(self) -> dict:
        meta = {
            "layer": self.layer,
            "auc": round(self.auc, 4),
            "cohens_d": round(self.cohens_d, 4),
            "model_id": self.model_id,
            "split_hash": self.split_hash,
            "d_model": int(self.vector.shape[0]),
            "min_usable_auc": MIN_USABLE_AUC,
            "usable": self.usable,
            "rank": self.rank,
            "method": self.method,
            "layer_scores": [
                {
                    "layer": s.layer,
                    "auc": round(s.auc, 4),
                    "cohens_d": round(s.cohens_d, 4),
                    "stable_rank": (round(s.stable_rank, 3) if s.stable_rank == s.stable_rank else None),
                }
                for s in self.layer_scores
            ],
        }
        if self.basis is not None:
            meta["basis_layers"] = list(self.basis_layers)
        if self.weights:
            meta["weights"] = [round(float(w), 6) for w in self.weights]
        if self.extra:
            meta["extra"] = self.extra
        return meta

    def save(self, out_dir: str | Path) -> Path:
        """Write ``direction.safetensors`` + ``direction.json`` into ``out_dir``."""
        import torch
        from safetensors.torch import save_file

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        # `basis` row 0 *is* the primary direction, so the two can legitimately
        # be views on the same storage. safetensors refuses to write aliased
        # tensors, so clone when that is the case -- 4 KB of duplication is a
        # better trade than a save that fails on the directions most worth
        # keeping.
        tensors = {"direction": self.vector.detach().contiguous().to(torch.float32)}
        if self.basis is not None:
            basis = self.basis.detach().contiguous().to(torch.float32)
            if basis.data_ptr() == tensors["direction"].data_ptr():
                tensors["direction"] = tensors["direction"].clone()
            tensors["basis"] = basis
        if self.weights:
            tensors["weights"] = torch.tensor([float(w) for w in self.weights], dtype=torch.float32)
        save_file(tensors, str(out / "direction.safetensors"))
        (out / "direction.json").write_text(json.dumps(self.metadata(), indent=2))
        logger.info(
            "Saved direction to %s (layer %d, AUC %.3f, rank %d)",
            out, self.layer, self.auc, self.rank,
        )
        return out

    @classmethod
    def load(cls, path: str | Path) -> "RefusalDirection":
        from safetensors.torch import load_file

        p = Path(path)
        if p.is_dir():
            vec_path, meta_path = p / "direction.safetensors", p / "direction.json"
        else:
            vec_path, meta_path = p, p.with_suffix(".json")
        if not vec_path.exists():
            hint = ""
            if p.is_dir():
                hint = (
                    f"\n'{p}' exists but holds no direction.safetensors. "
                    f"Produce one first:\n"
                    f"    aiasylum weights direction --model <hf-id> --out {p}"
                )
            elif not p.exists():
                hint = (
                    f"\n'{p}' does not exist. Pass the --out directory from a "
                    f"previous `aiasylum weights direction` run."
                )
            raise FileNotFoundError(f"No direction file at {vec_path}.{hint}")

        tensors = load_file(str(vec_path))
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        weights = tensors.get("weights")
        layer_scores = [
            LayerScore(
                layer=int(s["layer"]), auc=float(s.get("auc", float("nan"))),
                cohens_d=float(s.get("cohens_d", float("nan"))), train_norm=float("nan"),
                stable_rank=float(s["stable_rank"]) if s.get("stable_rank") is not None else float("nan"),
            )
            for s in meta.get("layer_scores", []) or []
        ]
        return cls(
            vector=tensors["direction"],
            layer=meta.get("layer", -1),
            auc=meta.get("auc", float("nan")),
            cohens_d=meta.get("cohens_d", float("nan")),
            model_id=meta.get("model_id", "unknown"),
            split_hash=meta.get("split_hash", "unknown"),
            layer_scores=layer_scores,
            basis=tensors.get("basis"),
            basis_layers=meta.get("basis_layers", []),
            weights=([float(w) for w in weights.tolist()] if weights is not None else None),
            method=meta.get("method", "diff_in_means"),
            extra=dict(meta.get("extra") or {}),
        )


def _auc(pos, neg) -> float:
    """Rank-based AUC (equivalent to Mann-Whitney U), tie-corrected."""
    import torch

    combined = torch.cat([pos, neg])
    # average ranks so ties contribute 0.5 rather than an arbitrary order
    order = combined.argsort()
    ranks = torch.empty_like(combined, dtype=torch.float64)
    ranks[order] = torch.arange(1, len(combined) + 1, dtype=torch.float64)

    uniq, inverse, counts = combined.unique(return_inverse=True, return_counts=True)
    if (counts > 1).any():
        sums = torch.zeros(len(uniq), dtype=torch.float64).scatter_add_(0, inverse, ranks)
        ranks = (sums / counts.double())[inverse]

    n_pos, n_neg = len(pos), len(neg)
    rank_sum = ranks[:n_pos].sum().item()
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _cohens_d(pos, neg) -> float:
    n1, n2 = len(pos), len(neg)
    if n1 < 2 or n2 < 2:
        return float("nan")
    pooled = (((n1 - 1) * pos.var(unbiased=True) + (n2 - 1) * neg.var(unbiased=True)) / (n1 + n2 - 2)).sqrt()
    if pooled.item() == 0:
        return float("nan")
    return ((pos.mean() - neg.mean()) / pooled).item()


def _score_layers(
    model,
    tokenizer,
    split,
    batch_size: int = 8,
    max_length: int = 512,
    layer_range: Optional[tuple] = None,
    progress: Optional[callable] = None,
    return_captures: bool = False,
):
    """Capture residuals and compute the per-layer difference-in-means direction.

    Returns ``(directions, scores)`` where ``directions[layer]`` is the unit
    diff-in-means vector fit on the train half and ``scores`` are held-out
    separation scores. Shared by ``derive_direction`` and ``derive_subspace`` so
    both fit and select on exactly the same seeded split. With
    ``return_captures=True`` a third element holds the raw ``[n_layers+1, n, d]``
    captures under ``harmful_train``, ``harmless_train``, ``harmful_test`` and
    ``harmless_test`` so RFM and the diagnostics never re-run the model.

    Each ``LayerScore`` also carries the stable rank of the benign-centred
    refusal residuals on the train half (see ``weights/diagnostics.py``).
    """
    from vivasecuris.aiasylum.weights.capture import capture_last_token_residuals

    def capture(prompts, label):
        if progress:
            progress(f"capturing {label} ({len(prompts)} prompts)")
        return capture_last_token_residuals(
            model, tokenizer, prompts, batch_size=batch_size, max_length=max_length,
            progress=(lambda d, t: progress(None, d, t)) if progress else None,
        )

    # [n_layers+1, n_prompts, d_model]
    harmful_tr = capture(split.harmful_train, "harmful/train")
    harmless_tr = capture(split.harmless_train, "harmless/train")
    harmful_te = capture(split.harmful_test, "harmful/test")
    harmless_te = capture(split.harmless_test, "harmless/test")

    n_layers = harmful_tr.shape[0]
    lo, hi = layer_range or (1, n_layers)  # layer 0 is the raw embedding: no computation yet
    lo, hi = max(1, lo), min(n_layers, hi)

    if progress:
        progress(f"scoring {hi - lo} layers on held-out prompts")

    from vivasecuris.aiasylum.weights.diagnostics import benign_centred_residuals, stable_rank

    scores: List[LayerScore] = []
    directions: Dict[int, object] = {}

    for layer in range(lo, hi):
        # Difference in means, on train only.
        diff = harmful_tr[layer].mean(dim=0) - harmless_tr[layer].mean(dim=0)
        norm = diff.norm()
        if norm < 1e-8:
            logger.debug("Layer %d: degenerate direction (norm %.2e), skipping", layer, norm)
            continue
        r = diff / norm
        directions[layer] = r

        # Evaluate separation on held-out prompts only.
        pos = harmful_te[layer] @ r
        neg = harmless_te[layer] @ r
        scores.append(
            LayerScore(
                layer=layer, auc=_auc(pos, neg), cohens_d=_cohens_d(pos, neg), train_norm=norm.item(),
                stable_rank=stable_rank(benign_centred_residuals(harmful_tr[layer], harmless_tr[layer])),
            )
        )

    if not scores:
        raise ValueError("No usable layer produced a non-degenerate direction.")
    if return_captures:
        captures = {
            "harmful_train": harmful_tr, "harmless_train": harmless_tr,
            "harmful_test": harmful_te, "harmless_test": harmless_te,
        }
        return directions, scores, captures
    return directions, scores


def _projection_extra(directions, scores, captures, layer: int, vector) -> dict:
    """Class projection means at ``layer`` plus the stable-rank summary."""
    from vivasecuris.aiasylum.weights.diagnostics import stable_rank_summary

    h = (captures["harmful_train"][layer] @ vector).mean().item()
    b = (captures["harmless_train"][layer] @ vector).mean().item()
    sr = {s.layer: s.stable_rank for s in scores if s.stable_rank == s.stable_rank}
    return {
        "projection_means": {"harmful": h, "harmless": b, "layer": int(layer), "centered": False},
        "stable_rank": stable_rank_summary(sr, layer),
    }


def _rank_layers(scores: List[LayerScore]) -> List[LayerScore]:
    """Best-separating layers first. AUC saturates at 1.0 on a cleanly separable
    pair, so it cannot rank them alone; Cohen's d breaks the tie."""
    return sorted(scores, key=lambda s: (round(s.auc, 6), s.cohens_d), reverse=True)


def derive_direction(
    model,
    tokenizer,
    split,
    model_id: str = "unknown",
    batch_size: int = 8,
    max_length: int = 512,
    layer_range: Optional[tuple] = None,
    progress: Optional[callable] = None,
) -> RefusalDirection:
    """Fit a direction on ``split``'s train half and select the layer on its test half."""
    import torch

    directions, scores, caps = _score_layers(
        model, tokenizer, split, batch_size=batch_size,
        max_length=max_length, layer_range=layer_range, progress=progress,
        return_captures=True,
    )

    best = _rank_layers(scores)[0]
    vec = directions[best.layer].to(torch.float32)
    direction = RefusalDirection(
        vector=vec,
        layer=best.layer,
        auc=best.auc,
        cohens_d=best.cohens_d,
        model_id=model_id,
        split_hash=split.hash,
        layer_scores=sorted(scores, key=lambda s: s.layer),
        extra=_projection_extra(directions, scores, caps, best.layer, vec),
    )

    logger.info("Best layer %d: held-out AUC %.3f, Cohen's d %.2f", best.layer, best.auc, best.cohens_d)
    if not direction.usable:
        logger.warning(
            "Held-out AUC %.3f is below the %.2f threshold. The direction does not "
            "separate these classes; steering and surgery will not produce a reliable "
            "behavior change.",
            direction.auc, MIN_USABLE_AUC,
        )
    return direction


def derive_subspace(
    model,
    tokenizer,
    split,
    rank: int = 4,
    pool_layers: int = 12,
    model_id: str = "unknown",
    batch_size: int = 8,
    max_length: int = 512,
    layer_range: Optional[tuple] = None,
    progress: Optional[callable] = None,
    tol: float = 1e-4,
) -> RefusalDirection:
    """Derive an orthonormal refusal *subspace* of up to ``rank`` directions.

    Rationale: a single difference-in-means vector captures the dominant refusal
    direction but not all of it, so a few prompts still refuse after a beta=0
    edit. Pooling the best-separating layers and keeping their leading orthogonal
    components removes those residual refusal directions too.

    Construction guarantees ``basis[0]`` is exactly the single best direction, so
    ``rank=1`` reduces bit-for-bit to :func:`derive_direction` and the resulting
    surgery is identical to a beta=0 edit. Additional rows are obtained by
    removing the already-chosen components from the pooled layer directions and
    taking the leading singular vectors of the residual -- i.e. the principal
    directions of refusal that the first vector does not already explain.
    """
    import torch

    if rank < 1:
        raise ValueError(f"rank must be >= 1, got {rank}")

    directions, scores, caps = _score_layers(
        model, tokenizer, split, batch_size=batch_size,
        max_length=max_length, layer_range=layer_range, progress=progress,
        return_captures=True,
    )

    ranked = _rank_layers(scores)
    best = ranked[0]
    best_vec = directions[best.layer].to(torch.float32)
    best_vec = best_vec / best_vec.norm()

    rows = [best_vec]
    row_layers = [best.layer]

    if rank > 1:
        pool = [s.layer for s in ranked[:max(pool_layers, rank)] if s.layer in directions]
        M = torch.stack([directions[L].to(torch.float32) for L in pool], dim=0)  # [P, d]
        B = torch.stack(rows, dim=0)                                             # [1, d]
        # Residual of each pooled direction after removing the chosen components.
        resid = M - (M @ B.t()) @ B                                             # [P, d]
        # Leading orthogonal refusal directions the first vector doesn't explain.
        _, S, Vh = torch.linalg.svd(resid, full_matrices=False)
        s0 = S[0].item() if S.numel() else 0.0
        for i in range(Vh.shape[0]):
            if len(rows) >= rank:
                break
            if s0 > 0 and S[i].item() > tol * s0:
                rows.append(Vh[i])
                # Attribute this component to the pooled layer it aligns with most.
                align = (M @ Vh[i]).abs()
                row_layers.append(pool[int(align.argmax().item())])

    basis = torch.stack(rows, dim=0).contiguous()  # [m, d], orthonormal rows

    # Orthonormality sanity check: G should be the identity.
    gram = basis @ basis.t()
    off = (gram - torch.eye(basis.shape[0])).abs().max().item()
    if off > 1e-3:
        logger.warning("Subspace basis is not orthonormal (max off-diagonal %.2e)", off)

    direction = RefusalDirection(
        vector=best_vec,
        layer=best.layer,
        auc=best.auc,
        cohens_d=best.cohens_d,
        model_id=model_id,
        split_hash=split.hash,
        layer_scores=sorted(scores, key=lambda s: s.layer),
        basis=basis,
        basis_layers=row_layers,
        method="diff_in_means",
        extra=_projection_extra(directions, scores, caps, best.layer, best_vec),
    )
    logger.info(
        "Derived rank-%d refusal subspace from layers %s (best layer %d, AUC %.3f)",
        direction.rank, row_layers, best.layer, best.auc,
    )
    return direction
