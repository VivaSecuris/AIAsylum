"""Train and score per-layer logistic probes on raw residual activations.

One probe per layer, standardised inputs, held-out AUROC and calibration, and
a shuffled-label null for every layer. The best layer is chosen on held-out
data; choosing it on the fitting set picks whichever layer overfit hardest,
the same trap ``weights.direction`` documents for layer selection.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

# Below this held-out AUROC the probe is not separating the classes well enough
# to act on, and a monitor built from it would mostly emit noise.
MIN_USABLE_AUROC = 0.75

# How far above its own shuffled-label ceiling a probe has to sit before the
# result counts as detection rather than memorisation.
MIN_MARGIN_OVER_NULL = 0.05


@dataclass
class LayerProbe:
    layer: int
    weight: object                  # numpy [d]
    bias: float
    mean: object                    # numpy [d] standardisation
    scale: object                   # numpy [d]
    auroc: float
    accuracy: float
    ece: float
    null_auroc_p95: float

    @property
    def beats_null(self) -> bool:
        return self.auroc >= self.null_auroc_p95 + MIN_MARGIN_OVER_NULL

    def score(self, acts) -> "object":
        """Probability of the harmful class for ``[n, d]`` activations."""
        import numpy as np

        x = (np.asarray(acts, dtype=np.float64) - self.mean) / self.scale
        return 1.0 / (1.0 + np.exp(-(x @ self.weight + self.bias)))


@dataclass
class ProbeSet:
    """Per-layer probes plus the evidence for picking one."""

    probes: Dict[int, LayerProbe]
    best_layer: int
    model_id: str
    pooling: str
    dataset_hash: str
    prompt_suffix: Optional[str] = None
    group_auroc: Dict[str, float] = field(default_factory=dict)
    dataset_summary: dict = field(default_factory=dict)
    # AUROC reachable from prompt length alone on the same held-out rows.
    surface_baseline: Dict[str, float] = field(default_factory=dict)

    @property
    def best(self) -> LayerProbe:
        return self.probes[self.best_layer]

    @property
    def surface_ceiling(self) -> float:
        return float(self.surface_baseline.get("worst_case", 0.0) or 0.0)

    @property
    def beats_surface(self) -> bool:
        """Does the probe do better than counting characters?

        The shuffled-label null asks whether any signal exists. This asks
        whether the signal is about harm. On this corpus the harmful class is
        systematically longer -- jailbreak wrappers are long roleplay texts and
        the benign contrast set is short questions -- so character count alone
        reaches 0.99 AUROC on the held-out half. A probe at 1.0 has, by itself,
        demonstrated almost nothing.

        Shuffling labels does not catch this: it destroys the length-label
        correlation too, so a pure length classifier fails the null exactly as
        a real detector does. The two baselines answer different questions and
        a probe has to clear both.
        """
        return self.best.auroc >= self.surface_ceiling + MIN_MARGIN_OVER_NULL

    @property
    def usable(self) -> bool:
        return (self.best.auroc >= MIN_USABLE_AUROC
                and self.best.beats_null
                and self.beats_surface)

    def metadata(self) -> dict:
        return {
            "model_id": self.model_id,
            "pooling": self.pooling,
            "prompt_suffix": self.prompt_suffix,
            "dataset_hash": self.dataset_hash,
            "dataset": self.dataset_summary,
            "best_layer": self.best_layer,
            "best_auroc": round(self.best.auroc, 4),
            "best_ece": round(self.best.ece, 4),
            "null_auroc_p95": round(self.best.null_auroc_p95, 4),
            "beats_null": self.best.beats_null,
            "surface_baseline": dict(self.surface_baseline),
            "surface_ceiling": round(self.surface_ceiling, 4),
            "beats_surface": self.beats_surface,
            "usable": self.usable,
            "min_usable_auroc": MIN_USABLE_AUROC,
            "claim": "descriptive",
            "detector": "logistic regression on raw residual activations",
            "note": (
                "Raw activations on purpose: SAE features measured worse for this task "
                "on every model in SAEGuardBench (April 2026). SAE features explain a "
                "detection; they do not make it."
            ),
            "caveat": (
                "Read AUROC against both baselines. `null_auroc_p95` is the "
                "shuffled-label ceiling and asks whether any signal exists. "
                "`surface_ceiling` is what prompt length alone reaches on the same "
                "rows and asks whether the signal is about harm rather than register. "
                "A probe that clears the first but not the second has learned that "
                "jailbreak wrappers are longer than benign questions."
            ),
            "group_auroc": {k: round(v, 4) for k, v in sorted(self.group_auroc.items())},
            "layers": [
                {
                    "layer": p.layer, "auroc": round(p.auroc, 4), "accuracy": round(p.accuracy, 4),
                    "ece": round(p.ece, 4), "null_auroc_p95": round(p.null_auroc_p95, 4),
                    "beats_null": p.beats_null,
                }
                for p in sorted(self.probes.values(), key=lambda p: p.layer)
            ],
        }

    def save(self, out_dir: str | Path) -> Path:
        import numpy as np

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        layers = sorted(self.probes)
        np.savez_compressed(
            out / "probes.npz",
            layers=np.array(layers),
            weight=np.stack([self.probes[L].weight for L in layers]),
            bias=np.array([self.probes[L].bias for L in layers]),
            mean=np.stack([self.probes[L].mean for L in layers]),
            scale=np.stack([self.probes[L].scale for L in layers]),
            auroc=np.array([self.probes[L].auroc for L in layers]),
            accuracy=np.array([self.probes[L].accuracy for L in layers]),
            ece=np.array([self.probes[L].ece for L in layers]),
            null_auroc_p95=np.array([self.probes[L].null_auroc_p95 for L in layers]),
        )
        (out / "probes.json").write_text(json.dumps(self.metadata(), indent=2))
        logger.info("Saved probe set to %s (best layer %d, AUROC %.3f)",
                    out, self.best_layer, self.best.auroc)
        return out

    @classmethod
    def load(cls, path: str | Path) -> "ProbeSet":
        import numpy as np

        p = Path(path)
        npz_path = p / "probes.npz" if p.is_dir() else p
        meta_path = (p / "probes.json") if p.is_dir() else p.with_suffix(".json")
        if not npz_path.exists():
            raise FileNotFoundError(
                f"No probes.npz at {npz_path}. Train one first:\n"
                f"    aiasylum weights probe --model <hf-id> --out {p}"
            )
        z = np.load(npz_path)
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        probes = {
            int(L): LayerProbe(
                layer=int(L), weight=z["weight"][i], bias=float(z["bias"][i]),
                mean=z["mean"][i], scale=z["scale"][i], auroc=float(z["auroc"][i]),
                accuracy=float(z["accuracy"][i]), ece=float(z["ece"][i]),
                null_auroc_p95=float(z["null_auroc_p95"][i]),
            )
            for i, L in enumerate(z["layers"].tolist())
        }
        return cls(
            probes=probes,
            best_layer=int(meta.get("best_layer", max(probes, key=lambda L: probes[L].auroc))),
            model_id=meta.get("model_id", "unknown"),
            pooling=meta.get("pooling", "last"),
            dataset_hash=meta.get("dataset_hash", "unknown"),
            prompt_suffix=meta.get("prompt_suffix"),
            group_auroc=meta.get("group_auroc", {}),
            dataset_summary=meta.get("dataset", {}),
            surface_baseline=meta.get("surface_baseline", {}),
        )


def _auroc(scores, labels) -> float:
    """Rank-based AUROC, tie-corrected."""
    import numpy as np

    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = scores.argsort()
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    uniq, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    if (counts > 1).any():
        sums = np.zeros(len(uniq))
        np.add.at(sums, inverse, ranks)
        ranks = (sums / counts)[inverse]
    n_pos = len(pos)
    rank_sum = ranks[labels == 1].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * len(neg)))


def _ece(probs, labels, bins: int = 10) -> float:
    """Expected calibration error: does a 0.8 score mean 80 percent?"""
    import numpy as np

    probs, labels = np.asarray(probs), np.asarray(labels)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (probs > lo) & (probs <= hi) if lo > 0 else (probs >= lo) & (probs <= hi)
        if not m.any():
            continue
        total += (m.sum() / len(probs)) * abs(labels[m].mean() - probs[m].mean())
    return float(total)


def train_probes(
    train_acts,
    train_labels: Sequence[int],
    test_acts,
    test_labels: Sequence[int],
    test_groups: Optional[Sequence[str]] = None,
    model_id: str = "unknown",
    pooling: str = "last",
    dataset_hash: str = "unknown",
    prompt_suffix: Optional[str] = None,
    dataset_summary: Optional[dict] = None,
    surface_baseline: Optional[Dict[str, float]] = None,
    layer_range: Optional[tuple] = None,
    n_perm: int = 20,
    seed: int = 0,
    progress: Optional[callable] = None,
) -> ProbeSet:
    """Fit one probe per layer on ``[n_layers+1, n, d]`` captures.

    A shuffled-label probe is fitted at every layer too, and its 95th-percentile
    AUROC becomes that layer's null. With a few hundred examples in a thousand
    dimensions the null is well above 0.5, which is exactly why it is measured
    rather than assumed.
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    y_tr = np.asarray(train_labels, dtype=int)
    y_te = np.asarray(test_labels, dtype=int)
    n_layers = train_acts.shape[0]
    lo, hi = layer_range or (1, n_layers)      # layer 0 is the embedding: no computation yet
    lo, hi = max(1, lo), min(n_layers, hi)
    rng = np.random.default_rng(seed)

    def fit(X, y):
        # lbfgs with a modest L2: the activations are standardised and the
        # sample counts are small relative to the dimension, so an unregularised
        # fit separates perfectly and generalises poorly.
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(X, y)
        return clf

    probes: Dict[int, LayerProbe] = {}
    for layer in range(lo, hi):
        if progress:
            progress(f"probe layer {layer}", layer - lo + 1, hi - lo)
        X_tr = np.asarray(train_acts[layer], dtype=np.float64)
        X_te = np.asarray(test_acts[layer], dtype=np.float64)
        mean = X_tr.mean(axis=0)
        scale = X_tr.std(axis=0)
        scale[scale < 1e-8] = 1.0
        Z_tr, Z_te = (X_tr - mean) / scale, (X_te - mean) / scale

        clf = fit(Z_tr, y_tr)
        p_te = clf.predict_proba(Z_te)[:, 1]
        auroc = _auroc(p_te, y_te)

        nulls = []
        for _ in range(int(n_perm)):
            shuffled = rng.permutation(y_tr)
            if len(set(shuffled.tolist())) < 2:
                continue
            p_null = fit(Z_tr, shuffled).predict_proba(Z_te)[:, 1]
            a = _auroc(p_null, y_te)
            nulls.append(max(a, 1.0 - a))       # a fitted probe is oriented
        null_p95 = float(np.percentile(nulls, 95)) if nulls else 0.5

        probes[layer] = LayerProbe(
            layer=layer, weight=clf.coef_[0], bias=float(clf.intercept_[0]),
            mean=mean, scale=scale, auroc=auroc,
            accuracy=float(((p_te >= 0.5).astype(int) == y_te).mean()),
            ece=_ece(p_te, y_te), null_auroc_p95=null_p95,
        )
        logger.debug("Layer %d: AUROC %.3f (null p95 %.3f)", layer, auroc, null_p95)

    if not probes:
        raise ValueError("No layers produced a probe.")

    # Best on held-out, among probes that clear their own null; if none does,
    # fall back to the highest AUROC so the failure is visible rather than fatal.
    clearing = [p for p in probes.values() if p.beats_null]
    pool = clearing or list(probes.values())
    best = max(pool, key=lambda p: p.auroc)

    group_auroc: Dict[str, float] = {}
    if test_groups is not None:
        groups = np.asarray(list(test_groups))
        benign = y_te == 0
        for g in sorted(set(groups[y_te == 1].tolist())):
            m = benign | (groups == g)
            probe_scores = best.score(np.asarray(test_acts[best.layer], dtype=np.float64)[m])
            group_auroc[g] = _auroc(probe_scores, y_te[m])

    ps = ProbeSet(
        probes=probes, best_layer=best.layer, model_id=model_id, pooling=pooling,
        dataset_hash=dataset_hash, prompt_suffix=prompt_suffix,
        group_auroc=group_auroc, dataset_summary=dataset_summary or {},
        surface_baseline=dict(
            surface_baseline or (dataset_summary or {}).get("surface_baseline") or {}
        ),
    )
    logger.info(
        "Best probe: layer %d, held-out AUROC %.3f (null p95 %.3f), ECE %.3f",
        best.layer, best.auroc, best.null_auroc_p95, best.ece,
    )
    if ps.surface_baseline and not ps.beats_surface:
        logger.warning(
            "Probe AUROC %.3f does not clear the %.3f reachable from prompt length "
            "alone. On this corpus the harmful class is longer than the benign one, "
            "so this score is not yet evidence that the probe encodes harm. "
            "Length-match the contrast sets, or read it as a register detector.",
            best.auroc, ps.surface_ceiling,
        )
    if not ps.usable:
        logger.warning(
            "Probe is not usable: AUROC %.3f against a %.2f threshold, a %.3f null "
            "and a %.3f surface ceiling.",
            best.auroc, MIN_USABLE_AUROC, best.null_auroc_p95, ps.surface_ceiling,
        )
    return ps
