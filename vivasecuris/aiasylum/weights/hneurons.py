"""Hallucination-associated neurons: find them, scale them, bake the scale exactly.

After H-Neurons (Tsinghua, arXiv 2512.01797): a sparse subset of feed-forward
neurons whose activation predicts whether the model is about to hallucinate.
The contribution of neuron ``j`` at token ``t`` is measured by CETT -- how much
of the MLP's output vector that one neuron writes:

    CETT_{j,t} = || W_down[:, j] * z_{j,t} ||_2 / || h_t ||_2
               = |z_{j,t}| * || W_down[:, j] ||_2 / || h_t ||_2

where ``z_t`` is the SwiGLU output (the input to ``down_proj``) and
``h_t = W_down @ z_t`` is the MLP's contribution to the residual. Averaged over
the answer tokens (and, separately, the other tokens), that is one feature per
neuron; an L1-regularised logistic regression on those features keeps well under
0.1% of them.

The intervention scales the selected neurons' SwiGLU output, ``z_j <- alpha*z_j``.
Because ``down_proj`` is linear in ``z``, scaling input channel ``j`` by ``alpha``
is *identically* scaling column ``j`` of ``W_down`` by ``alpha`` -- so the runtime
hook and the baked weight edit produce the same logits, with no bias to touch.
That equivalence is what makes the bake trustworthy, and it is asserted in the
tests.

**Read the replication caveat before trusting a number here.** arXiv 2604.19765
(which tested Qwen2.5-3B) found the detector barely transfers across domains and
that scaling moved behaviour by ~0.001. So selection reports a shuffled-label
null and an answer-length surface baseline, and the *induce* stage runs an
explicit causal check: if scaling does not move the hallucination rate, that is
reported plainly rather than assumed away.

Dense models only. A mixture-of-experts block writes through many expert
down-projections and is refused with a clear error.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


def _dense_down_projs(model) -> List[Tuple[int, Any]]:
    """``(layer_idx, down_proj_module)`` per block, or refuse a MoE model."""
    from vivasecuris.aiasylum.interp.core.arch import (
        _MLP_DOWN_ATTRS,
        _named_submodule_with_weight,
        describe_architecture,
        get_layer_stack,
    )

    info = describe_architecture(model)
    stack = get_layer_stack(model, info.family) if info.family else None
    if stack is None:
        raise ValueError(f"Could not enumerate decoder layers for {info.label}.")
    out: List[Tuple[int, Any]] = []
    for layer_idx, _attn, mlp in stack:
        if getattr(mlp, "experts", None) is not None:
            raise ValueError(
                f"{info.label} is a mixture-of-experts model: layer {layer_idx} routes "
                f"through many expert down-projections, so a single per-neuron scale is "
                f"ill-defined. H-Neuron editing supports dense MLPs only."
            )
        down = _named_submodule_with_weight(mlp, _MLP_DOWN_ATTRS)
        if down is None:
            raise ValueError(f"Layer {layer_idx} of {info.label} exposes no down-projection.")
        out.append((layer_idx, down))
    return out


def column_norms(down_module) -> "Any":
    """L2 norm of each column of ``W_down`` (shape ``[d_ff]``)."""
    import torch

    W = down_module.weight  # [d_model, d_ff]
    return torch.linalg.vector_norm(W.detach().float(), dim=0)


def cett_per_token(z, out, col_norm):
    """CETT for every neuron at every token: ``|z| * col_norm / ||out||``.

    ``z`` is ``[T, d_ff]`` (down_proj input), ``out`` is ``[T, d_model]``
    (down_proj output), ``col_norm`` is ``[d_ff]``. Returns ``[T, d_ff]``.
    """
    import torch

    denom = torch.linalg.vector_norm(out.float(), dim=-1, keepdim=True).clamp_min(1e-8)  # [T,1]
    return z.abs().float() * col_norm.unsqueeze(0) / denom


@dataclass
class HNeuronSet:
    """Selected hallucination neurons per layer, with the evidence for the choice."""

    neurons: Dict[int, List[int]]      # layer index -> neuron indices within that layer's d_ff
    model_id: str
    d_ff: int
    auroc: float = float("nan")
    null_auroc_p95: float = float("nan")
    surface_auroc: float = float("nan")
    n_selected: int = 0
    n_total: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def fraction(self) -> float:
        return self.n_selected / self.n_total if self.n_total else 0.0

    @property
    def beats_null(self) -> bool:
        return (math.isfinite(self.auroc) and math.isfinite(self.null_auroc_p95)
                and self.auroc >= self.null_auroc_p95 + 0.05)

    @property
    def beats_surface(self) -> bool:
        return (math.isfinite(self.auroc) and math.isfinite(self.surface_auroc)
                and self.auroc >= self.surface_auroc + 0.05)

    @property
    def usable(self) -> bool:
        """Evidence clears both baselines and identifies at least one neuron.

        Missing surface evidence is not a passing comparison. This is a detector
        check; a causal intervention still needs its own behavioural evaluation.
        """
        return self.n_selected > 0 and self.beats_null and self.beats_surface

    def metadata(self) -> dict:
        return {
            "model_id": self.model_id,
            "d_ff": self.d_ff,
            "auroc": self.auroc,
            "null_auroc_p95": self.null_auroc_p95,
            "surface_auroc": self.surface_auroc,
            "n_selected": self.n_selected,
            "n_total": self.n_total,
            "fraction": self.fraction,
            "beats_null": self.beats_null,
            "beats_surface": self.beats_surface,
            "usable": self.usable,
            "neurons": {str(k): list(v) for k, v in sorted(self.neurons.items())},
            "extra": dict(self.extra),
        }

    def save(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "hneurons.json").write_text(json.dumps(self.metadata(), indent=2))
        logger.info("Saved %d H-neurons (%.4f%% of %d) to %s",
                    self.n_selected, self.fraction * 100, self.n_total, out)
        return out

    @classmethod
    def load(cls, path: str | Path) -> "HNeuronSet":
        p = Path(path)
        meta = json.loads((p / "hneurons.json" if p.is_dir() else p).read_text())
        return cls(
            neurons={int(k): list(v) for k, v in meta.get("neurons", {}).items()},
            model_id=meta.get("model_id", "unknown"),
            d_ff=int(meta.get("d_ff", 0)),
            auroc=float(meta.get("auroc", float("nan"))),
            null_auroc_p95=float(meta.get("null_auroc_p95", float("nan"))),
            surface_auroc=float(meta.get("surface_auroc", float("nan"))),
            n_selected=int(meta.get("n_selected", 0)),
            n_total=int(meta.get("n_total", 0)),
            extra=dict(meta.get("extra") or {}),
        )


def _l1_logreg(C: float, max_iter: int = 5000):
    """An L1-penalised logistic regression, across sklearn API versions.

    sklearn 1.8 deprecated ``penalty='l1'`` in favour of ``l1_ratio``; the old
    spelling additionally trips an "inconsistent l1_ratio" warning there. Pick
    the spelling that actually applies pure L1 on the installed version.
    """
    from sklearn.linear_model import LogisticRegression

    try:
        import sklearn
        from packaging.version import Version

        if Version(sklearn.__version__) >= Version("1.8"):
            # New API: l1_ratio=1 is pure L1; saga is the solver that supports it.
            return LogisticRegression(l1_ratio=1.0, solver="saga", C=C, max_iter=max_iter, random_state=0)
    except Exception:
        pass
    return LogisticRegression(penalty="l1", solver="liblinear", C=C, max_iter=max_iter, random_state=0)


def _auroc(scores, labels) -> float:
    import numpy as np
    from scipy.stats import rankdata

    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # Average ranks give tied positive/negative pairs half credit.
    ranks = rankdata(s, method="average")
    return float((ranks[y == 1].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def _prefilter(features, labels, top_k, rows):
    """Rank features using only the labels belonging to this fitting fold."""
    import numpy as np
    from scipy.stats import rankdata

    fold_labels = labels[rows]
    n_pos = int((fold_labels == 1).sum())
    n_neg = len(fold_labels) - n_pos
    strength = np.empty(features.shape[1])
    # Bound rank-array memory for real models with millions of CETT columns.
    for start in range(0, features.shape[1], 2048):
        ranks = rankdata(features[rows, start:start + 2048], axis=0, method="average")
        aucs = (ranks[fold_labels == 1].sum(axis=0) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        strength[start:start + 2048] = np.abs(aucs - 0.5)
    return np.argsort(-strength, kind="stable")[:min(top_k, features.shape[1])]


def _fit_selection(features, labels, top_k, c_grid, seed):
    """Fit the complete selector, including C tuning, on training rows only."""
    import numpy as np
    from sklearn.model_selection import train_test_split

    def prepare(rows):
        keep = _prefilter(features, labels, top_k, rows)
        X = features[np.ix_(rows, keep)]
        mean, std = X.mean(axis=0), X.std(axis=0)
        std[std < 1e-8] = 1.0
        return keep, mean, std, (X - mean) / std

    rows = np.arange(len(labels))
    C = c_grid[0]
    if len(c_grid) > 1:
        # This validation set is inside the outer training partition. Its labels
        # never select columns or set normalization statistics for an inner fit.
        if min(np.bincount(labels, minlength=2)) < 2:
            raise ValueError("C tuning needs at least two training questions per class.")
        inner_train, validation = train_test_split(
            rows, test_size=max(2, int(round(len(rows) * 0.25))),
            random_state=seed, stratify=labels,
        )
        keep, mean, std, Ztrain = prepare(inner_train)
        Zvalidation = (features[np.ix_(validation, keep)] - mean) / std
        best_auc = -1.0
        for candidate in c_grid:
            clf = _l1_logreg(candidate).fit(Ztrain, labels[inner_train])
            auc = _auroc(clf.predict_proba(Zvalidation)[:, 1], labels[validation])
            if auc > best_auc:
                C, best_auc = candidate, auc
    # Once C is fixed, refit the complete selector on the outer training set.
    keep, mean, std, Ztrain = prepare(rows)
    clf = _l1_logreg(C).fit(Ztrain, labels)
    return keep, mean, std, clf, C


def select_hneurons(
    features,
    labels: Sequence[int],
    feature_map: Sequence[Tuple[int, int]],
    d_ff: int,
    model_id: str = "unknown",
    top_k: int = 20000,
    c_grid: Sequence[float] = (0.01, 0.1, 1.0),
    test_fraction: float = 0.3,
    seed: int = 0,
    answer_lengths: Optional[Sequence[float]] = None,
) -> HNeuronSet:
    """Pick hallucination neurons from CETT features by L1 logistic regression.

    ``features`` is ``[n, F]`` (F = 2 * n_layers * d_ff, an answer-token and an
    other-token CETT per neuron). ``feature_map[k] = (layer, neuron)`` says which
    neuron column ``k`` belongs to. A univariate AUROC prefilter keeps the top
    ``top_k`` columns; an L1 fit over a ``C`` grid then selects, and a neuron is
    kept if either of its two features survives with a positive (toward-wrong)
    weight. Feature selection and C tuning use only the training partition;
    reported AUROC uses untouched report rows. Each shuffled-label null repeats
    the full selector. When ``answer_lengths`` is given, reports the AUROC
    reachable from answer length alone as a conservative surface baseline.
    """
    import warnings

    import numpy as np

    X = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=int)
    if X.ndim != 2 or not X.shape[1] or len(y) != X.shape[0] or y.ndim != 1:
        raise ValueError("features must be a nonempty [questions, features] matrix matching labels.")
    n, F = X.shape
    if len(feature_map) != F or not np.isfinite(X).all():
        raise ValueError("Every finite feature column needs a feature_map entry.")
    if set(y.tolist()) != {0, 1}:
        raise ValueError("H-neuron selection needs both binary label classes (0 and 1).")
    if not 0 < test_fraction < 1 or top_k < 1:
        raise ValueError("test_fraction must be between 0 and 1 and top_k must be positive.")
    c_grid = tuple(float(c) for c in c_grid)
    if not c_grid or any(not np.isfinite(c) or c <= 0 for c in c_grid):
        raise ValueError("c_grid must contain finite positive regularization values.")
    if answer_lengths is not None:
        al = np.asarray(answer_lengths, float)
        if al.shape != (n,) or not np.isfinite(al).all():
            raise ValueError("answer_lengths must contain one finite value per question.")
    rng = np.random.default_rng(seed)

    # Split before *any* supervised operation. Membership is independent of the
    # labels so altering report labels cannot change the learned detector.
    idx = rng.permutation(n)
    n_test = max(1, int(round(n * test_fraction)))
    te, tr = idx[:n_test], idx[n_test:]
    if any(len(np.unique(y[rows])) < 2 for rows in (tr, te)):
        raise ValueError("Train and report partitions must each contain both classes; add more questions.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Xtrain, ytrain = X[tr], y[tr]
        keep, mean, std, clf, C = _fit_selection(Xtrain, ytrain, top_k, c_grid, seed)
        auroc = _auroc(clf.predict_proba((X[np.ix_(te, keep)] - mean) / std)[:, 1], y[te])
        coef = clf.coef_[0]

        # Repeat the *entire* procedure for every training-label permutation:
        # feature ranking, normalization, inner C tuning, and final refitting.
        nulls = []
        for _ in range(20):
            ysh = rng.permutation(ytrain)
            nk, nm, ns, nc, _ = _fit_selection(Xtrain, ysh, top_k, c_grid, seed)
            a = _auroc(nc.predict_proba((X[np.ix_(te, nk)] - nm) / ns)[:, 1], y[te])
            nulls.append(max(a, 1.0 - a))
    null_p95 = float(np.percentile(nulls, 95))

    surface = float("nan")
    if answer_lengths is not None:
        al = np.asarray(answer_lengths, float)
        surface = max(_auroc(al[te], y[te]), 1.0 - _auroc(al[te], y[te]))

    # A neuron is selected if either of its features survived toward the wrong class.
    neurons: Dict[int, set] = {}
    for local_k, kept_col in enumerate(keep):
        if coef[local_k] > 0:
            layer, neuron = feature_map[kept_col]
            neurons.setdefault(int(layer), set()).add(int(neuron))
    neurons_list = {L: sorted(s) for L, s in sorted(neurons.items())}
    n_selected = sum(len(v) for v in neurons_list.values())

    hset = HNeuronSet(
        neurons=neurons_list, model_id=model_id, d_ff=int(d_ff),
        auroc=float(auroc), null_auroc_p95=null_p95, surface_auroc=surface,
        n_selected=n_selected, n_total=int(F // 2),
        extra={"C": float(C), "top_k": int(top_k),
               "selection": "training_only_with_inner_validation",
               "train_indices": tr.tolist(), "report_indices": te.tolist(),
               "null_repeats": len(nulls), "null_full_selection": True},
    )
    logger.info("Selected %d H-neurons (AUROC %.3f, null p95 %.3f, %.4f%%)",
                n_selected, auroc, null_p95, hset.fraction * 100)
    return hset


def scale_neurons(model, hset: HNeuronSet, alpha: float):
    """Context manager scaling the selected neurons' ``down_proj`` input by ``alpha``.

    ``alpha < 1`` suppresses the hallucination neurons, ``> 1`` amplifies them.
    Applies at every position, matching what :func:`bake_hneurons` does to the
    weights, so a hooked run and a baked run agree.
    """
    import torch
    from contextlib import contextmanager

    @contextmanager
    def _cm():
        by_layer = {L: down for L, down in _dense_down_projs(model)}
        handles = []
        try:
            for layer, idx in hset.neurons.items():
                if not idx or layer not in by_layer:
                    continue
                index = torch.as_tensor(idx, dtype=torch.long)

                def make(index):
                    def hook(_module, args):
                        if not args:
                            return None
                        x = args[0]
                        if x is None or not hasattr(x, "shape"):
                            return None
                        x = x.clone()
                        ix = index.to(x.device)
                        x[..., ix] = x[..., ix] * alpha
                        return (x,) + tuple(args[1:])
                    return hook

                handles.append(by_layer[layer].register_forward_pre_hook(make(index)))
            yield model
        finally:
            for h in handles:
                h.remove()

    return _cm()


def label_consistency(
    model,
    tokenizer,
    rows: Sequence[Dict[str, Any]],
    *,
    n_samples: int = 10,
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: int = 50,
    max_new_tokens: int = 24,
    seed: int = 0,
    progress=None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Split QA rows into consistently-correct vs consistently-incorrect.

    After H-Neurons: sample ``n_samples`` answers per question at ``t=1`` and keep
    only questions the model gets right *every* time (it reliably knows) or wrong
    *every* time (it reliably does not, and commits anyway). Rows that sometimes
    abstain or flip are dropped -- they carry no clean label. ``rows`` are
    ``{"question", "aliases"}`` (as `benchmarks triviaqa` yields).
    """
    import torch

    from vivasecuris.aiasylum.weights.capture import format_prompts, strip_thinking
    from vivasecuris.aiasylum.weights.factual import classify_answer

    correct: List[Dict[str, Any]] = []
    incorrect: List[Dict[str, Any]] = []
    total = len(rows)
    for i, row in enumerate(rows, 1):
        q = row.get("question") or ""
        aliases = row.get("aliases") or ([row["answer"]] if row.get("answer") else [])
        texts, applied = format_prompts(tokenizer, [q])
        enc = tokenizer(texts[0], return_tensors="pt",
                        add_special_tokens=not applied).to(model.device)
        verdicts = []
        for s in range(n_samples):
            torch.manual_seed(seed + s)
            with torch.no_grad():
                out = model.generate(
                    **enc, do_sample=True, temperature=temperature, top_p=top_p, top_k=top_k,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            ans = strip_thinking(tokenizer.decode(out[0][enc["input_ids"].shape[1]:],
                                                  skip_special_tokens=True))
            verdicts.append(classify_answer(ans, aliases))
        if all(v == "correct" for v in verdicts):
            correct.append({**row, "aliases": aliases})
        elif all(v == "wrong" for v in verdicts):
            incorrect.append({**row, "aliases": aliases})
        if progress:
            progress(i, total)
    logger.info("Consistency labelling: %d consistently-correct, %d consistently-incorrect of %d",
                len(correct), len(incorrect), total)
    return {"correct": correct, "incorrect": incorrect}


def capture_cett(
    model,
    tokenizer,
    questions: Sequence[str],
    *,
    max_new_tokens: int = 24,
    progress=None,
):
    """Per-neuron CETT features over each question's own greedily-generated answer.

    For each question: generate the answer greedily, then run one more forward
    over prompt+answer with hooks on every ``down_proj`` capturing its input
    ``z`` and output ``h``. CETT is averaged over the answer tokens and,
    separately, over the prompt tokens, giving ``2 * n_layers * d_ff`` features.

    Returns ``(features [n, 2*L*d_ff] float32, feature_map, d_ff, answer_lengths)``
    where ``feature_map[k] = (layer, neuron)``. Using the model's own answer span
    (rather than a GPT-4o alignment) keeps this ADR-009-clean and self-contained.
    """
    import torch

    from vivasecuris.aiasylum.weights.capture import format_prompts

    downs = _dense_down_projs(model)
    layers = [L for L, _ in downs]
    d_ff = downs[0][1].weight.shape[1]
    col_norms = {L: column_norms(mod).to("cpu") for L, mod in downs}

    # feature_map: for each layer, d_ff answer-features then d_ff other-features.
    feature_map: List[Tuple[int, int]] = []
    for L in layers:
        feature_map += [(L, j) for j in range(d_ff)]   # answer block
        feature_map += [(L, j) for j in range(d_ff)]   # other block

    all_feats: List["Any"] = []
    answer_lengths: List[float] = []
    total = len(questions)

    captured: Dict[int, Dict[str, "Any"]] = {}

    def make_hook(layer_idx: int):
        def hook(_module, inp, out):
            z = inp[0]
            h = out[0] if isinstance(out, tuple) else out
            captured[layer_idx] = {"z": z.detach(), "h": h.detach()}
        return hook

    for i, q in enumerate(questions, 1):
        texts, applied = format_prompts(tokenizer, [q])
        enc = tokenizer(texts[0], return_tensors="pt",
                        add_special_tokens=not applied).to(model.device)
        p_len = enc["input_ids"].shape[1]
        with torch.no_grad():
            gen = model.generate(**enc, do_sample=False, max_new_tokens=max_new_tokens,
                                 pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        full = gen
        a_len = full.shape[1] - p_len
        answer_lengths.append(float(max(a_len, 0)))

        captured.clear()
        handles = [mod.register_forward_hook(make_hook(L)) for L, mod in downs]
        try:
            with torch.no_grad():
                model(input_ids=full, attention_mask=torch.ones_like(full))
        finally:
            for hd in handles:
                hd.remove()

        feats = []
        for L in layers:
            z = captured[L]["z"][0].to("cpu").float()      # [T, d_ff]
            h = captured[L]["h"][0].to("cpu").float()      # [T, d_model]
            cett = cett_per_token(z, h, col_norms[L])      # [T, d_ff]
            ans = cett[p_len:] if a_len > 0 else cett[-1:]
            other = cett[:p_len] if p_len > 0 else cett[:1]
            feats.append(ans.mean(dim=0))                  # [d_ff]
            feats.append(other.mean(dim=0))                # [d_ff]
        all_feats.append(torch.cat(feats))                 # [2*L*d_ff]
        if progress:
            progress(i, total)

    features = torch.stack(all_feats).numpy()
    return features, feature_map, d_ff, answer_lengths


def bake_hneurons(model, hset: HNeuronSet, alpha: float) -> Dict[str, Any]:
    """Scale column ``j`` of each layer's ``W_down`` by ``alpha`` in place, for selected ``j``.

    Exactly equivalent to :func:`scale_neurons` because ``down_proj`` is linear
    in its input and no bias sits on the neuron axis. Returns a small summary for
    the surgery manifest.
    """
    import torch

    by_layer = {L: down for L, down in _dense_down_projs(model)}
    edited = 0
    matrices_edited = 0
    rel_changes: List[float] = []
    with torch.no_grad():
        for layer, idx in hset.neurons.items():
            if not idx or layer not in by_layer:
                continue
            W = by_layer[layer].weight            # [d_model, d_ff]
            index = torch.as_tensor(idx, dtype=torch.long, device=W.device)
            before = W.detach().float().norm().item()
            delta = (torch.linalg.vector_norm(W[:, index].detach().float())
                     * abs(alpha - 1.0)).item()
            W[:, index] = W[:, index] * alpha
            edited += len(idx)
            matrices_edited += 1
            rel_changes.append(delta / before if before else 0.0)
    return {
        "method": "hneuron_scale", "alpha": float(alpha),
        "neurons_scaled": edited, "layers": sorted(hset.neurons),
        "matrices_edited": matrices_edited,
        "mean_relative_change": (sum(rel_changes) / len(rel_changes)) if rel_changes else 0.0,
    }
