"""Embedding cartography: align embedding spaces across models, and recover an
unknown local model's output-embedding geometry from queries alone.

Two composable capabilities, both defensive/interpretability research on local
open-weight models the operator controls.

1. **Cross-model alignment.** Two models trained separately do not share a
   coordinate frame, but they encode overlapping structure. Using tokens that
   exist in both vocabularies as *anchors*, a linear map (orthogonal Procrustes
   or ridge) reconstructs one model's embeddings from the other's, and
   anchor-relative representations (cosines to the anchor set) compare the two
   spaces with no learned map at all (Moschella et al. 2023). Retrieval
   precision on held-out anchors measures how well the spaces align.

2. **Black-box extraction.** A model's final layer maps a hidden state
   ``h in R^d`` to ``logits = h @ W_U.T (+ b)``, so the logit vectors a model
   emits lie in a ``d``-dimensional affine subspace of ``R^V``. Collecting many
   logit vectors and taking their SVD recovers the hidden size ``d`` (a knee in
   the spectrum) and the row-space of the output embedding ``W_U`` -- up to a
   ``d x d`` linear transform (Carlini et al. 2024). This exposes the *output*
   embedding geometry and hidden dimension; it does not recover the input
   embedding table, which logits never reveal.

The two compose: extraction leaves a residual linear transform, and anchor
alignment against a known reference model is exactly what resolves it, placing
an unknown model's recovered geometry into an interpretable frame.

**Scope.** Everything here runs on local models the operator controls; the
black-box oracle is guarded by :func:`weights.redteam.assert_local`, so no path
queries a hosted or third-party production API. When true weights are present
they are used only to *score* recovery, never by the estimator. The purpose is
to measure how much query access leaks about a model's embedding geometry.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# --- embedding extraction from weights (white-box) --------------------------

def embedding_matrices(model):
    """Return ``(E, W_U, tied)``: input embedding, output embedding, and whether tied.

    ``E`` is ``[V, d]`` (input token table); ``W_U`` is the unembedding ``[V, d]``
    (``lm_head.weight``). Small models often tie them (``arch.embeddings_are_tied``).
    """
    from vivasecuris.aiasylum.interp.core.arch import (
        _embedding_module,
        detect_architecture,
        embeddings_are_tied,
    )

    arch = detect_architecture(model)
    found = _embedding_module(model, arch) if arch else None
    E = None
    if found is not None:
        _, emb = found
        E = emb.weight.detach()
    elif hasattr(model, "get_input_embeddings"):
        ie = model.get_input_embeddings()
        E = ie.weight.detach() if ie is not None else None

    head = getattr(model, "lm_head", None)
    if head is None and hasattr(model, "get_output_embeddings"):
        head = model.get_output_embeddings()
    W_U = head.weight.detach() if (head is not None and hasattr(head, "weight")) else None
    if W_U is None:
        W_U = E
    return E, W_U, embeddings_are_tied(model)


# --- anchors: tokens present in both vocabularies ---------------------------

def shared_anchors(
    tok_a,
    tok_b,
    *,
    extra_words: Optional[Sequence[str]] = None,
    max_anchors: int = 4096,
    seed: int = 0,
) -> Tuple[List[int], List[int], List[str]]:
    """Strings that encode to exactly one real token in *both* tokenizers.

    Returns aligned ``(ids_a, ids_b, strings)``. The candidate pool is the raw
    vocabulary-string intersection plus any ``extra_words``; each candidate is
    kept only if :func:`remap.single_token_id` accepts it in both tokenizers
    (so a string that is one token in A but two in B is dropped -- its rows are
    not comparable). Deterministic given ``seed``.
    """
    import random

    from vivasecuris.aiasylum.weights.remap import single_token_id

    def vocab_size(tok) -> int:
        try:
            return int(getattr(tok, "vocab_size", 0) or len(tok.get_vocab()))
        except Exception:
            return 0

    # get_vocab keys are a tokenizer's internal spellings (e.g. GPT-2's "Ġthe"),
    # which differ across families; the *decoded* form of an id round-trips more
    # robustly, so build candidates by decoding the smaller vocab's ids and then
    # testing each string in both tokenizers.
    cand: set = set(extra_words or [])
    small_tok = tok_a if vocab_size(tok_a) <= vocab_size(tok_b) else tok_b
    n = min(vocab_size(small_tok), 200000)
    for tid in range(n):
        try:
            s = small_tok.decode([tid], skip_special_tokens=True)
        except Exception:
            continue
        if s and s.strip():
            cand.add(s)

    ids_a: List[int] = []
    ids_b: List[int] = []
    strings: List[str] = []
    seen: set = set()
    for s in sorted(cand):
        if s in seen:
            continue
        ia = single_token_id(tok_a, s)
        ib = single_token_id(tok_b, s)
        if ia is not None and ib is not None:
            seen.add(s)
            ids_a.append(int(ia))
            ids_b.append(int(ib))
            strings.append(s)

    if len(ids_a) > max_anchors:
        idx = list(range(len(ids_a)))
        random.Random(seed).shuffle(idx)
        idx = sorted(idx[:max_anchors])
        ids_a = [ids_a[i] for i in idx]
        ids_b = [ids_b[i] for i in idx]
        strings = [strings[i] for i in idx]
    logger.info("Found %d shared single-token anchors", len(ids_a))
    return ids_a, ids_b, strings


# --- linear alignment maps --------------------------------------------------

def _analysis_tensor(value, *, dtype=None):
    """Detached CPU arithmetic, including when model weights live on MPS/CUDA.

    The estimators need float64 SVD/solves. MPS cannot represent float64, and
    the two compared models need not even live on the same accelerator.
    Transfer before casting so none of that arithmetic runs on the model device.
    """
    import torch

    return value.detach().cpu().to(dtype=dtype or torch.float64)


def _anchor_rows(embeddings, ids):
    """Transfer only the selected rows before widening them to float64."""
    import torch

    indices = torch.as_tensor(list(ids), dtype=torch.long, device=embeddings.device)
    return _analysis_tensor(embeddings.detach().index_select(0, indices))


def procrustes(X, Y):
    """Orthogonal map ``M`` minimising ``||X M - Y||`` (requires equal width).

    ``M = U V^T`` from ``svd(X^T Y)``; applying it as ``X @ M`` rotates X's rows
    into Y's frame. Only defined when ``d_X == d_Y``.
    """
    import torch

    X = _analysis_tensor(X)
    Y = _analysis_tensor(Y)
    if X.shape[1] != Y.shape[1]:
        raise ValueError(f"Procrustes needs equal widths, got {X.shape[1]} and {Y.shape[1]}")
    U, _, Vh = torch.linalg.svd(X.t() @ Y, full_matrices=False)
    return (U @ Vh)


def ridge_map(X, Y, lam: float = 1.0):
    """Unconstrained least-squares map ``M`` (``d_X x d_Y``) with L2 ``lam``.

    Handles ``d_X != d_Y`` and small anchor counts. ``Y_hat = X @ M``.
    """
    import torch

    X = _analysis_tensor(X)
    Y = _analysis_tensor(Y)
    d = X.shape[1]
    A = X.t() @ X + lam * torch.eye(d, dtype=torch.float64)
    return torch.linalg.solve(A, X.t() @ Y)


def relative_repr(E, anchor_rows):
    """Anchor-relative coordinates: cosine of every row to each anchor row.

    Returns ``[V, n_anchors]``. Invariant to rotation/scaling of ``E`` (Moschella
    et al. 2023), so two models' relative reps are directly comparable even when
    their widths differ.
    """
    import torch

    E = _analysis_tensor(E)
    A = _analysis_tensor(anchor_rows)
    En = E / E.norm(dim=1, keepdim=True).clamp_min(1e-12)
    An = A / A.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return En @ An.t()


def retrieval_at_k(mapped, target_E, correct_rows: Sequence[int], ks: Sequence[int] = (1, 5),
                   chunk: int = 128) -> Dict[int, float]:
    """Precision@k: does ``mapped[i]``'s nearest neighbour in ``target_E`` equal ``correct_rows[i]``.

    Cosine similarity over every row of ``target_E``. Queries are processed in
    chunks in float32 so a full vocabulary (150k rows) stays within memory.
    Returns ``{k: precision}``.
    """
    import torch

    Tn = _analysis_tensor(target_E, dtype=torch.float32)
    Tn = Tn / Tn.norm(dim=1, keepdim=True).clamp_min(1e-12)
    m = _analysis_tensor(mapped, dtype=torch.float32)
    m = m / m.norm(dim=1, keepdim=True).clamp_min(1e-12)
    kmax = min(max(ks), Tn.shape[0])
    correct = torch.as_tensor(list(correct_rows)).view(-1, 1)
    tops = []
    for i in range(0, m.shape[0], chunk):
        sims = m[i:i + chunk] @ Tn.t()                  # [chunk, V]
        tops.append(sims.topk(kmax, dim=1).indices)
    topk = torch.cat(tops, dim=0)                       # [n_query, kmax]
    hits = (topk == correct)
    out = {}
    for k in ks:
        out[int(k)] = float(hits[:, :k].any(dim=1).float().mean().item())
    return out


# --- result types -----------------------------------------------------------

@dataclass
class AlignResult:
    model_a: str
    model_b: str
    n_anchors: int
    n_train: int
    n_test: int
    method: str
    cosine: float
    retrieval: Dict[int, float]
    null_shuffled: Dict[int, float] = field(default_factory=dict)
    null_random_init: Dict[int, float] = field(default_factory=dict)
    relative_agreement: Optional[float] = None
    relative_null: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        return {
            "model_a": self.model_a, "model_b": self.model_b,
            "n_anchors": self.n_anchors, "n_train": self.n_train, "n_test": self.n_test,
            "method": self.method, "cosine": self.cosine,
            "retrieval": {str(k): v for k, v in self.retrieval.items()},
            "null_shuffled": {str(k): v for k, v in self.null_shuffled.items()},
            "null_random_init": {str(k): v for k, v in self.null_random_init.items()},
            "relative_agreement": self.relative_agreement,
            "relative_null": self.relative_null,
            "extra": dict(self.extra),
        }

    def headline(self) -> str:
        p1 = self.retrieval.get(1, 0.0)
        n1 = self.null_shuffled.get(1, 0.0)
        pool = self.extra.get("retrieval_pool")
        pool_s = f" among {pool} rows" if pool else ""
        return (f"{self.method}: held-out P@1 {p1*100:.1f}%{pool_s} "
                f"(shuffled-anchor null {n1*100:.1f}%), cosine {self.cosine:.3f}")


@dataclass
class ExtractResult:
    model: str
    n_queries: int
    vocab_seen: int
    recovered_dim: int
    true_dim: Optional[int] = None
    spectrum: List[float] = field(default_factory=list)
    principal_angle_deg: Optional[float] = None
    subspace_overlap: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        return {
            "model": self.model, "n_queries": self.n_queries, "vocab_seen": self.vocab_seen,
            "recovered_dim": self.recovered_dim, "true_dim": self.true_dim,
            "spectrum": self.spectrum[:64],
            "principal_angle_deg": self.principal_angle_deg,
            "subspace_overlap": self.subspace_overlap,
            "extra": dict(self.extra),
        }

    def headline(self) -> str:
        s = f"recovered hidden dim {self.recovered_dim}"
        if self.true_dim is not None:
            s += f" (true {self.true_dim})"
        if self.subspace_overlap is not None:
            s += f", subspace overlap {self.subspace_overlap:.3f}"
        return s


# --- alignment driver -------------------------------------------------------

def align_from_matrices(
    E_a,
    E_b,
    ids_a: Sequence[int],
    ids_b: Sequence[int],
    *,
    model_a: str = "a",
    model_b: str = "b",
    method: str = "auto",
    test_fraction: float = 0.3,
    ridge_lambda: float = 1.0,
    seed: int = 0,
    random_init_E_b=None,
) -> AlignResult:
    """Fit an anchor map on a train split and score held-out retrieval, with nulls.

    ``method='auto'`` uses Procrustes when widths match, else ridge. Retrieval is
    over the anchor targets (``E_b[ids_b]``) so it is well-defined across widths.
    """
    import random

    import torch

    X = _anchor_rows(E_a, ids_a)     # [n, d_a], CPU float64
    Y = _anchor_rows(E_b, ids_b)     # [n, d_b], CPU float64
    n = X.shape[0]
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_test = max(1, int(round(n * test_fraction)))
    te, tr = idx[:n_test], idx[n_test:]

    use = method
    if method == "auto":
        use = "procrustes" if X.shape[1] == Y.shape[1] else "ridge"

    def fit(rows):
        Xt, Yt = X[rows], Y[rows]
        return procrustes(Xt, Yt) if use == "procrustes" else ridge_map(Xt, Yt, ridge_lambda)

    M = fit(tr)
    Yhat = X[te] @ M
    Yte = Y[te]
    cos = float((Yhat / Yhat.norm(dim=1, keepdim=True).clamp_min(1e-12)
                 * (Yte / Yte.norm(dim=1, keepdim=True).clamp_min(1e-12))).sum(dim=1).mean())
    # Retrieve among EVERY row of E_b, not just the held-out anchors: the correct
    # answer has to beat the whole vocabulary. Searching only the held-out anchor
    # set inflates P@1 (measured on gpt2 -> gpt2-medium: 99.6% vs 86.4%).
    correct = [int(ids_b[i]) for i in te]
    retr = retrieval_at_k(Yhat, E_b, correct)

    # Null 1: shuffle the train correspondence, refit, score on the same te.
    sh = tr[:]
    random.Random(seed + 1).shuffle(sh)
    Msh = procrustes(X[tr], Y[sh]) if use == "procrustes" else ridge_map(X[tr], Y[sh], ridge_lambda)
    null_sh = retrieval_at_k(X[te] @ Msh, E_b, correct)

    # Null 2: align to a random-init model of B's shape, if provided.
    null_ri: Dict[int, float] = {}
    if random_init_E_b is not None:
        Yr = _anchor_rows(random_init_E_b, ids_b)
        Mr = fit_pair(X[tr], Yr[tr], use, ridge_lambda)
        null_ri = retrieval_at_k(X[te] @ Mr, random_init_E_b, correct)

    return AlignResult(
        model_a=model_a, model_b=model_b, n_anchors=n, n_train=len(tr), n_test=len(te),
        method=use, cosine=cos, retrieval=retr, null_shuffled=null_sh,
        null_random_init=null_ri,
        extra={"ridge_lambda": ridge_lambda, "retrieval_pool": int(E_b.shape[0])},
    )


def fit_pair(X, Y, use: str, lam: float):
    return procrustes(X, Y) if use == "procrustes" else ridge_map(X, Y, lam)


def relative_agreement(E_a, E_b, ids_a, ids_b, *, center: bool = True, perm=None) -> float:
    """Mean cosine between the two models' anchor-relative reps of the anchors.

    Uses each anchor's similarity profile to all *other* anchors, so a model's
    width is irrelevant. High agreement = the two spaces place the anchors in the
    same relative configuration (the map-free alignment signal).

    ``center`` subtracts each model's anchor mean first. Token embeddings are
    anisotropic -- they share a large common direction -- and uncentered cosine
    profiles then look alike for *any* pair of models: on gpt2 vs gpt2-medium
    the uncentered score is 0.994 with a shuffled null of 0.979, i.e. it says
    nothing. Centered, it is 0.769 against a 0.243 null. ``perm`` reorders B's
    anchors to break the correspondence (the null).
    """
    import torch

    A = _anchor_rows(E_a, ids_a)
    B = _anchor_rows(E_b, ids_b)
    if perm is not None:
        B = B[torch.as_tensor(perm)]
    if center:
        A = A - A.mean(dim=0, keepdim=True)
        B = B - B.mean(dim=0, keepdim=True)
    Ra = relative_repr(A, A)          # [n, n]
    Rb = relative_repr(B, B)          # [n, n]
    Ra = Ra / Ra.norm(dim=1, keepdim=True).clamp_min(1e-12)
    Rb = Rb / Rb.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return float((Ra * Rb).sum(dim=1).mean().item())


def relative_agreement_with_null(E_a, E_b, ids_a, ids_b, *, center: bool = True,
                                 seed: int = 0) -> Dict[str, float]:
    """``{"real", "null"}``: centered relative agreement and its shuffled-anchor null."""
    import torch

    n = len(list(ids_a))
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    return {
        "real": relative_agreement(E_a, E_b, ids_a, ids_b, center=center),
        "null": relative_agreement(E_a, E_b, ids_a, ids_b, center=center, perm=perm),
    }


# --- black-box extraction (query-only local oracle) -------------------------

class LocalLogitOracle:
    """Query-only view of a *local* model: last-token logits, nothing else.

    Enforces the scope of this module -- the estimator sees only what a black box
    would return. ``assert_local`` refuses a hosted/Ollama id at construction, so
    there is no way to point this at a third-party production API.
    """

    def __init__(self, model, tokenizer, model_id: str = "local"):
        from vivasecuris.aiasylum.weights.redteam import assert_local

        if model_id and model_id != "local":
            assert_local(model_id)
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.n_queries = 0

    def logits(self, prompts: Sequence[str], *, batch_size: int = 8, max_length: int = 64):
        """Last-token logits ``[len(prompts), V]`` (float32, CPU)."""
        import torch

        from vivasecuris.aiasylum.weights.capture import format_prompts

        texts, applied = format_prompts(self.tokenizer, list(prompts))
        side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        out = []
        try:
            for i in range(0, len(texts), batch_size):
                chunk = texts[i:i + batch_size]
                enc = self.tokenizer(chunk, return_tensors="pt", padding=True,
                                     truncation=True, max_length=max_length,
                                     add_special_tokens=not applied).to(self.model.device)
                with torch.no_grad():
                    lg = self.model(**enc).logits[:, -1, :]
                out.append(lg.float().cpu())
                self.n_queries += len(chunk)
        finally:
            self.tokenizer.padding_side = side
        return torch.cat(out, dim=0)


def collect_logits(oracle: "LocalLogitOracle", prompts: Sequence[str], *,
                   col_subset: Optional[int] = 4096, must_include: Optional[Sequence[int]] = None,
                   seed: int = 0, batch_size: int = 8):
    """Gather a ``[N, V']`` logit matrix, optionally on a random column subset.

    The row-space dimension is ``d`` regardless of which columns are kept, so a
    subset of ``V' >> d`` columns recovers ``d`` and the subspace while keeping
    the matrix small (full ``V`` is 50k-150k wide). ``must_include`` forces those
    column indices to be present (used by recon so anchor tokens are recoverable).
    Returns ``(L, cols)`` where ``cols`` is the sorted kept indices (or ``None``
    when the full width was used).
    """
    import torch

    L = oracle.logits(prompts, batch_size=batch_size)     # [N, V]
    V = L.shape[1]
    if col_subset is None or col_subset >= V:
        return L, None
    g = torch.Generator().manual_seed(seed)
    keep = set(int(i) for i in (must_include or []))
    extra = [int(i) for i in torch.randperm(V, generator=g).tolist() if int(i) not in keep]
    need = max(0, col_subset - len(keep))
    keep_list = sorted(keep | set(extra[:need]))
    cols = torch.as_tensor(keep_list)
    return L[:, cols], cols


def recover_token_embeddings(L, d):
    """Per-column recovered embeddings ``[V', d]`` (token ``j`` = column ``j``).

    From centered ``L = M @ Vh`` with ``Vh = [d, V']``, column ``j`` of ``Vh`` is
    the recovered unembedding vector for the token at column ``j`` -- up to the
    same ``d x d`` transform for every token, which cross-model alignment then
    resolves.
    """
    return recover_subspace(L, d).t()                     # [V', d]


def recover_dimension(L, *, max_dim: Optional[int] = None) -> Tuple[int, List[float]]:
    """Recover the hidden size ``d`` from a centered logit matrix's spectrum.

    Rows of ``L`` lie in a ``d``-dim affine subspace, so centering removes the
    bias and the singular-value spectrum drops sharply after ``d``. ``d`` is the
    index of the largest relative gap ``s[i]/s[i+1]`` in the leading spectrum.
    """
    import torch

    Lc = _analysis_tensor(L)
    Lc = Lc - Lc.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(Lc)
    spectrum = [float(x) for x in s]
    horizon = min(len(spectrum) - 1, max_dim or len(spectrum) - 1)
    # Look for the knee among positive singular values only.
    eps = spectrum[0] * 1e-9 if spectrum else 0.0
    ratios = []
    for i in range(horizon):
        nxt = spectrum[i + 1]
        ratios.append(spectrum[i] / nxt if nxt > eps else float("inf"))
    d = int(max(range(len(ratios)), key=lambda i: ratios[i]) + 1) if ratios else len(spectrum)
    return d, spectrum


def recover_subspace(L, d: int):
    """Top-``d`` right singular vectors of centered ``L``: a basis for ``W_U``'s row-space.

    Returns ``[d, V']`` orthonormal rows spanning the recovered output-embedding
    directions (up to a ``d x d`` transform vs. the true ``W_U``).
    """
    import torch

    Lc = _analysis_tensor(L)
    Lc = Lc - Lc.mean(dim=0, keepdim=True)
    _, _, Vh = torch.linalg.svd(Lc, full_matrices=False)
    return Vh[:d]


def subspace_principal_angles(basis_a, basis_b) -> List[float]:
    """Principal angles (degrees) between two row-spaces given as orthonormal ``[k, V]``."""
    import math

    import torch

    A = _analysis_tensor(basis_a)
    B = _analysis_tensor(basis_b)
    A = torch.linalg.qr(A.t(), mode="reduced")[0]     # [V, k]
    B = torch.linalg.qr(B.t(), mode="reduced")[0]
    s = torch.linalg.svdvals(A.t() @ B).clamp(-1.0, 1.0)
    return [math.degrees(math.acos(float(x))) for x in s]


def score_recovery(recovered_basis, true_W_U, cols=None) -> Dict[str, Any]:
    """Compare a recovered subspace to the true ``W_U`` (evaluation only).

    Logits are ``L = H @ W_U.T + 1 b.T``; centering over queries leaves
    ``(H - H_mean) @ W_U.T``, whose rows lie in the **column space of W_U** in
    ``R^V``. So the reference basis is ``W_U``'s left singular vectors
    (restricted to ``cols`` when recovery used a column subset), uncentered.
    ``subspace_overlap`` = mean cosine of the principal angles (1 = exact).
    """
    import math

    import torch

    W = _analysis_tensor(true_W_U)
    if cols is not None:
        W = W[cols]                                   # [V', d]
    # Orthonormal basis for W_U's column space in R^V': left singular vectors.
    U = torch.linalg.svd(W, full_matrices=False)[0]   # [V', d]
    d = recovered_basis.shape[0]
    true_basis = U.t()[:d]                            # [<=d, V']
    angles = subspace_principal_angles(recovered_basis, true_basis)
    overlap = float(sum(math.cos(math.radians(a)) for a in angles) / len(angles)) if angles else 0.0
    return {
        "principal_angle_deg": (sum(angles) / len(angles)) if angles else None,
        "max_angle_deg": max(angles) if angles else None,
        "subspace_overlap": overlap,
    }


# A small English word bank; random short phrases from it give diverse last-token
# hidden states, which is what makes the logit matrix span the model's subspace.
_WORD_BANK = (
    "the a of and to in is was he for it with as his on be at by had not are but "
    "from or her she which you one we all their there when who will more if no out "
    "so up said what about them then would other into has two time could my than "
    "water people first over new sound take only little work know place year live "
    "back give most very after thing our just name good sentence man think say great "
    "where help through much before line right too mean old any same tell boy follow "
    "came want show also around form three small set put end does another well large "
    "must big even such because turn here why ask went men read need land different "
    "home us move try kind hand picture again change off play spell air away animal "
    "house point page letter mother answer found study still learn should world high"
).split()


def diverse_prompts(n: int, seed: int = 0, min_words: int = 3, max_words: int = 9) -> List[str]:
    """``n`` short, varied phrases from a fixed word bank (deterministic per seed)."""
    import random

    rng = random.Random(seed)
    out, seen = [], set()
    tries = 0
    while len(out) < n and tries < n * 20:
        tries += 1
        k = rng.randint(min_words, max_words)
        phrase = " ".join(rng.choice(_WORD_BANK) for _ in range(k))
        if phrase not in seen:
            seen.add(phrase)
            out.append(phrase)
    return out
