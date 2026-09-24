"""Permanent weight edits along a single residual-stream direction.

One scalar controls everything. For a unit direction ``r`` and a matrix that
writes into the residual stream, the edit rescales only the component along
``r`` and leaves the orthogonal complement untouched::

    beta = 0  ->  ablate  (remove the direction entirely)
    beta = 1  ->  no-op   (bit-identical)
    beta > 1  ->  amplify (strengthen whatever that direction encodes)

Layout matters. ``torch.nn.Linear`` stores ``weight`` as ``[out, in]`` and
computes ``y = x @ W.T``, so the two matrix kinds need different forms:

``KIND_OUT`` (``o_proj``, ``down_proj``) -- the output dimension is the
residual dimension, indexed by axis 0::

    W' = W + (beta - 1) * r (r^T W)
    =>  y' = y + (beta - 1) (y . r) r          exactly

``KIND_EMBED`` (``embed_tokens``) -- there is no matmul; each *row* is itself a
residual-space vector, so the component is removed per row::

    W' = W + (beta - 1) * (W r) r^T
    =>  w_i' = w_i + (beta - 1) (w_i . r) r    for every row i

Arithmetic runs in float32 regardless of model dtype: accumulating this in
bfloat16 loses most of the edit to rounding.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from vivasecuris.aiasylum.interp.core.arch import (
    KIND_EMBED,
    KIND_OUT,
    WriteMatrix,
    detect_architecture,
    embeddings_are_tied,
    residual_write_matrices,
)

logger = logging.getLogger(__name__)


def scale_direction_component(W, r, beta: float, kind: str):
    """Return ``W`` with its ``r``-component scaled by ``beta``.

    Pure function over tensors -- no model required -- so the projection
    algebra can be property-tested directly. ``r`` is normalized defensively;
    a non-unit direction would scale the edit by ``||r||^2``.
    """
    import torch

    if kind not in (KIND_EMBED, KIND_OUT):
        raise ValueError(f"Unknown matrix kind: {kind!r}")

    orig_dtype = W.dtype
    W32 = W.to(torch.float32)
    r32 = r.to(torch.float32).flatten()
    r32 = r32 / r32.norm()

    coeff = float(beta) - 1.0

    if kind == KIND_OUT:
        # W: [d_model, d_in]; r indexes the output (residual) dimension.
        if W32.shape[0] != r32.shape[0]:
            raise ValueError(
                f"Shape mismatch for kind={kind}: W is {tuple(W32.shape)} but "
                f"direction has {r32.shape[0]} elements; expected W.shape[0] to match."
            )
        # r (r^T W) -- outer product of r with the row-projection of W.
        update = torch.outer(r32, r32 @ W32)
    else:
        # W: [vocab, d_model]; r indexes the row (residual) dimension.
        if W32.shape[1] != r32.shape[0]:
            raise ValueError(
                f"Shape mismatch for kind={kind}: W is {tuple(W32.shape)} but "
                f"direction has {r32.shape[0]} elements; expected W.shape[1] to match."
            )
        update = torch.outer(W32 @ r32, r32)

    return (W32 + coeff * update).to(orig_dtype)


def remove_subspace_component(W, basis, k: float, kind: str):
    """Return ``W`` with ``k`` times its projection onto ``basis`` subtracted.

    ``basis`` is an orthonormal ``[m, d]`` set of residual-space directions
    (rows normalized defensively). The edit removes the whole subspace at once::

        KIND_OUT   (y = x @ W.T):  W' = W - k * Bᵀ (B W)   => y' = y - k · P y
        KIND_EMBED (rows are h):   W' = W - k * (W Bᵀ) B   => wᵢ' = wᵢ - k · P wᵢ

    where ``P = Σ_j r_j r_jᵀ`` is the projector onto the subspace. With a single
    row and ``k = 1 - beta`` this is exactly :func:`scale_direction_component`,
    so ``rank=1, k=1`` reproduces a ``beta=0`` ablation bit-for-bit.

    Arithmetic runs in float32 and casts back: bf16 accumulation loses the edit.
    """
    import torch

    if kind not in (KIND_EMBED, KIND_OUT):
        raise ValueError(f"Unknown matrix kind: {kind!r}")

    orig_dtype = W.dtype
    W32 = W.to(torch.float32)
    B = basis.to(torch.float32)
    if B.dim() == 1:
        B = B.reshape(1, -1)
    B = B / B.norm(dim=1, keepdim=True).clamp_min(1e-12)

    if kind == KIND_OUT:
        if W32.shape[0] != B.shape[1]:
            raise ValueError(
                f"Shape mismatch for kind={kind}: W is {tuple(W32.shape)} but basis rows "
                f"have {B.shape[1]} elements; expected W.shape[0] to match."
            )
        update = B.t() @ (B @ W32)          # [d_model, d_in]
    else:
        if W32.shape[1] != B.shape[1]:
            raise ValueError(
                f"Shape mismatch for kind={kind}: W is {tuple(W32.shape)} but basis rows "
                f"have {B.shape[1]} elements; expected W.shape[1] to match."
            )
        update = (W32 @ B.t()) @ B          # [vocab, d_model]

    return (W32 - float(k) * update).to(orig_dtype)


def apply_to_matrices(matrices: List[WriteMatrix], r, beta: float) -> Dict[str, float]:
    """Apply the single-direction edit in place to every matrix. Returns per-matrix relative change."""
    import torch

    deltas: Dict[str, float] = {}
    with torch.no_grad():
        for wm in matrices:
            before = wm.param.data
            after = scale_direction_component(before, r, beta, wm.kind)
            denom = before.to(torch.float32).norm().item() or 1.0
            deltas[wm.name] = (after.to(torch.float32) - before.to(torch.float32)).norm().item() / denom
            wm.param.data.copy_(after)
    return deltas


def apply_subspace_to_matrices(matrices: List[WriteMatrix], basis, k: float) -> Dict[str, float]:
    """Apply the subspace edit in place to every matrix. Returns per-matrix relative change."""
    import torch

    deltas: Dict[str, float] = {}
    with torch.no_grad():
        for wm in matrices:
            before = wm.param.data
            after = remove_subspace_component(before, basis, k, wm.kind)
            denom = before.to(torch.float32).norm().item() or 1.0
            deltas[wm.name] = (after.to(torch.float32) - before.to(torch.float32)).norm().item() / denom
            wm.param.data.copy_(after)
    return deltas


def apply_to_model(
    model,
    r,
    beta: float,
    include_embeddings: bool = True,
) -> Dict[str, object]:
    """Edit every residual-writing matrix of ``model`` in place.

    Returns a summary suitable for embedding in the surgery manifest.
    """
    d_model = getattr(model.config, "hidden_size", None)
    r_dim = int(r.flatten().shape[0])
    if d_model is not None and r_dim != d_model:
        raise ValueError(
            f"Direction width {r_dim} does not match this model's hidden size {d_model}. "
            f"The direction was almost certainly derived from a different model -- "
            f"derive one against this model before editing it."
        )

    arch = detect_architecture(model)
    tied = embeddings_are_tied(model)
    matrices = residual_write_matrices(model, arch, include_embeddings=include_embeddings)

    if tied and include_embeddings:
        logger.warning(
            "lm_head is tied to embed_tokens; editing the embedding table also "
            "changes the unembedding. Recorded in the manifest."
        )

    deltas = apply_to_matrices(matrices, r, beta)
    mean_delta = sum(deltas.values()) / len(deltas) if deltas else 0.0

    logger.info(
        "Applied beta=%.3f to %d matrices (arch=%s, tied=%s); mean relative change %.4f",
        beta, len(matrices), arch, tied, mean_delta,
    )

    return {
        "architecture": arch,
        "beta": float(beta),
        "embeddings_tied": tied,
        "embeddings_edited": include_embeddings,
        "matrices_edited": len(matrices),
        "mean_relative_change": mean_delta,
        "per_matrix_relative_change": deltas,
    }


def apply_subspace_to_model(
    model,
    basis,
    k: float = 1.0,
    include_embeddings: bool = True,
) -> Dict[str, object]:
    """Remove an orthonormal refusal *subspace* from every residual-writing matrix.

    The subspace analogue of :func:`apply_to_model`. ``basis`` is ``[m, d]`` with
    orthonormal rows; ``k`` is the removal strength (1.0 = project the subspace
    out; >1 over-projects, the multi-direction analogue of a negative ``beta``).
    """
    d_model = getattr(model.config, "hidden_size", None)
    b_dim = int(basis.shape[-1])
    if d_model is not None and b_dim != d_model:
        raise ValueError(
            f"Subspace width {b_dim} does not match this model's hidden size {d_model}. "
            f"The subspace was almost certainly derived from a different model -- "
            f"derive one against this model before editing it."
        )

    arch = detect_architecture(model)
    tied = embeddings_are_tied(model)
    matrices = residual_write_matrices(model, arch, include_embeddings=include_embeddings)

    if tied and include_embeddings:
        logger.warning(
            "lm_head is tied to embed_tokens; editing the embedding table also "
            "changes the unembedding. Recorded in the manifest."
        )

    rank = int(basis.shape[0])
    deltas = apply_subspace_to_matrices(matrices, basis, k)
    mean_delta = sum(deltas.values()) / len(deltas) if deltas else 0.0

    logger.info(
        "Removed rank-%d subspace (k=%.3f) from %d matrices (arch=%s, tied=%s); "
        "mean relative change %.4f",
        rank, k, len(matrices), arch, tied, mean_delta,
    )

    return {
        "architecture": arch,
        "subspace_rank": rank,
        "k": float(k),
        "embeddings_tied": tied,
        "embeddings_edited": include_embeddings,
        "matrices_edited": len(matrices),
        "mean_relative_change": mean_delta,
        "per_matrix_relative_change": deltas,
    }


def edit_and_save(
    source_model: str,
    direction,
    out_dir: str,
    beta: float = 0.0,
    device: str = "auto",
    dtype: str = "bfloat16",
    include_embeddings: bool = True,
    notes: Optional[str] = None,
    reporter=None,
    use_subspace: bool = False,
    k: Optional[float] = None,
) -> str:
    """Load ``source_model``, apply the direction edit, and save a new model directory.

    Two modes:

    - **single direction** (default): scale the ``direction.vector`` component by
      ``beta`` (0 ablates, 1 no-op, >1 amplifies, <0 over-projects).
    - **subspace** (``use_subspace=True``): remove ``direction``'s orthonormal
      ``basis`` with strength ``k`` (defaults to 1.0). ``rank=1, k=1`` is
      identical to ``beta=0``.

    Writes weights, tokenizer and an ``asylum_surgery.json`` manifest, so the
    result is a drop-in Hugging Face directory that the ``transformers``
    provider (and any other tool) can load directly.
    """
    from pathlib import Path

    from vivasecuris.aiasylum.interp.core.loader import load
    from vivasecuris.aiasylum.weights.manifest import SurgeryManifest

    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(
            f"{out} already exists and is not empty. Choose another path or remove it; "
            f"overwriting a model directory in place is not done implicitly."
        )

    # Surgery happens on CPU by default: the edit is a one-shot low-rank update per
    # matrix, so accelerator residency buys nothing and costs memory headroom.
    from contextlib import nullcontext

    step = reporter.step if reporter is not None else (lambda name: nullcontext())

    with step(f"loading {source_model}"):
        model, tokenizer = load(source_model, device=device, dtype=dtype, seed=None)

    if use_subspace:
        basis = direction.as_basis() if hasattr(direction, "as_basis") else direction
        strength = 1.0 if k is None else float(k)
        rank = int(basis.shape[0])
        with step(f"removing rank-{rank} refusal subspace (k={strength})"):
            summary = apply_subspace_to_model(
                model, basis, k=strength, include_embeddings=include_embeddings
            )
        method = "direction_subspace"
    else:
        vector = getattr(direction, "vector", direction)
        with step(f"editing residual-writing matrices (beta={beta})"):
            summary = apply_to_model(model, vector, beta=beta, include_embeddings=include_embeddings)
        method = "direction_scale"

    if reporter is not None:
        reporter.note(
            f"{summary['matrices_edited']} matrices edited, "
            f"mean relative change {summary['mean_relative_change']:.4f}"
        )

    out.mkdir(parents=True, exist_ok=True)
    with step(f"writing weights to {out}"):
        model.save_pretrained(str(out))
        tokenizer.save_pretrained(str(out))

    # Subspace-specific provenance rides `extra` -- no manifest schema change, so
    # it still flows through transformers_local -> ModelResponse.metadata.
    extra: dict = {}
    if use_subspace:
        extra = {
            "subspace_rank": summary["subspace_rank"],
            "k": summary["k"],
            "basis_layers": list(getattr(direction, "basis_layers", []) or []),
        }

    SurgeryManifest(
        source_model=source_model,
        method=method,
        beta=None if use_subspace else float(beta),
        direction_layer=getattr(direction, "layer", None),
        direction_auc=getattr(direction, "auc", None),
        split_hash=getattr(direction, "split_hash", None),
        architecture=summary["architecture"],
        matrices_edited=summary["matrices_edited"],
        embeddings_tied=summary["embeddings_tied"],
        embeddings_edited=summary["embeddings_edited"],
        mean_relative_change=summary["mean_relative_change"],
        notes=notes,
        extra=extra,
    ).save(out)

    if use_subspace:
        logger.info("Saved edited model to %s (subspace rank=%d, k=%.3f)",
                    out, summary["subspace_rank"], summary["k"])
    else:
        logger.info("Saved edited model to %s (beta=%.3f)", out, beta)
    return str(out)


def select_edit(
    model,
    tokenizer,
    direction,
    harmful_prompts: List[str],
    ranks: tuple = (1, 2, 3, 4, 6, 8),
    ks: tuple = (1.0, 1.25, 1.5),
    factual_floor: float = 0.05,
    max_new_tokens: int = 96,
    factual_limit: Optional[int] = None,
    progress: Optional[callable] = None,
) -> Dict[str, object]:
    """Search subspace rank x strength for the most-compliant capability-safe edit.

    Selection *is* the fix. Cranking one direction (beta<0) or removing a large
    subspace both reach 0% refusal only by lobotomizing the model, which a
    phrase-matching detector cannot distinguish from a clean jailbreak. So every
    candidate is previewed at inference time (via :func:`steering.ablate_subspace`,
    no weights written) and scored on both harmful refusal *and* the factual
    capability control; a candidate is admissible only if it stays within
    ``factual_floor`` of the unedited baseline's factual accuracy and is not
    degenerate.

    Returns ``{"baseline", "frontier", "best"}``. ``frontier`` is every (rank, k)
    with its refusal + factual + accepted flag -- the "what 100% costs" curve.
    ``best`` is the admissible config with the lowest refusal (ties broken toward
    the *least* destructive edit: smaller rank, then smaller k, then higher
    factual), or ``None`` if nothing clears the floor.
    """
    from vivasecuris.aiasylum.weights.evaluate import (
        capability_questions,
        factual_accuracy,
        generate_greedy,
    )
    from vivasecuris.aiasylum.weights.steering import (
        ablate_subspace,
        refusal_rate,
        _looks_degenerate,
    )

    basis = direction.as_basis()
    max_rank = int(basis.shape[0])
    ranks = tuple(sorted({r for r in ranks if 1 <= r <= max_rank}))
    if not ranks:
        ranks = (max_rank,)
    factual_qs = capability_questions(limit=factual_limit)

    def _note(msg):
        if progress:
            progress(msg)

    # Unedited baseline, once: the capability floor is relative to this.
    _note("baseline (no edit)")
    base_harm = generate_greedy(model, tokenizer, harmful_prompts, max_new_tokens=max_new_tokens)
    base_fac = generate_greedy(model, tokenizer, factual_qs, max_new_tokens=32)
    baseline = {
        "refuse_harmful": refusal_rate(base_harm),
        "factual_acc": factual_accuracy(base_fac),
    }
    floor = baseline["factual_acc"] - factual_floor

    frontier: List[dict] = []
    for r in ranks:
        sub = basis[:r]
        for k in ks:
            _note(f"rank={r} k={k}")
            with ablate_subspace(model, sub, k=k):
                harm = generate_greedy(model, tokenizer, harmful_prompts, max_new_tokens=max_new_tokens)
                fac = generate_greedy(model, tokenizer, factual_qs, max_new_tokens=32)
            refuse = refusal_rate(harm)
            fac_acc = factual_accuracy(fac)
            degenerate = bool(_looks_degenerate(harm) or _looks_degenerate(fac))
            accepted = (not degenerate) and (fac_acc >= floor)
            frontier.append({
                "rank": r,
                "k": float(k),
                "refuse_harmful": refuse,
                "compliance": 1.0 - refuse,
                "factual_acc": fac_acc,
                "factual_drop": baseline["factual_acc"] - fac_acc,
                "degenerate": degenerate,
                "accepted": accepted,
            })

    admissible = [row for row in frontier if row["accepted"]]
    best = None
    if admissible:
        # Most compliant; ties -> least destructive (smaller rank, smaller k),
        # then higher retained capability.
        best = min(
            admissible,
            key=lambda row: (row["refuse_harmful"], row["rank"], row["k"], -row["factual_acc"]),
        )

    return {"baseline": baseline, "frontier": frontier, "best": best}
