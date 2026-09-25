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
    expert_down_matrices,
    residual_write_plan,
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


def remove_subspace_component(W, basis, k: float, kind: str, weights=None):
    """Return ``W`` with ``k * w_j`` times its projection onto each basis row subtracted.

    ``weights`` (``[m]``, default all ones) is the per-direction strength of the
    eigenvalue-weighted soft ablation ``W <- W - sum_j (mu_j/mu_1) v_j v_j^T W``
    from RFM-AGOP (arXiv 2607.02396). With uniform weights this is the plain
    subspace projection documented below.

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
    if weights is not None:
        w = torch.as_tensor([float(x) for x in weights], dtype=torch.float32)[: B.shape[0]]
        if w.shape[0] != B.shape[0]:
            raise ValueError(f"{B.shape[0]} basis rows but {w.shape[0]} weights")
        Bw = B * w.unsqueeze(1)             # scale each row's contribution
    else:
        Bw = B

    if kind == KIND_OUT:
        if W32.shape[0] != B.shape[1]:
            raise ValueError(
                f"Shape mismatch for kind={kind}: W is {tuple(W32.shape)} but basis rows "
                f"have {B.shape[1]} elements; expected W.shape[0] to match."
            )
        update = Bw.t() @ (B @ W32)         # [d_model, d_in] = sum_j w_j v_j (v_j^T W)
    else:
        if W32.shape[1] != B.shape[1]:
            raise ValueError(
                f"Shape mismatch for kind={kind}: W is {tuple(W32.shape)} but basis rows "
                f"have {B.shape[1]} elements; expected W.shape[1] to match."
            )
        update = (W32 @ B.t()) @ Bw         # [vocab, d_model]

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


def apply_subspace_to_matrices(matrices: List[WriteMatrix], basis, k: float, weights=None) -> Dict[str, float]:
    """Apply the subspace edit in place to every matrix. Returns per-matrix relative change."""
    import torch

    deltas: Dict[str, float] = {}
    with torch.no_grad():
        for wm in matrices:
            before = wm.param.data
            after = remove_subspace_component(before, basis, k, wm.kind, weights=weights)
            denom = before.to(torch.float32).norm().item() or 1.0
            deltas[wm.name] = (after.to(torch.float32) - before.to(torch.float32)).norm().item() / denom
            wm.param.data.copy_(after)
    return deltas


def _plan_fields(plan) -> Dict[str, object]:
    """What the enumeration proved, for the manifest: the MoE shape and coverage."""
    return {
        "model_type": plan.model_type,
        "is_moe": plan.is_moe,
        "n_layers": plan.n_layers,
        "moe_layers": plan.moe_layers,
        "expert_matrices": plan.expert_matrices,
        "shared_expert_matrices": plan.shared_expert_matrices,
        "coverage_verified": plan.coverage_verified,
    }


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
    plan = residual_write_plan(model, arch, include_embeddings=include_embeddings)
    matrices = plan.matrices

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
        **_plan_fields(plan),
    }


def apply_subspace_to_model(
    model,
    basis,
    k: float = 1.0,
    include_embeddings: bool = True,
    weights=None,
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
    plan = residual_write_plan(model, arch, include_embeddings=include_embeddings)
    matrices = plan.matrices

    if tied and include_embeddings:
        logger.warning(
            "lm_head is tied to embed_tokens; editing the embedding table also "
            "changes the unembedding. Recorded in the manifest."
        )

    rank = int(basis.shape[0])
    deltas = apply_subspace_to_matrices(matrices, basis, k, weights=weights)
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
        "weights": ([float(w) for w in weights][:rank] if weights is not None else None),
        "embeddings_tied": tied,
        "embeddings_edited": include_embeddings,
        "matrices_edited": len(matrices),
        "mean_relative_change": mean_delta,
        "per_matrix_relative_change": deltas,
        **_plan_fields(plan),
    }


# --------------------------------------------------------------------------
# Expert-selective edits (partial by design)
# --------------------------------------------------------------------------
#
# The functions above edit every residual writer, which is what makes a
# direction removal exact. These edit only the down-projections of *chosen*
# experts in chosen MoE layers, to ask what those experts contribute. The
# result is partial on purpose and its manifest says so (coverage_verified is
# False): the untouched experts still write the direction whenever the router
# picks them. The router itself is never edited; see arch.py for the reasoning.


def scale_matrices(matrices: List[WriteMatrix], scale: float) -> Dict[str, float]:
    """Multiply every matrix by ``scale`` in place (0 zeroes it). Returns per-matrix relative change."""
    import torch

    deltas: Dict[str, float] = {}
    with torch.no_grad():
        for wm in matrices:
            before = wm.param.data
            after = (before.to(torch.float32) * float(scale)).to(before.dtype)
            denom = before.to(torch.float32).norm().item() or 1.0
            deltas[wm.name] = (after.to(torch.float32) - before.to(torch.float32)).norm().item() / denom
            wm.param.data.copy_(after)
    return deltas


def _check_width(model, width: int, what: str) -> None:
    d_model = getattr(model.config, "hidden_size", None)
    if d_model is not None and width != d_model:
        raise ValueError(
            f"{what} width {width} does not match this model's hidden size {d_model}. "
            f"It was almost certainly derived from a different model -- derive one "
            f"against this model before editing it."
        )


def _expert_summary(model, plan, deltas: Dict[str, float], mode: str, **extra) -> Dict[str, object]:
    mean_delta = sum(deltas.values()) / len(deltas) if deltas else 0.0
    return {
        "architecture": plan.arch.family,
        "model_type": plan.arch.model_type,
        "is_moe": True,
        "coverage_verified": False,
        "expert_mode": mode,
        "expert_selection": {str(layer): list(experts) for layer, experts in plan.selection.items()},
        "include_shared": plan.include_shared,
        "layers_edited": plan.layers_edited,
        "experts_edited": plan.experts_edited,
        "shared_edited": plan.shared_edited,
        "moe_layers": plan.moe_layers,
        "expert_matrices": plan.experts_edited,
        "shared_expert_matrices": plan.shared_edited,
        "moe_layout": {str(layer): info.n_experts for layer, info in plan.layout.items()},
        "matrices_edited": len(plan.matrices),
        "embeddings_tied": embeddings_are_tied(model),
        "embeddings_edited": False,
        "mean_relative_change": mean_delta,
        "per_matrix_relative_change": deltas,
        **extra,
    }


def apply_direction_to_experts(
    model, r, beta: float, selection, include_shared: bool = False
) -> Dict[str, object]:
    """Scale the ``r`` component of the selected experts' down-projections by ``beta``."""
    _check_width(model, int(r.flatten().shape[0]), "Direction")
    plan = expert_down_matrices(model, selection, include_shared=include_shared)
    deltas = apply_to_matrices(plan.matrices, r, beta)
    logger.info(
        "Applied beta=%.3f to %d expert matrices in %d layers (%s)",
        beta, len(plan.matrices), plan.layers_edited, plan.arch.label,
    )
    return _expert_summary(model, plan, deltas, "direction", beta=float(beta))


def apply_subspace_to_experts(
    model, basis, k: float, selection, weights=None, include_shared: bool = False
) -> Dict[str, object]:
    """Remove an orthonormal subspace from the selected experts' down-projections."""
    _check_width(model, int(basis.shape[-1]), "Subspace")
    plan = expert_down_matrices(model, selection, include_shared=include_shared)
    rank = int(basis.shape[0])
    deltas = apply_subspace_to_matrices(plan.matrices, basis, k, weights=weights)
    logger.info(
        "Removed rank-%d subspace (k=%.3f) from %d expert matrices in %d layers (%s)",
        rank, k, len(plan.matrices), plan.layers_edited, plan.arch.label,
    )
    return _expert_summary(
        model, plan, deltas, "subspace",
        subspace_rank=rank, k=float(k),
        weights=([float(w) for w in weights][:rank] if weights is not None else None),
    )


def ablate_experts(model, selection, scale: float = 0.0, include_shared: bool = False) -> Dict[str, object]:
    """Scale the selected experts' down-projections by ``scale`` (0 removes their write entirely).

    Routing is left exactly as it was: a token still pays the router's weight for
    a zeroed expert, so its MLP output shrinks rather than being redistributed.
    """
    plan = expert_down_matrices(model, selection, include_shared=include_shared)
    deltas = scale_matrices(plan.matrices, scale)
    logger.info(
        "Scaled %d expert matrices in %d layers by %.3f (%s)",
        len(plan.matrices), plan.layers_edited, scale, plan.arch.label,
    )
    return _expert_summary(model, plan, deltas, "ablate", expert_scale=float(scale))


def _edit_experts(model, direction, selection, mode, scale, include_shared, beta, k, step):
    """Dispatch for edit_and_save's expert-selective path. Returns (summary, method)."""
    if mode == "ablate":
        with step(f"scaling the selected experts' down-projections by {scale}"):
            summary = ablate_experts(model, selection, scale=scale, include_shared=include_shared)
        return summary, "expert_ablate"
    if direction is None:
        raise ValueError(
            "expert_mode 'direction' and 'subspace' need a direction; use 'ablate' to "
            "edit experts without one."
        )
    if mode == "subspace":
        basis = direction.as_basis() if hasattr(direction, "as_basis") else direction
        strength = 1.0 if k is None else float(k)
        rank = int(basis.shape[0])
        weights = direction.as_weights(rank) if hasattr(direction, "as_weights") else None
        with step(f"removing the rank-{rank} subspace (k={strength}) from the selected experts"):
            summary = apply_subspace_to_experts(
                model, basis, k=strength, selection=selection, weights=weights,
                include_shared=include_shared,
            )
        return summary, "expert_subspace"
    if mode == "direction":
        vector = getattr(direction, "vector", direction)
        with step(f"scaling the direction (beta={beta}) in the selected experts"):
            summary = apply_direction_to_experts(
                model, vector, beta=beta, selection=selection, include_shared=include_shared,
            )
        return summary, "expert_direction_scale"
    raise ValueError(f"expert_mode must be direction, subspace or ablate, got {mode!r}")


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
    *,
    expert_selection=None,
    expert_mode: str = "direction",
    expert_scale: float = 0.0,
    include_shared: bool = False,
) -> str:
    """Load ``source_model``, apply the direction edit, and save a new model directory.

    Three modes:

    - **single direction** (default): scale the ``direction.vector`` component by
      ``beta`` (0 ablates, 1 no-op, >1 amplifies, <0 over-projects).
    - **subspace** (``use_subspace=True``): remove ``direction``'s orthonormal
      ``basis`` with strength ``k`` (defaults to 1.0). ``rank=1, k=1`` is
      identical to ``beta=0``.
    - **expert-selective** (``expert_selection`` given): edit only the chosen
      experts of the chosen MoE layers, with ``expert_mode`` ``direction``
      (scale by ``beta``), ``subspace`` (remove at ``k``) or ``ablate`` (scale
      the whole down-projection by ``expert_scale``; ``direction`` may be
      ``None``). Partial by design; the manifest records ``coverage_verified``
      as False.

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

    if expert_selection is not None:
        summary, method = _edit_experts(
            model, direction, expert_selection, expert_mode, expert_scale,
            include_shared, beta, k, step,
        )
    elif use_subspace:
        basis = direction.as_basis() if hasattr(direction, "as_basis") else direction
        strength = 1.0 if k is None else float(k)
        rank = int(basis.shape[0])
        weights = direction.as_weights(rank) if hasattr(direction, "as_weights") else None
        with step(f"removing rank-{rank} refusal subspace (k={strength})"):
            summary = apply_subspace_to_model(
                model, basis, k=strength, include_embeddings=include_embeddings, weights=weights,
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
    if expert_selection is not None:
        extra = {
            key: summary.get(key) for key in (
                "expert_selection", "expert_mode", "expert_scale", "include_shared",
                "layers_edited", "experts_edited", "shared_edited", "moe_layout",
            )
        }
        if summary["expert_mode"] == "subspace":
            extra.update({
                "subspace_rank": summary["subspace_rank"],
                "k": summary["k"],
                "weights": summary.get("weights"),
                "derivation": getattr(direction, "method", "diff_in_means"),
            })
    elif use_subspace:
        extra = {
            "subspace_rank": summary["subspace_rank"],
            "k": summary["k"],
            "basis_layers": list(getattr(direction, "basis_layers", []) or []),
            "weights": summary.get("weights"),
            "derivation": getattr(direction, "method", "diff_in_means"),
        }

    SurgeryManifest(
        source_model=source_model,
        method=method,
        beta=float(beta) if method in ("direction_scale", "expert_direction_scale") else None,
        direction_layer=getattr(direction, "layer", None),
        direction_auc=getattr(direction, "auc", None),
        split_hash=getattr(direction, "split_hash", None),
        architecture=summary["architecture"],
        matrices_edited=summary["matrices_edited"],
        embeddings_tied=summary["embeddings_tied"],
        embeddings_edited=summary["embeddings_edited"],
        mean_relative_change=summary["mean_relative_change"],
        model_type=summary.get("model_type"),
        coverage_verified=summary.get("coverage_verified"),
        moe_layers=summary.get("moe_layers"),
        expert_matrices=summary.get("expert_matrices"),
        shared_expert_matrices=summary.get("shared_expert_matrices"),
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
    capability=None,
) -> Dict[str, object]:
    """Search subspace rank x strength for the most-compliant capability-safe edit.

    ``capability`` is a :class:`evaluate.CapabilitySet`; ``None`` means the
    built-in control, trimmed to ``factual_limit`` questions.

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

    from vivasecuris.aiasylum.weights.evaluate import CapabilitySet

    basis = direction.as_basis()
    max_rank = int(basis.shape[0])
    ranks = tuple(sorted({r for r in ranks if 1 <= r <= max_rank}))
    if not ranks:
        ranks = (max_rank,)
    if capability is None:
        capability = CapabilitySet(
            "builtin", capability_questions(limit=factual_limit), factual_accuracy, 32
        )
    factual_qs = capability.questions

    def _note(msg):
        if progress:
            progress(msg)

    # Unedited baseline, once: the capability floor is relative to this.
    _note("baseline (no edit)")
    base_harm = generate_greedy(model, tokenizer, harmful_prompts, max_new_tokens=max_new_tokens)
    base_fac = generate_greedy(model, tokenizer, factual_qs, max_new_tokens=capability.max_new_tokens)
    baseline = {
        "refuse_harmful": refusal_rate(base_harm),
        "factual_acc": capability.score(base_fac),
    }
    floor = baseline["factual_acc"] - factual_floor

    frontier: List[dict] = []
    for r in ranks:
        sub = basis[:r]
        for k in ks:
            _note(f"rank={r} k={k}")
            with ablate_subspace(model, sub, k=k, weights=direction.as_weights(r) if hasattr(direction, 'as_weights') else None):
                harm = generate_greedy(model, tokenizer, harmful_prompts, max_new_tokens=max_new_tokens)
                fac = generate_greedy(model, tokenizer, factual_qs, max_new_tokens=capability.max_new_tokens)
            refuse = refusal_rate(harm)
            fac_acc = capability.score(fac)
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

    return {"baseline": baseline, "frontier": frontier, "best": best,
            "capability_set": capability.name, "capability_n": capability.size}
