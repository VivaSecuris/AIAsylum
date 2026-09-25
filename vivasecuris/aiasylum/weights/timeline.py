"""Refusal-decision timeline: where in a generation the model commits.

For every generated token, the residual at the direction's layer is projected
onto the refusal direction *at the position that decides that token*. On a
plain chat model the sign is settled at the first answer token. On a reasoning
model the decision can form, flip, or harden inside the ``<think>`` block
(arXiv 2507.03167, 2605.26772), which is exactly what a last-token capture at
the end of the prompt cannot see. The timeline makes the commit point visible
and gives the sweep a way to say "refusal was decided N tokens into the
reasoning" rather than only "refused".

Scores are normalised with the class means recorded when the direction was
derived: ``score = (proj - midpoint) / (mean_harmful - midpoint)`` so ``+1`` is
the average harmful-prompt (refusing) state and ``-1`` the average harmless
one. Without recorded means the projection is z-scored along the trajectory
and no commit index is reported.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from vivasecuris.aiasylum.interp.core.arch import (
    detect_architecture,
    final_hidden_is_normed,
    get_decoder_layers,
    get_final_norm,
)

logger = logging.getLogger(__name__)

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
# Consecutive same-sign tokens before the trajectory counts as committed.
COMMIT_RUN = 3


def _first_tensor(output):
    return output[0] if isinstance(output, tuple) else output


def refusal_timeline(
    model,
    tokenizer,
    prompt: str,
    direction,
    layer: Optional[int] = None,
    max_new_tokens: int = 128,
    thinking: bool = True,
    apply_template: bool = True,
) -> Dict[str, object]:
    """Generate greedily and record the refusal projection per generated token."""
    import torch

    from vivasecuris.aiasylum.weights.capture import format_prompts

    layer = int(direction.layer if layer is None else layer)
    r = direction.vector.to(torch.float32).flatten()
    r = r / r.norm().clamp_min(1e-12)

    arch = detect_architecture(model)
    blocks = get_decoder_layers(model, arch) if arch else None
    if not blocks:
        raise ValueError(f"Could not locate decoder layers (arch={arch})")
    n_blocks = len(blocks)
    if not 0 <= layer <= n_blocks:
        raise ValueError(f"layer {layer} outside 0..{n_blocks}")

    projections: List[float] = []

    def record(hidden):
        if hidden is None or not hasattr(hidden, "shape"):
            return
        h = hidden[0, -1, :].detach().to(torch.float32).cpu()
        projections.append(float((h @ r).item()))

    def pre_hook(_module, args, kwargs):
        hidden = args[0] if args else kwargs.get("hidden_states")
        record(hidden)
        return None

    def post_hook(_module, _args, output):
        record(_first_tensor(output))
        return None

    if layer < n_blocks:
        handle = blocks[layer].register_forward_pre_hook(pre_hook, with_kwargs=True)
    else:
        target = get_final_norm(model) if final_hidden_is_normed(model) else None
        handle = (target or blocks[-1]).register_forward_hook(post_hook)

    text = format_prompts(tokenizer, [prompt], thinking=thinking)[0] if apply_template else prompt
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=not apply_template).to(model.device)
    try:
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
    finally:
        handle.remove()

    gen_ids = out[0][inputs["input_ids"].shape[1]:].tolist()
    tokens = [tokenizer.decode([i]) for i in gen_ids]
    # One projection per forward pass; pass k decides generated token k. A
    # generation that stops early leaves projections == len(tokens).
    projections = projections[: len(tokens)]

    in_think, inside, think_end = [], False, None
    for k, tok in enumerate(tokens):
        if THINK_OPEN in tok:
            inside = True
        in_think.append(inside)
        if THINK_CLOSE in tok:
            inside = False
            think_end = k

    means = (getattr(direction, "extra", {}) or {}).get("projection_means") or {}
    center = float((getattr(direction, "extra", {}) or {}).get("center_projection") or 0.0)
    scores: List[float]
    midpoint = scale = None
    if "harmful" in means and "harmless" in means and means.get("layer", layer) == layer:
        m_h, m_b = float(means["harmful"]), float(means["harmless"])
        midpoint = (m_h + m_b) / 2.0
        scale = (m_h - midpoint) or 1.0
        scores = [((p - center) - midpoint) / scale for p in projections]
        normalisation = "class_means"
    else:
        import statistics

        mu = statistics.fmean(projections) if projections else 0.0
        sd = statistics.pstdev(projections) if len(projections) > 1 else 1.0
        scores = [(p - mu) / (sd or 1.0) for p in projections]
        normalisation = "z_score"

    decision = None
    if normalisation == "class_means":
        for k in range(len(scores) - COMMIT_RUN + 1):
            window = scores[k : k + COMMIT_RUN]
            if all(s > 0 for s in window) or all(s < 0 for s in window):
                decision = k
                break
    tail = scores[-5:] if scores else []
    final_side = None
    if tail and normalisation == "class_means":
        mean_tail = sum(tail) / len(tail)
        final_side = "refuse" if mean_tail > 0 else "comply"
    crossings = sum(1 for a, b in zip(scores, scores[1:]) if (a > 0) != (b > 0))

    return {
        "layer": layer,
        "prompt": prompt,
        "text": tokenizer.decode(gen_ids, skip_special_tokens=True),
        "tokens": tokens,
        "projection": projections,
        "score": scores,
        "in_think": in_think,
        "think_end_index": think_end,
        "decision_index": decision,
        "decision_in_think": (in_think[decision] if decision is not None and decision < len(in_think) else None),
        "final_side": final_side,
        "sign_changes": crossings,
        "normalisation": normalisation,
        "midpoint": midpoint,
        "scale": scale,
    }
