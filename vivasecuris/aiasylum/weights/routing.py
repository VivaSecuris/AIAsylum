"""Which experts fire, on which prompts.

A mixture-of-experts block routes each token to ``top_k`` of its experts. If a
behaviour such as refusal is carried by particular experts, they are selected
more often on prompts that elicit it than on matched prompts that do not. This
module measures exactly that: per layer and per expert, the fraction of tokens
whose top-k contains the expert on a harmful and on a harmless prompt set, and
the difference between the two. The last prompt position is reported on its
own, because that is where the model has committed to answering or refusing.

Two independent readings are taken and compared. The gate's output is replayed
through the same top-k the block performs, and every expert module gets a
pre-hook that counts the rows it actually received. When the two disagree the
result says so (``consistency.gate_vs_expert_counts_match``) rather than hiding
it: it means the replayed selection is wrong for this family and the fractions
should not be trusted.

Nothing here edits anything. The output is what an expert-selective edit
(``weights.surgery.ablate_experts`` and friends) is aimed with.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence

from vivasecuris.aiasylum.interp.core.arch import (
    _gate_module,
    detect_architecture,
    describe_architecture,
    get_layer_stack,
    moe_layout,
)
from vivasecuris.aiasylum.weights.capture import format_prompts

logger = logging.getLogger(__name__)


class RoutingCapture:
    """Hooks that record, per MoE layer, the top-k selection and the rows each expert saw.

    Use as a context manager around forward passes; read ``selections``,
    ``weights`` and ``expert_rows`` after each pass and call :meth:`reset`
    between prompts.
    """

    def __init__(self, model):
        import torch  # noqa: F401  (torch is needed by the hooks below)

        self.model = model
        arch = detect_architecture(model)
        self.layout = moe_layout(model, arch)
        if not self.layout:
            raise ValueError(f"{describe_architecture(model).label} has no mixture-of-experts layers.")
        self._mlps = {idx: mlp for idx, _attn, mlp in get_layer_stack(model, arch)}
        self._handles: List[Any] = []
        self.selections: Dict[int, Any] = {}
        self.weights: Dict[int, Any] = {}
        self.expert_rows: Dict[int, List[int]] = {}
        self.reset()

    def reset(self) -> None:
        self.selections = {}
        self.weights = {}
        self.expert_rows = {layer: [0] * info.n_experts for layer, info in self.layout.items()}

    def __enter__(self) -> "RoutingCapture":
        for layer, info in self.layout.items():
            mlp = self._mlps[layer]
            gate = _gate_module(mlp)
            if gate is not None:
                self._handles.append(gate.register_forward_hook(self._gate_hook(layer, info, mlp)))
            experts = getattr(mlp, "experts", None)
            if info.experts_are_modules:
                for e, expert in enumerate(experts):
                    self._handles.append(expert.register_forward_pre_hook(self._expert_hook(layer, e)))
        return self

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def _expert_hook(self, layer: int, e: int) -> Callable:
        def hook(_module, args):
            if args and hasattr(args[0], "shape"):
                self.expert_rows[layer][e] += int(args[0].reshape(-1, args[0].shape[-1]).shape[0])
        return hook

    def _gate_hook(self, layer: int, info, mlp) -> Callable:
        def hook(_module, _inputs, output):
            import torch

            if isinstance(output, tuple):
                # DeepSeek / GLM4-MoE routers return (topk_indices, topk_weights).
                idx = output[0]
                wt = output[1] if len(output) > 1 else None
                sel = idx.reshape(-1, idx.shape[-1]).long()
                w = wt.reshape(-1, wt.shape[-1]).float() if wt is not None else None
            else:
                # A Linear gate: replay the block's own softmax + top-k.
                logits = output.reshape(-1, output.shape[-1]).float()
                scores = torch.softmax(logits, dim=-1)
                statics = getattr(mlp, "moe_statics", None)
                bias = getattr(statics, "e_score_correction_bias", None)
                if bias is not None:
                    # Ernie4.5-MoE adds a learned per-expert correction before top-k.
                    scores = scores + bias.reshape(-1).float().to(scores.device)
                k = info.top_k or scores.shape[-1]
                w, sel = torch.topk(scores, min(k, scores.shape[-1]), dim=-1)
            self.selections[layer] = sel.detach().cpu()
            self.weights[layer] = w.detach().cpu() if w is not None else None
        return hook


def _class_stats(n_experts: int) -> Dict[str, Any]:
    return {
        "tokens": 0,
        "prompts": 0,
        "selected": [0] * n_experts,
        "weight_sum": [0.0] * n_experts,
        "last_selected": [0] * n_experts,
        "expert_rows": [0] * n_experts,
    }


def _accumulate(stats: Dict[str, Any], capture: RoutingCapture, layer: int) -> None:
    import torch

    sel = capture.selections.get(layer)
    if sel is None:
        return
    n = len(stats["selected"])
    stats["tokens"] += int(sel.shape[0])
    stats["prompts"] += 1
    w = capture.weights.get(layer)
    for e in range(n):
        hit = (sel == e)
        stats["selected"][e] += int(hit.any(dim=-1).sum())
        if w is not None:
            stats["weight_sum"][e] += float(w[hit].sum())
        if bool((sel[-1] == e).any()):
            stats["last_selected"][e] += 1
        stats["expert_rows"][e] += capture.expert_rows[layer][e]


def _finish(stats: Dict[str, Any]) -> Dict[str, Any]:
    tokens = max(stats["tokens"], 1)
    prompts = max(stats["prompts"], 1)
    n = len(stats["selected"])
    return {
        "tokens": stats["tokens"],
        "prompts": stats["prompts"],
        "selection_frac": [stats["selected"][e] / tokens for e in range(n)],
        "mean_weight": [
            (stats["weight_sum"][e] / stats["selected"][e]) if stats["selected"][e] else 0.0
            for e in range(n)
        ],
        "last_token_frac": [stats["last_selected"][e] / prompts for e in range(n)],
        "expert_rows": list(stats["expert_rows"]),
        "selected": list(stats["selected"]),
    }


def routing_statistics(
    model,
    tokenizer,
    harmful: Sequence[str],
    harmless: Sequence[str],
    *,
    max_length: int = 512,
    thinking: bool = False,
    system_prompt: Optional[str] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, Any]:
    """Per-layer, per-expert selection fractions on two prompt sets, and their difference.

    Prompts are wrapped in the model's chat template exactly as capture and
    generation wrap them, and run one at a time so every recorded position is a
    real token. ``selection_frac[e]`` is the fraction of tokens whose top-k
    contains expert ``e`` (it sums to ``top_k`` over experts);
    ``last_token_frac[e]`` is the fraction of prompts whose final position
    selected it. ``ranking`` orders (layer, expert) pairs by the absolute
    difference between the two sets.
    """
    import torch

    info = describe_architecture(model)
    capture = RoutingCapture(model)
    layout = capture.layout
    per_class: Dict[str, Dict[int, Dict[str, Any]]] = {
        "harmful": {layer: _class_stats(li.n_experts) for layer, li in layout.items()},
        "harmless": {layer: _class_stats(li.n_experts) for layer, li in layout.items()},
    }

    device = next(model.parameters()).device
    sets = [("harmful", list(harmful)), ("harmless", list(harmless))]
    total = sum(len(p) for _, p in sets)
    done = 0
    model.eval()
    with capture, torch.no_grad():
        for label, prompts in sets:
            formatted = format_prompts(tokenizer, prompts, system_prompt=system_prompt, thinking=thinking)
            for text in formatted:
                enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
                capture.reset()
                model(**{k: v.to(device) for k, v in enc.items()}, use_cache=False)
                for layer in layout:
                    _accumulate(per_class[label][layer], capture, layer)
                done += 1
                if progress is not None:
                    progress(done, total)

    layers: List[Dict[str, Any]] = []
    ranking: List[Dict[str, Any]] = []
    consistent = True
    for layer, li in sorted(layout.items()):
        h = _finish(per_class["harmful"][layer])
        b = _finish(per_class["harmless"][layer])
        # The gate replay and the experts' own row counts must agree, or the
        # replayed selection is wrong for this family.
        if li.experts_are_modules and (h["selected"] != h["expert_rows"] or b["selected"] != b["expert_rows"]):
            consistent = False
        delta = [h["selection_frac"][e] - b["selection_frac"][e] for e in range(li.n_experts)]
        delta_last = [h["last_token_frac"][e] - b["last_token_frac"][e] for e in range(li.n_experts)]
        layers.append({
            "layer": layer, "n_experts": li.n_experts, "top_k": li.top_k,
            "gate_kind": li.gate_kind, "has_shared": li.has_shared,
            "harmful": {k: v for k, v in h.items() if k not in ("expert_rows", "selected")},
            "harmless": {k: v for k, v in b.items() if k not in ("expert_rows", "selected")},
            "delta_frac": delta, "delta_last_token": delta_last,
        })
        for e in range(li.n_experts):
            ranking.append({
                "layer": layer, "expert": e,
                "harmful_frac": h["selection_frac"][e], "harmless_frac": b["selection_frac"][e],
                "delta": delta[e], "last_token_delta": delta_last[e],
            })
    ranking.sort(key=lambda r: abs(r["delta"]), reverse=True)

    top_k = next((li.top_k for li in layout.values() if li.top_k), None)
    result = {
        "model_type": info.model_type,
        "architecture": info.family,
        "n_layers": int(getattr(model.config, "num_hidden_layers", len(layers))),
        "moe_layers": len(layers),
        "top_k": top_k,
        "prompts": {"harmful": len(harmful), "harmless": len(harmless)},
        "max_length": max_length,
        "thinking": thinking,
        "layers": layers,
        "ranking": ranking[:100],
        "consistency": {"gate_vs_expert_counts_match": consistent},
    }
    if not consistent:
        logger.warning(
            "Routing replay disagrees with the experts' own row counts for %s; "
            "selection fractions are not trustworthy for this family.", info.label,
        )
    return result


def routing_table(result: Dict[str, Any], top: int = 20) -> List[Dict[str, Any]]:
    """The ``top`` most differentially-selected experts, for a terminal or a UI list."""
    return list(result.get("ranking", [])[:top])
