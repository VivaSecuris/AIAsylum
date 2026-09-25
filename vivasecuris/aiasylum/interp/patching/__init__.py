"""Activation patching with real forward passes.

Every patch installs a hook on the target run, re-runs the model, and reads
the effect off the actual output logits. The previous implementation re-added
cached pieces into a single residual and projected it through ``lm_head``
without running the layers above it, so nothing it reported had passed
through the rest of the network. That is not a causal measurement, and the
``claim: causal`` label the API attaches to this analysis was not earned.

Three granularities, all measured the same way:

- ``layer``: replace the residual stream at hidden-state index ``L`` and one
  position with the source run's value. Index ``L < n_blocks`` is the input to
  block ``L`` (a pre-hook); index ``n_blocks`` is the final residual (a
  post-hook on the last block), the same convention as ``weights.steering``.
- ``head``: add ``(source − target)`` of one head's contribution to the
  attention sublayer output at one position. Per-head contributions come from
  :func:`interp.analysis.ov_qk.compute_per_head_outputs`, so attention and
  Q/K/V capture must be on.
- ``neuron``: add ``(source − target)`` of one neuron's contribution to the
  MLP sublayer output at one position. Needs pre-MLP capture (the input to
  the down projection).

Metrics per patch, read at the last position of the target sequence:

- ``logit_diff`` = logit(target's original top token) − logit(source's original
  top token). ``recovered`` is the fraction of the gap between the target run
  and the source run that the patch closes; 1.0 means the patched target now
  predicts exactly like the source at that margin, 0.0 means no effect.
- ``kl_to_source_after`` / ``kl_to_target_after``: where the patched
  distribution sits between the two originals.
- ``cos``/``delta`` before and after: the final residual of the (patched)
  target against the source's final residual. The dashboard plots the
  improvement, which is now "how far the final representation moved toward
  the source".

A null baseline accompanies the layer sweep: at the most effective layer, a
random perturbation of the same norm as the real patch is applied instead.
A real patch has to beat that number to mean anything.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from vivasecuris.aiasylum.interp.core.arch import final_hidden_is_normed, get_decoder_layers, get_final_norm
from vivasecuris.aiasylum.interp.core.config import Config
from vivasecuris.aiasylum.interp.core.hook_registry import _get_layer_stack, detect_architecture
from vivasecuris.aiasylum.interp.data.models import RunResult

logger = logging.getLogger(__name__)


PatchDirection = Literal["A_to_B", "B_to_A"]
PatchMode = Literal["last_token", "multi_position"]
PatchComponents = Literal["layer", "head", "neuron"]

# Default neuron budget when no explicit list is given: the neurons whose
# activation differs most between the runs at the patched position.
DEFAULT_NEURON_BUDGET = 50


@dataclass
class SinglePatchResult:
    """One (layer, position, direction) patch, measured on a real forward pass."""

    layer: int
    position: int
    direction: str

    # Final-residual geometry relative to the source run.
    cos_before: float
    delta_before: float
    cos_after: float
    delta_after: float

    # Next-token predictions: the two originals and the patched target.
    top_a_before: List[str]
    top_b_before: List[str]
    top_target_after: List[str]

    # Causal metrics.
    logit_diff_before: float
    logit_diff_after: float
    logit_diff_source: float
    recovered: float
    recovered_kl: float
    kl_to_target_after: float
    kl_to_source_after: float
    top_flipped: bool


@dataclass
class PatchExperiment:
    """One logical experiment: a component patched across positions."""

    id: str
    description: str
    patch_mode: str
    layer: int
    positions: List[int]
    direction: str
    results: List[SinglePatchResult]
    component: str = "layer"
    component_index: Optional[int] = None


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a_n = F.normalize(a.reshape(1, -1).float(), dim=-1)
    b_n = F.normalize(b.reshape(1, -1).float(), dim=-1)
    return float(torch.sum(a_n * b_n).item())


def _delta_norm(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(a.float() - b.float()).item())


def _kl(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    """KL(p || q) for two logit vectors."""
    p = F.log_softmax(p_logits.float(), dim=-1)
    q = F.log_softmax(q_logits.float(), dim=-1)
    return float((p.exp() * (p - q)).sum().item())


def _topk_tokens(logits: torch.Tensor, tokenizer, k: int) -> List[str]:
    top = torch.topk(logits.float(), k=min(k, logits.shape[-1])).indices
    return [tokenizer.decode([int(i)]) for i in top]


def _get_lm_head_weight(model) -> Optional[torch.Tensor]:
    """Best-effort retrieval of the unembedding matrix (kept for the scrub code)."""
    if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
        return model.lm_head.weight.detach().cpu().float()
    if hasattr(model, "embed_out") and hasattr(model.embed_out, "weight"):
        return model.embed_out.weight.detach().cpu().float()
    return None


def _get_mlp_down(model: Any, layer_idx: int) -> Optional[torch.Tensor]:
    """Return the MLP down projection weight ``[d_model, inter_dim]``."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        mlp = getattr(model.model.layers[layer_idx], "mlp", None)
        if mlp is not None and hasattr(mlp, "down_proj"):
            return mlp.down_proj.weight.detach().cpu().float()
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        mlp = getattr(model.gpt_neox.layers[layer_idx], "mlp", None)
        if mlp is not None and hasattr(mlp, "dense_4h_to_h"):
            return mlp.dense_4h_to_h.weight.detach().cpu().float()
    return None


def compute_per_neuron_mlp_outputs(
    result: RunResult,
    model: Any,
    layer_idx: int,
    start: int,
    window_len: int,
) -> Optional[torch.Tensor]:
    """Per-neuron MLP contribution ``[n_neurons, window_len, d_model]``.

    ``pre_mlp_activations`` holds the input to the down projection, which is
    already ``act(gate) * up`` for gated MLPs, so no activation function is
    applied here.
    """
    pre_mlp = (
        result.pre_mlp_activations.get(layer_idx)
        if getattr(result, "pre_mlp_activations", None) else None
    )
    if pre_mlp is None:
        return None
    down = _get_mlp_down(model, layer_idx)
    if down is None:
        return None
    pre = pre_mlp[0, start : start + window_len, :].float()       # [w, inter]
    output_bytes = pre.shape[0] * pre.shape[1] * down.shape[0] * 4
    if output_bytes > 256 * 1024**2:
        raise ValueError(
            f"Full per-neuron reconstruction needs {output_bytes / 1024**3:.1f} GiB "
            "for this layer. Reduce the analysis window or use neuron patching, "
            "which computes only selected contributions."
        )
    # out[n, t, :] = pre[t, n] * down[:, n]
    return torch.einsum("tn,dn->ntd", pre, down)


def _first_tensor(output: Any) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def _with_first(output: Any, new: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (new,) + tuple(output[1:])
    return new


# --------------------------------------------------------------------------
# Hooks
# --------------------------------------------------------------------------


def _replace_positions_pre_hook(pairs: Sequence[Tuple[int, torch.Tensor]]):
    """Pre-hook on a decoder block: overwrite the residual at given positions."""

    def hook(_module, args, kwargs):
        if args:
            hidden = args[0]
        else:
            hidden = kwargs.get("hidden_states")
        if hidden is None or not hasattr(hidden, "shape"):
            return None
        hidden = hidden.clone()
        for pos, vec in pairs:
            hidden[:, pos, :] = vec.to(device=hidden.device, dtype=hidden.dtype)
        if args:
            return (hidden,) + tuple(args[1:]), kwargs
        kwargs = dict(kwargs)
        kwargs["hidden_states"] = hidden
        return args, kwargs

    return hook


def _replace_positions_post_hook(pairs: Sequence[Tuple[int, torch.Tensor]]):
    """Post-hook on the last block: overwrite the final residual at positions."""

    def hook(_module, _args, output):
        hidden = _first_tensor(output)
        if hidden is None or not hasattr(hidden, "shape"):
            return None
        hidden = hidden.clone()
        for pos, vec in pairs:
            hidden[:, pos, :] = vec.to(device=hidden.device, dtype=hidden.dtype)
        return _with_first(output, hidden)

    return hook


def _add_delta_post_hook(pairs: Sequence[Tuple[int, torch.Tensor]]):
    """Post-hook on a sublayer (attention or MLP): add a delta at positions."""

    def hook(_module, _args, output):
        hidden = _first_tensor(output)
        if hidden is None or not hasattr(hidden, "shape"):
            return None
        hidden = hidden.clone()
        for pos, delta in pairs:
            hidden[:, pos, :] = hidden[:, pos, :] + delta.to(device=hidden.device, dtype=hidden.dtype)
        return _with_first(output, hidden)

    return hook


def _forward(model, input_ids: torch.Tensor, hooks: Sequence[Tuple[Any, str, Callable]]):
    """Run the model with hooks installed; return (last logits, last final hidden)."""
    handles = []
    try:
        for module, kind, fn in hooks:
            if kind == "pre":
                handles.append(module.register_forward_pre_hook(fn, with_kwargs=True))
            else:
                handles.append(module.register_forward_hook(fn))
        with torch.no_grad():
            out = model(
                input_ids=input_ids.to(model.device),
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        return out.logits[0, -1].float().cpu(), out.hidden_states[-1][0, -1].float().cpu()
    finally:
        for h in handles:
            h.remove()


def _final_target(model, blocks):
    """Module whose output is hidden-state index ``n_blocks``.

    When transformers returns the last hidden state after the final norm, the
    cached vector lives in post-norm space and must be written back on the
    norm's output; otherwise it is the last block's output.
    """
    if final_hidden_is_normed(model):
        norm = get_final_norm(model)
        if norm is not None:
            return norm
    return blocks[-1]


def patch_and_run(
    model,
    result_src: RunResult,
    result_tgt: RunResult,
    layer: int,
    pairs: Sequence[Tuple[int, int]],
):
    """Patch the residual at hidden-state index ``layer`` and re-run the target.

    ``pairs`` are ``(source_position, target_position)`` absolute token indices.
    Returns ``(last_logits, last_final_hidden)`` of the patched target run.
    Exposed on its own so a caller can patch a whole window jointly; the
    experiment runner patches positions independently.
    """
    arch = detect_architecture(model)
    blocks = get_decoder_layers(model, arch) if arch else None
    if not blocks:
        raise ValueError(f"Could not locate decoder layers for patching (arch={arch})")
    n_blocks = len(blocks)
    if not 0 <= layer <= n_blocks:
        raise ValueError(f"layer {layer} outside 0..{n_blocks}")
    vec_pairs = [(t, result_src.hidden_states[layer][0, s]) for s, t in pairs]
    if layer < n_blocks:
        hooks = [(blocks[layer], "pre", _replace_positions_pre_hook(vec_pairs))]
    else:
        hooks = [(_final_target(model, blocks), "post", _replace_positions_post_hook(vec_pairs))]
    return _forward(model, result_tgt.input_ids, hooks)


# --------------------------------------------------------------------------
# Experiment runner
# --------------------------------------------------------------------------


class _Pair:
    """Everything fixed about one patch direction, computed once."""

    def __init__(self, direction: str, src: RunResult, tgt: RunResult,
                 start_src: int, start_tgt: int, tokenizer, topk: int):
        self.direction = direction
        self.src, self.tgt = src, tgt
        self.start_src, self.start_tgt = start_src, start_tgt
        self.src_logits = src.logits[0, -1].float()
        self.tgt_logits = tgt.logits[0, -1].float()
        self.src_tok = int(self.src_logits.argmax().item())
        self.tgt_tok = int(self.tgt_logits.argmax().item())
        self.same_top = self.src_tok == self.tgt_tok
        self.ld_before = self.logit_diff(self.tgt_logits)
        self.ld_source = self.logit_diff(self.src_logits)
        # Distributional gap between the two originals; the KL-based recovery
        # is the fraction of it closed, defined even when both runs share a
        # top token and the logit-difference margin is zero.
        self.kl_gap = _kl(self.tgt_logits, self.src_logits)
        self.src_final = src.hidden_states[-1][0, -1].float()
        self.tgt_final = tgt.hidden_states[-1][0, -1].float()
        self.cos_before = _cosine(self.tgt_final, self.src_final)
        self.delta_before = _delta_norm(self.tgt_final, self.src_final)
        self.top_src = _topk_tokens(self.src_logits, tokenizer, topk)
        self.top_tgt = _topk_tokens(self.tgt_logits, tokenizer, topk)
        self.tokenizer, self.topk = tokenizer, topk

    def logit_diff(self, logits: torch.Tensor) -> float:
        return float((logits[self.tgt_tok] - logits[self.src_tok]).item())

    def recovered_logit_diff(self, ld_after: float) -> float:
        gap = self.ld_before - self.ld_source
        if abs(gap) < 1e-6:
            return float("nan")
        return float((self.ld_before - ld_after) / gap)

    def recovered_kl(self, logits: torch.Tensor) -> float:
        if self.kl_gap < 1e-9:
            return 0.0
        return float(1.0 - _kl(logits, self.src_logits) / self.kl_gap)

    def recovered(self, logits: torch.Tensor) -> float:
        """Primary recovery number: logit-difference based, KL based when the
        two runs already agree on the top token."""
        r = self.recovered_logit_diff(self.logit_diff(logits))
        if self.same_top or r != r:  # NaN check
            return self.recovered_kl(logits)
        return r

    def result(self, layer: int, pos: int, logits: torch.Tensor, final: torch.Tensor) -> SinglePatchResult:
        ld_after = self.logit_diff(logits)
        top_after = _topk_tokens(logits, self.tokenizer, self.topk)
        flipped = (not self.same_top) and int(logits.argmax().item()) == self.src_tok
        # A_to_B: source is A, so "top_a_before" is the source's prediction.
        top_a, top_b = (self.top_src, self.top_tgt) if self.direction == "A_to_B" else (self.top_tgt, self.top_src)
        return SinglePatchResult(
            layer=layer, position=pos, direction=self.direction,
            cos_before=self.cos_before, delta_before=self.delta_before,
            cos_after=_cosine(final, self.src_final), delta_after=_delta_norm(final, self.src_final),
            top_a_before=top_a, top_b_before=top_b, top_target_after=top_after,
            logit_diff_before=self.ld_before, logit_diff_after=ld_after,
            logit_diff_source=self.ld_source, recovered=self.recovered(logits),
            recovered_kl=self.recovered_kl(logits),
            kl_to_target_after=_kl(logits, self.tgt_logits),
            kl_to_source_after=_kl(logits, self.src_logits),
            top_flipped=flipped,
        )


def run_patching_experiments(
    model: Any,
    tokenizer: Any,
    config: Config,
    result_a: RunResult,
    result_b: RunResult,
    cos_mat: np.ndarray,
    dn_mat: np.ndarray,
    spike_layer: int,
    start_a: int,
    start_b: int,
    window_len: int,
) -> Dict[str, Any]:
    """Run activation patching in both directions with real forward passes.

    Layer-level patches sweep every hidden-state index unless
    ``config.patch_layers`` narrows them. Head- and neuron-level patches
    default to the spike layer. Positions are window-relative; the default
    is the last token of the aligned window.
    """
    arch = detect_architecture(model)
    blocks = get_decoder_layers(model, arch) if arch else None
    stack = _get_layer_stack(model, arch) if arch else None
    if not blocks or not stack:
        return {
            "enabled": False,
            "reason": f"Architecture {arch!r} does not expose decoder blocks; skipping patching.",
        }
    n_blocks = len(blocks)
    n_hidden = len(result_a.hidden_states)
    sublayers = {idx: (attn, mlp) for idx, attn, mlp in stack}

    # Reject invalid selections; filtering them would run a different experiment.
    if window_len < 1:
        raise ValueError("Activation patching requires a nonempty aligned token window")
    positions = config.patch_positions if config.patch_positions is not None else [window_len - 1]
    if not positions or any(p < 0 or p >= window_len for p in positions):
        raise ValueError(f"Patch positions must be between 0 and {window_len - 1} in the actual aligned window")
    patch_mode = "multi_position" if config.patch_positions is not None else "last_token"
    component = config.patch_components or "layer"
    if component not in ("layer", "head", "neuron"):
        raise ValueError(f"Unknown patch component: {component}")
    units = (config.patch_heads if component == "head" else config.patch_neurons) or []
    if units:
        layers = sorted({layer for layer, _ in units})
        if config.patch_layers is not None and set(config.patch_layers) != set(layers):
            raise ValueError("Selected patch layers must match the blocks in the head/neuron selection")
    elif config.patch_layers is not None:
        layers = list(config.patch_layers)
    else:
        layers = list(range(n_hidden)) if component == "layer" else [min(spike_layer, n_blocks - 1)]
    limit = n_hidden if component == "layer" else n_blocks
    if not layers or any(layer < 0 or layer >= limit for layer in layers):
        raise ValueError(f"Patch {component} layer indices must be between 0 and {limit - 1}")
    for layer, unit in units:
        if component == "head":
            count = getattr(model.config, "num_attention_heads", 0)
        else:
            captured = (result_a.pre_mlp_activations or {}).get(layer)
            count = captured.shape[-1] if captured is not None else 0
        if unit < 0 or unit >= count:
            raise ValueError(f"Patch {component} index {unit} is outside block {layer}'s range 0–{count - 1}")

    pairs = [
        _Pair("A_to_B", result_a, result_b, start_a, start_b, tokenizer, config.topk),
        _Pair("B_to_A", result_b, result_a, start_b, start_a, tokenizer, config.topk),
    ]

    experiments: List[PatchExperiment] = []
    forward_passes = 0
    notes: List[str] = []

    def run(pair: _Pair, hooks) -> Tuple[torch.Tensor, torch.Tensor]:
        nonlocal forward_passes
        forward_passes += 1
        return _forward(model, pair.tgt.input_ids, hooks)

    # ---- layer level -------------------------------------------------------
    if component == "layer":
        for pair in pairs:
            for layer in layers:
                results: List[SinglePatchResult] = []
                for pos in positions:
                    s, t = pair.start_src + pos, pair.start_tgt + pos
                    vec = pair.src.hidden_states[layer][0, s]
                    if layer < n_blocks:
                        hooks = [(blocks[layer], "pre", _replace_positions_pre_hook([(t, vec)]))]
                    else:
                        hooks = [(_final_target(model, blocks), "post", _replace_positions_post_hook([(t, vec)]))]
                    logits, final = run(pair, hooks)
                    results.append(pair.result(layer, pos, logits, final))
                experiments.append(PatchExperiment(
                    id=f"{pair.direction.lower()}_{patch_mode}_layer{layer}",
                    description=(f"Residual patch ({pair.direction.replace('_', ' ')}) at hidden "
                                 f"index {layer}, positions={positions}"),
                    patch_mode=patch_mode, layer=layer, positions=list(positions),
                    direction=pair.direction, results=results, component="layer",
                ))

    # ---- head level --------------------------------------------------------
    elif component == "head":
        from vivasecuris.aiasylum.interp.analysis.ov_qk import compute_per_head_outputs

        for pair in pairs:
            for layer in layers:
                if layer not in sublayers:
                    continue
                per_src = compute_per_head_outputs(pair.src, model, layer, pair.start_src, window_len)
                per_tgt = compute_per_head_outputs(pair.tgt, model, layer, pair.start_tgt, window_len)
                if per_src is None or per_tgt is None:
                    raise ValueError(f"Block {layer}: per-head outputs unavailable (need attention + Q/K/V capture)")
                n_heads = int(per_src.shape[0])
                wanted = [h for (ly, h) in (config.patch_heads or []) if ly == layer] or list(range(n_heads))
                attn_module = sublayers[layer][0]
                for h in wanted:
                    if h >= n_heads:
                        continue
                    results = []
                    for pos in positions:
                        delta = per_src[h, pos] - per_tgt[h, pos]
                        hooks = [(attn_module, "post", _add_delta_post_hook([(pair.start_tgt + pos, delta)]))]
                        logits, final = run(pair, hooks)
                        results.append(pair.result(layer, pos, logits, final))
                    experiments.append(PatchExperiment(
                        id=f"head_{pair.direction.lower()}_L{layer}_H{h}",
                        description=f"Head patch L{layer} H{h} ({pair.direction})",
                        patch_mode=patch_mode, layer=layer, positions=list(positions),
                        direction=pair.direction, results=results,
                        component="head", component_index=h,
                    ))

    # ---- neuron level ------------------------------------------------------
    else:
        for pair in pairs:
            for layer in layers:
                if layer not in sublayers:
                    continue
                pre_src = (pair.src.pre_mlp_activations or {}).get(layer)
                pre_tgt = (pair.tgt.pre_mlp_activations or {}).get(layer)
                down = _get_mlp_down(model, layer)
                if pre_src is None or pre_tgt is None or down is None:
                    raise ValueError(f"Block {layer}: per-neuron outputs unavailable (need pre-MLP capture)")
                pre_src = pre_src[0, pair.start_src:pair.start_src + window_len]
                pre_tgt = pre_tgt[0, pair.start_tgt:pair.start_tgt + window_len]
                n_neurons = int(pre_src.shape[-1])
                wanted = [n for (ly, n) in (config.patch_neurons or []) if ly == layer]
                if not wanted:
                    # Rank by how much the neuron's contribution differs at the last position.
                    diff = (pre_src[positions[-1]] - pre_tgt[positions[-1]]).abs() * down.norm(dim=0)
                    wanted = torch.topk(diff, k=min(DEFAULT_NEURON_BUDGET, n_neurons)).indices.tolist()
                mlp_module = sublayers[layer][1]
                for n in wanted:
                    if n < 0 or n >= n_neurons:
                        continue
                    results = []
                    for pos in positions:
                        delta = (pre_src[pos, n] - pre_tgt[pos, n]) * down[:, n]
                        hooks = [(mlp_module, "post", _add_delta_post_hook([(pair.start_tgt + pos, delta)]))]
                        logits, final = run(pair, hooks)
                        results.append(pair.result(layer, pos, logits, final))
                    experiments.append(PatchExperiment(
                        id=f"neuron_{pair.direction.lower()}_L{layer}_N{n}",
                        description=f"Neuron patch L{layer} N{n} ({pair.direction})",
                        patch_mode=patch_mode, layer=layer, positions=list(positions),
                        direction=pair.direction, results=results,
                        component="neuron", component_index=int(n),
                    ))

    # ---- summary and null baseline ----------------------------------------
    summary: Dict[str, Any] = {}
    baseline: Dict[str, Any] = {}
    for pair in pairs:
        mine = [e for e in experiments if e.direction == pair.direction and e.results]
        if not mine:
            continue
        best = max(mine, key=lambda e: max(r.recovered for r in e.results))
        best_r = max(best.results, key=lambda r: r.recovered)
        summary[pair.direction] = {
            "best_layer": best.layer,
            "best_component": best.component,
            "best_component_index": best.component_index,
            "best_recovered": best_r.recovered,
            "best_recovered_kl": best_r.recovered_kl,
            "best_top_flipped": best_r.top_flipped,
            "same_top_token": pair.same_top,
        }
        if component == "layer":
            # Same norm as the real patch, random direction: what "any perturbation
            # of this size" does at the layer where the real patch worked best.
            pos = best_r.position
            s, t = pair.start_src + pos, pair.start_tgt + pos
            real = pair.src.hidden_states[best.layer][0, s].float() - pair.tgt.hidden_states[best.layer][0, t].float()
            gen = torch.Generator().manual_seed(int(getattr(config, "seed", 0) or 0))
            noise = torch.randn(real.shape, generator=gen)
            noise = noise / noise.norm().clamp_min(1e-12) * real.norm()
            vec = pair.tgt.hidden_states[best.layer][0, t].float() + noise
            if best.layer < n_blocks:
                hooks = [(blocks[best.layer], "pre", _replace_positions_pre_hook([(t, vec)]))]
            else:
                hooks = [(_final_target(model, blocks), "post", _replace_positions_post_hook([(t, vec)]))]
            logits, _ = run(pair, hooks)
            baseline[pair.direction] = {
                "kind": "random_perturbation_same_norm",
                "layer": best.layer,
                "position": pos,
                "recovered": pair.recovered(logits),
                "recovered_kl": pair.recovered_kl(logits),
            }
            summary[pair.direction]["beats_null"] = (
                best_r.recovered > baseline[pair.direction]["recovered"] + 0.05
            )

    return {
        "enabled": True,
        "claim": "causal",
        "patch_mode": patch_mode,
        "component": component,
        "layer": spike_layer,
        "layers": layers,
        "positions": positions,
        "forward_passes": forward_passes,
        "notes": notes,
        "summary": summary,
        "baseline": baseline,
        "experiments": [
            {
                "id": exp.id,
                "description": exp.description,
                "patch_mode": exp.patch_mode,
                "layer": exp.layer,
                "positions": exp.positions,
                "direction": exp.direction,
                "component": exp.component,
                "component_index": exp.component_index,
                "results": [asdict(r) for r in exp.results],
            }
            for exp in experiments
        ],
    }
