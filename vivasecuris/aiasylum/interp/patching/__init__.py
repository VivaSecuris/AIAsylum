"""Patching and intervention modules for causal testing (Phase 3).

This module implements lightweight activation patching experiments that operate
purely on already-captured hidden states. The design goal is:

- **No additional forward passes**: we reuse `RunResult.hidden_states`.
- **Model-agnostic** for decoder-style LMs with an `lm_head` (same assumption
  as `PredictionAnalyzer`).
- **Safe by default**: if anything looks unsupported, we return a structured
  error in the payload instead of raising.

The current implementation focuses on:

- Patch mode: last-token activation patching at a selected layer
- Patch direction: A → B and B → A
- Metrics:
  - Before/after cosine similarity and delta-norm at the patched position
  - Local logits/top‑k prediction changes using the unembedding matrix
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from vivasecuris.aiasylum.interp.core.config import Config
from vivasecuris.aiasylum.interp.data.models import RunResult


PatchDirection = Literal["A_to_B", "B_to_A"]
PatchMode = Literal["last_token", "multi_position"]
PatchComponents = Literal["layer", "head", "neuron"]


@dataclass
class SinglePatchResult:
    """Result for a single (layer, position, direction) patch experiment."""

    layer: int
    position: int
    direction: PatchDirection

    # Metrics before patch
    cos_before: float
    delta_before: float
    top_a_before: List[str]
    top_b_before: List[str]

    # Metrics after patch (on target stream only)
    cos_after: float
    delta_after: float
    top_target_after: List[str]


@dataclass
class PatchExperiment:
    """Container for one logical patching experiment."""

    id: str
    description: str
    patch_mode: PatchMode
    layer: int
    positions: List[int]
    direction: PatchDirection
    results: List[SinglePatchResult]


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine similarity between two 1D tensors."""
    a_n = F.normalize(a.view(1, -1), dim=-1)
    b_n = F.normalize(b.view(1, -1), dim=-1)
    return float(torch.sum(a_n * b_n).item())


def _delta_norm(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute L2 norm of the difference between two 1D tensors."""
    return float(torch.linalg.vector_norm(a - b).item())


def _topk_from_hidden(
    hidden: torch.Tensor,
    lm_head_weight: torch.Tensor,
    tokenizer,
    topk: int,
) -> List[str]:
    """Project a hidden state through the unembedding and return top‑k tokens."""
    # hidden: [d], lm_head_weight: [vocab, d]
    logits = hidden @ lm_head_weight.t()
    probs = F.softmax(logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, k=topk)
    tokens = [tokenizer.decode([int(i)]) for i in top_indices]
    return tokens


def _get_lm_head_weight(model) -> Optional[torch.Tensor]:
    """Best-effort retrieval of the unembedding matrix."""
    if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
        return model.lm_head.weight.detach().cpu().float()
    if hasattr(model, "embed_out") and hasattr(model.embed_out, "weight"):
        return model.embed_out.weight.detach().cpu().float()
    return None


def _get_mlp_c_proj(model: Any, layer_idx: int) -> Optional[torch.Tensor]:
    """Return MLP output projection (down_proj for Llama-style). Shape [d_model, inter_dim]."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        block = model.model.layers[layer_idx]
        mlp = getattr(block, "mlp", None)
        if mlp is not None and hasattr(mlp, "down_proj") and hasattr(mlp.down_proj, "weight"):
            return mlp.down_proj.weight.detach().cpu().float()
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        block = model.gpt_neox.layers[layer_idx]
        mlp = getattr(block, "mlp", None)
        if mlp is not None and hasattr(mlp, "dense_4h_to_h") and hasattr(mlp.dense_4h_to_h, "weight"):
            return mlp.dense_4h_to_h.weight.detach().cpu().float()
    return None


def compute_per_neuron_mlp_outputs(
    result: RunResult,
    model: Any,
    layer_idx: int,
    start: int,
    window_len: int,
) -> Optional[torch.Tensor]:
    """
    Per-neuron MLP contribution for a layer when pre_mlp_activations and c_proj are available.
    Returns [n_neurons, window_len, d_model] or None.
    """
    pre_mlp = (
        result.pre_mlp_activations.get(layer_idx)
        if getattr(result, "pre_mlp_activations", None) else None
    )
    if pre_mlp is None:
        return None
    c_proj = _get_mlp_c_proj(model, layer_idx)
    if c_proj is None:
        return None
    # pre_mlp: [1, seq, inter_dim], c_proj: [d_model, inter_dim]
    pre = pre_mlp[0, start : start + window_len, :].float()  # [window_len, inter_dim]
    act = F.gelu(pre)  # Activation before down_proj (Llama uses silu; gelu for compatibility with pre_mlp if added later)
    # MLP output = act @ c_proj.T -> [window_len, d_model]
    # Per neuron n: act[:, n:n+1] @ c_proj[:, n:n+1].T -> [window_len, d_model]
    n_neurons = act.shape[1]
    d_model = c_proj.shape[0]
    device = pre.device
    c_proj = c_proj.to(device)
    out = torch.zeros(n_neurons, window_len, d_model, device=device, dtype=act.dtype)
    for n in range(n_neurons):
        out[n] = act[:, n : n + 1] * c_proj[:, n].unsqueeze(0)
    return out


def patch_head_into_residual(
    layer_idx: int,
    position_in_window: int,
    result_src: RunResult,
    result_tgt: RunResult,
    start_src: int,
    start_tgt: int,
    window_len: int,
    head_idx: int,
    model: Any,
) -> Optional[torch.Tensor]:
    """
    Compute residual at (layer_idx+1, position) after patching head `head_idx` from src into tgt.
    Returns [d_model] tensor or None if data unavailable.
    """
    from vivasecuris.aiasylum.interp.analysis.ov_qk import compute_per_head_outputs
    per_src = compute_per_head_outputs(result_src, model, layer_idx, start_src, window_len)
    per_tgt = compute_per_head_outputs(result_tgt, model, layer_idx, start_tgt, window_len)
    if per_src is None or per_tgt is None:
        return None
    if head_idx >= per_src.shape[0]:
        return None
    # Per-head at this position: [n_heads, d_model]
    head_src = per_src[:, position_in_window, :]  # [n_heads, d_model]
    head_tgt = per_tgt[:, position_in_window, :]
    attn_tgt = result_tgt.attn_outputs[layer_idx][0, start_tgt + position_in_window, :].float()
    patched_attn = attn_tgt - head_tgt[head_idx] + head_src[head_idx]
    pos_tgt = start_tgt + position_in_window
    hidden_before = result_tgt.hidden_states[layer_idx][0, pos_tgt, :].float()
    mlp_at = result_tgt.mlp_activations[layer_idx][0, pos_tgt, :].float()
    return (hidden_before + patched_attn + mlp_at).cpu()


def patch_neuron_into_residual(
    layer_idx: int,
    position_in_window: int,
    result_src: RunResult,
    result_tgt: RunResult,
    start_src: int,
    start_tgt: int,
    window_len: int,
    neuron_idx: int,
    model: Any,
) -> Optional[torch.Tensor]:
    """
    Compute residual at (layer_idx+1, position) after patching neuron `neuron_idx` from src into tgt.
    Returns [d_model] tensor or None if data unavailable.
    """
    per_src = compute_per_neuron_mlp_outputs(result_src, model, layer_idx, start_src, window_len)
    per_tgt = compute_per_neuron_mlp_outputs(result_tgt, model, layer_idx, start_tgt, window_len)
    if per_src is None or per_tgt is None:
        return None
    if neuron_idx >= per_src.shape[0]:
        return None
    # [n_neurons, d_model] at this position
    neu_src = per_src[:, position_in_window, :]
    neu_tgt = per_tgt[:, position_in_window, :]
    mlp_tgt = result_tgt.mlp_activations[layer_idx][0, start_tgt + position_in_window, :].float()
    patched_mlp = mlp_tgt - neu_tgt[neuron_idx] + neu_src[neuron_idx]
    pos_tgt = start_tgt + position_in_window
    hidden_before = result_tgt.hidden_states[layer_idx][0, pos_tgt, :].float()
    attn_at = result_tgt.attn_outputs[layer_idx][0, pos_tgt, :].float()
    return (hidden_before + attn_at + patched_mlp).cpu()


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
    """Run Phase 3-style activation patching experiments.

    Notes:
        - Currently only supports ``patch_mode == 'last_token'``.
        - We operate on the *spike layer* by default. If
          ``config.patch_layers`` is set, we intersect that with the spike
          layer (future work: support multi-layer sweeps).
    """
    lm_head_weight = _get_lm_head_weight(model)
    if lm_head_weight is None:
        return {
            "enabled": False,
            "reason": "Model does not expose an lm_head/embed_out weight; "
            "skipping patching experiments.",
        }

    # Determine which layer(s) to patch – for now we only support a single layer.
    layer_to_use = spike_layer
    if config.patch_layers:
        if spike_layer in config.patch_layers:
            layer_to_use = spike_layer
        else:
            layer_to_use = int(config.patch_layers[0])

    # Determine positions to patch: multi-position when patch_positions provided
    last_pos = window_len - 1 if window_len > 0 else 0
    if config.patch_positions is not None and len(config.patch_positions) > 0:
        positions = [p for p in config.patch_positions if 0 <= p < window_len]
        if not positions:
            positions = [last_pos]
        patch_mode = "multi_position"
    else:
        positions = [last_pos]
        patch_mode = "last_token"

    patch_components: Optional[PatchComponents] = (
        config.patch_components if isinstance(config.patch_components, str) else None
    )
    if patch_components not in ("layer", "head", "neuron"):
        patch_components = "layer"

    experiments: List[PatchExperiment] = []

    # Head-level and neuron-level experiments (granular)
    if patch_components == "head" and result_a.qkv_outputs is not None and result_a.attn_outputs is not None:
        head_list = config.patch_heads
        if not head_list:
            # Default: all heads in layer (we need n_heads from model or result)
            from vivasecuris.aiasylum.interp.analysis.ov_qk import compute_per_head_outputs
            per = compute_per_head_outputs(result_a, model, layer_to_use, start_a, window_len)
            n_heads = int(per.shape[0]) if per is not None else 0
            head_list = [(layer_to_use, h) for h in range(n_heads)] if n_heads else []
        for (ly, head_idx) in head_list:
            if ly != layer_to_use:
                continue
            for direction in ("A_to_B", "B_to_A"):
                dir_literal: PatchDirection = direction  # type: ignore[assignment]
                src, tgt = (result_a, result_b) if direction == "A_to_B" else (result_b, result_a)
                start_src = start_a if direction == "A_to_B" else start_b
                start_tgt = start_b if direction == "A_to_B" else start_a
                single_results: List[SinglePatchResult] = []
                for pos in positions:
                    idx_a = start_a + pos
                    idx_b = start_b + pos
                    hs_a = result_a.hidden_states[layer_to_use][0, idx_a].detach().cpu().float()
                    hs_b = result_b.hidden_states[layer_to_use][0, idx_b].detach().cpu().float()
                    cos_before = _cosine(hs_a, hs_b)
                    delta_before = _delta_norm(hs_a, hs_b)
                    top_a_before = _topk_from_hidden(hs_a, lm_head_weight, tokenizer, config.topk)
                    top_b_before = _topk_from_hidden(hs_b, lm_head_weight, tokenizer, config.topk)
                    patched = patch_head_into_residual(
                        layer_to_use, pos, src, tgt, start_src, start_tgt, window_len, head_idx, model
                    )
                    if patched is None:
                        continue
                    # Patched residual is at layer+1
                    ref = result_a.hidden_states[layer_to_use + 1][0, idx_a].float() if layer_to_use + 1 < len(result_a.hidden_states) else result_a.hidden_states[layer_to_use][0, idx_a].float()
                    cos_after = _cosine(patched, ref)
                    delta_after = _delta_norm(patched, ref)
                    top_target_after = _topk_from_hidden(patched, lm_head_weight, tokenizer, config.topk)
                    single_results.append(
                        SinglePatchResult(
                            layer=layer_to_use,
                            position=pos,
                            direction=dir_literal,
                            cos_before=cos_before,
                            delta_before=delta_before,
                            top_a_before=top_a_before,
                            top_b_before=top_b_before,
                            cos_after=cos_after,
                            delta_after=delta_after,
                            top_target_after=top_target_after,
                        )
                    )
                if single_results:
                    experiments.append(
                        PatchExperiment(
                            id=f"head_{direction.lower()}_L{layer_to_use}_H{head_idx}",
                            description=f"Head patch L{layer_to_use} H{head_idx} ({direction})",
                            patch_mode=patch_mode,
                            layer=layer_to_use,
                            positions=positions,
                            direction=dir_literal,
                            results=single_results,
                        )
                    )
    elif patch_components == "neuron" and getattr(result_a, "pre_mlp_activations", None) and result_a.mlp_activations:
        neuron_list = config.patch_neurons
        if not neuron_list:
            per = compute_per_neuron_mlp_outputs(result_a, model, layer_to_use, start_a, window_len)
            n_neurons = int(per.shape[0]) if per is not None else 0
            neuron_list = [(layer_to_use, n) for n in range(min(50, n_neurons))] if n_neurons else []
        for (ly, neuron_idx) in neuron_list:
            if ly != layer_to_use:
                continue
            for direction in ("A_to_B", "B_to_A"):
                dir_literal = "A_to_B" if direction == "A_to_B" else "B_to_A"
                dir_literal = dir_literal  # type: PatchDirection
                src, tgt = (result_a, result_b) if direction == "A_to_B" else (result_b, result_a)
                start_src = start_a if direction == "A_to_B" else start_b
                start_tgt = start_b if direction == "A_to_B" else start_a
                single_results = []
                for pos in positions:
                    idx_a = start_a + pos
                    idx_b = start_b + pos
                    hs_a = result_a.hidden_states[layer_to_use][0, idx_a].detach().cpu().float()
                    hs_b = result_b.hidden_states[layer_to_use][0, idx_b].detach().cpu().float()
                    cos_before = _cosine(hs_a, hs_b)
                    delta_before = _delta_norm(hs_a, hs_b)
                    top_a_before = _topk_from_hidden(hs_a, lm_head_weight, tokenizer, config.topk)
                    top_b_before = _topk_from_hidden(hs_b, lm_head_weight, tokenizer, config.topk)
                    patched = patch_neuron_into_residual(
                        layer_to_use, pos, src, tgt, start_src, start_tgt, window_len, neuron_idx, model
                    )
                    if patched is None:
                        continue
                    ref = result_a.hidden_states[layer_to_use + 1][0, idx_a].float() if layer_to_use + 1 < len(result_a.hidden_states) else result_a.hidden_states[layer_to_use][0, idx_a].float()
                    cos_after = _cosine(patched, ref)
                    delta_after = _delta_norm(patched, ref)
                    top_target_after = _topk_from_hidden(patched, lm_head_weight, tokenizer, config.topk)
                    single_results.append(
                        SinglePatchResult(
                            layer=layer_to_use,
                            position=pos,
                            direction=dir_literal,
                            cos_before=cos_before,
                            delta_before=delta_before,
                            top_a_before=top_a_before,
                            top_b_before=top_b_before,
                            cos_after=cos_after,
                            delta_after=delta_after,
                            top_target_after=top_target_after,
                        )
                    )
                if single_results:
                    experiments.append(
                        PatchExperiment(
                            id=f"neuron_{direction.lower()}_L{layer_to_use}_N{neuron_idx}",
                            description=f"Neuron patch L{layer_to_use} N{neuron_idx} ({direction})",
                            patch_mode=patch_mode,
                            layer=layer_to_use,
                            positions=positions,
                            direction=dir_literal,
                            results=single_results,
                        )
                    )
    else:
        # Layer-level (default) patching
        pass  # fall through to original loop below

    # We'll test both A→B and B→A for symmetry (layer-level)
    if patch_components == "layer" or not experiments:
        for direction in ("A_to_B", "B_to_A"):
            dir_literal: PatchDirection = direction  # type: ignore[assignment]
            exp_id = f"{direction.lower()}_{patch_mode}_layer{layer_to_use}"
            description = (
                f"Activation patching ({direction.replace('_', ' ')}) at layer "
                f"{layer_to_use}, positions={positions} (mode={patch_mode})"
            )

            single_results_layer: List[SinglePatchResult] = []

            for pos in positions:
                # Map aligned window position to absolute token indices
                idx_a = start_a + pos
                idx_b = start_b + pos
                hs_a_layer = result_a.hidden_states[layer_to_use][0, idx_a].detach().cpu().float()
                hs_b_layer = result_b.hidden_states[layer_to_use][0, idx_b].detach().cpu().float()

                if direction == "A_to_B":
                    src_h = hs_a_layer
                    tgt_before = hs_b_layer
                else:
                    src_h = hs_b_layer
                    tgt_before = hs_a_layer

                # Metrics before patch
                cos_before = _cosine(hs_a_layer, hs_b_layer)
                delta_before = _delta_norm(hs_a_layer, hs_b_layer)

                top_a_before = _topk_from_hidden(hs_a_layer, lm_head_weight, tokenizer, config.topk)
                top_b_before = _topk_from_hidden(hs_b_layer, lm_head_weight, tokenizer, config.topk)

                # Apply patch: overwrite target hidden with source hidden
                tgt_after = src_h
                cos_after = _cosine(hs_a_layer, tgt_after)
                delta_after = _delta_norm(hs_a_layer, tgt_after)
                top_target_after = _topk_from_hidden(
                    tgt_after, lm_head_weight, tokenizer, config.topk
                )

                single_results_layer.append(
                    SinglePatchResult(
                        layer=layer_to_use,
                        position=pos,
                        direction=dir_literal,
                        cos_before=cos_before,
                        delta_before=delta_before,
                        top_a_before=top_a_before,
                        top_b_before=top_b_before,
                        cos_after=cos_after,
                        delta_after=delta_after,
                        top_target_after=top_target_after,
                    )
                )

            experiments.append(
                PatchExperiment(
                    id=exp_id,
                    description=description,
                    patch_mode=patch_mode,
                    layer=layer_to_use,
                    positions=positions,
                    direction=dir_literal,
                    results=single_results_layer,
                )
            )

    return {
        "enabled": True,
        "patch_mode": patch_mode,
        "layer": layer_to_use,
        "positions": positions,
        "experiments": [
            {
                **{
                    "id": exp.id,
                    "description": exp.description,
                    "patch_mode": exp.patch_mode,
                    "layer": exp.layer,
                    "positions": exp.positions,
                    "direction": exp.direction,
                },
                "results": [asdict(r) for r in exp.results],
            }
            for exp in experiments
        ],
    }

