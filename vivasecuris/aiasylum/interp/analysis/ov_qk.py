"""OV and QK circuit analysis: per-head outputs and QK attention patterns."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from vivasecuris.aiasylum.interp.data.models import RunResult
from vivasecuris.aiasylum.interp.core.hook_registry import (
    detect_architecture,
    ARCH_GPT_NEOX,
    _LLAMA_STYLE_TYPES,
)


def _get_w_o(model: Any, layer_idx: int, arch: str) -> Optional[torch.Tensor]:
    """Return output projection weight W_O for a layer. Shape [d_model, d_model] or [d_model, n_heads*head_dim]."""
    arch = detect_architecture(model) if not arch else arch
    if arch in _LLAMA_STYLE_TYPES and hasattr(model, "model") and hasattr(model.model, "layers"):
        block = model.model.layers[layer_idx]
        attn = getattr(block, "self_attn", None)
        if attn is not None and hasattr(attn, "o_proj") and hasattr(attn.o_proj, "weight"):
            return attn.o_proj.weight.detach().cpu().float()
    if arch == ARCH_GPT_NEOX and hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        block = model.gpt_neox.layers[layer_idx]
        attn = getattr(block, "attention", None)
        if attn is not None and hasattr(attn, "dense") and hasattr(attn.dense, "weight"):
            return attn.dense.weight.detach().cpu().float()
    return None


def compute_per_head_outputs(
    result: RunResult,
    model: Any,
    layer_idx: int,
    start: int,
    window_len: int,
) -> Optional[torch.Tensor]:
    """
    Compute per-head output contributions for a layer (OV circuit).

    Uses attention_weights, qkv_outputs["v"], and W_O from model.
    Returns tensor [n_heads, window_len, d_model] or None if data missing.
    """
    if result.attention_weights is None or layer_idx >= len(result.attention_weights):
        return None
    qkv = result.qkv_outputs if getattr(result, "qkv_outputs", None) else None
    if not qkv or layer_idx not in qkv or "v" not in qkv[layer_idx]:
        return None
    w_o = _get_w_o(model, layer_idx, None)
    if w_o is None:
        return None

    attn = result.attention_weights[layer_idx]  # [1, n_heads, seq, seq]
    v = qkv[layer_idx]["v"]  # [1, seq, n_heads, head_dim] or [1, n_heads, seq, head_dim]
    device = attn.device
    attn = attn.float()
    v = v.float().to(device)
    w_o = w_o.to(device)

    # Slice to window
    attn_w = attn[0, :, start : start + window_len, start : start + window_len]  # [n_heads, w, w]
    v_w = v[0, start : start + window_len, :, :]  # [w, n_heads, head_dim]
    n_heads = attn_w.shape[0]
    head_dim = v_w.shape[2]
    d_model = w_o.shape[0]

    # Per head: O_h = attn_h @ V_h -> [window_len, head_dim]
    head_outputs = []
    for h in range(n_heads):
        attn_h = attn_w[h]  # [w, w]
        v_h = v_w[:, h, :]  # [w, head_dim]
        o_h = torch.mm(attn_h, v_h)  # [w, head_dim]
        w_o_h = w_o[:, h * head_dim : (h + 1) * head_dim]  # [d_model, head_dim]
        out_h = torch.mm(o_h, w_o_h.t())  # [w, d_model]
        head_outputs.append(out_h)
    return torch.stack(head_outputs, dim=0)  # [n_heads, window_len, d_model]


def compute_qk_pattern(
    q: torch.Tensor,
    k: torch.Tensor,
    head_dim: int,
    scale: bool = True,
) -> torch.Tensor:
    """
    Compute attention pattern from Q and K: softmax(Q @ K.T / sqrt(head_dim)).

    q: [seq, n_heads, head_dim] or [n_heads, seq, head_dim]
    k: [seq, n_kv_heads, head_dim] or [n_heads, seq, head_dim]
    Returns [n_heads, seq, seq] or similar depending on GQA.
    """
    if q.dim() == 3 and q.shape[1] != k.shape[1]:
        # GQA: k has fewer heads; expand or use same for all q heads
        pass
    # Assume q, k same shape [seq, n_heads, head_dim]
    if q.dim() == 3:
        scores = torch.matmul(q, k.transpose(-2, -1))
    else:
        scores = torch.matmul(q, k.transpose(-2, -1))
    if scale:
        scores = scores / (head_dim ** 0.5)
    return F.softmax(scores.float(), dim=-1)


def run_ov_qk_analysis(
    result: RunResult,
    model: Any,
    layer_idx: int,
    start: int,
    window_len: int,
) -> Dict[str, Any]:
    """
    Run OV/QK analysis for one layer: per-head norms and optional QK pattern check.

    Returns dict with per_head_outputs (list of norms or full tensors), ov_available, qk_available.
    """
    out: Dict[str, Any] = {"layer_idx": layer_idx, "ov_available": False, "qk_available": False}
    per_head = compute_per_head_outputs(result, model, layer_idx, start, window_len)
    if per_head is not None:
        out["ov_available"] = True
        out["per_head_output_norms"] = torch.linalg.norm(per_head, dim=-1).cpu().numpy().tolist()
        out["n_heads"] = per_head.shape[0]
    qkv = getattr(result, "qkv_outputs", None)
    if qkv and layer_idx in qkv and "q" in qkv[layer_idx] and "k" in qkv[layer_idx]:
        q = qkv[layer_idx]["q"][0, start : start + window_len, :, :].float()
        k = qkv[layer_idx]["k"][0, start : start + window_len, :, :].float()
        head_dim = q.shape[-1]
        qk_pattern = compute_qk_pattern(q, k, head_dim)
        out["qk_available"] = True
        if result.attention_weights is not None and layer_idx < len(result.attention_weights):
            attn = result.attention_weights[layer_idx][0, :, start : start + window_len, start : start + window_len].float()
            diff = (qk_pattern - attn).abs().max().item()
            out["qk_vs_actual_max_diff"] = diff
    return out
