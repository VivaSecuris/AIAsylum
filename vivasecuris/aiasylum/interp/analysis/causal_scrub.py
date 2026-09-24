"""Causal scrubbing and minimal sufficient circuit discovery."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from vivasecuris.aiasylum.interp.data.models import RunResult
from vivasecuris.aiasylum.interp.analysis.ov_qk import compute_per_head_outputs
from vivasecuris.aiasylum.interp.patching import compute_per_neuron_mlp_outputs

Component = Tuple[int, str, int]  # (layer, "head"|"neuron", idx)


def _scrub_mask(
    components_to_scrub: List[Component],
    layer_idx: int,
    kind: str,
    n_units: int,
) -> torch.Tensor:
    """Boolean mask of size n_units: True = keep, False = scrub."""
    mask = torch.ones(n_units, dtype=torch.bool)
    for (ly, k, idx) in components_to_scrub:
        if ly == layer_idx and k == kind and 0 <= idx < n_units:
            mask[idx] = False
    return mask


def scrub_components(
    result: RunResult,
    position: int,
    start: int,
    window_len: int,
    components_to_scrub: List[Component],
    baseline: str = "zero",
    model: Optional[Any] = None,
) -> Optional[torch.Tensor]:
    """
    Compute residual at the final layer at `position` with given components scrubbed.

    Args:
        result: RunResult with hidden_states, attn_outputs, mlp_activations, and optionally
                qkv_outputs / pre_mlp_activations for per-head/per-neuron.
        position: Absolute token index in the sequence.
        start: Start of the aligned window.
        window_len: Window length.
        components_to_scrub: List of (layer_idx, "head"|"neuron", unit_idx) to replace with baseline.
        baseline: "zero" or "mean". Mean = mean over positions in the same run for that component.
        model: Model (for W_O and c_proj); required for per-head/per-neuron decomposition.

    Returns:
        Final-layer residual tensor [d_model] at position, or None if data unavailable.
    """
    if model is None or position < start or position >= start + window_len:
        return None
    pos_in_window = position - start
    num_layers = len(result.hidden_states) - 1
    device = result.hidden_states[0].device
    dtype = result.hidden_states[0].dtype
    h = result.hidden_states[0][0, position, :].float().to(device)

    for layer_idx in range(num_layers):
        # Attn: sum over heads; scrub some to baseline
        per_head = compute_per_head_outputs(result, model, layer_idx, start, window_len)
        if per_head is not None:
            n_heads = per_head.shape[0]
            keep = _scrub_mask(components_to_scrub, layer_idx, "head", n_heads).to(device)
            head_out = per_head[:, pos_in_window, :].to(device)  # [n_heads, d_model]
            if baseline == "mean":
                head_mean = head_out.mean(dim=0, keepdim=True).expand_as(head_out)
                head_out = torch.where(keep.unsqueeze(1), head_out, head_mean)
            else:
                head_out = torch.where(keep.unsqueeze(1), head_out, torch.zeros_like(head_out))
            attn_out = head_out.sum(dim=0)
        else:
            attn_out = result.attn_outputs[layer_idx][0, position, :].float().to(device)
            # Cannot scrub heads without per-head data
        # MLP: sum over neurons; scrub some to baseline
        per_neuron = compute_per_neuron_mlp_outputs(result, model, layer_idx, start, window_len)
        if per_neuron is not None:
            n_neurons = per_neuron.shape[0]
            keep = _scrub_mask(components_to_scrub, layer_idx, "neuron", n_neurons).to(device)
            neu_out = per_neuron[:, pos_in_window, :].to(device)  # [n_neurons, d_model]
            if baseline == "mean":
                neu_mean = neu_out.mean(dim=0, keepdim=True).expand_as(neu_out)
                neu_out = torch.where(keep.unsqueeze(1), neu_out, neu_mean)
            else:
                neu_out = torch.where(keep.unsqueeze(1), neu_out, torch.zeros_like(neu_out))
            mlp_out = neu_out.sum(dim=0)
        else:
            mlp_out = result.mlp_activations[layer_idx][0, position, :].float().to(device)
        h = h + attn_out + mlp_out

    return h.cpu()


def run_scrub_experiment(
    result: RunResult,
    position: int,
    start: int,
    window_len: int,
    components_to_scrub: List[Component],
    baseline: str,
    model: Any,
    lm_head_weight: torch.Tensor,
) -> Dict[str, Any]:
    """
    Run a scrubbing experiment: compute logits after scrubbing and compare to full run.

    Returns dict with scrubbed_logits, full_logits, logit_l2, logit_kl (if applicable).
    """
    full_h = result.hidden_states[-1][0, position, :].float()
    full_logits = (full_h.unsqueeze(0) @ lm_head_weight.t().float()).squeeze(0)
    scrubbed_h = scrub_components(
        result, position, start, window_len, components_to_scrub, baseline, model
    )
    if scrubbed_h is None:
        return {
            "scrub_available": False,
            "reason": "Could not compute scrubbed residual (missing per-head/per-neuron or model).",
        }
    scrubbed_logits = (scrubbed_h.unsqueeze(0) @ lm_head_weight.t().float()).squeeze(0)
    logit_l2 = float(torch.linalg.norm(scrubbed_logits - full_logits).item())
    full_probs = F.softmax(full_logits, dim=-1)
    scrubbed_probs = F.softmax(scrubbed_logits, dim=-1)
    logit_kl = float(F.kl_div(scrubbed_probs.log(), full_probs, reduction="sum").item())
    return {
        "scrub_available": True,
        "components_scrubbed": len(components_to_scrub),
        "logit_l2": logit_l2,
        "logit_kl": logit_kl,
        "baseline": baseline,
    }


def residual_after_patching_components(
    result_a: RunResult,
    result_b: RunResult,
    components_to_patch: List[Component],
    position: int,
    start_a: int,
    start_b: int,
    window_len: int,
    model: Any,
) -> Optional[torch.Tensor]:
    """
    Final-layer residual at position when we patch A by replacing the given components with B's.
    """
    if position < start_a or position >= start_a + window_len:
        return None
    pos_in_window = position - start_a
    num_layers = len(result_a.hidden_states) - 1
    device = result_a.hidden_states[0].device
    h = result_a.hidden_states[0][0, position, :].float().to(device)
    patch_set = set(components_to_patch)

    for layer_idx in range(num_layers):
        per_head_a = compute_per_head_outputs(result_a, model, layer_idx, start_a, window_len)
        per_head_b = compute_per_head_outputs(result_b, model, layer_idx, start_b, window_len)
        if per_head_a is not None and per_head_b is not None:
            head_a = per_head_a[:, pos_in_window, :].to(device)
            head_b = per_head_b[:, pos_in_window, :].to(device)
            attn_out = head_a.clone()
            for h_idx in range(head_a.shape[0]):
                if (layer_idx, "head", h_idx) in patch_set:
                    attn_out[h_idx] = head_b[h_idx]
            attn_out = attn_out.sum(dim=0)
        else:
            attn_out = result_a.attn_outputs[layer_idx][0, position, :].float().to(device)

        per_neu_a = compute_per_neuron_mlp_outputs(result_a, model, layer_idx, start_a, window_len)
        per_neu_b = compute_per_neuron_mlp_outputs(result_b, model, layer_idx, start_b, window_len)
        if per_neu_a is not None and per_neu_b is not None:
            neu_a = per_neu_a[:, pos_in_window, :].to(device)
            neu_b = per_neu_b[:, pos_in_window, :].to(device)
            mlp_out = neu_a.clone()
            for n_idx in range(neu_a.shape[0]):
                if (layer_idx, "neuron", n_idx) in patch_set:
                    mlp_out[n_idx] = neu_b[n_idx]
            mlp_out = mlp_out.sum(dim=0)
        else:
            mlp_out = result_a.mlp_activations[layer_idx][0, position, :].float().to(device)
        h = h + attn_out + mlp_out
    return h.cpu()


def find_minimal_circuit_greedy(
    result_a: RunResult,
    result_b: RunResult,
    start_a: int,
    start_b: int,
    window_len: int,
    model: Any,
    lm_head_weight: torch.Tensor,
    layer_range: Optional[Tuple[int, int]] = None,
    top_k_heads: int = 5,
    top_k_neurons: int = 5,
    metric: str = "logit_l2",
    max_components: int = 20,
) -> Dict[str, Any]:
    """
    Greedily find a minimal set of components (heads + neurons) that best restore B's logits
    when patching from A (add components that reduce the difference between
    "A patched with B's components" and "B" at the last token).

    Returns dict with circuit: List[(layer, "head"|"neuron", idx)], metric_history, final_metric.
    """
    last_pos = window_len - 1
    pos_a = start_a + last_pos
    pos_b = start_b + last_pos
    target_logits = (
        result_b.hidden_states[-1][0, pos_b, :].float().cpu()
        @ lm_head_weight.t().float()
    )
    num_layers = len(result_a.hidden_states) - 1
    if layer_range is None:
        layer_range = (0, num_layers)
    lo, hi = layer_range
    candidates: List[Component] = []
    for layer_idx in range(lo, hi):
        per_head = compute_per_head_outputs(result_a, model, layer_idx, start_a, window_len)
        if per_head is not None:
            n_heads = per_head.shape[0]
            for h in range(min(top_k_heads, n_heads)):
                candidates.append((layer_idx, "head", h))
        per_neu = compute_per_neuron_mlp_outputs(result_a, model, layer_idx, start_a, window_len)
        if per_neu is not None:
            n_neurons = per_neu.shape[0]
            for n in range(min(top_k_neurons, n_neurons)):
                candidates.append((layer_idx, "neuron", n))
    chosen: List[Component] = []
    remaining = list(candidates)
    metric_history: List[float] = []

    def logit_diff(patched_h: torch.Tensor) -> float:
        logits = patched_h.unsqueeze(0) @ lm_head_weight.t().float()
        if metric == "logit_l2":
            return float(torch.linalg.norm(logits.squeeze(0) - target_logits).item())
        return float(
            F.kl_div(
                F.log_softmax(logits, dim=-1).squeeze(0),
                F.softmax(target_logits, dim=-1),
                reduction="sum",
            ).item()
        )

    current_diff = logit_diff(
        result_a.hidden_states[-1][0, pos_a, :].float().cpu()
    )
    metric_history.append(current_diff)

    for _ in range(max_components):
        if not remaining:
            break
        best_candidate: Optional[Component] = None
        best_diff = current_diff
        for c in remaining:
            trial = chosen + [c]
            patched_h = residual_after_patching_components(
                result_a, result_b, trial, pos_a, start_a, start_b, window_len, model
            )
            if patched_h is None:
                continue
            d = logit_diff(patched_h)
            if d < best_diff:
                best_diff = d
                best_candidate = c
        if best_candidate is None:
            break
        chosen.append(best_candidate)
        remaining.remove(best_candidate)
        patched_h = residual_after_patching_components(
            result_a, result_b, chosen, pos_a, start_a, start_b, window_len, model
        )
        current_diff = best_diff
        metric_history.append(current_diff)

    return {
        "circuit": chosen,
        "metric": metric,
        "metric_history": metric_history,
        "final_metric": current_diff,
        "num_candidates": len(candidates),
    }
