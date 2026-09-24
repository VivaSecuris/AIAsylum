"""Architecture detection and enumeration of residual-stream-writing matrices.

Adapted from labotomy (``llm_prompt_diff.core.hook_registry``), trimmed to the
detection/layer-stack core and extended with :func:`residual_write_matrices`,
which is what the surgery pass actually edits.

The distinction that matters here: a matrix *writes* to the residual stream if
its output is added into it. Those are the embedding table and, per block, the
attention out-projection and the MLP down-projection. ``lm_head`` *reads* from
the residual stream, so it is not a surgery target -- except that many small
models tie it to the embedding table, in which case editing one edits both.
That is detected and reported rather than silently happening.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

from vivasecuris.aiasylum.interp.core.hook_registry import (
    ARCH_GEMMA,
    ARCH_GPT_NEOX,
    ARCH_LLAMA,
    ARCH_MISTRAL,
    ARCH_MIXTRAL,
    ARCH_QWEN2,
    ArchInfo,
    _LLAMA_STYLE_TYPES,
    _get_layer_stack as get_layer_stack,
    describe_architecture,
    detect_architecture,
)

# How a matrix relates to the residual stream, which fixes the projection form.
KIND_EMBED = "embed"  # rows are residual vectors:      W -= (W @ r) (x) r
KIND_OUT = "out"      # columns map into residual dim:  W -= r (x) (r^T @ W)

# Attention out-projection names, in probe order: llama-style, then gpt_neox.
_ATTN_OUT_ATTRS = ("o_proj", "dense")

# MLP down-projection names, in probe order. `w2` is Mixtral's name for it inside
# MixtralBlockSparseTop2MLP -- a down_proj-only probe misses every Mixtral expert.
_MLP_DOWN_ATTRS = ("down_proj", "dense_4h_to_h", "w2")

# Feed-forward submodules that hold a single shared expert alongside the routed ones.
_SHARED_EXPERT_ATTRS = ("shared_expert", "shared_mlp")


def get_final_norm(model: nn.Module) -> Optional[nn.Module]:
    """The normalisation applied to the final residual before the unembedding.

    A logit lens that skips it reads un-normalised residuals through a matrix
    trained on normalised ones; on RMSNorm models the norms differ by an order
    of magnitude across depth, so the raw lens is a different measurement at
    every layer. Returns ``None`` when no known attribute is found.
    """
    for path in (
        ("model", "norm"),                 # llama, mistral, qwen2/3, gemma, mixtral
        ("gpt_neox", "final_layer_norm"),  # gpt_neox
        ("transformer", "ln_f"),           # gpt2-style
        ("model", "final_layernorm"),      # phi-style
    ):
        obj: Any = model
        for name in path:
            obj = getattr(obj, name, None)
            if obj is None:
                break
        if isinstance(obj, nn.Module):
            return obj
    return None


def final_hidden_is_normed(model: nn.Module) -> bool:
    """Whether ``output_hidden_states`` ends with the post-norm final residual.

    transformers changed this convention over time (4.57 returns the last
    entry after the final norm; older releases returned it before). Everything
    that reads or patches hidden-state index ``n_blocks`` has to know which,
    so it is probed once per model with a one-token forward pass and cached
    on the module rather than inferred from a version string.
    """
    cached = getattr(model, "_aiasylum_final_hidden_is_normed", None)
    if cached is not None:
        return bool(cached)
    result = False
    try:
        head = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
        if head is not None:
            ids = torch.tensor([[1]], device=model.device)
            with torch.no_grad():
                out = model(input_ids=ids, output_hidden_states=True, return_dict=True)
                direct = head(out.hidden_states[-1])
            result = bool(torch.allclose(direct.float(), out.logits.float(), atol=1e-4, rtol=1e-4))
    except Exception:  # pragma: no cover - probe is best-effort
        result = False
    try:
        setattr(model, "_aiasylum_final_hidden_is_normed", result)
    except Exception:
        pass
    return result


def get_decoder_layers(model: nn.Module, arch: str):
    """Return the decoder blocks themselves, for pre-hook steering."""
    if arch in _LLAMA_STYLE_TYPES and hasattr(model, "model"):
        blocks = getattr(model.model, "layers", None)
        return list(blocks) if blocks is not None else None
    if arch == ARCH_GPT_NEOX and hasattr(model, "gpt_neox"):
        blocks = getattr(model.gpt_neox, "layers", None)
        return list(blocks) if blocks is not None else None
    return None


@dataclass
class WriteMatrix:
    """A parameter that writes into the residual stream."""

    name: str
    param: Any  # torch.nn.Parameter
    kind: str   # KIND_EMBED | KIND_OUT


def _embedding_module(model: nn.Module, arch: str) -> Optional[Tuple[str, nn.Module]]:
    if arch in _LLAMA_STYLE_TYPES and hasattr(model, "model"):
        emb = getattr(model.model, "embed_tokens", None)
        if emb is not None:
            return "model.embed_tokens", emb
    if arch == ARCH_GPT_NEOX and hasattr(model, "gpt_neox"):
        emb = getattr(model.gpt_neox, "embed_in", None)
        if emb is not None:
            return "gpt_neox.embed_in", emb
    return None


def embeddings_are_tied(model: nn.Module) -> bool:
    """True when ``lm_head`` shares storage with the input embedding table.

    Qwen2.5 at 0.5B/1.5B/3B ties them; 7B and up do not. When tied, projecting
    the refusal direction out of the embedding table also removes it from the
    unembedding. That is usually desirable, but it is not what the caller
    literally asked for, so it gets recorded in the manifest.
    """
    head = getattr(model, "lm_head", None)
    if head is None or not hasattr(head, "weight"):
        return False
    arch = detect_architecture(model)
    found = _embedding_module(model, arch) if arch else None
    if found is None:
        return False
    _, emb = found
    if not hasattr(emb, "weight"):
        return False
    return head.weight.data_ptr() == emb.weight.data_ptr()


def _named_submodule_with_weight(parent: nn.Module, names: Tuple[str, ...]) -> Optional[nn.Module]:
    """First submodule of ``parent`` matching one of ``names`` and carrying a weight.

    Selection is by *name*, never by shape. A router gate is an ``nn.Linear`` that
    reads from the residual stream exactly as ``lm_head`` does, and any shape-based
    heuristic would eventually pick one up and edit it.
    """
    for attr in names:
        mod = getattr(parent, attr, None)
        if mod is not None and getattr(mod, "weight", None) is not None:
            return mod
    return None


def _describe_fused_experts(experts: nn.Module) -> str:
    """Name the offending parameter and its shape, for the rejection message."""
    for attr in _MLP_DOWN_ATTRS:
        param = getattr(experts, attr, None)
        if param is not None and hasattr(param, "shape"):
            return f"{attr} is a {tuple(param.shape)} tensor"
    return f"{type(experts).__name__} exposes no recognizable down-projection"


def attn_out_matrices(attn: nn.Module, layer_idx: int) -> List[WriteMatrix]:
    """The attention out-projection, whose output is added to the residual stream."""
    out_proj = _named_submodule_with_weight(attn, _ATTN_OUT_ATTRS)
    if out_proj is None:
        return []
    return [WriteMatrix(f"layers.{layer_idx}.attn_out.weight", out_proj.weight, KIND_OUT)]


def mlp_down_matrices(mlp: nn.Module, layer_idx: int, label: str = "") -> List[WriteMatrix]:
    """Every down-projection in this block's feed-forward, dense or sparse.

    A mixture-of-experts block writes ``sum_e g_e(x) * D_e a_e(x)`` into the residual,
    where ``g_e`` is a *scalar* router weight. Because the projection operator is the
    same for every expert and the gates are scalars, editing each expert's ``D_e``
    independently scales the whole block's residual contribution by exactly beta --
    the per-expert edit is exact, not an approximation. It only holds if *every*
    expert is edited: one missed expert reinjects the direction whenever it is routed
    to, which is what the coverage assertion in :func:`residual_write_plan` exists
    to make impossible.
    """
    # 1. Dense block -- one down-projection.
    down = _named_submodule_with_weight(mlp, _MLP_DOWN_ATTRS)
    if down is not None:
        return [WriteMatrix(f"layers.{layer_idx}.mlp_down.weight", down.weight, KIND_OUT)]

    experts = getattr(mlp, "experts", None)
    if experts is None:
        return []

    # 2. Sparse block with per-expert modules (Qwen2/Qwen3-MoE, Mixtral).
    if isinstance(experts, nn.ModuleList):
        out: List[WriteMatrix] = []
        for e, expert in enumerate(experts):
            expert_down = _named_submodule_with_weight(expert, _MLP_DOWN_ATTRS)
            if expert_down is None:
                raise ValueError(
                    f"Layer {layer_idx} expert {e} of {label or 'this model'} exposes no "
                    f"recognizable down-projection (looked for {', '.join(_MLP_DOWN_ATTRS)}). "
                    f"Editing the other experts would leave this one writing the direction "
                    f"back into the residual stream, so enumeration stops here."
                )
            out.append(WriteMatrix(
                f"layers.{layer_idx}.mlp_experts.{e}.down.weight", expert_down.weight, KIND_OUT
            ))

        # Qwen2-MoE carries a shared expert evaluated for every token; Qwen3-MoE and
        # Mixtral do not. Its output joins the same sum, so it is edited the same way.
        for attr in _SHARED_EXPERT_ATTRS:
            shared = getattr(mlp, attr, None)
            if shared is None:
                continue
            shared_down = _named_submodule_with_weight(shared, _MLP_DOWN_ATTRS)
            if shared_down is not None:
                out.append(WriteMatrix(
                    f"layers.{layer_idx}.mlp_shared_expert.down.weight",
                    shared_down.weight, KIND_OUT,
                ))
            break

        return out

    # 3. Fused expert tensors -- rejected rather than half-handled.
    #
    # gpt_oss stores experts as raw [n_experts, d_ff, d_model] parameters applied as
    # `x @ down_proj[e]`, so the residual axis is LAST -- transposed from the
    # nn.Linear [out, in] convention every projection in weights/surgery.py assumes.
    # On a square matrix that would succeed silently and edit the wrong axis, which is
    # exactly the failure test_weights_surgery.py exists to catch, reintroduced a level
    # up. These architectures also carry down_proj_bias and an attention output bias:
    # residual-space constants that no weight projection can reach, so a weight-only
    # edit there is provably incomplete no matter how the tensors are handled.
    #
    # Supporting them needs a third matrix kind (residual axis last) plus bias editing,
    # and a checkpoint to validate against. Until then this raises.
    raise ValueError(
        f"Layer {layer_idx} of {label or 'this model'} uses fused expert tensors "
        f"({_describe_fused_experts(experts)}), which this surgery pass cannot edit "
        f"correctly: the residual axis is last rather than first, and the block's bias "
        f"terms write into the residual stream where no weight projection reaches them. "
        f"Editing it would produce a model whose manifest claims an edit the weights do "
        f"not contain."
    )


@dataclass
class WritePlan:
    """The enumeration result, with enough detail to prove it was complete."""

    matrices: List[WriteMatrix]
    arch: ArchInfo
    per_layer: Dict[int, Dict[str, int]] = field(default_factory=dict)
    candidates_found: int = 0
    deduplicated: int = 0
    expert_matrices: int = 0
    shared_expert_matrices: int = 0
    moe_layers: int = 0
    n_layers: int = 0

    @property
    def coverage_verified(self) -> bool:
        """True only when every layer contributed on both sides. See residual_write_plan."""
        return bool(self.per_layer) and all(
            counts["attn"] > 0 and counts["mlp"] > 0 for counts in self.per_layer.values()
        )


def residual_write_plan(
    model: nn.Module,
    arch: Optional[str] = None,
    include_embeddings: bool = True,
) -> WritePlan:
    """Enumerate every parameter whose output lands in the residual stream.

    Deduplicated by storage pointer, so a tied embedding/unembedding pair is
    returned once and therefore edited once. Applying the projection twice
    would square it, which is not a no-op for any beta other than 0 or 1.

    Raises rather than returning a partial enumeration. Editing some of the
    matrices that write a direction and not others produces a model that still
    carries the direction, alongside a manifest that says it does not -- which is
    worse than refusing to edit at all.
    """
    info = describe_architecture(model)
    arch = arch or info.family
    if arch is None:
        raise ValueError(
            f"Unsupported architecture for {info.label}: could not locate a decoder "
            f"layer stack. Supported: llama, mistral, mixtral, gemma, qwen2/qwen3, gpt_neox."
        )

    stack = get_layer_stack(model, arch)
    if stack is None:
        raise ValueError(
            f"Could not enumerate decoder layers for {info.label}. Blocks expose no "
            f"recognizable attention and feed-forward pair."
        )

    matrices: List[WriteMatrix] = []
    seen: set = set()
    candidates = 0

    def add(matrix: Optional[WriteMatrix]) -> None:
        nonlocal candidates
        if matrix is None or matrix.param is None:
            return
        candidates += 1
        ptr = matrix.param.data_ptr()
        if ptr in seen:
            logger.debug("Skipping %s: shares storage with an earlier matrix", matrix.name)
            return
        seen.add(ptr)
        matrices.append(matrix)

    if include_embeddings:
        found = _embedding_module(model, arch)
        if found is not None:
            emb_name, emb = found
            add(WriteMatrix(f"{emb_name}.weight", emb.weight, KIND_EMBED))

    per_layer: Dict[int, Dict[str, int]] = {}
    expert_matrices = shared_expert_matrices = moe_layers = 0

    for layer_idx, attn, mlp in stack:
        attn_mats = attn_out_matrices(attn, layer_idx)
        mlp_mats = mlp_down_matrices(mlp, layer_idx, label=info.label)

        if any(".mlp_experts." in m.name for m in mlp_mats):
            moe_layers += 1
        expert_matrices += sum(".mlp_experts." in m.name for m in mlp_mats)
        shared_expert_matrices += sum(".mlp_shared_expert." in m.name for m in mlp_mats)

        # Counted before deduplication: a layer whose only matrix is a tied duplicate
        # has still been covered, and must not read as missing.
        per_layer[layer_idx] = {"attn": len(attn_mats), "mlp": len(mlp_mats)}

        for matrix in (*attn_mats, *mlp_mats):
            add(matrix)

    _assert_full_coverage(model, info, per_layer)

    if not matrices:
        raise ValueError(f"Found no residual-writing matrices for {info.label}.")

    plan = WritePlan(
        matrices=matrices,
        arch=info,
        per_layer=per_layer,
        candidates_found=candidates,
        deduplicated=candidates - len(matrices),
        expert_matrices=expert_matrices,
        shared_expert_matrices=shared_expert_matrices,
        moe_layers=moe_layers,
        n_layers=len(per_layer),
    )

    logger.info(
        "Enumerated %d residual-writing matrices across %d layers (%s%s); %d deduplicated",
        len(matrices), plan.n_layers, info.label,
        f", {expert_matrices} expert matrices in {moe_layers} MoE layers" if expert_matrices else "",
        plan.deduplicated,
    )
    return plan


def _assert_full_coverage(
    model: nn.Module,
    info: ArchInfo,
    per_layer: Dict[int, Dict[str, int]],
) -> None:
    """Every decoder layer must contribute on both sides, or we refuse to edit.

    This is the check that makes silent under-enumeration structurally impossible.
    Before it existed, a Qwen-MoE model enumerated its attention out-projections and
    none of its experts, edited a third of what it should have, and wrote a manifest
    that looked exactly like a clean run. The observable tell was a sweep that said
    'causal' followed by an edit that changed nothing.
    """
    missing_attn = sorted(i for i, c in per_layer.items() if c["attn"] == 0)
    missing_mlp = sorted(i for i, c in per_layer.items() if c["mlp"] == 0)
    if missing_attn or missing_mlp:
        raise ValueError(
            f"Residual-write enumeration is incomplete for {info.label}: "
            f"layers with no attention-side matrix: {missing_attn or 'none'}; "
            f"layers with no MLP-side matrix: {missing_mlp or 'none'}. "
            f"Those layers would still write the direction back into the residual "
            f"stream after the edit. Add the layout to arch.py rather than editing "
            f"partially."
        )

    expected = getattr(getattr(model, "config", None), "num_hidden_layers", None)
    if expected is not None and len(per_layer) != int(expected):
        raise ValueError(
            f"Enumerated {len(per_layer)} decoder layers for {info.label} but the config "
            f"declares {int(expected)}. Some blocks were skipped, and editing the rest "
            f"would leave the direction intact in the ones that were missed."
        )


def residual_write_matrices(
    model: nn.Module,
    arch: Optional[str] = None,
    include_embeddings: bool = True,
) -> List[WriteMatrix]:
    """Every parameter whose output lands in the residual stream.

    Thin wrapper over :func:`residual_write_plan` for callers that only need the
    matrices themselves.
    """
    return residual_write_plan(model, arch, include_embeddings).matrices
