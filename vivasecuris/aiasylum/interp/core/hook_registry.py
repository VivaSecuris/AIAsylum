"""Architecture detection and activation hook registration for multi-arch support.

Captures per-layer attention output and MLP output (and optionally pre-MLP
intermediate activations) so residual stream decomposition and logit
attribution can be computed without extra forward passes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Normalized architecture names we support (Ollama/Llama-style only)
ARCH_LLAMA = "llama"
ARCH_MISTRAL = "mistral"
ARCH_GEMMA = "gemma"
ARCH_QWEN2 = "qwen2"
ARCH_GPT_NEOX = "gpt_neox"
ARCH_MIXTRAL = "mixtral"

# Model types that use the same layer structure as Llama (model.model.layers, self_attn, mlp).
# Mixtral is listed here because it uses model.model.layers and self_attn like the rest; its
# feed-forward submodule is named block_sparse_moe rather than mlp, which _get_layer_stack
# handles. It gets its own constant rather than aliasing to mistral so the arch string is
# never itself a lie.
_LLAMA_STYLE_TYPES = (
    ARCH_LLAMA,
    ARCH_MISTRAL,
    ARCH_GEMMA,
    ARCH_QWEN2,
    ARCH_MIXTRAL,
)

# Feed-forward submodule names, in probe order. Mixtral calls it block_sparse_moe; a few
# community architectures call it feed_forward.
_FFN_ATTRS = ("mlp", "block_sparse_moe", "feed_forward")


@dataclass(frozen=True)
class ArchInfo:
    """What we know about a model's architecture, including what we had to guess.

    ``family`` is the normalized ARCH_* bucket that decides which code path runs.
    ``model_type`` is ``config.model_type`` verbatim -- the only value that should
    ever appear in an error message, because ``family`` may be the result of the
    structural fallback and therefore name an architecture the model is not. A
    Mixtral checkpoint reported as 'llama' sent one debugging session down the
    wrong path; naming the real type costs nothing and ends that class of confusion.
    """

    family: Optional[str]
    model_type: Optional[str]
    from_structure: bool
    is_moe: bool

    @property
    def label(self) -> str:
        """How to name this model in a message meant for a human."""
        if self.model_type and self.family and self.model_type != self.family:
            if self.from_structure:
                return f"model_type={self.model_type!r} (matched structurally as the {self.family} family)"
            return f"model_type={self.model_type!r} ({self.family} family)"
        return f"model_type={self.model_type!r}" if self.model_type else f"architecture {self.family!r}"


def _ffn_module(block: nn.Module) -> Optional[nn.Module]:
    """The block's feed-forward submodule, whatever this architecture calls it."""
    for attr in _FFN_ATTRS:
        mod = getattr(block, attr, None)
        if mod is not None:
            return mod
    return None


def _looks_sparse(ffn: nn.Module) -> bool:
    """True when this feed-forward module routes across experts rather than one MLP."""
    return getattr(ffn, "experts", None) is not None


def _iter_blocks(model: nn.Module) -> List[nn.Module]:
    for holder, attr in ((getattr(model, "model", None), "layers"),
                         (getattr(model, "gpt_neox", None), "layers")):
        if holder is not None:
            blocks = getattr(holder, attr, None)
            if blocks is not None:
                return list(blocks)
    return []


def describe_architecture(model: nn.Module) -> ArchInfo:
    """Detect the architecture and record how confident that detection is.

    ``is_moe`` is decided structurally rather than from ``model_type``, because
    Qwen picks dense-vs-sparse per layer (``mlp_only_layers``, ``decoder_sparse_step``):
    a single stack can be mixed, and the config name alone does not say which layers.
    """
    model_type = None
    if hasattr(model, "config"):
        raw = getattr(model.config, "model_type", None)
        if raw:
            model_type = str(raw).lower()

    family: Optional[str] = None
    if model_type:
        if model_type.startswith("llama"):
            family = ARCH_LLAMA
        elif model_type == "mistral":
            family = ARCH_MISTRAL
        elif model_type.startswith("mixtral"):
            family = ARCH_MIXTRAL
        elif model_type in ("gemma", "gemma2", "gemma3"):
            family = ARCH_GEMMA
        # Qwen2, Qwen2-MoE, Qwen3 and Qwen3-MoE all use the Llama block layout.
        # The MoE variants differ only in what `mlp` holds, which arch.py handles.
        elif model_type.startswith("qwen2") or model_type.startswith("qwen3"):
            family = ARCH_QWEN2
        elif model_type == "gpt_neox":
            family = ARCH_GPT_NEOX

    from_structure = False
    if family is None:
        # Structural fallback. This is load-bearing -- it is why unknown Llama clones
        # work at all -- but it means `family` is a guess, so it is flagged as one.
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            family, from_structure = ARCH_LLAMA, True
        elif hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
            family, from_structure = ARCH_GPT_NEOX, True

    is_moe = any(
        _looks_sparse(ffn)
        for ffn in (_ffn_module(b) for b in _iter_blocks(model))
        if ffn is not None
    )

    return ArchInfo(family=family, model_type=model_type,
                    from_structure=from_structure, is_moe=is_moe)


def detect_architecture(model: nn.Module) -> Optional[str]:
    """Detect model architecture from config or structure.

    Returns one of ARCH_* constants or None if unsupported.
    Only Llama-style and related Ollama-compatible architectures are supported.

    Use :func:`describe_architecture` when building an error message -- this
    returns the normalized family, which may be a structural guess.
    """
    return describe_architecture(model).family


def _get_layer_stack(model: nn.Module, arch: str) -> Optional[List[Tuple[Any, nn.Module, nn.Module]]]:
    """Return list of (layer_index, attn_module, mlp_module) for each block.

    attn_module and mlp_module are the submodules whose *output* is the
    tensor added to the residual (attention output and MLP output).
    """
    if arch in _LLAMA_STYLE_TYPES and hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = []
        for i, block in enumerate(model.model.layers):
            attn = getattr(block, "self_attn", None)
            # Mixtral names its feed-forward `block_sparse_moe`; its output is what lands
            # in the residual just as `mlp`'s does, and _extract_output_tensor already
            # unwraps the (hidden, router_logits) tuple every MoE block returns.
            mlp = _ffn_module(block)
            if attn is not None and mlp is not None:
                layers.append((i, attn, mlp))
        return layers if layers else None

    if arch == ARCH_GPT_NEOX and hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        layers = []
        for i, block in enumerate(model.gpt_neox.layers):
            attn = getattr(block, "attention", None)
            mlp = _ffn_module(block)
            if attn is not None and mlp is not None:
                layers.append((i, attn, mlp))
        return layers if layers else None

    return None


def _extract_output_tensor(output: Any) -> torch.Tensor:
    """Extract the main tensor from a sublayer output (may be tuple)."""
    if isinstance(output, tuple):
        return output[0]
    return output


class ActivationHooks:
    """Registers and holds forward hooks for attention and MLP outputs (and optional pre-MLP)."""

    def __init__(
        self,
        model: nn.Module,
        capture_attn_output: bool = True,
        capture_mlp_output: bool = True,
        capture_pre_mlp: bool = False,
    ):
        self.model = model
        self.capture_attn_output = capture_attn_output
        self.capture_mlp_output = capture_mlp_output
        self.capture_pre_mlp = capture_pre_mlp

        self._handles: List[Any] = []
        self.attn_outputs: Dict[int, torch.Tensor] = {}
        self.mlp_activations: Dict[int, torch.Tensor] = {}
        self.pre_mlp_activations: Dict[int, torch.Tensor] = {}

        self._arch = detect_architecture(model)
        self._layer_stack = _get_layer_stack(model, self._arch) if self._arch else None

    @property
    def supported(self) -> bool:
        return self._layer_stack is not None

    def register(self) -> None:
        """Register all hooks. No-op if architecture unsupported."""
        if not self.supported:
            logger.warning("Activation hooks: architecture %s not supported, skipping", self._arch)
            return
        self.attn_outputs = {}
        self.mlp_activations = {}
        self.pre_mlp_activations = {}

        for layer_idx, attn_module, mlp_module in self._layer_stack:
            if self.capture_attn_output:
                handle = attn_module.register_forward_hook(
                    self._make_attn_hook(layer_idx)
                )
                self._handles.append(handle)
            if self.capture_mlp_output:
                handle = mlp_module.register_forward_hook(
                    self._make_mlp_hook(layer_idx)
                )
                self._handles.append(handle)
            if self.capture_pre_mlp:
                # Pre-MLP: hook inside MLP. Architecture-specific.
                inner_handle = self._register_pre_mlp_hook(layer_idx, mlp_module)
                if inner_handle is not None:
                    self._handles.append(inner_handle)

    def _make_attn_hook(self, layer_idx: int) -> Callable[..., None]:
        def hook(_module: nn.Module, _input: Any, output: Any) -> None:
            t = _extract_output_tensor(output)
            self.attn_outputs[layer_idx] = t.detach().cpu().float()
        return hook

    def _make_mlp_hook(self, layer_idx: int) -> Callable[..., None]:
        def hook(_module: nn.Module, _input: Any, output: Any) -> None:
            t = _extract_output_tensor(output)
            self.mlp_activations[layer_idx] = t.detach().cpu().float()
        return hook

    def _register_pre_mlp_hook(self, layer_idx: int, mlp_module: nn.Module) -> Optional[Any]:
        """Register hook for pre-MLP (intermediate) activations if possible."""
        # Llama/Mistral/Gemma: MLP has gate_proj, up_proj, down_proj. Intermediate = silu(gate)*up
        # (computed in forward, not a single submodule output). Pre-MLP would require
        # a forward hook that captures the intermediate inside the MLP forward; left for future work.
        return None

    def remove(self) -> None:
        """Remove all registered hooks."""
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles = []

    def get_attn_outputs_tuple(self) -> Optional[Tuple[torch.Tensor, ...]]:
        """Return attn_outputs as a tuple ordered by layer index (0..L-1)."""
        if not self.attn_outputs:
            return None
        n = max(self.attn_outputs.keys()) + 1
        if any(i not in self.attn_outputs for i in range(n)):
            return None
        return tuple(self.attn_outputs[i] for i in range(n))

    def get_mlp_activations_copy(self) -> Dict[int, torch.Tensor]:
        """Return a copy of mlp_activations dict (caller may clear after use)."""
        return dict(self.mlp_activations)

    def get_pre_mlp_copy(self) -> Optional[Dict[int, torch.Tensor]]:
        """Return a copy of pre_mlp_activations or None if empty."""
        if not self.pre_mlp_activations:
            return None
        return dict(self.pre_mlp_activations)


def _get_config_value(config: Any, key: str, default: Optional[int] = None) -> Optional[int]:
    """Get int config value if present."""
    if config is None:
        return default
    v = getattr(config, key, None)
    return int(v) if v is not None else default


class QKVHooks:
    """Captures per-layer Q, K, V for OV/QK analysis. Multi-arch."""

    def __init__(self, model: nn.Module):
        self.model = model
        self._handles: List[Any] = []
        self.qkv_outputs: Dict[int, Dict[str, torch.Tensor]] = {}
        self._arch = detect_architecture(model)
        self._layer_stack = _get_layer_stack(model, self._arch) if self._arch else None
        self._config = getattr(model, "config", None)

    @property
    def supported(self) -> bool:
        return self._layer_stack is not None

    def register(self) -> None:
        if not self.supported:
            return
        self.qkv_outputs = {}
        if self._arch in _LLAMA_STYLE_TYPES:
            self._register_llama_style()
        elif self._arch == ARCH_GPT_NEOX:
            self._register_gpt_neox()

    def _register_llama_style(self) -> None:
        n_heads = _get_config_value(self._config, "num_attention_heads", 32)
        n_kv_heads = _get_config_value(self._config, "num_key_value_heads", n_heads)
        head_dim = _get_config_value(self._config, "head_dim") or (
            _get_config_value(self._config, "hidden_size", 4096) // n_heads
        )
        for layer_idx, attn_module, _ in self._layer_stack:
            q_proj = getattr(attn_module, "q_proj", None)
            k_proj = getattr(attn_module, "k_proj", None)
            v_proj = getattr(attn_module, "v_proj", None)
            if not (q_proj and k_proj and v_proj):
                continue

            def make_q_hook(li: int):
                def hook(_m: nn.Module, _in: Any, out: Any) -> None:
                    t = _extract_output_tensor(out).detach().cpu().float()
                    # [1, seq, n_heads*head_dim]
                    if self.qkv_outputs.get(li) is None:
                        self.qkv_outputs[li] = {}
                    self.qkv_outputs[li]["q"] = t.reshape(1, t.shape[1], n_heads, head_dim)
                return hook

            def make_k_hook(li: int):
                def hook(_m: nn.Module, _in: Any, out: Any) -> None:
                    t = _extract_output_tensor(out).detach().cpu().float()
                    if self.qkv_outputs.get(li) is None:
                        self.qkv_outputs[li] = {}
                    self.qkv_outputs[li]["k"] = t.reshape(1, t.shape[1], n_kv_heads, head_dim)
                return hook

            def make_v_hook(li: int):
                def hook(_m: nn.Module, _in: Any, out: Any) -> None:
                    t = _extract_output_tensor(out).detach().cpu().float()
                    if self.qkv_outputs.get(li) is None:
                        self.qkv_outputs[li] = {}
                    self.qkv_outputs[li]["v"] = t.reshape(1, t.shape[1], n_kv_heads, head_dim)
                return hook

            self._handles.append(q_proj.register_forward_hook(make_q_hook(layer_idx)))
            self._handles.append(k_proj.register_forward_hook(make_k_hook(layer_idx)))
            self._handles.append(v_proj.register_forward_hook(make_v_hook(layer_idx)))

    def _register_gpt_neox(self) -> None:
        n_heads = _get_config_value(self._config, "num_attention_heads", 16)
        head_dim = _get_config_value(self._config, "hidden_size", 6144) // n_heads
        for layer_idx, attn_module, _ in self._layer_stack:
            query = getattr(attn_module, "query", None)
            key = getattr(attn_module, "key", None)
            value = getattr(attn_module, "value", None)
            if not (query and key and value):
                continue

            def make_hook(li: int, which: str):
                def hook(_m: nn.Module, _in: Any, out: Any) -> None:
                    t = _extract_output_tensor(out).detach().cpu().float()
                    if self.qkv_outputs.get(li) is None:
                        self.qkv_outputs[li] = {}
                    self.qkv_outputs[li][which] = t.reshape(1, t.shape[1], n_heads, head_dim)
                return hook

            self._handles.append(query.register_forward_hook(make_hook(layer_idx, "q")))
            self._handles.append(key.register_forward_hook(make_hook(layer_idx, "k")))
            self._handles.append(value.register_forward_hook(make_hook(layer_idx, "v")))

    def remove(self) -> None:
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles = []

    def get_qkv_copy(self) -> Optional[Dict[int, Dict[str, torch.Tensor]]]:
        if not self.qkv_outputs:
            return None
        return {k: {kk: vv.clone() for kk, vv in v.items()} for k, v in self.qkv_outputs.items()}
