"""Model execution and forward pass handling."""

import torch
from typing import List, Optional, Dict, Any
import logging
from transformers import PreTrainedModel, PreTrainedTokenizer

from vivasecuris.aiasylum.interp.data.models import RunResult
from vivasecuris.aiasylum.interp.core.hook_registry import ActivationHooks, QKVHooks

logger = logging.getLogger(__name__)


class ModelRunner:
    """Handles model forward passes and activation capture."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        debug_mode: bool = False,
        config: Optional[Any] = None,
    ):
        """
        Initialize model runner.

        Args:
            model: Pre-trained model
            tokenizer: Pre-trained tokenizer
            debug_mode: Whether to capture additional debug info (attention, MLP, attn outputs)
            config: Optional Config; when set, used for enable_attn_output_capture / enable_pre_mlp_capture
        """
        self.model = model
        self.tokenizer = tokenizer
        self.debug_mode = debug_mode
        self.config = config
        self.hooks = []
        self.mlp_activations = {}
        self._activation_hooks: Optional[ActivationHooks] = None
        self._qkv_hooks: Optional[QKVHooks] = None

    def run_once(
        self,
        prompt: str,
        max_length: int = 2048,
    ) -> RunResult:
        """
        Run a single forward pass and capture activations.
        
        Args:
            prompt: Input prompt text
            max_length: Maximum sequence length
            
        Returns:
            RunResult with captured activations
        """
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        input_ids = inputs["input_ids"].to(self.model.device)
        
        # Get token strings
        token_strs = [
            self.tokenizer.decode([token_id]) for token_id in input_ids[0]
        ]
        
        # Register hooks if needed
        if self.debug_mode:
            self._register_debug_hooks()

        # SDPA and flash kernels never materialise attention weights, so
        # `output_attentions=True` returns None under them. Switch to eager for
        # the capture pass and back afterwards, so generation elsewhere keeps
        # the fast kernel.
        restore_impl = None
        if self.debug_mode:
            impl = getattr(getattr(self.model, "config", None), "_attn_implementation", None)
            if impl not in (None, "eager") and hasattr(self.model, "set_attn_implementation"):
                try:
                    self.model.set_attn_implementation("eager")
                    restore_impl = impl
                except Exception as exc:  # pragma: no cover - depends on transformers version
                    logger.warning("Could not switch attention to eager for capture: %s", exc)

        # Forward pass
        try:
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    output_hidden_states=True,
                    output_attentions=self.debug_mode,
                    return_dict=True,
                )
        finally:
            if restore_impl is not None:
                try:
                    self.model.set_attn_implementation(restore_impl)
                except Exception:  # pragma: no cover
                    pass
        
        # Extract optional debug info from hooks before removing them
        mlp_activations = None
        attn_outputs = None
        pre_mlp_activations = None
        qkv_outputs = None
        if self._activation_hooks is not None and self._activation_hooks.supported:
            attn_outputs = self._activation_hooks.get_attn_outputs_tuple()
            mlp_activations = self._activation_hooks.get_mlp_activations_copy()
            pre_mlp_activations = self._activation_hooks.get_pre_mlp_copy()
        if self._qkv_hooks is not None and self._qkv_hooks.supported:
            qkv_outputs = self._qkv_hooks.get_qkv_copy()

        # Remove hooks
        if self.debug_mode:
            self._remove_hooks()

        # Extract hidden states (tuple of L+1 tensors)
        hidden_states = tuple(
            hs.cpu().float() for hs in outputs.hidden_states
        )

        # Extract logits
        logits = outputs.logits.cpu().float()

        # Extract optional debug info
        attention_weights = None
        if self.debug_mode and hasattr(outputs, "attentions"):
            attention_weights = outputs.attentions

        if attn_outputs is None and mlp_activations is None and self.debug_mode and self.mlp_activations:
            # Legacy path when hook registry not used (e.g. unsupported arch)
            mlp_activations = self.mlp_activations.copy()
            self.mlp_activations.clear()

        return RunResult(
            input_ids=input_ids.cpu(),
            token_strs=token_strs,
            hidden_states=hidden_states,
            logits=logits,
            attention_weights=attention_weights,
            mlp_activations=mlp_activations,
            attn_outputs=attn_outputs,
            pre_mlp_activations=pre_mlp_activations,
            qkv_outputs=qkv_outputs,
        )

    def _register_debug_hooks(self) -> None:
        """Register hooks for capturing attention outputs and MLP activations (multi-arch)."""
        self.hooks = []
        self.mlp_activations = {}

        capture_attn_out = True
        capture_mlp = True
        capture_pre_mlp = False
        if self.config is not None:
            capture_attn_out = getattr(
                self.config, "enable_attn_output_capture", True
            )
            capture_mlp = getattr(self.config, "enable_mlp_capture", True)
            capture_pre_mlp = getattr(
                self.config, "enable_pre_mlp_capture", False
            )

        self._activation_hooks = ActivationHooks(
            self.model,
            capture_attn_output=capture_attn_out,
            capture_mlp_output=capture_mlp,
            capture_pre_mlp=capture_pre_mlp,
        )
        if self._activation_hooks.supported:
            self._activation_hooks.register()
        capture_qkv = self.config is not None and getattr(self.config, "enable_qkv_capture", False)
        if capture_qkv:
            self._qkv_hooks = QKVHooks(self.model)
            if self._qkv_hooks.supported:
                self._qkv_hooks.register()
            else:
                self._qkv_hooks = None
        else:
            self._qkv_hooks = None
        if not self._activation_hooks.supported:
            self._activation_hooks = None

    def _remove_hooks(self) -> None:
        """Remove all registered hooks."""
        if self._activation_hooks is not None:
            self._activation_hooks.remove()
            self._activation_hooks = None
        if self._qkv_hooks is not None:
            self._qkv_hooks.remove()
            self._qkv_hooks = None
        for hook in self.hooks:
            try:
                hook.remove()
            except Exception:
                pass
        self.hooks = []
