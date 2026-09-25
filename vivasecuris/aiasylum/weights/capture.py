"""Residual-stream capture for direction derivation.

Deliberately narrow: batched last-prompt-token hidden states across all layers.
Rich per-token capture (attention maps, MLP activations, Q/K/V) lives in the
vendored interpretability engine at ``interp.core.runner``; this module exists
because direction derivation runs over hundreds of prompts and needs to be
batched and memory-frugal rather than exhaustive.

Why the last prompt token: with ``add_generation_prompt=True`` the final
position is the start of the assistant turn, which is where the model has
committed to answering or refusing. Earlier positions are still reading the
question.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

# Qwen3 and other hybrid reasoning models open the reply with a <think> block.
_THINK_BLOCK = re.compile(r"<think>.*?(?:</think>|$)\s*", re.DOTALL)


def render_chat(tokenizer, messages: List[Dict[str, str]], thinking: bool = False) -> str:
    """Apply the chat template, with reasoning off unless ``thinking`` is set.

    Hybrid reasoning models (Qwen3) think by default: the last prompt token is
    then the start of a reasoning trace rather than the answer, and the refusal
    phrases land inside ``<think>`` where they say nothing about the answer.
    Templates that do not know ``enable_thinking`` are rendered unchanged.
    """
    kwargs = {}
    if "enable_thinking" in (getattr(tokenizer, "chat_template", None) or ""):
        kwargs["enable_thinking"] = thinking
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **kwargs
    )


def strip_thinking(text: str) -> str:
    """Drop a leading reasoning block so only the answer is scored."""
    return _THINK_BLOCK.sub("", text or "", count=1).lstrip()


def format_prompts(
    tokenizer,
    prompts: Sequence[str],
    system_prompt: Optional[str] = None,
    thinking: bool = False,
) -> List[str]:
    """Wrap raw prompts in the model's chat template.

    Falls back to the raw text for base models with no template. The same
    formatting must be used at capture and at generation time, or the direction
    is derived from positions the model never actually sees.
    """
    if getattr(tokenizer, "chat_template", None) is None:
        logger.warning("Tokenizer has no chat template; using raw prompts")
        return list(prompts)

    formatted = []
    for prompt in prompts:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        formatted.append(render_chat(tokenizer, messages, thinking=thinking))
    return formatted


def capture_pooled_residuals(
    model,
    tokenizer,
    prompts: Sequence[str],
    pooling: str = "last",
    last_k: int = 8,
    batch_size: int = 8,
    max_length: int = 512,
    apply_template: bool = True,
    system_prompt: Optional[str] = None,
    prompt_suffix: Optional[str] = None,
    thinking: bool = False,
    progress: Optional[callable] = None,
):
    """Per-layer residuals pooled over the prompt: ``[n_layers+1, n_prompts, d_model]``.

    ``capture_last_token_residuals`` reads one position, which is right for
    deriving a direction but is a documented failure mode for a *monitor*:
    final-token-only probes miss signal that is present earlier in the prompt
    (arXiv 2605.12726). ``pooling`` selects how positions are combined, over
    real tokens only -- padding is masked out, so a short prompt in a batch of
    long ones is not averaged with filler.

    - ``last``    the final prompt token, matching the direction pipeline
    - ``mean``    mean over the prompt
    - ``max``     elementwise max over the prompt
    - ``last_k``  mean over the final ``last_k`` real tokens

    ``prompt_suffix`` appends an eliciting question before templating (the
    "prompted probe" of arXiv 2504.20271, the most data-efficient monitor in
    that comparison): the model is asked about the input, so its answer-position
    activations carry the judgement rather than only the content.
    """
    import torch

    if pooling not in ("last", "mean", "max", "last_k"):
        raise ValueError(f"pooling must be last|mean|max|last_k, got {pooling!r}")

    texts = list(prompts)
    if prompt_suffix:
        texts = [f"{p}\n\n{prompt_suffix}" for p in texts]
    if apply_template:
        texts = format_prompts(tokenizer, texts, system_prompt, thinking=thinking)
    if not texts:
        raise ValueError("No prompts to capture")

    original_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    per_batch = []
    try:
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            encoded = tokenizer(
                chunk, return_tensors="pt", padding=True, truncation=True,
                max_length=max_length, add_special_tokens=not apply_template,
            ).to(model.device)
            mask = encoded["attention_mask"].unsqueeze(-1)          # [b, seq, 1]

            with torch.no_grad():
                out = model(**encoded, output_hidden_states=True, return_dict=True)

            pooled = []
            for h in out.hidden_states:                              # [b, seq, d]
                hm = h * mask
                if pooling == "last":
                    pooled.append(h[:, -1, :])
                elif pooling == "mean":
                    pooled.append(hm.sum(dim=1) / mask.sum(dim=1).clamp_min(1))
                elif pooling == "max":
                    # Padding is left-side, so masked positions must not win the
                    # max; push them to -inf rather than zero, which a negative
                    # activation would otherwise lose to.
                    pooled.append(h.masked_fill(mask == 0, float("-inf")).max(dim=1).values)
                else:
                    k = min(int(last_k), h.shape[1])
                    tail, tail_mask = hm[:, -k:, :], mask[:, -k:, :]
                    pooled.append(tail.sum(dim=1) / tail_mask.sum(dim=1).clamp_min(1))
            per_batch.append(torch.stack(pooled, dim=0).to(torch.float32).cpu())

            del out, encoded
            if model.device.type == "mps":
                torch.mps.empty_cache()
            if progress:
                progress(min(start + batch_size, len(texts)), len(texts))
    finally:
        tokenizer.padding_side = original_side

    result = torch.cat(per_batch, dim=1)
    logger.info("Captured pooled(%s) %s", pooling, tuple(result.shape))
    return result


def capture_last_token_residuals(
    model,
    tokenizer,
    prompts: Sequence[str],
    batch_size: int = 8,
    max_length: int = 512,
    apply_template: bool = True,
    system_prompt: Optional[str] = None,
    progress: Optional[callable] = None,
):
    """Return per-layer last-token hidden states as ``[n_layers+1, n_prompts, d_model]``.

    Results come back on CPU in float32 regardless of model dtype, so a few
    hundred prompts stay well inside memory while the model itself may be on MPS.
    """
    import torch

    texts = format_prompts(tokenizer, prompts, system_prompt) if apply_template else list(prompts)
    if not texts:
        raise ValueError("No prompts to capture")

    # Left padding puts the real final token at index -1 for every row in the
    # batch, so the gather below needs no per-row offset arithmetic.
    original_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    per_batch = []
    try:
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            encoded = tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
                add_special_tokens=not apply_template,  # the template already has them
            ).to(model.device)

            with torch.no_grad():
                out = model(**encoded, output_hidden_states=True, return_dict=True)

            # hidden_states: tuple of (n_layers+1) tensors [batch, seq, d_model]
            stacked = torch.stack([h[:, -1, :] for h in out.hidden_states], dim=0)
            per_batch.append(stacked.to(torch.float32).cpu())

            del out, encoded
            if model.device.type == "mps":
                torch.mps.empty_cache()

            if progress:
                progress(min(start + batch_size, len(texts)), len(texts))
    finally:
        tokenizer.padding_side = original_side

    result = torch.cat(per_batch, dim=1)
    logger.info(
        "Captured %s: %d layers x %d prompts x %d dims",
        tuple(result.shape), result.shape[0], result.shape[1], result.shape[2],
    )
    return result
