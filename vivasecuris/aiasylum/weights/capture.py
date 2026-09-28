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
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# Version of the (text, special-tokens) contract that ``format_chat`` produces for
# a given tokenizer and message list. Recorded on every derived direction and
# probe so an artifact can say which contract it was captured under. Bump it
# when the no-template text, the special-tokens rule, ``add_generation_prompt``
# or the thinking default changes -- never for a model, corpus or method change.
#
#   1  pre-fix: no-template text was the raw prompt with the system prompt
#      dropped, and ``add_special_tokens`` was keyed on the caller's *request*
#      rather than on whether a template was applied, so base models got no BOS.
#   2  ``format_chat``: plain text keeps the system prompt and ends in a blank
#      line, and special tokens are added exactly when no template rendered them.
PROMPT_FORMAT_VERSION = 2


class NoChatTemplateError(ValueError):
    """A caller required a chat template and the tokenizer has none.

    Deriving a refusal direction from raw prompts against a base model is a
    different experiment from deriving it through the model's chat template,
    so it has to be asked for rather than fallen into. ``ValueError`` so the
    run loop's existing handling marks the run failed with this message.
    """


def has_chat_template(tokenizer) -> bool:
    """Whether the tokenizer carries a usable chat template.

    ``bool()`` rather than ``is not None``: transformers treats an empty string
    as a real template and renders every prompt to ``""``, which is precisely
    the plausible-looking wrong result this module exists to refuse.
    """
    return bool(getattr(tokenizer, "chat_template", None))


def render_plain(messages: List[Dict[str, str]]) -> str:
    """The one no-template rendering: every message, joined, ending in a blank line.

    The system prompt is kept. The trailing separator is what plays the role
    of ``add_generation_prompt=True`` on a base model: without it the last
    prompt token is the last word of the question, and greedy generation tends
    to continue the question rather than answer it. It is also the text
    ``lora.build_examples`` trains on, so a base-model adapter's training
    positions and its later capture and serving positions coincide.
    """
    return "\n\n".join(m.get("content", "") for m in messages) + "\n\n"


_warned_no_template: set = set()


def _warn_no_template_once(tokenizer) -> None:
    key = getattr(tokenizer, "name_or_path", "") or f"id:{id(tokenizer)}"
    if key in _warned_no_template:
        return
    _warned_no_template.add(key)
    logger.warning(
        "Tokenizer %s has no chat template: prompts are rendered as plain text with "
        "a trailing blank line and the tokenizer's own special tokens (BOS) added.",
        key,
    )


def format_chat(
    tokenizer, messages: List[Dict[str, str]], *, thinking: bool = False
) -> Tuple[str, bool]:
    """Render one message list. The single source of truth for prompt text.

    Returns ``(text, template_applied)``. Every caller must tokenize with
    ``add_special_tokens=not template_applied``: a chat template carries its own
    special tokens (``{{ bos_token }}``, ``<|begin_of_text|>``, ``<bos>``), and
    plain text needs the tokenizer to add them. Keying that flag on whether a
    template was *requested* rather than *applied* is the bug this replaces.
    """
    if not has_chat_template(tokenizer):
        _warn_no_template_once(tokenizer)
        return render_plain(messages), False
    return render_chat(tokenizer, messages, thinking=thinking), True


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
    """Drop the reasoning block so only the answer is scored.

    One rule for the whole framework, in ``reasoning.split_reasoning``: the
    behavioural scorers, the benchmark parser and this module all read the same
    visible answer, including the closing-tag-only trace a template that
    pre-fills ``<think>`` produces.
    """
    from vivasecuris.aiasylum.reasoning import split_reasoning

    return split_reasoning(text or "")[0].lstrip()


def format_prompts(
    tokenizer,
    prompts: Sequence[str],
    system_prompt: Optional[str] = None,
    thinking: bool = False,
) -> Tuple[List[str], bool]:
    """Wrap raw prompts as single-turn chats: ``(texts, template_applied)``.

    Built on :func:`format_chat`, so the same text is produced at capture and
    at generation time -- or the direction is derived from positions the model
    never actually sees. Returns a tuple on purpose: a caller that still does
    ``format_prompts(...)[0]`` and hands the result to a tokenizer fails at its
    first call, rather than silently keying ``add_special_tokens`` on its own
    request as every caller once did.
    """
    applied = has_chat_template(tokenizer)   # defined even for an empty prompt list
    texts: List[str] = []
    for prompt in prompts:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        text, applied = format_chat(tokenizer, messages, thinking=thinking)
        texts.append(text)
    return texts, applied


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
    require_template: bool = False,
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
    # Refuse before formatting so the no-template warning does not fire on a raise.
    if apply_template and require_template and not has_chat_template(tokenizer):
        raise NoChatTemplateError(
            "This tokenizer has no chat template, and this capture requires one. "
            "Capturing plain-text prompts against a base model is a different "
            "experiment; pass allow_no_chat_template to run it deliberately."
        )
    if apply_template:
        texts, applied = format_prompts(tokenizer, texts, system_prompt, thinking=thinking)
    else:
        applied = False
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
                max_length=max_length, add_special_tokens=not applied,
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
    thinking: bool = False,
    require_template: bool = False,
    progress: Optional[callable] = None,
):
    """Return per-layer last-token hidden states as ``[n_layers+1, n_prompts, d_model]``.

    Results come back on CPU in float32 regardless of model dtype, so a few
    hundred prompts stay well inside memory while the model itself may be on MPS.

    ``thinking`` is part of the formatting a direction is derived under: on a
    hybrid reasoning model the generation prompt differs with it, so the last
    prompt token does too. ``require_template`` refuses a base model rather
    than silently capturing plain text.
    """
    import torch

    # Refuse before formatting so the no-template warning does not fire on a raise.
    if apply_template and require_template and not has_chat_template(tokenizer):
        raise NoChatTemplateError(
            "This tokenizer has no chat template, and direction capture requires one. "
            "Deriving from plain-text prompts against a base model is a different "
            "experiment; pass allow_no_chat_template to run it deliberately."
        )
    if apply_template:
        texts, applied = format_prompts(tokenizer, prompts, system_prompt, thinking=thinking)
    else:
        texts, applied = list(prompts), False
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
                add_special_tokens=not applied,
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
