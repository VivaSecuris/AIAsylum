"""Token remap: emit one token wherever the model would have emitted another.

The last thing a causal LM does is turn the final residual ``h`` into one score
per vocabulary entry. With ``W_U`` the output head (``lm_head.weight``, shape
``[vocab, d_model]``) and ``w_v`` its row for token ``v``::

    l_v = w_v . h

Nothing else reads ``W_U``, so it is a lookup table: whichever token's row sits
at index ``v`` is the token that gets ``v``'s score. Moving a row moves a token.

Two layers, mirroring ``steer`` (preview) before ``ablate`` (saved edit):

**Decode time** -- :class:`TokenRemapProcessor` rewrites the scores at each
generation step. Nothing is written, and it can do what weights cannot: set a
score to ``-inf``, so the source token is banned outright.

**Weights** -- :func:`apply_token_remap` moves rows of the head. Three modes,
for a source row ``a`` and a target row ``b``:

``swap``   exchange the rows             ``l_a`` and ``l_b`` trade places. Exact,
                                         and applying it again restores every bit.
``copy``   ``w_b <- w_a``, ``w_a <- 0``   ``b`` gets ``a``'s score; the model can no
                                         longer say ``b`` when it means ``b``.
``merge``  ``w_b <- w_b + w_a``,         ``l_b = old l_b + old l_a``: ``b`` wins when
           ``w_a <- 0``                  either was wanted. Approximate.

Why the source row is zeroed rather than banned: ``l_a = w_a . h`` is finite for
every finite row, so no row makes it ``-inf`` for all ``h``. A zero row gives
``l_a = 0`` always, which loses to any token the model actually wants. That is a
suppression, not a ban.

``swap`` and ``copy`` move bit patterns and do no arithmetic, so they are exact
in any dtype, bfloat16 included. ``merge`` adds, so it runs in float32 and casts
back, like the projection edits in :mod:`weights.surgery`.

Three things make "one token" harder than it sounds:

*Tokens are not words.* ``Hello``, `` Hello``, ``hello`` and `` hello`` are four
different rows, and a target must be exactly one token. On the Qwen2.5
tokenizer ``Bye`` is two tokens and `` Bye`` is one.

*Tied embeddings.* Qwen2.5 at 0.5B/1.5B/3B shares one tensor between the head
and the input embedding table. A head-only edit must untie them first, or
``save_pretrained`` drops ``lm_head.weight`` as a duplicate and the edit is gone
on reload.

*The model reads its own output.* After emitting ``b`` the next step embeds
``b``. Head-only, the model hears itself say ``b`` and continues from there.
With ``inputs=True`` the embedding rows move too, so it keeps believing it said
``a`` -- at the price that a ``b`` typed by the user is now read as ``a``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from vivasecuris.aiasylum.interp.core.arch import detect_architecture

logger = logging.getLogger(__name__)

METHOD = "token_remap"
MODES = ("swap", "copy", "merge")
DEFAULT_CHECK_PROMPT = "The capital of France is"

# Keys of the summary that belong in the manifest's ``extra``.
_EXTRA_KEYS = (
    "mode", "pairs", "skipped", "head_untied", "inputs_edited", "tied_before",
    "shared_table", "vocab_size", "reversible", "logit_check",
)

_CASES: Tuple[Tuple[str, Callable[[str], str]], ...] = (
    ("as written", lambda s: s),
    ("lower", str.lower),
    ("upper", str.upper),
    ("capitalised", str.capitalize),
)


class RemapError(ValueError):
    """The requested remap cannot be expressed as a move of head rows."""


@dataclass(frozen=True)
class TokenPair:
    src_id: int
    src_text: str
    dst_id: int
    dst_text: str

    def as_dict(self) -> Dict[str, object]:
        return {"src_id": int(self.src_id), "src_text": self.src_text,
                "dst_id": int(self.dst_id), "dst_text": self.dst_text}

    def describe(self) -> str:
        return f"{self.src_text!r} ({self.src_id}) -> {self.dst_text!r} ({self.dst_id})"


@dataclass
class TokenRemap:
    """Resolved pairs, plus the source spellings that could not be paired."""

    pairs: List[TokenPair]
    skipped: List[Dict[str, object]] = field(default_factory=list)
    exact: bool = False

    @property
    def src_ids(self) -> List[int]:
        return [int(p.src_id) for p in self.pairs]

    @property
    def dst_ids(self) -> List[int]:
        return [int(p.dst_id) for p in self.pairs]

    def describe(self) -> List[str]:
        return [p.describe() for p in self.pairs]


# --------------------------------------------------------------------------
# Resolution: words to single-token ids
# --------------------------------------------------------------------------


def token_ids(tokenizer, text: str) -> List[int]:
    return [int(i) for i in tokenizer.encode(text, add_special_tokens=False)]


def single_token_id(tokenizer, text: str) -> Optional[int]:
    """The id ``text`` encodes to, or ``None`` when it is not exactly one real token.

    A word-level tokenizer maps a spelling it has never seen to its unknown
    token. That is one id, but it is not that spelling's row.
    """
    ids = token_ids(tokenizer, text)
    if len(ids) != 1:
        return None
    unk = getattr(tokenizer, "unk_token_id", None)
    if unk is not None and ids[0] == unk:
        return None
    return ids[0]


def _why_not_single(ids: Sequence[int]) -> str:
    if len(ids) == 1:
        return "not in the vocabulary (it encodes to the unknown token)"
    return f"{len(ids)} tokens {list(ids)} in this tokenizer"


def _shapes(word: str) -> List[Tuple[str, Callable[[str], str]]]:
    """Every (leading space, case) form of ``word`` as a text transform.

    The first entry is the word exactly as typed; the rest are built from its
    stripped core. Each transform can be applied to another word to get the
    same-shaped spelling of *that* word, which is how variants are paired.
    """
    shapes: List[Tuple[str, Callable[[str], str]]] = [("as typed", None)]  # type: ignore[list-item]
    for case_name, fn in _CASES:
        for space in ("", " "):
            label = f"{'leading space, ' if space else ''}{case_name}"
            shapes.append((label, (lambda w, _s=space, _f=fn: _s + _f(w.strip()))))
    return shapes


def surface_variants(word: str) -> List[Tuple[str, str]]:
    """``(label, text)`` for every spelling of ``word``, deduplicated, order kept."""
    out: List[Tuple[str, str]] = []
    seen = set()
    for label, fn in _shapes(word):
        text = word if fn is None else fn(word)
        if text.strip() and text not in seen:
            seen.add(text)
            out.append((label, text))
    return out


def source_variant_ids(tokenizer, word: str) -> List[int]:
    """Ids of the single-token spellings of ``word``, in variant order."""
    ids: List[int] = []
    for _label, text in surface_variants(word):
        tid = single_token_id(tokenizer, text)
        if tid is not None and tid not in ids:
            ids.append(tid)
    return ids


def single_token_spellings(tokenizer, word: str) -> List[Tuple[str, int]]:
    """``(text, id)`` for each spelling of ``word`` that is exactly one token."""
    out: List[Tuple[str, int]] = []
    for _label, text in surface_variants(word):
        tid = single_token_id(tokenizer, text)
        if tid is not None:
            out.append((text, tid))
    return out


def _not_single_message(tokenizer, role: str, text: str, flag: str) -> str:
    ids = token_ids(tokenizer, text)
    spellings = single_token_spellings(tokenizer, text)
    head = (
        f"{role} {text!r} is {_why_not_single(ids)}. The output head has "
        f"one row per token, so it must be a single token."
    )
    if not spellings:
        return head + " No spelling of it is a single token; choose a different word."
    listed = ", ".join(f"{t!r} ({i})" for t, i in spellings)
    return head + f" Single-token spellings: {listed}. Pass one of these to {flag} (quote a leading space)."


def resolve_remap(tokenizer, src: str, dst: str, *, exact: bool = False) -> TokenRemap:
    """Turn a source and a target word into pairs of single-token ids.

    ``exact=True`` takes both strings literally and requires each to be one
    token. Otherwise every spelling of ``src`` that is one token is paired with
    the same-shaped spelling of ``dst``; a spelling whose target is not one
    token, or whose ids are already in use, lands in ``skipped`` with the
    reason. Raises :class:`RemapError` when nothing can be paired.
    """
    if not src or not src.strip():
        raise RemapError("The source word is empty.")
    if not dst or not dst.strip():
        raise RemapError("The target word is empty.")

    if exact:
        src_id = single_token_id(tokenizer, src)
        if src_id is None:
            raise RemapError(_not_single_message(tokenizer, "Source", src, "--from"))
        dst_id = single_token_id(tokenizer, dst)
        if dst_id is None:
            raise RemapError(_not_single_message(tokenizer, "Target", dst, "--to"))
        if src_id == dst_id:
            raise RemapError(f"Source and target are the same token ({src_id}); there is nothing to remap.")
        return TokenRemap([TokenPair(src_id, src, dst_id, dst)], [], exact=True)

    pairs: List[TokenPair] = []
    skipped: List[Dict[str, object]] = []
    seen_text = set()
    used: Dict[int, str] = {}
    for _label, fn in _shapes(src):
        src_text = src if fn is None else fn(src)
        dst_text = dst if fn is None else fn(dst)
        if not src_text.strip() or src_text in seen_text:
            continue
        seen_text.add(src_text)
        src_id = single_token_id(tokenizer, src_text)
        if src_id is None:
            continue  # this spelling of the source is not a row of the head at all
        dst_ids = token_ids(tokenizer, dst_text)
        entry: Dict[str, object] = {"src_text": src_text, "src_id": src_id,
                                    "dst_text": dst_text, "dst_ids": dst_ids}
        dst_id = single_token_id(tokenizer, dst_text)
        if dst_id is None:
            entry["reason"] = (
                f"target is {len(dst_ids)} tokens" if len(dst_ids) != 1
                else "target is not in the vocabulary"
            )
            skipped.append(entry)
            continue
        if dst_id == src_id:
            entry["reason"] = "source and target are the same token"
            skipped.append(entry)
            continue
        clash = used.get(src_id) or used.get(dst_id)
        if clash:
            entry["reason"] = f"a token of this pair is already used by {clash}"
            skipped.append(entry)
            continue
        pair = TokenPair(src_id, src_text, dst_id, dst_text)
        used[src_id] = used[dst_id] = pair.describe()
        pairs.append(pair)

    if not pairs:
        if single_token_id(tokenizer, dst) is None and not single_token_spellings(tokenizer, dst):
            raise RemapError(_not_single_message(tokenizer, "Target", dst, "--to"))
        if not source_variant_ids(tokenizer, src):
            raise RemapError(_not_single_message(tokenizer, "Source", src, "--from"))
        reasons = "; ".join(f"{s['src_text']!r} -> {s['dst_text']!r}: {s['reason']}" for s in skipped)
        raise RemapError(
            f"No spelling of {src!r} could be paired with a single-token spelling of {dst!r} "
            f"({reasons}). Choose the pair yourself with --exact."
        )
    return TokenRemap(pairs, skipped, exact=False)


def _pair_ids(pairs: Sequence[object]) -> Tuple[List[int], List[int]]:
    """Source and target ids from ``TokenPair`` objects or ``(src, dst)`` tuples."""
    src: List[int] = []
    dst: List[int] = []
    for p in pairs:
        if isinstance(p, TokenPair):
            src.append(int(p.src_id))
            dst.append(int(p.dst_id))
        else:
            a, b = p  # type: ignore[misc]
            src.append(int(a))
            dst.append(int(b))
    return src, dst


def validate_pairs(pairs: Sequence[object], n_rows: Optional[int] = None) -> Tuple[List[int], List[int]]:
    """Check the pairs describe disjoint moves. Returns ``(src_ids, dst_ids)``.

    Every id may appear once. A repeated id would make ``swap`` something other
    than its own inverse, and would give ``copy`` two sources for one row.
    """
    src, dst = _pair_ids(pairs)
    if not src:
        raise RemapError("The remap holds no pairs.")
    seen = set()
    for a, b in zip(src, dst):
        if a == b:
            raise RemapError(f"Source and target are the same token ({a}).")
        for tid in (a, b):
            if tid < 0:
                raise RemapError(f"Token id {tid} is negative.")
            if n_rows is not None and tid >= n_rows:
                raise RemapError(f"Token id {tid} is past the end of a {n_rows}-row table.")
            if tid in seen:
                raise RemapError(
                    f"Token id {tid} appears in more than one pair. Each row can take part in "
                    f"one move; run separate remaps against the source model instead."
                )
            seen.add(tid)
    return src, dst


# --------------------------------------------------------------------------
# Decode time
# --------------------------------------------------------------------------


class TokenRemapProcessor:
    """Give each target the better of its own score and its source's, then ban the source.

    Duck-typed ``transformers`` logits processor (``LogitsProcessorList`` only
    inspects the call signature), so importing this module does not import
    transformers. ``max`` rather than assignment keeps a target the model
    already preferred; if the source was the argmax, the target becomes it.
    """

    def __init__(self, pairs: Sequence[object]) -> None:
        self.src_ids, self.dst_ids = validate_pairs(pairs)

    def __call__(self, input_ids, scores):
        import torch

        src = torch.as_tensor(self.src_ids, device=scores.device, dtype=torch.long)
        dst = torch.as_tensor(self.dst_ids, device=scores.device, dtype=torch.long)
        out = scores.clone()
        moved = scores[:, src]  # read every source before writing anything
        out[:, dst] = torch.maximum(scores[:, dst], moved)
        out[:, src] = float("-inf")
        return out


def remap_processor_list(remap: TokenRemap):
    from transformers import LogitsProcessorList

    return LogitsProcessorList([TokenRemapProcessor(remap.pairs)])


def _encode_prompt(model, tokenizer, prompt: str, apply_template: bool = True):
    from vivasecuris.aiasylum.weights.capture import format_prompts

    texts, applied = format_prompts(tokenizer, [prompt]) if apply_template else ([prompt], False)
    return tokenizer(texts[0], return_tensors="pt", add_special_tokens=not applied).to(model.device)


def generate_once(
    model,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 32,
    apply_template: bool = True,
    logits_processor=None,
) -> str:
    """One greedy completion through the provider's own kwargs builder."""
    import torch

    from vivasecuris.aiasylum.models.transformers_local import generation_kwargs

    inputs = _encode_prompt(model, tokenizer, prompt, apply_template)
    gen = generation_kwargs(0, max_new_tokens, tokenizer.pad_token_id or tokenizer.eos_token_id)
    if logits_processor is not None:
        gen["logits_processor"] = logits_processor
    with torch.no_grad():
        out = model.generate(**inputs, **gen)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def preview_remap(
    model, tokenizer, prompt: str, remap: TokenRemap, *,
    max_new_tokens: int = 32, apply_template: bool = True,
) -> Tuple[str, str]:
    """``(baseline, remapped)`` for one prompt. No weight is touched."""
    baseline = generate_once(model, tokenizer, prompt, max_new_tokens=max_new_tokens,
                             apply_template=apply_template)
    remapped = generate_once(model, tokenizer, prompt, max_new_tokens=max_new_tokens,
                             apply_template=apply_template,
                             logits_processor=remap_processor_list(remap))
    return baseline, remapped


# --------------------------------------------------------------------------
# Weights
# --------------------------------------------------------------------------


def _shares_storage(a, b) -> bool:
    return (
        a is not None and b is not None and hasattr(a, "weight") and hasattr(b, "weight")
        and a.weight.data_ptr() == b.weight.data_ptr()
    )


def untie_output_embeddings(model) -> bool:
    """Give the output head its own copy of a shared table. True when it untied.

    The parameter inside the existing module is replaced, so module identity and
    any hooks survive. ``config.tie_word_embeddings`` is cleared so
    ``save_pretrained`` writes both tensors and ``from_pretrained`` does not tie
    them again on reload. The class attribute ``_tied_weights_keys`` is left
    alone: it is shared by every instance in the process, and it only matters
    for tensors that still share storage.
    """
    import torch
    from torch import nn

    head = model.get_output_embeddings()
    emb = model.get_input_embeddings()
    if not _shares_storage(head, emb):
        return False
    with torch.no_grad():
        head.weight = nn.Parameter(
            head.weight.detach().clone(), requires_grad=head.weight.requires_grad
        )
    model.config.tie_word_embeddings = False
    get_text_config = getattr(model.config, "get_text_config", None)
    if callable(get_text_config):
        try:
            text_config = get_text_config(decoder=True)
        except TypeError:
            text_config = get_text_config()
        if text_config is not None and text_config is not model.config:
            text_config.tie_word_embeddings = False
    logger.info("Untied lm_head from the embedding table; the head now has its own storage")
    return True


def _frobenius(W, chunk: int = 8192) -> float:
    """Frobenius norm in float32, over row blocks, without copying the table."""
    import torch

    total = 0.0
    for start in range(0, int(W.shape[0]), chunk):
        block = W[start:start + chunk].to(torch.float32)
        total += float((block * block).sum())
    return total ** 0.5


def move_rows(weight, src_ids: Sequence[int], dst_ids: Sequence[int], mode: str) -> float:
    """Apply one row operation in place. Returns the relative Frobenius change."""
    import torch

    if mode not in MODES:
        raise RemapError(f"mode must be one of {', '.join(MODES)}; got {mode!r}")
    with torch.no_grad():
        W = weight.data
        src = torch.as_tensor(list(src_ids), device=W.device, dtype=torch.long)
        dst = torch.as_tensor(list(dst_ids), device=W.device, dtype=torch.long)
        touched = torch.cat([src, dst])
        before = W[touched].to(torch.float32)
        norm = _frobenius(W)
        if mode == "swap":
            # The right-hand side is an advanced index, so it is a copy taken
            # before anything is written: a true exchange.
            W[touched] = W[torch.cat([dst, src])]
        elif mode == "copy":
            W[dst] = W[src]
            W[src] = 0
        else:  # merge
            merged = W[dst].to(torch.float32) + W[src].to(torch.float32)
            W[dst] = merged.to(W.dtype)
            W[src] = 0
        delta = float((W[touched].to(torch.float32) - before).norm())
    return delta / norm if norm > 0 else 0.0


def apply_token_remap(
    model,
    tokenizer,
    remap: TokenRemap,
    *,
    mode: str = "swap",
    inputs: bool = False,
) -> Dict[str, object]:
    """Move rows of the output head in place. Returns a manifest-ready summary.

    ======  ==========  =====================================================
    model   ``inputs``  what happens
    ======  ==========  =====================================================
    tied    False       untie, then edit the head only
    tied    True        one operation on the shared table does both edits
    untied  False       edit the head only
    untied  True        edit the head, then the embedding table
    ======  ==========  =====================================================
    """
    if mode not in MODES:
        raise RemapError(f"mode must be one of {', '.join(MODES)}; got {mode!r}")
    head = model.get_output_embeddings()
    emb = model.get_input_embeddings()
    if head is None or not hasattr(head, "weight"):
        raise RemapError("This model exposes no output head to edit.")
    if inputs and (emb is None or not hasattr(emb, "weight")):
        raise RemapError("This model exposes no input embedding table to edit.")

    n_rows = int(head.weight.shape[0])
    if inputs:
        n_rows = min(n_rows, int(emb.weight.shape[0]))
    # Checked against the table, not the tokenizer: real checkpoints carry
    # spare rows past the last tokenizer entry.
    src, dst = validate_pairs(remap.pairs, n_rows=n_rows)

    tied_before = _shares_storage(head, emb)
    head_untied = False
    shared_table = False
    changes: List[float] = []

    if tied_before and not inputs:
        head_untied = untie_output_embeddings(model)
        head = model.get_output_embeddings()
        changes.append(move_rows(head.weight, src, dst, mode))
    elif tied_before and inputs:
        shared_table = True
        logger.warning(
            "lm_head is tied to the embedding table; one row operation edits both. "
            "Recorded in the manifest."
        )
        changes.append(move_rows(head.weight, src, dst, mode))
    else:
        changes.append(move_rows(head.weight, src, dst, mode))
        if inputs:
            changes.append(move_rows(emb.weight, src, dst, mode))

    tied_after = _shares_storage(model.get_output_embeddings(), model.get_input_embeddings())
    summary: Dict[str, object] = {
        "architecture": detect_architecture(model),
        "model_type": getattr(model.config, "model_type", None),
        "matrices_edited": len(changes),
        "embeddings_tied": tied_after,
        "embeddings_edited": bool(inputs),
        "mean_relative_change": sum(changes) / len(changes),
        "coverage_verified": None,
        "mode": mode,
        "pairs": [p.as_dict() if isinstance(p, TokenPair) else
                  {"src_id": int(p[0]), "dst_id": int(p[1])} for p in remap.pairs],
        "skipped": list(remap.skipped),
        "head_untied": head_untied,
        "inputs_edited": bool(inputs),
        "tied_before": tied_before,
        "shared_table": shared_table,
        "vocab_size": int(head.weight.shape[0]),
        "reversible": mode == "swap",
    }
    logger.info(
        "Token remap (%s) over %d pair(s): %d table(s) edited, tied %s -> %s, untied=%s",
        mode, len(src), len(changes), tied_before, tied_after, head_untied,
    )
    return summary


def manifest_extra(summary: Dict[str, object]) -> Dict[str, object]:
    return {key: summary[key] for key in _EXTRA_KEYS if key in summary}


# --------------------------------------------------------------------------
# The check that the edit did what it says
# --------------------------------------------------------------------------


def next_token_logits(model, tokenizer, prompt: str, *, apply_template: bool = True):
    """Scores for the token after ``prompt``: a float32 ``[vocab]`` tensor on CPU."""
    import torch

    inputs = _encode_prompt(model, tokenizer, prompt, apply_template)
    with torch.no_grad():
        logits = model(**inputs).logits
    return logits[0, -1].detach().to(torch.float32).cpu().clone()


def prompt_mentions_pairs(tokenizer, prompt: str, remap: TokenRemap, *, apply_template: bool = True) -> bool:
    """Whether the formatted prompt contains a token the remap moves.

    With the embedding rows edited such a prompt reads differently afterwards,
    so its residual changes and the logit comparison no longer isolates the head.
    """
    from vivasecuris.aiasylum.weights.capture import format_prompts

    texts, applied = format_prompts(tokenizer, [prompt]) if apply_template else ([prompt], False)
    ids = set(tokenizer(texts[0], add_special_tokens=not applied)["input_ids"])
    return bool(ids & (set(remap.src_ids) | set(remap.dst_ids)))


def check_remap_logits(before, after, remap: TokenRemap, mode: str, *,
                       atol: float = 1e-3, rtol: float = 1e-3) -> Dict[str, object]:
    """Compare next-token scores before and after a head edit.

    The moved columns must hold what the mode promises and every other column
    must be unchanged. A moved row is computed at a new position in the matrix
    product, so its score is compared within a tolerance rather than bit for
    bit; ``merge`` rounds the summed row into the model's dtype and gets a
    wider one.
    """
    import torch

    if mode not in MODES:
        raise RemapError(f"mode must be one of {', '.join(MODES)}; got {mode!r}")
    src = torch.as_tensor(remap.src_ids, dtype=torch.long)
    dst = torch.as_tensor(remap.dst_ids, dtype=torch.long)
    before = before.to(torch.float32).cpu()
    after = after.to(torch.float32).cpu()

    if mode == "swap":
        want_dst, want_src = before[src], before[dst]
    elif mode == "copy":
        want_dst, want_src = before[src], torch.zeros_like(before[src])
    else:
        want_dst, want_src = before[dst] + before[src], torch.zeros_like(before[src])
        atol, rtol = max(atol, 0.25), max(rtol, 0.02)

    def worst(got, want) -> Tuple[float, bool]:
        diff = (got - want).abs()
        ok = bool((diff <= atol + rtol * want.abs()).all())
        return (float(diff.max()) if diff.numel() else 0.0), ok

    dst_diff, dst_ok = worst(after[dst], want_dst)
    src_diff, src_ok = worst(after[src], want_src)

    others = torch.ones(before.shape[-1], dtype=torch.bool)
    others[src] = False
    others[dst] = False
    other_diff = float((after[others] - before[others]).abs().max()) if bool(others.any()) else 0.0
    other_ok = other_diff <= atol

    problems = []
    if not dst_ok:
        problems.append(f"target columns are off by up to {dst_diff:.4g}")
    if not src_ok:
        problems.append(f"source columns are off by up to {src_diff:.4g}")
    if not other_ok:
        problems.append(f"untouched columns moved by up to {other_diff:.4g}")
    return {
        "ok": not problems,
        "mode": mode,
        "max_target_diff": dst_diff,
        "max_source_diff": src_diff,
        "max_other_diff": other_diff,
        "detail": "; ".join(problems) or "moved columns match and nothing else changed",
    }


# --------------------------------------------------------------------------
# Save
# --------------------------------------------------------------------------


def _refuse_non_empty(out_dir: str) -> None:
    from pathlib import Path

    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(
            f"{out} already exists and is not empty. Choose another path or remove it; "
            f"overwriting a model directory in place is not done implicitly."
        )


def apply_and_save_remap(
    model,
    tokenizer,
    remap: TokenRemap,
    out_dir: str,
    *,
    source_model: str,
    mode: str = "swap",
    inputs: bool = False,
    notes: Optional[str] = None,
    reporter=None,
    check_prompt: Optional[str] = DEFAULT_CHECK_PROMPT,
) -> Tuple[str, Dict[str, object]]:
    """Edit ``model`` in place, check the scores moved as promised, and save it.

    Returns ``(path, summary)``. ``summary["logit_check"]`` holds the comparison,
    or says why it was skipped. A failed check raises before anything is
    written: this is the coverage assertion for the edit.
    """
    from contextlib import nullcontext

    from vivasecuris.aiasylum.weights.surgery import save_edited_model

    _refuse_non_empty(out_dir)
    step = reporter.step if reporter is not None else (lambda name: nullcontext())

    before = None
    skipped_check = None
    if not check_prompt:
        skipped_check = "no check prompt given"
    elif inputs and prompt_mentions_pairs(tokenizer, check_prompt, remap):
        skipped_check = "the check prompt contains a remapped token and the embedding rows are edited"
    else:
        with step("scoring the next token before the edit"):
            before = next_token_logits(model, tokenizer, check_prompt)

    label = f"moving {len(remap.pairs)} row pair(s) of the output head ({mode})"
    if inputs:
        label += " and the embedding table"
    with step(label):
        summary = apply_token_remap(model, tokenizer, remap, mode=mode, inputs=inputs)

    if before is None:
        summary["logit_check"] = {"ok": None, "skipped": skipped_check}
    else:
        with step("scoring the next token after the edit"):
            after = next_token_logits(model, tokenizer, check_prompt)
        check = check_remap_logits(before, after, remap, mode)
        check["prompt"] = check_prompt
        summary["logit_check"] = check
        if not check["ok"]:
            raise RuntimeError(
                f"The remap did not move the scores as promised ({check['detail']}). "
                f"Nothing was written."
            )

    if reporter is not None:
        reporter.note(
            f"{summary['matrices_edited']} table(s) edited, "
            f"mean relative change {summary['mean_relative_change']:.6f}"
        )

    path = save_edited_model(
        model, tokenizer, str(out_dir),
        source_model=source_model, direction=None, method=METHOD, summary=summary,
        notes=notes, extra=manifest_extra(summary), reporter=reporter,
    )
    return path, summary


def remap_and_save(
    source_model: str,
    src: str,
    dst: str,
    out_dir: str,
    *,
    exact: bool = False,
    mode: str = "swap",
    inputs: bool = False,
    device: str = "cpu",
    dtype: str = "bfloat16",
    notes: Optional[str] = None,
    reporter=None,
) -> str:
    """Load ``source_model``, remap ``src`` to ``dst``, and save a new model directory."""
    from contextlib import nullcontext

    from vivasecuris.aiasylum.interp.core.loader import load

    _refuse_non_empty(out_dir)
    step = reporter.step if reporter is not None else (lambda name: nullcontext())
    with step(f"loading {source_model}"):
        model, tokenizer = load(source_model, device=device, dtype=dtype, seed=None)
    remap = resolve_remap(tokenizer, src, dst, exact=exact)
    path, _summary = apply_and_save_remap(
        model, tokenizer, remap, out_dir, source_model=source_model, mode=mode,
        inputs=inputs, notes=notes, reporter=reporter,
    )
    return path
