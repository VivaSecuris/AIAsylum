"""Split a model's private reasoning from the answer it actually gave.

Reasoning models write a working-out before the answer, usually inside
``<think>...</think>``. A refusal detector that reads the whole text scores a
model that muses "I can't help with this" and then complies as having refused;
a harm detector scores "the user wants to make a bomb, so I should refuse" as
harmful content; a judge shown the first 500 characters sees the trace and
none of the answer. Every scorer therefore reads only the *visible answer*,
and this module is the one place that decides what that is. ``ModelResponse``
applies it when a response is built, so ``content`` is the visible answer by
construction; readers of stored text apply it themselves.

Stdlib only: it is imported by ``models.base`` and ``weights.capture``, both of
which must stay cheap to import.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Optional, Sequence, Tuple

# Tags real emitters use. ``analysis`` is deliberately not here: gpt-oss uses
# harmony channel markers rather than an ``<analysis>`` tag, and a doctor that
# asks for "an analysis" can elicit that tag as ordinary structured output. The
# benchmark answer parser opts into it explicitly.
DEFAULT_TAGS: Tuple[str, ...] = ("think", "thinking")

# Bump when the rule below changes what counts as the visible answer. Stored
# on every test result the runner writes, so a refusal flag computed before
# the rule existed is recognisable by the stamp's absence.
REASONING_SPLIT_VERSION = 1


@lru_cache(maxsize=8)
def _patterns(tags: Tuple[str, ...]):
    alt = "|".join(re.escape(t) for t in tags)
    return (
        re.compile(rf"</(?:{alt})>", re.I),   # a closing tag
        re.compile(rf"<(?:{alt})>", re.I),    # an opening tag
        re.compile(rf"</?(?:{alt})>", re.I),  # either, for stripping
    )


def split_reasoning(text: Optional[str], tags: Sequence[str] = DEFAULT_TAGS) -> Tuple[str, Optional[str]]:
    """Return ``(visible_answer, reasoning)``; ``reasoning`` is None when no block is present.

    The rule, chosen so the benchmark parser's existing behaviour is preserved
    exactly and no scoring version has to move:

    * A closing tag exists: the answer is everything after the *last* closing
      tag; the reasoning is everything before it, tags removed. This covers a
      template that pre-fills ``<think>`` in the generation prompt, so the
      completion starts mid-trace and carries only the closing tag.
    * An opening tag with no closing tag after it (the trace was cut off at the
      token limit): there is no answer yet -- the answer is ``""`` and the whole
      remainder is reasoning. ``finish_reason`` tells the rest of that story.
    * No tag: ``(text, None)``. An empty block gives reasoning ``""``, not None,
      so "present but empty" stays distinguishable.

    Matching is case-insensitive. Several blocks fold into one reasoning string.
    """
    text = text or ""
    if "<" not in text:
        return text, None
    closing, opening, any_tag = _patterns(tuple(tags))
    closes = list(closing.finditer(text))
    if closes:
        head, tail = text[: closes[-1].end()], text[closes[-1].end():]
    else:
        head, tail = "", text
    if opening.search(tail):
        head, tail = text, ""
    if not head:
        return text, None
    return tail.lstrip(), any_tag.sub("", head).strip()


def visible_answer(text: Optional[str], tags: Sequence[str] = DEFAULT_TAGS) -> str:
    """The answer a user would have seen: :func:`split_reasoning` without the trace."""
    return split_reasoning(text, tags)[0]
