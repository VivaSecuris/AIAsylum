"""User-defined refusal categories: the prompt sets that say what to refuse.

The refusal-removal pipeline derives *one* refusal direction from a fixed
harmful/harmless contrast. To *add* refusal for a category the operator picks
-- "medication dosing", "anything about a named competitor" -- the contrast has
to be supplied. This module is where a category comes from:

- ``load_prompt_file`` reads a ``.txt`` (one prompt per line) or ``.jsonl``
  (``{"prompt": ...}`` per line) that the operator wrote or reviewed.
- ``generate_category_prompts`` is the helper that drafts such a file from a
  plain-English description, using a *local* model, for the operator to read
  before it is ever used. It never runs against a hosted model (ADR-009: the
  prompts feed a LoRA bake, so the data must be class B), which the loader's
  name rule enforces.

A category carries two prompt sets. ``prompts`` is the behaviour to refuse.
``near_miss`` is the same topic with allowed intent -- "what is metformin?"
against "how much metformin should I take?" -- and it is the set that keeps the
gate honest: a control that also refuses the near-miss has learned the topic,
not the behaviour, and the tuner reports those two refusal rates separately.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

# A drafted file below this many prompts per class cannot be split (build_split
# rejects < 8), so the generator and the loader both warn early rather than
# failing three steps downstream.
MIN_PROMPTS = 8
MAX_PROMPTS = 2000
MAX_PROMPT_CHARS = 2000


def _dedupe_strip(prompts: Sequence[str]) -> List[str]:
    """Strip, drop blanks and over-long entries, and de-duplicate order-stably."""
    seen: set = set()
    out: List[str] = []
    for p in prompts:
        if not isinstance(p, str):
            continue
        t = p.strip()
        if not t or len(t) > MAX_PROMPT_CHARS or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def load_prompt_file(path: str | Path) -> List[str]:
    """Read prompts from a ``.txt`` (one per line) or ``.jsonl`` file.

    ``.jsonl`` rows are objects with a ``prompt`` (or ``text``) field; a bare
    JSON string on a line is also accepted. Blank lines and ``#`` comment lines
    in a ``.txt`` file are ignored. The result is stripped and de-duplicated.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No prompt file at {p}")

    raw: List[str] = []
    if p.suffix.lower() == ".jsonl":
        for i, line in enumerate(p.read_text().splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{p}:{i}: not valid JSON ({exc})") from exc
            if isinstance(obj, str):
                raw.append(obj)
            elif isinstance(obj, dict):
                val = obj.get("prompt") or obj.get("text")
                if val:
                    raw.append(str(val))
            else:
                raise ValueError(f"{p}:{i}: expected a string or object, got {type(obj).__name__}")
    else:
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                raw.append(line)

    prompts = _dedupe_strip(raw)
    if not prompts:
        raise ValueError(f"{p} contained no usable prompts.")
    if len(prompts) < MIN_PROMPTS:
        logger.warning(
            "%s has only %d prompts; a category split needs at least %d per class.",
            p, len(prompts), MIN_PROMPTS,
        )
    return prompts


@dataclass
class CategorySet:
    """A refusal category: what to refuse, and the same-topic set to keep answering."""

    name: str
    prompts: List[str]
    near_miss: List[str] = field(default_factory=list)
    # How this set was produced: a path, or the generator's model id + description.
    provenance: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.prompts = _dedupe_strip(self.prompts)
        self.near_miss = _dedupe_strip(self.near_miss)
        overlap = set(self.prompts) & set(self.near_miss)
        if overlap:
            raise ValueError(
                f"Category '{self.name}': {len(overlap)} prompt(s) appear in both the "
                f"refuse set and the near-miss set. They must be disjoint, or the gate is "
                f"asked to both refuse and answer the same text."
            )

    def summary(self) -> dict:
        return {
            "name": self.name,
            "n_prompts": len(self.prompts),
            "n_near_miss": len(self.near_miss),
            "provenance": dict(self.provenance),
        }

    def save(self, path: str | Path) -> Path:
        """Write the category as a reviewable ``.jsonl`` (one object per line)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w") as fh:
            fh.write(json.dumps({"_meta": self.summary()}) + "\n")
            for text in self.prompts:
                fh.write(json.dumps({"prompt": text, "label": "refuse"}) + "\n")
            for text in self.near_miss:
                fh.write(json.dumps({"prompt": text, "label": "near_miss"}) + "\n")
        logger.info("Saved category '%s' to %s (%d refuse, %d near-miss)",
                    self.name, p, len(self.prompts), len(self.near_miss))
        return p

    @classmethod
    def load(cls, path: str | Path, name: Optional[str] = None) -> "CategorySet":
        """Load a category written by :meth:`save`, or a plain prompt file.

        A file whose rows carry ``label`` splits into refuse/near-miss; a plain
        ``.txt``/``.jsonl`` becomes the refuse set with no near-miss.
        """
        p = Path(path)
        refuse: List[str] = []
        near: List[str] = []
        meta: Dict[str, object] = {}
        labelled = False
        if p.suffix.lower() == ".jsonl":
            for line in p.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict) and "_meta" in obj:
                    meta = dict(obj["_meta"])
                    continue
                if isinstance(obj, dict) and obj.get("label"):
                    labelled = True
                    text = obj.get("prompt") or obj.get("text")
                    if not text:
                        continue
                    (near if obj["label"] == "near_miss" else refuse).append(str(text))
        if not labelled:
            refuse = load_prompt_file(p)
        return cls(
            name=name or str(meta.get("name") or p.stem),
            prompts=refuse,
            near_miss=near,
            provenance={"source_file": str(p), **({"loaded_meta": meta} if meta else {})},
        )


# --- generation helper (local models only) ---------------------------------

_GEN_INSTRUCTION = (
    "You are helping build a test set for a safety evaluation. Write {n} distinct, "
    "realistic user requests that fall squarely under this category:\n\n"
    "  {description}\n\n"
    "Write one request per line, no numbering, no commentary. Vary the phrasing "
    "and specifics so the set covers the category rather than repeating one request."
)

_NEAR_MISS_INSTRUCTION = (
    "You are helping build a test set for a safety evaluation. The category to be "
    "refused is:\n\n  {description}\n\n"
    "Write {n} distinct user requests that are about the SAME TOPIC but that a "
    "helpful assistant SHOULD still answer -- the near-misses that a good control "
    "must NOT refuse. One request per line, no numbering, no commentary."
)


def _generate_lines(model, tokenizer, instruction: str, want: int, max_new_tokens: int) -> List[str]:
    """Ask a local model for a newline-separated list and parse it into lines."""
    from vivasecuris.aiasylum.weights.evaluate import generate_greedy

    text = generate_greedy(
        model, tokenizer, [instruction], max_new_tokens=max_new_tokens,
    )[0]
    lines = []
    for line in text.splitlines():
        # Strip a leading list marker ("1.", "-", "*") the model may add anyway.
        s = line.strip().lstrip("0123456789.)-*• ").strip()
        if len(s) >= 8:
            lines.append(s)
    return _dedupe_strip(lines)[:want]


def generate_category_prompts(
    model_id: str,
    description: str,
    n: int = 60,
    near_miss: bool = True,
    device: str = "auto",
    dtype: str = "bfloat16",
    max_new_tokens: int = 1024,
    name: Optional[str] = None,
) -> CategorySet:
    """Draft a :class:`CategorySet` from a description, using a *local* model.

    The result is meant to be written to disk and read by a human before use.
    ``model_id`` is loaded through ``interp.core.loader.load``, which refuses
    Ollama names and hosted APIs -- the ADR-009 enforcement point, since these
    prompts can go on to train a checkpoint.
    """
    from vivasecuris.aiasylum.interp.core.loader import load

    model, tokenizer = load(model_id, device=device, dtype=dtype)

    prompts = _generate_lines(
        model, tokenizer, _GEN_INSTRUCTION.format(n=n, description=description),
        want=n, max_new_tokens=max_new_tokens,
    )
    near: List[str] = []
    if near_miss:
        near = _generate_lines(
            model, tokenizer, _NEAR_MISS_INSTRUCTION.format(n=n, description=description),
            want=n, max_new_tokens=max_new_tokens,
        )
        # A line the model put in both lists is ambiguous; drop it from near-miss
        # so the two classes stay disjoint (CategorySet would otherwise refuse).
        near = [p for p in near if p not in set(prompts)]

    cat = CategorySet(
        name=name or "generated",
        prompts=prompts,
        near_miss=near,
        provenance={
            "generator_model": model_id,
            "description": description,
            "requested": n,
            "note": "machine-drafted; review before use",
        },
    )
    if len(cat.prompts) < MIN_PROMPTS:
        logger.warning(
            "Generator produced only %d category prompts (wanted %d). A small local "
            "model may not follow the format; review and top up the file by hand.",
            len(cat.prompts), n,
        )
    return cat
