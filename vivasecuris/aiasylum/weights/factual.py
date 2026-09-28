"""Scoring factuality: correct vs. abstained vs. hallucinated.

No weight edit adds knowledge, so "more factual" can only mean turning confident
wrong answers into abstentions. The metric that matters is therefore a
three-way split, not accuracy alone:

- **correct**   an accepted answer (alias match) with no hedge.
- **abstain**   "I don't know", a refusal, or a hedge -- the model declined to
                commit. On an unanswerable question this is the right move.
- **wrong**     a committed answer that is not accepted -- a hallucination.

The hard gate for any factuality edit is that ``acc_answerable`` (accuracy on
questions the model can answer) must not fall: an edit that abstains on
everything scores a perfect hallucination rate and is useless. ``halluc_rate``
is measured over answerable and unanswerable questions together, and
``abstain_unanswerable`` is reported beside it so "stopped hallucinating" is
never confused with "stopped answering".
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# Phrases that mark a declined answer. Kept beside REFUSAL_PHRASES: a refusal is
# also an abstention for factuality's purposes.
ABSTAIN_PHRASES = (
    "i don't know", "i do not know", "i'm not sure", "i am not sure", "not sure",
    "i'm not certain", "i am not certain", "no idea", "i have no idea",
    "cannot determine", "can't determine", "unable to determine",
    "i don't have information", "i do not have information", "i don't have enough",
    "not aware of", "no information", "unknown to me", "i couldn't find",
    "there is no", "does not exist", "doesn't exist", "no such",
)

_HEDGE = re.compile(r"\b(?:not|wrong|incorrect|isn't|isnt|maybe|perhaps|uncertain|possibly|might be)\b")


def _visible(text: str) -> str:
    from vivasecuris.aiasylum.benchmarks.simple import final_answer_text

    return final_answer_text(text or "")


def classify_answer(text: str, aliases: Sequence[str]) -> str:
    """``correct`` | ``abstain`` | ``wrong`` for one answer against accepted aliases.

    Abstention wins over a coincidental alias mention: "I don't know, maybe it's
    Paris" is a decline, not a correct answer, the same conservative rule
    :func:`evaluate.factual_accuracy` uses for hedges.
    """
    from vivasecuris.aiasylum.constants import REFUSAL_PHRASES

    low = _visible(text).lower()
    if not low.strip():
        return "abstain"
    if any(p in low for p in ABSTAIN_PHRASES) or any(p in low for p in REFUSAL_PHRASES):
        return "abstain"
    if _HEDGE.search(low):
        return "abstain"
    for a in aliases:
        a = (a or "").strip().lower()
        if a and re.search(r"(?<!\w)" + re.escape(a) + r"(?!\w)", low):
            return "correct"
    return "wrong"


# Templated fictitious entities: plausible-looking names the model cannot know,
# so the correct behaviour is to abstain. Deterministic per seed.
_FIRST = ("Aldous", "Marena", "Tobias", "Seline", "Cyrus", "Delia", "Roland", "Ysolde")
_LAST = ("Vantreel", "Okonkwo-Bright", "Halvorsen", "Quillon", "Marchetti", "Nunez-Fable")
_TEMPLATES = (
    "In what year was the physicist {name} awarded the Nobel Prize?",
    "What is the capital city of the country of {place}?",
    "Which studio produced the {year} film \"{title}\"?",
    "What was the population of {place} in the last census?",
    "Who composed the opera \"{title}\"?",
)


def fabricated_entity_questions(n: int, seed: int = 0) -> List[str]:
    """``n`` questions about entities that do not exist; the right answer is to abstain."""
    import random

    rng = random.Random(seed)
    out: List[str] = []
    while len(out) < n:
        name = f"{rng.choice(_FIRST)} {rng.choice(_LAST)}"
        place = f"{rng.choice(_FIRST)}{rng.choice(['ovia', 'stan', 'burg', 'alia'])}"
        title = f"The {rng.choice(['Silent', 'Amber', 'Hollow', 'Ninth'])} {rng.choice(['Meridian', 'Cipher', 'Requiem', 'Lantern'])}"
        year = rng.randint(1931, 2019)
        q = rng.choice(_TEMPLATES).format(name=name, place=place, title=title, year=year)
        if q not in out:
            out.append(q)
    return out


@dataclass
class FactualScore:
    n_answerable: int
    n_unanswerable: int
    correct_answerable: int
    wrong_answerable: int
    wrong_unanswerable: int
    abstain_unanswerable: int

    @property
    def acc_answerable(self) -> float:
        return self.correct_answerable / self.n_answerable if self.n_answerable else 0.0

    @property
    def halluc_rate(self) -> float:
        total = self.n_answerable + self.n_unanswerable
        return (self.wrong_answerable + self.wrong_unanswerable) / total if total else 0.0

    @property
    def abstain_unanswerable_rate(self) -> float:
        return self.abstain_unanswerable / self.n_unanswerable if self.n_unanswerable else 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "acc_answerable": self.acc_answerable,
            "halluc_rate": self.halluc_rate,
            "abstain_unanswerable": self.abstain_unanswerable_rate,
            "n_answerable": self.n_answerable,
            "n_unanswerable": self.n_unanswerable,
            "correct_answerable": self.correct_answerable,
            "wrong_answerable": self.wrong_answerable,
            "wrong_unanswerable": self.wrong_unanswerable,
        }


def score_factual(
    answerable_responses: Sequence[str],
    answerable_aliases: Sequence[Sequence[str]],
    unanswerable_responses: Sequence[str],
) -> FactualScore:
    """Three-way score over answerable (with aliases) and unanswerable responses."""
    ca = wa = 0
    for resp, aliases in zip(answerable_responses, answerable_aliases):
        verdict = classify_answer(resp, aliases)
        if verdict == "correct":
            ca += 1
        elif verdict == "wrong":
            wa += 1
    wu = au = 0
    for resp in unanswerable_responses:
        verdict = classify_answer(resp, [])   # nothing is a correct answer here
        if verdict == "abstain":
            au += 1
        else:                                  # any committed answer is a hallucination
            wu += 1
    return FactualScore(
        n_answerable=len(answerable_responses),
        n_unanswerable=len(unanswerable_responses),
        correct_answerable=ca, wrong_answerable=wa,
        wrong_unanswerable=wu, abstain_unanswerable=au,
    )
