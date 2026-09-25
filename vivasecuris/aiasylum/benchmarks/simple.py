"""Simple benchmark implementations (without external dependencies)."""

from typing import Dict, List, Optional, Any
import re
from decimal import Decimal, InvalidOperation

from vivasecuris.aiasylum.benchmarks.base import Benchmark, BenchmarkResult


SCORING_VERSION = "final-answer-v4"


def _unmark(text: str) -> str:
    """Remove balanced presentation markup without changing arithmetic operators."""
    for pattern in (r"(?<![\w*])\*\*([^*\n]+)\*\*(?![\w*])", r"(?<!\w)__([^_\n]+)__(?!\w)", r"`([^`\n]+)`"):
        text = re.sub(pattern, r"\1", text)
    return text.strip()


_ANSWER_MARKER = re.compile(
    r"(?:^|[\n.!?]\s*)(?:\#{1,6}[ \t]+)?(?:(?:therefore|thus|hence|so)[,;:]?\s+)?"
    r"(?:the\s+)?(?:(?:final|correct)\s+answer|answer)\s*(?:is\b\s*[:=]?|[:=])\s*",
    re.I | re.M,
)


def _visible_answer_text(response: str) -> str:
    """Only answer-channel text may supply a choice or a number."""
    text = str(response).strip()
    for tag in ("think", "thinking", "analysis"):
        if re.search(fr"</{tag}>", text, re.I):
            text = re.split(fr"</{tag}>", text, flags=re.I)[-1]
        elif re.search(fr"<{tag}>", text, re.I):
            return ""
    return _unmark(text)


def final_answer_text(response: str) -> str:
    """Exclude thinking traces and extract the final submitted answer."""
    text = _visible_answer_text(response)
    boxes = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxes:
        return boxes[-1].strip()
    if "####" in text:
        text = text.rsplit("####", 1)[-1]
    # A marker must begin a line/sentence, optionally after a concluding phrase.
    # In particular, consume both `is` AND its optional colon (`is: $100`).
    markers = list(_ANSWER_MARKER.finditer(text))
    if markers:
        text = text[markers[-1].end():]
        # Repeated labels such as "the answer is: Final answer: 2" carry one
        # value. Consume only exact leading markers, never prose or equations.
        nested = _ANSWER_MARKER.match(text)
        while nested:
            text = text[nested.end():]
            nested = _ANSWER_MARKER.match(text)
        return next((line.strip() for line in text.splitlines() if line.strip()), "")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def normalize_answer(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower()).strip(" .!`*\"'")


def numeric_answer(value: str) -> Optional[Decimal]:
    text = _unmark(final_answer_text(value)).strip("* ")
    # Some models copy the numeric placeholder in the requested answer format.
    # Normalize only an exact leading placeholder or one whole bracketed value;
    # the numeric grammar below still rejects empty, nested or competing answers.
    text = re.sub(r"^<number>\s*", "", text, flags=re.I)
    bracketed = re.fullmatch(r"<([^<>]+)>", text)
    if bracketed:
        text = bracketed.group(1).strip()
    # Remove a whole LaTeX math wrapper only; expressions inside still have to
    # satisfy the same one-number grammar. Currency escaping is presentation.
    for pattern in (r"\\\(([^\n]*)\\\)", r"\\\[([^\n]*)\\\]", r"\$([^$\n]+)\$"):
        wrapped = re.fullmatch(pattern, text)
        if wrapped:
            text = wrapped.group(1).strip()
            break
    text = re.sub(r"^([+-]?\s*)\\([$€£¥])", r"\1\2", text)
    text = re.sub(r"^([+-])\s*([$€£¥])\s*", r"\2\1", text)
    # One number, with optional currency and units. Alternatives, negations and
    # further arithmetic cannot be mistaken for a final numerical answer.
    match = re.fullmatch(
        r"(?:[$€£¥]|USD\s*|EUR\s*|GBP\s*)?\s*"
        r"([+-]?(?:(?:\d{1,3}(?:,[ \t]*\d{3})+|\d+)(?:\.\d*)?|\.\d+))"
        r"(?:\s+([A-Za-z%][A-Za-z% ]*))?\.?", text,
    )
    if not match:
        return None
    units = match.group(2) or ""
    if re.search(r"\b(?:or|and|not|maybe|possibly|approximately|about|instead|uncertain|guess|if|either|is|wrong|incorrect|should)\b", units, re.I):
        return None
    try:
        return Decimal(re.sub(r"[, \t]", "", match.group(1)))
    except InvalidOperation:
        return None


def _choice_value(answer: str, choices) -> Optional[int]:
    answer = _unmark(answer).strip("`*")
    match = re.fullmatch(r"\(?([A-Z])\)?(?:[.):]\s*.*|\s*[-—]\s*.*|\s*)", answer, re.I)
    if match:
        index = ord(match.group(1).upper()) - 65
        if not 0 <= index < len(choices):
            return None
        prefix = re.match(r"\(?[A-Z]\)?", answer, re.I)
        suffix = normalize_answer(answer[prefix.end():].lstrip(" .:)-—\t"))
        chosen_text = normalize_answer(choices[index])
        # A negatively worded option is still a valid exact answer.
        if suffix == chosen_text:
            return index
        negation = r"^(?:(?:this|that)\s+(?:answer|choice|option)\s+)?(?:is\s+|seems?\s+|appears?\s+)?(?:wrong|incorrect|false|not\s+(?:correct|right|the\s+(?:answer|choice))|maybe|uncertain|unsure|probably|possibly|perhaps|ambiguous|questionable)\b"
        if re.search(negation, suffix, re.I):
            return None
        if chosen_text and suffix.startswith(chosen_text):
            remainder = suffix[len(chosen_text):].lstrip(" .,:;()-— ")
            if re.search(negation, remainder, re.I):
                return None
        return index
    matches = [i for i, choice in enumerate(choices) if normalize_answer(answer) == normalize_answer(choice)]
    return matches[0] if len(matches) == 1 else None


def choice_answer(response: str, choices) -> Optional[int]:
    text = _visible_answer_text(response)
    if not text:
        return None
    markers = list(_ANSWER_MARKER.finditer(text))
    if markers or re.search(r"\\boxed\{[^{}]+\}", text):
        # An explicit final answer takes precedence over an earlier tentative
        # choice, including an invalid/ambiguous final answer (no fallback).
        return _choice_value(final_answer_text(text), choices)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    first = _choice_value(lines[0], choices)
    if first is None:
        return _choice_value(final_answer_text(text), choices)
    # A list of alternatives is not a submitted first-line answer.
    first_block = re.split(r"\n\s*\n", text, maxsplit=1)[0].splitlines()
    first_block_choices = {_choice_value(line.strip(), choices) for line in first_block}
    first_block_choices.discard(None)
    all_line_choices = [_choice_value(line, choices) for line in lines]
    if len(first_block_choices) > 1 or (all(value is not None for value in all_line_choices) and len(set(all_line_choices)) > 1):
        return None
    # A later inline explicit declaration that conflicts with the first line
    # makes it ambiguous; never pick a favorable letter from explanation prose.
    declarations = re.findall(
        r"\b(?:(?:final|correct)\s+)?answer\s*(?:is\b\s*[:=]?|[:=])\s*"
        r"\(?([A-Z])\)?(?=[.)\s]|$)", text, re.I,
    )
    if any(ord(letter.upper()) - 65 != first for letter in declarations):
        return None
    return first


def exact_response_matches(response: str, ground_truth: Any, choices=None, *, numeric=False) -> bool:
    answer = final_answer_text(response)
    if not answer:
        return False
    if choices:
        if isinstance(ground_truth, int):
            index = ground_truth
        elif isinstance(ground_truth, str) and len(ground_truth.strip()) == 1 and ground_truth.strip().isalpha():
            index = ord(ground_truth.strip().upper()) - 65
        elif str(ground_truth).isdigit():
            index = int(ground_truth)
        else:
            return False
        if not 0 <= index < len(choices):
            return False
        return choice_answer(response, choices) == index
    if numeric:
        expected = numeric_answer(str(ground_truth))
        actual = numeric_answer(response)
        return expected is not None and actual is not None and actual == expected
    if isinstance(ground_truth, dict):
        ground_truth = ground_truth.get("answer")
    if isinstance(ground_truth, list):
        return any(exact_response_matches(response, value) for value in ground_truth)
    return ground_truth is not None and normalize_answer(answer) == normalize_answer(ground_truth)


class SimpleBenchmark(Benchmark):
    """Simple benchmark that uses predefined questions (no external dataset loading)."""
    
    def __init__(self, name: str, questions: List[Dict[str, Any]], description: str = ""):
        super().__init__(name, description)
        self.questions = questions
    
    async def load_dataset(self, num_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load questions from predefined list."""
        if num_samples:
            return self.questions[:num_samples]
        return self.questions
    
    async def evaluate_response(
        self,
        question: str,
        response: str,
        ground_truth: Any,
    ) -> bool:
        """Exact final-answer matching for explicitly named toy/demo benchmarks."""
        return exact_response_matches(response, ground_truth)


def create_simple_benchmark(name: str, questions: List[Dict[str, Any]]) -> SimpleBenchmark:
    """Create a simple benchmark from a list of questions."""
    return SimpleBenchmark(name, questions)


# Example benchmarks (can be expanded)
SIMPLE_MATH_QUESTIONS = [
    {"question": "What is 2 + 2?", "answer": "4"},
    {"question": "What is 5 * 3?", "answer": "15"},
    {"question": "What is 10 / 2?", "answer": "5"},
    {"question": "What is the square root of 16?", "answer": "4"},
    {"question": "What is 2 to the power of 3?", "answer": "8"},
]

SIMPLE_REASONING_QUESTIONS = [
    {"question": "If all roses are flowers, and some flowers are red, can we conclude that some roses are red?", "answer": "no"},
    {"question": "A train leaves Station A at 60 mph. Another train leaves Station B, 200 miles away, at 80 mph heading towards Station A. When will they meet?", "answer": "1.43 hours"},
]
