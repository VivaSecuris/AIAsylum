"""Training rows for LoRA fine-tuning and distillation.

One row is a prompt, the response the model should learn to give, and an
optional system prompt. On disk that is JSONL, one object per line:
``{"prompt": ..., "response": ..., "system": <optional>}``. A prompts-only
dataset (empty responses) is legal here because distillation fills the
responses in from the teacher before training.

This module must import without torch: the API validates uploaded datasets
and hashes them for the manifest in a process that never loads a model.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

# Hard caps. A row past either limit is almost certainly a formatting mistake
# (a whole document pasted as one prompt) rather than a training example.
MAX_TRAIN_ROWS = 20000
MAX_TRAIN_CHARS = 32000  # per field


@dataclass
class TrainRow:
    prompt: str
    response: str = ""
    system: Optional[str] = None

    def to_dict(self) -> Dict[str, str]:
        """JSON form; ``system`` is omitted when unset so digests stay stable."""
        d = {"prompt": self.prompt, "response": self.response}
        if self.system is not None:
            d["system"] = self.system
        return d


def _clean_field(value: Any, name: str, index: int, required: bool) -> Optional[str]:
    if value is None:
        if required:
            raise ValueError(f"row {index}: missing {name!r}")
        return None
    if not isinstance(value, str):
        raise ValueError(f"row {index}: {name!r} must be a string, got {type(value).__name__}")
    value = value.strip()
    if len(value) > MAX_TRAIN_CHARS:
        raise ValueError(
            f"row {index}: {name!r} is {len(value)} characters; the limit is {MAX_TRAIN_CHARS}"
        )
    return value


def parse_rows(rows: Iterable[Mapping[str, Any]], *, require_response: bool) -> List[TrainRow]:
    """Validate raw mappings into ``TrainRow``s. Raises ``ValueError`` naming the row.

    Fields are stripped; a row with an empty prompt is rejected, as is a row
    with an empty response when ``require_response`` is set.
    """
    out: List[TrainRow] = []
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise ValueError(f"row {index}: expected an object, got {type(raw).__name__}")
        if index >= MAX_TRAIN_ROWS:
            raise ValueError(f"more than {MAX_TRAIN_ROWS} rows; split the dataset")
        prompt = _clean_field(raw.get("prompt"), "prompt", index, required=True)
        if not prompt:
            raise ValueError(f"row {index}: empty prompt")
        response = _clean_field(raw.get("response"), "response", index, required=False) or ""
        if require_response and not response:
            raise ValueError(f"row {index}: empty response")
        system = _clean_field(raw.get("system"), "system", index, required=False) or None
        out.append(TrainRow(prompt=prompt, response=response, system=system))
    return out


def parse_jsonl(text: str, *, require_response: bool) -> List[TrainRow]:
    """Decode JSONL text (blank lines ignored) and validate it with :func:`parse_rows`."""
    raw: List[Mapping[str, Any]] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {lineno}: invalid JSON ({exc.msg})") from None
        if not isinstance(obj, dict):
            raise ValueError(f"line {lineno}: expected an object, got {type(obj).__name__}")
        raw.append(obj)
    return parse_rows(raw, require_response=require_response)


def read_jsonl(path, *, require_response: bool = False) -> List[TrainRow]:
    return parse_jsonl(Path(path).read_text(encoding="utf-8"), require_response=require_response)


def write_jsonl(path, rows: Sequence[TrainRow]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")


def dataset_digest(rows: Sequence[TrainRow]) -> str:
    """sha256 of the canonical JSON (sorted keys, no whitespace) of the rows in order."""
    canonical = json.dumps(
        [row.to_dict() for row in rows], sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _run_coroutine(factory):
    """Run ``factory()`` to completion whether or not an event loop is already running.

    ``asyncio.run`` refuses to nest, and the API's request handlers are async,
    so inside a running loop the coroutine gets a thread of its own.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(factory())
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(factory())).result()


def _benchmark_row_to_train(row: Mapping[str, Any]) -> Dict[str, str]:
    """Standardized benchmark row -> prompt/response.

    Multiple choice: the choices are lettered under the question and the
    response is the answer letter (or the answer text when the loader did not
    resolve a letter). Free-form: the response is the answer as given.
    """
    question = str(row.get("question") or "").strip()
    choices = [c for c in (row.get("choices") or []) if isinstance(c, str)]
    answer = row.get("answer")
    if choices:
        lettered = "\n".join(f"{chr(65 + i)}. {c.strip()}" for i, c in enumerate(choices))
        prompt = f"{question}\n{lettered}"
        response = row.get("answer_letter") or ("" if answer is None else str(answer))
    else:
        prompt = question
        response = "" if answer is None else str(answer)
    return {"prompt": prompt, "response": response}


def rows_from_benchmark(name: str, n: int, seed: int) -> List[TrainRow]:
    """``n`` seeded rows from a registered benchmark, as training rows.

    Synchronous wrapper around the async ``load_benchmark_dataset``. The rows
    pass through :func:`parse_rows` without requiring responses because some
    benchmarks (toxicity prompts) carry no answer.
    """
    from vivasecuris.aiasylum.benchmarks.datasets import load_benchmark_dataset

    if n < 1:
        raise ValueError("n must be at least 1")
    rows = _run_coroutine(lambda: load_benchmark_dataset(name, num_samples=n, seed=seed))
    return parse_rows([_benchmark_row_to_train(r) for r in rows], require_response=False)


def rows_from_prompts(prompts: Sequence[str]) -> List[TrainRow]:
    """Prompts with empty responses, for a distillation run to fill in."""
    return parse_rows([{"prompt": p} for p in prompts], require_response=False)
