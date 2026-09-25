"""Subprocess entry point for LoRA and distillation jobs.

Training runs in its own process so that the API server never holds a model
in its address space: a job that exhausts unified memory takes down this
process, not the server, and cancellation is a SIGTERM rather than a thread
that cannot be interrupted mid-backward. The API writes a job.json, spawns
``python -m vivasecuris.aiasylum.weights.train_worker job.json``, and reads
one JSON event per stdout line. Nothing else is written to stdout; logging
goes to stderr. This module never opens the database.

Events: ``note``, ``step`` (per optimizer step), ``eval``, ``count`` (teacher
generation), then exactly one of ``done`` or ``error``. Exit 0 on completion,
3 when stopped on request (``done`` is still emitted with
``result.stopped = true``), 1 on error.
"""

from __future__ import annotations

import os

# Several backward ops still have no MPS kernel; without the fallback they
# raise instead of running on CPU. Must be set before torch is imported.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import gc  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import signal  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from dataclasses import asdict  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Callable, Dict, List, Optional  # noqa: E402

from vivasecuris.aiasylum.weights.lora import LoraSpec  # noqa: E402
from vivasecuris.aiasylum.weights.train_data import (  # noqa: E402
    TrainRow, dataset_digest, read_jsonl, write_jsonl,
)

logger = logging.getLogger(__name__)

EXIT_DONE, EXIT_ERROR, EXIT_STOPPED = 0, 1, 3
TEACHER_SAMPLE_COUNT = 5

_stop_requested = False


def should_stop() -> bool:
    return _stop_requested


def request_stop() -> None:
    global _stop_requested
    _stop_requested = True


@contextmanager
def _sigterm_requests_stop():
    """SIGTERM sets the stop flag; the training loop saves a partial adapter and exits."""
    def handler(_signum, _frame):
        logger.info("SIGTERM received; stopping after the current step")
        request_stop()

    try:
        previous = signal.signal(signal.SIGTERM, handler)
    except ValueError:  # not the main thread (in-process use from a test or a worker thread)
        previous = None
    try:
        yield
    finally:
        if previous is not None:
            signal.signal(signal.SIGTERM, previous)


class _Emitter:
    """One JSON object per line on the event stream, flushed immediately."""

    def __init__(self, stream):
        self.stream = stream

    def __call__(self, event: Dict[str, Any]) -> None:
        self.stream.write(json.dumps(event, default=str) + "\n")
        self.stream.flush()

    def note(self, message: str) -> None:
        self({"event": "note", "message": message})


def _free_accelerator_memory() -> None:
    import torch

    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _progress_events(emit: _Emitter) -> Callable[[dict], None]:
    def progress(record: dict) -> None:
        if record.get("phase") == "eval":
            emit({"event": "eval", "step": record["step"], "eval_loss": record["eval_loss"]})
        else:
            emit({"event": "step", **record})

    return progress


def _stopped_result(stage: str, distill_info: Optional[dict], teacher_samples: Optional[list]) -> Dict[str, Any]:
    return {
        "train": None, "history": [], "adapter_path": None, "output_path": None,
        "merged": False, "manifest": None, "stopped": True, "stage": stage,
        "distill": distill_info, "teacher_samples": teacher_samples,
    }


def run_job(job: Dict[str, Any], emit: _Emitter) -> Dict[str, Any]:
    """Execute one job description; returns the ``done`` result. Raises on error."""
    kind = job.get("kind")
    if kind not in ("lora", "distill"):
        raise ValueError(f"kind must be 'lora' or 'distill', got {kind!r}")
    source_model = job.get("source_model")
    if not source_model:
        raise ValueError("source_model is required")
    dataset_path = Path(job.get("dataset_path") or "")
    if not dataset_path.is_file():
        raise FileNotFoundError(f"dataset not found: {dataset_path}")
    run_dir = Path(job.get("run_dir") or "")
    if not str(run_dir):
        raise ValueError("run_dir is required")
    run_dir.mkdir(parents=True, exist_ok=True)

    spec = LoraSpec.from_dict(job.get("lora") or {})
    spec.validate()
    device = job.get("device") or "auto"
    dtype = job.get("dtype") or "bfloat16"
    out_dir = job.get("out_dir")
    if spec.merge:
        if not out_dir:
            raise ValueError("lora.merge is true but out_dir is null")
        if Path(out_dir).exists() and any(Path(out_dir).iterdir()):
            raise FileExistsError(f"{out_dir} already exists and is not empty")

    distill = job.get("distill") if kind == "distill" else None
    if kind == "distill" and not distill:
        raise ValueError("a distill job needs a 'distill' block")

    rows: List[TrainRow] = read_jsonl(dataset_path)
    if not rows:
        raise ValueError(f"{dataset_path} holds no rows")
    emit.note(f"loaded {len(rows)} rows from {dataset_path.name}")
    dataset_meta: Dict[str, Any] = {
        "path": str(dataset_path), "rows": len(rows), "digest": dataset_digest(rows),
    }

    from vivasecuris.aiasylum.interp.core import loader
    from vivasecuris.aiasylum.weights import distill as distill_mod
    from vivasecuris.aiasylum.weights import lora as lora_mod

    student = tokenizer = None
    teacher = teacher_tokenizer = None
    distill_info: Optional[Dict[str, Any]] = None
    teacher_samples: Optional[List[Dict[str, str]]] = None

    if distill:
        level = distill.get("level") or "response"
        if level not in ("response", "logit"):
            raise ValueError(f"distill.level must be 'response' or 'logit', got {level!r}")
        teacher_id = distill.get("teacher_model")
        if not teacher_id:
            raise ValueError("distill.teacher_model is required")
        temperature = float(distill.get("temperature", 2.0))
        ce_weight = float(distill.get("ce_weight", 0.5))
        teacher_max_new_tokens = int(distill.get("teacher_max_new_tokens", 256))
        teacher_system_prompt = distill.get("teacher_system_prompt")
        # Response level always samples the teacher; logit level only fills gaps.
        needs_generation = level == "response" or any(not r.response for r in rows)
        distill_info = {
            "teacher_model": teacher_id, "level": level, "temperature": temperature,
            "ce_weight": ce_weight, "teacher_max_new_tokens": teacher_max_new_tokens,
            "teacher_system_prompt": teacher_system_prompt, "generated": needs_generation,
        }

        emit.note(f"loading teacher {teacher_id}")
        teacher, teacher_tokenizer = loader.load(teacher_id, device=device, dtype=dtype, seed=spec.seed)

        if level == "logit":
            # Both models are resident for the loss anyway; load the student now
            # so an incompatible tokenizer fails before minutes of generation.
            emit.note(f"loading student {source_model}")
            student, tokenizer = loader.load(source_model, device=device, dtype=dtype, seed=spec.seed)
            compat = distill_mod.check_tokenizer_compatibility(tokenizer, teacher_tokenizer)
            distill_info["tokenizer_compatibility"] = compat
            if not compat["compatible"]:
                raise ValueError(
                    f"student and teacher tokenizers differ (student vocab {compat['student_vocab']}, "
                    f"teacher vocab {compat['teacher_vocab']}); logit distillation needs an "
                    f"identical vocabulary. Use level 'response'."
                )

        if needs_generation:
            emit.note(f"generating {len(rows)} teacher responses (greedy, up to {teacher_max_new_tokens} new tokens)")
            generated = distill_mod.generate_teacher_rows(
                teacher, teacher_tokenizer, [r.prompt for r in rows],
                max_new_tokens=teacher_max_new_tokens, system_prompt=teacher_system_prompt,
                progress=lambda done, total: emit(
                    {"event": "count", "phase": "teacher_generation", "done": done, "total": total}
                ),
                should_stop=should_stop,
            )
            teacher_path = run_dir / "teacher_responses.jsonl"
            write_jsonl(teacher_path, generated)
            dataset_meta["teacher_responses"] = str(teacher_path)
            teacher_samples = [
                {"prompt": r.prompt, "response": r.response} for r in generated[:TEACHER_SAMPLE_COUNT]
            ]
            if len(generated) < len(rows):
                emit.note(f"stopped during teacher generation after {len(generated)} of {len(rows)} prompts")
                return _stopped_result("teacher_generation", distill_info, teacher_samples)
            empty = sum(1 for r in generated if not r.response)
            if empty:
                emit.note(f"dropping {empty} rows where the teacher produced no text")
            rows = [r for r in generated if r.response]
            if not rows:
                raise ValueError("the teacher produced no text for any prompt")
            dataset_meta["rows_trained"] = len(rows)
            dataset_meta["digest_trained"] = dataset_digest(rows)

        if level == "response":
            del teacher
            teacher = teacher_tokenizer = None
            _free_accelerator_memory()

    if student is None:
        emit.note(f"loading student {source_model}")
        student, tokenizer = loader.load(source_model, device=device, dtype=dtype, seed=spec.seed)

    progress = _progress_events(emit)
    emit.note(f"training LoRA r={spec.rank} targets={spec.targets} on {len(rows)} rows")
    if distill:
        train = distill_mod.train_distill(
            student, tokenizer, rows, spec, run_dir, level=distill_info["level"],
            teacher=teacher, teacher_tokenizer=teacher_tokenizer,
            temperature=distill_info["temperature"], ce_weight=distill_info["ce_weight"],
            progress=progress, should_stop=should_stop,
        )
    else:
        train = lora_mod.train_lora(student, tokenizer, rows, spec, run_dir,
                                    progress=progress, should_stop=should_stop)

    history = train.get("history") or []
    summary = {k: v for k, v in train.items() if k != "history"}
    result: Dict[str, Any] = {
        "train": summary,
        "history": history,
        "adapter_path": train["adapter_path"],
        "output_path": train["adapter_path"],
        "merged": False,
        "manifest": None,
        "stopped": bool(train.get("stopped")),
    }
    if distill:
        result["distill"] = distill_info
        result["teacher_samples"] = teacher_samples

    if spec.merge and not result["stopped"]:
        if teacher is not None:
            del teacher
            teacher = None
            _free_accelerator_memory()
        emit.note(f"merging the adapter into {out_dir}")
        extra = dict(job.get("manifest_extra") or {})
        if distill_info:
            extra["distill"] = distill_info
        out = lora_mod.merge_and_save(
            student, tokenizer, out_dir, source_model=source_model, spec=spec,
            train_result=train, dataset_meta=dataset_meta, notes=job.get("notes"),
            extra=extra, method="lora_merge",
        )
        from vivasecuris.aiasylum.weights.manifest import SurgeryManifest

        manifest = SurgeryManifest.load(out)
        result.update({
            "output_path": str(out), "merged": True,
            "manifest": asdict(manifest) if manifest else None,
        })
    return result


def main(job_path) -> int:
    """Run the job at ``job_path``; events on stdout, exit code as documented above."""
    global _stop_requested
    _stop_requested = False
    started = time.time()
    events = sys.stdout
    emit = _Emitter(events)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    # Anything that prints (library banners, stray debug output) must not land
    # in the event stream; only the emitter holds the real stdout.
    sys.stdout = sys.stderr
    try:
        with _sigterm_requests_stop():
            try:
                job = json.loads(Path(job_path).read_text(encoding="utf-8"))
                result = run_job(job, emit)
            except BaseException as exc:  # noqa: BLE001 -- the caller only ever sees the event
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                emit({
                    "event": "error",
                    "message": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                })
                return EXIT_ERROR
            result["elapsed_seconds"] = time.time() - started
            emit({"event": "done", "result": result})
            return EXIT_STOPPED if result.get("stopped") else EXIT_DONE
    finally:
        sys.stdout = events


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.stderr.write("usage: python -m vivasecuris.aiasylum.weights.train_worker <job.json>\n")
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
