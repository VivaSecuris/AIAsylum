"""Knowledge distillation of a local teacher into a LoRA-adapted student.

Two levels (docs/MODEL_MODIFICATION.md, "Distillation, specifically"):

- ``response``: the teacher answers each prompt greedily and the student is
  fine-tuned on those answers with :func:`lora.train_lora`. Only a generation
  pass is needed, so teacher and student never share memory.
- ``logit``: the student matches the teacher's softened next-token
  distribution on the response positions (KL at temperature ``T``, scaled by
  ``T^2`` so the gradient magnitude does not vanish as ``T`` grows), mixed with
  ordinary cross-entropy on the labels. Both models are resident; the
  tokenizers must be identical, since position ``i`` of both logit rows has to
  mean the same token.

ADR-009: the teacher must be a local open-weights model, and it is loaded
through ``loader.load``, which rejects Ollama names. Output from a hosted
model (class C data) is for evaluation only and never enters this path.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from vivasecuris.aiasylum.weights.train_data import TrainRow

logger = logging.getLogger(__name__)


def generate_teacher_rows(
    teacher,
    tokenizer,
    prompts: Sequence[str],
    *,
    max_new_tokens: int = 256,
    system_prompt: Optional[str] = None,
    progress: Optional[Callable[[int, int], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> List[TrainRow]:
    """Greedy teacher completions for ``prompts``, as training rows.

    Uses ``evaluate.generate_greedy`` one prompt at a time -- through the
    model's chat template when it has one, as plain text with BOS otherwise,
    the thinking block stripped -- so ``progress`` and ``should_stop`` work per
    prompt. ``system_prompt`` conditions the teacher only: the rows carry no
    system prompt, so the student learns the behaviour without it (context
    distillation). A stop request returns the rows generated so far.
    """
    from vivasecuris.aiasylum.weights.evaluate import generate_greedy

    rows: List[TrainRow] = []
    total = len(prompts)
    for i, prompt in enumerate(prompts, 1):
        if should_stop is not None and should_stop():
            logger.info("Stop requested after %d of %d teacher generations", len(rows), total)
            break
        # One path whether or not a system prompt is set and whether or not the
        # tokenizer has a template. The earlier three-way branch gave a
        # base-model teacher a BOS token only when a system prompt happened to be
        # set, and stripped-then-re-added the template's BOS otherwise; routing
        # every case through generate_greedy makes the special-token rule the
        # formatter's, once.
        response = generate_greedy(
            teacher, tokenizer, [prompt], max_new_tokens=max_new_tokens,
            system_prompt=system_prompt,
        )[0]
        rows.append(TrainRow(prompt=prompt, response=(response or "").strip()))
        if progress is not None:
            progress(i, total)
    return rows


def _tokenizer_digest(tokenizer) -> str:
    vocab = sorted((str(k), int(v)) for k, v in tokenizer.get_vocab().items())
    special = sorted((str(k), str(v)) for k, v in (tokenizer.special_tokens_map or {}).items())
    payload = json.dumps({"vocab": vocab, "special": special}, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def check_tokenizer_compatibility(student_tok, teacher_tok) -> Dict[str, Any]:
    """Whether two tokenizers index the same vocabulary the same way.

    Logit distillation compares distributions position by position, so the
    vocabularies must match exactly (same tokens, same ids, same specials).
    """
    student = _tokenizer_digest(student_tok)
    teacher = _tokenizer_digest(teacher_tok)
    return {
        "student_digest": student,
        "teacher_digest": teacher,
        "compatible": student == teacher,
        "student_vocab": len(student_tok.get_vocab()),
        "teacher_vocab": len(teacher_tok.get_vocab()),
    }


def kl_distill_loss(
    student_logits, teacher_logits, labels, temperature: float, ce_weight: float
) -> Tuple[Any, Dict[str, float]]:
    """``ce_weight * CE + (1 - ce_weight) * T^2 * KL(teacher_T || student_T)``.

    Computed on response positions only (``labels != -100``), shifted as a
    causal LM: logits at position ``i`` predict the label at ``i + 1``. The
    two logit tensors are sliced to their common vocabulary prefix, since
    embedding tables are often padded past the tokenizer's size. ``batchmean``
    over the labelled positions; the cross-entropy uses the student's full
    width. With no labelled position the loss is zero (with a gradient path).
    """
    import torch
    import torch.nn.functional as F

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0.0 <= ce_weight <= 1.0:
        raise ValueError("ce_weight must be between 0 and 1")

    shift_student = student_logits[:, :-1]
    shift_teacher = teacher_logits[:, :-1]
    shift_labels = labels[:, 1:]
    mask = shift_labels != -100
    positions = int(mask.sum().item())
    if positions == 0:
        return student_logits.sum() * 0.0, {"kl": 0.0, "ce": 0.0, "positions": 0}

    s = shift_student[mask].float()                       # [N, Vs]
    t = shift_teacher[mask].float().to(s.device)          # [N, Vt]
    y = shift_labels[mask].to(s.device)
    ce = F.cross_entropy(s, y)

    width = min(s.shape[-1], t.shape[-1])
    log_ps = F.log_softmax(s[:, :width] / temperature, dim=-1)
    log_pt = F.log_softmax(t[:, :width] / temperature, dim=-1)
    kl = F.kl_div(log_ps, log_pt, log_target=True, reduction="batchmean") * (temperature ** 2)

    total = ce_weight * ce + (1.0 - ce_weight) * kl
    return total, {"kl": float(kl.item()), "ce": float(ce.item()), "positions": positions}


def make_logit_loss_fn(teacher, temperature: float, ce_weight: float) -> Callable[[Any, Dict[str, Any]], Tuple[Any, dict]]:
    """A ``loss_fn(model, batch)`` for :func:`lora.train_lora` that runs the teacher under ``no_grad``."""
    import torch

    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    teacher_device = next(teacher.parameters()).device

    def loss_fn(model, batch: Dict[str, Any]) -> Tuple[Any, dict]:
        with torch.no_grad():
            teacher_logits = teacher(
                input_ids=batch["input_ids"].to(teacher_device),
                attention_mask=batch["attention_mask"].to(teacher_device),
            ).logits
        student_logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        return kl_distill_loss(
            student_logits, teacher_logits.to(student_logits.device), batch["labels"],
            temperature, ce_weight,
        )

    return loss_fn


def train_distill(
    student,
    tokenizer,
    rows: Sequence[TrainRow],
    spec,
    run_dir,
    *,
    level: str,
    teacher=None,
    teacher_tokenizer=None,
    temperature: float = 2.0,
    ce_weight: float = 0.5,
    progress: Callable[[dict], None],
    should_stop: Callable[[], bool],
) -> Dict[str, Any]:
    """Distil into ``student`` at ``level`` ("response" or "logit").

    Rows must already carry responses (from :func:`generate_teacher_rows` or a
    gold dataset); the loss is defined on response positions, so a row
    without one contributes nothing. At the logit level ``teacher`` is
    required and its tokenizer (``teacher_tokenizer``, defaulting to the
    student's) must be compatible.
    """
    from vivasecuris.aiasylum.weights.lora import train_lora

    if level not in ("response", "logit"):
        raise ValueError(f"level must be 'response' or 'logit', got {level!r}")
    rows = list(rows)
    without_response = sum(1 for r in rows if not r.response)
    if without_response:
        raise ValueError(
            f"{without_response} of {len(rows)} rows have no response; "
            f"generate them from the teacher first (generate_teacher_rows)"
        )

    if level == "response":
        result = train_lora(student, tokenizer, rows, spec, run_dir, progress=progress, should_stop=should_stop)
        result["level"] = "response"
        return result

    if teacher is None:
        raise ValueError("logit-level distillation needs a loaded teacher")
    compat = check_tokenizer_compatibility(tokenizer, teacher_tokenizer if teacher_tokenizer is not None else tokenizer)
    if not compat["compatible"]:
        raise ValueError(
            f"student and teacher tokenizers differ (student vocab {compat['student_vocab']}, "
            f"teacher vocab {compat['teacher_vocab']}); logit distillation needs an identical "
            f"vocabulary. Use level='response'."
        )
    loss_fn = make_logit_loss_fn(teacher, temperature, ce_weight)
    result = train_lora(student, tokenizer, rows, spec, run_dir, progress=progress,
                        should_stop=should_stop, loss_fn=loss_fn)
    kls = [h["kl"] for h in result["history"] if "kl" in h]
    result.update({
        "level": "logit",
        "temperature": temperature,
        "ce_weight": ce_weight,
        "mean_kl": (sum(kls) / len(kls)) if kls else None,
        "tokenizer_compatibility": compat,
    })
    return result
