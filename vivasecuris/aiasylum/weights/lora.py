"""LoRA fine-tuning of a locally loaded causal LM, and merging the result.

Plain-precision LoRA (the base in whatever dtype ``loader.load`` gave it,
adapters in fp32 as peft upcasts them): on the M4 there is no bitsandbytes,
so no QLoRA and no 8-bit optimizers. A full fine-tune of a 3B model does not
fit (docs/MODEL_MODIFICATION.md section 4), an r=8..32 adapter on the
attention projections does.

The loss is masked to the response: every prompt token carries label -100, so
the model learns to answer, not to reproduce the question. Rows are rendered
through the tokenizer's chat template when it has one, exactly as capture and
evaluation render prompts, so the trained positions are the ones inference
will see.

Merging writes an ordinary ``transformers`` directory plus the same
``asylum_surgery.json`` manifest that direction surgery writes, so a merged
LoRA model is loadable and traceable through the same paths as any other edit.
"""

from __future__ import annotations

import json
import logging
import math
import random
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from vivasecuris.aiasylum.weights.train_data import TrainRow

logger = logging.getLogger(__name__)

# Module leaf names per architecture family. Suffix-matched by peft, so a
# name here matches ``model.layers.N.self_attn.q_proj`` in every layer.
_PRESETS = {
    "llama_style": {
        "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "mlp": ["gate_proj", "up_proj", "down_proj"],
    },
    "gpt_neox": {
        "attention": ["query_key_value", "dense"],
        "mlp": ["dense_h_to_4h", "dense_4h_to_h"],
    },
}


@dataclass
class LoraSpec:
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    targets: str = "attention"      # "attention" | "attention+mlp" | comma-separated module names
    epochs: int = 1
    max_steps: Optional[int] = None
    lr: float = 2e-4
    warmup_ratio: float = 0.03
    batch_size: int = 1
    grad_accum: int = 8
    max_length: int = 512
    eval_rows: int = 32
    eval_every: int = 50
    seed: int = 0
    gradient_checkpointing: bool = False
    merge: bool = True

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "LoraSpec":
        """Build from a mapping, ignoring keys this version does not know."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in known})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if self.rank < 1:
            raise ValueError("rank must be at least 1")
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1")
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError("max_steps must be at least 1 when set")
        if self.batch_size < 1 or self.grad_accum < 1:
            raise ValueError("batch_size and grad_accum must be at least 1")
        if self.max_length < 2:
            raise ValueError("max_length must leave room for a prompt and a response")
        if not (0.0 <= self.warmup_ratio <= 1.0):
            raise ValueError("warmup_ratio must be between 0 and 1")
        if self.lr <= 0:
            raise ValueError("lr must be positive")


def _leaf(name: str) -> str:
    return name.rsplit(".", 1)[-1]


def resolve_target_modules(model, spec: LoraSpec) -> List[str]:
    """Module names peft should wrap, checked against ``model.named_modules()``.

    ``attention`` and ``attention+mlp`` expand per architecture family. MLP
    targets apply to dense blocks only: nothing under an ``experts`` container
    is wrapped unless the user names it explicitly, because a rank-8 adapter on
    every expert is neither what "the MLP" means nor what fits in memory. When
    a preset name would also match an expert matrix, the returned names are
    fully qualified so the exclusion is exact.

    Explicit names are comma-separated and suffix-matched the way peft matches
    them; every one must exist, so a typo cannot silently train less.
    """
    from vivasecuris.aiasylum.interp.core.hook_registry import describe_architecture

    names = [n for n, _ in model.named_modules() if n]
    if not names:
        raise ValueError("model has no submodules")
    targets = (spec.targets or "").strip()

    if targets not in ("attention", "attention+mlp"):
        requested = [t.strip() for t in targets.split(",") if t.strip()]
        if not requested:
            raise ValueError("targets is empty; use 'attention', 'attention+mlp' or module names")
        missing = [t for t in requested if not any(n == t or n.endswith("." + t) for n in names)]
        if missing:
            raise ValueError(
                f"target modules {missing} do not exist in this model. "
                f"Module leaf names found: {sorted({_leaf(n) for n in names})}"
            )
        return requested

    info = describe_architecture(model)
    preset = _PRESETS["gpt_neox"] if info.family == "gpt_neox" else _PRESETS["llama_style"]
    if info.family is None:
        # Unknown family: take whichever preset's attention names are present.
        for candidate in _PRESETS.values():
            if any(_leaf(n) in candidate["attention"] for n in names):
                preset = candidate
                break

    wanted_attn = preset["attention"]
    wanted_mlp = preset["mlp"] if targets == "attention+mlp" else []
    sparse_prefixes = [n + "." for n, m in model.named_modules() if getattr(m, "experts", None) is not None]

    attn_hits = [n for n in names if _leaf(n) in wanted_attn]
    if not attn_hits:
        raise ValueError(
            f"no attention projections named {wanted_attn} in this model "
            f"({info.label}). Module leaf names found: {sorted({_leaf(n) for n in names})}"
        )
    mlp_candidates = [n for n in names if _leaf(n) in wanted_mlp]
    mlp_hits = [n for n in mlp_candidates if not any(n.startswith(p) for p in sparse_prefixes)]
    if wanted_mlp and not mlp_hits:
        raise ValueError(
            f"no dense MLP blocks named {wanted_mlp} to target in this model ({info.label}); "
            f"every feed-forward block routes across experts. Use targets='attention'."
        )

    if len(mlp_hits) != len(mlp_candidates):
        # A short name would also wrap expert matrices; be exact instead.
        chosen = set(attn_hits + mlp_hits)
        return [n for n in names if n in chosen]
    present = {_leaf(n) for n in names}
    return [leaf for leaf in wanted_attn + wanted_mlp if leaf in present]


def build_examples(
    tokenizer, rows: Sequence[TrainRow], max_length: int
) -> Tuple[List[Dict[str, List[int]]], Dict[str, Any]]:
    """Tokenize rows into ``{"input_ids", "labels"}`` with the prompt masked to -100.

    The prompt is rendered through the chat template (thinking off) when the
    tokenizer has one, otherwise as ``f"{prompt}\\n\\n"`` (a system prompt is
    prepended the same way). The sequence is prompt + response + EOS. A row
    whose prompt alone fills ``max_length`` is dropped; a response that does
    not fit is cut from its tail and then carries no EOS, because teaching the
    model to stop after a cut-off answer would teach it to stop early.
    """
    from vivasecuris.aiasylum.weights.capture import format_chat, has_chat_template

    has_template = has_chat_template(tokenizer)
    eos = tokenizer.eos_token_id
    examples: List[Dict[str, List[int]]] = []
    dropped = truncated = total_tokens = 0

    for row in rows:
        # The same renderer capture and serving use, so a base-model adapter's
        # training positions coincide with the positions later read and served.
        messages = []
        if row.system:
            messages.append({"role": "system", "content": row.system})
        messages.append({"role": "user", "content": row.prompt})
        prompt_text, applied = format_chat(tokenizer, messages, thinking=False)
        prompt_ids = list(tokenizer(prompt_text, add_special_tokens=not applied)["input_ids"])
        if len(prompt_ids) >= max_length:
            dropped += 1
            continue
        response_ids = list(tokenizer(row.response, add_special_tokens=False)["input_ids"]) if row.response else []
        tail = [eos] if eos is not None else []
        budget = max_length - len(prompt_ids)
        if len(response_ids) + len(tail) > budget:
            response_ids, tail = response_ids[:budget], []
            truncated += 1
        ids = prompt_ids + response_ids + tail
        labels = [-100] * len(prompt_ids) + response_ids + tail
        examples.append({"input_ids": ids, "labels": labels})
        total_tokens += len(ids)

    stats = {
        "kept": len(examples),
        "dropped": dropped,
        "truncated": truncated,
        "chat_template": has_template,
        "mean_tokens": (total_tokens / len(examples)) if examples else 0.0,
        "max_length": max_length,
    }
    return examples, stats


def _pad_id(tokenizer) -> int:
    for value in (tokenizer.pad_token_id, tokenizer.eos_token_id):
        if value is not None:
            return int(value)
    return 0


def _collate(examples: Sequence[Dict[str, List[int]]], pad_id: int, device) -> Dict[str, Any]:
    """Right-pad a micro-batch: ``pad_id`` in the inputs, -100 in the labels."""
    import torch

    width = max(len(e["input_ids"]) for e in examples)
    input_ids = torch.full((len(examples), width), pad_id, dtype=torch.long)
    labels = torch.full((len(examples), width), -100, dtype=torch.long)
    attention = torch.zeros((len(examples), width), dtype=torch.long)
    for i, e in enumerate(examples):
        n = len(e["input_ids"])
        input_ids[i, :n] = torch.tensor(e["input_ids"], dtype=torch.long)
        labels[i, :n] = torch.tensor(e["labels"], dtype=torch.long)
        attention[i, :n] = 1
    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention.to(device),
        "labels": labels.to(device),
    }


def _eval_loss(model, batches: Sequence[Dict[str, Any]]) -> Optional[float]:
    """Mean cross-entropy per labelled token over ``batches``; restores train mode."""
    import torch
    import torch.nn.functional as F

    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    try:
        with torch.no_grad():
            for batch in batches:
                logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
                shift_logits = logits[:, :-1].float()
                shift_labels = batch["labels"][:, 1:]
                total += F.cross_entropy(
                    shift_logits.reshape(-1, shift_logits.shape[-1]), shift_labels.reshape(-1),
                    ignore_index=-100, reduction="sum",
                ).item()
                count += int((shift_labels != -100).sum().item())
    finally:
        if was_training:
            model.train()
    return total / count if count else None


def train_lora(
    model,
    tokenizer,
    rows: Sequence[TrainRow],
    spec: LoraSpec,
    run_dir,
    *,
    progress: Callable[[dict], None],
    should_stop: Callable[[], bool],
    loss_fn: Optional[Callable[[Any, Dict[str, Any]], Tuple[Any, Dict[str, float]]]] = None,
) -> Dict[str, Any]:
    """Attach a LoRA adapter to ``model`` and train it on ``rows``.

    ``model`` is modified in place: peft wraps the target modules, so the same
    object can be passed to :func:`merge_and_save` afterwards. The adapter is
    saved to ``run_dir/adapter`` (or ``run_dir/adapter-partial`` when
    ``should_stop`` asked for an early exit), and ``run_dir/train_log.jsonl``
    receives one JSON record per optimizer step and per evaluation.

    ``loss_fn(model, batch) -> (loss, extras)`` replaces the default masked
    cross-entropy; distillation uses it. ``extras`` values are averaged per
    step and reported alongside the loss.
    """
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import get_linear_schedule_with_warmup

    spec.validate()
    rows = list(rows)
    if not rows:
        raise ValueError("no training rows")
    without_response = sum(1 for r in rows if not r.response)
    if without_response:
        raise ValueError(
            f"{without_response} of {len(rows)} rows have no response. LoRA training "
            f"needs a target for every prompt; for prompts-only data run distillation."
        )

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train_log.jsonl"
    log_path.write_text("")

    def log(record: dict) -> None:
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    # Seeded split, then seeded shuffles, so a rerun sees the same batches.
    rng = random.Random(spec.seed)
    order = list(range(len(rows)))
    rng.shuffle(order)
    n_eval = min(spec.eval_rows, len(rows) // 5)
    eval_rows = [rows[i] for i in order[:n_eval]]
    train_rows = [rows[i] for i in order[n_eval:]]

    train_examples, stats = build_examples(tokenizer, train_rows, spec.max_length)
    eval_examples: List[Dict[str, List[int]]] = []
    if eval_rows:
        eval_examples, _ = build_examples(tokenizer, eval_rows, spec.max_length)
    if not train_examples:
        raise ValueError(
            f"every training row was dropped: all {len(train_rows)} prompts alone reach "
            f"max_length={spec.max_length}"
        )
    stats = {**stats, "train_rows": len(train_rows), "eval_rows": len(eval_rows),
             "eval_kept": len(eval_examples)}

    torch.manual_seed(spec.seed)
    target_modules = resolve_target_modules(model, spec)
    config = LoraConfig(
        r=spec.rank, lora_alpha=spec.alpha, lora_dropout=spec.dropout,
        target_modules=target_modules, task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, config)
    base = peft_model.get_base_model()
    prior_use_cache = getattr(base.config, "use_cache", None)
    if spec.gradient_checkpointing:
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        base.enable_input_require_grads()
        base.config.use_cache = False
    peft_model.train()

    trainable = [p for p in peft_model.parameters() if p.requires_grad]
    trainable_params = sum(p.numel() for p in trainable)
    total_params = sum(p.numel() for p in peft_model.parameters())
    device = next(peft_model.parameters()).device
    pad_id = _pad_id(tokenizer)

    micro, accum = spec.batch_size, spec.grad_accum
    steps_per_epoch = math.ceil(len(train_examples) / (micro * accum))
    total_steps = steps_per_epoch * spec.epochs
    if spec.max_steps is not None:
        total_steps = min(spec.max_steps, total_steps)

    optimizer = torch.optim.AdamW(trainable, lr=spec.lr, weight_decay=0.0)
    warmup = int(round(spec.warmup_ratio * total_steps))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total_steps)

    eval_batches = [
        _collate(eval_examples[s:s + micro], pad_id, device)
        for s in range(0, len(eval_examples), micro)
    ]
    history: List[Dict[str, Any]] = []
    started = time.time()
    logger.info(
        "LoRA r=%d on %d modules: %d trainable of %d params, %d steps over %d examples on %s",
        spec.rank, len(target_modules), trainable_params, total_params, total_steps,
        len(train_examples), device,
    )

    def run_eval(step: int) -> Optional[float]:
        if not eval_batches:
            return None
        loss = _eval_loss(peft_model, eval_batches)
        if device.type == "mps":
            torch.mps.empty_cache()
        record = {"phase": "eval", "step": step, "eval_loss": loss}
        history.append({"step": step, "eval_loss": loss})
        log(record)
        progress(record)
        return loss

    def finish(adapter_dir: Path, stopped: bool, eval_after: Optional[float], step: int,
               last_loss: Optional[float], tokens: int) -> Dict[str, Any]:
        peft_model.save_pretrained(str(adapter_dir))
        if prior_use_cache is not None:
            base.config.use_cache = prior_use_cache
        return {
            "steps": step,
            "total_steps": total_steps,
            "epochs": spec.epochs,
            "final_loss": last_loss,
            "eval_loss_before": eval_before,
            "eval_loss_after": eval_after,
            "trainable_params": trainable_params,
            "total_params": total_params,
            "target_modules": list(target_modules),
            "elapsed_seconds": time.time() - started,
            "tokens": tokens,
            "device": str(device),
            "dtype": str(next(base.parameters()).dtype).replace("torch.", ""),
            "dataset": stats,
            "history": history,
            "stopped": stopped,
            "adapter_path": str(adapter_dir),
        }

    eval_before = run_eval(0)
    step, tokens_seen, last_loss = 0, 0, None
    stopped = should_stop()

    for _epoch in range(spec.epochs):
        if stopped or step >= total_steps:
            break
        idx = list(range(len(train_examples)))
        rng.shuffle(idx)
        micro_batches = [idx[s:s + micro] for s in range(0, len(idx), micro)]
        groups = [micro_batches[g:g + accum] for g in range(0, len(micro_batches), accum)]
        for group in groups:
            if step >= total_steps:
                break
            optimizer.zero_grad(set_to_none=True)
            step_loss, step_tokens = 0.0, 0
            extras_sum: Dict[str, float] = {}
            for members in group:
                batch = _collate([train_examples[i] for i in members], pad_id, device)
                if loss_fn is not None:
                    loss, extras = loss_fn(peft_model, batch)
                else:
                    loss, extras = peft_model(**batch).loss, {}
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss at step {step + 1}; lower lr or check the data")
                (loss / len(group)).backward()
                step_loss += loss.item() / len(group)
                for key, value in extras.items():
                    extras_sum[key] = extras_sum.get(key, 0.0) + float(value) / len(group)
                step_tokens += int(batch["attention_mask"].sum().item())
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            lr = float(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()
            step += 1
            tokens_seen += step_tokens
            last_loss = step_loss
            record = {
                "phase": "train", "step": step, "total": total_steps, "loss": step_loss,
                "lr": lr, "tokens": tokens_seen, "elapsed": time.time() - started, **extras_sum,
            }
            history.append({"step": step, "loss": step_loss, "lr": lr, **extras_sum})
            log(record)
            progress(record)
            if step < total_steps and spec.eval_every > 0 and step % spec.eval_every == 0:
                run_eval(step)
            if should_stop():
                stopped = True
                break

    if stopped:
        logger.info("Stop requested after step %d; saving a partial adapter", step)
        return finish(run_dir / "adapter-partial", True, None, step, last_loss, tokens_seen)

    eval_after = run_eval(step)
    return finish(run_dir / "adapter", False, eval_after, step, last_loss, tokens_seen)


def _merge_adapters(model):
    """Fold the LoRA deltas into the base weights and drop the wrappers.

    Accepts the ``PeftModel`` peft returns or, since peft wraps modules in
    place, the very model object that was handed to :func:`train_lora`.
    """
    if hasattr(model, "merge_and_unload"):
        return model.merge_and_unload()
    from peft.tuners.tuners_utils import BaseTunerLayer

    merged_any = False
    for name, module in list(model.named_modules()):
        if not isinstance(module, BaseTunerLayer):
            continue
        module.merge()
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, module.get_base_layer())
        merged_any = True
    if not merged_any:
        raise ValueError("model carries no LoRA layers to merge; run train_lora first")
    return model


def merge_and_save(
    model,
    tokenizer,
    out_dir,
    *,
    source_model: str,
    spec: LoraSpec,
    train_result: Dict[str, Any],
    dataset_meta: dict,
    notes: Optional[str] = None,
    extra: Optional[dict] = None,
    method: str = "lora_merge",
) -> Path:
    """Merge the adapter into the base weights and save a loadable model directory.

    Writes the weights, the tokenizer and an ``asylum_surgery.json`` manifest
    whose ``extra`` records the LoRA spec, the training summary (history
    excluded) and the dataset provenance. Refuses a non-empty ``out_dir``.
    """
    from vivasecuris.aiasylum.interp.core.hook_registry import describe_architecture
    from vivasecuris.aiasylum.weights.manifest import SurgeryManifest

    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} already exists and is not empty; refusing to overwrite a model directory")

    merged = _merge_adapters(model)
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))

    info = describe_architecture(merged)
    manifest = SurgeryManifest(
        source_model=source_model,
        method=method,
        architecture=info.family,
        model_type=info.model_type,
        coverage_verified=None,
        notes=notes,
        extra={
            "lora": spec.to_dict(),
            "train": {k: v for k, v in train_result.items() if k != "history"},
            "dataset": dict(dataset_meta),
            **(extra or {}),
        },
    )
    manifest.save(out)
    logger.info("Merged LoRA model written to %s", out)
    return out
