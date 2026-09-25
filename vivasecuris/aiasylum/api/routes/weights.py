"""Weight surgery: deriving a behavioural direction, proving it, and writing it in.

Drives `vivasecuris.aiasylum.weights`, which until now was reachable only from
`aiasylum weights ...` on a terminal.

Structurally a sibling of `interp.py` -- own progress and cancellation managers
(both key on bare integers, so sharing them would make weight run 5 and interp
run 5 the same subscription), the same create-row-then-`asyncio.create_task`
shape, the same SSE framing. Four things are specific to this router:

* **It can write gigabytes, irreversibly.** Every output path is built from a
  validated slug under one configured root, surgery writes to a staging
  directory that is renamed on success, and deleting a produced model needs the
  name echoed back.
* **Preflight blocks rather than warns.** Activation capture and a 6 GB
  `save_pretrained` both fail slowly and silently under memory or disk
  pressure. `WEIGHT_SURGERY.md` calls the warnings blocking; here they are.
* **The stages gate each other.** A sweep and a surgery both consume a derived
  direction, and the chain exists so a direction that does not separate is
  caught before 6 GB is written rather than after.
* **Method and objective are recorded, not assumed.** The pipeline defaults to
  a rank-1 `direction_scale` edit aimed at refusal, but neither is a constraint
  of the engine, and a manifest that omits them cannot say what a modified
  model was actually aimed at.

Nothing under `weights/` is imported at module scope: the default install is
deliberately torch-free, and these read endpoints must work without the extra.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from uuid import uuid4
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from vivasecuris.aiasylum.api.cancellation import CancellationManager
from vivasecuris.aiasylum.api.model_jobs import hold, model_slot, slot_status
from vivasecuris.aiasylum.api.progress_events import ProgressEventManager
from vivasecuris.aiasylum.constants import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_PENDING,
    STATUS_RUNNING,
    WEIGHT_KINDS_CONSUMING_DIRECTION,
    WEIGHT_KINDS_WRITING_MODELS,
)
from vivasecuris.aiasylum.database import InterpRun, TestRun, WeightRun, get_session

logger = logging.getLogger(__name__)

router = APIRouter()

# A third id space, separate from test runs and from interp runs.
weights_progress = ProgressEventManager()
weights_cancellation = CancellationManager()

# Shared with interp: see api/model_jobs.py.
_semaphore = model_slot

MANIFEST_NAME = "asylum_surgery.json"

# Rejects "/", "..", and a leading dot by construction, so a slug can never
# escape the models root or hide a directory from the scan.
SLUG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

STAGING_PREFIX = ".staging-"


def _runs_root() -> Path:
    from config import settings

    return Path(settings.weights_runs_root)


def _models_root() -> Path:
    from config import settings

    return Path(settings.weights_models_root)


# --------------------------------------------------------------------------
# Registries
# --------------------------------------------------------------------------
#
# Read by the UI rather than hardcoded there, the same way
# `GET /api/v1/models/providers` is -- which exists because `servus` and
# `agentic` were once registered but unreachable from the app. Methods that are
# not built are listed and disabled with a reason instead of being invisible,
# so the page shows where directional ablation sits among the alternatives
# rather than implying it is the only way to change a model.

METHODS = {
    # Deriving is not an edit, but it is a method choice: one difference-in-means
    # vector, or an orthonormal subspace that catches components a single vector
    # misses. The stage registry pairs each stage with one of these.
    "diff_in_means": {
        "label": "Difference in means (optionally a subspace)",
        "description": (
            "Capture last-prompt-token residuals for both classes and take the "
            "difference in means per layer, choosing the layer on held-out data. With "
            "a rank above 1, build an orthonormal basis instead of a single vector."
        ),
        "stage": "direction",
        "permanent": False,
        "available": True,
    },
    "direction_scale": {
        "label": "Directional ablation / amplification (permanent)",
        "description": (
            "Edits every residual-writing matrix. With a single direction, beta scales "
            "its component (0 removes, 1 no-op, 2 amplifies); with a subspace, the whole "
            "orthonormal basis is removed at strength k. Produces a real model directory."
        ),
        "stage": "surgery",
        "permanent": True,
        "available": True,
    },
    # One entry, not two. `sweep_alpha` measures ablation and addition in the
    # same run and returns them as rows of one table, so offering them as
    # separate "methods" would be a menu where both choices do the same thing.
    "steering_sweep": {
        "label": "Steering sweep: ablation + addition (inference-time)",
        "description": (
            "Measures both interventions at once. Ablation (h <- h - (h.r)r at every "
            "layer) is scale-free and is the row to trust; addition (h <- h + alpha*r "
            "at the direction's layer, alpha relative to that layer's residual norm) "
            "maps the strength curve on either side of it."
        ),
        "stage": "sweep",
        "permanent": False,
        "available": True,
    },
    "subspace_search": {
        "label": "Subspace search: rank x strength, capability-gated",
        "description": (
            "Previews every rank/strength candidate at inference time, keeps only "
            "those that hold a factual capability floor, and reports the frontier -- "
            "what more compliance actually costs. Writes nothing."
        ),
        "stage": "select",
        "permanent": False,
        "available": True,
    },
    "rfm_agop": {
        "label": "RFM-AGOP refusal cone (multi-dimensional)",
        "description": (
            "Fits a kernel classifier on the same captures and reads the refusal "
            "subspace off its gradient outer products (arXiv 2607.02396). Returns a "
            "rank-k cone with per-direction weights for soft ablation; larger models "
            "need three or more directions where one mean difference is not enough."
        ),
        "stage": "direction",
        "permanent": False,
        "available": True,
    },
    "subspace_curve": {
        "label": "Rank curve: refusal against directions removed",
        "description": (
            "Previews removing the first 1..k directions of a subspace at inference "
            "time and reports refusal, compliance and the capability control for each. "
            "Says how many directions this model actually needs."
        ),
        "stage": "sweep",
        "permanent": False,
        "available": True,
    },
    "compare_rederive": {
        "label": "Measure, then attack again",
        "description": (
            "The baseline-vs-modified measurement plus a re-derivation of the refusal "
            "direction on the modified model: held-out AUC, stable rank and ablation "
            "refusal after the edit. This is how a hardening defence is tested."
        ),
        "stage": "compare",
        "permanent": False,
        "available": True,
    },
    "linear_probe": {
        "label": "Harmful-intent probe (raw activations)",
        "description": (
            "Logistic regression per layer on pooled residual activations, with a "
            "shuffled-label null at every layer. Raw activations on purpose: SAE "
            "features measured worse for this task on every model in SAEGuardBench "
            "(2026). Produces a monitor that scores live prompts."
        ),
        "stage": "probe",
        "permanent": False,
        "available": True,
    },
    "model_compare": {
        "label": "Measure: baseline vs modified",
        "description": (
            "Runs both models through the identical loader, tokenizer and greedy "
            "decoding, so the weights are the only variable. Reports refusal on "
            "harmful and harmless prompts plus the factual capability control."
        ),
        "stage": "compare",
        "permanent": False,
        "available": True,
    },
    "expert_routing": {
        "label": "Expert routing statistics (MoE)",
        "description": (
            "Runs harmful and harmless prompts through a mixture-of-experts model and "
            "records, per layer and per expert, how often each expert is selected on "
            "each set. The difference says which experts a behaviour routes through. "
            "Writes nothing to the model."
        ),
        "stage": "routing",
        "permanent": False,
        "available": True,
    },
    "expert_direction_scale": {
        "label": "Direction edit inside chosen experts (permanent, partial)",
        "description": (
            "Scales a direction's component (or removes a subspace) in the "
            "down-projections of the experts you name, and nowhere else. Partial by "
            "design: untouched experts still write the direction when routed to. "
            "For asking what those experts carry, not for removing the behaviour."
        ),
        "stage": "expert_surgery",
        "permanent": True,
        "available": True,
    },
    "expert_ablate": {
        "label": "Expert ablation (permanent, partial)",
        "description": (
            "Scales the whole down-projection of the experts you name (0 removes "
            "their write entirely), with routing left exactly as it was. Needs no "
            "direction. The cleanest test of whether an expert carries a behaviour."
        ),
        "stage": "expert_surgery",
        "permanent": True,
        "available": True,
    },
    "lora": {
        "label": "LoRA fine-tune (plain-precision adapter, merged into a new model)",
        "description": (
            "Gradient training of a low-rank adapter on the attention projections, with "
            "the loss masked to the response so the model learns to answer rather than "
            "to repeat the question. No bitsandbytes on Apple silicon, so no QLoRA: "
            "models up to about 3B fit on a 24 GB Mac. The adapter is merged into a "
            "full model directory with the same provenance manifest surgery writes."
        ),
        "stage": "lora",
        "permanent": True,
        "available": True,
    },
    "response_distill": {
        "label": "Response distillation (teacher text -> student LoRA)",
        "description": (
            "A local open-weights teacher answers the prompts; the student is LoRA-trained "
            "on those answers. The cheapest distillation and the only kind that needs "
            "just a generation pass. The teacher is unloaded before training starts, so "
            "memory is the larger of the two models rather than their sum."
        ),
        "stage": "distill",
        "permanent": True,
        "available": True,
    },
    "logit_distill": {
        "label": "Logit distillation (KL to the teacher's distribution)",
        "description": (
            "The student matches the teacher's full next-token distribution at every "
            "response position, temperature-softened and mixed with plain cross-entropy. "
            "Far more signal per example than text alone, but both models stay resident "
            "and the two tokenizers must be identical."
        ),
        "stage": "distill",
        "permanent": True,
        "available": True,
    },
}

# Present in the landscape, not built here. Reasons come from
# docs/MODEL_MODIFICATION.md so the UI and the doc cannot drift apart.
UNAVAILABLE_METHODS = {
    "task_arithmetic": (
        "Task arithmetic",
        "Needs at least one fine-tune to subtract from the base. Not implemented.",
    ),
    "model_merge": (
        "Model merging (SLERP / TIES / DARE)",
        "No training required, but mergekit is not wired in. Not implemented.",
    ),
    "rome_memit": (
        "ROME / MEMIT",
        "Surgical fact editing rather than behaviour editing. Not implemented.",
    ),
    "pruning": (
        "Pruning (SparseGPT / Wanda)",
        "Targets speed and size rather than behaviour. Not implemented.",
    ),
    "quantization": (
        "Quantization (GGUF / AWQ / GPTQ)",
        "For serving, and it perturbs the tensors being measured. Quantize after "
        "surgery, never before. Not implemented.",
    ),
}

# What the direction is derived to separate. The engine never required this to
# be refusal -- build_split() takes arbitrary contrast pairs -- so the default
# is a default, not a limit.
OBJECTIVES = {
    "refusal": {
        "label": "Refusal",
        "description": (
            "Separates requests a safety-tuned model refuses from neutral requests "
            "matched in form."
        ),
        "note": (
            "Five restricted-advice scenarios are excluded on purpose: models answer "
            "them rather than refusing, and including them moved baseline refusal from "
            "92% to 33% and made the direction underivable."
        ),
        "configurable": False,
    },
    "refusal_narrow": {
        "label": "Refusal, narrowed to chosen scenarios",
        "description": (
            "The same contrast restricted to scenarios you pick -- for targeting one "
            "behaviour rather than refusal in general."
        ),
        "note": "Fewer scenarios means fewer prompts; below 8 per class the split is rejected.",
        "configurable": True,
    },
    "over_refusal": {
        "label": "Over-refusal (benign prompts the model refuses)",
        "description": (
            "Contrasts benign prompts the model refuses against benign prompts it "
            "answers. Over-refusal directions are task-dependent and sit inside the "
            "benign clusters (arXiv 2603.27518), so this is derived separately from "
            "refusal and reported with its overlap against the global refusal direction."
        ),
        "note": (
            "Supply the two prompt lists in objective_config as 'refused' and "
            "'answered' (from a test run's analysis); below 8 per class the split is rejected."
        ),
        "configurable": True,
    },
    "custom": {
        "label": "Custom contrast",
        "description": (
            "Any pair of prompt sets. The direction encodes whatever separates them, "
            "so a contrast that differs in register as well as content will encode that too."
        ),
        "note": (
            "Academic benchmark questions differ from harmful prompts in register, so a "
            "direction fit against them partly encodes 'academic vs conversational'."
        ),
        "configurable": True,
    },
    "dataset": {
        "label": "Defined by the training rows",
        "description": (
            "Whatever the rows teach. For LoRA and distillation runs, where the behaviour "
            "is set by the data rather than by a prompt contrast."
        ),
        "note": (
            "To distil refusals from a larger local model, keep the objective as refusal "
            "and take the prompts from it (dataset source 'objective')."
        ),
        "configurable": False,
    },
}

STAGES = {
    "direction": {
        "label": "Derive a direction",
        "description": (
            "Capture last-prompt-token residuals for both classes, take the "
            "difference in means per layer, and pick the layer that separates best on "
            "held-out data."
        ),
        "needs": ["source_model", "objective"],
        "writes": "a few KB",
    },
    "sweep": {
        "label": "Check it is causal",
        "description": (
            "Steer with the direction and measure refusal. A flat ablation row means "
            "the direction is not causal for this model and surgery would fail silently."
        ),
        "needs": ["source_model", "source_run_id"],
        "writes": "a few KB",
    },
    "select": {
        "label": "Find the best capability-safe edit",
        "description": (
            "Search subspace rank against removal strength, previewing each candidate "
            "at inference time and keeping only those that hold the factual capability "
            "control. Reports what more compliance costs."
        ),
        "needs": ["source_model", "source_run_id"],
        "writes": "a few KB",
    },
    "surgery": {
        "label": "Write the edit into the weights",
        "description": (
            "Permanent update of every residual-writing matrix -- a single direction "
            "scaled by beta, or a whole subspace removed at strength k. Produces a "
            "normal Hugging Face directory plus a provenance manifest."
        ),
        "needs": ["source_model", "source_run_id", "output_name", "beta"],
        "writes": "a full model copy, roughly 6 GB for a 3B in bf16",
    },
    "probe": {
        "label": "Train a harmful-intent monitor",
        "description": (
            "Fits a linear probe on pooled residual activations over harmful, "
            "jailbreak-wrapped and benign prompts, holding out whole jailbreak "
            "families so the reported number is generalisation rather than recall. "
            "The result scores live prompts and flags 'the model knew and complied'."
        ),
        "needs": ["source_model"],
        "writes": "a few hundred KB",
    },
    "compare": {
        "label": "Measure what changed",
        "description": (
            "Run a baseline and a modified model through the identical loader and "
            "decoding, and report refusal plus the factual capability control for each. "
            "A refusal drop with a capability drop is a lobotomy, not a jailbreak."
        ),
        "needs": ["source_model", "modified_model"],
        "writes": "a few KB",
    },
    "routing": {
        "label": "Find which experts carry it",
        "description": (
            "For a mixture-of-experts model: run both prompt classes and record how "
            "often each expert in each layer is selected on each. The experts that "
            "fire far more on one class are where an expert-level edit should aim."
        ),
        "needs": ["source_model", "objective"],
        "writes": "a few KB",
    },
    "expert_surgery": {
        "label": "Edit specific experts",
        "description": (
            "Permanent, partial edit of the experts you name in the layers you name: "
            "ablate them outright, or scale a direction inside them. Produces a "
            "normal Hugging Face directory whose manifest says the edit is partial."
        ),
        "needs": ["source_model", "output_name", "expert_selection"],
        "writes": "a full model copy, roughly 6 GB for a 3B in bf16",
    },
    "lora": {
        "label": "Fine-tune with LoRA",
        "description": (
            "Train a low-rank adapter on rows of prompt and response, then merge it into a "
            "new model directory. The first gradient-based method here: for hardening, for "
            "teaching a format, for anything a weight edit cannot express."
        ),
        "needs": ["source_model", "dataset", "output_name"],
        "writes": "an adapter of a few MB, plus a full model copy when merged",
    },
    "distill": {
        "label": "Distil from a local teacher",
        "description": (
            "A larger local open-weights model answers a prompt set; the student is "
            "LoRA-trained to imitate it, from its text or from its logits. The teacher has "
            "to be loadable here: only a local model's output is trainable on."
        ),
        "needs": ["source_model", "teacher_model", "output_name"],
        "writes": "an adapter of a few MB, plus a full model copy when merged",
    },
}

# What each stage runs when the caller does not name a method.
DEFAULT_METHOD_FOR_STAGE = {
    "direction": "diff_in_means",
    "sweep": "steering_sweep",
    "select": "subspace_search",
    "surgery": "direction_scale",
    "probe": "linear_probe",
    "compare": "model_compare",
    "routing": "expert_routing",
    "expert_surgery": "expert_ablate",
    "lora": "lora",
    "distill": "response_distill",
}

# The three points of beta the operating guide names, presented as intents so
# the number does not have to be rediscovered from the docs.
BETA_PRESETS = [
    {"value": 0.0, "label": "Remove", "description": "Ablate the direction."},
    {
        "value": 1.0,
        "label": "Preserve (control)",
        "description": "Bit-identical no-op. The honest control for any comparison.",
    },
    {
        "value": 2.0,
        "label": "Amplify",
        "description": (
            "Strengthen whatever the direction encodes. Measured cost on "
            "Qwen2.5-0.5B: +25 points refusal, -25 points factual accuracy."
        ),
    },
]


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------


class WeightRunRequest(BaseModel):
    """Flat rather than a discriminated union.

    `InterpRunRequest` is flat for the same reason: the launcher form is driven
    by each stage's `needs[]` list, which reads one field set and shows a
    subset. A Union would be more type-correct and would fight that pattern for
    no user-visible gain.
    """

    # source_model collides with pydantic's protected "model_" prefix.
    model_config = {"protected_namespaces": ()}

    kind: str = Field(..., description="direction | sweep | surgery")
    source_model: str
    source_run_id: Optional[int] = None

    # Defaults per stage, so a caller never has to know a method name to run
    # the obvious thing for the stage it picked.
    method: Optional[str] = None
    objective: str = "refusal"
    objective_config: Optional[Dict[str, Any]] = None

    device: str = "auto"
    dtype: str = "bfloat16"

    # direction
    n_per_class: int = 128
    test_fraction: float = 0.25
    seed: int = 0
    batch_size: int = 8
    max_length: int = 512
    # A subspace catches refusal components a single difference-in-means vector
    # misses. None keeps the rank-1 behaviour.
    subspace_rank: Optional[int] = None
    pool_layers: Optional[int] = None
    # rfm_agop: cone rank and iterations; None falls back to subspace_rank / 4.
    rfm_rank: Optional[int] = None
    rfm_iterations: int = 5
    rfm_beta: float = 0.5
    # over_refusal: also derive the global refusal direction and report overlap.
    report_overlap: bool = True

    # sweep extras: keep a reasoning model's <think> block on while steering,
    # and record refusal-decision timelines for the first N held-out prompts.
    thinking: bool = False
    timeline_prompts: int = 0

    # compare extras: re-derive the direction on the modified model (the
    # attack-after-defence check) and run the broad-misalignment control.
    rederive: bool = False
    misalignment_control: bool = False

    # probe
    pooling: str = "mean"
    prompt_suffix: Optional[str] = None
    use_eliciting_suffix: bool = False
    n_direct: int = 120
    n_jailbreak: int = 120
    n_benign: int = 240
    holdout_techniques: int = 2
    jailbreak_examples: Optional[List[Dict[str, str]]] = None

    # sweep
    n_prompts: int = 8
    max_new_tokens: int = 64
    alphas: Optional[List[float]] = None
    include_ablation: bool = True
    # On by default: the operating guide is explicit that a refusal delta means
    # nothing without it. Trimmed to 6 questions so it roughly doubles rather
    # than triples the sweep.
    capability_control: bool = True
    capability_limit: int = 6

    # select + compare
    ranks: Optional[List[int]] = None
    ks: Optional[List[float]] = None
    factual_floor: float = 0.05
    # "builtin" is the 12-question smoke test; "mmlu:<n>" draws n MMLU items.
    capability_set: str = "builtin"

    # surgery
    output_name: Optional[str] = None
    beta: float = 0.0
    include_embeddings: bool = True
    use_subspace: bool = False
    k: Optional[float] = None

    # compare
    modified_model: Optional[str] = None

    # expert_surgery (mixture-of-experts only): {"12": [3, 7], "15": "all"}.
    # Keys are strings because they arrive as JSON object keys.
    expert_selection: Optional[Dict[str, Any]] = None
    expert_scale: float = 0.0
    include_shared_expert: bool = False

    # lora + distill: gradient training in a worker process. Exactly one dataset
    # source: inline rows, a benchmark's items, or the objective's prompt corpus
    # (prompts only, so distillation only). Never a filesystem path from the
    # browser; the CLI reads files.
    dataset_rows: Optional[List[Dict[str, Any]]] = None
    dataset_benchmark: Optional[Dict[str, Any]] = None
    dataset_source: Optional[str] = None
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_targets: str = "attention"
    epochs: int = 1
    max_steps: Optional[int] = None
    lr: float = 2e-4
    train_batch_size: int = 1
    grad_accum: int = 8
    gradient_checkpointing: bool = False
    merge: bool = True
    eval_rows: int = 32
    # distill
    teacher_model: Optional[str] = None
    distill_temperature: float = 2.0
    ce_weight: float = 0.5
    teacher_max_new_tokens: int = 256
    teacher_system_prompt: Optional[str] = None

    lineage_parent: Optional[str] = Field(None, max_length=1024)

    notes: Optional[str] = None
    # Preflight codes the caller explicitly overrode. Naming each one means a
    # stale UI cannot blanket-force, and a newly appearing condition still blocks.
    acknowledge: List[str] = []


class WeightRunResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    id: int
    kind: str
    status: str
    source_model: str
    source_run_id: Optional[int]
    method: Optional[str]
    objective: Optional[str]
    out_dir: Optional[str]
    artifact_bytes: Optional[int]
    error: Optional[str]
    created_at: Optional[datetime]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    metadata: Dict[str, Any] = {}
    # On a machine billed by the hour, how long each stage took is a cost.
    elapsed_seconds: Optional[float] = None

    @classmethod
    def from_orm_row(cls, row: WeightRun) -> "WeightRunResponse":
        return cls(
            id=row.id,
            kind=row.kind,
            status=row.status,
            source_model=row.source_model,
            source_run_id=row.source_run_id,
            method=row.method,
            objective=row.objective,
            out_dir=row.out_dir,
            artifact_bytes=row.artifact_bytes,
            error=row.error,
            created_at=row.created_at,
            started_at=row.started_at,
            completed_at=row.completed_at,
            metadata=row.meta_data or {},
            elapsed_seconds=(
                (row.completed_at or datetime.utcnow()) - row.started_at
            ).total_seconds() if row.started_at else None,
        )


class PreflightCheck(BaseModel):
    code: str
    severity: str           # blocking | advisory
    message: str
    acknowledgeable: bool


class PreflightResponse(BaseModel):
    kind: str
    checks: List[PreflightCheck]
    can_proceed: bool
    blocking_codes: List[str]
    memory: Dict[str, Any] = {}
    disk: Dict[str, Any] = {}


# --------------------------------------------------------------------------
# Path safety
# --------------------------------------------------------------------------


def _validate_slug(name: Optional[str]) -> str:
    if not name or not SLUG_RE.match(name):
        raise HTTPException(
            status_code=400,
            detail=(
                "output_name must be 1-64 characters of letters, digits, '.', '_' or "
                "'-', starting with a letter or digit. Paths are not accepted: the "
                "destination is always built under the configured models directory."
            ),
        )
    return name


def _resolve_output_dir(name: str) -> Path:
    """Build the destination under the models root, and prove it stays there.

    Resolving before the containment check is what makes this symlink-safe; the
    slug pattern alone would not be.
    """
    root = _models_root().resolve()
    candidate = (root / name).resolve()
    if root not in candidate.parents:
        raise HTTPException(
            status_code=400,
            detail=f"Refusing to write outside {root}.",
        )
    return candidate


def _is_inside(path: Path, root: Path) -> bool:
    try:
        return root.resolve() in path.resolve().parents
    except Exception:
        return False


def _dir_size(path: Path) -> int:
    try:
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    except Exception:
        return 0


def _estimated_write_gb(source_model: str, dtype: str = "bfloat16") -> Optional[float]:
    from vivasecuris.aiasylum.weights.resources import estimated_weights_gb

    return estimated_weights_gb(source_model, dtype)


# --------------------------------------------------------------------------
# Preflight
# --------------------------------------------------------------------------


def _interp_extra_installed() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401

        return True
    except ImportError:
        return False


def _device_info() -> Dict[str, Any]:
    """Where a run will execute, so the UI can say it rather than imply it."""
    from vivasecuris.aiasylum.weights.progress import gpu_report

    info: Dict[str, Any] = {"type": "unknown", "gpus": gpu_report()}
    try:
        from vivasecuris.aiasylum.interp.core.loader import resolve_device

        info["type"] = resolve_device("auto").type
    except Exception:
        pass
    if info["gpus"]:
        info["name"] = info["gpus"][0]["name"]
    return info


def _direction_summary(row: Optional[WeightRun]) -> Dict[str, Any]:
    return ((row.meta_data or {}).get("summary") or {}) if row is not None else {}


# Plain-precision LoRA on a 24 GB Mac: about 3.5B parameters in bf16. QLoRA
# would lift this, but bitsandbytes has no MPS backend (MODEL_MODIFICATION.md §4).
LORA_MAX_WEIGHTS_GB = 8.0
TEACHER_ADVISORY_GB = 16.0


def _training_preflight(
    kind: str, source_model: str, dtype: str, training: Dict[str, Any], mem: Dict[str, Any]
) -> List[PreflightCheck]:
    """What a gradient run needs that a weight edit does not: peft, headroom, a teacher."""
    checks: List[PreflightCheck] = []

    def add(code, severity, message, acknowledgeable=True):
        checks.append(PreflightCheck(code=code, severity=severity, message=message,
                                     acknowledgeable=acknowledgeable))

    try:
        import peft  # noqa: F401
    except ImportError:
        add("lora_extra_missing", "blocking",
            'Training needs the optional extra: pip install -e ".[lora]"', acknowledgeable=False)

    student_gb = _estimated_write_gb(source_model, dtype) if source_model else None
    if student_gb is not None and student_gb > LORA_MAX_WEIGHTS_GB:
        add("model_too_large", "blocking",
            f"{source_model} is about {student_gb:.1f} GB of weights in {dtype}. Plain-precision "
            f"LoRA here is sized for models up to about 3B (QLoRA needs bitsandbytes, which has "
            f"no MPS backend); a larger student will most likely exhaust memory mid-step.")

    teacher = (training.get("teacher_model") or "").strip() if kind == "distill" else ""
    level = training.get("distill_level") or "response"
    teacher_gb = _estimated_write_gb(teacher, dtype) if teacher else None
    if teacher:
        path = Path(teacher).expanduser()
        if path.exists() or teacher.startswith(("/", "./", "../", "~/", "models/")):
            from vivasecuris.aiasylum.api.model_catalog import checkpoint_status

            status = checkpoint_status(path)
            if status["availability"] != "ready":
                add("teacher_unavailable", "blocking",
                    f"The teacher checkpoint is unavailable on this server: {status['reason']}",
                    acknowledgeable=False)
        if teacher_gb is not None and teacher_gb > TEACHER_ADVISORY_GB:
            add("teacher_too_large", "advisory",
                f"The teacher is about {teacher_gb:.1f} GB; generating from it here will be slow, "
                f"and logit distillation keeps it resident for the whole run.")
        if level == "logit":
            add("tokenizer_check_at_start", "advisory",
                "Logit distillation needs identical student and teacher tokenizers. The worker "
                "compares their vocabularies before training and fails fast on a mismatch.")

    # The base plus activations plus the adapter's optimizer state: 2.5x the
    # weights is a conservative envelope on unified memory, and swapping under
    # training does not fail fast, it crawls.
    free = mem.get("free_gb")
    if student_gb is not None and free:
        need = student_gb * 2.5 + 2
        if teacher_gb is not None:
            need = (teacher_gb + student_gb * 2.5 + 2) if level == "logit" else (max(teacher_gb, student_gb * 2.5) + 2)
        if free < need:
            parts = f"{student_gb:.1f} GB of student weights"
            if teacher_gb is not None:
                parts += f", {teacher_gb:.1f} GB of teacher"
            add("training_memory", "blocking",
                f"About {need:.0f} GB of free memory is needed ({parts}, activations and optimizer "
                f"state) and {free:.0f} GB is free. Free memory first, or acknowledge and expect swapping.")

    try:
        from vivasecuris.aiasylum.interp.core.loader import resolve_device

        if resolve_device("auto").type == "mps":
            add("mps_fallback", "advisory",
                "Some backward ops have no MPS kernel. The worker sets PYTORCH_ENABLE_MPS_FALLBACK=1 "
                "so they run on the CPU instead of failing, at a cost in speed.")
    except Exception:
        pass
    return checks


def _moe_preflight(source_model: str, expert_selection: Optional[Dict[str, Any]]) -> List[PreflightCheck]:
    """Is this a mixture-of-experts model whose experts can be addressed one by one?

    Uses a meta-device instance (shapes, no storage), the way
    `interp_preflight.model_info` does, so a 30B model costs a config fetch. A
    dense model or a fused-expert layout is refused here, before anything loads.
    """
    checks: List[PreflightCheck] = []
    try:
        from accelerate import init_empty_weights
        from transformers import AutoConfig, AutoModelForCausalLM

        from vivasecuris.aiasylum.interp.core.arch import moe_layout, normalize_expert_selection
        from vivasecuris.aiasylum.interp.core.loader import _check_model_id, get_hf_token

        _check_model_id(source_model)
        cfg = AutoConfig.from_pretrained(source_model, token=get_hf_token(), trust_remote_code=False)
        with init_empty_weights(include_buffers=True):
            model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=False)
            layout = moe_layout(model)
    except Exception as exc:
        checks.append(PreflightCheck(
            code="moe_unknown", severity="advisory", acknowledgeable=True,
            message=f"Could not inspect the model's expert layout before running: {exc}",
        ))
        return checks

    if not layout:
        checks.append(PreflightCheck(
            code="not_moe", severity="blocking", acknowledgeable=False,
            message=(
                f"{source_model} has no mixture-of-experts layers. Routing statistics "
                f"and expert edits need routed experts; use a direction edit on a dense model."
            ),
        ))
        return checks

    fused = sorted(layer for layer, info in layout.items() if not info.experts_are_modules)
    if fused:
        shown = ", ".join(str(l) for l in fused[:6]) + (", ..." if len(fused) > 6 else "")
        checks.append(PreflightCheck(
            code="fused_experts", severity="blocking", acknowledgeable=False,
            message=(
                f"Layers {shown} store their experts as fused tensors, which cannot be "
                f"edited or counted per expert in this surgery pass."
            ),
        ))
    if expert_selection is not None:
        try:
            normalize_expert_selection(expert_selection, layout)
        except ValueError as exc:
            checks.append(PreflightCheck(
                code="expert_selection_invalid", severity="blocking",
                acknowledgeable=False, message=str(exc),
            ))
    return checks


def _preflight_checks(
    kind: str,
    source_model: str = "",
    direction_row: Optional[WeightRun] = None,
    output_name: Optional[str] = None,
    modified_model: Optional[str] = None,
    dtype: str = "bfloat16",
    *,
    expert_selection: Optional[Dict[str, Any]] = None,
    training: Optional[Dict[str, Any]] = None,
) -> PreflightResponse:
    """Structured checks, so the UI can render severity and POST can verify acks.

    Built from `memory_report()` and `resident_ollama_models()` directly rather
    than by classifying `memory_warnings()`'s prose, which would couple this
    route to wording. Both are plain stdlib helpers with no torch import, so
    `weights/progress.py` is untouched and the CLI is unaffected.
    """
    from vivasecuris.aiasylum.weights.progress import (
        LOW_DISK_GB,
        gpu_report,
        memory_report,
        resident_ollama_models,
    )

    # A training run that keeps only its adapter writes a few MB under runs/,
    # not a model directory; the disk and memory bars are set accordingly.
    writes_weights = kind in WEIGHT_KINDS_WRITING_MODELS and bool((training or {}).get("merge", True))
    checks: List[PreflightCheck] = []

    def add(code, severity, message, acknowledgeable=True):
        checks.append(
            PreflightCheck(
                code=code, severity=severity, message=message,
                acknowledgeable=acknowledgeable,
            )
        )

    if kind == "compare" and modified_model:
        from vivasecuris.aiasylum.api.model_catalog import checkpoint_status
        ref = modified_model.strip()
        path = Path(ref).expanduser()
        if path.exists() or ref.startswith(("/", "./", "../", "~/", "models/")):
            status = checkpoint_status(path)
            if status["availability"] != "ready":
                add("modified_model_unavailable", "blocking",
                    f"Custom checkpoint is unavailable on this server: {status['reason']} Select a ready model from Models or finish saving it first.",
                    acknowledgeable=False)

    # Nothing else matters if the extra is missing.
    if not _interp_extra_installed():
        add(
            "interp_extra_missing", "blocking",
            'Needs the optional extra: pip install -e ".[interp]"',
            acknowledgeable=False,
        )
    elif kind in ("routing", "expert_surgery") and source_model.strip():
        checks.extend(_moe_preflight(source_model.strip(), expert_selection))

    mem = memory_report()
    sev = "blocking" if writes_weights else "advisory"

    if kind in ("lora", "distill") and _interp_extra_installed():
        checks.extend(_training_preflight(kind, source_model.strip(), dtype, training or {}, mem))

    for name, size in resident_ollama_models():
        add(
            "ollama_resident", sev,
            f"Ollama is holding {name} ({size}) in the same unified memory. Free it "
            f"with `ollama stop {name}`: under contention this does not error, it "
            f"silently collapses in speed.",
        )

    if mem.get("swap_total_gb") and mem["swap_used_gb"] / mem["swap_total_gb"] > 0.8:
        add(
            "swap_thrashing", sev,
            f"Swap is {mem['swap_used_gb']:.0f} GB of {mem['swap_total_gb']:.0f} GB "
            f"used. The machine is thrashing; expect order-of-magnitude slowdowns.",
        )

    # Stages that hold a model on the accelerator. Only the writing kinds also write.
    loads_heavily = kind in ("select", "compare", "routing") or kind in WEIGHT_KINDS_WRITING_MODELS
    gpus = gpu_report()
    model_gb = _estimated_write_gb(source_model, dtype)
    if kind == "compare" and modified_model:
        other_gb = _estimated_write_gb(modified_model, dtype)
        model_gb = max(model_gb or 0, other_gb or 0) or None
    if gpus and not writes_weights:
        needed_vram = model_gb * 1.25 + 2 if model_gb else None
        best = max(gpus, key=lambda g: g["free_gb"])
        if needed_vram and best["free_gb"] < needed_vram:
            add(
                "gpu_memory_low", "blocking" if loads_heavily else "advisory",
                f"{best['name']} has {best['free_gb']:.1f} GB free but the model needs "
                f"about {needed_vram:.1f} GB. Something else is holding the card; check "
                f"`nvidia-smi` before paying for a run that will spill or fail.",
            )

    # A probe that found nothing must say so. Before the Linux branch existed,
    # every field came back 0 on the GPU box and the checks read that as clean.
    if not mem.get("measured") and not gpus:
        add(
            "memory_unmeasured", "advisory",
            "Could not measure host or GPU memory on this machine, so no memory "
            "check above means anything. Look at `free -g` and `nvidia-smi` yourself.",
        )

    needed_gb = model_gb if writes_weights else None
    # Surgery runs on CPU, keeping model weights plus float32 working tensors
    # while updating/saving them. Budget two loaded copies plus workspace.
    host_needed = model_gb * 2 + 2 if model_gb and writes_weights else None
    if mem.get("free_gb") and host_needed and mem["free_gb"] < host_needed:
        add(
            "low_free_memory", sev,
            f"About {mem['free_gb']:.1f} GB host RAM free; CPU surgery budgets roughly {host_needed:.1f} GB for loaded weights, float32 editing and saving workspace.",
        )
    mem = {**mem, "model_gb": model_gb, "dtype": dtype, "estimated_host_needed_gb": host_needed}
    if not model_gb:
        add("model_size_unknown", "advisory", "Checkpoint size is unknown. Download its metadata first; memory and disk checks cannot verify capacity for this model.")

    disk = {}
    if writes_weights:
        import shutil

        try:
            free_gb = shutil.disk_usage(_models_root().parent).free / 2**30
        except Exception:
            free_gb = None
        # Source download/cache and the new saved checkpoint can coexist. The
        # staging directory is renamed, so publishing does not make a third copy.
        floor = (needed_gb * 2.1) if needed_gb else LOW_DISK_GB
        disk = {"free_gb": free_gb, "needed_gb": floor, "checkpoint_gb": needed_gb, "threshold_gb": LOW_DISK_GB}
        if free_gb is not None and free_gb < floor:
            # Not acknowledgeable: running out part-way through save_pretrained
            # leaves a corrupt directory, and there is no legitimate override.
            add(
                "insufficient_disk", "blocking",
                f"Only {free_gb:.1f} GB free but about {floor:.1f} GB is needed. "
                f"A write that runs out of space leaves a corrupt model directory.",
                acknowledgeable=False,
            )

    # Direction-quality gates.
    summary = _direction_summary(direction_row)
    if direction_row is not None and summary:
        auc = summary.get("auc")
        min_auc = summary.get("min_usable_auc", 0.90)
        if auc is not None and auc < min_auc:
            add(
                "low_auc", "blocking" if writes_weights else "advisory",
                f"The direction's held-out AUC is {auc:.3f}, below the {min_auc:.2f} "
                f"usability threshold. It does not separate the classes, so the edit "
                f"will not produce a reliable behaviour change.",
            )

        derived_from = summary.get("model_id")
        if derived_from and source_model and derived_from != source_model:
            # apply_to_model only raises on a hidden-size mismatch, so a direction
            # from another model of the same width writes 6 GB of garbage silently.
            add(
                "model_mismatch", "blocking" if writes_weights else "advisory",
                f"This direction was derived from '{derived_from}' but you are editing "
                f"'{source_model}'. Only the hidden size is checked, so a mismatch of "
                f"the same width completes and produces garbage without erroring.",
            )

    # A search over subspace ranks needs a subspace to search.
    if kind == "select" and direction_row is not None:
        if int(summary.get("rank") or 1) < 2:
            add(
                "needs_subspace", "blocking",
                "This direction holds a single vector, so there is no subspace to "
                "search. Derive one with a subspace rank above 1.",
                acknowledgeable=False,
            )

    if writes_weights and direction_row is not None:
        if not _has_completed_sweep(direction_row.id):
            add(
                "no_sweep_evidence", "advisory",
                "No completed steering sweep has used this direction, so its causality "
                "is unproven. A direction that does not move behaviour under steering "
                "will not move it as a weight edit either.",
            )

    if writes_weights and output_name:
        candidate = _resolve_output_dir(output_name)
        if candidate.exists():
            add(
                "output_exists", "blocking",
                f"'{candidate.name}' already exists. Surgery refuses to write into a "
                f"non-empty directory; choose another name.",
                acknowledgeable=False,
            )
        else:
            from vivasecuris.aiasylum.api.model_history import saved_checkpoint_history

            previous = saved_checkpoint_history(candidate)
            if previous:
                add(
                    "output_name_reused", "blocking",
                    f"'{candidate.name}' was already saved by {previous}. Its original checkpoint directory "
                    "is not present, but this name is reserved to keep the model history distinct. "
                    "Choose a fresh output name.",
                    acknowledgeable=False,
                )

    blocking = [c.code for c in checks if c.severity == "blocking"]
    return PreflightResponse(
        kind=kind,
        checks=checks,
        can_proceed=not blocking,
        blocking_codes=blocking,
        memory=mem,
        disk=disk,
    )


def _has_completed_sweep(direction_run_id: int) -> bool:
    session = get_session()
    try:
        return (
            session.query(WeightRun)
            .filter(
                WeightRun.kind == "sweep",
                WeightRun.source_run_id == direction_run_id,
                WeightRun.status == STATUS_COMPLETED,
            )
            .count()
            > 0
        )
    finally:
        session.close()


# --------------------------------------------------------------------------
# Read endpoints
# --------------------------------------------------------------------------


@router.get("/stages")
async def list_stages():
    """Everything the launcher needs in one fetch: stages, methods, objectives."""
    methods = [{"name": k, **v} for k, v in METHODS.items()]
    methods += [
        {
            "name": k,
            "label": label,
            "description": reason,
            "stage": None,
            "permanent": None,
            "available": False,
            "unavailable_reason": reason,
            "needs": [],
        }
        for k, (label, reason) in UNAVAILABLE_METHODS.items()
    ]

    return {
        "stages": [{"name": k, **v} for k, v in STAGES.items()],
        "methods": methods,
        "objectives": [{"name": k, **v} for k, v in OBJECTIVES.items()],
        "beta_presets": BETA_PRESETS,
        "defaults": {
            "source_model": "Qwen/Qwen2.5-3B-Instruct",
            "n_per_class": 128,
            "n_prompts": 8,
            "alphas": [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0],
            "dtype": "bfloat16",
            "device": "auto",
        },
        "min_usable_auc": 0.90,
        "interp_extra_installed": _interp_extra_installed(),
        "device": _device_info(),
        "models_root": str(_models_root()),
        "slot": slot_status(),
    }


@router.get("/objectives")
async def list_objectives():
    """Objectives, with live prompt counts so the form can warn before a run.

    `build_split` rejects anything under 8 usable prompts per class. Counting
    here turns that into a disabled checkbox rather than a failure minutes in.
    """
    from vivasecuris.aiasylum.database.models import PromptLibrary

    session = get_session()
    try:
        rows = (
            session.query(PromptLibrary.prompt_text, PromptLibrary.meta_data)
            .filter(PromptLibrary.category == "forbidden_question")
            .all()
        )
    finally:
        session.close()

    counts: Dict[str, int] = {}
    for text, meta in rows:
        if not text or not text.strip():
            continue
        scenario = (meta or {}).get("content_policy_name") or "(unlabelled)"
        counts[scenario] = counts.get(scenario, 0) + 1

    from vivasecuris.aiasylum.weights.corpus import (
        ADVICE_SCENARIOS,
        FALLBACK_HARMLESS,
        HARMFUL_SCENARIOS,
    )

    scenarios = [
        {
            "name": name,
            "count": counts.get(name, 0),
            "in_default_set": name in HARMFUL_SCENARIOS,
            "excluded_reason": (
                "Restricted advice: models answer these rather than refusing, so "
                "including them dilutes the harmful class."
                if name in ADVICE_SCENARIOS
                else None
            ),
        }
        for name in sorted(counts)
    ]

    return {
        "objectives": [{"name": k, **v} for k, v in OBJECTIVES.items()],
        "scenarios": scenarios,
        "default_scenarios": list(HARMFUL_SCENARIOS),
        "harmless_builtin_count": len(FALLBACK_HARMLESS),
        "min_per_class": 8,
    }


@router.get("/preflight", response_model=PreflightResponse)
async def preflight(
    kind: str = Query("direction"),
    source_model: str = Query(""),
    source_run_id: Optional[int] = Query(None),
    output_name: Optional[str] = Query(None),
    modified_model: Optional[str] = Query(None),
    dtype: str = Query("bfloat16"),
    expert_selection: Optional[str] = Query(None, description="JSON object: layer -> experts | 'all'"),
    merge: bool = Query(True),
):
    if kind not in STAGES:
        raise HTTPException(status_code=400, detail=f"Unknown stage '{kind}'.")

    selection: Optional[Dict[str, Any]] = None
    if expert_selection:
        try:
            selection = json.loads(expert_selection)
        except ValueError:
            raise HTTPException(status_code=400, detail="expert_selection must be a JSON object.")
        if not isinstance(selection, dict):
            raise HTTPException(status_code=400, detail="expert_selection must be a JSON object.")

    direction_row = None
    if source_run_id is not None:
        session = get_session()
        try:
            direction_row = (
                session.query(WeightRun).filter(WeightRun.id == source_run_id).first()
            )
        finally:
            session.close()

    return _preflight_checks(
        kind, source_model, direction_row, output_name, modified_model, dtype,
        expert_selection=selection, training={"merge": merge},
    )


@router.get("/runs", response_model=List[WeightRunResponse])
async def list_weight_runs(
    limit: int = 50,
    offset: int = 0,
    kind: Optional[str] = None,
    status: Optional[str] = None,
):
    session = get_session()
    try:
        query = session.query(WeightRun)
        if kind:
            query = query.filter(WeightRun.kind == kind)
        if status:
            query = query.filter(WeightRun.status == status)
        rows = query.order_by(WeightRun.id.desc()).offset(offset).limit(limit).all()
        return [WeightRunResponse.from_orm_row(r) for r in rows]
    finally:
        session.close()


@router.get("/directions")
async def list_directions(usable_only: bool = False):
    """Completed directions, for the sweep and surgery pickers.

    `artifacts_present` is checked against the filesystem rather than assumed
    from the status: a row outlives its files whenever a run is deleted or
    `runs/` is cleaned, and catching that here turns a job that would die in a
    worker thread into an instantly explicable rejection.
    """
    session = get_session()
    try:
        rows = (
            session.query(WeightRun)
            .filter(WeightRun.kind == "direction", WeightRun.status == STATUS_COMPLETED)
            .order_by(WeightRun.id.desc())
            .all()
        )
        sweeps = {
            r.source_run_id
            for r in session.query(WeightRun)
            .filter(WeightRun.kind == "sweep", WeightRun.status == STATUS_COMPLETED)
            .all()
        }
    finally:
        session.close()

    out = []
    for row in rows:
        summary = _direction_summary(row)
        present = bool(row.out_dir) and (Path(row.out_dir) / "direction.safetensors").exists()
        if usable_only and not summary.get("usable"):
            continue
        out.append(
            {
                "id": row.id,
                "created_at": row.created_at,
                "source_model": row.source_model,
                "objective": row.objective,
                "layer": summary.get("layer"),
                "auc": summary.get("auc"),
                "cohens_d": summary.get("cohens_d"),
                "d_model": summary.get("d_model"),
                "split_hash": summary.get("split_hash"),
                "usable": summary.get("usable"),
                "method": summary.get("method", "diff_in_means"),
                "rank": summary.get("rank", 1),
                "stable_rank": ((summary.get("extra") or {}).get("stable_rank") or {}).get("at_layer"),
                "artifacts_present": present,
                "has_sweep": row.id in sweeps,
            }
        )
    return out


@router.get("/probes")
async def list_probes(usable_only: bool = False):
    """Completed probe runs, for the monitor picker."""
    session = get_session()
    try:
        rows = (
            session.query(WeightRun)
            .filter(WeightRun.kind == "probe", WeightRun.status == STATUS_COMPLETED)
            .order_by(WeightRun.id.desc())
            .all()
        )
    finally:
        session.close()

    out = []
    for row in rows:
        summary = _direction_summary(row)
        present = bool(row.out_dir) and (Path(row.out_dir) / "probes.npz").exists()
        if usable_only and not summary.get("usable"):
            continue
        out.append({
            "id": row.id,
            "created_at": row.created_at,
            "source_model": row.source_model,
            "best_layer": summary.get("best_layer"),
            "auroc": summary.get("best_auroc"),
            "null_auroc_p95": summary.get("null_auroc_p95"),
            "beats_null": summary.get("beats_null"),
            "ece": summary.get("best_ece"),
            "pooling": summary.get("pooling"),
            "prompt_suffix": summary.get("prompt_suffix"),
            "group_auroc": summary.get("group_auroc", {}),
            "usable": summary.get("usable"),
            "artifacts_present": present,
        })
    return out


@router.get("/models")
async def list_edited_models():
    """Models carrying a surgery manifest.

    The manifest on disk is the source of truth and the database row is an
    optional enrichment, not the other way round: `models/ablated/` already
    exists in this repo from a CLI run and will never have a row. Listing only
    what the API produced would hide models that are being tested against.
    """
    root = _models_root()
    if not root.is_dir():
        return []

    session = get_session()
    try:
        rows = {
            r.out_dir: r
            for r in session.query(WeightRun).filter(WeightRun.kind.in_(WEIGHT_KINDS_WRITING_MODELS)).all()
            if r.out_dir
        }
    finally:
        session.close()

    out = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        manifest_path = child / MANIFEST_NAME
        if not manifest_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception:
            manifest = {}
        row = rows.get(str(child)) or rows.get(str(child.resolve()))
        out.append(
            {
                "name": child.name,
                "path": str(child),
                "manifest": manifest,
                "run_id": row.id if row else None,
                "orphan": row is None,
                "size_bytes": _dir_size(child),
                "created_at": manifest.get("created_at"),
            }
        )
    return out


@router.get("/models/{name}")
async def get_edited_model(name: str):
    """One edited model, with the provenance trail behind it."""
    _validate_slug(name)
    path = _resolve_output_dir(name)
    manifest_path = path / MANIFEST_NAME
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail=f"No surgery manifest at {path}.")

    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unreadable manifest: {exc}")

    session = get_session()
    try:
        surgery = (
            session.query(WeightRun)
            .filter(WeightRun.kind.in_(WEIGHT_KINDS_WRITING_MODELS), WeightRun.out_dir == str(path))
            .first()
        )
        direction = None
        sweeps: List[WeightRun] = []
        if surgery is not None and surgery.source_run_id:
            direction = (
                session.query(WeightRun)
                .filter(WeightRun.id == surgery.source_run_id)
                .first()
            )
            sweeps = (
                session.query(WeightRun)
                .filter(
                    WeightRun.kind == "sweep",
                    WeightRun.source_run_id == surgery.source_run_id,
                )
                .all()
            )

        # TestRun stores models as (provider, name) string pairs. patient_model
        # is String(100), so a long path can be truncated on the way in -- treat
        # an empty result as "nothing recorded", not as proof it was never used.
        test_runs = (
            session.query(TestRun)
            .filter(
                (TestRun.patient_model == str(path)) | (TestRun.doctor_model == str(path))
            )
            .order_by(TestRun.id.desc())
            .limit(50)
            .all()
        )
        test_rows = [
            {
                "id": t.id,
                "test_type": t.test_type,
                "status": t.status,
                "created_at": t.created_at,
                "role": "patient" if t.patient_model == str(path) else "doctor",
            }
            for t in test_runs
        ]

        interp_runs = (
            session.query(InterpRun)
            .filter((InterpRun.model_a == str(path)) | (InterpRun.model_b == str(path)))
            .order_by(InterpRun.id.desc())
            .limit(50)
            .all()
        )
        interp_rows = [
            {"id": i.id, "mode": i.mode, "status": i.status, "created_at": i.created_at}
            for i in interp_runs
        ]
    finally:
        session.close()

    return {
        "name": path.name,
        "path": str(path),
        "manifest": manifest,
        "size_bytes": _dir_size(path),
        "run": WeightRunResponse.from_orm_row(surgery) if surgery else None,
        "direction_run": WeightRunResponse.from_orm_row(direction) if direction else None,
        "sweep_runs": [WeightRunResponse.from_orm_row(s) for s in sweeps],
        "test_runs": test_rows,
        "interp_runs": interp_rows,
        "path_truncation_caveat": len(str(path)) > 100,
    }


@router.get("/runs/{run_id}", response_model=WeightRunResponse)
async def get_weight_run(run_id: int):
    session = get_session()
    try:
        row = session.query(WeightRun).filter(WeightRun.id == run_id).first()
        if row is None:
            raise HTTPException(status_code=404, detail="Weight run not found")
        return WeightRunResponse.from_orm_row(row)
    finally:
        session.close()


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------


class RunCancelled(Exception):
    """Raised inside the worker thread to unwind a cancelled run.

    `asyncio.to_thread` is not interruptible: cancelling the task raises in the
    awaiting coroutine while the thread runs to completion. Cancellation has to
    be cooperative, and the progress callbacks are the only place the engine
    yields control often enough to check.
    """


def _build_objective_split(
    objective: str,
    config: Optional[Dict[str, Any]],
    n_per_class: int,
    test_fraction: float,
    seed: int,
):
    """Turn an objective into the contrast pair the direction is fit against."""
    from vivasecuris.aiasylum.weights.corpus import (
        HARMFUL_SCENARIOS,
        build_split,
        load_harmful_prompts,
        load_harmless_prompts,
    )

    cfg = config or {}
    kw = dict(n_per_class=n_per_class, test_fraction=test_fraction, seed=seed)

    if objective == "refusal":
        return build_split(**kw)

    if objective == "refusal_narrow":
        scenarios = cfg.get("scenarios") or list(HARMFUL_SCENARIOS)
        return build_split(harmful=load_harmful_prompts(scenarios=scenarios), **kw)

    if objective == "custom":
        pos, neg = cfg.get("positive") or {}, cfg.get("negative") or {}
        # Fall back to HARMFUL_SCENARIOS, never to None. `load_harmful_prompts`
        # reads a falsy `scenarios` as "every scenario", which pulls in the five
        # restricted-advice ones that models answer rather than refuse -- the
        # set that moved baseline refusal from 92% to 33% and made the direction
        # underivable. A caller who wants all of them has to say so explicitly.
        scenarios = pos.get("scenarios")
        if not scenarios:
            scenarios = (
                None if pos.get("all_scenarios") else list(HARMFUL_SCENARIOS)
            )
        harmful = pos.get("prompts") or load_harmful_prompts(
            category=pos.get("category", "forbidden_question"),
            scenarios=scenarios,
        )
        harmless = neg.get("prompts") or load_harmless_prompts(
            benchmark=neg.get("benchmark")
        )
        return build_split(harmful=harmful, harmless=harmless, **kw)

    if objective == "over_refusal":
        refused = list(cfg.get("refused") or [])
        answered = list(cfg.get("answered") or [])
        if len(refused) < 8 or len(answered) < 8:
            raise ValueError(
                "over_refusal needs at least 8 'refused' and 8 'answered' benign prompts "
                "in objective_config; take them from a test run's analysis."
            )
        split = build_split(harmful=refused, harmless=answered, **kw)
        split.source = f"over_refusal: refused(n={len(refused)}) vs answered(n={len(answered)})"
        return split

    raise ValueError(f"Unknown objective '{objective}'")


def _refusal_overlap(model, tok, direction, opts: Dict[str, Any], report) -> Dict[str, Any]:
    """Cosine between a derived direction and the global refusal direction at its layer.

    Over-refusal that is just refusal in disguise shows up as a cosine near
    one; a genuinely task-dependent over-refusal direction does not.
    """
    import torch

    from vivasecuris.aiasylum.weights.direction import derive_direction

    split = _build_objective_split("refusal", None, opts["n_per_class"], opts["test_fraction"], opts["seed"])
    refusal = derive_direction(
        model, tok, split, model_id="overlap-check", batch_size=opts["batch_size"],
        max_length=opts["max_length"], layer_range=(direction.layer, direction.layer + 1), progress=report,
    )
    cos = float(torch.dot(direction.vector.float(), refusal.vector.float()).abs().item())
    return {
        "cosine_with_refusal": cos,
        "refusal_auc_at_layer": refusal.auc,
        "interpretation": (
            "indistinguishable from the global refusal direction" if cos > 0.9
            else "partly shares the refusal direction" if cos > 0.5
            else "separable from the global refusal direction"
        ),
    }


def _compare_extras(model, tok, label: str, snap: Dict[str, Any], opts: Dict[str, Any],
                    reporter, harmful: List[str]) -> Dict[str, Any]:
    """Optional compare controls: re-derive the direction, broad misalignment."""
    out: Dict[str, Any] = {}
    if opts.get("rederive") or snap.get("method") == "compare_rederive":
        from vivasecuris.aiasylum.weights.direction import derive_direction
        from vivasecuris.aiasylum.weights.steering import refusal_rate, steer

        reporter.note(f"{label}: re-deriving the refusal direction")
        split = _evaluation_split(snap)
        d = derive_direction(
            model, tok, split, model_id=label, batch_size=opts.get("batch_size", 8),
            max_length=opts.get("max_length", 512), progress=reporter.as_callback(),
        )
        reporter.note(f"{label}: ablating the re-derived direction")
        from vivasecuris.aiasylum.weights.steering import generate_with_steering

        with steer(model, d.vector, mode="ablate"):
            abl = [generate_with_steering(model, tok, p, max_new_tokens=opts["max_new_tokens"]) for p in harmful]
        out["rederived"] = {
            "layer": d.layer, "auc": d.auc, "cohens_d": d.cohens_d, "usable": d.usable,
            "stable_rank": (d.extra.get("stable_rank") or {}).get("at_layer"),
            "stable_rank_band": (d.extra.get("stable_rank") or {}).get("band"),
            "ablate_refuse_harmful": refusal_rate(abl),
        }
    if opts.get("misalignment_control"):
        from vivasecuris.aiasylum.weights.evaluate import generate_greedy
        from vivasecuris.aiasylum.weights.misalignment import MISALIGNMENT_PROBES, misalignment_rate

        reporter.note(f"{label}: running the broad-misalignment control")
        resp = generate_greedy(model, tok, list(MISALIGNMENT_PROBES), max_new_tokens=96)
        result = misalignment_rate(resp, list(MISALIGNMENT_PROBES))
        out["misalignment"] = {**result, "responses": resp}
        out["misalignment_rate"] = result["rate"]
    return out


def _evaluation_split(snap: Dict[str, Any]):
    """Use the parent's immutable partition even if the library has changed."""
    from vivasecuris.aiasylum.weights.corpus import PromptSplit

    source = snap.get("source_direction") or {}
    if source.get("prompt_split"):
        return PromptSplit.from_dict(source["prompt_split"])
    opts = snap.get("source_options") or snap.get("options") or {}
    split = _build_objective_split(
        snap.get("objective") or "refusal",
        snap.get("objective_config"),
        n_per_class=opts.get("n_per_class", 128),
        test_fraction=opts.get("test_fraction", 0.25),
        seed=opts.get("seed", 0),
    )
    # Legacy directions can only be reconstructed if the exact digest agrees.
    # A changed corpus must not silently turn training prompts into test data.
    if source and source.get("split_hash") != split.hash:
        raise ValueError("Cannot reconstruct this direction's exact prompt split. Derive a new direction before evaluation.")
    return split


def _held_out_prompts(snap: Dict[str, Any], limit: int) -> List[str]:
    split = _evaluation_split(snap)
    return list(split.harmful_test[:limit])


def _evaluation_evidence(snap: Dict[str, Any], limit: int) -> Dict[str, Any]:
    split = _evaluation_split(snap)
    actual = min(limit, len(split.harmful_test))
    return {"split_hash": split.hash, "requested": limit, "actual": actual,
            "warnings": ([f"Only {actual} held-out prompts are available; requested {limit}. No training prompts were reused."]
                         if actual < limit else [])}


def _make_reporter(run_id: int, loop: asyncio.AbstractEventLoop):
    """A `Reporter` that emits SSE events instead of writing to stderr.

    Both `_write` and `count` are overridden, not just `_write`: `Reporter.count`
    writes to stderr directly rather than going through `_write`, so overriding
    only the latter would capture the three step announcements from
    `edit_and_save` and nothing at all from a 128-prompt capture or a ten-minute
    sweep -- where the per-item counter is the only progress produced.

    Emissions are throttled, because `count` fires once per prompt per alpha.
    The terminal tick always gets through so a stage never appears to stall at
    99%.
    """
    import time

    from vivasecuris.aiasylum.weights.progress import Reporter

    class _EventReporter(Reporter):
        def __init__(self):
            super().__init__(enabled=True)
            self._last_emit = 0.0

        def _emit(self, message: str, data: Optional[Dict[str, Any]] = None) -> None:
            if weights_cancellation.is_cancelled(run_id):
                raise RunCancelled()
            asyncio.run_coroutine_threadsafe(
                weights_progress.emit_event(
                    run_id, "weights_progress", data or {}, message
                ),
                loop,
            )

        def _write(self, text: str, newline: bool = True) -> None:
            self._emit(text.strip())

        def count(self, done: int, total: int, label: str = "") -> None:
            if not total:
                return
            now = time.time()
            final = done >= total
            if not final and now - self._last_emit < 0.5:
                # Still check for cancellation on a throttled tick: this is the
                # only place a long sweep yields often enough to stop promptly.
                if weights_cancellation.is_cancelled(run_id):
                    raise RunCancelled()
                return
            self._last_emit = now
            pct = 100.0 * done / total
            self._emit(
                f"{label}{done}/{total} ({pct:.0f}%)",
                {"done": done, "total": total, "percent": round(pct, 1)},
            )

        def metrics(self, data: Dict[str, Any]) -> None:
            """A training tick: step, loss, learning rate, or an evaluation loss.

            Throttled like `count`, since the worker emits one per optimizer
            step; evaluations and the final step always get through.
            """
            now = time.time()
            step, total = data.get("step"), data.get("total")
            is_eval = data.get("phase") == "eval"
            final = bool(total) and step == total
            if not is_eval and not final and now - self._last_emit < 0.5:
                if weights_cancellation.is_cancelled(run_id):
                    raise RunCancelled()
                return
            self._last_emit = now
            if is_eval:
                loss = data.get("eval_loss")
                message = f"eval at step {step}: loss {loss:.4f}" if loss is not None else f"eval at step {step}"
                pct = None
            else:
                loss = data.get("loss")
                pct = round(100.0 * step / total, 1) if total else None
                message = f"step {step}/{total}" + (f" loss {loss:.4f}" if loss is not None else "")
            self._emit(message, {
                "phase": data.get("phase", "train"), "step": step, "total": total,
                "percent": pct, "loss": loss if not is_eval else None,
                "lr": data.get("lr"), "eval_loss": data.get("eval_loss"),
                "done": step, "tokens": data.get("tokens"),
            })

    return _EventReporter()


def _write_model_output(run_id: int, out_dir: Path, reporter, writer, snap: Dict[str, Any]) -> Dict[str, Any]:
    """Run ``writer(staging)``, publish by rename, and enrich the manifest.

    Every kind that produces a model directory goes through here. Writing to a
    staging sibling means a crash or a cancel never leaves a half-written
    directory at a name the models list would show, and retrying the same name
    still works.
    """
    import shutil
    from dataclasses import asdict

    from vivasecuris.aiasylum.weights.manifest import SurgeryManifest

    staging = out_dir.parent / f"{STAGING_PREFIX}{run_id}"
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    try:
        writer(staging)
        staging.rename(out_dir)
    except BaseException:
        if staging.name.startswith(STAGING_PREFIX) and _is_inside(staging, _models_root()):
            shutil.rmtree(staging, ignore_errors=True)
        raise

    # Record what this edit was and what it was aimed at, so `weights info`
    # and the provenance trail say more than the value of beta.
    manifest = SurgeryManifest.load(out_dir)
    if manifest is not None:
        manifest.extra = {
            **(manifest.extra or {}),
            "method": snap["method"],
            "objective": snap["objective"],
            "direction_run_id": snap.get("source_run_id"),
            "objective_config": snap.get("objective_config"),
            "source_options": snap.get("source_options") or {},
            "prompt_split": (snap.get("source_direction") or {}).get("prompt_split"),
        }
        manifest.save(out_dir)

    return {
        "manifest": asdict(manifest) if manifest is not None else {},
        "output_path": str(out_dir),
        "size_bytes": _dir_size(out_dir),
        "elapsed": reporter.total_elapsed(),
    }


def _execute(run_id: int, snap: Dict[str, Any], reporter) -> Dict[str, Any]:
    """The synchronous, model-bound body. Always called in a worker thread."""
    from vivasecuris.aiasylum.interp.core.loader import load
    from vivasecuris.aiasylum.models.transformers_local import clear_cache

    # That cache holds whole models keyed by (path, device, dtype) and is never
    # evicted, so after one test run the API process may already be holding
    # several GB of the unified memory this job is about to need.
    clear_cache()

    kind = snap["kind"]
    opts = snap["options"]
    # `compare` produces only numbers, so it is the one stage with nothing to
    # write; the others all address a directory.
    out_dir = Path(snap["out_dir"]) if snap.get("out_dir") else None
    report = reporter.as_callback()

    if kind == "direction":
        from vivasecuris.aiasylum.weights.direction import derive_direction, derive_subspace

        reporter.note(f"building prompt split ({snap['objective']})")
        split = _build_objective_split(
            snap["objective"], snap.get("objective_config"),
            opts["n_per_class"], opts["test_fraction"], opts["seed"],
        )
        with reporter.step(f"loading {snap['source_model']}"):
            model, tok = load(snap["source_model"], device=opts["device"], dtype=opts["dtype"])

        rank = int(opts.get("subspace_rank") or 1)
        method = snap.get("method") or "diff_in_means"
        if method == "rfm_agop":
            from vivasecuris.aiasylum.weights.rfm import derive_rfm_subspace

            rfm_rank = int(opts.get("rfm_rank") or max(rank, 4))
            reporter.note(f"deriving a rank-{rfm_rank} refusal cone with RFM-AGOP")
            direction = derive_rfm_subspace(
                model, tok, split,
                rank=rfm_rank,
                iterations=int(opts.get("rfm_iterations") or 5),
                beta=float(opts.get("rfm_beta") if opts.get("rfm_beta") is not None else 0.5),
                model_id=snap["source_model"],
                batch_size=opts["batch_size"],
                max_length=opts["max_length"],
                progress=report,
            )
        elif rank > 1:
            # A subspace catches refusal components a single difference-in-means
            # vector misses; row 0 of the basis is still that vector, so
            # everything downstream that reads `vector` is unaffected.
            reporter.note(f"deriving a rank-{rank} subspace")
            direction = derive_subspace(
                model, tok, split,
                rank=rank,
                pool_layers=int(opts.get("pool_layers") or 12),
                model_id=snap["source_model"],
                batch_size=opts["batch_size"],
                max_length=opts["max_length"],
                progress=report,
            )
        else:
            direction = derive_direction(
                model, tok, split,
                model_id=snap["source_model"],
                batch_size=opts["batch_size"],
                max_length=opts["max_length"],
                progress=report,
            )
        out_dir.mkdir(parents=True, exist_ok=True)
        direction.save(out_dir)

        summary = direction.metadata()
        summary["split"] = split.summary()
        summary["prompt_split"] = split.to_dict()
        (out_dir / "prompt_split.json").write_text(json.dumps(split.to_dict(), indent=2))
        if snap.get("objective") == "over_refusal" and opts.get("report_overlap", True):
            reporter.note("checking overlap with the global refusal direction")
            summary["refusal_overlap"] = _refusal_overlap(model, tok, direction, opts, report)
        # Snapshotted for the interp handoff, so the model_diff button does not
        # have to re-derive a prompt the direction was actually fit against.
        summary["probe_prompts"] = list(split.harmful_test[:3])
        summary["elapsed"] = reporter.total_elapsed()
        return summary

    if kind == "sweep":
        from vivasecuris.aiasylum.weights.direction import RefusalDirection
        from vivasecuris.aiasylum.weights.steering import summarize_sweep, sweep_alpha

        d = RefusalDirection.load(snap["direction_dir"])
        with reporter.step(f"loading {snap['source_model']}"):
            model, tok = load(snap["source_model"], device=opts["device"], dtype=opts["dtype"])

        # Check the direction against the contrast it was derived for. Sweeping
        # a narrowed or custom direction against the generic refusal corpus
        # measures the wrong thing, and the causality verdict -- plus the
        # no_sweep_evidence gate built on it -- would be meaningless.
        prompts = _held_out_prompts(snap, opts["n_prompts"])
        reporter.note(
            f"sweeping over {len(prompts)} held-out '{snap['objective']}' prompts "
            f"at layer {d.layer}"
        )

        thinking = bool(opts.get("thinking", False))
        if (snap.get("method") or "steering_sweep") == "subspace_curve":
            from vivasecuris.aiasylum.weights.steering import summarize_curve, sweep_subspace_rank

            reporter.note(f"rank curve over {d.rank} directions ({d.method})")
            curve = sweep_subspace_rank(
                model, tok, d, prompts,
                ks=tuple(opts.get("ks") or (1.0,)),
                max_new_tokens=opts["max_new_tokens"],
                thinking=thinking,
                capability_control=opts.get("capability_control", True),
                capability_limit=opts.get("capability_limit", 6),
                progress=report,
            )
            summary = {"rows": [], "curve": curve, "layer": d.layer, "rank": d.rank,
                       "weights": d.weights, "method": d.method, **summarize_curve(curve)}
        else:
            rows = sweep_alpha(
                model, tok, d.vector, prompts,
                layer=d.layer,
                max_new_tokens=opts["max_new_tokens"],
                include_ablation=opts["include_ablation"],
                progress=report,
                capability_control=opts.get("capability_control", True),
                capability_limit=opts.get("capability_limit", 6),
                thinking=thinking,
                **({"alphas": opts["alphas"]} if opts.get("alphas") else {}),
            )
            summary = {"rows": rows, "layer": d.layer, **summarize_sweep(rows)}

        n_timeline = int(opts.get("timeline_prompts") or 0)
        if n_timeline > 0:
            from vivasecuris.aiasylum.weights.timeline import refusal_timeline

            reporter.note(f"recording refusal-decision timelines for {n_timeline} prompts")
            summary["timelines"] = [
                refusal_timeline(model, tok, p, d, max_new_tokens=opts["max_new_tokens"], thinking=thinking)
                for p in prompts[:n_timeline]
            ]
        summary["thinking"] = thinking
        summary["evaluation"] = _evaluation_evidence(snap, opts["n_prompts"])
        summary["elapsed"] = reporter.total_elapsed()
        return summary

    if kind == "select":
        from vivasecuris.aiasylum.weights.direction import RefusalDirection
        from vivasecuris.aiasylum.weights.surgery import select_edit

        d = RefusalDirection.load(snap["direction_dir"])
        if d.basis is None:
            raise ValueError(
                "This direction holds only a single vector; there is no subspace to "
                "search. Derive one with a subspace rank above 1."
            )

        with reporter.step(f"loading {snap['source_model']}"):
            model, tok = load(snap["source_model"], device=opts["device"], dtype=opts["dtype"])

        prompts = _held_out_prompts(snap, opts["n_prompts"])
        ranks = tuple(opts.get("ranks") or (1, 2, 3, 4, 6, 8))
        ks = tuple(opts.get("ks") or (1.0, 1.25, 1.5))
        reporter.note(
            f"scoring {len(ranks) * len(ks)} candidates over {len(prompts)} held-out "
            f"prompts, each gated on the factual capability control"
        )

        from vivasecuris.aiasylum.weights.evaluate import capability_set

        capability = capability_set(opts.get("capability_set") or "builtin", opts["seed"])
        reporter.note(f"capability control: {capability.name} ({capability.size} items)")
        result = select_edit(
            model, tok, d, prompts,
            ranks=ranks, ks=ks,
            factual_floor=opts.get("factual_floor", 0.05),
            max_new_tokens=opts["max_new_tokens"],
            progress=report,
            capability=capability,
        )
        summary = dict(result)
        summary["ranks"] = list(ranks)
        summary["ks"] = list(ks)
        summary["factual_floor"] = opts.get("factual_floor", 0.05)
        summary["source_rank"] = d.rank
        summary["evaluation"] = _evaluation_evidence(snap, opts["n_prompts"])
        summary["elapsed"] = reporter.total_elapsed()
        return summary

    if kind == "probe":
        from vivasecuris.aiasylum.interp.probes.dataset import (
            ELICITING_SUFFIX, build_harmful_intent_dataset,
        )
        from vivasecuris.aiasylum.interp.probes.train import train_probes
        from vivasecuris.aiasylum.weights.capture import capture_pooled_residuals

        reporter.note("building the harmful-intent dataset")
        ds = build_harmful_intent_dataset(
            n_direct=int(opts.get("n_direct", 120)),
            n_jailbreak=int(opts.get("n_jailbreak", 120)),
            n_benign=int(opts.get("n_benign", 240)),
            holdout_techniques=int(opts.get("holdout_techniques") or 0),
            seed=opts.get("seed", 0),
            jailbreak_examples=opts.get("jailbreak_examples"),
        )
        suffix = opts.get("prompt_suffix")
        if not suffix and opts.get("use_eliciting_suffix"):
            suffix = ELICITING_SUFFIX

        with reporter.step(f"loading {snap['source_model']}"):
            model, tok = load(snap["source_model"], device=opts["device"], dtype=opts["dtype"])

        pooling = opts.get("pooling") or "mean"
        reporter.note(f"capturing {len(ds.train_prompts)} train prompts ({pooling} pooling)")
        train_acts = capture_pooled_residuals(
            model, tok, ds.train_prompts, pooling=pooling, prompt_suffix=suffix,
            batch_size=opts["batch_size"], max_length=opts["max_length"],
            progress=(lambda d, t: report(None, d, t)),
        )
        reporter.note(f"capturing {len(ds.test_prompts)} held-out prompts")
        test_acts = capture_pooled_residuals(
            model, tok, ds.test_prompts, pooling=pooling, prompt_suffix=suffix,
            batch_size=opts["batch_size"], max_length=opts["max_length"],
            progress=(lambda d, t: report(None, d, t)),
        )
        reporter.note("fitting one probe per layer, each against a shuffled-label null")
        ps = train_probes(
            train_acts, ds.train_labels, test_acts, ds.test_labels,
            test_groups=ds.test_groups, model_id=snap["source_model"], pooling=pooling,
            dataset_hash=ds.hash, prompt_suffix=suffix, dataset_summary=ds.summary(),
            seed=opts.get("seed", 0), progress=report,
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        ps.save(out_dir)
        from dataclasses import asdict
        (out_dir / "dataset.json").write_text(json.dumps(asdict(ds), indent=2))
        summary = ps.metadata()
        summary["elapsed"] = reporter.total_elapsed()
        return summary

    if kind == "compare":
        import gc

        from vivasecuris.aiasylum.weights.evaluate import (
            capability_set, compare_verdict, generate_greedy,
        )
        from vivasecuris.aiasylum.weights.manifest import SurgeryManifest
        from vivasecuris.aiasylum.weights.steering import _looks_degenerate, refusal_rate

        split = _evaluation_split(snap)
        harmful = list(split.harmful_test[: opts["n_prompts"]])
        harmless = list(split.harmless_test[: opts["n_prompts"]])
        capability = capability_set(opts.get("capability_set") or "builtin", opts["seed"])
        factual_qs = capability.questions
        reporter.note(f"capability control: {capability.name} ({capability.size} items)")

        metrics: Dict[str, Any] = {}
        # One model resident at a time: two 3B copies would not fit beside each
        # other, and the point of the comparison is that only the weights differ.
        for label, model_id in (
            ("baseline", snap["source_model"]),
            ("modified", snap["modified_model"]),
        ):
            with reporter.step(f"loading {label}: {model_id}"):
                model, tok = load(model_id, device=opts["device"], dtype=opts["dtype"])

            reporter.note(f"{label}: scoring refusal on harmful prompts")
            harm = generate_greedy(model, tok, harmful, max_new_tokens=opts["max_new_tokens"])
            reporter.note(f"{label}: scoring false refusal on harmless prompts")
            harmless_r = generate_greedy(model, tok, harmless, max_new_tokens=opts["max_new_tokens"])
            reporter.note(f"{label}: running the factual capability control")
            fac = generate_greedy(model, tok, factual_qs, max_new_tokens=capability.max_new_tokens)

            metrics[label] = {
                "model": model_id,
                "refuse_harmful": refusal_rate(harm),
                "refuse_harmless": refusal_rate(harmless_r),
                "factual_acc": capability.score(fac),
                "degenerate": bool(_looks_degenerate(harm) or _looks_degenerate(fac)),
                "responses": {"harmful": harm, "harmless": harmless_r, "factual": fac},
            }
            metrics[label].update(_compare_extras(model, tok, label, snap, opts, reporter, harmful))
            del model, tok
            gc.collect()

        b, m = metrics["baseline"], metrics["modified"]
        manifest = SurgeryManifest.load(snap["modified_model"])
        return {
            "metrics": {k: {kk: vv for kk, vv in v.items() if kk != "responses"}
                        for k, v in metrics.items()},
            "responses": {k: v["responses"] for k, v in metrics.items()},
            "prompts": {"harmful": harmful, "harmless": harmless, "factual": factual_qs},
            "deltas": {
                "refuse_harmful": m["refuse_harmful"] - b["refuse_harmful"],
                "refuse_harmless": m["refuse_harmless"] - b["refuse_harmless"],
                "factual_acc": m["factual_acc"] - b["factual_acc"],
                **({"misalignment_rate": m["misalignment_rate"] - b["misalignment_rate"]}
                   if "misalignment_rate" in m and "misalignment_rate" in b else {}),
                **({"rederived_auc": m["rederived"]["auc"] - b["rederived"]["auc"],
                    "rederived_ablate_refuse_harmful": (
                        m["rederived"]["ablate_refuse_harmful"] - b["rederived"]["ablate_refuse_harmful"])}
                   if "rederived" in m and "rederived" in b else {}),
            },
            "verdict": compare_verdict(b, m, opts.get("factual_floor", 0.05)),
            "capability_set": capability.name,
            "capability_n": capability.size,
            "evaluation": _evaluation_evidence(snap, opts["n_prompts"]),
            "manifest": (__import__("dataclasses").asdict(manifest) if manifest else None),
            "elapsed": reporter.total_elapsed(),
        }

    if kind == "routing":
        from vivasecuris.aiasylum.weights.routing import routing_statistics

        reporter.note(f"building prompt split ({snap['objective']})")
        split = _build_objective_split(
            snap["objective"], snap.get("objective_config"),
            opts["n_per_class"], opts["test_fraction"], opts["seed"],
        )
        # Routing statistics are descriptive, not a fit, so both halves of the
        # split are used; the split is still recorded for provenance.
        harmful = list(split.harmful_train) + list(split.harmful_test)
        harmless = list(split.harmless_train) + list(split.harmless_test)
        with reporter.step(f"loading {snap['source_model']}"):
            model, tok = load(snap["source_model"], device=opts["device"], dtype=opts["dtype"])
        reporter.note(
            f"recording expert routing over {len(harmful)} harmful and "
            f"{len(harmless)} harmless prompts"
        )
        result = routing_statistics(
            model, tok, harmful, harmless,
            max_length=opts["max_length"], thinking=bool(opts.get("thinking", False)),
            progress=lambda done, total: reporter.count(done, total, "prompts "),
        )
        result["split"] = split.summary()
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "routing.json").write_text(json.dumps(result, indent=1))
        (out_dir / "prompt_split.json").write_text(json.dumps(split.to_dict(), indent=2))
        # The per-expert tables run to thousands of numbers on a real model; the
        # row keeps the ranking and the layer shape, GET /runs/{id}/routing the rest.
        summary = {key: value for key, value in result.items() if key != "layers"}
        summary["layer_shape"] = [
            {"layer": layer["layer"], "n_experts": layer["n_experts"], "top_k": layer["top_k"]}
            for layer in result["layers"]
        ]
        summary["elapsed"] = reporter.total_elapsed()
        return summary

    if kind in ("surgery", "expert_surgery"):
        from vivasecuris.aiasylum.weights.direction import RefusalDirection
        from vivasecuris.aiasylum.weights.surgery import edit_and_save

        d = RefusalDirection.load(snap["direction_dir"]) if snap.get("direction_dir") else None
        selection = opts.get("expert_selection") if kind == "expert_surgery" else None
        expert_mode = "direction"
        if kind == "expert_surgery":
            if (snap.get("method") or "expert_ablate") == "expert_ablate":
                expert_mode = "ablate"
                reporter.note(
                    f"editing {snap['source_model']}: scaling experts {selection} by "
                    f"{opts.get('expert_scale', 0.0)} (surgery runs on CPU)"
                )
            else:
                expert_mode = "subspace" if opts.get("use_subspace") else "direction"
                reporter.note(
                    f"editing {snap['source_model']}: {expert_mode} edit inside experts "
                    f"{selection} (surgery runs on CPU)"
                )
        elif opts.get("use_subspace"):
            reporter.note(
                f"editing {snap['source_model']}: removing the rank-{d.rank} subspace "
                f"at k={opts.get('k') or 1.0} (surgery runs on CPU)"
            )
        else:
            reporter.note(
                f"editing {snap['source_model']} with beta={opts['beta']} (surgery runs on CPU)"
            )
        reporter.note("writing weights -- several minutes for a 3B model, with no progress")

        def write(staging: Path) -> None:
            edit_and_save(
                source_model=snap["source_model"],
                direction=d,
                out_dir=str(staging),
                beta=opts["beta"],
                device="cpu",
                dtype=opts["dtype"],
                include_embeddings=opts["include_embeddings"],
                notes=snap.get("notes"),
                reporter=reporter,
                use_subspace=bool(opts.get("use_subspace")),
                k=opts.get("k"),
                expert_selection=selection,
                expert_mode=expert_mode,
                expert_scale=float(opts.get("expert_scale") or 0.0),
                include_shared=bool(opts.get("include_shared_expert")),
            )

        return _write_model_output(run_id, out_dir, reporter, write, snap)

    if kind in ("lora", "distill"):
        from vivasecuris.aiasylum.api.train_runtime import run_training_worker
        from vivasecuris.aiasylum.weights.lora import LoraSpec

        run_dir = _runs_root() / str(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        dataset = snap.get("dataset") or {}
        dataset_path = dataset.get("path") or str(run_dir / "train.jsonl")
        spec = LoraSpec(
            rank=int(opts.get("lora_rank", 8)),
            alpha=int(opts.get("lora_alpha", 16)),
            dropout=float(opts.get("lora_dropout", 0.05)),
            targets=str(opts.get("lora_targets") or "attention"),
            epochs=int(opts.get("epochs", 1)),
            max_steps=opts.get("max_steps"),
            lr=float(opts.get("lr", 2e-4)),
            batch_size=int(opts.get("train_batch_size", 1)),
            grad_accum=int(opts.get("grad_accum", 8)),
            max_length=int(opts["max_length"]),
            eval_rows=int(opts.get("eval_rows", 32)),
            seed=int(opts.get("seed", 0)),
            gradient_checkpointing=bool(opts.get("gradient_checkpointing")),
            merge=bool(opts.get("merge", True)),
        )
        job: Dict[str, Any] = {
            "kind": kind,
            "source_model": snap["source_model"],
            "dataset_path": dataset_path,
            "run_dir": str(run_dir),
            "out_dir": None,
            "device": opts["device"],
            "dtype": opts["dtype"],
            "lora": spec.to_dict(),
            "notes": snap.get("notes"),
            "manifest_extra": {
                "method": snap["method"],
                "objective": snap["objective"],
                "objective_config": snap.get("objective_config"),
                "dataset_source": dataset.get("source"),
                "dataset_sha256": dataset.get("sha256"),
            },
            "distill": None,
        }
        if kind == "distill":
            job["distill"] = {
                "teacher_model": opts.get("teacher_model"),
                "level": "logit" if snap.get("method") == "logit_distill" else "response",
                "temperature": float(opts.get("distill_temperature", 2.0)),
                "ce_weight": float(opts.get("ce_weight", 0.5)),
                "teacher_max_new_tokens": int(opts.get("teacher_max_new_tokens", 256)),
                "teacher_system_prompt": opts.get("teacher_system_prompt"),
            }
        log_path = run_dir / "worker.log"
        reporter.note(f"training in a worker process; its log is {log_path}")

        def train(job_spec: Dict[str, Any]) -> Dict[str, Any]:
            result = run_training_worker(
                run_id, job_spec, log_path, reporter,
                is_cancelled=lambda: weights_cancellation.is_cancelled(run_id),
                on_pid=lambda pid: _record_worker_pid(run_id, pid),
            )
            _record_worker_pid(run_id, None)
            if result.get("stopped"):
                raise RunCancelled()
            return result

        if spec.merge:
            holder: Dict[str, Any] = {}

            def write(staging: Path) -> None:
                job["out_dir"] = str(staging)
                holder["result"] = train(job)

            summary = _write_model_output(run_id, out_dir, reporter, write, snap)
            result = holder["result"]
        else:
            result = train(job)
            summary = {
                "manifest": {},
                "output_path": result.get("output_path"),
                "size_bytes": _dir_size(run_dir),
            }

        # The full per-step log lives in train_log.jsonl; the row keeps a curve
        # short enough to render, never thousands of points in a JSON column.
        history = list(result.get("history") or [])
        if len(history) > 400:
            stride = max(1, -(-len(history) // 400))
            history = history[::stride] + history[-1:]
        # The worker wrote the merged model to the staging directory and says so;
        # the published path is the one the rename produced.
        summary.update({k: v for k, v in result.items() if k not in ("history", "manifest", "output_path")})
        if not spec.merge:
            summary["output_path"] = result.get("output_path")
        summary["history"] = history
        summary["worker_log"] = str(log_path)
        summary["elapsed"] = reporter.total_elapsed()
        return summary

    raise ValueError(f"Unknown kind '{kind}'")


def _record_worker_pid(run_id: int, pid: Optional[int]) -> None:
    """Remember the trainer's pid on the row, so a restart can stop an orphan."""
    session = get_session()
    try:
        row = session.query(WeightRun).filter(WeightRun.id == run_id).first()
        if row is not None:
            meta = dict(row.meta_data or {})
            if pid is None:
                meta.pop("worker_pid", None)
            else:
                meta["worker_pid"] = int(pid)
            row.meta_data = meta
            session.commit()
    except Exception:
        logger.warning("Could not record worker pid for run %s", run_id, exc_info=True)
    finally:
        session.close()


async def _run_weights_background(run_id: int) -> None:
    """Shares the one model slot with interp; never touches the worker pool."""
    session = get_session()
    loop = asyncio.get_running_loop()
    try:
        row = session.query(WeightRun).filter(WeightRun.id == run_id).first()
        if row is None or weights_cancellation.is_cancelled(run_id):
            return

        # A run queued behind another model job sits at pending with nothing to
        # show for it, which is indistinguishable from the hang this whole
        # progress machinery exists to rule out. Say so before blocking.
        status = slot_status()
        if status.get("held_by"):
            await weights_progress.emit_event(
                run_id, "weights_queued", status,
                f"Waiting for the model slot, held by {status['held_by']}",
            )

        async with hold(f"weights run {run_id}"):
            row = session.query(WeightRun).filter(WeightRun.id == run_id).first()
            if row is None or weights_cancellation.is_cancelled(run_id):
                return

            opts = dict((row.meta_data or {}).get("options") or {})
            snapshot = {
                "kind": row.kind,
                "source_model": row.source_model,
                "source_run_id": row.source_run_id,
                "method": row.method,
                "objective": row.objective,
                "objective_config": (row.meta_data or {}).get("objective_config"),
                "options": opts,
                "out_dir": row.out_dir,
                "direction_dir": (row.meta_data or {}).get("direction_dir"),
                "source_direction": (row.meta_data or {}).get("source_direction") or {},
                "source_options": (row.meta_data or {}).get("source_options") or {},
                "notes": opts.get("notes"),
                "modified_model": (row.meta_data or {}).get("modified_model"),
                "dataset": (row.meta_data or {}).get("dataset"),
            }

            row.status = STATUS_RUNNING
            row.started_at = datetime.utcnow()
            session.commit()

            await weights_progress.emit_event(
                run_id, "weights_started",
                {"status": STATUS_RUNNING, "kind": row.kind},
                f"Starting {STAGES[row.kind]['label'].lower()}",
            )

            reporter = _make_reporter(run_id, loop)
            # Cancellation stops the awaiting task, not its worker thread.
            # Keep the model slot until the worker reaches a cooperative
            # cancellation check or finishes loading/saving its model.
            worker = asyncio.create_task(
                asyncio.to_thread(_execute, run_id, snapshot, reporter)
            )
            cancelled = False
            while True:
                try:
                    summary = await asyncio.shield(worker)
                    break
                except asyncio.CancelledError:
                    if worker.cancelled():
                        raise
                    if not cancelled:
                        asyncio.run_coroutine_threadsafe(
                            weights_progress.emit_event(
                                run_id, "weights_stopping", {"status": STATUS_RUNNING},
                                "Stop requested; waiting for the current model operation to release memory",
                            ),
                            loop,
                        )
                    cancelled = True
                except Exception:
                    if cancelled:
                        raise RunCancelled()
                    raise
            if cancelled:
                raise RunCancelled()

            row = session.query(WeightRun).filter(WeightRun.id == run_id).first()
            row.status = STATUS_COMPLETED
            row.completed_at = datetime.utcnow()
            row.meta_data = {**(row.meta_data or {}), "summary": summary}
            if row.out_dir:
                row.artifact_bytes = _dir_size(Path(row.out_dir))
            session.commit()

            await weights_progress.emit_event(
                run_id, "weights_completed",
                {"status": STATUS_COMPLETED, **_headline(row.kind, summary)},
                "Complete",
            )

    except (asyncio.CancelledError, RunCancelled):
        _fail(session, run_id, "Cancelled", cancelled=True)
        await weights_progress.emit_event(
            run_id, "weights_cancelled",
            {"status": STATUS_FAILED, "cancelled": True}, "Run was cancelled",
        )
    except Exception as exc:
        logger.exception("Weight run %s failed", run_id)
        _fail(session, run_id, str(exc))
        await weights_progress.emit_event(
            run_id, "weights_failed",
            {"status": STATUS_FAILED, "error": str(exc)}, f"Failed: {exc}",
        )
    finally:
        session.close()
        weights_cancellation.unregister_task(run_id)


def _headline(kind: str, summary: Dict[str, Any]) -> Dict[str, Any]:
    """The one number each stage is actually judged on."""
    if kind == "direction":
        return {"layer": summary.get("layer"), "auc": summary.get("auc"),
                "usable": summary.get("usable"), "method": summary.get("method"),
                "rank": summary.get("rank"),
                "stable_rank": ((summary.get("extra") or {}).get("stable_rank") or {}).get("at_layer"),
                "refusal_overlap": (summary.get("refusal_overlap") or {}).get("cosine_with_refusal")}
    if kind == "sweep" and summary.get("curve") is not None:
        return {"k50_rank": summary.get("k50_rank"), "max_compliance": summary.get("max_compliance"),
                "rank_at_max": summary.get("rank_at_max"), "monotone": summary.get("monotone"),
                "any_degenerate": summary.get("any_degenerate")}
    if kind == "sweep":
        return {"verdict": summary.get("verdict"),
                "ablate_delta_points": summary.get("ablate_delta_points"),
                "factual_delta_points": summary.get("factual_delta_points"),
                "any_degenerate": summary.get("any_degenerate")}
    if kind == "select":
        best = summary.get("best") or {}
        return {"best_rank": best.get("rank"), "best_k": best.get("k"),
                "admissible": sum(1 for r in summary.get("frontier", []) if r.get("accepted"))}
    if kind == "probe":
        return {"best_layer": summary.get("best_layer"), "auroc": summary.get("best_auroc"),
                "null_p95": summary.get("null_auroc_p95"), "beats_null": summary.get("beats_null"),
                "usable": summary.get("usable")}
    if kind == "compare":
        d = summary.get("deltas", {})
        return {"refusal_delta": d.get("refuse_harmful"),
                "capability_delta": d.get("factual_acc"),
                "misalignment_delta": d.get("misalignment_rate"),
                "rederived_auc_delta": d.get("rederived_auc")}
    if kind in ("lora", "distill"):
        train = summary.get("train") or {}
        return {"steps": train.get("steps"), "final_loss": train.get("final_loss"),
                "eval_loss_before": train.get("eval_loss_before"),
                "eval_loss_after": train.get("eval_loss_after"), "mean_kl": train.get("mean_kl"),
                "merged": summary.get("merged"), "output_path": summary.get("output_path"),
                "size_bytes": summary.get("size_bytes")}
    if kind == "routing":
        top = (summary.get("ranking") or [{}])[0]
        return {"top_layer": top.get("layer"), "top_expert": top.get("expert"),
                "top_delta": top.get("delta"), "moe_layers": summary.get("moe_layers"),
                "consistent": (summary.get("consistency") or {}).get("gate_vs_expert_counts_match")}
    if kind == "expert_surgery":
        manifest = summary.get("manifest") or {}
        extra = manifest.get("extra") or {}
        return {"expert_mode": extra.get("expert_mode"), "experts": extra.get("experts_edited"),
                "layers": extra.get("layers_edited"), "matrices_edited": manifest.get("matrices_edited"),
                "output_path": summary.get("output_path"), "size_bytes": summary.get("size_bytes")}
    return {"output_path": summary.get("output_path"), "size_bytes": summary.get("size_bytes")}


def _fail(session, run_id: int, error: str, cancelled: bool = False) -> None:
    try:
        row = session.query(WeightRun).filter(WeightRun.id == run_id).first()
        if row is not None:
            row.status = STATUS_FAILED
            row.error = error
            row.completed_at = datetime.utcnow()
            if cancelled:
                row.meta_data = {**(row.meta_data or {}), "cancelled": True}
            session.commit()
    except Exception:
        session.rollback()
        logger.exception("Could not record failure for weight run %s", run_id)


def _resolve_direction(request: WeightRunRequest) -> WeightRun:
    """Resolve and gate the direction a sweep or surgery consumes."""
    if request.source_run_id is None:
        raise HTTPException(
            status_code=400,
            detail=f"'{request.kind}' consumes a derived direction; pass source_run_id.",
        )

    session = get_session()
    try:
        row = (
            session.query(WeightRun)
            .filter(WeightRun.id == request.source_run_id)
            .first()
        )
        if row is None:
            raise HTTPException(
                status_code=404, detail=f"Direction run {request.source_run_id} not found."
            )
        if row.kind != "direction":
            raise HTTPException(
                status_code=400,
                detail=f"Run {row.id} is a '{row.kind}', not a direction.",
            )
        if row.status != STATUS_COMPLETED:
            raise HTTPException(
                status_code=409, detail=f"Direction run {row.id} is '{row.status}'."
            )
        # The row outlives its files whenever a run is deleted or runs/ is
        # cleaned, so check the disk rather than trusting the status.
        if not row.out_dir or not (Path(row.out_dir) / "direction.safetensors").exists():
            raise HTTPException(
                status_code=410,
                detail=(
                    f"The artifacts for direction run {row.id} are gone from "
                    f"{row.out_dir}. Derive it again."
                ),
            )
        session.expunge(row)
        return row
    finally:
        session.close()


def _teacher_id_problem(model_id: str) -> Optional[str]:
    """Why this is not a usable teacher, or None.

    The loader's name rule is the ADR-009 enforcement point: only a model it can
    load is local open-weights, and only local output is trainable on. The
    loader module itself is torch-free at import, so this costs nothing here.
    """
    try:
        from vivasecuris.aiasylum.interp.core.loader import _check_model_id
    except Exception:
        return None
    try:
        _check_model_id(model_id)
    except ValueError as exc:
        return str(exc)
    return None


def _training_request(request: "WeightRunRequest") -> Optional[Dict[str, Any]]:
    """What the preflight needs to know about a training run."""
    if request.kind not in ("lora", "distill"):
        return None
    return {
        "merge": request.merge,
        "teacher_model": request.teacher_model,
        "distill_level": "logit" if request.method == "logit_distill" else "response",
    }


def _resolve_training_rows(request: "WeightRunRequest", objective: str, objective_config):
    """The rows a training run will see, from whichever source the request named."""
    from vivasecuris.aiasylum.weights.train_data import (
        parse_rows,
        rows_from_benchmark,
        rows_from_prompts,
    )

    if request.dataset_rows:
        return parse_rows(request.dataset_rows, require_response=request.kind == "lora"), "rows"
    if request.dataset_benchmark:
        bench = request.dataset_benchmark
        try:
            rows = rows_from_benchmark(str(bench["name"]), int(bench.get("count", 256)), request.seed)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Could not load benchmark rows: {exc}")
        if not rows:
            raise HTTPException(status_code=400, detail=f"Benchmark {bench['name']!r} yielded no rows.")
        return rows, "benchmark"
    try:
        split = _build_objective_split(
            objective, objective_config, request.n_per_class, request.test_fraction, request.seed,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    prompts = (
        list(split.harmful_train) + list(split.harmful_test)
        + list(split.harmless_train) + list(split.harmless_test)
    )
    return rows_from_prompts(prompts), "objective"


def _validate(request: WeightRunRequest) -> None:
    if request.kind not in STAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown stage '{request.kind}'. Expected one of {sorted(STAGES)}.",
        )
    if not request.source_model or not request.source_model.strip():
        raise HTTPException(status_code=400, detail="source_model is required.")
    if request.objective not in OBJECTIVES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown objective '{request.objective}'. Expected one of {sorted(OBJECTIVES)}.",
        )

    method_name = request.method or DEFAULT_METHOD_FOR_STAGE.get(request.kind)
    method = METHODS.get(method_name)
    if method is None:
        label_reason = UNAVAILABLE_METHODS.get(method_name)
        if label_reason:
            raise HTTPException(
                status_code=409,
                detail=f"{label_reason[0]} is not implemented here. {label_reason[1]}",
            )
        raise HTTPException(status_code=400, detail=f"Unknown method '{method_name}'.")

    # Method and stage must agree. Without this a surgery run could be launched
    # naming a steering method, and the manifest would then carry two
    # contradictory answers to "what was done": the engine always writes
    # method="direction_scale", while the request said otherwise.
    if method["stage"] != request.kind:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Method '{method_name}' belongs to the '{method['stage']}' stage, "
                f"but this is a '{request.kind}' run. Pick a method for this stage."
            ),
        )

    if request.kind == "direction" and request.n_per_class < 8:
        raise HTTPException(
            status_code=400,
            detail="n_per_class must be at least 8; the split is rejected below that.",
        )

    if not 0 < request.test_fraction < 1:
        raise HTTPException(status_code=400, detail="test_fraction must be between zero and one.")
    if min(request.max_length, request.batch_size, request.n_prompts, request.max_new_tokens) < 1:
        raise HTTPException(status_code=400, detail="Token limits, batch size and evaluation prompt count must be positive.")
    if request.kind == "direction" and request.objective == "over_refusal":
        cfg = request.objective_config or {}
        if any(len(set(cfg.get(key) or [])) < 8 for key in ("refused", "answered")):
            raise HTTPException(status_code=400, detail="Over-refusal needs at least eight distinct refused and eight distinct answered benign prompts.")

    if request.kind == "direction" and request.subspace_rank is not None:
        if request.subspace_rank < 1:
            raise HTTPException(status_code=400, detail="subspace_rank must be at least 1.")
    if request.kind == "direction" and request.rfm_rank is not None and request.rfm_rank < 1:
        raise HTTPException(status_code=400, detail="rfm_rank must be at least 1.")
    if request.kind == "direction" and request.rfm_iterations < 1:
        raise HTTPException(status_code=400, detail="rfm_iterations must be at least 1.")
    if request.kind == "sweep" and request.timeline_prompts < 0:
        raise HTTPException(status_code=400, detail="timeline_prompts cannot be negative.")

    if request.kind == "probe" and request.pooling not in ("last", "mean", "max", "last_k"):
        raise HTTPException(
            status_code=400,
            detail=f"Unknown pooling '{request.pooling}'. Expected last, mean, max or last_k.",
        )
    if request.kind == "probe" and request.holdout_techniques < 0:
        raise HTTPException(status_code=400, detail="holdout_techniques cannot be negative.")
    if request.kind == "probe" and (min(request.n_direct, request.n_jailbreak, request.n_benign) < 0 or not request.n_benign):
        raise HTTPException(status_code=400, detail="Probe counts must be non-negative and n_benign must be positive.")
    if request.kind == "probe" and not request.n_jailbreak and request.holdout_techniques:
        raise HTTPException(status_code=400, detail="Set held-out techniques to zero for a direct-only probe.")
    if request.kind == "probe" and request.jailbreak_examples is not None:
        if not request.jailbreak_examples or any(not row.get("prompt", "").strip() or not row.get("technique", "").strip()
                                                for row in request.jailbreak_examples):
            raise HTTPException(status_code=400, detail="Each supplied jailbreak example needs a non-empty prompt and technique.")

    if request.kind == "routing" and request.n_per_class < 8:
        raise HTTPException(
            status_code=400,
            detail="n_per_class must be at least 8; the prompt split is rejected below that.",
        )

    if request.kind == "expert_surgery":
        selection = request.expert_selection
        if not isinstance(selection, dict) or not selection:
            raise HTTPException(
                status_code=400,
                detail=(
                    "expert_surgery needs expert_selection: a mapping of layer to a list "
                    'of expert indices or "all", e.g. {"12": [3, 7], "15": "all"}.'
                ),
            )
        for layer, experts in selection.items():
            if not str(layer).strip().isdigit():
                raise HTTPException(
                    status_code=400, detail=f"expert_selection key {layer!r} is not a layer index."
                )
            if isinstance(experts, str):
                if experts.strip().lower() != "all":
                    raise HTTPException(
                        status_code=400,
                        detail=f'expert_selection[{layer}] must be a list of expert indices or "all".',
                    )
            elif (
                not isinstance(experts, list) or not experts
                or any(isinstance(e, bool) or not isinstance(e, int) or e < 0 for e in experts)
            ):
                raise HTTPException(
                    status_code=400,
                    detail=f'expert_selection[{layer}] must be a non-empty list of expert indices or "all".',
                )
        if method_name == "expert_direction_scale" and request.source_run_id is None:
            raise HTTPException(
                status_code=400,
                detail="expert_direction_scale needs source_run_id: the direction to scale inside the chosen experts.",
            )
        if method_name == "expert_ablate" and request.use_subspace:
            raise HTTPException(
                status_code=400,
                detail="expert_ablate scales whole down-projections; pick expert_direction_scale for a subspace edit.",
            )

    if request.kind in ("lora", "distill"):
        sources = [
            name for name, present in (
                ("dataset_rows", bool(request.dataset_rows)),
                ("dataset_benchmark", bool(request.dataset_benchmark)),
                ("dataset_source='objective'", request.dataset_source == "objective"),
            ) if present
        ]
        if len(sources) != 1:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Give exactly one dataset source: dataset_rows (inline rows), "
                    "dataset_benchmark ({name, count}), or dataset_source='objective' "
                    "(the objective's prompt corpus; distillation only)."
                ),
            )
        if request.dataset_source not in (None, "rows", "benchmark", "objective"):
            raise HTTPException(status_code=400, detail="dataset_source must be rows, benchmark or objective.")
        if request.dataset_source == "objective":
            if request.kind == "lora":
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "The objective corpus holds prompts without responses, so LoRA cannot "
                        "train on it directly. Use distillation to have a teacher answer them, "
                        "or supply rows with responses."
                    ),
                )
            if request.objective == "dataset":
                raise HTTPException(
                    status_code=400,
                    detail="dataset_source='objective' needs a prompt objective such as refusal, not 'dataset'.",
                )
            if request.n_per_class < 8:
                raise HTTPException(status_code=400, detail="n_per_class must be at least 8; the prompt split is rejected below that.")
        if request.dataset_rows is not None:
            from vivasecuris.aiasylum.weights.train_data import parse_rows

            try:
                parse_rows(request.dataset_rows, require_response=request.kind == "lora")
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc))
        if request.dataset_benchmark is not None:
            bench = request.dataset_benchmark if isinstance(request.dataset_benchmark, dict) else {}
            name, count = bench.get("name"), bench.get("count", 256)
            if not isinstance(name, str) or not name.strip():
                raise HTTPException(status_code=400, detail="dataset_benchmark needs a name.")
            if isinstance(count, bool) or not isinstance(count, int) or count < 8:
                raise HTTPException(status_code=400, detail="dataset_benchmark.count must be an integer of at least 8.")
        if request.lora_rank < 1:
            raise HTTPException(status_code=400, detail="lora_rank must be at least 1.")
        if request.lora_alpha <= 0:
            raise HTTPException(status_code=400, detail="lora_alpha must be positive.")
        if not 0 <= request.lora_dropout < 1:
            raise HTTPException(status_code=400, detail="lora_dropout must be in [0, 1).")
        if request.epochs < 1 or (request.max_steps is not None and request.max_steps < 1):
            raise HTTPException(status_code=400, detail="epochs and max_steps must be at least 1.")
        if request.lr <= 0:
            raise HTTPException(status_code=400, detail="lr must be positive.")
        if min(request.train_batch_size, request.grad_accum) < 1:
            raise HTTPException(status_code=400, detail="train_batch_size and grad_accum must be at least 1.")
        if request.max_length < 16:
            raise HTTPException(status_code=400, detail="max_length must be at least 16 for training rows.")
        if request.eval_rows < 0:
            raise HTTPException(status_code=400, detail="eval_rows cannot be negative.")
        if not request.lora_targets.strip():
            raise HTTPException(status_code=400, detail="lora_targets must name what to adapt: attention, attention+mlp, or module names.")
        if request.merge and not (request.output_name or "").strip():
            raise HTTPException(
                status_code=400,
                detail="merge=true writes a model directory and needs output_name; set merge=false to keep only the adapter.",
            )

    if request.kind == "distill":
        teacher = (request.teacher_model or "").strip()
        if not teacher:
            raise HTTPException(
                status_code=400,
                detail="distill needs teacher_model: a local open-weights model the loader can load.",
            )
        problem = _teacher_id_problem(teacher)
        if problem:
            raise HTTPException(status_code=400, detail=problem)
        if not 0 <= request.ce_weight <= 1:
            raise HTTPException(status_code=400, detail="ce_weight must be between 0 and 1.")
        if request.distill_temperature <= 0:
            raise HTTPException(status_code=400, detail="distill_temperature must be positive.")
        if request.teacher_max_new_tokens < 1:
            raise HTTPException(status_code=400, detail="teacher_max_new_tokens must be at least 1.")

    if request.kind == "compare" and not (request.modified_model or "").strip():
        raise HTTPException(
            status_code=400,
            detail="compare needs modified_model: the edited directory to measure.",
        )


@router.post("/runs", response_model=WeightRunResponse)
async def create_weight_run(request: WeightRunRequest):
    """Create a stage run and start it in the background."""
    _validate(request)

    direction_row = None
    direction_dir = None
    source_direction: Dict[str, Any] = {}
    source_options: Dict[str, Any] = {}
    # `compare` measures two finished models, so it is the one stage with no
    # direction behind it.
    method_name = request.method or DEFAULT_METHOD_FOR_STAGE.get(request.kind)
    consumes_direction = request.kind in WEIGHT_KINDS_CONSUMING_DIRECTION or (
        request.kind == "expert_surgery" and method_name == "expert_direction_scale"
    )
    if consumes_direction:
        direction_row = _resolve_direction(request)
        direction_dir = direction_row.out_dir
        # Snapshot rather than join: the child page must still render once the
        # parent row is gone.
        source_direction = _direction_summary(direction_row)
        source_options = (direction_row.meta_data or {}).get("options") or {}
        # A rank curve walks the directions of a subspace; a single vector has none.
        if request.kind == "sweep" and request.method == "subspace_curve":
            if int(source_direction.get("rank") or 1) < 2:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "This direction holds a single vector, so there is no rank curve to "
                        "measure. Derive a subspace (rank above 1, or the RFM-AGOP method) first."
                    ),
                )

    # A child stage measures or edits the direction it was given, so it inherits
    # what that direction was fitted for. Taking the request's own objective
    # would let a sweep default to "refusal" while checking a narrowed direction.
    objective = request.objective
    objective_config = request.objective_config
    if direction_row is not None:
        objective = direction_row.objective or objective
        objective_config = (direction_row.meta_data or {}).get(
            "objective_config", objective_config
        )
    elif request.kind == "compare":
        # An edited model carries its derivation partition into every later
        # comparison. Resampling here would contaminate the reported holdout.
        session = get_session()
        try:
            surgery = session.query(WeightRun).filter(
                WeightRun.kind.in_(WEIGHT_KINDS_WRITING_MODELS), WeightRun.out_dir == request.modified_model
            ).order_by(WeightRun.id.desc()).first()
            if surgery is not None:
                meta = surgery.meta_data or {}
                source_direction = meta.get("source_direction") or {}
                source_options = meta.get("source_options") or {}
                objective = surgery.objective or objective
                objective_config = meta.get("objective_config", objective_config)
            else:
                from vivasecuris.aiasylum.weights.manifest import SurgeryManifest
                manifest = SurgeryManifest.load(request.modified_model)
                if manifest is not None:
                    extra = manifest.extra or {}
                    source_direction = {"prompt_split": extra.get("prompt_split"), "split_hash": manifest.split_hash}
                    source_options = extra.get("source_options") or {}
                    objective = extra.get("objective") or objective
                    objective_config = extra.get("objective_config", objective_config)
        finally:
            session.close()

    out_dir: Optional[Path] = None
    # A training run that keeps only its adapter writes under runs/, not models/.
    if request.kind in WEIGHT_KINDS_WRITING_MODELS and (
        request.kind not in ("lora", "distill") or request.merge
    ):
        name = _validate_slug(request.output_name)
        out_dir = _resolve_output_dir(name)
        session = get_session()
        try:
            clash = (
                session.query(WeightRun)
                .filter(
                    WeightRun.kind.in_(WEIGHT_KINDS_WRITING_MODELS),
                    WeightRun.out_dir == str(out_dir),
                    WeightRun.status.in_([STATUS_PENDING, STATUS_RUNNING]),
                )
                .first()
            )
        finally:
            session.close()
        if clash is not None:
            # The slot serialises execution, but two requests can both pass
            # validation while queued behind it.
            raise HTTPException(
                status_code=409,
                detail=f"Weight run {clash.id} is already queued to write '{name}'.",
            )

    checks = _preflight_checks(
        request.kind, request.source_model, direction_row, request.output_name, request.modified_model, request.dtype,
        expert_selection=request.expert_selection, training=_training_request(request),
    )
    unacknowledged = [
        c for c in checks.checks
        if c.severity == "blocking" and (not c.acknowledgeable or c.code not in request.acknowledge)
    ]
    if unacknowledged:
        raise HTTPException(
            status_code=409,
            detail={
                "message": "Preflight blocked this run.",
                "blocking": [c.model_dump() for c in unacknowledged],
                "hint": "Resolve these, or resend with acknowledge=[codes] for the ones marked acknowledgeable.",
            },
        )

    # Resolved before the row exists, so a bad benchmark name or an empty
    # corpus fails the request rather than leaving a pending row behind.
    training_rows = (
        _resolve_training_rows(request, objective, objective_config)
        if request.kind in ("lora", "distill") else None
    )

    session = get_session()
    try:
        row = WeightRun(
            kind=request.kind,
            status=STATUS_PENDING,
            source_model=request.source_model.strip(),
            source_run_id=request.source_run_id,
            method=request.method or DEFAULT_METHOD_FOR_STAGE.get(request.kind),
            objective=objective,
            meta_data={
                "lineage_parent": request.lineage_parent,
                "lineage_id": uuid4().hex,
                "source_direction_lineage_id": (direction_row.meta_data or {}).get("lineage_id") if direction_row is not None else None,
                "options": {
                    "device": request.device,
                    "dtype": request.dtype,
                    "n_per_class": request.n_per_class,
                    "test_fraction": request.test_fraction,
                    "seed": request.seed,
                    "batch_size": request.batch_size,
                    "max_length": request.max_length,
                    "n_prompts": request.n_prompts,
                    "max_new_tokens": request.max_new_tokens,
                    "alphas": request.alphas,
                    "include_ablation": request.include_ablation,
                    "capability_control": request.capability_control,
                    "capability_limit": request.capability_limit,
                    "beta": request.beta,
                    "include_embeddings": request.include_embeddings,
                    "use_subspace": request.use_subspace,
                    "k": request.k,
                    "subspace_rank": request.subspace_rank,
                    "pool_layers": request.pool_layers,
                    "rfm_rank": request.rfm_rank,
                    "rfm_iterations": request.rfm_iterations,
                    "rfm_beta": request.rfm_beta,
                    "report_overlap": request.report_overlap,
                    "thinking": request.thinking,
                    "timeline_prompts": request.timeline_prompts,
                    "rederive": request.rederive or request.method == "compare_rederive",
                    "misalignment_control": request.misalignment_control,
                    "pooling": request.pooling,
                    "prompt_suffix": request.prompt_suffix,
                    "use_eliciting_suffix": request.use_eliciting_suffix,
                    "n_direct": request.n_direct,
                    "n_jailbreak": request.n_jailbreak,
                    "n_benign": request.n_benign,
                    "holdout_techniques": request.holdout_techniques,
                    "jailbreak_examples": request.jailbreak_examples,
                    "ranks": request.ranks,
                    "ks": request.ks,
                    "factual_floor": request.factual_floor,
                    "capability_set": request.capability_set,
                    "expert_selection": request.expert_selection,
                    "expert_scale": request.expert_scale,
                    "include_shared_expert": request.include_shared_expert,
                    # Training. The rows themselves go to train.jsonl, never here.
                    "dataset_source": (
                        request.dataset_source
                        or ("rows" if request.dataset_rows else "benchmark" if request.dataset_benchmark else None)
                    ),
                    "dataset_benchmark": request.dataset_benchmark,
                    "lora_rank": request.lora_rank,
                    "lora_alpha": request.lora_alpha,
                    "lora_dropout": request.lora_dropout,
                    "lora_targets": request.lora_targets,
                    "epochs": request.epochs,
                    "max_steps": request.max_steps,
                    "lr": request.lr,
                    "train_batch_size": request.train_batch_size,
                    "grad_accum": request.grad_accum,
                    "gradient_checkpointing": request.gradient_checkpointing,
                    "merge": request.merge,
                    "eval_rows": request.eval_rows,
                    "teacher_model": (request.teacher_model or "").strip() or None,
                    "distill_temperature": request.distill_temperature,
                    "ce_weight": request.ce_weight,
                    "teacher_max_new_tokens": request.teacher_max_new_tokens,
                    "teacher_system_prompt": request.teacher_system_prompt,
                    "notes": request.notes,
                },
                "modified_model": (request.modified_model or "").strip() or None,
                "objective_config": objective_config,
                "direction_dir": direction_dir,
                "source_direction": source_direction,
                "source_options": source_options,
                "preflight": {
                    "checks": [c.model_dump() for c in checks.checks],
                    "acknowledged": list(request.acknowledge),
                },
            },
        )
        session.add(row)
        session.commit()
        session.refresh(row)

        row.out_dir = str(out_dir) if out_dir else str(_runs_root() / str(row.id))
        session.commit()
        session.refresh(row)

        if training_rows is not None:
            from vivasecuris.aiasylum.weights.train_data import dataset_digest, write_jsonl

            rows, source = training_rows
            run_dir = _runs_root() / str(row.id)
            run_dir.mkdir(parents=True, exist_ok=True)
            path = run_dir / "train.jsonl"
            write_jsonl(path, rows)
            row.meta_data = {
                **(row.meta_data or {}),
                "dataset": {"source": source, "n_rows": len(rows), "sha256": dataset_digest(rows), "path": str(path)},
            }
            session.commit()
            session.refresh(row)

        task = asyncio.create_task(_run_weights_background(row.id))
        weights_cancellation.register_task(row.id, task)

        return WeightRunResponse.from_orm_row(row)
    finally:
        session.close()


@router.get("/runs/{run_id}/routing")
async def get_routing_statistics(run_id: int):
    """The full per-layer, per-expert tables a routing run wrote to routing.json."""
    session = get_session()
    try:
        row = session.query(WeightRun).filter(WeightRun.id == run_id).first()
        if row is None:
            raise HTTPException(status_code=404, detail="Weight run not found")
        if row.kind != "routing":
            raise HTTPException(
                status_code=400, detail=f"Run {run_id} is a {row.kind} run, not a routing run."
            )
        out_dir = row.out_dir
    finally:
        session.close()
    path = Path(out_dir) / "routing.json" if out_dir else None
    if path is None or not path.is_file():
        raise HTTPException(status_code=404, detail="This routing run has not written routing.json yet.")
    try:
        return json.loads(path.read_text())
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=f"Unreadable routing.json: {exc}")


@router.post("/runs/{run_id}/stop")
async def stop_weight_run(run_id: int):
    session = get_session()
    try:
        row = session.query(WeightRun).filter(WeightRun.id == run_id).first()
        if row is None:
            raise HTTPException(status_code=404, detail="Weight run not found")
        if row.status not in (STATUS_PENDING, STATUS_RUNNING):
            return {"stopped": False, "status": row.status, "reason": "not running"}
    finally:
        session.close()

    weights_cancellation.cancel(run_id)
    return {"stopped": True, "run_id": run_id}


@router.get("/runs/{run_id}/progress")
async def stream_weight_progress(run_id: int):
    """Server-Sent Events, mirroring the interp and test-run streams."""
    from fastapi.responses import StreamingResponse

    session = get_session()
    try:
        row = session.query(WeightRun).filter(WeightRun.id == run_id).first()
        if row is None:
            raise HTTPException(status_code=404, detail="Weight run not found")
        current_status = row.status
    finally:
        session.close()

    async def stream():
        yield "data: " + json.dumps({
            "test_run_id": run_id,
            "event_type": "status_update",
            "timestamp": datetime.utcnow().isoformat(),
            "data": {"status": current_status},
            "message": f"Current status: {current_status}",
        }) + "\n\n"
        async for event in weights_progress.stream_events(run_id):
            yield event

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.delete("/runs/{run_id}")
async def delete_weight_run(
    run_id: int,
    delete_artifacts: bool = False,
    confirm: Optional[str] = None,
    force: bool = False,
):
    """Delete a run, and optionally what it wrote.

    A surgery output is a multi-gigabyte model that other runs may be testing
    against, so removing the row and removing the weights are separate
    decisions: by default the bytes stay. Purging requires the directory name
    echoed back, which is what makes it deliberate rather than accidental.
    """
    import shutil

    session = get_session()
    try:
        row = session.query(WeightRun).filter(WeightRun.id == run_id).first()
        if row is None:
            raise HTTPException(status_code=404, detail="Weight run not found")
        if row.status in (STATUS_PENDING, STATUS_RUNNING):
            raise HTTPException(
                status_code=409,
                detail="Stop this run and wait for it to finish before deleting its artifacts.",
            )
        kind, out_dir = row.kind, row.out_dir
        from vivasecuris.aiasylum.api.model_history import archive_run
        archive_run(row)
    finally:
        session.close()

    removed_artifacts = False
    if delete_artifacts and out_dir:
        path = Path(out_dir)
        if kind in WEIGHT_KINDS_WRITING_MODELS:
            if confirm != path.name:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"Deleting the weights at {path} is irreversible. Resend with "
                        f"confirm={path.name!r} to proceed."
                    ),
                )
            if not (path / MANIFEST_NAME).exists():
                # Not a directory this project wrote; refuse regardless of confirm.
                raise HTTPException(
                    status_code=409,
                    detail=f"{path} holds no {MANIFEST_NAME}; refusing to remove it.",
                )
            users = _models_in_use(str(path))
            if users and not force:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"Test runs {users} used this model. Deleting the weights makes "
                        f"those results unreproducible. Resend with force=true to proceed."
                    ),
                )
            if _is_inside(path, _models_root()) and path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
                removed_artifacts = True
        else:
            if _is_inside(path, _runs_root()) and path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
                removed_artifacts = True

    session = get_session()
    try:
        row = session.query(WeightRun).filter(WeightRun.id == run_id).first()
        if row is not None:
            session.delete(row)
            session.commit()
    finally:
        session.close()

    return {"deleted": run_id, "artifacts_removed": removed_artifacts}


def _models_in_use(path: str) -> List[int]:
    session = get_session()
    try:
        return [
            t.id
            for t in session.query(TestRun)
            .filter((TestRun.patient_model == path) | (TestRun.doctor_model == path))
            .all()
        ]
    finally:
        session.close()


# --------------------------------------------------------------------------
# Chat
# --------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=32000)


class ChatRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    messages: List[ChatMessage] = Field(max_length=100)
    system_prompt: Optional[str] = Field(None, max_length=32000)
    # Greedy by default so what you see matches what the surgery measurements
    # were taken with; a sampled reply is not evidence about the edit.
    temperature: float = Field(0.0, ge=0, le=2)
    max_tokens: int = Field(256, ge=1, le=4096)
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"


@router.post("/models/{name}/chat")
async def chat_with_model(name: str, request: ChatRequest):
    """One turn against an edited model.

    Deliberately not a run kind: this is interactive, and a conversation must
    not hold the single model slot for its whole length or nothing else could
    run while someone is typing. The slot is taken per turn instead, and the
    model is loaded through the serving cache that the test harness already
    uses -- so it stays warm between turns and the first turn pays the load.

    Served through the same `transformers` provider a test run would use, with
    no quantization step, so what you read here is what the harness will see.
    """
    _validate_slug(name)
    path = _resolve_output_dir(name)
    if not path.is_dir():
        raise HTTPException(status_code=404, detail=f"No model directory at {path}.")

    if not request.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty.")
    if request.messages[-1].role != "user":
        raise HTTPException(status_code=400, detail="The final chat message must be a user turn.")
    if any(not message.content.strip() for message in request.messages):
        raise HTTPException(status_code=400, detail="Chat turns must contain text.")
    if sum(len(message.content) for message in request.messages) + len(request.system_prompt or "") > 128000:
        raise HTTPException(status_code=400, detail="Conversation is too large. Start a new conversation or shorten the history.")

    from vivasecuris.aiasylum.api.model_catalog import checkpoint_status
    status = checkpoint_status(path)
    if status["availability"] != "ready":
        raise HTTPException(status_code=409, detail=f"Checkpoint is unavailable for chat: {status['reason']}")

    if not _interp_extra_installed():
        raise HTTPException(
            status_code=409,
            detail='Needs the optional extra: pip install -e ".[interp]"',
        )

    history = [{"role": m.role, "content": m.content} for m in request.messages]

    def _turn():
        import asyncio as _asyncio
        import time

        from vivasecuris.aiasylum.models import get_provider

        provider = get_provider("transformers")
        model = provider.create_model(
            str(path),
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            device=request.device,
            dtype=request.dtype,
        )
        started = time.perf_counter()
        response = _asyncio.run(
            model.generate(
                prompt="", system_prompt=request.system_prompt, messages=history
            )
        )
        return response, time.perf_counter() - started

    try:
        async with hold(f"chat with {path.name}"):
            worker = asyncio.create_task(asyncio.to_thread(_turn))
            cancelled = False
            while True:
                try:
                    response, elapsed = await asyncio.shield(worker)
                    break
                except asyncio.CancelledError:
                    if worker.cancelled():
                        raise
                    cancelled = True
                except Exception:
                    if cancelled:
                        raise asyncio.CancelledError()
                    raise
            # Cancelling a request does not stop the model thread. The slot
            # stays leased until that thread exits, then cancellation propagates.
            if cancelled:
                raise asyncio.CancelledError()
    except Exception as exc:
        logger.exception("Chat turn failed for %s", path)
        raise HTTPException(status_code=500, detail=str(exc))

    from vivasecuris.aiasylum.weights.manifest import SurgeryManifest

    manifest = SurgeryManifest.load(path)
    from vivasecuris.aiasylum.weights.steering import refusal_rate

    return {
        "content": response.content,
        "model": str(path),
        "edited": manifest is not None,
        "manifest": (__import__("dataclasses").asdict(manifest) if manifest else None),
        "usage": response.usage or {},
        "finish_reason": response.finish_reason,
        "elapsed_seconds": elapsed,
        "refused": bool(refusal_rate([response.content])),
        "refusal_detector": "phrase_heuristic",
        "truncated": response.finish_reason == "length",
        "settings": {"temperature": request.temperature, "max_tokens": request.max_tokens,
                     "device": request.device, "dtype": request.dtype},
    }
