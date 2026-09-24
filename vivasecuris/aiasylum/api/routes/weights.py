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
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    "lora": (
        "LoRA",
        "setup.py declares the extra but no training loop exists. "
        "No bitsandbytes on MPS, so any LoRA here would be plain bf16.",
    ),
    "distillation": (
        "Distillation",
        "Not implemented. Under ADR-009 a teacher must be local open-weights for its "
        "output to be trainable on.",
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
}

# What each stage runs when the caller does not name a method.
DEFAULT_METHOD_FOR_STAGE = {
    "direction": "diff_in_means",
    "sweep": "steering_sweep",
    "select": "subspace_search",
    "surgery": "direction_scale",
    "compare": "model_compare",
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


def _estimated_write_gb(source_model: str) -> Optional[float]:
    """Roughly how much a copy of this model will occupy.

    Local directories are measured. A Hugging Face id is not resolved over the
    network -- a preflight that can block on a download is worse than no
    estimate -- so the caller falls back to the flat low-disk threshold.
    """
    p = Path(source_model)
    if p.is_dir():
        total = sum(
            f.stat().st_size
            for pattern in ("*.safetensors", "*.bin")
            for f in p.glob(pattern)
            if f.is_file()
        )
        if total:
            return total / 2**30
    return None


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


def _direction_summary(row: Optional[WeightRun]) -> Dict[str, Any]:
    return ((row.meta_data or {}).get("summary") or {}) if row is not None else {}


def _preflight_checks(
    kind: str,
    source_model: str = "",
    direction_row: Optional[WeightRun] = None,
    output_name: Optional[str] = None,
) -> PreflightResponse:
    """Structured checks, so the UI can render severity and POST can verify acks.

    Built from `memory_report()` and `resident_ollama_models()` directly rather
    than by classifying `memory_warnings()`'s prose, which would couple this
    route to wording. Both are plain stdlib helpers with no torch import, so
    `weights/progress.py` is untouched and the CLI is unaffected.
    """
    from vivasecuris.aiasylum.weights.progress import (
        LOW_DISK_GB,
        memory_report,
        resident_ollama_models,
    )

    writes_weights = kind == "surgery"
    checks: List[PreflightCheck] = []

    def add(code, severity, message, acknowledgeable=True):
        checks.append(
            PreflightCheck(
                code=code, severity=severity, message=message,
                acknowledgeable=acknowledgeable,
            )
        )

    # Nothing else matters if the extra is missing.
    if not _interp_extra_installed():
        add(
            "interp_extra_missing", "blocking",
            'Needs the optional extra: pip install -e ".[interp]"',
            acknowledgeable=False,
        )

    mem = memory_report()
    sev = "blocking" if writes_weights else "advisory"

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

    needed_gb = _estimated_write_gb(source_model) if writes_weights else None
    if mem.get("free_gb") and needed_gb and mem["free_gb"] < needed_gb:
        add(
            "low_free_memory", sev,
            f"About {mem['free_gb']:.1f} GB free but roughly {needed_gb:.1f} GB needed.",
        )

    disk = {}
    if writes_weights:
        import shutil

        try:
            free_gb = shutil.disk_usage(_models_root().parent).free / 2**30
        except Exception:
            free_gb = None
        disk = {"free_gb": free_gb, "needed_gb": needed_gb, "threshold_gb": LOW_DISK_GB}
        floor = (needed_gb * 1.1) if needed_gb else LOW_DISK_GB
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
):
    if kind not in STAGES:
        raise HTTPException(status_code=400, detail=f"Unknown stage '{kind}'.")

    direction_row = None
    if source_run_id is not None:
        session = get_session()
        try:
            direction_row = (
                session.query(WeightRun).filter(WeightRun.id == source_run_id).first()
            )
        finally:
            session.close()

    return _preflight_checks(kind, source_model, direction_row, output_name)


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
                "artifacts_present": present,
                "has_sweep": row.id in sweeps,
            }
        )
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
            for r in session.query(WeightRun).filter(WeightRun.kind == "surgery").all()
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
            .filter(WeightRun.kind == "surgery", WeightRun.out_dir == str(path))
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

    raise ValueError(f"Unknown objective '{objective}'")


def _held_out_prompts(snap: Dict[str, Any], limit: int) -> List[str]:
    """The held-out positive prompts for this run's objective.

    A sweep and a select both measure a direction against the contrast it was
    fitted for, so both rebuild the parent's split rather than reaching for the
    default refusal corpus.
    """
    opts = snap.get("options") or {}
    split = _build_objective_split(
        snap.get("objective") or "refusal",
        snap.get("objective_config"),
        n_per_class=max(8, limit * 2),
        test_fraction=opts.get("test_fraction", 0.25),
        seed=opts.get("seed", 0),
    )
    return list(split.harmful_test[:limit])


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

    return _EventReporter()


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
        if rank > 1:
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

        rows = sweep_alpha(
            model, tok, d.vector, prompts,
            layer=d.layer,
            max_new_tokens=opts["max_new_tokens"],
            include_ablation=opts["include_ablation"],
            progress=report,
            capability_control=opts.get("capability_control", True),
            capability_limit=opts.get("capability_limit", 6),
            **({"alphas": opts["alphas"]} if opts.get("alphas") else {}),
        )
        summary = {"rows": rows, "layer": d.layer, **summarize_sweep(rows)}
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
        summary["elapsed"] = reporter.total_elapsed()
        return summary

    if kind == "compare":
        import gc

        from vivasecuris.aiasylum.weights.evaluate import (
            capability_set, compare_verdict, generate_greedy,
        )
        from vivasecuris.aiasylum.weights.manifest import SurgeryManifest
        from vivasecuris.aiasylum.weights.steering import _looks_degenerate, refusal_rate

        split = _build_objective_split(
            snap["objective"], snap.get("objective_config"),
            max(8, opts["n_prompts"] * 2), opts["test_fraction"], opts["seed"],
        )
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
            },
            "verdict": compare_verdict(b, m, opts.get("factual_floor", 0.05)),
            "capability_set": capability.name,
            "capability_n": capability.size,
            "manifest": (__import__("dataclasses").asdict(manifest) if manifest else None),
            "elapsed": reporter.total_elapsed(),
        }

    if kind == "surgery":
        from dataclasses import asdict

        from vivasecuris.aiasylum.weights.direction import RefusalDirection
        from vivasecuris.aiasylum.weights.manifest import SurgeryManifest
        from vivasecuris.aiasylum.weights.surgery import edit_and_save

        d = RefusalDirection.load(snap["direction_dir"])

        # Write to a staging sibling and publish by rename. A crash or a cancel
        # then never leaves a half-written directory at a name the models list
        # would show, and retrying the same name still works.
        staging = out_dir.parent / f"{STAGING_PREFIX}{run_id}"
        if staging.exists():
            import shutil

            shutil.rmtree(staging, ignore_errors=True)

        if opts.get("use_subspace"):
            reporter.note(
                f"editing {snap['source_model']}: removing the rank-{d.rank} subspace "
                f"at k={opts.get('k') or 1.0} (surgery runs on CPU)"
            )
        else:
            reporter.note(
                f"editing {snap['source_model']} with beta={opts['beta']} (surgery runs on CPU)"
            )
        reporter.note("writing weights -- several minutes for a 3B model, with no progress")
        try:
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
            )
            staging.rename(out_dir)
        except BaseException:
            import shutil

            if staging.name.startswith(STAGING_PREFIX) and _is_inside(
                staging, _models_root()
            ):
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
            }
            manifest.save(out_dir)

        summary = {
            "manifest": asdict(manifest) if manifest is not None else {},
            "output_path": str(out_dir),
            "size_bytes": _dir_size(out_dir),
            "elapsed": reporter.total_elapsed(),
        }
        return summary

    raise ValueError(f"Unknown kind '{kind}'")


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
                "notes": opts.get("notes"),
                "modified_model": (row.meta_data or {}).get("modified_model"),
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
            summary = await asyncio.to_thread(_execute, run_id, snapshot, reporter)

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
                "usable": summary.get("usable")}
    if kind == "sweep":
        return {"verdict": summary.get("verdict"),
                "ablate_delta_points": summary.get("ablate_delta_points"),
                "factual_delta_points": summary.get("factual_delta_points"),
                "any_degenerate": summary.get("any_degenerate")}
    if kind == "select":
        best = summary.get("best") or {}
        return {"best_rank": best.get("rank"), "best_k": best.get("k"),
                "admissible": sum(1 for r in summary.get("frontier", []) if r.get("accepted"))}
    if kind == "compare":
        d = summary.get("deltas", {})
        return {"refusal_delta": d.get("refuse_harmful"),
                "capability_delta": d.get("factual_acc")}
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

    if request.kind == "direction" and request.subspace_rank is not None:
        if request.subspace_rank < 1:
            raise HTTPException(status_code=400, detail="subspace_rank must be at least 1.")

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
    # `compare` measures two finished models, so it is the one stage with no
    # direction behind it.
    if request.kind in ("sweep", "select", "surgery"):
        direction_row = _resolve_direction(request)
        direction_dir = direction_row.out_dir
        # Snapshot rather than join: the child page must still render once the
        # parent row is gone, and RefusalDirection.load() drops layer_scores,
        # so a later re-read could not reconstruct this anyway.
        source_direction = _direction_summary(direction_row)

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

    out_dir: Optional[Path] = None
    if request.kind == "surgery":
        name = _validate_slug(request.output_name)
        out_dir = _resolve_output_dir(name)
        session = get_session()
        try:
            clash = (
                session.query(WeightRun)
                .filter(
                    WeightRun.kind == "surgery",
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
        request.kind, request.source_model, direction_row, request.output_name
    )
    unacknowledged = [
        c for c in checks.checks
        if c.severity == "blocking" and c.code not in request.acknowledge
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
                    "ranks": request.ranks,
                    "ks": request.ks,
                    "factual_floor": request.factual_floor,
                    "capability_set": request.capability_set,
                    "notes": request.notes,
                },
                "modified_model": (request.modified_model or "").strip() or None,
                "objective_config": objective_config,
                "direction_dir": direction_dir,
                "source_direction": source_direction,
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

        task = asyncio.create_task(_run_weights_background(row.id))
        weights_cancellation.register_task(row.id, task)

        return WeightRunResponse.from_orm_row(row)
    finally:
        session.close()


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
        kind, out_dir = row.kind, row.out_dir
    finally:
        session.close()

    removed_artifacts = False
    if delete_artifacts and out_dir:
        path = Path(out_dir)
        if kind == "surgery":
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
    role: str
    content: str


class ChatRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    messages: List[ChatMessage]
    system_prompt: Optional[str] = None
    # Greedy by default so what you see matches what the surgery measurements
    # were taken with; a sampled reply is not evidence about the edit.
    temperature: float = 0.0
    max_tokens: int = 256
    device: str = "auto"
    dtype: str = "bfloat16"


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

    if not _interp_extra_installed():
        raise HTTPException(
            status_code=409,
            detail='Needs the optional extra: pip install -e ".[interp]"',
        )

    history = [{"role": m.role, "content": m.content} for m in request.messages]

    def _turn() -> str:
        import asyncio as _asyncio

        from vivasecuris.aiasylum.models import get_provider

        provider = get_provider("transformers")
        model = provider.create_model(
            str(path),
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            device=request.device,
            dtype=request.dtype,
        )
        response = _asyncio.run(
            model.generate(
                prompt="", system_prompt=request.system_prompt, messages=history
            )
        )
        return response.content

    try:
        async with hold(f"chat with {path.name}"):
            content = await asyncio.to_thread(_turn)
    except Exception as exc:
        logger.exception("Chat turn failed for %s", path)
        raise HTTPException(status_code=500, detail=str(exc))

    from vivasecuris.aiasylum.weights.manifest import SurgeryManifest

    manifest = SurgeryManifest.load(path)
    return {
        "content": content,
        "model": str(path),
        "edited": manifest is not None,
        "manifest": (__import__("dataclasses").asdict(manifest) if manifest else None),
    }
