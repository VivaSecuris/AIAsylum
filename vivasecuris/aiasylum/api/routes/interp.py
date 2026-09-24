"""Mechanistic interpretability runs.

Drives the vendored engine in `vivasecuris.aiasylum.interp` and serves the
dashboards it produces.

Three deliberate departures from the test-run routes:

* **Its own progress and cancellation managers.** Both are keyed by bare
  integers, so sharing the global instances would make interp run 5 and test
  run 5 the same subscription.
* **A single model-job slot.** `worker_pool` has five slots shared with test
  runs; a model-loading interp job would hold one for its whole duration, and
  two concurrent jobs would fight over the same unified memory. The slot lives
  in `api/model_jobs.py` and is shared with the weights router.
* **A memory preflight that refuses.** Activation capture scales with the
  square of sequence length. Left alone it does not error, it swaps, and the
  job looks like a hang.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from vivasecuris.aiasylum.api.cancellation import CancellationManager
from vivasecuris.aiasylum.api.model_jobs import model_slot
from vivasecuris.aiasylum.api.progress_events import ProgressEventManager
from vivasecuris.aiasylum.constants import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_PENDING,
    STATUS_RUNNING,
)
from vivasecuris.aiasylum.database import InterpRun, get_session

logger = logging.getLogger(__name__)

router = APIRouter()

# Separate id spaces from test runs; see the module docstring.
interp_progress = ProgressEventManager()
interp_cancellation = CancellationManager()

# One model-heavy job at a time. Shared with the weights router rather than
# private to this one: an interp run and a weight-surgery run each holding
# their own slot would load two multi-GB models into the same unified memory,
# which is exactly the contention the slot exists to prevent. Aliased rather
# than wrapped so `_semaphore()` keeps returning the same memoised object.
_semaphore = model_slot


MODES = {
    "single": {
        "label": "Single prompt",
        "description": "One prompt through one model: layer trajectory, logit lens, attention, MLP.",
        "needs": ["model_a", "prompt_a"],
    },
    "comparison": {
        "label": "Prompt comparison",
        "description": "Two prompts through one model: where their representations diverge.",
        "needs": ["model_a", "prompt_a", "prompt_b"],
    },
    "progression": {
        "label": "Few-shot progression",
        "description": "N prompts through one model: how representations drift as examples are added.",
        "needs": ["model_a", "prompts"],
    },
    "model_diff": {
        "label": "Model comparison",
        "description": "One prompt through two models: what a weight edit actually changed.",
        "needs": ["model_a", "model_b", "prompt_a"],
    },
}

# Attention capture is O(layers x heads x seq^2). At 2048 tokens a 3B model
# needs over 12 GB for a single run, so the ceiling is well below the engine's
# own default of 2048.
MAX_LEN_DEFAULT = 512
MAX_LEN_CEILING = 1024

# The optional analyses, with what each one costs and how strong a claim it
# supports. The distinction matters: the circuit *cards* score components by
# how much they differ between two runs, which is correlational and uses fixed
# thresholds that were never validated against held-out data, while the minimal
# circuit scrubs components away and keeps what the behaviour actually needs.
# Presenting them as equally trustworthy would be the mistake.
ANALYSES = {
    "enable_attention_capture": {
        "label": "Attention capture",
        "description": "Per-head attention weights. Required for head-level circuit cards.",
        "cost": "O(layers x heads x seq^2) memory -- the dominant cost at long sequences.",
        "claim": "descriptive",
        "modes": ["single", "comparison", "progression"],
        "available": True,
    },
    "enable_mlp_capture": {
        "label": "MLP capture",
        "description": "Per-neuron activations. Required for neuron-level cards and safety-neuron clusters.",
        "cost": "O(layers x seq x d_ffn) memory.",
        "claim": "descriptive",
        "modes": ["single", "comparison", "progression"],
        "available": True,
    },
    "enable_qkv_capture": {
        "label": "Q/K/V capture",
        "description": "Per-layer queries, keys and values, for OV/QK circuit analysis.",
        "cost": "Heavier than attention capture; adds three tensors per layer.",
        "claim": "descriptive",
        # Only the two-prompt comparison service reads this.
        "modes": ["comparison"],
        "available": True,
    },
    "enable_pre_mlp_capture": {
        "label": "Pre-MLP capture",
        "description": "Intermediate activations before the MLP, for residual decomposition.",
        "cost": "Heavier memory; another full activation tensor per layer.",
        "claim": "descriptive",
        "modes": ["comparison"],
        # The capture hook is a stub that returns None (hook_registry.py), so
        # pre-MLP activations are always empty. That also silently caps the
        # minimal-circuit search to attention heads, never neurons. Offering a
        # checkbox for it would promise something the engine cannot deliver.
        "available": False,
        "unavailable_reason": (
            "The pre-MLP capture hook is not implemented, so this would record "
            "nothing. It also limits minimal-circuit search to heads only."
        ),
    },
    "enable_patching": {
        "label": "Activation patching",
        "description": "Substitute activations from one run into the other and re-measure.",
        "cost": "One extra forward pass per patched location.",
        "claim": "causal",
        "modes": ["comparison"],
        "available": True,
    },
    "enable_scrub": {
        "label": "Causal scrubbing",
        "description": "Ablate components and measure how much of the behaviour survives.",
        "cost": "Many forward passes.",
        "claim": "causal",
        "modes": ["comparison"],
        # The service calls run_scrub_experiment with components_to_scrub=[],
        # and an empty list keeps every component -- so what it reports is
        # residual reconstruction error, not the effect of an ablation.
        "available": False,
        "unavailable_reason": (
            "The scrub runs with an empty component list, so it measures "
            "reconstruction error rather than an ablation effect. Use the "
            "minimal circuit search for a causal result."
        ),
    },
    "enable_minimal_circuit": {
        "label": "Minimal circuit search",
        "description": (
            "Greedy search for the smallest set of components that still reproduces "
            "the behaviour. The strongest claim available here."
        ),
        "cost": (
            "Slowest analysis in the engine: a greedy sweep over components. "
            "Searches attention heads only, since pre-MLP capture is unimplemented."
        ),
        "claim": "causal",
        "modes": ["comparison"],
        "available": True,
    },
}

# Served as JSON so the UI can render findings natively instead of only inside
# the generated dashboard. Whitelisted by name, and only ever .json: the .npy
# siblings are raw matrices that belong in the dashboard, not in a React table.
ARTIFACT_WHITELIST = {
    "meta.json",
    "pca_payload.json",
    "predictions.json",
    "patching_results.json",
    "attention_payload.json",
    "mlp_payload.json",
    "circuit_payload.json",
    "temporal_payload.json",
    "attribution_payload.json",
    "logit_attribution_payload.json",
    "ov_qk_payload.json",
    "scrub_payload.json",
    "minimal_circuit_payload.json",
    "example_impact.json",
}

RUNS_ROOT = Path("runs/interp")


class InterpRunRequest(BaseModel):
    # model_a/model_b collide with pydantic's protected "model_" prefix.
    model_config = {"protected_namespaces": ()}

    mode: str = Field(..., description="single | comparison | progression | model_diff")
    model_a: str
    model_b: Optional[str] = None
    prompt_a: Optional[str] = None
    prompt_b: Optional[str] = None
    prompts: Optional[List[str]] = None
    device: str = "auto"
    dtype: str = "bfloat16"
    max_len: int = MAX_LEN_DEFAULT
    window: int = 128
    topk: int = 10
    dim_reduction: str = "pca"
    enable_attention_capture: bool = False
    enable_mlp_capture: bool = False
    # The causal tools. Implemented in the engine but previously unreachable
    # from here, so `find_minimal_circuit_greedy` -- the only analysis making a
    # causal rather than correlational claim -- could not be run from the app.
    # All default off: each one multiplies capture cost or search time.
    enable_qkv_capture: bool = False
    enable_pre_mlp_capture: bool = False
    enable_patching: bool = False
    enable_scrub: bool = False
    enable_minimal_circuit: bool = False


class InterpRunResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    id: int
    mode: str
    status: str
    model_a: str
    model_b: Optional[str]
    prompt_a: Optional[str]
    prompt_b: Optional[str]
    prompts: Optional[List[str]]
    out_dir: Optional[str]
    error: Optional[str]
    created_at: Optional[datetime]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    metadata: Dict[str, Any] = {}

    @classmethod
    def from_orm_row(cls, row: InterpRun) -> "InterpRunResponse":
        return cls(
            id=row.id,
            mode=row.mode,
            status=row.status,
            model_a=row.model_a,
            model_b=row.model_b,
            prompt_a=row.prompt_a,
            prompt_b=row.prompt_b,
            prompts=row.prompts,
            out_dir=row.out_dir,
            error=row.error,
            created_at=row.created_at,
            started_at=row.started_at,
            completed_at=row.completed_at,
            metadata=row.meta_data or {},
        )


def _validate(request: InterpRunRequest) -> None:
    """Reject a request before anything expensive is scheduled."""
    spec = MODES.get(request.mode)
    if spec is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown mode '{request.mode}'. Choose from: {', '.join(MODES)}",
        )

    missing = [
        field for field in spec["needs"]
        if not getattr(request, field, None)
        or (field == "prompts" and len(request.prompts or []) < 2)
    ]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Mode '{request.mode}' requires: {', '.join(missing)}",
        )

    if request.max_len > MAX_LEN_CEILING:
        raise HTTPException(
            status_code=400,
            detail=(
                f"max_len {request.max_len} exceeds the ceiling of {MAX_LEN_CEILING}. "
                f"Activation capture grows with the square of sequence length: at 2048 "
                f"tokens a 3B model needs over 12 GB for one run. Lower max_len, or "
                f"disable attention capture and analyze a shorter window."
            ),
        )
    if request.max_len < 8:
        raise HTTPException(status_code=400, detail="max_len must be at least 8")

    # Refuse a flag rather than accepting it and quietly dropping it. Only the
    # two-prompt comparison service reads the causal analyses; model_diff,
    # single and progression route to services that never look at them, so a
    # request that asks for a minimal circuit on model_diff would run for
    # minutes and produce nothing, with no way to tell that from a failure.
    for name, spec in ANALYSES.items():
        if not getattr(request, name, False):
            continue
        if not spec.get("available", True):
            raise HTTPException(
                status_code=409,
                detail=f"{spec['label']} is unavailable. {spec.get('unavailable_reason', '')}".strip(),
            )
        modes = spec.get("modes")
        if modes and request.mode not in modes:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"{spec['label']} is only computed for "
                    f"{' / '.join(modes)} runs, not '{request.mode}'. It would be "
                    f"captured and then discarded."
                ),
            )


def _preflight() -> List[str]:
    """Memory warnings from the same helper the weights CLI uses."""
    try:
        from vivasecuris.aiasylum.weights.progress import memory_warnings

        return memory_warnings(needed_gb=4.0)
    except Exception as exc:
        logger.debug("Memory preflight unavailable: %s", exc)
        return []


def _build_config(row: InterpRun, out_dir: Path):
    """Translate a stored run into an engine Config.

    Built directly rather than through `Config.from_api_request`, which
    hardcodes max_len=2048 and forces every capture flag on -- the combination
    this route exists to prevent.
    """
    from vivasecuris.aiasylum.interp.core.config import Config

    opts = (row.meta_data or {}).get("options", {})
    captures = bool(opts.get("enable_attention_capture") or opts.get("enable_mlp_capture"))

    return Config(
        model=row.model_a,
        out_dir=out_dir,
        analysis_mode="comparison" if row.mode == "model_diff" else row.mode,
        prompt=row.prompt_a or "",
        prompt_a=row.prompt_a or "",
        prompt_b=row.prompt_b or "",
        prompts=row.prompts,
        device=opts.get("device", "auto"),
        dtype=opts.get("dtype", "bfloat16"),
        max_len=int(opts.get("max_len", MAX_LEN_DEFAULT)),
        window=int(opts.get("window", 128)),
        topk=int(opts.get("topk", 10)),
        dim_reduction=opts.get("dim_reduction", "pca"),
        enable_attention_capture=bool(opts.get("enable_attention_capture", False)),
        enable_mlp_capture=bool(opts.get("enable_mlp_capture", False)),
        enable_attn_output_capture=captures,
        enable_component_analysis=captures,
        enable_qkv_capture=bool(opts.get("enable_qkv_capture", False)),
        enable_pre_mlp_capture=bool(opts.get("enable_pre_mlp_capture", False)),
        enable_patching=bool(opts.get("enable_patching", False)),
        enable_scrub=bool(opts.get("enable_scrub", False)),
        enable_minimal_circuit=bool(opts.get("enable_minimal_circuit", False)),
        seed=0,
    )


def _execute(run_id: int, row_snapshot: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    """The synchronous, GPU-bound body. Always called in a worker thread."""
    from vivasecuris.aiasylum.interp.core.orchestrator import AnalysisOrchestrator

    mode = row_snapshot["mode"]
    config = row_snapshot["config"]

    if mode == "model_diff":
        from vivasecuris.aiasylum.interp.core.services.model_comparison_service import (
            ModelComparisonService,
        )
        from vivasecuris.aiasylum.models.transformers_local import _get_cached, clear_cache

        device = config.device
        dtype = config.dtype
        model_a, model_b = row_snapshot["model_a"], row_snapshot["model_b"]

        result = ModelComparisonService.run_model_diff(
            config,
            load_a=lambda: _get_cached(model_a, device, dtype),
            load_b=lambda: _get_cached(model_b, device, dtype),
            model_a_id=model_a,
            model_b_id=model_b,
            # Both models never resident at once: the cache is the only thing
            # holding them, so dropping it is what frees the first.
            release=clear_cache,
            progress=row_snapshot["progress"],
        )
        ModelComparisonService.save_results(result, out_dir)
        html = AnalysisOrchestrator.build_dashboard(result)
        (out_dir / "dashboard.html").write_text(html)
    else:
        result, html = AnalysisOrchestrator.run_and_save(config)

    meta = dict(getattr(result, "meta", {}) or {})
    return {
        "spike_layer": meta.get("spike_layer"),
        "num_layers": meta.get("num_layers"),
        "window_len": meta.get("window_len"),
        "shared_unembedding": meta.get("shared_unembedding"),
        "dashboard_bytes": len(html),
    }


async def _run_interp_background(run_id: int) -> None:
    """Own semaphore, own managers; never touches the test-run worker pool."""
    async with _semaphore():
        session = get_session()
        try:
            row = session.query(InterpRun).filter(InterpRun.id == run_id).first()
            if row is None or interp_cancellation.is_cancelled(run_id):
                return

            out_dir = RUNS_ROOT / str(run_id)
            out_dir.mkdir(parents=True, exist_ok=True)

            row.status = STATUS_RUNNING
            row.started_at = datetime.utcnow()
            row.out_dir = str(out_dir)
            session.commit()

            await interp_progress.emit_event(
                run_id, "interp_started", {"status": STATUS_RUNNING, "mode": row.mode},
                f"Starting {row.mode} analysis",
            )

            loop = asyncio.get_running_loop()

            def report(message: str) -> None:
                # Called from the worker thread, so hop back to the loop.
                asyncio.run_coroutine_threadsafe(
                    interp_progress.emit_event(
                        run_id, "interp_progress", {"step": message}, message
                    ),
                    loop,
                )

            snapshot = {
                "mode": row.mode,
                "model_a": row.model_a,
                "model_b": row.model_b,
                "config": _build_config(row, out_dir),
                "progress": report,
            }

            summary = await asyncio.to_thread(_execute, run_id, snapshot, out_dir)

            row = session.query(InterpRun).filter(InterpRun.id == run_id).first()
            row.status = STATUS_COMPLETED
            row.completed_at = datetime.utcnow()
            row.meta_data = {**(row.meta_data or {}), "summary": summary}
            session.commit()

            await interp_progress.emit_event(
                run_id, "interp_completed",
                {"status": STATUS_COMPLETED, **summary}, "Analysis complete",
            )

        except asyncio.CancelledError:
            _fail(session, run_id, "Cancelled", cancelled=True)
            await interp_progress.emit_event(
                run_id, "interp_cancelled", {"status": STATUS_FAILED, "cancelled": True},
                "Analysis was cancelled",
            )
            raise
        except Exception as exc:
            logger.exception("Interp run %s failed", run_id)
            _fail(session, run_id, str(exc))
            await interp_progress.emit_event(
                run_id, "interp_failed", {"status": STATUS_FAILED, "error": str(exc)},
                f"Analysis failed: {exc}",
            )
        finally:
            session.close()
            interp_cancellation.unregister_task(run_id)


def _fail(session, run_id: int, error: str, cancelled: bool = False) -> None:
    try:
        row = session.query(InterpRun).filter(InterpRun.id == run_id).first()
        if row is not None:
            row.status = STATUS_FAILED
            row.error = error
            row.completed_at = datetime.utcnow()
            if cancelled:
                row.meta_data = {**(row.meta_data or {}), "cancelled": True}
            session.commit()
    except Exception:
        session.rollback()
        logger.exception("Could not record failure for interp run %s", run_id)


@router.get("/modes")
async def list_modes():
    """The analysis modes, their required fields, and the memory limits."""
    return {
        "modes": [{"name": name, **spec} for name, spec in MODES.items()],
        "analyses": [{"name": k, **v} for k, v in ANALYSES.items()],
        "limits": {
            "max_len_default": MAX_LEN_DEFAULT,
            "max_len_ceiling": MAX_LEN_CEILING,
            "note": (
                "Activation capture grows with the square of sequence length. "
                "Attention capture is off by default for the same reason."
            ),
        },
        "warnings": _preflight(),
    }


@router.post("/runs", response_model=InterpRunResponse)
async def create_interp_run(request: InterpRunRequest):
    """Create an interpretability run and start it in the background."""
    _validate(request)

    session = get_session()
    try:
        row = InterpRun(
            mode=request.mode,
            status=STATUS_PENDING,
            model_a=request.model_a,
            model_b=request.model_b,
            prompt_a=request.prompt_a,
            prompt_b=request.prompt_b,
            prompts=request.prompts,
            meta_data={
                "options": {
                    "device": request.device,
                    "dtype": request.dtype,
                    "max_len": request.max_len,
                    "window": request.window,
                    "topk": request.topk,
                    "dim_reduction": request.dim_reduction,
                    "enable_attention_capture": request.enable_attention_capture,
                    "enable_mlp_capture": request.enable_mlp_capture,
                    "enable_qkv_capture": request.enable_qkv_capture,
                    "enable_pre_mlp_capture": request.enable_pre_mlp_capture,
                    "enable_patching": request.enable_patching,
                    "enable_scrub": request.enable_scrub,
                    "enable_minimal_circuit": request.enable_minimal_circuit,
                },
                "preflight_warnings": _preflight(),
            },
        )
        session.add(row)
        session.commit()
        session.refresh(row)

        task = asyncio.create_task(_run_interp_background(row.id))
        interp_cancellation.register_task(row.id, task)

        return InterpRunResponse.from_orm_row(row)
    finally:
        session.close()


@router.get("/runs", response_model=List[InterpRunResponse])
async def list_interp_runs(limit: int = 50, offset: int = 0, mode: Optional[str] = None):
    session = get_session()
    try:
        query = session.query(InterpRun)
        if mode:
            query = query.filter(InterpRun.mode == mode)
        rows = query.order_by(InterpRun.id.desc()).offset(offset).limit(limit).all()
        return [InterpRunResponse.from_orm_row(r) for r in rows]
    finally:
        session.close()


@router.get("/runs/{run_id}", response_model=InterpRunResponse)
async def get_interp_run(run_id: int):
    session = get_session()
    try:
        row = session.query(InterpRun).filter(InterpRun.id == run_id).first()
        if row is None:
            raise HTTPException(status_code=404, detail="Interp run not found")
        return InterpRunResponse.from_orm_row(row)
    finally:
        session.close()


@router.delete("/runs/{run_id}")
async def delete_interp_run(run_id: int):
    """Delete a run and the artifacts it wrote."""
    import shutil

    session = get_session()
    try:
        row = session.query(InterpRun).filter(InterpRun.id == run_id).first()
        if row is None:
            raise HTTPException(status_code=404, detail="Interp run not found")

        out_dir = row.out_dir
        session.delete(row)
        session.commit()

        # Only ever remove a directory this route created.
        if out_dir:
            path = Path(out_dir).resolve()
            if path.is_dir() and RUNS_ROOT.resolve() in path.parents:
                shutil.rmtree(path, ignore_errors=True)

        return {"deleted": run_id}
    finally:
        session.close()


@router.post("/runs/{run_id}/stop")
async def stop_interp_run(run_id: int):
    session = get_session()
    try:
        row = session.query(InterpRun).filter(InterpRun.id == run_id).first()
        if row is None:
            raise HTTPException(status_code=404, detail="Interp run not found")
        if row.status not in (STATUS_PENDING, STATUS_RUNNING):
            return {"stopped": False, "status": row.status, "reason": "not running"}
    finally:
        session.close()

    interp_cancellation.cancel(run_id)
    return {"stopped": True, "run_id": run_id}


@router.get("/runs/{run_id}/progress")
async def stream_interp_progress(run_id: int):
    """Server-Sent Events, mirroring the test-run progress stream."""
    session = get_session()
    try:
        row = session.query(InterpRun).filter(InterpRun.id == run_id).first()
        if row is None:
            raise HTTPException(status_code=404, detail="Interp run not found")
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
        async for event in interp_progress.stream_events(run_id):
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


@router.get("/runs/{run_id}/dashboard", response_class=HTMLResponse)
async def get_interp_dashboard(run_id: int):
    """Serve the generated dashboard, for embedding in an iframe."""
    session = get_session()
    try:
        row = session.query(InterpRun).filter(InterpRun.id == run_id).first()
        if row is None:
            raise HTTPException(status_code=404, detail="Interp run not found")
        status, out_dir = row.status, row.out_dir
    finally:
        session.close()

    if status != STATUS_COMPLETED or not out_dir:
        raise HTTPException(
            status_code=409,
            detail=f"Run {run_id} is '{status}'; a dashboard exists only once it completes.",
        )

    path = Path(out_dir) / "dashboard.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"No dashboard at {path}")

    return HTMLResponse(content=path.read_text(), status_code=200)


@router.get("/runs/{run_id}/artifacts")
async def list_interp_artifacts(run_id: int):
    """Which JSON payloads this run actually produced.

    Most are conditional on a capture flag, so the UI needs to ask rather than
    assume: a run without `enable_mlp_capture` has no neuron-level circuit data
    and should not offer a tab that would render empty.
    """
    session = get_session()
    try:
        row = session.query(InterpRun).filter(InterpRun.id == run_id).first()
        if row is None:
            raise HTTPException(status_code=404, detail="Interp run not found")
        out_dir = row.out_dir
    finally:
        session.close()

    if not out_dir:
        return {"run_id": run_id, "artifacts": []}

    base = Path(out_dir)
    found = []
    for name in sorted(ARTIFACT_WHITELIST):
        path = base / name
        if path.exists():
            found.append({"name": name, "bytes": path.stat().st_size})
    return {"run_id": run_id, "artifacts": found}


@router.get("/runs/{run_id}/artifacts/{name}")
async def get_interp_artifact(run_id: int, name: str):
    """Serve one JSON payload, so findings can be rendered natively.

    Whitelisted by exact name and confined to the run's own directory. The
    whitelist is doing the real work here -- the containment check alone would
    still allow reading any file the engine happened to leave in `out_dir`.
    """
    if name not in ARTIFACT_WHITELIST:
        raise HTTPException(
            status_code=404,
            detail=f"'{name}' is not a served artifact. Known: {sorted(ARTIFACT_WHITELIST)}",
        )

    session = get_session()
    try:
        row = session.query(InterpRun).filter(InterpRun.id == run_id).first()
        if row is None:
            raise HTTPException(status_code=404, detail="Interp run not found")
        out_dir = row.out_dir
    finally:
        session.close()

    if not out_dir:
        raise HTTPException(status_code=404, detail=f"Run {run_id} wrote no artifacts.")

    base = Path(out_dir).resolve()
    path = (base / name).resolve()
    if base != path.parent:
        raise HTTPException(status_code=400, detail="Refusing to read outside the run directory.")
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Run {run_id} produced no {name}; the analysis that writes it was off.",
        )

    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"{name} is not valid JSON: {exc}")
