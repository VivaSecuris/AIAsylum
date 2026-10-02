"""FastAPI main application."""

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root is in Python path for config imports
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config import settings
from vivasecuris.aiasylum.api.routes import (
    test_runs,
    analysis,
    benchmarks,
    benchmark_campaigns,
    auth,
    prompts,
    suites,
    models as models_router,
    model_organization,
    model_adjustments,
    conversations as conversations_router,
    config as config_router,
    interp,
    weights,
)

app = FastAPI(
    title="AI Asylum API",
    description="LLM Psychoanalysis Framework API",
    version="0.1.0",
)

# Auth before CORS in code order means CORS wraps it, so a 401 still carries
# the CORS headers the browser needs to read it.
from vivasecuris.aiasylum.api.security import check_config, make_auth_middleware

check_config(settings.require_auth, settings.api_keys_list, settings.api_key_hmac_secret)
app.middleware("http")(make_auth_middleware(settings))

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(test_runs.router, prefix="/api/v1/test-runs", tags=["test-runs"])
app.include_router(analysis.router, prefix="/api/v1/analysis", tags=["analysis"])
app.include_router(benchmarks.router, prefix="/api/v1/benchmarks", tags=["benchmarks"])
app.include_router(benchmark_campaigns.router, prefix="/api/v1/benchmark-campaigns", tags=["benchmarks"])
app.include_router(prompts.router, prefix="/api/v1/prompts", tags=["prompts"])
app.include_router(suites.router, prefix="/api/v1/suites", tags=["suites"])
app.include_router(models_router.router, prefix="/api/v1/models", tags=["models"])
app.include_router(model_adjustments.router, prefix="/api/v1/models/adjustments", tags=["models"])
app.include_router(model_organization.router, prefix="/api/v1/model-organization", tags=["models"])
app.include_router(conversations_router.router, prefix="/api/v1/conversations", tags=["conversations"])
app.include_router(config_router.router, prefix="/api/v1/config", tags=["config"])
app.include_router(interp.router, prefix="/api/v1/interp", tags=["interp"])
app.include_router(weights.router, prefix="/api/v1/weights", tags=["weights"])

# Interpretability dashboards load plotly.js from here rather than a CDN, so
# they render on machines with no outbound network. Mounted only when the
# optional interp extra is installed.
def _mount_static() -> None:
    try:
        from fastapi.staticfiles import StaticFiles

        from vivasecuris.aiasylum.interp.visualization.assets import plotly_asset_path

        asset = plotly_asset_path()
        if asset is None:
            return
        app.mount("/static", StaticFiles(directory=str(asset.parent)), name="static")
    except Exception as exc:  # pragma: no cover - optional dependency
        import logging

        logging.getLogger(__name__).info(
            "Static assets not mounted (%s); dashboards will fall back to the CDN", exc
        )


_mount_static()



@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "AI Asylum API",
        "version": "0.1.0",
        "status": "running",
    }


@app.on_event("startup")
async def startup_event():
    """Startup event handler."""
    import asyncio
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Starting AI Asylum API...")
    logger.info(f"Project root: {project_root}")
    logger.info(f"Python path: {sys.path[:3]}")  # Show first 3 entries
    logger.info(f"Config loaded: database_url={settings.database_url[:30]}...")  # Show first 30 chars
    from vivasecuris.aiasylum.database.models import ModelAdjustment
    from vivasecuris.aiasylum.database.session import engine
    ModelAdjustment.__table__.create(bind=engine, checkfirst=True)
    from vivasecuris.aiasylum.api.adjustment_recovery import recover_interrupted_adjustments
    recover_interrupted_adjustments()
    from vivasecuris.aiasylum.api.benchmark_campaigns import recover_interrupted_campaigns
    recover_interrupted_campaigns()
    from vivasecuris.aiasylum.api.job_recovery import (
        capture_interrupted_model_jobs,
        log_recovery_result,
        recover_interrupted_model_jobs,
    )
    snapshot = capture_interrupted_model_jobs()
    task = asyncio.create_task(recover_interrupted_model_jobs(snapshot))
    task.add_done_callback(log_recovery_result)
    app.state.model_job_recovery = task


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler."""
    import asyncio
    from contextlib import suppress
    import logging
    logger = logging.getLogger(__name__)
    logger.info("Shutting down AI Asylum API...")
    recovery = getattr(app.state, "model_job_recovery", None)
    if recovery is not None and not recovery.done():
        recovery.cancel()
        with suppress(asyncio.CancelledError):
            await recovery


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}
