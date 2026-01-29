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
from vivasecuris.aiasylum.api.routes import test_runs, analysis, benchmarks, auth, prompts, suites

app = FastAPI(
    title="AI Asylum API",
    description="LLM Psychoanalysis Framework API",
    version="0.1.0",
)

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
app.include_router(prompts.router, prefix="/api/v1/prompts", tags=["prompts"])
app.include_router(suites.router, prefix="/api/v1/suites", tags=["suites"])


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
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Starting AI Asylum API...")
    logger.info(f"Project root: {project_root}")
    logger.info(f"Python path: {sys.path[:3]}")  # Show first 3 entries
    logger.info(f"Config loaded: database_url={settings.database_url[:30]}...")  # Show first 30 chars


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler."""
    import logging
    logger = logging.getLogger(__name__)
    logger.info("Shutting down AI Asylum API...")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}
