"""FastAPI main application."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}
