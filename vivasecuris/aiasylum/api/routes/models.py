"""API routes for model management (e.g. Ollama list/pull)."""

import logging
from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from vivasecuris.aiasylum.models.ollama import OllamaProvider

logger = logging.getLogger(__name__)

router = APIRouter()


class PullModelRequest(BaseModel):
    name: str


@router.get("/ollama", response_model=List[str])
async def list_ollama_models():
    """List Ollama models currently installed (from Ollama /api/tags)."""
    try:
        provider = OllamaProvider()
        models = await provider.list_available_models()
        return models
    except Exception as e:
        logger.warning(f"Failed to list Ollama models: {e}")
        raise HTTPException(status_code=502, detail=f"Could not reach Ollama: {e}")


@router.post("/ollama/pull")
async def pull_ollama_model(body: PullModelRequest):
    """Pull an Ollama model by name. Idempotent if already present."""
    model_name = (body.name or "").strip()
    if not model_name:
        raise HTTPException(status_code=400, detail="name is required")
    try:
        provider = OllamaProvider()
        result = await provider.pull_model(model_name)
        if result.get("error"):
            raise HTTPException(status_code=502, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"Failed to pull Ollama model {model_name}: {e}")
        raise HTTPException(status_code=502, detail=str(e))
