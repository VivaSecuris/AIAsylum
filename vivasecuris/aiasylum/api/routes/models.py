"""API routes for model management (e.g. Ollama list/pull)."""

import logging
from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from vivasecuris.aiasylum.models.ollama import OllamaProvider
from vivasecuris.aiasylum.models.registry import list_providers

logger = logging.getLogger(__name__)

router = APIRouter()


class PullModelRequest(BaseModel):
    name: str


@router.get("/providers")
async def list_model_providers():
    """Every provider the backend can create, with how to pick a model for it.

    The UI reads this instead of hard-coding a list, which is how `servus` and
    `agentic` ended up registered but unreachable from the app.
    """
    from config import settings

    out = []
    for info in list_providers():
        data = info.to_dict()
        # Report configuration state so the UI can grey out what cannot work.
        if info.requires_api_key:
            data["configured"] = bool(getattr(settings, info.requires_api_key, None))
        else:
            data["configured"] = True
        if info.requires_extra:
            try:
                import torch  # noqa: F401
                import transformers  # noqa: F401
                data["available"] = True
            except ImportError:
                data["available"] = False
                data["unavailable_reason"] = (
                    f'Needs the optional extra: pip install -e ".[{info.requires_extra}]"'
                )
        else:
            data["available"] = True
        out.append(data)
    return out


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
