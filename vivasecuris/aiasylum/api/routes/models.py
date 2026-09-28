"""API routes for model management (e.g. Ollama list/pull)."""

import logging
import asyncio
import time
from typing import List, Literal, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from vivasecuris.aiasylum.api.model_downloads import get_manager
from vivasecuris.aiasylum.models.ollama import OllamaProvider
from vivasecuris.aiasylum.models.registry import list_providers

logger = logging.getLogger(__name__)

router = APIRouter()


class ModelChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=32000)


class ModelChatRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    provider: str = Field(min_length=1, max_length=80)
    model: str = Field(min_length=1, max_length=2048)
    messages: List[ModelChatMessage] = Field(max_length=100)
    system_prompt: Optional[str] = Field(None, max_length=32000)
    temperature: Optional[float] = Field(None, ge=0, le=2)
    top_p: Optional[float] = Field(None, gt=0, le=1)
    max_tokens: Optional[int] = Field(None, ge=1, le=32768)
    seed: Optional[int] = Field(None, ge=0, le=2**32 - 1)
    enable_cot: bool = False
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"


@router.post("/chat")
async def chat_with_model(request: ModelChatRequest):
    """One stateless turn through any registered provider, with optional ReACT."""
    from vivasecuris.aiasylum.api.model_chat import chat_response, close_chat_model, generate_chat, validate_chat
    from vivasecuris.aiasylum.api.model_jobs import hold
    from vivasecuris.aiasylum.models import get_provider
    from vivasecuris.aiasylum.models.registry import get_provider_info

    provider_name, model_name = request.provider.strip().lower(), request.model.strip()
    info = get_provider_info(provider_name)
    if info is None or not model_name:
        raise HTTPException(status_code=400, detail="A registered provider and nonempty model are required.")
    history = [m.model_dump() for m in request.messages]
    validate_chat(history, request.system_prompt)

    async def turn():
        started = time.perf_counter()
        provider = model = None
        try:
            provider = get_provider(provider_name)
            options = {key: getattr(request, key) for key in ("temperature", "max_tokens") if getattr(request, key) is not None}
            if info.name == "transformers":
                options.update(device=request.device, dtype=request.dtype)
            model = provider.create_model(model_name, **options)
            response, generation = await generate_chat(model, history, request)
            return chat_response(response, time.perf_counter() - started, generation)
        finally:
            await close_chat_model(model, provider)
            if info.name == "transformers":
                from vivasecuris.aiasylum.models.transformers_local import clear_cache
                clear_cache()

    try:
        if info.kind != "local":
            return await turn()
        # Cancellation cannot release the shared accelerator lease while the
        # blocking worker is still generating.
        async with hold(f"chat with {model_name}"):
            worker = asyncio.create_task(asyncio.to_thread(lambda: asyncio.run(turn())))
            cancelled = False
            while True:
                try:
                    result = await asyncio.shield(worker)
                    break
                except asyncio.CancelledError:
                    if worker.cancelled():
                        raise
                    cancelled = True
                except Exception:
                    if cancelled:
                        raise asyncio.CancelledError()
                    raise
            if cancelled:
                raise asyncio.CancelledError()
            return result
    except ImportError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Chat failed for %s/%s", provider_name, model_name)
        raise HTTPException(status_code=502, detail=str(exc))


class PullModelRequest(BaseModel):
    name: str


class DownloadRequest(BaseModel):
    repo_id: str
    revision: str = "main"


class DeleteCustomRequest(BaseModel):
    names: List[str]


@router.post("/custom/delete")
def delete_custom_models(body: DeleteCustomRequest):
    """Remove explicitly selected custom weights, preserving run history."""
    from vivasecuris.aiasylum.api.custom_checkpoints import delete_custom_checkpoints

    try:
        return delete_custom_checkpoints(body.names)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@router.get("/catalog")
def model_catalog():
    """Base and custom checkpoints visible to this API server, without downloads."""
    from vivasecuris.aiasylum.api.model_catalog import build_model_catalog

    return build_model_catalog()


@router.get("/discover")
def discover_models(q: str = Query(default="", max_length=80)):
    """Common suggestions or public Hub search; never download checkpoints."""
    from vivasecuris.aiasylum.api.model_discovery import discover_models as discover

    return discover(q)


@router.get("/huggingface/status")
def huggingface_status():
    """Report only whether this server has a token, without validating access."""
    from vivasecuris.aiasylum.api.model_discovery import huggingface_status as status

    return status()


@router.get("/lineage")
def model_lineage():
    """Recorded creation, analysis, and modification branches across origins."""
    from vivasecuris.aiasylum.api.model_lineage import build_model_lineage

    return build_model_lineage()


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
async def list_ollama_models(chat_only: bool = True):
    """List installed Ollama models (from /api/tags).

    By default only models that can chat: every picker that uses this list sends
    chat requests, and an embedding model there fails the run.
    """
    try:
        provider = OllamaProvider()
        models = await provider.list_available_models(chat_only=chat_only)
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


# -- Hugging Face checkpoints on this server ---------------------------------
# The server's Hub cache is what the `transformers` provider and every weights
# and interp run load from. Unlike an Ollama pull these are background jobs
# with progress and cancellation: on a GPU box one pull is tens of gigabytes.


@router.get("/downloads")
def list_downloads():
    """Every download this server process has run, newest first, plus cache disk space."""
    manager = get_manager()
    return {"downloads": manager.list(), **manager.disk()}


@router.post("/downloads", status_code=202)
def start_download(body: DownloadRequest):
    """Fetch a checkpoint into the server's cache. Returns at once; poll the download."""
    try:
        return get_manager().start(body.repo_id, body.revision)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/downloads/{download_id}")
def get_download(download_id: str):
    try:
        return get_manager().get(download_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.post("/downloads/{download_id}/cancel")
def cancel_download(download_id: str):
    """Stop the worker. Bytes already fetched stay, and a later download resumes from them."""
    try:
        return get_manager().cancel(download_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.delete("/cache/{repo_id:path}")
def delete_cached_model(repo_id: str):
    """Remove a cached base model from this server to free disk."""
    try:
        return get_manager().delete_cached(repo_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
