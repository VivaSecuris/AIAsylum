"""Shared stateless generation for general and checkpoint chat endpoints."""

from copy import deepcopy
import inspect
import logging

from fastapi import HTTPException

logger = logging.getLogger(__name__)


def validate_chat(history, system_prompt):
    if not history:
        raise HTTPException(status_code=400, detail="messages must not be empty.")
    if history[-1]["role"] != "user":
        raise HTTPException(status_code=400, detail="The final chat message must be a user turn.")
    if any(not message["content"].strip() for message in history):
        raise HTTPException(status_code=400, detail="Chat turns must contain text.")
    if sum(len(message["content"]) for message in history) + len(system_prompt or "") > 128000:
        raise HTTPException(status_code=400, detail="Conversation is too large. Start a new conversation or shorten the history.")


async def generate_chat(model, history, request):
    """Keep the selected persona exact and apply only the requested ReACT wrapper."""
    from vivasecuris.aiasylum.cot import ReACTReasoner

    messages = deepcopy(history)
    if request.system_prompt:
        messages.insert(0, {"role": "system", "content": request.system_prompt})
    requested = {key: getattr(request, key, None) for key in ("temperature", "top_p", "max_tokens", "seed")}
    requested["enable_cot"] = request.enable_cot
    sent = {key: value for key, value in requested.items() if key != "enable_cot" and value is not None}
    # ReACT's context helper otherwise falls back to 0.7 when any override is
    # present. Resolve the selected model's defaults before going through it.
    sent.setdefault("temperature", getattr(model, "temperature", 0.7))
    sent.setdefault("max_tokens", getattr(model, "max_tokens", 4096))
    notes = []
    if "seed" in sent and not getattr(model, "supports_seed", True):
        sent.pop("seed")
        notes.append("This provider does not support generation seeds; the requested seed was omitted.")
    if request.enable_cot:
        response = await ReACTReasoner(model).reason(
            prompt=messages[-1]["content"], messages=messages[:-1], gen_overrides=sent,
        )
    else:
        response = await model.generate(prompt="", messages=messages, **sent)
    response.metadata = dict(response.metadata or {})
    response.metadata.setdefault("request_system_prompts", [m["content"] for m in messages if m["role"] == "system"])
    response.metadata.setdefault("request_system_prompts_source", "model_input")
    response.metadata["framework_cot_enabled"] = request.enable_cot
    applied = {**sent, "enable_cot": request.enable_cot}
    if requested["seed"] is not None and "seed" not in sent:
        applied["seed"] = None
    for key, value in (response.metadata.get("sampling") or {}).items():
        applied[key] = value
        if key in sent and value is None:
            notes.append(f"The selected model does not accept {key} in this request; it was omitted.")
    return response, {"requested": requested, "applied": applied, "notes": notes}


async def close_chat_model(model, provider=None):
    """Chat factories are per request; release their HTTP transports afterwards."""
    close = getattr(model, "close", None)
    if close is None:
        client = getattr(model, "client", None) or getattr(provider, "client", None)
        close = getattr(client, "aclose", None) or getattr(client, "close", None)
    if callable(close):
        try:
            result = close()
            if inspect.isawaitable(result):
                await result
        except Exception:
            logger.warning("Could not close chat transport", exc_info=True)


def chat_response(response, elapsed, generation):
    metadata = response.metadata or {}
    return {
        "content": response.content, "reasoning": metadata.get("reasoning"),
        "reasoning_source": metadata.get("reasoning_source"),
        "model": response.model, "provider": response.provider,
        "finish_reason": response.finish_reason, "usage": response.usage or {},
        "metadata": metadata, "elapsed_seconds": elapsed, "generation": generation,
    }
