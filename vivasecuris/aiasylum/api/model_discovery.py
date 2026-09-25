"""Public Hub discovery, separate from checkpoint inventory and downloads.

Discovery never sends the server's Hugging Face credential. Public and gated
repository metadata is public; listing it does not prove access or runtime
compatibility. An empty search works offline using the bundled common list.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import httpx

from config import settings

MAX_RESULTS = 20
MAX_QUERY_LENGTH = 80
HUB_SEARCH_URL = "https://huggingface.co/api/models"
_REPO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}/[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")
_SPECIALIZED_FORMAT = re.compile(
    r"(?:^|[^a-z0-9])(?:gguf|ggml|mlx|onnx|gptq|awq|exl2|exllama(?:v2)?|"
    r"bitsandbytes|bnb(?:[-_]?[48][-_]?bit)?|[248][-_]?bit|int[248]|nf4|fp8)(?:$|[^a-z0-9])",
    re.IGNORECASE,
)
_SPECIALIZED_TAGS = {"gguf", "ggml", "mlx", "gptq", "awq", "exl2", "exllama", "exllamav2", "bitsandbytes"}


def _searchable_checkpoint(item: Any) -> bool:
    """Hide obvious alternate formats, even when Hub metadata says Transformers.

    Conversion repos commonly inherit the original model's task and library.
    Names and format tags therefore matter too. This is a discovery filter,
    not a guarantee that every retained checkpoint passes runtime preflight.
    """
    if not isinstance(item, dict):
        return False
    if item.get("library_name", "transformers") != "transformers" or item.get("pipeline_tag", "text-generation") != "text-generation":
        return False
    repo_id = item.get("id")
    if not isinstance(repo_id, str) or _SPECIALIZED_FORMAT.search(repo_id):
        return False
    tags = item.get("tags")
    if isinstance(tags, list) and any(isinstance(tag, str) and tag.casefold() in _SPECIALIZED_TAGS for tag in tags):
        return False
    # An `onnx` tag alone is not an exclusion: official native checkpoints
    # such as SmolLM2-135M-Instruct also ship optional ONNX exports.
    return True


def _family(repo_id: str) -> str:
    name = repo_id.lower()
    for needle, family in (("smollm", "SmolLM"), ("qwen", "Qwen"), ("llama", "Llama"),
                           ("gemma", "Gemma"), ("mistral", "Mistral"), ("phi", "Phi")):
        if needle in name:
            return family
    return repo_id.split("/", 1)[0]


def _model(item: Any, source: str) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    repo_id = item.get("id", item.get("repo_id"))
    if not isinstance(repo_id, str) or not _REPO_ID.fullmatch(repo_id) or ".." in repo_id or "--" in repo_id:
        return None
    gated = item.get("gated", False)
    if not isinstance(gated, bool) and gated not in ("auto", "manual"):
        gated = False
    tags = item.get("tags", [])
    tags = [tag[:160] for tag in tags if isinstance(tag, str)][:40] if isinstance(tags, list) else []
    downloads = item.get("downloads")
    family = item.get("family")
    return {
        "id": repo_id, "name": repo_id, "family": family if isinstance(family, str) and family else _family(repo_id),
        "gated": gated, "private": bool(item.get("private", False)),
        "downloads": downloads if type(downloads) is int and downloads >= 0 else None,
        "tags": tags, "source": source,
    }


def common_models(path: Path | None = None) -> list[dict[str, Any]]:
    path = path or settings.project_root / "config" / "common_models.json"
    try:
        payload = json.loads(path.read_text())
    except (OSError, UnicodeError, ValueError):
        return []
    rows = payload.get("models", []) if isinstance(payload, dict) else []
    seen = set()
    models = []
    for row in rows if isinstance(rows, list) else []:
        model = _model(row, "curated")
        if model and model["id"] not in seen:
            seen.add(model["id"])
            models.append(model)
    return models[:MAX_RESULTS]


def discover_models(query: str = "") -> dict[str, Any]:
    if len(query) > MAX_QUERY_LENGTH:
        raise ValueError(f"Search must be at most {MAX_QUERY_LENGTH} characters.")
    query = query.strip()
    common = common_models()
    if not query:
        return {"models": common, "source": "curated"}
    matching_common = [row for row in common if query.casefold() in f'{row["id"]} {row["family"]}'.casefold()]

    # Use the fixed public endpoint instead of the token-aware Hub client.
    # Explicitly request only the fields displayed by the picker, not every
    # repository's file list or model card. No model files are fetched here.
    try:
        response = httpx.get(
            HUB_SEARCH_URL,
            params={"search": query, "pipeline_tag": "text-generation", "filter": "transformers",
                    "sort": "downloads", "direction": -1, "limit": MAX_RESULTS,
                    "expand": ["gated", "private", "downloads", "tags", "pipeline_tag", "library_name"]},
            timeout=5.0, follow_redirects=False,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            raise ValueError("Unexpected Hub response")
        models = []
        seen = set()
        for row in payload[:MAX_RESULTS]:
            if not _searchable_checkpoint(row):
                continue
            model = _model(row, "huggingface")
            if model and not model["private"] and model["id"] not in seen:
                seen.add(model["id"])
                models.append(model)
        # Some official checkpoints have incomplete Hub task/library tags.
        # Keep matching curated entries discoverable in that case, while live
        # metadata takes precedence for entries the Hub actually returned.
        live = {row["id"]: row for row in models}
        matching_ids = {row["id"] for row in matching_common}
        models = [live.get(row["id"], row) for row in matching_common] + [row for row in models if row["id"] not in matching_ids]
        return {"models": models[:MAX_RESULTS], "source": "huggingface"}
    except (httpx.HTTPError, ValueError):
        return {
            "models": matching_common,
            "source": "curated",
            "warning": "Hugging Face search is unavailable. Showing matching common models; you can also enter a repository ID.",
        }


def _has_hub_token() -> bool:
    # This loader module is deliberately torch-free at import time. Reuse its
    # exact environment / Hub CLI cache / legacy token-file resolution so a
    # "configured" badge agrees with both downloads and analysis jobs.
    from vivasecuris.aiasylum.interp.core.loader import get_hf_token

    return bool(get_hf_token())


def huggingface_status() -> dict[str, Any]:
    try:
        configured = _has_hub_token()
    except (OSError, UnicodeError):
        return {"token_configured": False, "message": "Could not read Hugging Face token configuration on this server. Public model discovery does not require sign-in."}
    if configured:
        message = "A Hugging Face token is configured on this server. Gated models still require approval for that account; token validity and access have not been checked."
    else:
        message = "Public models do not require sign-in. For gated or private models, configure a read token on this server (HF_TOKEN or hf auth login); signing into the Hugging Face website alone does not connect the server."
    return {"token_configured": configured, "message": message}
