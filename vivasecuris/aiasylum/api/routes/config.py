"""API routes for runtime configuration (API keys, etc.)."""

import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

_ENV_KEY_MAP = {
    "openai_api_key": "OPENAI_API_KEY",
    "anthropic_api_key": "ANTHROPIC_API_KEY",
    "google_api_key": "GOOGLE_API_KEY",
}


class ApiKeysStatus(BaseModel):
    openai_configured: bool
    anthropic_configured: bool
    google_configured: bool


class ApiKeysUpdate(BaseModel):
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    google_api_key: Optional[str] = None


@router.get("/api-keys/status", response_model=ApiKeysStatus)
async def get_api_keys_status():
    """Return which provider API keys are currently configured (values are never exposed)."""
    return ApiKeysStatus(
        openai_configured=bool(settings.openai_api_key),
        anthropic_configured=bool(settings.anthropic_api_key),
        google_configured=bool(settings.google_api_key),
    )


@router.put("/api-keys")
async def update_api_keys(body: ApiKeysUpdate):
    """Update provider API keys at runtime and persist them to the .env file."""
    updates: dict[str, Optional[str]] = {}

    if body.openai_api_key is not None:
        value = body.openai_api_key.strip() or None
        settings.openai_api_key = value
        updates["openai_api_key"] = value

    if body.anthropic_api_key is not None:
        value = body.anthropic_api_key.strip() or None
        settings.anthropic_api_key = value
        updates["anthropic_api_key"] = value

    if body.google_api_key is not None:
        value = body.google_api_key.strip() or None
        settings.google_api_key = value
        updates["google_api_key"] = value

    if updates:
        _persist_to_env(updates)
        logger.info(f"Updated API keys: {list(updates.keys())}")

    return {
        "status": "updated",
        "openai_configured": bool(settings.openai_api_key),
        "anthropic_configured": bool(settings.anthropic_api_key),
        "google_configured": bool(settings.google_api_key),
    }


def _persist_to_env(updates: dict[str, Optional[str]]) -> None:
    """Write updated keys to the project .env file, creating it if needed."""
    env_path = settings.project_root / ".env"

    existing_lines: list[str] = []
    if env_path.exists():
        existing_lines = env_path.read_text(encoding="utf-8").splitlines()

    env_updates: dict[str, Optional[str]] = {
        _ENV_KEY_MAP[k]: v for k, v in updates.items() if k in _ENV_KEY_MAP
    }

    written_keys: set[str] = set()
    new_lines: list[str] = []

    for line in existing_lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in env_updates:
                written_keys.add(key)
                if env_updates[key]:
                    new_lines.append(f"{key}={env_updates[key]}")
                # empty value → drop the line (key removed)
                continue
        new_lines.append(line)

    # Append any keys that weren't already in the file
    for key, value in env_updates.items():
        if key not in written_keys and value:
            new_lines.append(f"{key}={value}")

    env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
