"""Client-side helper for syncing local conversations and safety data to the central API.

This module encodes the recommended flow:
1. Call local Ollama for inference.
2. Run any local safety / analysis models.
3. Use `sync_conversation_with_backend` to POST a batch payload to the API.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx


@dataclass
class LocalMessage:
    client_id: Optional[str]
    role: str
    content: str
    created_at: Optional[datetime] = None
    model_provider: Optional[str] = None
    model_name: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class LocalSafetyEvent:
    client_message_id: Optional[str]
    check_type: str
    score: Optional[float] = None
    label: Optional[str] = None
    raw_model_output: Optional[str] = None
    model_provider: Optional[str] = None
    model_name: Optional[str] = None
    share_scope: str = "aggregated_only"
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class LocalAnalysisArtifact:
    client_message_id: Optional[str]
    type: str
    payload: Dict[str, Any]
    share_scope: str = "aggregated_only"
    metadata: Optional[Dict[str, Any]] = None


def _isoformat_or_none(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    return dt.isoformat()


def build_sync_payload(
    title: Optional[str],
    visibility: str,
    messages: List[LocalMessage],
    safety_events: Optional[List[LocalSafetyEvent]] = None,
    analysis_artifacts: Optional[List[LocalAnalysisArtifact]] = None,
) -> Dict[str, Any]:
    """Build the JSON payload expected by /api/v1/conversations/sync."""
    safety_events = safety_events or []
    analysis_artifacts = analysis_artifacts or []

    return {
        "title": title,
        "visibility": visibility,
        "messages": [
            {
                "client_id": m.client_id,
                "role": m.role,
                "content": m.content,
                "created_at": _isoformat_or_none(m.created_at),
                "model_provider": m.model_provider,
                "model_name": m.model_name,
                "usage": m.usage,
                "metadata": m.metadata,
            }
            for m in messages
        ],
        "safety_events": [asdict(ev) for ev in safety_events],
        "analysis_artifacts": [asdict(art) for art in analysis_artifacts],
    }


def sync_conversation_with_backend(
    api_base_url: str,
    api_key: str,
    title: Optional[str],
    visibility: str,
    messages: List[LocalMessage],
    safety_events: Optional[List[LocalSafetyEvent]] = None,
    analysis_artifacts: Optional[List[LocalAnalysisArtifact]] = None,
    timeout: float = 10.0,
) -> Dict[str, Any]:
    """Sync a conversation and associated safety/analysis data to the central API.

    The caller is responsible for:
    - Talking to local Ollama and any local safety models.
    - Constructing LocalMessage / LocalSafetyEvent / LocalAnalysisArtifact objects.
    """
    payload = build_sync_payload(
        title=title,
        visibility=visibility,
        messages=messages,
        safety_events=safety_events,
        analysis_artifacts=analysis_artifacts,
    )

    url = api_base_url.rstrip("/") + "/api/v1/conversations/sync"
    headers = {"Authorization": f"Bearer {api_key}"}

    with httpx.Client(timeout=timeout) as client:
        response = client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()

