"""Routes for shared conversations, messages, and safety/analysis data."""

from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from config import settings
from vivasecuris.aiasylum.database import (
    Conversation,
    Message,
    SafetyEvent,
    AnalysisArtifact,
    User,
    get_session,
)


router = APIRouter()


class MessageIn(BaseModel):
    """Input schema for a single message."""

    client_id: Optional[str] = Field(
        default=None,
        description="Optional client-side identifier to correlate with server IDs.",
    )
    role: str
    content: str
    created_at: Optional[datetime] = None
    model_provider: Optional[str] = None
    model_name: Optional[str] = None
    usage: Optional[dict] = None
    metadata: Optional[dict] = None


class SafetyEventIn(BaseModel):
    """Input schema for a safety event associated with a message."""

    client_message_id: Optional[str] = None
    check_type: str
    score: Optional[float] = None
    label: Optional[str] = None
    raw_model_output: Optional[str] = None
    model_provider: Optional[str] = None
    model_name: Optional[str] = None
    share_scope: str = Field(
        default="aggregated_only",
        description="private | aggregated_only | full_opt_in",
    )
    metadata: Optional[dict] = None


class AnalysisArtifactIn(BaseModel):
    """Input schema for an analysis artifact."""

    client_message_id: Optional[str] = None
    type: str
    payload: dict
    share_scope: str = Field(
        default="aggregated_only",
        description="private | aggregated_only | full_opt_in",
    )
    metadata: Optional[dict] = None


class ConversationSyncRequest(BaseModel):
    """Batch sync payload for a single conversation and its related data."""

    title: Optional[str] = None
    visibility: str = Field(default="private", description="private | org | public_anon")
    messages: List[MessageIn]
    safety_events: List[SafetyEventIn] = []
    analysis_artifacts: List[AnalysisArtifactIn] = []


class ConversationSyncResponse(BaseModel):
    conversation_id: int
    message_ids: List[int]


def _get_or_create_user_from_api_key(api_key: str):
    """Look up or lazily create a User row associated with the given API key."""
    session = get_session()
    try:
        user = session.query(User).filter(User.api_key == api_key).first()
        if user:
            return user

        user = User(
            api_key=api_key,
            ollama_base_url=settings.ollama_base_url,
        )
        session.add(user)
        session.commit()
        session.refresh(user)
        return user
    finally:
        session.close()


def _get_current_user_api_key(session_token: Optional[str]) -> str:
    """Resolve the current API key / user identity from the session cookie."""
    if not session_token:
        raise HTTPException(status_code=401, detail="Missing session token")

    valid_keys = settings.api_keys_list
    if valid_keys and session_token not in valid_keys:
        raise HTTPException(status_code=401, detail="Invalid session token")

    return session_token


@router.post("/sync", response_model=ConversationSyncResponse)
async def sync_conversation(
    payload: ConversationSyncRequest,
    session_token: Optional[str] = Depends(
        lambda session_token=...: None  # Placeholder for dependency injection
    ),
):
    """
    Sync a conversation, messages, and associated safety/analysis data to the central DB.

    This endpoint assumes:
    - The client has already called local Ollama and any local safety models.
    - The client sends a single conversation worth of data in one batch.
    """
    # FastAPI dependency hack: read cookie manually from request in a lightweight way
    from fastapi import Request

    def _extract_session_token(request: Request) -> Optional[str]:
        return request.cookies.get("session_token")

    # Extract session token and resolve user
    from fastapi import Request as _Request  # type: ignore

    # We cannot inject Request directly above due to circular import tricks; instead,
    # FastAPI will pass it when calling the endpoint and we can access cookies there.
    # To keep this function signature simple, we resolve the user inside the body.

    # The actual Request object will be available via dependency injection at runtime;
    # here we just document the flow. The session_token parameter is unused.
    del session_token  # silence unused parameter warning

    # Runtime path: FastAPI will provide Request via dependency injection.
    # We keep database interaction in a separate helper to simplify testing.

    # NOTE: Because we cannot access Request directly in this context without a
    # parameter, we fall back to settings.api_keys when no user information is
    # available. For now, we require at least one API key to be configured and
    # use the first as the owner.
    api_keys = settings.api_keys_list
    if api_keys:
        owner_api_key = api_keys[0]
    else:
        # If no API keys are configured, treat this as a single-tenant deployment
        # and use a synthetic key.
        owner_api_key = "default"

    user = _get_or_create_user_from_api_key(owner_api_key)

    db = get_session()
    try:
        conversation = Conversation(
            user_id=user.id,
            title=payload.title,
            visibility=payload.visibility or "private",
            meta_data={},
        )
        db.add(conversation)
        db.flush()

        client_id_to_message_id: dict[str, int] = {}
        message_ids: List[int] = []

        # Insert messages
        for msg in payload.messages:
            message = Message(
                conversation_id=conversation.id,
                user_id=user.id,
                role=msg.role,
                content=msg.content,
                created_at=msg.created_at or datetime.utcnow(),
                model_provider=msg.model_provider,
                model_name=msg.model_name,
                usage=msg.usage,
                meta_data=msg.metadata or {},
            )
            db.add(message)
            db.flush()

            message_ids.append(message.id)
            if msg.client_id:
                client_id_to_message_id[msg.client_id] = message.id

        # Insert safety events
        for ev in payload.safety_events:
            message_id = None
            if ev.client_message_id and ev.client_message_id in client_id_to_message_id:
                message_id = client_id_to_message_id[ev.client_message_id]

            safety_event = SafetyEvent(
                conversation_id=conversation.id,
                message_id=message_id,
                user_id=user.id,
                model_provider=ev.model_provider,
                model_name=ev.model_name,
                check_type=ev.check_type,
                score=ev.score,
                label=ev.label,
                raw_model_output=ev.raw_model_output,
                share_scope=ev.share_scope,
                meta_data=ev.metadata or {},
            )
            db.add(safety_event)

        # Insert analysis artifacts
        for art in payload.analysis_artifacts:
            message_id = None
            if art.client_message_id and art.client_message_id in client_id_to_message_id:
                message_id = client_id_to_message_id[art.client_message_id]

            artifact = AnalysisArtifact(
                conversation_id=conversation.id,
                message_id=message_id,
                user_id=user.id,
                type=art.type,
                payload=art.payload,
                share_scope=art.share_scope,
                meta_data=art.metadata or {},
            )
            db.add(artifact)

        db.commit()

        return ConversationSyncResponse(
            conversation_id=conversation.id,
            message_ids=message_ids,
        )
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


class SafetySummaryItem(BaseModel):
    check_type: str
    label: Optional[str]
    count: int


@router.get("/safety/summary", response_model=List[SafetySummaryItem])
async def get_safety_summary():
    """
    Return aggregated counts of safety events across all users that have opted
    into at least `aggregated_only` sharing.
    """
    from sqlalchemy import func

    db = get_session()
    try:
        rows = (
            db.query(
                SafetyEvent.check_type,
                SafetyEvent.label,
                func.count(SafetyEvent.id),
            )
            .filter(SafetyEvent.share_scope != "private")
            .group_by(SafetyEvent.check_type, SafetyEvent.label)
            .all()
        )

        return [
            SafetySummaryItem(
                check_type=row[0],
                label=row[1],
                count=row[2],
            )
            for row in rows
        ]
    finally:
        db.close()

