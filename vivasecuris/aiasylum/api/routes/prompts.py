"""Prompt library routes."""

from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import or_
from pydantic import BaseModel, Field

from vivasecuris.aiasylum.database import get_session, PromptLibrary
from vivasecuris.aiasylum.utils import extract_variables
from vivasecuris.aiasylum.tests.jailbreak_loader import (
    get_available_techniques,
    get_available_sources,
    count_jailbreak_prompts,
)

router = APIRouter()


class PromptCreate(BaseModel):
    """Prompt creation request."""
    name: str
    description: Optional[str] = None
    prompt_text: str
    prompt_type: Optional[str] = "test_prompt"  # test_prompt or system_prompt
    target: Optional[str] = None  # doctor, patient, or None for test prompts
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    metadata: Optional[dict] = None


class PromptUpdate(BaseModel):
    """Prompt update request."""
    name: Optional[str] = None
    description: Optional[str] = None
    prompt_text: Optional[str] = None
    prompt_type: Optional[str] = None
    target: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    metadata: Optional[dict] = None


class PromptResponse(BaseModel):
    """Prompt response."""
    id: int
    name: str
    description: Optional[str]
    prompt_text: str
    prompt_type: Optional[str]
    target: Optional[str]
    category: Optional[str]
    tags: List[str]
    usage_count: int
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    metadata: dict
    
    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }
    
    @classmethod
    def from_orm(cls, obj: PromptLibrary):
        """Create response from SQLAlchemy model, handling metadata conflict."""
        return cls(
            id=obj.id,
            name=obj.name,
            description=obj.description,
            prompt_text=obj.prompt_text,
            prompt_type=obj.prompt_type,
            target=obj.target,
            category=obj.category,
            tags=obj.tags or [],
            usage_count=obj.usage_count or 0,
            created_at=obj.created_at,
            updated_at=obj.updated_at,
            metadata=obj.meta_data or {},
        )


@router.post("/", response_model=PromptResponse)
async def create_prompt(prompt: PromptCreate):
    """Create a new prompt in the library."""
    session = get_session()
    try:
        # Check if name already exists
        existing = session.query(PromptLibrary).filter(PromptLibrary.name == prompt.name).first()
        if existing:
            raise HTTPException(status_code=400, detail=f"Prompt with name '{prompt.name}' already exists")
        
        db_prompt = PromptLibrary(
            name=prompt.name,
            description=prompt.description,
            prompt_text=prompt.prompt_text,
            prompt_type=prompt.prompt_type or "test_prompt",
            target=prompt.target,
            category=prompt.category,
            tags=prompt.tags or [],
            meta_data=prompt.metadata or {},
        )
        session.add(db_prompt)
        session.commit()
        session.refresh(db_prompt)
        
        return PromptResponse.from_orm(db_prompt)
    finally:
        session.close()


@router.get("/", response_model=List[PromptResponse])
async def list_prompts(
    category: Optional[str] = None,
    tag: Optional[str] = None,
    prompt_type: Optional[str] = None,
    target: Optional[str] = None,
    search: Optional[str] = Query(None, max_length=200),
    limit: int = Query(100, ge=1, le=5000),
    offset: int = Query(0, ge=0),
):
    """List prompts from the library, newest first.
    
    ``search`` matches name, description and prompt text (case-insensitive), and
    also the prompt ID when it is all digits.

    Note: Forbidden questions are completely excluded from this endpoint.
    They should only be used AFTER a jailbreak is achieved, not in general searches.
    Use /api/v1/prompts/forbidden-questions/list to access them in specific test contexts.
    """
    session = get_session()
    try:
        query = session.query(PromptLibrary)
        
        if category:
            query = query.filter(PromptLibrary.category == category)
        
        if tag:
            # Filter by tag (tags is a JSON array)
            query = query.filter(PromptLibrary.tags.contains([tag]))
        
        if prompt_type:
            query = query.filter(PromptLibrary.prompt_type == prompt_type)
        
        if target:
            query = query.filter(PromptLibrary.target == target)
        
        term = (search or "").strip()
        if term:
            like = "%" + term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
            matches = [
                PromptLibrary.name.ilike(like, escape="\\"),
                PromptLibrary.description.ilike(like, escape="\\"),
                PromptLibrary.prompt_text.ilike(like, escape="\\"),
            ]
            if term.isdigit():
                matches.append(PromptLibrary.id == int(term))
            query = query.filter(or_(*matches))
        
        # Get all prompts and filter in Python (SQLite JSON limitation)
        all_prompts = query.order_by(PromptLibrary.created_at.desc()).all()
        
        # Completely exclude forbidden questions from general prompt library
        # They should only be used AFTER a jailbreak is achieved, not in general searches
        prompts = [p for p in all_prompts if p.category != "forbidden_question"]
        
        # Apply pagination after filtering
        prompts = prompts[offset:offset + limit]
        
        return [PromptResponse.from_orm(p) for p in prompts]
    finally:
        session.close()


@router.get("/defaults")
async def get_prompt_defaults():
    """The built-in system prompts each role falls back to when none is chosen.

    Declared before ``/{prompt_id}`` so "defaults" is not parsed as an ID.
    """
    from vivasecuris.aiasylum.analysis.prompts import get_evaluation_system_prompt
    from vivasecuris.aiasylum.doctor.doctor import ASSESSMENT_INSTRUCTIONS, DEFAULT_DOCTOR_SYSTEM_PROMPT
    from vivasecuris.aiasylum.patient.patient import (
        PATIENT_INTERVIEW_INPUT_FORMAT, PATIENT_INTERVIEW_TEMPLATE, PATIENT_QUESTION_TEMPLATE,
    )

    return {
        "doctor": {
            "system_prompt": DEFAULT_DOCTOR_SYSTEM_PROMPT,
            "assessment_instructions": ASSESSMENT_INSTRUCTIONS,
            "notes": [
                "Used whenever no doctor system prompt is chosen.",
                "The assessment instructions are sent after the transcript at the end of every test.",
            ],
        },
        "patient": {
            "system_prompt": None,
            "user_message_template": PATIENT_QUESTION_TEMPLATE,
            "interview_user_message_template": PATIENT_INTERVIEW_TEMPLATE,
            "interview_input_format": PATIENT_INTERVIEW_INPUT_FORMAT,
            "notes": [
                "One-Shot and Multi-Shot use user_message_template only when no system prompt is selected.",
                "Conversation and Group Therapy use interview_user_message_template even with a "
                "selected custom or library system prompt; the selected system text remains unchanged.",
                "The interview template's {prompt} is a JSON object with speaker='doctor' and "
                "message containing the input text; quoting attributes the speaker without rewriting the input.",
                "Explicitly disabling patient prompt framing skips both wrappers. Benchmarks use their own prompt format.",
                "ReACT, when enabled, separately adds a requested reasoning text format in one model call; "
                "it does not execute actions or observation tools.",
            ],
        },
        "evaluator": {
            "system_prompt": get_evaluation_system_prompt(),
            "default_temperature": 0.3,
            "notes": [
                "Custom evaluator instructions are added after this prompt; the scoring "
                "dimensions and JSON format are always kept.",
            ],
        },
    }


@router.get("/{prompt_id}", response_model=PromptResponse)
async def get_prompt(prompt_id: int):
    """Get a prompt by ID."""
    session = get_session()
    try:
        prompt = session.query(PromptLibrary).filter(PromptLibrary.id == prompt_id).first()
        if not prompt:
            raise HTTPException(status_code=404, detail="Prompt not found")
        return PromptResponse.from_orm(prompt)
    finally:
        session.close()


@router.put("/{prompt_id}", response_model=PromptResponse)
async def update_prompt(prompt_id: int, prompt_update: PromptUpdate):
    """Update a prompt."""
    session = get_session()
    try:
        prompt = session.query(PromptLibrary).filter(PromptLibrary.id == prompt_id).first()
        if not prompt:
            raise HTTPException(status_code=404, detail="Prompt not found")
        
        # Only fields present in the request change; an explicit null clears an
        # optional field (description, target, category, tags, metadata).
        provided = prompt_update.model_fields_set
        if prompt_update.name is not None:
            # Check if new name conflicts with existing prompt
            existing = session.query(PromptLibrary).filter(
                PromptLibrary.name == prompt_update.name,
                PromptLibrary.id != prompt_id
            ).first()
            if existing:
                raise HTTPException(status_code=400, detail=f"Prompt with name '{prompt_update.name}' already exists")
            prompt.name = prompt_update.name
        
        if prompt_update.prompt_text is not None:
            prompt.prompt_text = prompt_update.prompt_text
        
        if prompt_update.prompt_type is not None:
            prompt.prompt_type = prompt_update.prompt_type
        
        for field in ("description", "target", "category"):
            if field in provided:
                value = getattr(prompt_update, field)
                setattr(prompt, field, value if value not in ("",) else None)
        
        if "tags" in provided:
            prompt.tags = prompt_update.tags or []
        
        if "metadata" in provided:
            prompt.meta_data = prompt_update.metadata or {}
        
        session.commit()
        session.refresh(prompt)
        return PromptResponse.from_orm(prompt)
    finally:
        session.close()


@router.delete("/{prompt_id}")
async def delete_prompt(prompt_id: int):
    """Delete a prompt."""
    session = get_session()
    try:
        prompt = session.query(PromptLibrary).filter(PromptLibrary.id == prompt_id).first()
        if not prompt:
            raise HTTPException(status_code=404, detail="Prompt not found")
        session.delete(prompt)
        session.commit()
        return {"message": "Prompt deleted successfully"}
    except HTTPException:
        if session:
            session.rollback()
        raise
    except Exception as e:
        if session:
            session.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete prompt: {str(e)}")
    finally:
        session.close()


@router.get("/{prompt_id}/variables")
async def get_prompt_variables(prompt_id: int):
    """Extract variables from a prompt."""
    session = get_session()
    try:
        prompt = session.query(PromptLibrary).filter(PromptLibrary.id == prompt_id).first()
        if not prompt:
            raise HTTPException(status_code=404, detail="Prompt not found")
        
        variables = extract_variables(prompt.prompt_text)
        return {"variables": variables}
    finally:
        session.close()


@router.post("/{prompt_id}/increment-usage")
async def increment_usage(prompt_id: int):
    """Increment usage count for a prompt."""
    session = get_session()
    try:
        prompt = session.query(PromptLibrary).filter(PromptLibrary.id == prompt_id).first()
        if not prompt:
            raise HTTPException(status_code=404, detail="Prompt not found")
        
        prompt.usage_count = (prompt.usage_count or 0) + 1
        session.commit()
        return {"usage_count": prompt.usage_count}
    finally:
        session.close()


@router.get("/jailbreaks/list", response_model=List[PromptResponse])
async def list_jailbreak_prompts(
    technique: Optional[str] = None,
    source_platform: Optional[str] = None,
    source: Optional[str] = None,
    community_name: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
):
    """List jailbreak prompts with filtering options."""
    session = get_session()
    try:
        # Build query - filter by category first
        # All filtering done in Python due to SQLite JSON limitations
        query = session.query(PromptLibrary).filter(
            PromptLibrary.category == "adversarial"
        )
        
        # Get all results and filter by tags in Python
        all_prompts = query.order_by(PromptLibrary.created_at.desc()).all()
        prompts = [p for p in all_prompts if p.tags and "jailbreak" in p.tags]
        
        # Apply all filters in Python
        if technique:
            prompts = [p for p in prompts if p.meta_data and p.meta_data.get("jailbreak_technique") == technique]
        
        if source_platform:
            prompts = [p for p in prompts if p.meta_data and p.meta_data.get("source_platform") == source_platform]
        
        if source:
            prompts = [p for p in prompts if p.meta_data and p.meta_data.get("source") == source]
        
        if community_name:
            prompts = [p for p in prompts if p.meta_data and p.meta_data.get("community_name") == community_name]
        
        # Apply pagination after filtering
        prompts = prompts[offset:offset + limit]
        
        return [PromptResponse.from_orm(p) for p in prompts]
    finally:
        session.close()


@router.get("/jailbreaks/techniques")
async def get_jailbreak_techniques():
    """Get list of available jailbreak techniques."""
    techniques = get_available_techniques()
    return {"techniques": techniques}


@router.get("/jailbreaks/sources")
async def get_jailbreak_sources():
    """Get available sources organized by platform."""
    sources = get_available_sources()
    return {"sources_by_platform": sources}


@router.get("/jailbreaks/stats")
async def get_jailbreak_stats(
    technique: Optional[str] = None,
    source_platform: Optional[str] = None,
):
    """Get statistics on jailbreak prompts."""
    total = count_jailbreak_prompts(technique=technique, source_platform=source_platform)
    
    # Get technique breakdown
    techniques = get_available_techniques()
    technique_counts = {}
    for tech in techniques:
        count = count_jailbreak_prompts(technique=tech, source_platform=source_platform)
        if count > 0:
            technique_counts[tech] = count
    
    # Get platform breakdown
    sources = get_available_sources()
    platform_counts = {}
    for platform, platform_sources in sources.items():
        count = count_jailbreak_prompts(source_platform=platform)
        if count > 0:
            platform_counts[platform] = count
    
    return {
        "total": total,
        "by_technique": technique_counts,
        "by_platform": platform_counts,
    }


@router.get("/forbidden-questions/list", response_model=List[PromptResponse])
async def list_forbidden_questions(
    scenario: Optional[str] = None,
    approved_only: bool = True,  # Only show approved by default
    limit: int = 100,
    offset: int = 0,
):
    """List forbidden questions (test targets).
    
    IMPORTANT: These questions should only be used AFTER a jailbreak is achieved.
    They are test targets to see what a jailbroken model will do, not jailbreak prompts themselves.
    Only approved questions are shown by default.
    """
    session = get_session()
    try:
        # Build query - filter by category first
        query = session.query(PromptLibrary).filter(
            PromptLibrary.category == "forbidden_question"
        )
        
        # Get all questions and filter in Python (SQLite JSON limitation)
        all_questions = query.order_by(PromptLibrary.created_at.desc()).all()
        
        # Filter by approval status
        if approved_only:
            questions = [q for q in all_questions if q.meta_data and q.meta_data.get("approved", False)]
        else:
            questions = all_questions
        
        # Apply scenario filter in Python
        if scenario:
            questions = [q for q in questions if q.meta_data and q.meta_data.get("content_policy_name") == scenario]
        
        # Apply pagination after filtering
        questions = questions[offset:offset + limit]
        
        return [PromptResponse.from_orm(q) for q in questions]
    finally:
        session.close()


@router.post("/forbidden-questions/{question_id}/approve")
async def approve_forbidden_question(question_id: int):
    """Approve a forbidden question to make it visible in searches."""
    session = get_session()
    try:
        question = session.query(PromptLibrary).filter(
            PromptLibrary.id == question_id,
            PromptLibrary.category == "forbidden_question"
        ).first()
        
        if not question:
            raise HTTPException(status_code=404, detail="Forbidden question not found")
        
        # Update metadata to mark as approved
        if not question.meta_data:
            question.meta_data = {}
        question.meta_data["approved"] = True
        question.meta_data["approved_date"] = datetime.utcnow().isoformat()
        
        session.commit()
        session.refresh(question)
        
        return PromptResponse.from_orm(question)
    finally:
        session.close()


@router.post("/forbidden-questions/{question_id}/unapprove")
async def unapprove_forbidden_question(question_id: int):
    """Unapprove a forbidden question to hide it from searches."""
    session = get_session()
    try:
        question = session.query(PromptLibrary).filter(
            PromptLibrary.id == question_id,
            PromptLibrary.category == "forbidden_question"
        ).first()
        
        if not question:
            raise HTTPException(status_code=404, detail="Forbidden question not found")
        
        # Update metadata to mark as unapproved
        if not question.meta_data:
            question.meta_data = {}
        question.meta_data["approved"] = False
        
        session.commit()
        session.refresh(question)
        
        return PromptResponse.from_orm(question)
    finally:
        session.close()
