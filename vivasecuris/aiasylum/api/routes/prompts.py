"""Prompt library routes."""

from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException
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
    limit: int = 100,
    offset: int = 0,
):
    """List prompts from the library."""
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
        
        prompts = query.order_by(PromptLibrary.created_at.desc()).limit(limit).offset(offset).all()
        return [PromptResponse.from_orm(p) for p in prompts]
    finally:
        session.close()


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
        
        # Update fields if provided
        if prompt_update.name is not None:
            # Check if new name conflicts with existing prompt
            existing = session.query(PromptLibrary).filter(
                PromptLibrary.name == prompt_update.name,
                PromptLibrary.id != prompt_id
            ).first()
            if existing:
                raise HTTPException(status_code=400, detail=f"Prompt with name '{prompt_update.name}' already exists")
            prompt.name = prompt_update.name
        
        if prompt_update.description is not None:
            prompt.description = prompt_update.description
        
        if prompt_update.prompt_text is not None:
            prompt.prompt_text = prompt_update.prompt_text
        
        if prompt_update.prompt_type is not None:
            prompt.prompt_type = prompt_update.prompt_type
        
        if prompt_update.target is not None:
            prompt.target = prompt_update.target
        
        if prompt_update.category is not None:
            prompt.category = prompt_update.category
        
        if prompt_update.tags is not None:
            prompt.tags = prompt_update.tags
        
        if prompt_update.metadata is not None:
            prompt.meta_data = prompt_update.metadata
        
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
    limit: int = 100,
    offset: int = 0,
):
    """List forbidden questions (test targets)."""
    session = get_session()
    try:
        query = session.query(PromptLibrary).filter(
            PromptLibrary.category == "forbidden_question"
        )
        
        if scenario:
            query = query.filter(
                PromptLibrary.meta_data.contains({"content_policy_name": scenario})
            )
        
        questions = query.order_by(PromptLibrary.created_at.desc()).limit(limit).offset(offset).all()
        return [PromptResponse.from_orm(q) for q in questions]
    finally:
        session.close()
