"""Prompt library routes."""

from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from vivasecuris.aiasylum.database import get_session, PromptLibrary

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
    created_at: str
    updated_at: Optional[str]
    metadata: dict
    
    class Config:
        from_attributes = True


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
        
        return db_prompt
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
        return prompts
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
        return prompt
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
        return prompt
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
