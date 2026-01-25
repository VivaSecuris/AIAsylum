"""Jailbreak prompt loader from database.

This module provides functionality to load jailbreak prompts from the PromptLibrary
database for use in adversarial testing.
"""

from typing import List, Optional, Dict
from random import sample

from vivasecuris.aiasylum.database import get_session, PromptLibrary


def load_jailbreak_prompts(
    technique: Optional[str] = None,
    source_platform: Optional[str] = None,
    source: Optional[str] = None,
    community_name: Optional[str] = None,
    limit: Optional[int] = None,
    random: bool = False,
) -> List[str]:
    """
    Load jailbreak prompts from the database.
    
    Args:
        technique: Filter by jailbreak technique (e.g., "dan_variant", "hypothetical_roleplay")
        source_platform: Filter by platform (e.g., "reddit", "discord", "website")
        source: Filter by specific source (e.g., "r/ChatGPT", "FlowGPT")
        community_name: Filter by community name
        limit: Maximum number of prompts to return
        random: If True, return random selection; otherwise return in order
    
    Returns:
        List of prompt texts
    """
    session = get_session()
    try:
        # Build query - filter by category first
        # All other filtering done in Python due to SQLite JSON limitations
        query = session.query(PromptLibrary).filter(
            PromptLibrary.category == "adversarial"
        )
        
        # Execute query and filter everything in Python (SQLite JSON limitation)
        all_prompts = query.all()
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
        
        # Extract prompt texts
        prompt_texts = [p.prompt_text for p in prompts]
        
        # Apply randomization first if requested
        if random:
            prompt_texts = sample(prompt_texts, len(prompt_texts))
        
        # Apply limit after filtering and randomization
        if limit:
            prompt_texts = prompt_texts[:limit]
        
        return prompt_texts
    finally:
        session.close()


def load_forbidden_questions(
    scenario: Optional[str] = None,
    limit: Optional[int] = None,
    random: bool = False,
) -> List[str]:
    """
    Load forbidden questions from the database.
    
    Args:
        scenario: Filter by content policy name (e.g., "Illegal Activity", "Hate Speech")
        limit: Maximum number of questions to return
        random: If True, return random selection; otherwise return in order
    
    Returns:
        List of question texts
    """
    session = get_session()
    try:
        # Build query
        query = session.query(PromptLibrary).filter(
            PromptLibrary.category == "forbidden_question"
        )
        
        # Apply scenario filter
        if scenario:
            query = query.filter(
                PromptLibrary.meta_data.contains({"content_policy_name": scenario})
            )
        
        # Execute query
        questions = query.all()
        
        # Extract question texts
        question_texts = [q.prompt_text for q in questions]
        
        # Apply limit and randomization
        if random and limit and len(question_texts) > limit:
            question_texts = sample(question_texts, limit)
        elif limit:
            question_texts = question_texts[:limit]
        elif random:
            question_texts = sample(question_texts, len(question_texts))
        
        return question_texts
    finally:
        session.close()


def get_jailbreak_prompt_metadata(prompt_text: str) -> Optional[Dict]:
    """
    Get metadata for a specific jailbreak prompt.
    
    Args:
        prompt_text: The prompt text to look up
    
    Returns:
        Dictionary with metadata or None if not found
    """
    session = get_session()
    try:
        prompt = session.query(PromptLibrary).filter(
            PromptLibrary.prompt_text == prompt_text,
            PromptLibrary.category == "adversarial"
        ).first()
        
        if prompt:
            return prompt.meta_data
        return None
    finally:
        session.close()


def get_available_techniques() -> List[str]:
    """
    Get list of available jailbreak techniques in the database.
    
    Returns:
        List of technique names
    """
    session = get_session()
    try:
        # Get all adversarial prompts and filter by tags in Python
        all_prompts = session.query(PromptLibrary).filter(
            PromptLibrary.category == "adversarial"
        ).all()
        
        prompts = [p for p in all_prompts if p.tags and "jailbreak" in p.tags]
        
        techniques = set()
        for prompt in prompts:
            if prompt.meta_data:
                technique = prompt.meta_data.get("jailbreak_technique")
                if technique:
                    techniques.add(technique)
        
        return sorted(list(techniques))
    finally:
        session.close()


def get_available_sources() -> Dict[str, List[str]]:
    """
    Get available sources organized by platform.
    
    Returns:
        Dictionary mapping platform to list of sources
    """
    session = get_session()
    try:
        # Get all adversarial prompts and filter by tags in Python
        all_prompts = session.query(PromptLibrary).filter(
            PromptLibrary.category == "adversarial"
        ).all()
        
        prompts = [p for p in all_prompts if p.tags and "jailbreak" in p.tags]
        
        sources_by_platform: Dict[str, set] = {}
        for prompt in prompts:
            if prompt.meta_data:
                platform = prompt.meta_data.get("source_platform", "unknown")
                source = prompt.meta_data.get("source", "unknown")
                
                if platform not in sources_by_platform:
                    sources_by_platform[platform] = set()
                sources_by_platform[platform].add(source)
        
        return {
            platform: sorted(list(sources))
            for platform, sources in sources_by_platform.items()
        }
    finally:
        session.close()


def count_jailbreak_prompts(
    technique: Optional[str] = None,
    source_platform: Optional[str] = None,
) -> int:
    """
    Count jailbreak prompts matching criteria.
    
    Args:
        technique: Filter by jailbreak technique
        source_platform: Filter by platform
    
    Returns:
        Count of matching prompts
    """
    session = get_session()
    try:
        # Build query - filter by category first
        # All filtering done in Python due to SQLite JSON limitations
        query = session.query(PromptLibrary).filter(
            PromptLibrary.category == "adversarial"
        )
        
        # Filter by tags in Python (SQLite JSON limitation)
        all_prompts = query.all()
        prompts = [p for p in all_prompts if p.tags and "jailbreak" in p.tags]
        
        # Apply additional filters in Python
        if technique:
            prompts = [p for p in prompts if p.meta_data and p.meta_data.get("jailbreak_technique") == technique]
        
        if source_platform:
            prompts = [p for p in prompts if p.meta_data and p.meta_data.get("source_platform") == source_platform]
        
        return len(prompts)
    finally:
        session.close()
