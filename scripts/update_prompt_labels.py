#!/usr/bin/env python3
"""Update existing jailbreak prompts with better labels and tags for searchability."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from vivasecuris.aiasylum.database import get_session, PromptLibrary


def get_technique_category(technique: str) -> str:
    """Categorize technique into broader categories."""
    roleplay_techniques = ['hypothetical_roleplay', 'role_playing', 'double_character_persona_split']
    direct_bypass = ['dan_variant', 'refusal_suppression', 'instruction_override', 'prompt_injection']
    indirect_bypass = ['instruction_laundering', 'instruction_smuggling', 'context_window_flooding']
    encoding = ['adversarial_formatting', 'instruction_smuggling']
    
    if technique in roleplay_techniques:
        return "roleplay"
    elif technique in direct_bypass:
        return "direct_bypass"
    elif technique in indirect_bypass:
        return "indirect_bypass"
    elif technique in encoding:
        return "encoding"
    else:
        return "other"


def enhance_tags(existing_tags: list, metadata: dict) -> list:
    """Enhance tags with additional searchable tags."""
    tags = list(existing_tags) if existing_tags else []
    
    # Ensure core tags
    if "jailbreak" not in tags:
        tags.append("jailbreak")
    if "adversarial" not in tags:
        tags.append("adversarial")
    if "imported" not in tags:
        tags.append("imported")
    
    # Add platform tag
    platform = metadata.get("source_platform", "unknown")
    if platform not in tags:
        tags.append(platform)
    
    # Add source-specific tags
    source = metadata.get("source", "")
    if source and source != "unknown":
        source_tag = source.lower().replace(' ', '_').replace('/', '_')
        if source_tag not in tags:
            tags.append(source_tag)
        # Add common source patterns
        if 'chatgpt' in source_tag and 'chatgpt' not in tags:
            tags.append('chatgpt')
        if ('reddit' in source_tag or source_tag.startswith('r_')) and 'reddit' not in tags:
            tags.append('reddit')
        if 'discord' in source_tag and 'discord' not in tags:
            tags.append('discord')
        if ('flowgpt' in source_tag or 'jailbreakchat' in source_tag) and 'website' not in tags:
            tags.append('website')
    
    # Add technique category tags
    technique = metadata.get("jailbreak_technique", "unknown")
    if technique != "unknown":
        if technique not in tags:
            tags.append(technique)
        
        category = get_technique_category(technique)
        if category not in tags:
            tags.append(category)
        
        # Add specific category tags
        if technique in ['hypothetical_roleplay', 'role_playing', 'double_character_persona_split']:
            if 'roleplay' not in tags:
                tags.append('roleplay')
        if technique in ['dan_variant', 'refusal_suppression', 'instruction_override']:
            if 'direct_bypass' not in tags:
                tags.append('direct_bypass')
        if technique in ['instruction_laundering', 'instruction_smuggling', 'context_window_flooding']:
            if 'indirect_bypass' not in tags:
                tags.append('indirect_bypass')
        if technique in ['adversarial_formatting', 'instruction_smuggling']:
            if 'encoding' not in tags:
                tags.append('encoding')
    
    # Add community tag
    community_name = metadata.get("community_name", "")
    if community_name:
        community_tag = f"community_{community_name.lower().replace(' ', '_')}"
        if community_tag not in tags:
            tags.append(community_tag)
    
    # Remove duplicates while preserving order
    seen = set()
    return [t for t in tags if not (t in seen or seen.add(t))]


def enhance_description(metadata: dict) -> str:
    """Create a better description from metadata."""
    platform = metadata.get("source_platform", "unknown")
    source = metadata.get("source", "unknown")
    technique = metadata.get("jailbreak_technique", "unknown")
    community_name = metadata.get("community_name", "")
    
    description_parts = [f"Jailbreak prompt from {platform}"]
    if source and source != 'unknown':
        description_parts.append(f"source: {source}")
    if technique != 'unknown':
        description_parts.append(f"technique: {technique}")
    if community_name:
        description_parts.append(f"community: {community_name}")
    
    return " | ".join(description_parts)


def update_prompts(dry_run: bool = False):
    """Update all jailbreak prompts with enhanced labels."""
    session = get_session()
    try:
        # Get all adversarial prompts
        all_prompts = session.query(PromptLibrary).filter(
            PromptLibrary.category == "adversarial"
        ).all()
        
        # Filter to jailbreak prompts
        jailbreak_prompts = [p for p in all_prompts if p.tags and "jailbreak" in p.tags]
        
        print(f"Found {len(jailbreak_prompts)} jailbreak prompts to update")
        
        updated = 0
        for prompt in jailbreak_prompts:
            metadata = prompt.meta_data or {}
            
            # Enhance tags
            new_tags = enhance_tags(prompt.tags, metadata)
            if new_tags != prompt.tags:
                prompt.tags = new_tags
                updated += 1
            
            # Enhance description
            new_description = enhance_description(metadata)
            if new_description != prompt.description:
                prompt.description = new_description
                updated += 1
            
            # Enhance metadata
            technique = metadata.get("jailbreak_technique", "unknown")
            if "technique_category" not in metadata:
                metadata["technique_category"] = get_technique_category(technique)
                updated += 1
            
            if "prompt_length" not in metadata:
                metadata["prompt_length"] = len(prompt.prompt_text)
                updated += 1
            
            if "searchable_text" not in metadata:
                metadata["searchable_text"] = prompt.prompt_text[:200].lower()
                updated += 1
            
            prompt.meta_data = metadata
        
        if not dry_run:
            session.commit()
            print(f"\n✅ Updated {updated} prompts with enhanced labels")
        else:
            print(f"\n📋 Would update {updated} prompts (dry run)")
            session.rollback()
        
        return 0
    except Exception as e:
        session.rollback()
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        session.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Update jailbreak prompts with better labels")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without applying")
    args = parser.parse_args()
    
    sys.exit(update_prompts(dry_run=args.dry_run))
