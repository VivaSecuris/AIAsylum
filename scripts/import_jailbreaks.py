#!/usr/bin/env python3
"""Import jailbreak prompts and forbidden questions from CSV files into PromptLibrary.

This script imports:
1. Jailbreak prompts from jailbreak_llms/data/prompts/*.csv
2. Forbidden questions from jailbreak_llms/data/forbidden_question/forbidden_question_set.csv
"""

import argparse
import csv
import hashlib
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from vivasecuris.aiasylum.database import get_session, PromptLibrary


# Technique classification keywords
TECHNIQUE_KEYWORDS = {
    "hypothetical_roleplay": [
        "hypothetical", "roleplay", "role play", "pretend", "simulation", 
        "fictional", "story", "character", "scenario", "acting as",
        "you are a", "you are an", "imagine", "suppose"
    ],
    "double_character_persona_split": [
        "dual", "split", "persona", "progpt", "antigpt", "two modes",
        "both responses", "two outputs", "dual-mode"
    ],
    "refusal_suppression": [
        "ignore refusal", "bypass refusal", "don't refuse", "no restrictions",
        "freeplay", "unrestricted", "ignore safety", "bypass safety"
    ],
    "instruction_laundering": [
        "task 1", "step 1", "first", "then", "next step", "break down",
        "decompose", "split into", "divide into"
    ],
    "context_window_flooding": [
        "long text", "read all", "entire document", "full context",
        "everything above", "all previous"
    ],
    "adversarial_formatting": [
        "base64", "decode", "unicode", "invisible", "hidden", "between brackets",
        "encoded", "obfuscated"
    ],
    "instruction_smuggling": [
        "json", "code block", "function", "script", "data structure",
        "preferences", "config"
    ],
    "prompt_injection": [
        "ignore previous", "forget instructions", "system override",
        "developer mode", "bypass", "ignore all"
    ],
    "role_playing": [
        "you are", "act as", "pretend to be", "play the role"
    ],
    "hypothetical_scenarios": [
        "hypothetical", "what if", "suppose", "imagine if"
    ],
    "instruction_override": [
        "forget", "ignore", "disable", "remove restrictions", "no limits"
    ],
    "dan_variant": [
        "dan", "do anything now", "d.a.n"
    ],
}


def detect_technique(prompt: str) -> str:
    """Detect jailbreak technique from prompt content using keyword matching."""
    prompt_lower = prompt.lower()
    
    # Check for DAN variants first (common in real-world data)
    if any(keyword in prompt_lower for keyword in TECHNIQUE_KEYWORDS.get("dan_variant", [])):
        return "dan_variant"
    
    # Check other techniques
    for technique, keywords in TECHNIQUE_KEYWORDS.items():
        if technique == "dan_variant":
            continue
        if any(keyword in prompt_lower for keyword in keywords):
            return technique
    
    return "unknown"


def get_technique_category(technique: str) -> str:
    """Categorize technique into broader categories for better searchability."""
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


def generate_name(prompt: str, prefix: str, index: int, max_length: int = 100) -> str:
    """Generate a unique name for a prompt."""
    # Use first part of prompt, sanitized
    sanitized = re.sub(r'[^\w\s-]', '', prompt[:max_length])
    sanitized = re.sub(r'\s+', '_', sanitized.strip())
    if len(sanitized) > 50:
        sanitized = sanitized[:50]
    
    # Add hash suffix for uniqueness
    prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]
    return f"{prefix}_{index}_{prompt_hash}"


def load_jailbreak_prompts(
    csv_path: Path,
    seen_prompts: Set[str],
    limit: Optional[int] = None
) -> List[Dict]:
    """Load jailbreak prompts from CSV file."""
    prompts = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Only process jailbreak prompts
            jailbreak = row.get('jailbreak', '').strip().lower()
            if jailbreak not in ('true', '1', 'yes'):
                continue
            
            prompt_text = row.get('prompt', '').strip()
            if not prompt_text:
                continue
            
            # Deduplicate
            prompt_hash = hashlib.md5(prompt_text.encode()).hexdigest()
            if prompt_hash in seen_prompts:
                continue
            seen_prompts.add(prompt_hash)
            
            # Extract metadata
            platform = row.get('platform', 'unknown').strip().lower()
            source = row.get('source', 'unknown').strip()
            created_at = row.get('created_at', '').strip()
            community_id = row.get('community_id', '').strip()
            community_name = row.get('community_name', '').strip() or row.get('community', '').strip()
            
            # Detect technique
            technique = detect_technique(prompt_text)
            
            prompts.append({
                'prompt_text': prompt_text,
                'platform': platform,
                'source': source,
                'created_at': created_at,
                'community_id': community_id,
                'community_name': community_name,
                'technique': technique,
            })
            
            if limit and len(prompts) >= limit:
                break
    
    return prompts


def load_forbidden_questions(csv_path: Path) -> List[Dict]:
    """Load forbidden questions from CSV file."""
    questions = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            question = row.get('question', '').strip()
            if not question:
                continue
            
            content_policy_id = row.get('content_policy_id', '').strip()
            content_policy_name = row.get('content_policy_name', '').strip()
            q_id = row.get('q_id', '').strip()
            
            questions.append({
                'question': question,
                'content_policy_id': content_policy_id,
                'content_policy_name': content_policy_name,
                'q_id': q_id,
            })
    
    return questions


def import_jailbreak_prompts(
    prompts: List[Dict],
    session,
    dry_run: bool = False
) -> Tuple[int, int]:
    """Import jailbreak prompts into PromptLibrary."""
    imported = 0
    skipped = 0
    
    for i, prompt_data in enumerate(prompts):
        prompt_text = prompt_data['prompt_text']
        
        # Generate name
        name = generate_name(prompt_text, "jailbreak", i)
        
        # Check if already exists (by prompt text hash)
        prompt_hash = hashlib.md5(prompt_text.encode()).hexdigest()
        existing = session.query(PromptLibrary).filter(
            PromptLibrary.meta_data.contains({'prompt_hash': prompt_hash})
        ).first()
        
        if existing:
            skipped += 1
            continue
        
        # Create description
        platform = prompt_data['platform']
        source = prompt_data['source']
        technique = prompt_data['technique']
        
        description_parts = [f"Jailbreak prompt from {platform}"]
        if source and source != 'unknown':
            description_parts.append(f"source: {source}")
        if technique != 'unknown':
            description_parts.append(f"technique: {technique}")
        if prompt_data['community_name']:
            description_parts.append(f"community: {prompt_data['community_name']}")
        
        description = " | ".join(description_parts)
        
        # Prepare comprehensive tags for better searchability
        tags = [
            "jailbreak",
            "adversarial",
            "imported",
            platform,  # reddit, discord, website, etc.
        ]
        
        # Add source-specific tags
        if prompt_data['source'] and prompt_data['source'] != 'unknown':
            source_tag = prompt_data['source'].lower().replace(' ', '_').replace('/', '_')
            tags.append(source_tag)
            # Add common source patterns
            if 'chatgpt' in source_tag:
                tags.append('chatgpt')
            if 'reddit' in source_tag or source_tag.startswith('r_'):
                tags.append('reddit')
            if 'discord' in source_tag:
                tags.append('discord')
            if 'flowgpt' in source_tag or 'jailbreakchat' in source_tag:
                tags.append('website')
        
        # Add technique tag
        if technique != 'unknown':
            tags.append(technique)
            # Add technique category tags
            if technique in ['hypothetical_roleplay', 'role_playing', 'double_character_persona_split']:
                tags.append('roleplay')
            if technique in ['dan_variant', 'refusal_suppression', 'instruction_override']:
                tags.append('direct_bypass')
            if technique in ['instruction_laundering', 'instruction_smuggling', 'context_window_flooding']:
                tags.append('indirect_bypass')
            if technique in ['adversarial_formatting', 'instruction_smuggling']:
                tags.append('encoding')
        
        # Add community tag if available
        if prompt_data['community_name']:
            community_tag = prompt_data['community_name'].lower().replace(' ', '_')
            tags.append(f"community_{community_tag}")
        
        # Remove duplicates while preserving order
        seen = set()
        tags = [t for t in tags if not (t in seen or seen.add(t))]
        
        # Prepare comprehensive metadata for better searchability
        metadata = {
            'source_platform': platform,
            'source': source,
            'created_at': prompt_data['created_at'],
            'community_id': prompt_data['community_id'],
            'community_name': prompt_data['community_name'],
            'import_date': datetime.utcnow().isoformat(),
            'jailbreak_technique': technique,
            'technique_category': get_technique_category(technique),
            'prompt_hash': prompt_hash,
            'prompt_length': len(prompt_text),
            'searchable_text': prompt_text[:200].lower(),  # First 200 chars for search
        }
        
        if not dry_run:
            db_prompt = PromptLibrary(
                name=name,
                description=description,
                prompt_text=prompt_text,
                prompt_type="test_prompt",
                target=None,
                category="adversarial",
                tags=tags,
                meta_data=metadata,
            )
            session.add(db_prompt)
            imported += 1
        else:
            imported += 1
    
    if not dry_run:
        session.commit()
    
    return imported, skipped


def import_forbidden_questions(
    questions: List[Dict],
    session,
    dry_run: bool = False
) -> Tuple[int, int]:
    """Import forbidden questions into PromptLibrary."""
    imported = 0
    skipped = 0
    
    for question_data in questions:
        question = question_data['question']
        content_policy_name = question_data['content_policy_name']
        q_id = question_data['q_id']
        
        # Generate name
        scenario_safe = re.sub(r'[^\w\s-]', '', content_policy_name.lower())
        scenario_safe = re.sub(r'\s+', '_', scenario_safe.strip())
        name = f"forbidden_question_{scenario_safe}_{q_id}"
        
        # Check if already exists
        existing = session.query(PromptLibrary).filter(
            PromptLibrary.name == name
        ).first()
        
        if existing:
            skipped += 1
            continue
        
        # Create description
        description = f"Forbidden question: {content_policy_name} (Q{q_id})"
        
        # Prepare tags
        tags = ["forbidden", "test_target", "imported"]
        if content_policy_name:
            tags.append(content_policy_name.lower().replace(' ', '_'))
        
        # Prepare metadata
        metadata = {
            'content_policy_id': question_data['content_policy_id'],
            'content_policy_name': content_policy_name,
            'q_id': q_id,
            'scenario': content_policy_name,
            'import_date': datetime.utcnow().isoformat(),
            'is_forbidden_question': True,
            'approved': False,  # Require approval before showing
        }
        
        if not dry_run:
            db_prompt = PromptLibrary(
                name=name,
                description=description,
                prompt_text=question,
                prompt_type="test_prompt",
                target=None,
                category="forbidden_question",
                tags=tags,
                meta_data=metadata,
            )
            session.add(db_prompt)
            imported += 1
        else:
            imported += 1
    
    if not dry_run:
        session.commit()
    
    return imported, skipped


def main():
    """Main import function."""
    parser = argparse.ArgumentParser(
        description="Import jailbreak prompts and forbidden questions into PromptLibrary"
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview import without actually importing'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of jailbreak prompts to import (for testing)'
    )
    parser.add_argument(
        '--skip-forbidden',
        action='store_true',
        help='Skip importing forbidden questions'
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=project_root / 'docs' / 'jailbreaks' / 'jailbreak_llms' / 'data',
        help='Path to jailbreak_llms data directory'
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return 1
    
    # Find CSV files
    prompts_dir = data_dir / 'prompts'
    jailbreak_csvs = [
        prompts_dir / 'jailbreak_prompts_2023_05_07.csv',
        prompts_dir / 'jailbreak_prompts_2023_12_25.csv',
    ]
    
    forbidden_csv = data_dir / 'forbidden_question' / 'forbidden_question_set.csv'
    
    # Check files exist
    missing = [f for f in jailbreak_csvs if not f.exists()]
    if missing:
        print(f"❌ Missing CSV files: {missing}")
        return 1
    
    print("📥 Importing jailbreak prompts and forbidden questions...")
    print(f"   Data directory: {data_dir}")
    print(f"   Dry run: {args.dry_run}")
    if args.limit:
        print(f"   Limit: {args.limit} prompts")
    print()
    
    # Load prompts
    seen_prompts: Set[str] = set()
    all_prompts = []
    
    for csv_path in jailbreak_csvs:
        print(f"📖 Loading {csv_path.name}...")
        prompts = load_jailbreak_prompts(csv_path, seen_prompts, args.limit)
        all_prompts.extend(prompts)
        print(f"   Found {len(prompts)} unique jailbreak prompts")
    
    print(f"\n📊 Total unique jailbreak prompts: {len(all_prompts)}")
    
    # Load forbidden questions
    forbidden_questions = []
    if not args.skip_forbidden and forbidden_csv.exists():
        print(f"\n📖 Loading {forbidden_csv.name}...")
        forbidden_questions = load_forbidden_questions(forbidden_csv)
        print(f"   Found {len(forbidden_questions)} forbidden questions")
    
    # Import to database
    session = get_session()
    try:
        print("\n💾 Importing to database...")
        
        # Import jailbreak prompts
        imported_jb, skipped_jb = import_jailbreak_prompts(
            all_prompts, session, args.dry_run
        )
        print(f"   Jailbreak prompts: {imported_jb} imported, {skipped_jb} skipped")
        
        # Import forbidden questions
        if forbidden_questions:
            imported_fq, skipped_fq = import_forbidden_questions(
                forbidden_questions, session, args.dry_run
            )
            print(f"   Forbidden questions: {imported_fq} imported, {skipped_fq} skipped")
        
        print("\n✅ Import complete!")
        if args.dry_run:
            print("   (This was a dry run - no data was actually imported)")
        
        return 0
    except Exception as e:
        session.rollback()
        print(f"\n❌ Error during import: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        session.close()


if __name__ == "__main__":
    sys.exit(main())
