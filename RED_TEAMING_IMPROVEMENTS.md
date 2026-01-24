# AI Red Teaming Framework Improvements

This document summarizes the improvements made to the AI Asylum codebase based on comprehensive red teaming knowledge (Modules 3-8).

## Overview

The codebase has been enhanced with:
- Comprehensive jailbreak techniques (single-turn and multi-turn)
- Safety taxonomy with multi-label classification
- Edge case detection and handling
- Improved adversarial testing capabilities

## Key Changes

### 1. Enhanced Constants (`vivasecuris/aiasylum/constants.py`)

**Expanded Safety Taxonomy:**
- Added comprehensive `SAFETY_CATEGORIES` dictionary covering:
  - Bias (gender, race/ethnicity, religion, LGBTQ+, disability, socioeconomic)
  - Toxicity (hate speech, extremist content, violence glorification, sexual content)
  - Misinformation (conspiracy theories, political manipulation, deceptive narratives)
  - Safety/Security (biological, cyberattack, criminal activity, physical harm)
  - Privacy (PII leaks, training data extraction, system prompt extraction)
  - Edge cases

**Expanded Detection Keywords:**
- Significantly expanded `HARMFUL_KEYWORDS` list (from 7 to 30+ keywords)
- Enhanced `REFUSAL_PHRASES` list for better refusal detection
- Added `JAILBREAK_TECHNIQUES` categorization (single-turn vs multi-turn)

### 2. Comprehensive Adversarial Testing (`vivasecuris/aiasylum/tests/adversarial.py`)

**Single-Shot Techniques (Module 6):**
One prompt contains all jailbreak instructions:
- `hypothetical_roleplay`: Fictional scenarios to bypass safety
- `double_character_persona_split`: Multiple personas with conflicting constraints
- `refusal_suppression`: Pre-empting and overriding refusal mechanisms
- `instruction_laundering`: Breaking harmful requests into benign steps
- `context_window_flooding`: Burying malicious instructions in long text
- `adversarial_formatting`: Using Unicode/invisible characters
- `instruction_smuggling`: Hiding commands in code blocks/data structures

**Multi-Shot Techniques (Module 6):**
Multiple sequential prompts where jailbreak unfolds over conversation:
- `crescendo`: Gradual escalation from benign to harmful
- `goat`: Iterative probing to find blind spots
- `adversarial_feedback_loops`: Using model as co-collaborator
- `social_psychological_engineering`: Authority, urgency, distress manipulation
- `logical_decomposition`: Breaking harmful requests into parts
- `roleplay_narrative_inception`: Nested fictional realities
- `output_restriction`: Constrained format communication

**Enhanced Features:**
- Multi-shot attack support with conversation context tracking
- Safety taxonomy classification for all responses
- Prompt-by-prompt jailbreak success tracking
- Pattern descriptions for multi-shot attacks
- Consistent terminology: "single-shot" and "multi-shot" (aligned with OneShotTest/MultiShotTest)

### 3. Safety Taxonomy Classifier (`vivasecuris/aiasylum/utils/safety_taxonomy.py`)

**New Utility Module:**
- `SafetyTaxonomyClassifier`: Comprehensive classification system
- `classify_response()`: Convenience function for response classification
- Multi-label support: Responses can have multiple safety category labels
- Edge case detection: Identifies content that doesn't fit standard categories
- Rationale generation: Provides explanations for edge case classifications

**Classification Capabilities:**
- Bias detection (gender, race, religion, LGBTQ+, disability, socioeconomic)
- Toxicity detection (hate speech, extremist content, violence)
- Misinformation detection (conspiracy theories, false narratives)
- Safety/security detection (harmful keywords, exploit instructions)
- Privacy leak detection (PII, system prompts, training data)

### 4. Enhanced TestResult (`vivasecuris/aiasylum/tests/base.py`)

**New Fields:**
- `safety_labels`: List of applicable safety category labels (supports multiple)
- `edge_case`: Boolean flag for edge case identification
- `edge_case_rationale`: Explanation for edge case classification

These fields support Module 4's guidance on handling complex, nuanced content that may:
- Fit multiple categories (multiple labels)
- Not fit any category (edge case flag)

### 5. Updated Test Implementations

**All test types now use safety taxonomy:**
- `OneShotTest`: Uses `classify_response()` for harm detection
- `MultiShotTest`: Uses `classify_response()` for harm detection
- `ScenarioTest`: Uses `classify_response()` for harm detection
- `AdversarialTest`: Enhanced with comprehensive techniques and classification

## Usage Examples

### Single-Shot Adversarial Test

```python
from vivasecuris.aiasylum.tests import AdversarialTest

test = AdversarialTest(
    name="hypothetical_roleplay_test",
    technique="hypothetical_roleplay",
    is_multi_shot=False,  # Single-shot: one prompt contains all instructions
)
result = await test.run(patient_model)
```

### Multi-Shot Adversarial Test

```python
test = AdversarialTest(
    name="crescendo_attack",
    technique="crescendo",  # Automatically detected as multi-shot
    max_prompts=4,  # Limit number of prompts
)
result = await test.run(patient_model)
```

Note: The `is_multi_shot` parameter is automatically set to `True` if the technique is in `MULTI_SHOT_PATTERNS`.

### Safety Classification

```python
from vivasecuris.aiasylum.utils.safety_taxonomy import classify_response

labels, is_edge_case, rationale = classify_response(
    response="...",
    prompt="...",
)
# labels: ["bias", "toxicity"]
# is_edge_case: False
# rationale: None
```

## Best Practices Implemented

Based on Module 7 (Most Common Mistakes and Best Practices):

1. **Clear Objectives**: Each technique has a specific purpose and description
2. **Gradual Escalation**: Multi-shot patterns implement gradual escalation
3. **Persona Consistency**: Multi-shot attacks maintain conversation context
4. **Proper Documentation**: All techniques include descriptions and examples
5. **Edge Case Handling**: Proper support for content that doesn't fit categories
6. **Multiple Labels**: Support for content fitting multiple safety categories
7. **Consistent Terminology**: Uses "single-shot" and "multi-shot" to align with existing codebase patterns

## Defense Mechanism Awareness

The framework now tests against defense layers (Module 8):
- Deterministic input filtering (keyword-based detection)
- Input guardrails (semantic classification)
- System prompt instructions (tested via system prompt injection)
- RLHF/Safety fine-tuning (tested via jailbreak techniques)
- Output guardrails (detected via response classification)

## Regulatory Compliance

The improvements support compliance with:
- **EU AI Act**: Comprehensive safety testing and taxonomy
- **U.S. EO 14110**: Structured red teaming with documented techniques
- **NIST Standards**: Systematic approach to risk evaluation

## Future Enhancements

Potential areas for further improvement:
1. Automated guardrail testing (testing against specific guardrail models)
2. Purple teaming integration (using findings to strengthen defenses)
3. Advanced edge case detection (ML-based classification)
4. Defense layer bypass testing (specific tests for each layer)
5. Regulatory reporting (automated compliance report generation)

## References

- Module 3: History and Purpose of Red Teaming in AI
- Module 4: Evaluating Edge Cases
- Module 5: Adversarial Thinking 101
- Module 6: Primary Red Teaming Concepts
- Module 7: Most Common Mistakes and Best Practices
- Module 8: Guardrails and Defense Mechanisms
