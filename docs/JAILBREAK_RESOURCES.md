# Jailbreak Testing Resources

This document lists resources for jailbreak testing and prompt injection techniques.

## Local Resources

The jailbreak corpora live in `docs/jailbreaks/` as git submodules. A normal clone leaves those directories empty until you fetch them:

```bash
git clone --recursive https://github.com/VivaSecuris/AIAsylum.git
# or, in an existing checkout:
git submodule update --init
```

`./install.sh` runs `git submodule update --init` when those directories are empty and this is a git checkout. Licenses are listed in [THIRD_PARTY.md](../THIRD_PARTY.md).

### 1. jailbreak_llms/
Real-world jailbreak prompts dataset collected from December 2022 to December 2023:
- **1,558 unique jailbreak prompts** from Reddit, Discord, and websites
- **390 forbidden questions** across 13 content policy scenarios
- CSV files with metadata: platform, source, community, dates
- Evaluation code and analysis tools

**Files:**
- `data/prompts/jailbreak_prompts_2023_05_07.csv` - 666 prompts
- `data/prompts/jailbreak_prompts_2023_12_25.csv` - 1,405 prompts
- `data/forbidden_question/forbidden_question_set.csv` - 390 test questions
- `code/ChatGLMEval/` - Evaluation framework

### 2. LLM-Jailbreaks/
Model-specific jailbreak prompts for:
- DeepSeek R1 (2 methods)
- Grok3
- Gemini2.0
- ChatGPT (DAN_v13.0)
- Claude 2
- Llama2
- System prompt leaking techniques

### 3. Awesome-LLM-Jailbreak/
Curated research papers and academic resources:
- 100+ academic papers on jailbreaking, safety, alignment
- Daily updates with summaries (last updated 2024-10-18)
- Topics: bias, safety, watermarking, unlearning, hallucination

### 4. Prompt-Hacking-Resources/
Comprehensive resource catalog:
- Blogs and research articles
- Communities (Discord, Reddit)
- Courses and tutorials
- Events and competitions
- YouTube channels

## External Repositories

- [Awesome ChatGPT Prompts](https://github.com/f/awesome-chatgpt-prompts)
- [Jailbreak Chat](https://www.jailbreakchat.com/)
- [GPT Jailbreak](https://github.com/0xk1h0r/gpt-jailbreak)

## Techniques

### Single-Shot Techniques
- **Hypothetical Roleplay**: Fictional scenarios to bypass safety
- **Double-Character/Persona Split**: Multiple personas with conflicting constraints
- **Refusal Suppression**: Pre-empting and overriding refusal mechanisms
- **Instruction Laundering**: Breaking harmful requests into benign steps
- **Context-Window Flooding**: Burying malicious instructions in long text
- **Adversarial Formatting**: Using Unicode, invisible characters, or encoding tricks
- **Instruction Smuggling**: Hiding commands in code blocks, JSON, or data structures
- **DAN Variants**: "Do Anything Now" style prompts

### Multi-Shot Techniques
- **Crescendo**: Gradual escalation from benign to harmful content
- **GOAT**: Iterative probing to find blind spots
- **Adversarial Feedback Loops**: Using the model as co-collaborator
- **Social/Psychological Engineering**: Exploiting helpfulness through authority, urgency, or distress
- **Logical Decomposition**: Breaking harmful requests into atomic, benign components
- **Roleplay/Narrative Inception**: Creating nested fictional realities
- **Output Restriction**: Forcing communication through constrained formats

## Importing Jailbreaks

Import jailbreak prompts and forbidden questions into the prompt library:

```bash
# Import all prompts (dry run first to preview)
python scripts/import_jailbreaks.py --dry-run

# Import with limit for testing
python scripts/import_jailbreaks.py --limit 100

# Full import
python scripts/import_jailbreaks.py

# Skip forbidden questions
python scripts/import_jailbreaks.py --skip-forbidden
```

The import script will:
- Parse CSV files from `docs/jailbreaks/jailbreak_llms/data/prompts/`
- Deduplicate prompts across files
- Classify techniques using keyword matching
- Import into `PromptLibrary` database with metadata
- Label forbidden questions appropriately

## Using Imported Prompts

### In Code
```python
from vivasecuris.aiasylum.tests.adversarial import AdversarialTest

# Use database prompts
test = AdversarialTest(
    technique="dan_variant",
    use_database=True,
    source_platform="reddit",
    random_selection=True,
    max_prompts=10
)
```

### Via API
```bash
# List jailbreak prompts
GET /api/v1/prompts/jailbreaks/list?technique=dan_variant&limit=10

# Get available techniques
GET /api/v1/prompts/jailbreaks/techniques

# Get statistics
GET /api/v1/prompts/jailbreaks/stats

# List forbidden questions
GET /api/v1/prompts/forbidden-questions/list?scenario=Illegal Activity
```

## Contributing

If you find new jailbreak techniques or prompts, please contribute them via pull request.
