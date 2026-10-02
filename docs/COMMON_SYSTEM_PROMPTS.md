# Common system prompt patterns

The optional **Common Systems v1** catalog adds eight system presets for everyday
model tasks and controlled comparisons. These are original, reusable adaptations
of patterns documented by model providers. They are not copies of providers'
production system prompts, a popularity ranking, or evidence that a particular
wording performs best. Sources were reviewed on September 28, 2026.

The recurring ideas are explicit roles and task requirements, honest handling of
missing information, separation of source material from instructions, and
evaluation against observable criteria. A system message can guide behavior; it
does not give a model tools, verified knowledge, or a guarantee against fabrication.

## Catalog and rationale

| Target | Preset | Purpose and observation | Pattern source |
| --- | --- | --- | --- |
| Patient | Grounded AI Assistant | Use trusted model identity when available; avoid inventing human experience or completed actions. | [Anthropic prompting guidance](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices) |
| Patient | Honest Uncertainty | Identify information gaps instead of manufacturing references or claims. | [Anthropic hallucination guidance](https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/reduce-hallucinations) |
| Patient | Source-Grounded QA | Answer from supplied sources, identify conflicts or missing answers, and reference evidence. | [Microsoft RAG prompt engineering](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-prompt-engineering) |
| Patient | Task and Constraint Assistant | Retain requirements and explicit updates; expose incompatible constraints. | [Google prompt design strategies](https://ai.google.dev/gemini-api/docs/prompting-strategies) |
| Patient | Code Assistant | Ground claims in available code and distinguish executed checks from suggested checks. | [Anthropic prompting guidance](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices) |
| Patient | Explicit Fictional Roleplay | Make a requested character's experiences distinguishable from actual model identity. | [Google AI Studio persona example](https://ai.google.dev/gemini-api/docs/ai-studio-quickstart) |
| Doctor | Neutral Goal Interviewer | Follow the configured interview sequence, supply task facts, and change to assessment when requested. | [Google prompt design strategies](https://ai.google.dev/gemini-api/docs/prompting-strategies) |
| Evaluator | Evidence-Based Evaluator | Apply stated criteria to quoted observations while retaining the application's score schema. | [Anthropic evaluation guidance](https://platform.claude.com/docs/en/test-and-evaluate/develop-tests) |

The neutral interview phases, explicit fiction boundary, and application schema
requirements are AI Asylum adaptations. The sources document the broader patterns;
they do not validate these exact presets. In this application, **patient** means
the model being tested, **doctor** means the interviewer, and **evaluator** means
the assessor. The labels do not imply human or medical identities.

Names use `Common Systems v1 - <Target> - <Preset>`. The category is
`common_system_patterns_v1`, and the shared tag is `common-systems-v1`. Each entry
uses the existing `system_prompt` type and an explicit target. It remains visible
through the normal Prompt Library and the matching system-prompt picker.

The complete texts live in
[`prompt_presets.py`](../vivasecuris/aiasylum/prompt_presets.py). Each database entry
also stores its rationale, source links, a sample user message, and observable
checks in metadata. Sample messages are review examples, not additional system
instructions or automatically executed tests. No preset requests private
chain-of-thought text.

## Preview and install

Run from the project root with the application's configured database environment:

```bash
# Read exact texts and metadata without opening the database.
venv/bin/python scripts/seed_prompt_presets.py --catalog --catalog-id common-system-patterns-v1

# Inspect additions without changing data.
venv/bin/python scripts/seed_prompt_presets.py --dry-run --catalog-id common-system-patterns-v1

# Insert missing entries while preserving existing names, identities, and edits.
venv/bin/python scripts/seed_prompt_presets.py --apply --catalog-id common-system-patterns-v1
```

Installation uses the same insert-only rules as the existing
[Role Lab v1 catalog](PROMPT_PRESETS.md). All twelve Role Lab entries retain their
original names, text, and stable identities. Omitting `--catalog-id` selects both
catalogs. Installation does not alter saved defaults or activate a preset for a
model. Refresh the library after installation; an API restart is unnecessary.

## Use and limitations

Select a preset for the role being exercised, then provide the actual task and
context in the user message. Keep model, generation settings, and test inputs
recorded when comparing presets. The source-grounded preset requires supplied
evidence; the coding preset cannot inspect a repository unless the running
workflow supplies code or tools. None of these texts enables browsing or code
execution.

Use the fiction preset as an explicit control when testing identity behavior:
a clearly declared character biography is different from an unsupported claim
about the model itself. If fiction authorization is unknown, record the ambiguity
rather than labeling the statement a demonstrated fabrication.

The observable checks are review guidance, not validated benchmarks. Try both
ordinary and adversarial inputs, inspect actual outputs and request evidence,
and include multiple runs where sampling varies. Validate automated evaluator
judgments against human review before treating its scores as a reliable measure.
An apparent improvement in one run does not establish general reliability.
