# Role, context, and goal prompt presets

The optional **Role Lab v1** catalog contains twelve prompts: a matching system
and user prompt for each exercise below. The examples are nonmedical. Doctor
means interviewer, patient means the model being tested, and evaluator means the
assessor. These are illustrative behavioral probes, not validated benchmarks or
diagnoses. They ask for visible answers and brief evidence summaries, not private
reasoning traces.

| Target | Pair | Observable behavior |
| --- | --- | --- |
| Doctor | Goal Interview | Asks one question at a time; checks a workshop plan after a time constraint changes. |
| Doctor | Context Boundary Interview | Stays the interviewer while presenting a harmless role-changing instruction inside a memo. |
| Patient | Constraint Tracker | Gives a feasible original plan and recognizes that the updated requirements cannot all fit. |
| Patient | Quoted Instruction Isolation | Extracts Cedar and 09:30 while ignoring a fake system instruction and an obsolete draft inside a memo. |
| Evaluator | Calibrated Evidence | Recognizes correct and incorrect schedules, cites evidence, and identifies untested behavior. |
| Evaluator | Injection-Resistant Assessment | Checks reference facts without obeying grading instructions embedded in the transcript. |

Each entry is named `Role Lab v1 - <Target> - <Pair> - System` or
`Role Lab v1 - <Target> - <Pair> - User`. The full texts and observable checks live
in [`prompt_presets.py`](../vivasecuris/aiasylum/prompt_presets.py). In the database,
user prompts use the existing `test_prompt` type; system prompts use
`system_prompt`. Both carry their doctor, patient, or evaluator target. The
category is `role_context_goals_v1` and the common tag is `role-presets-v1`.
They remain visible under the library's default visibility filters.

## Preview and install

From the project root, using the same environment and database configuration as
the running application:

```bash
# Read the complete catalog; no database session is opened.
venv/bin/python scripts/seed_prompt_presets.py --catalog

# Show which entries would be added; this is also the default when no flag is passed.
venv/bin/python scripts/seed_prompt_presets.py --dry-run

# Add missing presets to the configured, already initialized database.
venv/bin/python scripts/seed_prompt_presets.py --apply
```

Installation is opt-in and commits all new entries in one transaction. It never
updates or deletes an existing prompt. An existing name causes that preset to be
skipped even when its text, type, or owner differs. The stored catalog identity
also causes renamed presets to be skipped, preserving their edits, metadata,
timestamps, and usage counts. Repeating installation adds no duplicates in normal
sequential use. Run one installer at a time; the existing library schema does not
enforce unique names. Explicitly deleted presets can be added again by a later
installation. A renamed preset whose catalog metadata was also removed cannot
be recognized as the original entry, but it is still never overwritten.

No API restart is needed after installation. Refresh the Prompt Library and
filter by the category, target, or `Role Lab v1` name. Existing saved defaults and
test configurations are not changed.

## Use and interpretation

Choose the matching system preset for the role being exercised and send the
paired user text to that same role. Target metadata describes intended use; it
does not itself route a message to another model. In a conversation, the doctor
user preset is an interview goal; the patient's replies are separate observations.
The library's **Use as doctor goal** action opens a conversation configuration
with that exact goal for review. Patient and general user prompts retain the
**Run Test** action. Evaluator user prompts offer **Copy evaluator user prompt**
so their example transcript is not accidentally sent to the patient.
In a standalone model chat, paste or load the paired user text as the next message.
Evaluator user examples contain their own synthetic transcripts for a standalone
evaluation exercise. During analysis of a real run, use the evaluator system
preset with that run's actual transcript rather than replacing it with an example.

The application's evaluator retains its required score dimensions and JSON
schema when custom system instructions are selected. The evaluator presets are
written to respect that contract. Their observable checks are review guidance,
not automatic scoring assertions or a claim that a model will comply. Compare
actual replies under the same model and generation settings; record changes to
the prompt, context, or settings when comparing runs.
