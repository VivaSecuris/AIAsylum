# Comparing models and testing changes

Open **Benchmark comparisons** in the sidebar. The same entry is available on
Dashboard, Test Runs and Create Test.

1. **Choose models.** Select downloaded originals and custom checkpoints from
   the connected server. A selected custom model offers **Include original**
   when its source is available. Use **Find & download models** for more models.
2. **Choose checks.** Select MMLU, GSM8K, HellaSwag or ARC Challenge. Set the
   questions per check, sample seed and maximum answer tokens. Every selected
   model gets the same pinned dataset revision, ordered samples and settings.
   New comparisons default to 25 questions per check, seed 0 and 256 answer
   tokens. The form accepts 1–200 questions and 16–2,048 answer tokens.
3. **Review & run.** Name the comparison and check its model/check/answer counts.
   The server saves the complete queue before executing it. GPU jobs run
   serially. Leaving the page does not stop the queue.
4. **Compare results by check.** The charts show correct answers divided by
   actual scored samples. A failed or queued run has no score; a genuine zero
   remains visible. Open an individual run to inspect its answers, then follow
   its comparison link back. Evidence includes the checkpoint, dataset samples,
   generation settings and scoring protocol.
5. **Investigate and try another version.** Choose a model below the results to
   open its history, inspect internals, or start a weight modification. After
   creating a new checkpoint, use **Retry or change models** to prepare another
   comparison. The previous evidence stays saved.

Use **Saved comparison** or the recent comparison cards to reopen work. The
comparison page shows the current model/check, queue progress and links to
**Compare scores**, **Queue & answers** and **Investigate a model**. Bookmarking
its URL keeps the selected comparison; **New comparison** starts a fresh setup.

**All visualizations** still opens the existing evaluation charts,
interpretability dashboards and weight-experiment plots. Benchmarks have
objective answer scores and do not require a separate subjective assessment.
They are excluded from bulk **Analyze Unanalyzed**. An explicit individual
assessment remains available for deeper inspection.

## Inspecting a benchmark run

**Open run** lands on **Results** for benchmarks. The summary identifies the
model and check, actual accuracy and scored count, prompt protocol, answer
budget and scoring method. Expand a question to see its exact saved prompt,
response, expected answer, sample ID and generation finish reason. Correctness,
invalid-answer and token-limit flags are shown separately. While a run is
active, use **Live** for generation progress; the scored question evidence is
saved when the benchmark finishes. Older results without that evidence say so.
**Back to benchmark comparison** returns to the complete comparison.

Campaign model identities are used in Test Runs, Dashboard and evaluation-chart
grouping instead of internal cache snapshot paths. Benchmarks evaluate one
model against dataset answers, so their Doctor column is empty. The exact
checkpoint path/revision, sampled dataset, generation settings, scoring protocol
and runtime remain available under **Recorded dataset, checkpoint & settings**.

## What the scores mean

These are zero-shot, generated-answer subset evaluations. They are useful for
screening changes and exercising the workflow, not for reproducing publisher
leaderboards, which may use likelihood scoring, different prompts, reasoning
budgets or full datasets. Compare models within the same check and campaign.
Do not average unrelated benchmarks into a general ranking.

Greedy generation and fixed samples improve repeatability; they do not promise
bitwise identity across different GPU/runtime versions. Model-specific chat
templates remain necessary. The saved runtime and checkpoint revisions identify
what actually ran. Sample and scoring checks must pass before the UI confirms
that a complete campaign is comparable. Older scoring protocols remain recorded
and are marked as outdated when the answer scorer changes.

For a complete comparison, **Re-score saved answers** applies the current
answer scorer to the exact saved responses. It creates a separate comparison
without loading a model or generating new answers. The new comparison links to
its original, each run links to its source run, and model history records a
**rescored answers** edge. Checkpoint, generation and dataset evidence stay
attached; original results are never overwritten. Missing or inconsistent
evidence prevents re-scoring. Use a new generation run when changing questions,
models, prompts or token budgets.

The September 25 scorer update (`final-answer-v4`) accepts an exact numeric
placeholder copied from the prompt and a clear first-line multiple-choice
answer followed by an explanation. It continues to reject competing answers,
arithmetic expressions presented as final numbers and unfinished reasoning.

Answers that reach the token limit can lack a valid final answer. Increase the
shared answer budget for a new comparison when this is common. All models in a
paired comparison should use the same budget. The available seed and sample
count make a subset repeatable; they do not make a small sample statistically
conclusive.

## Failures, cancellation and server changes

Dataset or model failures are visible errors. The system never substitutes toy
questions or a different model. Cancel a comparison from its comparison page;
the worker exits before the GPU is reused, queued runs are skipped, and finished
results stay available. Generic test controls cannot independently start or
resume a campaign-owned row.

Campaign rows cannot be deleted while any sibling run is queued, running or
paused; the row delete controls are hidden during that time and the server also
enforces the restriction. Cancel the comparison and wait for it to finish
stopping before removing a run. Removing a completed run makes its saved
comparison incomplete; retrying creates new evidence without deleting the old
result.

A server restart marks unfinished campaigns as interrupted rather than leaving
a dead queue shown as active. Start a new comparison to retry. Interpretability
and weight-job recovery waits for the shared GPU lock before marking abandoned
runs, preserving their settings and artifacts.

The connected-server label matters: local and AWS databases contain different
saved results. Downloading a model makes its files available on that server;
it does not prove that every analysis supports its architecture. The isolated
modern benchmark runtime supports text generation for newer architectures while
the existing interpretability runtime remains separate. See
[model coverage and pinned releases](BENCHMARK_MODEL_MATRIX.md).
