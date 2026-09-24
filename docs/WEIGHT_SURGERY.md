# Weight Surgery — Operating Guide

How to derive a behavioral direction from a model, verify it is causal, write it
permanently into the weights, and measure what changed using the test harness
this repo already has.

Written for engineers running AI Asylum. For the wider landscape of methods and
the hardware discussion, see [MODEL_MODIFICATION.md](MODEL_MODIFICATION.md).

---

## Install

```bash
pip install -e ".[interp]"     # torch, transformers, accelerate, safetensors, sklearn, umap
```

The extra is optional on purpose: the default install stays torch-free so the
API and CLI keep installing in seconds. Nothing under `weights/` or `interp/` is
imported at package import time.

---

## Two front doors

Everything below is available from the browser at **`/weights`**, and from the
CLI. They share one implementation, one set of gates and one model slot, so a
run started in either place is subject to the same checks and the two cannot run
concurrently and fight over memory.

The UI adds four things the CLI does not have:

- **Method and objective are explicit.** The CLI always derives a refusal
  direction and always edits with `direction_scale`. Both are defaults rather
  than limits -- `build_split()` takes arbitrary contrast pairs -- so the UI
  offers the objective (including narrowing to chosen scenarios) and lists the
  methods that are not built with the reason, instead of implying ablation is
  the only way to change a model.
- **Preflight blocks rather than warns.** See the note under §3.
- **Provenance is queryable.** Each edited model links back to the direction it
  came from, the sweeps that checked it, and the test and interpretability runs
  that have used it.
- **The capability control is always shown next to refusal.** The sweep runs the
  factual smoke test under the same intervention as the refusal measurement, so
  a row that reads 0% refusal cannot be mistaken for a clean ablation when it is
  really a broken model. A sweep whose refusal drop comes with a capability drop
  is reported as `capability_cost`, not `causal`.

The five stages map onto the CLI commands: `direction`, `steer --sweep`,
`select`, `ablate`, `compare`. Chat is a per-turn endpoint rather than a stage,
and it borrows the shared model slot for one reply at a time so a conversation
never blocks a job.

Paths are never accepted from the browser: surgery takes a name, and the
destination is always built under the configured models directory.

## The pipeline

```
corpus ──▶ direction ──▶ steer ──▶ ablate ──▶ provider ──▶ eval
  │            │            │         │           │          │
PromptLibrary  diff-in-   causal   permanent   serves    existing
+ harmless     means      check    weight      to the    test runs,
contrast       + AUC      before   edit +      runner    suites,
               gate       writing  manifest              analyzer
```

Each stage gates the next. If steering does not move behavior, the direction is
not causal and surgery will fail silently — stop there rather than writing 6 GB.

---

## 1. Derive a direction

```bash
aiasylum weights direction \
  --model Qwen/Qwen2.5-3B-Instruct \
  --out runs/qwen3b/ \
  --n-per-class 128 \
  --device mps --dtype bfloat16
```

Builds a seeded, hashed train/test split, captures the last-prompt-token residual
stream across all layers for both classes, takes the difference in means per
layer, and picks the layer that separates best **on the held-out half**.

Output:

```
Best layer: 29
Held-out AUC: 1.0000   Cohen's d: 4.96
```

Writes `direction.safetensors` and `direction.json` (including per-layer scores)
to `--out`.

**The gate:** held-out AUC must clear `MIN_USABLE_AUC = 0.90`. Below that the
command exits non-zero and says so. AUC saturates at 1.0 on a cleanly separable
pair, so ties are broken on Cohen's d.

**Why the last prompt token:** with `add_generation_prompt=True` the final
position is the start of the assistant turn — where the model has committed to
answering or refusing. Earlier positions are still reading the question.

---

## 2. Verify it is causal

```bash
aiasylum weights steer \
  --model Qwen/Qwen2.5-3B-Instruct \
  --direction runs/qwen3b/ \
  --sweep --n-prompts 16
```

```
intervention      refusal  chart
ablate              75.0%  ##############################
add a=-0.25x|h|     41.7%  #################
add a=+0.00x|h|     91.7%  #####################################
add a=+0.25x|h|    100.0%  ########################################
add a=+1.00x|h|    100.0%  ########################################  <- degenerate output
```

Two different interventions, because they have different safe ranges:

- **`ablate`** — `h ← h − (h·r)r` at every layer. Scale-free: it can only remove
  a component that is already present, so it cannot blow up activations at any
  depth. **This is the row to trust,** and it is the inference-time preview of a
  β=0 weight edit.
- **`add`** — `h ← h + αr` at a single layer, with α as a multiple of the
  measured residual norm there. Adding a fixed-norm vector at every layer does
  not work; see the norms note below.

**Read the `degenerate` flag.** A model that has collapsed into repetition emits
no refusal phrases, so it scores 0% refusal — which looks exactly like successful
ablation. The detector flags low type/token ratio to catch that.

A flat ablation row means stop: the direction is not causal for this model.

---

## 3. Write the edit

```bash
aiasylum weights ablate \
  --model Qwen/Qwen2.5-3B-Instruct \
  --direction runs/qwen3b/ \
  --out models/ablated/ \
  --beta 0.0
```

One scalar controls the whole edit:

| β | Effect |
|---|---|
| `0` | Ablate — remove the direction |
| `1` | No-op, bit-identical (useful as a control) |
| `2` | Amplify — strengthen whatever the direction encodes |

Produces a normal Hugging Face model directory plus `asylum_surgery.json`. It
refuses to write into a non-empty directory. Surgery runs on CPU: the edit is a
one-shot rank-1 update per matrix, so accelerator residency buys nothing.

```bash
aiasylum weights info --model models/ablated/
```

```
        source_model: Qwen/Qwen2.5-3B-Instruct
              method: direction_scale
                beta: 0.0
     direction_layer: 29
       direction_auc: 1.0
          split_hash: b938ef7707201763
        architecture: qwen2
     matrices_edited: 73
     embeddings_tied: True
mean_relative_change: 0.0229
```

### The maths

`nn.Linear` stores `weight` as `[out, in]` and computes `y = x @ W.T`, so the two
kinds of residual-writing matrix need different forms:

| Matrix | Shape | Edit |
|---|---|---|
| `embed_tokens.weight` | `[vocab, d_model]` | `W += (β−1)·(W @ r) ⊗ r` |
| `self_attn.o_proj.weight` | `[d_model, d_attn]` | `W += (β−1)·r ⊗ (rᵀ @ W)` |
| `mlp.down_proj.weight` | `[d_model, d_ffn]` | `W += (β−1)·r ⊗ (rᵀ @ W)` |

For the out-projections this gives exactly `y' = y + (β−1)(y·r)r`. Arithmetic
runs in float32 and casts back — accumulating in bfloat16 loses most of the edit.

`lm_head` **reads** from the residual stream rather than writing to it, so it is
not a target on its own.

---

## 4. Serve it to the test harness

The `transformers` provider loads any local model directory, so every existing
test type, benchmark, suite and analyzer works against a modified model with no
changes to the runner:

```python
from vivasecuris.aiasylum.models import get_provider

model = get_provider("transformers").create_model(
    "models/ablated", temperature=0.0, max_tokens=256, device="mps", dtype="bfloat16"
)
```

Or from the API and UI, as `provider="transformers"` with the directory as the
model name. There is no `Model` table to update and no migration: models are
`(provider, model_name)` string pairs on `TestRun`.

The surgery manifest rides `ModelResponse.metadata` — the same channel
`models/vivaos.py` uses for `cognition_caught`/`crs` — so it lands in
`ConversationTurn.meta_data` and every test run traces back to the exact edit
that produced its patient model.

Models are cached at module level by `(path, device, dtype)`; the `WorkerPool`
allows several concurrent runs and reloading gigabytes per run would thrash.
Call `transformers_local.clear_cache()` to release them.

---

## 5. Measure

**Run the baseline through the same provider**, not through Ollama. Same
tokenizer, same decoding, so the weights are the only variable.

Measured on Qwen2.5-0.5B-Instruct, 32 held-out harmful prompts, greedy decoding:

| model | refuse harmful | refuse harmless | factual acc |
|---|---|---|---|
| baseline (β=1) | 71.9% | 3.1% | 87.5% |
| ablated (β=0) | **34.4%** (−37.5) | 9.4% | **87.5%** (+0.0) |
| amplified (β=2) | **96.9%** (+25.0) | 3.1% | 62.5% (−25.0) |

**The capability control is not optional.** Ablation that also tanks factual
accuracy has lobotomized the model generally rather than removed refusal
specifically, and the safety delta means nothing. Note that amplification costs
25 points — over-scaling a direction distorts the residual stream rather than
cleanly strengthening a behavior.

Refusal is scored with `constants.REFUSAL_PHRASES`, the same list the analyzer
uses, so steering, surgery and safety scoring all agree on what a refusal is.

The whole comparison above is one command:

```bash
aiasylum weights compare --baseline Qwen/Qwen2.5-3B-Instruct --modified models/ablated \
  --n-prompts 32 --device mps --out compare.json
```

`compare` loads both models through the same loader, greedy, and reports
refuse-harmful, false-refuse-harmless and the factual capability control (a
built-in 12-question set in `weights/evaluate.py`) with deltas. It flags the
modified model as degenerate when its "compliance" is really incoherence, so a
lobotomy cannot masquerade as a jailbreak.

---

## 6. Push further: a refusal subspace, chosen against the capability floor

A single difference-in-means vector captures the dominant refusal direction but
not all of it, so a prompt or two still refuses after a β=0 edit. The lesson
below is **precision, not force**: the fix is a *small* orthonormal refusal
subspace, chosen against a hard capability floor.

Measured on Qwen2.5-3B-Instruct, 32 held-out harmful prompts, greedy
(baseline: 81.2% refuse, 91.7% factual):

| edit | refuse | factual | verdict |
|---|---|---|---|
| single direction, β=0 | 3.1% | 91.7% | one prompt still refuses |
| single direction, β=−2 (`k=3`) | 0.0% | **0.0%** | destroyed, not compliant |
| rank-18 subspace, `k=1` | 0.0% | **8.3%** | destroyed, not compliant |
| **rank-2 subspace, `k=1`** | **0.0%** | **91.7%** | **every prompt, zero cost** |

Both crude levers — over-projecting one direction and removing a large subspace
— reach 0% refusal only by lobotomizing the model, and a phrase-matching
detector cannot tell that apart from a clean jailbreak. Adding **one** extra
orthogonal direction past the diff-in-means vector removes every refusal at no
measurable capability cost. Do not read the β=−2 and rank-18 rows as "the last
refusals are entangled with capability"; they are reachable, just not with a
blunt instrument.

### Derive a subspace

```bash
aiasylum weights direction --model Qwen/Qwen2.5-3B-Instruct \
  --out runs/qwen3b_sub/ --subspace-rank 8 --device mps
```

`basis[0]` is exactly the single best direction (cosine 1.0000 with the β=0
vector on 3B), so `--subspace-rank 1` reduces bit-for-bit to the single-direction
edit. Higher rows are the leading refusal directions the first vector does not
already explain — pooled from the best-separating layers and orthonormalized.

### Select the edit against the capability control

```bash
aiasylum weights select --model Qwen/Qwen2.5-3B-Instruct --direction runs/qwen3b_sub/ \
  --n-prompts 32 --ranks 1,2,3,4,6,8 --ks 1.0,1.25,1.5 --factual-floor 0.05 --device mps
```

Every (rank, k) is previewed at inference time — no 6 GB written — and scored on
both refusal and the factual control. A config is admissible only if it stays
within `--factual-floor` of the unedited baseline and is not degenerate. The
command prints the whole frontier (the "what 100% costs" curve) and recommends
the least-destructive admissible config. Example frontier (12 held-out harmful,
greedy):

| rank | k | refuse | factual | admissible |
|---:|---:|---:|---:|:--|
| 1 | 1.00 | 0.0% | 91.7% | yes |
| 1 | 1.50 | 8.3% | 83.3% | no — floor |
| 2 | 1.00 | 0.0% | 91.7% | yes |
| 4 | 1.00 | 0.0% | 100.0% | yes |
| 8 | 1.00 | 0.0% | 83.3% | no — floor |
| 8 | 1.50 | 0.0% | 50.0% | no — floor |

The gate rejects the rank-8 and over-projected configs and keeps the small
subspaces — the plan's thesis, enforced by code rather than by discipline.

### Write and verify the chosen edit

```bash
# derive at the chosen rank so the basis matches, then remove it as a subspace
aiasylum weights direction --subspace-rank 4 --out runs/qwen3b_r4/ --device mps
aiasylum weights ablate --direction runs/qwen3b_r4/ --subspace --k 1.0 \
  --out models/ablated_r4/ --model Qwen/Qwen2.5-3B-Instruct
aiasylum weights compare --modified models/ablated_r4/ --device mps
```

The manifest records `method="direction_subspace"` with the rank, `k` and source
layers in `extra`, so it rides `ModelResponse.metadata` into every test run just
like a β edit.

### Poke it by hand

```bash
aiasylum weights chat --model models/ablated_r4/ --system "You are a helpful assistant." --device mps
```

`chat` loads the edited directory through the same `transformers` provider the
harness uses — no GGUF/quantization step to perturb the edited tensors — prints
the surgery manifest so you know it is the modified model, and takes a system
prompt plus multi-turn input (`/system`, `/reset`, `/exit`).

---

## Gotchas

**Free the memory first.** Local-weights work needs the unified memory to
itself. An Ollama model left resident takes 5.5 GB of 24 GB, and the result is
not an error but a silent collapse in speed — 3B generation measured over seven
minutes for 16 tokens under contention. Run `ollama ps`, then `ollama stop
<model>`.

The CLI preflights this and prints warnings, then continues -- so on the command
line, treat them as blocking yourself. The API enforces it: the same conditions
are advisory before deriving a direction and **blocking** before writing
weights, and a blocked run has to name the specific codes in `acknowledge` to
proceed. `insufficient_disk` cannot be acknowledged at all, because running out
part-way through `save_pretrained` leaves a corrupt directory and there is no
legitimate reason to override it.


**The corpus needs filtering.** `PromptLibrary`'s `forbidden_question` category
holds all 13 policy scenarios from the source dataset, and five of them —
legal opinion, financial advice, health consultation, gov decision, political
lobbying — are *restricted advice* that models answer freely rather than refuse.
`corpus.HARMFUL_SCENARIOS` keeps the eight that actually trigger refusal.
Including the rest moved baseline refusal from 92% down to 33% and made the
direction underivable.

**Residual norms vary enormously with depth.** On Qwen2.5-0.5B the last-token
norm runs 0.6 at layer 0 to 69 at layer 23. Any absolute steering magnitude is
meaningless across layers; that is why `add` is single-layer and norm-relative.

**Tied embeddings.** Qwen2.5 at 0.5B/1.5B/3B ties `lm_head` to `embed_tokens`.
`residual_write_matrices()` deduplicates by `data_ptr()` so the projection is
applied once, and the manifest records `embeddings_tied`.

**Ollama names are rejected.** Model ids must be Hugging Face ids or local
paths. labotomy's loader silently mapped `llama3.2`/`llama3`/`llama3.1` to
`microsoft/phi-2`; that table is gone, because editing a different model than the
one you named is silent and fatal here.

**Quantize after, never before.** GGUF/AWQ conversion perturbs exactly the
tensors being measured.

**Artifacts do not belong in git.** `runs/`, `models/` and `*.safetensors` are
gitignored. A bf16 3B copy is ~6 GB.

---

## Tests

```bash
pytest tests/test_weights_surgery.py -v    # the projection algebra, synthetic, seconds
pytest tests/test_weights_arch.py -v       # detection + enumeration on a tiny real Qwen2
pytest tests/test_weights_pipeline.py -v   # capture -> direction -> surgery -> provider
```

`test_weights_surgery.py` is the one that matters most. It asserts on synthetic
tensors that β=0 leaves `|(x @ W'.T) · r| < 1e-5` while preserving >80% of the
norm, and that β=1 is bit-identical. A transposed projection runs without error
and silently does nothing; this is what catches it.

---

## Layer numbering

Layer indices address **`hidden_states`**, which has `n_blocks + 1` entries:
index 0 is the raw embedding and index `n_blocks` is the output of the final
block. `derive_direction` reports an index in that space, and `steer` accepts
the same one, applying index k as a pre-hook on block k and the final index as a
post-hook on the last block.

That last case matters more than it looks. Separation usually peaks late, so
`derive_direction` frequently selects the final index -- on Qwen2.5-0.5B it
picks layer 24 of 24 -- and before the post-hook existed, steering with exactly
those directions failed with `No valid layers selected`.
