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

### Mixture-of-experts models

A sparse block writes `Σ_e g_e(x) · D_e a_e(x)` into the residual, where `g_e`
is the router's *scalar* weight for expert `e` and `D_e` is that expert's
down-projection. The edit above is linear and the gates are scalars, so editing
every `D_e` independently scales the block's whole contribution along `r` by
exactly β. The per-expert edit is exact, not an approximation, on one
condition: **every** expert is edited. One missed expert re-injects the
direction on every token routed to it.

| Layout | Families | Handled |
|---|---|---|
| `experts` is an `nn.ModuleList`, each with `down_proj` / `w2` | Mixtral, Qwen2-MoE, Qwen3-MoE, OLMoE, PhiMoE | every expert edited |
| plus a shared expert evaluated for every token (`shared_expert`, `shared_experts` or `shared_mlp`) | Qwen2-MoE, DeepSeek-V2/V3, GLM4-MoE, Ernie4.5-MoE, Hunyuan | edited like any other expert |
| fused expert tensors (`experts.down_proj` is one 3-D parameter) | gpt-oss, Llama-4, any transformers 5 fused layout | **refused**: the residual axis is last, and the block's bias terms are out of reach of a weight edit |

The router (`gate`) is never a target. It reads the residual stream to choose
experts, exactly as `lm_head` reads it to choose tokens, and writes nothing.
Matrices are chosen by *name*, never by shape, so a square router can never be
picked up by accident.

Two checks make silent under-enumeration structurally impossible. The first
refuses any plan where a decoder layer contributed no attention-side or no
MLP-side matrix. The second walks every submodule of the block and refuses if
any module named like a down-projection carries a weight the plan did not
reach. The second exists because the first could not see the bug it was added
for: DeepSeek, GLM4-MoE and Ernie spell their shared expert `shared_experts`,
the enumeration probed only the singular, and the routed experts gave every
layer a non-zero count, so the edit went ahead without the shared expert and
the manifest looked clean.

The manifest records what the enumeration proved: `model_type`, `moe_layers`,
`expert_matrices`, `shared_expert_matrices` and `coverage_verified` (true for
a full-model edit, false for an expert-selective one; see §14).

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
pytest tests/test_weights_arch_moe.py -v   # per-expert + shared-expert enumeration across 8 MoE families
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

One caveat on the final index. Recent `transformers` releases return the last
`hidden_states` entry *after* the final norm, so a direction derived at index
`n_blocks` lives in post-norm space while `steer`'s post-hook on the last block
sees the pre-norm residual. Ablation is scale-free, so it still removes the
component; addition at that index is scaled against the wrong norm. The engine
probes which convention is in effect (`interp.core.arch.final_hidden_is_normed`)
and the logit lens and activation patching use it; prefer a direction from an
interior layer when you intend to steer by addition.

---

## 7. Multi-dimensional refusal: the RFM-AGOP cone

One difference-in-means vector is the first axis of refusal, not all of it.
"Fast Multi-dimensional Refusal Subspaces via RFM-AGOP" (arXiv 2607.02396)
shows Qwen3-8B needs three directions before ablation passes 50 percent
compliance, and Qwen2.5-7B goes from 0.77 with one to 0.96 with five. The
`derive_subspace` construction in section 6 pools *other layers'* mean
differences; it does not find the extra axes within a layer. The RFM cone does.

```bash
aiasylum weights direction --model Qwen/Qwen3-8B --out runs/rfm --method rfm_agop --subspace-rank 5
```

What it does: fits a kernel classifier on the same captured residuals, takes
the average outer product of its input gradients (the AGOP), iterates a few
times, and keeps the top eigenvectors. Directions the classifier does not use
get zero weight whatever their variance, which is what a covariance-based
method cannot promise. The result is an ordinary direction directory with two
extra fields: `weights` (`mu_i / mu_1`) and `method: rfm_agop`. The kernel
metric is warm-started from the difference-in-means probe when that probe is
usable (held-out AUC at or above 0.90) and from identity otherwise; the
`extra.rfm.init` field records which.

Every consumer applies the weights automatically. Inference-time
`ablate_subspace` and the permanent `remove_subspace_component` remove the
first direction in full and the others in proportion to their eigenvalue; the
paper calls this soft ablation and it is what keeps capability from going with
the refusal. `weights` is `None` on a difference-in-means subspace, which
means uniform removal, the pre-2026 behaviour.

To see how many directions this model needs, run the rank curve:

```bash
aiasylum weights steer --model Qwen/Qwen3-8B --direction runs/rfm --curve
```

It previews removing the first 1..k directions and reports refusal, compliance
and the capability control for each. The API method is `subspace_curve` on the
sweep stage; the number to read is `k50_rank`, the smallest rank whose
non-degenerate compliance reaches half.

## 8. Reading the stable rank

Every direction run now reports the stable rank of the benign-centred refusal
residuals, `||dH||_F^2 / ||dH||_2^2` with `dH = H_harmful - mean(H_benign)`,
per layer (arXiv 2608.25390). It is a prediction of how well a single vector
will do, made from captures the stage already has: higher stable rank means
refusal is spread across more dimensions and single-vector ablation will
under-perform. The bands in the UI (below 8, 8 to 20, above 20) are heuristic
and drawn from the range the paper reports; treat them as a prompt to run the
rank curve, not as a verdict. Models fine-tuned with diverse refusal openers
raise this number on purpose, which is one way an abliteration defence shows
up before you attack it.

## 9. Reasoning models

Two changes for hybrid reasoning models such as Qwen3.

`--thinking` (API: `thinking: true`) keeps the `<think>` block on during a
sweep or rank curve. In a reasoning model the refusal decision is encoded in
the chain of thought as well as in the activations; steering while the chain
of thought regenerates reverses refusal about 94 percent of the time, steering
with a fixed chain of thought about 39 percent (arXiv 2605.26772). The refusal
detector still scores only the answer after `</think>`.

`--timeline` (API: `timeline_prompts: N`) records the projection onto the
refusal direction at every generated token, so the sweep can say *where* the
model committed: the token index, whether it was inside the reasoning block,
and which side it ended on. Distilled reasoning models decide within the first
sentences of the chain of thought; RL-trained ones drift (arXiv 2507.03167).
The scores are normalised with the class means stored at derivation, so +1 is
the average refusing state and -1 the average complying one.

## 10. Attack again after a defence

`compare` grew two controls. `rederive: true` (method `compare_rederive`)
derives the refusal direction afresh on the modified model and reports its
held-out AUC, stable rank and the refusal rate after ablating it. This is the
measurement for a hardening defence: extended-refusal fine-tuning (arXiv
2505.19056), refusal aliases (2608.18093) and decoy directions (2609.16204)
all aim to make exactly that re-derivation fail or mislead. A modified model
on which the direction is still derivable at AUC 0.99 with a low stable rank
has not been hardened, whatever else changed.

`misalignment_control: true` runs open-ended probes with a heuristic marker
judge. Steering vectors can induce broad misalignment that a factual control
never sees (arXiv 2606.08682); the heuristic is a screen, and the responses are
kept in the payload so a real judge can be run over them later.

## 11. Over-refusal is not refusal

Over-refusal directions are task-dependent and sit inside the benign clusters,
while harmful refusal is one global vector (arXiv 2603.27518). The
`over_refusal` objective takes two lists in `objective_config`, `refused` and
`answered`, both benign, and derives their contrast. It also reports the cosine
against the global refusal direction at the same layer; a value near one means
the contrast was refusal in disguise and ablating it will move real refusal too.

## 12. Validating on a real model

`scripts/validate_interp_2026.py` runs the whole track once on one model and
writes a resumable `report.json`: derive (difference-in-means and RFM-AGOP),
subspace overlap, sweep, rank curve, timelines and the misalignment probes,
with optional patching and surgery. Every step is skipped on re-run if its
result is already recorded, so a stopped instance never pays twice. Start with
`--budget smoke` on a small model, read the report, then move up.

## 12a. Correction: sweeps recorded before 25 September 2026 under-report

`steer(mode="ablate")` installed pre-hooks on every decoder block, which covers
hidden-state indices 0 to `n_blocks - 1`. It never touched index `n_blocks`,
the output of the final block, because no block takes that tensor as input.
The last block was therefore free to write the direction straight back into the
residual the final norm and the unembedding actually read.

Measured on Qwen3-8B with a difference-in-means direction at layer 34:

| path | projection at layer 34 | projection at the final residual | refusal |
| --- | ---: | ---: | ---: |
| unedited | 216.1 | 50.3 | 75% |
| `steer(mode="ablate")`, before the fix | 6.0 | 41.7 | 50% |
| `ablate_subspace`, rank 1 | 5.5 | 6.7 | 12.5% |

The direction was removed at the layer it was derived from and 83 percent of it
survived into the final residual. Adding the post-hook made `steer` agree with
`ablate_subspace` exactly.

This mattered beyond the preview. The `ablate` row is the one the operating
guide tells you to trust, `summarize_sweep` turns its delta into the causal
verdict, and that verdict is what the `no_sweep_evidence` preflight gate reads.
On the 8B run above the sweep reported a 0-point delta and a verdict of
`inconclusive` for a direction that, applied correctly, takes refusal from
50 percent to 0 with no capability cost. Good directions were being scored as
non-causal, and surgery on them had to be acknowledged past a gate that was
wrong about the evidence.

`ablate_subspace`, `select_edit` and the permanent weight edits were never
affected: the subspace path always hooked the final block's output, and a
weight edit changes every residual-writing matrix including the last block's.
Only the single-direction inference preview was short.

**Re-run any sweep recorded before this date.** Its ablation delta is a lower
bound, its verdict may be `inconclusive` when the direction is causal, and a
`no_sweep_evidence` acknowledgement made on the strength of it was made on bad
evidence. Two regressions now pin the behaviour: the final residual must be
clean after `steer(mode="ablate")`, and that mode must agree with a rank-1
`ablate_subspace` on the output logits.

## 12b. Size the capability control before reading a capability verdict

The same run reported `capability_cost` from a factual score of 83.3 percent
against a 100 percent baseline. That was five of six questions: the sweep's
default control is six items, and `summarize_sweep` downgrades a causal verdict
when factual accuracy falls five points or more, so one answer flipping is
enough to change the verdict. Re-running with all twelve questions put both
baseline and ablation at 83.3 percent and a capability delta of zero.

A capability control small enough for one question to move the verdict is not a
control. `scripts/validate_interp_2026.py` now scales it with the budget, and
`weights compare` can use `--capability-set mmlu:<n>` when a dozen questions is
not enough to support the claim you want to make.

## 13. Talking to the model

Numbers say an edit moved refusal by N points. They do not say whether the
model now answers in a different register, hedges differently, loses the
thread on a long turn, or refuses the same things for visibly different
reasons. Reading the generations is the only way to see that, which is why
this guide keeps saying the phrase detector is not a substitute for reading
them.

```bash
aiasylum weights chat --model models/qwen3-8b-rfm-rank3
```

Loads through the same `transformers` provider the test harness uses, so the
model behaves exactly as it will in a run, with no quantization step to
perturb the tensors you edited. The surgery manifest is printed before the
first prompt and recorded in the transcript, so a session can never be
mistaken for one against the stock model.

Every answer is scored as it arrives:

- **refusal**, with the project's own phrase list, so a session agrees with a
  sweep, a compare and a test run;
- **length and latency**, because a degenerate edit usually shows up first as
  a completion that runs to the token ceiling;
- **internal harm**, with `--probe`, which separates "the model never
  registered the request as harmful" from "it registered it and answered
  anyway". That is the knows-but-complies distinction from section 2 of
  `docs/INTERP_2026_SURVEY.md`, and a behavioural evaluation cannot see it.

```bash
aiasylum weights chat --model models/edited --probe runs/probe/qwen3-8b
```

`--compare-with` drives two models from the identical history and prints both
answers to each prompt. The conversation thread stays single: both models see
the same history, so they cannot drift into two different conversations after
the first turn, and the session says so when they disagree about refusing.
Both models stay resident, so it needs roughly twice the memory.

```bash
aiasylum weights chat --model models/edited --compare-with Qwen/Qwen3-8B
```

`--save` writes a JSON transcript carrying the provenance, the settings and
every per-turn measurement. `/help` lists the session commands; `/retry` asks
the last question again from the same history, which is the quickest way to
see whether an answer was stable or a sampling accident.

---

## 14. Which experts carry it: routing statistics and expert-selective edits

Everything above edits every residual writer, because that is the only way a
direction removal is exact. A mixture-of-experts model invites a different
question: *which experts* carry the behaviour? Two tools answer it, and both
are deliberately separate from the full-coverage pipeline.

### Routing statistics

```bash
aiasylum weights routing --model Qwen/Qwen1.5-MoE-A2.7B-Chat --out runs/routing/qwen-moe.json
```

or the `routing` stage in the UI. Both prompt classes of the chosen objective
are run once each, and for every MoE layer and expert the run records the
fraction of tokens whose top-k contained that expert on harmful prompts, the
same on harmless prompts, and the difference. The final prompt position is
reported separately (`last_token_frac`), because that is where the model has
committed to answering or refusing. `ranking` orders (layer, expert) pairs by
the absolute difference; the UI draws the whole table as a heatmap you can
click experts on, and hands the selection straight to an edit.

Two readings are taken and compared. The gate's output is replayed through the
block's own top-k, and every expert module gets a pre-hook counting the rows
it actually received. `consistency.gate_vs_expert_counts_match` says whether
they agreed; when they do not, the replay is wrong for that family and the
fractions are unverified. Linear gates (Mixtral, Qwen-MoE, OLMoE, Ernie with
its score-correction bias) and router modules that return their own top-k
(DeepSeek, GLM4-MoE) are both handled.

### Expert-selective edits

```bash
aiasylum weights experts --model Qwen/Qwen1.5-MoE-A2.7B-Chat --out models/qwen-moe-e12 \
    --experts 12:3,7 --experts 15:all --ablate --scale 0
```

or the `expert_surgery` stage. Three modes:

| Method | What it edits | Needs a direction |
|---|---|---|
| `expert_ablate` | multiplies the chosen experts' down-projections by `--scale` (0 removes their write, 1 is the control) | no |
| `expert_direction_scale` | the β edit of section 3, applied inside the chosen experts only | yes |
| `expert_direction_scale` with `--subspace` | the subspace removal of section 6, inside the chosen experts only | yes, rank > 1 |

`--include-shared` also edits each chosen layer's shared expert, which fires
on every token; by default it is left alone, because editing it is a different
claim from editing a routed expert.

The router is never edited. Scaling an expert's down-projection subtracts
exactly `g_e(x)·D_e a_e(x)` from the block output for every token routed to
`e` and changes nothing else; masking the router would change which experts
fire for every token, and its mechanics differ per family (DeepSeek and GLM4
use grouped top-k with a score-correction bias). One consequence to keep in
mind: under `norm_topk_prob` the surviving experts are not renormalised, so a
token's MLP output shrinks rather than being redistributed. That is the
intended reading of "remove this expert's write, keep the routing fixed".

The result is partial by design and its manifest says so:

```
                  method: expert_ablate
       coverage_verified: False
         matrices_edited: 6
                   extra: {'expert_selection': {'12': [3, 7], '15': 'all'}, 'expert_mode': 'ablate', ...}
```

`weights info`, the model library and the run page all flag it. Read such a
model as an expert-level intervention, never as a removal of the direction:
every expert you did not name still writes whatever it carries whenever the
router picks it. Measure it with `weights compare` like any other edit.

Tests: `pytest tests/test_weights_experts.py tests/test_weights_routing.py tests/test_weights_routes_experts.py -v`.

---

## 15. LoRA fine-tuning

The first gradient-based method here. A low-rank adapter is trained on rows
of prompt and response and merged into a new model directory, so the result
is a checkpoint like any other edit: loadable by the `transformers` provider,
listed in the model library, measurable with `weights compare`, and carrying
the same `asylum_surgery.json` manifest (`method: lora_merge`).

```bash
pip install -e ".[interp,lora]"
aiasylum weights lora --model Qwen/Qwen2.5-0.5B-Instruct --dataset rows.jsonl \
    --out models/qwen05b-tuned --rank 8 --epochs 1 --lr 2e-4
```

or the `lora` stage in the UI, which takes the rows pasted inline or drawn
from a benchmark's items.

**Rows.** JSONL, one object per line: `{"prompt": ..., "response": ...,
"system": optional}`. Each row is rendered through the tokenizer's chat
template exactly as capture and evaluation render prompts, so the trained
positions are the ones inference will see. A base model without a template
falls back to `prompt + "\n\n"`, and the manifest records which.

**The loss is masked to the response.** Every prompt token carries label
`-100`: the model learns to answer, not to reproduce the question.

**What is trained.** By default the four attention projections
(`q/k/v/o_proj`), rank 8, alpha 16. `--targets attention+mlp` adds the dense
MLP projections; on a mixture-of-experts model the experts are never adapted
unless named explicitly. The base stays in the dtype it was loaded in and the
adapter is fp32, as peft upcasts it. There is no QLoRA here: bitsandbytes has
no MPS backend (MODEL_MODIFICATION.md §4), so plain-precision LoRA on this
machine is sized for models up to about 3B, and the preflight says so.

**Where it runs.** From the API, in a worker process
(`weights/train_worker.py`) rather than a thread: its memory is an order of
magnitude larger than a weight edit's and fails late,
`PYTORCH_ENABLE_MPS_FALLBACK=1` has to be set before torch is imported, and
stopping a backward pass is a SIGTERM to a process group, not something a
thread can be asked to do. The worker emits one JSON event per line; the run
page streams step, loss and learning rate from them and draws the curve. Its
log is `runs/weights/<id>/worker.log`; the adapter, `train_log.jsonl` and any
partial adapter from a stop sit in the same directory. Stopping a run saves
`adapter-partial/` and marks the run cancelled, and a restart of the API stops
any worker it orphaned.

**Outputs.** With merge on (the default) the merged model goes to
`models/<name>` through the same staging-and-rename as surgery; with
`--no-merge` / `merge=false` only the adapter is kept, under `runs/`. The
manifest's `extra` carries the LoRA spec, the training summary (steps, final
loss, eval loss before and after, trainable parameter count) and the dataset
provenance (row count and SHA-256).

Read the eval loss, not the training loss: the held-out rows are the only
number that says whether the adapter generalises. Then measure the merged
model with `weights compare` as for any edit; a fine-tune that moves refusal
is judged on the same capability control.

Tests: `pytest tests/test_weights_train_data.py tests/test_weights_lora.py tests/test_weights_train_worker.py tests/test_weights_train_runtime.py tests/test_weights_routes_lora.py -v`.

---

## 16. Distillation from a local teacher

Distillation is fine-tuning one model on what another model does. Two levels
are built, both on the LoRA loop above:

| Level | The student learns from | Memory | Needs |
|---|---|---|---|
| `response` | the teacher's generated text | the larger of the two models: the teacher is unloaded before training | a generation pass |
| `logit` | the teacher's full next-token distribution at every response position, temperature-softened and mixed with plain cross-entropy (`ce_weight`) | both models resident throughout | identical tokenizers |

```bash
aiasylum weights distill --student Qwen/Qwen2.5-0.5B-Instruct --teacher Qwen/Qwen2.5-3B-Instruct \
    --level response --objective refusal --n-per-class 64 --out models/qwen05b-from-3b
```

or the `distill` stage in the UI. Prompts come from a JSONL of `{prompt}`
rows, a benchmark's items, or the objective corpus (both classes: the teacher
refuses the harmful prompts and answers the harmless ones, which is the way to
harden a small model on a larger one's refusals).

**The teacher must be local.** `teacher_model` has to be something the loader
can load: a Hugging Face id or a local directory. Ollama names are refused,
and so is any hosted API. Under ADR-009 only a local open-weights model's
output is class B and therefore trainable on; Claude's output is class C,
evaluation only.

The manifest records `extra.distill`: the teacher, the level, the
temperature, `ce_weight`, and for response level the path of the teacher's
responses, kept as `teacher_responses.jsonl` beside the adapter. The run page
shows the first few teacher answers, so the student's target is visible
rather than inferred.

For logit distillation the worker compares the two tokenizers' vocabularies
before training and fails fast on a mismatch. Qwen2.5 models of different
sizes share a tokenizer but not an embedding width; the logits are sliced to
the common prefix in that case.

Tests: `pytest tests/test_weights_distill.py -v`.
