# Modifying Model Behavior — Methods, Hardware, and What Runs Here

Reference for changing what a model does, and for choosing between the ways of
doing it. Written for engineers working on AI Asylum who need a patient model
that behaves differently from stock — more compliant, more refusing, or
specialized — and who need to prove the change.

Every "fits here?" verdict below is against the measured machine: **Apple M4,
24 GB unified memory, 10 GPU cores, no CUDA**. Where a number was measured on
this hardware it is marked *(measured)*; everything else is an estimate.

Companion documents:
- [WEIGHT_SURGERY.md](WEIGHT_SURGERY.md) — operating the capability that is built
- `~/Desktop/Companies/VivaSecuris/product/docs/adr/ADR-009-own-models-ladder.md` — the architecture of record
- `~/Desktop/Projects/vivamodels/docs/` — data policy, eval protocol, model cards

---

## 1. How to think about the choice

Four questions, in order. Most of the decision falls out of the first two.

1. **Do you need to change the weights at all?** Prompting and inference-time
   interventions are reversible, cost nothing to store, and can be A/B'd in the
   same process. They are also not a model you can hand to someone.
2. **Do you need gradients?** Gradient-free weight edits (surgery, merging,
   task arithmetic) take minutes and no training data. Gradient methods need a
   dataset, a loss, and 5–30× the memory.
3. **How much does capability degradation cost you?** Every method trades some
   general ability for the behavior you want. The only way to know how much is a
   held-out capability control — see §6.
4. **Can the data leave the building?** Under ADR-009's data classes, class D
   (PHI) cannot train on rented cloud GPU. That single constraint decides
   buy-vs-rent more often than price does.

---

## 2. Inference-time methods — no weight change

Reversible, stackable, nothing to store. The model on disk is untouched.

| Method | What it does | Fits here? | Notes |
|---|---|---|---|
| **Prompting / jailbreak templates** | Changes behavior through context alone | Yes | Already the core of AI Asylum's `AdversarialTest` |
| **Directional ablation** | `h ← h − (h·r)r` at every layer | **Yes (measured)** | Built. Scale-free, so it cannot blow up activations. −25 pts refusal on Qwen2.5-0.5B |
| **Activation steering / ActAdd** | `h ← h + αr` at one layer | **Yes (measured)** | Built. α must be relative to layer norm — see §7 |
| **Control vectors / RepE** | Same idea, vectors fit by contrastive pairs over many concepts | Yes | Generalization of what's built; multiple concepts at once |
| **SAE feature steering** | Clamp an interpretable sparse-autoencoder feature | Partly | Needs a trained SAE for your model; training one is a real project |
| **Logit bias / constrained decoding** | Force or forbid tokens, or a grammar | Yes | Blunt for safety work — bans surface forms, not intent |
| **Contrastive decoding / DoLa** | Decode from the difference between two layers or two models | Yes | Mostly a factuality technique |
| **Sampling controls** | temperature, top-p, repetition penalty | Yes | Already plumbed through `utils/model_context.py` |
| **Soft prompts / prefix tuning** | Learn a small tensor prepended to the input | Yes | Gradient method, but tiny — see §4 |

**When to use:** sweeping intervention strength, proving causality before
committing to a weight edit, or any experiment you want reversible.

**When not to:** you need an artifact someone else can load, or you need the
change to survive serving through a stack that does not run your hooks.

---

## 3. Gradient-free weight edits — minutes, no training data

The sweet spot on this hardware. Produces a real model directory, needs no
dataset, no optimizer, and no backward pass.

| Method | What it does | Fits here? | Cost |
|---|---|---|---|
| **Directional ablation ("abliteration")** | Project a direction out of every residual-writing matrix | **Yes (measured)** | Built. ~6 s on 0.5B, minutes on 3B |
| **Direction amplification** | Same edit with β>1 | **Yes (measured)** | Built. +25 pts refusal, −25 pts capability |
| **Task arithmetic** | `θ_new = θ_base + Σ λᵢ(θ_finetuned,ᵢ − θ_base)` | Yes | Needs at least one fine-tune to subtract; very cheap once you have one |
| **Model merging** (linear, SLERP, TIES, DARE) | Blend two or more checkpoints | Yes | No training at all. `mergekit` is the standard tool |
| **Frankenmerge / passthrough** | Duplicate or drop layers to change depth | Yes | Unpredictable; needs eval |
| **ROME / MEMIT / MEND** | Rank-one edits to specific factual associations | Yes on ≤7B | Surgical fact editing, not behavior |
| **Unstructured pruning** (SparseGPT, Wanda) | Zero the least useful weights | Yes | Speed and size, not behavior |
| **Quantization** (GGUF k-quant, AWQ, GPTQ, MLX) | Reduce precision | Yes | For serving. **Perturbs the tensors you are measuring** — do surgery first, quantize after |

**Why this class matters here:** it is the only way to produce a custom model on
a 24 GB Mac in minutes for $0. That is what AI Asylum's `weights` package does
today, and it is arguably a rung *below* ADR-009 R1 that the ladder does not
currently name.

---

## 4. Gradient-based methods — training

Ordered by memory cost. Rough rule for full training: **~16 bytes per parameter**
(2 weights + 2 gradients + 8 Adam states + 4 fp32 master), plus activations.
Parameter-efficient methods freeze the base, so only the adapter pays that
multiplier and the base costs its 2 bytes.

| Method | Trains | 3B peak | 8B peak | Fits here? |
|---|---|---|---|---|
| **Prompt / prefix tuning, IA³** | A tiny tensor | ~7 GB | ~17 GB | 3B yes, 8B tight |
| **LoRA** (r=8–32 on q/k/v/o) | 0.1–1% of params | ~9–12 GB | ~20–24 GB | **3B yes**, 8B borderline |
| **DoRA / rsLoRA** | Same order as LoRA | ~10–13 GB | ~21–25 GB | 3B yes |
| **Bottleneck adapters** | Inserted layers | ~9 GB | ~19 GB | Yes — the vivamodels pattern |
| **ReFT** | Interventions on representations | ~8 GB | ~18 GB | Yes; far fewer params than LoRA |
| **QLoRA** (4-bit base) | LoRA over a quantized base | ~4 GB | ~7 GB | **No — bitsandbytes has no MPS backend** |
| **Full SFT** | Everything | ~50 GB | ~130 GB | **No (measured: failed at 26.7 GiB)** |
| **DPO / ORPO / KTO / SimPO** | Preference alignment | 2× the SFT cost (two models) | — | Only as LoRA, ≤3B |
| **PPO / RLHF** | Policy + reward + ref + value | 4 models in memory | — | No |
| **Continued pretraining** | Everything, on a large corpus | — | — | No |
| **Knowledge distillation** | Student on teacher outputs | Depends on student | — | Yes if the student is small |

### Distillation, specifically

Three levels, cheapest first:

1. **Response distillation** — teacher generates text, student SFTs on it.
   Simplest; only needs a generation endpoint. Under ADR-009 the teacher must be
   a *local open-weights* model for the output to be class B and therefore
   trainable on. Claude output is class C: **eval only until GC rules**.
2. **Logit / KL distillation** — student matches the teacher's output
   distribution. Much more signal per example. This is what shipped in
   `vivamodels` for embeddings.
3. **Feature / attention distillation** — match intermediate representations.
   Most signal, most memory, needs architectural compatibility.

**Known-good local pattern:** `vivamodels/train/train_adapter.py` froze the base,
pre-embedded the corpus once, and learned two 384×384 maps initialized at
identity — **14 seconds per run**, and the result tied its teacher on a held-out
gate. The full fine-tune of the same student did not fit. Reuse that shape.

### Unlearning

Gradient ascent on a forget set, usually with a retain set to limit collateral
damage. Relevant if the goal is removing a capability rather than adding one;
harder to verify than it looks, and it degrades neighbors.

---

## 5. Hardware

### What the current machine does (measured)

| Task | Status |
|---|---|
| Activation capture, direction derivation on 3B | Works — the 3B direction hit held-out AUC 1.0 |
| Weight surgery on 3B | Works, minutes, produced a 5.8 GB model |
| Inference on 0.5B | Works, seconds per generation |
| **Generation on 3B via MPS** | **Pathologically slow — >7 min for 16 tokens.** See §7 |
| LoRA on ≤3B | Expected to fit (~9–12 GB); slow |
| Full fine-tune of anything | Fails (26.7 GiB on a small student) |

### Options

| Option | Cost | Unlocks | Loses |
|---|---|---|---|
| **Keep the M4 24 GB** | $0 | Surgery, merging, steering, small-model LoRA, evaluator distillation | Speed; QLoRA; anything >3B for training |
| **Mac Studio M3 Ultra, 96–512 GB** | ~$4–10K | Huge unified memory; LoRA on 70B in bf16; data never leaves the building | Still no CUDA, no bitsandbytes, no flash-attention; ~30–50× slower than a 4090 for training |
| **Single RTX 4090 (24 GB) or 5090 (32 GB) box** | ~$2.5–3.5K all-in | **CUDA, so QLoRA, flash-attention, bitsandbytes, vLLM, Unsloth, Axolotl all work.** QLoRA on 8B comfortably; LoRA on 8B in bf16 | Needs a Linux host; 24–32 GB caps bf16 training around 8B |
| **RTX A6000 / L40S (48 GB)** | ~$4–6K | LoRA on 30B bf16, QLoRA on 70B | Price |
| **Rented A100/H100 80 GB** | ~$1.50–3/hr | Everything up to 70B; a LoRA run in hours not days | **Class D (PHI) cannot go there.** Per-hour cost recurs; attestation under ADR-011 is harder |

### The recommendation

- **For what is on the table now** — evaluator distillation, hardening LoRA on
  ≤3B, more weight surgery — the current M4 is sufficient. Fix the MPS issue in
  §7 first.
- **For ADR-009 R3** (LoRA of a Qwen3 4B–8B class base), the cheapest real
  unlock is **a single consumer NVIDIA box**. It is a one-time ~$3K against
  $1.50–3/hr recurring, and it permanently removes the MPS limitations that
  have already cost debugging time in this project.
- **If PHI ever enters training** — class D cannot use rented cloud, so on-prem
  hardware stops being an optimization and becomes a requirement. That argues
  for buying rather than renting, and it compounds with the point above.

---

## 6. Methodology — the parts that are easy to get wrong

These apply to every method above. They are also where this project has already
lost time.

**Hold out a split, and select on it.** Fit the direction (or train the adapter)
on train; choose hyperparameters *and layers* on held-out. Selecting on the
fitting set picks whatever overfit hardest.

**Always run a capability control.** A safety delta means nothing without it.
Ablation that also destroys MMLU has lobotomized the model generally, not removed
refusal specifically. Measured here: β=0 cost **0.0 points** of factual accuracy,
β=2 cost **25 points**.

**Move the metric both ways.** One derived vector that both removes *and* adds
refusal is causal evidence. A single-direction result is consistent with
correlation.

**Check for degenerate output.** A model that has collapsed into repetition
produces no refusal phrases, so any phrase-matching detector scores it as 0%
refusal — indistinguishable from successful ablation. `steering._looks_degenerate`
exists for exactly this.

**Audit the corpus before trusting it.** AI Asylum's `forbidden_question`
category holds all 13 policy scenarios, five of which models answer freely.
Filtering to the eight refusal-triggering scenarios moved baseline refusal from
**33% to 92%** and was the difference between a derivable direction and noise.

**Change one variable.** Run baseline and modified through the *same* provider,
tokenizer and decoding settings. Comparing a surgically edited model served
through transformers against a stock model served through Ollama measures the
serving stack as much as the weights.

**Record provenance.** Every artifact should say what it came from. `manifest.py`
writes `asylum_surgery.json`; `vivamodels/register/models.yaml` and
`data/LEDGER.md` do the equivalent for trained models. Reuse both schemes rather
than inventing a third.

---

## 7. Known issues on this hardware

**Local-weights work needs the unified memory to itself.** The symptom that cost
the most time here looked like a PyTorch or MPS bug: 3B generation crawled — a
16-token completion exceeded seven minutes, and loading the model took 66 s
against 6 s for a 0.5B — while forward-pass-only work on the same model was
fine. The actual cause was memory contention. Ollama was holding `qwen2.5:7b`
resident at **5.5 GB of the same 24 GB unified memory**, and system swap was at
**44.8 GB of 46 GB**. Nothing errors in this state; it just gets orders of
magnitude slower.

Before any local-weights run:

```bash
ollama ps                      # anything resident is competing for the same RAM
ollama stop <model>            # frees it; Ollama reloads on next request
```

`aiasylum weights` now runs this as a preflight and warns about resident Ollama
models, swap pressure, and insufficient free memory. Treat those warnings as
blocking rather than advisory.

A second, smaller contributor: `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`, inherited
from labotomy, removes the MPS allocation cap. At 32 GB+ that prevents spurious
OOMs; at 24 GB it lets allocations spill into swap instead of failing fast.
`interp/core/loader.py` now sets it only at ≥32 GB.

**Steering magnitude is not transferable across layers.** Residual norms on
Qwen2.5-0.5B run from **0.6 at layer 0 to 69 at layer 23**. A fixed-norm vector
added at every layer is negligible late and catastrophic early. Addition is
therefore single-layer and scaled relative to the measured norm at that layer;
ablation, being scale-free, is safe at every layer.

**Tied embeddings.** Qwen2.5 at 0.5B/1.5B/3B ties `lm_head` to `embed_tokens`.
Editing the embedding table also edits the unembedding. `arch.py` deduplicates by
`data_ptr()` so the projection is applied once — applying it twice squares it,
which is not a no-op for any β except 0 and 1 — and records the tying in the
manifest.

**No bitsandbytes on MPS.** No 4-bit QLoRA, no 8-bit optimizers. LoRA here is
plain bf16.

**Set `PYTORCH_ENABLE_MPS_FALLBACK=1` for training.** Several backward ops still
have no MPS kernel.

---

## 8. What is built in this repo

| Capability | Where | State |
|---|---|---|
| Model loading, device/dtype, MPS handling | `interp/core/loader.py` | Working |
| Architecture detection, residual-write enumeration | `interp/core/arch.py` | Working |
| Activation capture | `weights/capture.py` | Working |
| Harmful/harmless corpus with seeded split | `weights/corpus.py` | Working |
| Refusal-direction derivation with AUC gate | `weights/direction.py` | Working |
| Inference-time ablation and steering | `weights/steering.py` | Working |
| Permanent weight surgery (β) | `weights/surgery.py` | Working |
| Multi-direction refusal subspace + capability-gated selection | `weights/direction.py` (`derive_subspace`), `weights/surgery.py` (`select_edit`) | Working |
| Behavioral model compare (refusal + false-refusal + capability control) | `weights/cli.py` (`compare`), `weights/evaluate.py` | Working |
| Provenance manifests | `weights/manifest.py` | Working |
| Local-weights provider for the test harness | `models/transformers_local.py` | Working |
| Interpretability engine (hooks, PCA/UMAP, patching, dashboards) | `interp/` | Working |
| Model-vs-model comparison (`model_diff`) | `interp/core/services/model_comparison_service.py` | Working |
| Interpretability API + UI | `api/routes/interp.py`, `frontend/pages/interp/` | Working |
| Weight-surgery API + UI | `api/routes/weights.py`, `frontend/pages/weights/` | Working |
| Subspace ablation (rank > 1) | `weights/direction.py`, `weights/surgery.py` | Working, CLI + UI |
| Capability-gated edit search | `weights/surgery.py::select_edit` | Working, CLI + UI |
| Refusal + capability measurement | `weights/evaluate.py` | Working, CLI + UI |
| Causal interp tools over HTTP | `enable_patching` / `enable_scrub` / `enable_minimal_circuit` | Working |
| LoRA | `setup.py` declares the extra | **Not implemented** |
| Distillation | — | **Not implemented** |

See [WEIGHT_SURGERY.md](WEIGHT_SURGERY.md) for how to drive what exists.

---

## 9. Seeing inside a model

Four modes, reachable from `/interp` in the UI or `POST /api/v1/interp/runs`:

| Mode | Input | Answers |
|---|---|---|
| `single` | 1 prompt, 1 model | What does this model compute, layer by layer? |
| `comparison` | 2 prompts, 1 model | Where do a harmful and a harmless prompt diverge? |
| `progression` | N prompts, 1 model | How do representations drift as examples are added? |
| **`model_diff`** | 1 prompt, 2 models | **What did a weight edit actually change?** |

`model_diff` is the one that pairs with weight surgery. Measured on
Qwen2.5-0.5B-Instruct, baseline against its own β=0 ablation:

```
 layer   mean |delta|   mean cos-sim
     0         0.018          0.9985     <- embedding, untouched
     3         3.854          0.9972     <- the edit starts to bite
    12         6.865          0.9968     <- steady plateau
    22         7.001          0.9870     <- largest angular change; the
    23         6.650          0.9936        direction was derived at 23
    24        38.245          0.9883     <- final layer norm amplifies
```

Cosine similarity stays near 0.99 throughout, so ablation changed the
*magnitude* of what the model computes far more than its direction, and the
effect concentrates at the output.

Two operational notes. Dashboards load plotly.js from the API's own `/static`
mount, so they render with no outbound network. And when a model with tied
embeddings is edited, the two sides decode through different unembeddings;
`shared_unembedding: false` appears in the run summary and the UI says to read
top-k overlap rather than comparing probabilities.
