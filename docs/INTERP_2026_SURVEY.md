# Mechanistic interpretability, 2026: what changed and what aiasylum does about it

Survey date: 2026-09-24. Scope: work published roughly January to September 2026, plus the late-2025 anchors it builds on, filtered for what matters to a platform that (a) derives and removes refusal directions, (b) tests models against a jailbreak corpus, and (c) wants causal rather than correlational claims about model internals.

Each section lists the papers, what the technique is in one or two sentences, and what it means for the `interp/` and `weights/` modules. The implementation roadmap that follows from this survey is in the plan file referenced at the end and is mirrored by the registries in `api/routes/interp.py` and `api/routes/weights.py`, which mark each technique as built or not built with a reason.

---

## 1. Refusal geometry

The module's core assumption, inherited from Arditi et al. 2024, is that refusal is mediated by a single direction. The 2026 literature keeps the direction but shows it is the first axis of a low-dimensional subspace, that the subspace's dimensionality is a property of the training data, and that reasoning models move the decision into the chain of thought.

| Paper | Venue | Technique | Code |
|---|---|---|---|
| Fast Multi-dimensional Refusal Subspaces via RFM-AGOP ([2607.02396](https://arxiv.org/abs/2607.02396)) | ICML 2026 MI workshop | Recursive Feature Machine on last-token residuals: fit a Mahalanobis-Laplace kernel machine on harmful vs harmless activations, take the Average Gradient Outer Product `M = (1/n) Σ ∇f ∇fᵀ`, EMA-smooth, iterate; warm-start `M₀ = β wwᵀ + (1−β) Σ_{X,10}` from the difference-in-means probe `w`. Top-k eigenvectors form the refusal cone. Ablation is eigenvalue-weighted: `W ← W − Σᵢ (μᵢ/μ₁) v̂ᵢ v̂ᵢᵀ W`. Runs in seconds on a laptop (one forward pass plus small eigendecompositions). | not released |
| Refusal geometry reflects refusal training ([2608.25390](https://arxiv.org/abs/2608.25390)) | preprint, Aug 2026 | Stable rank `sr(M) = ‖M‖²_F / ‖M‖²₂` of the benign-centred refusal residuals `ΔH` at refusal-mediating layers predicts how well a single-vector ablation works. Repetitive refusal prefixes in training concentrate refusal into few dimensions; diverse prefixes raise the stable rank and weaken the attack. Observed sr 11 to 33 on OLMo-2-1B. | not released |
| There Is More to Refusal in LLMs than a Single Direction ([2602.02132](https://arxiv.org/abs/2602.02132)) | EMNLP 2026 | SAE decomposition of refusal shows a shared core plus style-specific and domain-specific latents. Linear interventions along any refusal direction produce near-identical behavioural trade-offs, flattening that structure. | see paper |
| Over-Refusal and Representation Subspaces ([2603.27518](https://arxiv.org/abs/2603.27518)) | EMNLP 2026 | Harmful refusal is one global vector; over-refusal of benign prompts is task-dependent and lives inside benign task clusters. Global ablation fixes over-refusal only incidentally while disrupting the whole refusal mechanism. | see paper |
| The Geometry of Refusal: Concept Cones ([2502.17420v2](https://arxiv.org/abs/2502.17420), Feb 2026 revision) | ICML 2025, revised | Gradient-based Refusal Cone Optimization finds polyhedral cones of refusal directions; introduces representational independence (orthogonality is not enough). Hours of compute; RFM-AGOP is the cheap replacement. | yes |
| Where Do Reasoning Models Refuse? ([2507.03167v3](https://arxiv.org/abs/2507.03167)) | preprint, 2026 revision | Sentence-level resampling through the chain of thought on DeepSeek-R1-Distill, Qwen3-8B, GPT-OSS-20B. Distilled models decide refusal in the first sentences; RL-trained models decide gradually. Directional ablation works less reliably and costs capability. | [github](https://github.com/kureha-yamaguchi/reasoning-manipulation) |
| Chain-of-Thought Disrupts Simple Steering of Refusal ([2605.26772](https://arxiv.org/abs/2605.26772)) | preprint | Refusal in reasoning models is encoded jointly in activations and in the CoT text. Steering while the CoT regenerates reverses refusal 94 percent of the time; steering with a fixed CoT only 39 percent. | not released |

Abliteration defences, which the pipeline should be evaluated against:

| Paper | Technique |
|---|---|
| An Embarrassingly Simple Defense Against LLM Abliteration Attacks ([2505.19056](https://arxiv.org/abs/2505.19056)) | Extended-refusal fine-tuning spreads the refusal signal across token positions; refusal drops at most 10 points under abliteration versus 70 to 80 for baselines. |
| Abliteration Mitigation via Refusal Aliases ([2608.18093](https://arxiv.org/abs/2608.18093)) | Rank-k edits to residual-writing matrices replace the refusal-vector output with a low-variance random alias; downstream Q/K/V/unembed readers are SVD-patched to preserve behaviour. Only tested against directional ablation. |
| Decoy Direction Optimization ([2609.16204](https://arxiv.org/abs/2609.16204), Sept 2026) | Post-hoc weight edit injects a high-magnitude nonlinear decoy into MLP neurons so difference-in-means finds the decoy. 30 to 450 times cheaper than trained defences; 65 percent worst-case attack success under an adaptive attacker. |
| Activation Steering Induces Emergent Misalignment ([2606.08682](https://arxiv.org/abs/2606.08682)) | Steering vectors, including on Qwen3.5, cause broad misalignment on unrelated prompts, and the steered harmful outputs are more coherent than fine-tuned ones. A capability-only control misses this. |

What it means here. `derive_subspace` builds its extra directions from the SVD of per-layer means, which is not the same object as the RFM cone and has no per-direction weight. The stable-rank number costs nothing beyond the captures the direction stage already makes. The `refusal_narrow` objective conflates over-refusal with refusal. Sweeps on Qwen3 should run with thinking enabled so steering acts during CoT regeneration. The `compare` stage needs a broad-misalignment control next to the factual one.

## 2. Jailbreak mechanics and monitoring

| Paper | Venue | Technique | Code |
|---|---|---|---|
| Do SAE Features Actually Help Detect Jailbreaks? SAEGuardBench / InterpGuard ([zenodo](https://zenodo.org/records/19535387)) | Apr 2026 | 8 detectors, 4 paradigms, 6 datasets, Gemma-2-2B to Llama-3.3-70B. Raw-activation linear probes beat SAE-feature probes on every model (Gemma-2-2B 0.949 vs 0.712 AUROC; Llama-3.1-8B 0.867 vs 0.477). The SAE reconstruction objective discards low-variance directions that carry the safety signal. InterpGuard: detect with a raw probe, explain with SAE features. | [github](https://github.com/ronyrahmaan/saeguardbench) |
| Investigating task-specific prompts and SAEs for activation monitoring ([2504.20271](https://arxiv.org/abs/2504.20271)) | 2025 | Prompted last-token probes (append an eliciting question, read the last token) are the most data-efficient monitors and generalise under shift. | yes |
| Before the Last Token: Diagnosing Final-Token Safety Probe Failures ([2605.12726](https://arxiv.org/abs/2605.12726)) | 2026 | Final-token-only probes fail in diagnosable ways; pool over several positions. | see paper |
| Robust Harmful Features Under Jailbreak Attacks ([2606.28153](https://arxiv.org/abs/2606.28153)) | ICML 2026 oral | Two head populations: early-layer Adversarially Compromised Heads that attack-template tokens suppress, and mid-layer Safety-Aligned Heads that keep firing when the attack succeeds. Suppressing a few ACHs alone induces jailbreak-like behaviour. Reading persistent activations without training gives a detector competitive with LlamaGuard-3/4, Qwen3Guard and WildGuard across Gemma-2 2B/9B/27B, Llama-2-7B, Llama-3-8B, Qwen 7B/14B. | [github](https://github.com/MorningYin/Robust-Harmful-Features) |
| Circuit Discovery Helps Detect LLM Jailbreaking ([2608.27504](https://arxiv.org/abs/2608.27504)) | 2026 | Edge Attribution Patching (zero-ablation corruption, score `(e_corr − e_clean)ᵀ ∂L/∂e_clean`, two forward and one backward pass over 1.6M edges) and learned-mask subnetwork probing on LLaMA-2-7B-chat with GCG suffixes. Ablating the top 0.05 to 5 percent of edges at the first token raises refusal from about 28 percent to as much as 80. Benign capability was not measured. | not released |
| From Concept-Aligned Tokens to Vulnerable Features ([2604.23130](https://arxiv.org/abs/2604.23130)) | 2026 | Single harmful tokens localise sparse SAE feature subgroups on Gemma-2-2B; amplifying them raises harmfulness. | see paper |
| ADAG: Automatically Describing Attribution Graphs ([2604.07615](https://arxiv.org/abs/2604.07615)) | 2026 | Attribution profiles plus an LLM explainer-simulator loop; found steerable clusters behind a jailbreak in Llama-3.1-8B-Instruct. | see paper |
| RelP: Relevance Patching ([2508.21258](https://arxiv.org/abs/2508.21258)); When Attribution Patching Lies ([2606.09899](https://arxiv.org/abs/2606.09899)) | 2025, 2026 | Plain attribution patching correlates 0.006 with true patching on GPT-2-Large MLP outputs; RelP's LRP coefficients reach 0.956. HVP is a second-order correction that stays tractable at 8B. | yes / see paper |

What it means here. A harmful-intent probe on raw residuals is the right detector for test runs, with SAE features reserved for explanation. The head-level analysis the module already does is correlational and can be upgraded to the ACH/SAH protocol with an ablation check. Any circuit claim needs a real forward-pass patching primitive first; the current one re-adds cached pieces without re-running the model.

## 3. Dictionaries: SAEs, transcoders, attribution graphs

| Item | What is new |
|---|---|
| Qwen-Scope ([qwen.ai](https://qwen.ai/blog?id=qwen-scope)) | Official SAEs across Qwen3 and Qwen3.5 layers. |
| Qwen3-Instruct SAEs, "Discovering Millions of Interpretable Features" ([2606.26620](https://arxiv.org/abs/2606.26620)) | Layer-wise SAEs on residual, MLP and attention outputs for Qwen3-1.7B and 4B, a residual subset for 8B, trained on FineWeb-Edu; includes a refusal-steering case study. |
| circuit-tracer ([github](https://github.com/safety-research/circuit-tracer)) | Transcoders and cross-layer transcoders for Qwen3 0.6B to 14B, Gemma-2/3, Llama-3.x, GPT-OSS-20B, with feature interventions and Neuronpedia hosting. |
| CLT-Forge ([2603.21014](https://arxiv.org/abs/2603.21014), [github](https://github.com/LLM-Interp/CLT-Forge)) | Training cross-layer transcoders end to end, with circuit-tracer integration. |
| Turn-Averaged SAEs (Anthropic Circuits Update, [June 2026](https://transformer-circuits.pub/2026/june-update/index.html)) | Average the residual over all tokens of a turn, train an SAE on that vector. Features come out at the level of "wrong answer in a number puzzle", "evaluation awareness", "distress", instead of token-level concepts. |
| Downstream Connections Predict Which Features Will Steer (Anthropic, [May 2026](https://transformer-circuits.pub/2026/may-update/index.html)) | Two features with the same top activations and logit-lens readout can steer differently; the downstream edges in the attribution graph predict which one will. |
| Sanity Checks for SAEs: Do SAEs Beat Random Baselines? ([2602.14111](https://arxiv.org/abs/2602.14111)); Are SAE Benchmarks Reliable? ([2605.18229](https://arxiv.org/abs/2605.18229)); Where You Measure Decides What You Measure ([2608.13337](https://arxiv.org/abs/2608.13337)) | Random dictionaries match trained SAEs on auto-interpretability, sparse probing and causal editing metrics. Position choice changes ablation-based SAE evaluations. |
| Feature Superposition in Neural Networks: From Theory to Practice ([2609.06862](https://arxiv.org/abs/2609.06862), Sept 2026) | Survey; states plainly that reconstruction alone does not establish feature identity or causal use. |
| LLMs Can Annotate Attribution Graphs ([2608.02632](https://arxiv.org/abs/2608.02632)) | Automatic supernode grouping from feature descriptions; recovers intermediate steps in 97 of 100 capitals cases. |

What it means here. SAE work is now practical on Qwen3 without training anything. There are no pretrained dictionaries for Qwen2.5. Every SAE-based claim must ship with a random-dictionary baseline or it is not distinguishable from noise under current metrics.

## 4. New lenses and introspection

| Paper | Technique | Code |
|---|---|---|
| Verbalizable Representations Form a Global Workspace in Language Models ([2607.15495](https://arxiv.org/abs/2607.15495), Anthropic, July 2026) | The Jacobian lens: `lens_l(h) = unembed(J_l h)` with `J_l = E[∂h_final/∂h_l]` averaged over about 1000 sequences of 128 tokens (100 prompts already usable). Unlike the logit lens it is a linearised causal readout of what an activation disposes the model to say. Middle layers (about 40 to 90 percent depth) form a workspace: swapping lens coordinates changes what the model reports thinking; unspoken intermediates ("spider" while counting legs) and suppressed intentions ("secretly", "trick") are readable. | [anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens), Apache-2.0, HF transformers only; third-party fits for Qwen3.5-4B and Qwen3.5-397B |
| Activation Oracles ([2512.15674](https://arxiv.org/abs/2512.15674)); Building Better Activation Oracles + AObench ([2606.02609](https://arxiv.org/abs/2606.02609)); Confidence and Calibration ([2605.26045](https://arxiv.org/abs/2605.26045)) | Fine-tuned LLMs that take another model's activations as input and answer questions about them. Hallucination and text-inversion confounds remain; AObench is the first evaluation suite. | AObench open |
| Measuring Activation Control in LLMs ([2608.21664](https://arxiv.org/abs/2608.21664)) | Most models can shift the direction and magnitude of their own residual stream on instruction, sometimes enough to degrade linear probes, activation oracles and the Jacobian lens. Recommends tracking activation controllability. | see paper |
| Legibility is Not Interpretability ([2609.04194](https://arxiv.org/abs/2609.04194), Sept 2026) | Step importance in a CoT measured as Monte Carlo advantage is only partially recoverable from the text, even by fine-tuned critics. | see paper |
| AuditBench ([2602.22755](https://arxiv.org/abs/2602.22755)); The Model Organism Lottery ([2607.01033](https://arxiv.org/abs/2607.01033)) | 56 models with 14 planted behaviours; black-box tools beat white-box ones inside an investigator agent. Detectability of a planted behaviour depends heavily on how it was trained in; post-hoc SFT organisms overstate what interpretability can find. | models and agent released |

What it means here. The existing logit lens skips the final norm and is not causal; the Jacobian lens is a drop-in upgrade with a reference implementation that runs on Qwen. An intent readout over agentic transcripts is a concrete new test. Monitors need a robustness test against models that can move their own activations.

## 5. Weight-space interpretability (documented, not scheduled)

Stochastic Parameter Decomposition ([2506.20790](https://arxiv.org/abs/2506.20790)), Targeted Parameter Decomposition ([2607.13047](https://arxiv.org/abs/2607.13047), 7 percent of the FLOPs of a full decomposition for a targeted submodel), Sparse Weight Decomposition for circuit extraction ([2608.03913](https://arxiv.org/abs/2608.03913), GPT-2 / Qwen2.5 / Qwen3.5-27B, under 1 percent of the data transcoders need), weight-sparse transformers ([2511.13653](https://arxiv.org/abs/2511.13653)) and interpretable individual parameters ([2607.02964](https://arxiv.org/abs/2607.02964)), interference weights (Anthropic, [2026](https://transformer-circuits.pub/2026/interference_effectiveness_helpfulness/index.html)), WASD critical neurons ([2603.18474](https://arxiv.org/abs/2603.18474)). These decompose weights rather than activations. They are the most promising path to circuits that do not depend on a prompt, but every method is still demonstrated on models well under 1B parameters or on toy tasks, so nothing here is scheduled.

---

## Gap map

| Capability | Module today | Target |
|---|---|---|
| Refusal subspace | SVD of pooled per-layer means (`weights/direction.py: derive_subspace`) | RFM-AGOP cone with eigenvalue weights and a refusal-vs-k curve |
| Abliteration resistance | none | stable rank per layer; re-derive after any hardening |
| Over-refusal | shares the refusal contrast | task-conditioned objective with overlap report |
| Reasoning models | `strip_thinking` before phrase matching | thinking-aware sweeps; refusal-decision timeline |
| Probes and monitoring | none | multi-layer raw probes, prompted variant, "knows but complies" flag on test runs |
| Head-level safety | correlational circuit cards | ACH/SAH classification, training-free detector, ACH ablation |
| Causal attribution | patching without a forward pass | forward-pass patching; node-level EAP with verification |
| Lens | `h @ W_Uᵀ` without the final norm | final-norm logit lens; Jacobian lens |
| Dictionaries | none | Qwen3 SAE loader, feature steering, turn-averaged SAE, circuit-tracer graphs |
| Null baselines | none | required on every payload labelled causal |

## Caveats encoded in the UI

- A payload may be labelled `claim: causal` only if it carries a null-baseline block (random directions, shuffled labels, or a random dictionary).
- SAE features explain; they do not detect. Detection numbers come from raw-activation probes.
- Interpretability results on a model that was fine-tuned post hoc to plant a behaviour overstate what the same tools find on naturally trained models.
- A capability control alone does not clear a steered or edited model; a broad-misalignment control runs alongside it.
- Reasoning-model refusal numbers are reported with thinking enabled, and the refusal decision is timed within the CoT rather than assumed at the first answer token.

## Roadmap

Phase 0 fixes the engine's credibility gaps (final-norm lens, real patching, relabelled head roles, null-baseline helper, pre-MLP capture). Phase 1 upgrades refusal geometry in `weights/`. Phase 2 adds probes, safety heads and gradient attribution. Phase 3 adds the Jacobian lens. Phase 4 adds the Qwen3 dictionary stack. Phase 5 builds composite tests: an abliteration-hardness scorecard, jailbreak mechanism fingerprinting, knows-but-complies severity, refusal-decision timelines with CoT resampling, a monitor-robustness test, an attack-after-defence loop, and cross-lingual subspace overlap. Registries in `api/routes/weights.py` and `api/routes/interp.py` are the source of truth for what has shipped.
