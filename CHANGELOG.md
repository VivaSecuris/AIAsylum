# Changelog

All notable changes to AI Asylum will be documented in this file.

## [Unreleased]

### Changed
- Create Test is organised by step: test design, patient, doctor, evaluator, run. Each LLM step has its own model, system prompt (built-in, library, custom, or verbatim for the patient) and advanced settings (temperature, top-p, max tokens, ReACT).
- One-Shot and Multi-Shot runs use a separate doctor for the end-of-test assessment. Previously the model under test assessed itself.
- **Scoring:** custom evaluator prompts, including library prompt "Default Evaluator", are added after the built-in scoring prompt instead of replacing it. The eight score dimensions and the JSON format are always kept. Runs scored with a custom evaluator prompt before this change may have used invented dimension names.
- Suites take a doctor. Group therapy suites run one session with every model as a patient.
- Ollama model pickers list only models that can chat (embedding models are excluded).

### Fixed
- Interview inputs quote and label the doctor's message; patient ReACT instructions keep all reasoning fields about the patient's own answer. Selected system prompts remain unchanged.
- Create Test explains application framing and ReACT effects, records the actual interview wrapper in resolved configuration, and exposes adaptive-strategy state. Finished runs leave the disappearing Live tab for an available results panel.
- Added a reproducible system/user prompt diagnostic with response capture, causal-prefix checks, and explicitly labeled activation and role-logit measurements.
- Conversation patients keep their interview speaker framing when using custom system prompts or ReACT; group histories now attribute each speaker once from the receiving patient's perspective.
- Conversation turns retain the responding model and the system prompts observed on generation requests, shown in both Transcript and Live views. System-prompt pickers reject stale library selections belonging to another role.
- H-neuron selection now isolates report labels, refits shuffled-label controls, handles tied AUROC scores, and checks both null and answer-length baselines before claiming a usable detector.
- Conditional-steering controls are exported only after passing reporting and requested sampled checks; result views and CLI output distinguish rejected, accepted, and unverified controls.
- Embedding analysis uses consistent CPU numerical operations for accelerator-loaded models, and full-vocabulary reconstruction works in the API and CLI.
- Autotune releases live snapshot references and backup tensors before reloading a saved checkpoint for verification.
- Saved test and weight-run replay preserves recorded defaults, new-stage options, and available artifact dependencies.
- Anthropic sampling options respect model/thinking restrictions; group-therapy seeds are handled per patient; cached-model deletion respects active and queued model jobs.
- Writes to a row's `meta_data` dict now persist: run renames, pause and cancel flags, and the analysis run's assessment ID were previously lost.
- The chain-of-thought path no longer sends the system prompt twice.
- Autotune (weight surgery) no longer fails on Apple GPUs (MPS) with a CPU/GPU device mismatch.
- Test-prompt search reaches the whole library, not only the newest 100 prompts.

## [0.1.0] - 2024-01-01

### Added
- Initial release
- Multi-provider support (OpenAI, Anthropic, Google, Ollama)
- Test framework (conversations, scenarios, adversarial)
- Doctor/Patient model system
- Test execution runner
- Analysis service
- FastAPI backend
- CLI interface
- Database models and migrations
- Authentication system
- Basic frontend structure

### Features
- Multi-turn conversation testing
- Scenario-based testing
- Adversarial/jailbreak testing
- Safety scoring
- Assessment generation
- Test result storage

### Known Limitations
- Benchmark integration is placeholder
- Frontend UI is basic structure only
- Deep analysis features (activation patching) not fully implemented
- RAG support is placeholder
