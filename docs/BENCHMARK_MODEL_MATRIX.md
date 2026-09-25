# Benchmark model selection — verified 24 September 2026

This is a bounded comparison of current public model families on the AWS
96 GiB GPU, with cached Qwen3 and our custom checkpoints retained as baselines.
It is not a claim to cover every model or to reproduce publisher leaderboards.
Model release dates below come from publisher announcements; Hub creation and
modification timestamps are not substituted for release dates.

| Candidate | Release | Weight parameters / BF16 storage | Access and runtime |
| --- | --- | --- | --- |
| [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | 2 March 2026 | 4.660B total including vision, about 9.32 GB | Public, Apache 2.0, native `qwen3_5`; modern runtime required |
| [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | 2 March 2026 | 9.653B total, about 19.31 GB | Public, Apache 2.0, native `qwen3_5`; modern runtime required |
| [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | 14 August 2026 | 27.781B total, about 55.56 GB | Public, Apache 2.0, native `qwen3_5`; modern runtime required |
| [google/gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it) | 3 June 2026 | 11.960B total, about 23.92 GB | Public, Apache 2.0, native `gemma4_unified`; modern runtime required |
| [mistralai/Ministral-3-8B-Instruct-2512-BF16](https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512-BF16) | Family: 2 December 2025 | 8.918B total, about 17.84 GB | Public, Apache 2.0; `mistral3` with `ministral3` text config, modern runtime and mistral-common required |
| [HuggingFaceTB/SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) | 8 July 2025 | 3.075B, about 6.15 GB | Public, Apache 2.0, native `smollm3`, supported by existing Transformers 4.57.6 |

Storage estimates are parameter bytes, not total peak memory. Inference adds
KV cache, activations and temporary workspaces. Run models serially with a
bounded context and output budget. None of these six needs a Hugging Face login.

Release evidence: [Qwen release chronology](https://github.com/QwenLM/Qwen3.8/blob/main/README.md),
[Google Gemma 4 12B announcement](https://blog.google/innovation-and-ai/technology/developers-tools/introducing-gemma-4-12b/),
[Mistral 3 announcement](https://mistral.ai/news/mistral-3/),
[SmolLM3 announcement](https://huggingface.co/blog/smollm3).

## Pinned revisions

Anonymous Hub API metadata and public config files were read on the verification
date. The campaign should record these revisions along with its dataset hash.

```text
Qwen/Qwen3.5-4B 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
Qwen/Qwen3.5-9B c202236235762e1c871ad0ccb60c8ee5ba337b9a
Qwen/Qwen3.8-27B 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
google/gemma-4-12B-it 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7
mistralai/Ministral-3-8B-Instruct-2512-BF16 f6fae9795746f63c9be8344932f01275f3c63734
HuggingFaceTB/SmolLM3-3B a07cc9a04f16550a088caea529712d1d335b0ac1
```

## Explicit exclusions and compatibility

- [Microsoft Phi-4-reasoning-vision-15B](https://huggingface.co/microsoft/Phi-4-reasoning-vision-15B)
  was released 4 March 2026, is public/MIT, and has 15.120B BF16 parameters
  (~30.24 GB). It requires repository custom modeling and processing code.
  It fits the GPU but is excluded from the native-model comparison pending a
  pinned-code review and generation adapter. Phi-4-mini-flash-reasoning also
  requires custom code and Mamba dependencies. Compatible older baselines are
  `microsoft/Phi-4-mini-reasoning` (3.836B) and `microsoft/phi-4` (14.660B).
- [Meta Llama 4 Scout](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct)
  is gated and has 108.642B **total** BF16 parameters (~217.28 GB); its 17B
  active count does not make its complete weights fit this GPU. Skip native
  BF16 testing here. Older Llama 3.2 models can be added after account access.
- [Mistral Small 4](https://mistral.ai/news/mistral-small-4/) has 119B total
  parameters; native BF16 weights exceed 96 GiB. Quantized tests would need a
  separate clearly labeled campaign. The same applies to larger Qwen3.8 MoEs.

Read-only inspection of the actual AWS runtime confirmed Transformers 4.57.6
does not register `qwen3_5`, `gemma4`, `gemma4_unified`, `ministral3`, or
`phi4flash`. A config's recorded `transformers_version` is not proof that the
architecture is available in that release. The benchmark worker uses a separate
Python environment with [Transformers 5.17.0](https://pypi.org/project/transformers/5.17.0/)
and reuses the existing CUDA Torch installation through a `.pth` path. The API,
interpretability and weight-surgery runtime remain on their validated versions.

Configure `AIASYLUM_BENCHMARK_PYTHON=/home/ubuntu/aiasylum/venv-benchmark/bin/python`
on the server. Recreate the environment with
`bash scripts/setup_benchmark_runtime.sh` (Transformers 5.17.0, mistral-common
1.12.0, datasets 5.0.1). The API invokes `scripts/run_benchmark_job.py <existing-run-id>`,
stores runtime versions in the run metadata and writes private logs to
`runs/benchmark-jobs/<run-id>.log`. Cancellation terminates the worker process
group before releasing its OS-held GPU lock. New architecture support here is
for text generation benchmarks; it does not establish interpretability support.

The initial runtime check loaded all six candidates' configurations and
tokenizers. Qwen3.5-4B then produced the expected answer to a short arithmetic
prompt on CUDA in BF16, using its pinned revision. This is a loader smoke test,
not a benchmark score. Qwen's optional linear-attention and causal-convolution
extensions were absent; Transformers used correct but slower PyTorch kernels.

Dataset loading was verified on AWS with datasets 5.0.1. Each split loaded
real rows and repeated seed-0 selections had identical ordered sample hashes:

| Dataset / configuration / split | Rows | Verified revision |
| --- | ---: | --- |
| `cais/mmlu` / `all` / `test` | 14,042 | `c30699e8356da336a370243923dbaf21066bb9fe` |
| `openai/gsm8k` / `main` / `test` | 1,319 | `740312add88f781978c0658806c59bc2815b9866` |
| `Rowan/hellaswag` / default / `validation` | 10,042 | `218ec52e09a7e7462a5400043bb9a69a41d06b76` |
| `allenai/ai2_arc` / `ARC-Challenge` / `test` | 1,172 | `210d026faf9955653af8916fad021475a3f00453` |

Use the same sampled dataset revision, ordered-question hash, sample count,
prompt protocol, seed, greedy decoding and output-token limit for every model.
Report invalid/truncated responses and failed runs visibly. Small sample runs
are workflow validation and directional evidence, not statistically definitive
rankings or official benchmark scores.
