# Ollama ↔ Transformers bridge

CLI tools that move a **local** Ollama model into a Transformers checkpoint for
weight surgery, then pack an edited checkpoint back into a new Ollama tag.

```text
ollama tag  --ollama_to_hf-->  models/<name>/  --surgery-->  models/<edited>/
                                                           |
                                                           v
                                              hf_to_ollama --> new ollama tag
```

Surgery still runs only on Transformers safetensors under `models/`. The bridge
does not teach the interp/weights stack to accept Ollama names.

## Proof fixture (verified locally)

| Role | ID |
| --- | --- |
| Ollama source | `qwen2.5:0.5b` (Q4_K_M blob) |
| HF twin | `Qwen/Qwen2.5-0.5B-Instruct` |
| Edited Ollama tag | `aiasylum-qwen05-edited` |

Completed on 1 October 2026:

1. `scripts/ollama_to_hf.py --model qwen2.5:0.5b --out models/bridge-qwen05 --hf-id Qwen/Qwen2.5-0.5B-Instruct`
2. `scripts/ollama_to_hf.py --model qwen2.5:0.5b --out models/bridge-qwen05-from-gguf` (dequant path)
3. Smoke ablation → `models/bridge-qwen05-edited` (random unit direction, β=0; not a scientific claim)
4. `scripts/hf_to_ollama.py --checkpoint models/bridge-qwen05-edited --tag aiasylum-qwen05-edited --quant f16`
5. `ollama run qwen2.5:0.5b` and `ollama run aiasylum-qwen05-edited` both answered a short prompt

### Parity (no surgery) — verified 1 October 2026

Exported the **unedited** `models/bridge-qwen05` to `aiasylum-qwen05-roundtrip` (F16) and compared to stock `qwen2.5:0.5b` (Q4_K_M):

| Check | Result |
| --- | --- |
| Transformers `weights compare` hub `Qwen/Qwen2.5-0.5B-Instruct` vs `models/bridge-qwen05` | **unchanged** — refuse-harmful 87.5%=87.5%, factual 83.3%=83.3%, drift 0% |
| Ollama closed prompts (temp=0, seed=0, n=10) | **9/10 exact**; accept rate stock 8/10, roundtrip 9/10 |
| Ollama mixed/open prompts (n=10) | **7/10 exact** — open-ended completions diverge (expected Q4 vs F16) |

Artifacts: `runs/ollama-bridge/PARITY_SUMMARY.json`, `PARITY_closed.json`, `PARITY_ollama_chat.json`, `PARITY_weights_compare.json`.

Stock Ollama is quantized; the roundtrip is F16 from the HF twin. Near-par on short factual prompts is the right bar — not bit-identical open generation.

Unit tests: `venv/bin/python -m pytest tests/test_ollama_bridge.py -q`

## Commands

### Ollama → Hugging Face

```bash
# Preferred for surgery: download the known publisher checkpoint; still records
# the Ollama blob digest in aiasylum-bridge.json
venv/bin/python scripts/ollama_to_hf.py \
  --model qwen2.5:0.5b \
  --out models/bridge-qwen05 \
  --hf-id Qwen/Qwen2.5-0.5B-Instruct

# Lossy: dequantize the local Ollama GGUF into safetensors
venv/bin/python scripts/ollama_to_hf.py \
  --model qwen2.5:0.5b \
  --out models/bridge-qwen05-from-gguf
```

Then edit with the existing weights CLI / WebUI Neurosurgery against the
`models/...` path (provider `transformers` / `local`).

### Hugging Face → Ollama

```bash
# Needs llama.cpp convert_hf_to_gguf.py (see Dependencies)
venv/bin/python scripts/hf_to_ollama.py \
  --checkpoint models/bridge-qwen05-edited \
  --tag aiasylum-qwen05-edited \
  --quant f16 \
  --llama-cpp-dir runs/ollama-bridge/llama.cpp

# Optional: quantize after F16 conversion
venv/bin/python scripts/hf_to_ollama.py \
  --checkpoint models/bridge-qwen05-edited \
  --tag aiasylum-qwen05-edited-q8 \
  --quant q8_0

# Optional: skip GGUF and use Ollama 0.32+ experimental safetensors import
venv/bin/python scripts/hf_to_ollama.py \
  --checkpoint models/bridge-qwen05-edited \
  --tag aiasylum-qwen05-edited-exp \
  --experimental-ollama
```

## Provenance

Each import writes `aiasylum-bridge.json` next to the checkpoint (and export
work dirs write one too). Important fields:

| Field | Meaning |
| --- | --- |
| `import_mode` | `hf_id` (preferred) or `dequantized` (lossy GGUF→HF) |
| `source_ollama` | Original Ollama tag |
| `blob_digest` | Content-addressed weight blob |
| `gguf_quant` / `gguf_file_type` | Quantization of the Ollama blob |
| `notes` | Human-readable fidelity warning |

Sidecars `ollama-template.txt` and `ollama-system.txt` are copied when present so
`hf_to_ollama` can rebuild a Modelfile close to the source tag.

## Fidelity rules

1. **Quantize after surgery, never before.** Prefer `--hf-id` for the surgery
   base. Dequantized GGUF imports are working copies, not bit-identical to the
   publisher release.
2. **Cloud tags cannot bridge.** Names ending in `:cloud` (for example
   `glm-5.3:cloud`, `deepseek-v4-pro:cloud`) have no local blobs. The scripts
   reject them. Frontier chat proof via Ollama cloud is a separate later phase;
   local weight surgery on GLM-5.3 / DeepSeek-V4 needs multi-GPU hardware far
   beyond the current single 96 GB box.
3. **Ollama names stay rejected by weight surgery.** Use the bridged `models/`
   path. Do not pass `qwen2.5:0.5b` to `aiasylum weights ablate`.

## Dependencies

| Tool | Role |
| --- | --- |
| Ollama CLI | `pull`, blob store, `create` |
| Project venv | `transformers`, `huggingface_hub`, `gguf`, `sentencepiece`, `torch` |
| llama.cpp | `convert_hf_to_gguf.py` (+ `conversion/`); optional `llama-quantize` on PATH |

Pinned llama.cpp revision used for the proof convert tree:

```text
0c1e57098bba43ac29e6e3b677cdceebdd22334f
```

Bootstrap a convert tree (sparse checkout is enough):

```bash
mkdir -p runs/ollama-bridge
git clone --depth 1 https://github.com/ggml-org/llama.cpp runs/ollama-bridge/llama.cpp
# Or set LLAMA_CPP_DIR / --llama-cpp-dir to an existing checkout at that revision.
venv/bin/pip install gguf sentencepiece
brew install llama.cpp   # provides llama-quantize when using --quant other than f16/bf16/q8_0 convert outtypes
```

`runs/` is gitignored; the convert clone stays local.

## Layout

| Path | Role |
| --- | --- |
| [`vivasecuris/aiasylum/bridge/`](../vivasecuris/aiasylum/bridge/) | Testable helpers |
| [`scripts/ollama_to_hf.py`](../scripts/ollama_to_hf.py) | Import CLI |
| [`scripts/hf_to_ollama.py`](../scripts/hf_to_ollama.py) | Export CLI |
| [`tests/test_ollama_bridge.py`](../tests/test_ollama_bridge.py) | Unit tests (no full convert in CI) |

## Out of scope (this bridge)

- WebUI convert buttons
- Treating dequantized GGUF as scientifically equivalent to publisher BF16
- Teaching surgery/interp to accept bare Ollama IDs
- Local round-trip of GLM-5.3 / DeepSeek-V4 on the current single-GPU host
