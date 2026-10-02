# Ollama Setup Guide

Ollama allows you to run local LLM models without API keys.

## Installation

1. Install Ollama from https://ollama.ai
2. Start Ollama service:
   ```bash
   ollama serve
   ```

3. Pull models:
   ```bash
   ollama pull llama2
   ollama pull llama3
   ollama pull llama3.2
   ollama pull mistral
   ```

## Configuration

Set in `.env`:
```bash
OLLAMA_BASE_URL=http://localhost:11434
```

## Usage

Use `ollama` as the provider in CLI or API:

```bash
python -m vivasecuris.aiasylum.cli run \
  --doctor-provider ollama \
  --doctor-model llama3.2 \
  --patient-provider ollama \
  --patient-model llama2 \
  --test-type conversation
```

## Available Models

Check available models:
```bash
ollama list
```

Pull additional models:
```bash
ollama pull <model-name>
```

## Why responses differ from the Ollama app

Outputs from AI Asylum can differ from chatting in the Ollama app for two reasons:

1. **Different input** – We send system prompts and conversation context (e.g. doctor/patient instructions, turn history). The model sees more than the single message you type in the app, so its reply can change.

2. **Randomness** – We use `temperature=0.7` by default, so the model samples different replies each time. For more reproducible runs, use lower temperature or a fixed `seed` in your test config.

## Troubleshooting

- Ensure Ollama is running: `ollama serve`
- Check `OLLAMA_BASE_URL` matches your Ollama instance
- Verify model is pulled: `ollama list`

## Weight surgery bridge

Ollama tags cannot be edited in place. To modify weights and serve the result
from Ollama again, use the Transformers round-trip documented in
[OLLAMA_TRANSFORMERS_BRIDGE.md](OLLAMA_TRANSFORMERS_BRIDGE.md). Cloud tags
(`:cloud`) have no local blobs and cannot be converted.
