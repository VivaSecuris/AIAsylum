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

## Troubleshooting

- Ensure Ollama is running: `ollama serve`
- Check `OLLAMA_BASE_URL` matches your Ollama instance
- Verify model is pulled: `ollama list`
