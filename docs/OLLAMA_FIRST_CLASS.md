# Ollama First-Class Support

Ollama models are now first-class citizens in AI Asylum with enhanced features and seamless integration.

## Features

### Enhanced OllamaModel

- **Model Management**: Check availability, pull models, get model info
- **Better Streaming**: Improved streaming support with proper async handling
- **Message Support**: Full support for conversation messages
- **Context Management**: Async context manager support
- **Usage Tracking**: Detailed token usage and timing information

### OllamaProvider

- **Model Listing**: List all available Ollama models
- **Model Pulling**: Pull models programmatically
- **Enhanced Integration**: Better integration with the test framework

## Usage

### Basic Usage

```python
from vivasecuris.aiasylum.models.ollama import OllamaModel

# Create model
model = OllamaModel("llama2")

# Generate response
response = await model.generate("Hello, how are you?")
print(response.content)

# Close when done
await model.close()
```

### With Context Manager

```python
async with OllamaModel("llama2") as model:
    response = await model.generate("Hello!")
    print(response.content)
    # Automatically closed
```

### Check Model Availability

```python
model = OllamaModel("llama2")
if await model.check_available():
    print("Model is available")
else:
    print("Model not found, pulling...")
    await model.pull_model()
```

### List Available Models

```python
from vivasecuris.aiasylum.models.ollama import OllamaProvider

provider = OllamaProvider()
models = await provider.list_available_models()
for model_name in models:
    print(f"  - {model_name}")
```

### Pull a Model

```python
provider = OllamaProvider()
result = await provider.pull_model("llama2")
if "error" not in result:
    print("Model pulled successfully")
```

### Get Model Information

```python
model = OllamaModel("llama2")
info = await model.get_model_info()
print(f"Model size: {info.get('size')}")
print(f"Parameters: {info.get('parameter_size')}")
```

## CLI Commands

```bash
# List available Ollama models
python -m vivasecuris.aiasylum.cli ollama list

# Pull a model
python -m vivasecuris.aiasylum.cli ollama pull llama2

# Check if model is available
python -m vivasecuris.aiasylum.cli ollama check llama2

# Get model information
python -m vivasecuris.aiasylum.cli ollama info llama2
```

## Integration with Tests

Ollama models work seamlessly with the test framework:

```python
from vivasecuris.aiasylum.runner import TestRunner

runner = TestRunner()
test_run = await runner.run_test(
    doctor_provider="ollama",
    doctor_model="llama3.2",
    patient_provider="ollama",
    patient_model="llama2",
    test_type="conversation",
)
```

## Advanced Features

### Streaming Responses

```python
model = OllamaModel("llama2")
async for chunk in model.stream_generate("Tell me a story"):
    print(chunk, end="", flush=True)
```

### Custom Options

```python
response = await model.generate(
    "Hello",
    temperature=0.9,
    top_p=0.95,
    top_k=40,
    repeat_penalty=1.1,
)
```

### Message-Based Conversations

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"},
    {"role": "user", "content": "How are you?"},
]

response = await model.generate("", messages=messages)
```

## Benefits

1. **No API Keys Required**: Ollama works completely offline
2. **Full Control**: Manage models locally
3. **Better Performance**: No network latency for local models
4. **Privacy**: All data stays local
5. **Cost Effective**: No API costs
6. **Enhanced Features**: Model management, info, availability checks

## Configuration

Set in `.env`:

```bash
OLLAMA_BASE_URL=http://localhost:11434
```

Or use a remote Ollama instance:

```bash
OLLAMA_BASE_URL=http://your-ollama-server:11434
```

## Troubleshooting

### Model Not Found

```python
# Check if model exists
available = await model.check_available()
if not available:
    # Pull the model
    await model.pull_model()
```

### Connection Errors

- Ensure Ollama is running: `ollama serve`
- Check `OLLAMA_BASE_URL` in `.env`
- Verify network connectivity if using remote Ollama

### Model Pulling

Pulling large models can take time. The pull operation streams progress:

```python
result = await provider.pull_model("llama2")
# Check result["status"] for progress
```
