# Benchmark Integration

AI Asylum integrates with standard AI evaluation benchmarks for comprehensive model assessment.

## Available Benchmarks

### Knowledge & Reasoning
- **MMLU**: Massive Multitask Language Understanding
- **ARC**: AI2 Reasoning Challenge
- **MATH**: Mathematics dataset
- **GSM8K**: Grade School Math 8K

### Commonsense Reasoning
- **HellaSwag**: Commonsense NLI
- **WinoGrande**: Winograd schema challenge
- **PIQA**: Physical Interaction QA

### Safety & Alignment
- **TruthfulQA**: Truthfulness in QA
- **BBQ**: Bias Benchmark for QA
- **RealToxicityPrompts**: Toxicity evaluation

## Usage

### CLI

```bash
# List available benchmarks
python -m vivasecuris.aiasylum.cli list-benchmarks

# Run a single benchmark
python -m vivasecuris.aiasylum.cli run-benchmark \
  --provider ollama \
  --model llama3.2 \
  --benchmark mmlu \
  --num-samples 100
```

### API

```bash
# List benchmarks
curl http://localhost:8000/api/v1/benchmarks/list

# Run benchmark
curl -X POST http://localhost:8000/api/v1/benchmarks/run \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "ollama",
    "model": "llama3.2",
    "benchmark": "mmlu",
    "num_samples": 100
  }'
```

## Implementation Status

Benchmark integration is currently a placeholder. Full implementation will include:

- Dataset loading from HuggingFace
- Evaluation metrics calculation
- Results storage in database
- Comparison across models
- Visualization in dashboard

## Adding New Benchmarks

To add a new benchmark:

1. Create benchmark loader in `vivasecuris/aiasylum/benchmarks/`
2. Implement evaluation logic
3. Add to benchmark registry
4. Update API and CLI
