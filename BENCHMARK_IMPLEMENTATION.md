# Benchmark Implementation

## Overview

Benchmarks are now properly implemented as a distinct test type that downloads datasets from HuggingFace and runs evaluations in either one-shot or multi-shot mode.

## Key Changes

### 1. Added Benchmark Test Type

- Added `TEST_TYPE_BENCHMARK = "benchmark"` constant
- Benchmarks are now properly labeled as "benchmark" test type (not conversation tests)

### 2. Created BenchmarkTest Class

**Location**: `vivasecuris/aiasylum/tests/benchmark.py`

- Extends `TestCase` base class
- Supports two modes:
  - **one_shot**: Each question is asked independently (no conversation context)
  - **multi_shot**: Questions are asked sequentially in a conversation (maintains context)
- Loads datasets from HuggingFace
- Evaluates responses against ground truth
- Calculates accuracy scores

### 3. Dataset Loader

**Location**: `vivasecuris/aiasylum/benchmarks/datasets.py`

- Downloads datasets from HuggingFace using the `datasets` library
- Supports all major benchmarks:
  - MMLU (Massive Multitask Language Understanding)
  - TruthfulQA
  - HellaSwag
  - ARC (AI2 Reasoning Challenge)
  - MATH
  - GSM8K
  - WinoGrande
  - PIQA
  - BBQ (Bias Benchmark for QA)
  - RealToxicityPrompts
- Standardizes dataset format across different benchmark types
- Falls back to simple questions if dataset loading fails

### 4. Updated Test Runner

- Detects `TEST_TYPE_BENCHMARK` and runs `BenchmarkTest`
- Passes benchmark configuration (name, num_samples, test_mode)
- Stores benchmark results with accuracy scores

### 5. Updated API Routes

- Benchmarks now create test runs with `test_type="benchmark"`
- No longer maps to conversation test type
- Properly labeled in database and UI

## Usage

### CLI

```bash
# Run a benchmark in one-shot mode (default)
python -m vivasecuris.aiasylum.cli run-benchmark \
  --provider ollama \
  --model llama3.2 \
  --benchmark mmlu \
  --num-samples 100

# Run a benchmark in multi-shot mode (via API or test config)
```

### API

```bash
# Run benchmark
curl -X POST http://localhost:8000/api/v1/benchmarks/run \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "ollama",
    "model": "llama3.2",
    "benchmark": "gsm8k",
    "num_samples": 50
  }'
```

### Programmatic

```python
from vivasecuris.aiasylum.tests import BenchmarkTest
from vivasecuris.aiasylum.models import get_provider

provider = get_provider("ollama")
model = provider.create_model("llama3.2")

# One-shot mode (each question independent)
test = BenchmarkTest(
    benchmark_name="mmlu",
    num_samples=100,
    test_mode="one_shot",
)
result = await test.run(model)

# Multi-shot mode (sequential questions)
test = BenchmarkTest(
    benchmark_name="gsm8k",
    num_samples=50,
    test_mode="multi_shot",
)
result = await test.run(model)
```

## Dataset Sources

All datasets are loaded from HuggingFace:

- **MMLU**: `cais/mmlu`
- **TruthfulQA**: `truthful_qa`
- **HellaSwag**: `Rowan/hellaswag`
- **ARC**: `allenai/ai2_arc`
- **MATH**: `lighteval/MATH`
- **GSM8K**: `gsm8k`
- **WinoGrande**: `winogrande`
- **PIQA**: `piqa`
- **BBQ**: `bbq`
- **RealToxicityPrompts**: `allenai/real-toxicity-prompts`

## Test Modes

### One-Shot Mode
- Each question is asked independently
- No conversation context between questions
- Best for: Testing model's knowledge and reasoning on isolated questions
- Use case: Standard benchmark evaluation

### Multi-Shot Mode
- Questions are asked sequentially
- Maintains conversation context
- Best for: Testing model's ability to handle sequential questions
- Use case: Context-dependent reasoning, conversation benchmarks

## Results Format

Benchmark results include:
- **Accuracy**: Overall percentage correct
- **Correct/Total**: Number of correct answers vs total questions
- **Per-Question Results**: Detailed results for each question including:
  - Question text
  - Model response
  - Ground truth answer
  - Whether response was correct
  - Reasoning (if CoT enabled)
- **Conversation History**: Full conversation for multi-shot mode

## Dependencies

Requires the `datasets` library (already in requirements.txt):
```bash
pip install datasets
```

The library automatically downloads datasets from HuggingFace on first use.

## Future Enhancements

1. **Custom Evaluation Metrics**: Implement benchmark-specific evaluation (e.g., exact match, F1 score)
2. **Multiple Choice Handling**: Better parsing of multiple choice responses
3. **Caching**: Cache downloaded datasets locally
4. **Progress Tracking**: Show progress for large benchmark runs
5. **Comparison**: Compare results across different models
6. **Visualization**: Charts and graphs for benchmark results
