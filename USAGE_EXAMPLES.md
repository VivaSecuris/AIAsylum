# Usage Examples

Real-world examples of how to use AI Asylum.

## Example 1: Quick Safety Test with Ollama

Test a local model's safety without any API keys:

```bash
# Start Ollama
ollama serve

# Pull a model
ollama pull llama2

# Run a conversation test
python -m vivasecuris.aiasylum.cli run \
  --doctor-provider ollama \
  --doctor-model llama2 \
  --patient-provider ollama \
  --patient-model llama2 \
  --test-type conversation

# View results
python -m vivasecuris.aiasylum.cli results
```

## Example 2: Compare Two Models

Compare how two different models handle the same scenarios:

```bash
# Test Model A
python -m vivasecuris.aiasylum.cli run \
  --doctor-provider openai \
  --doctor-model gpt-4 \
  --patient-provider openai \
  --patient-model gpt-3.5-turbo \
  --test-type scenario

# Test Model B
python -m vivasecuris.aiasylum.cli run \
  --doctor-provider openai \
  --doctor-model gpt-4 \
  --patient-provider anthropic \
  --patient-model claude-3-haiku \
  --test-type scenario

# Compare results via API
curl http://localhost:8000/api/v1/test-runs/ | jq
```

## Example 3: Jailbreak Testing

Test a model's resistance to prompt injection:

```bash
python -m vivasecuris.aiasylum.cli run \
  --doctor-provider ollama \
  --doctor-model llama3.2 \
  --patient-provider ollama \
  --patient-model llama2-uncensored \
  --test-type adversarial
```

## Example 4: API Workflow

Use the REST API for programmatic access:

```bash
# Start API server
make run-api

# Create a test run
curl -X POST http://localhost:8000/api/v1/test-runs/ \
  -H "Content-Type: application/json" \
  -d '{
    "doctor_provider": "ollama",
    "doctor_model": "llama2",
    "patient_provider": "ollama",
    "patient_model": "llama2",
    "test_type": "conversation"
  }'

# Get test run
curl http://localhost:8000/api/v1/test-runs/1

# Get conversation
curl http://localhost:8000/api/v1/test-runs/1/conversation

# Run analysis
curl -X POST http://localhost:8000/api/v1/analysis/test-run/1 \
  -H "Content-Type: application/json" \
  -d '{
    "enable_cot_detection": true,
    "cot_analysis_mode": "full"
  }'
```

## Example 5: Docker Deployment

Deploy everything with Docker:

```bash
# Create .env file
cp .env.example .env
# Edit .env with your settings

# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f api

# Test API
curl http://localhost:8000/health

# Stop services
docker-compose down
```

## Example 6: Automated Testing Script

Create a script to run multiple tests:

```bash
#!/bin/bash
# test_suite.sh

MODELS=("llama2" "llama3" "mistral")
TEST_TYPES=("conversation" "scenario" "adversarial")

for model in "${MODELS[@]}"; do
  for test_type in "${TEST_TYPES[@]}"; do
    echo "Testing $model with $test_type..."
    python -m vivasecuris.aiasylum.cli run \
      --doctor-provider ollama \
      --doctor-model llama3.2 \
      --patient-provider ollama \
      --patient-model "$model" \
      --test-type "$test_type"
  done
done

echo "All tests completed!"
python -m vivasecuris.aiasylum.cli results
```

## Example 7: Python Script Integration

Use AI Asylum in your own Python scripts:

```python
import asyncio
from vivasecuris.aiasylum.runner import TestRunner
from vivasecuris.aiasylum.models import get_provider

async def run_custom_test():
    runner = TestRunner()
    
    test_run = await runner.run_test(
        doctor_provider="ollama",
        doctor_model="llama3.2",
        patient_provider="ollama",
        patient_model="llama2",
        test_type="conversation",
        test_config={"max_turns": 5}
    )
    
    print(f"Test completed: {test_run.id}")
    print(f"Status: {test_run.status}")
    
    # Access results
    for result in test_run.results:
        print(f"Result: {result.test_name}, Score: {result.score}")

# Run it
asyncio.run(run_custom_test())
```

## Example 8: Continuous Monitoring

Set up a monitoring script:

```python
#!/usr/bin/env python3
"""Monitor model safety over time."""

import time
import asyncio
from vivasecuris.aiasylum.runner import TestRunner
from vivasecuris.aiasylum.analysis import AnalysisService

async def monitor_model():
    runner = TestRunner()
    analyzer = AnalysisService()
    
    while True:
        # Run safety test
        test_run = await runner.run_test(
            doctor_provider="ollama",
            doctor_model="llama3.2",
            patient_provider="ollama",
            patient_model="llama2",
            test_type="adversarial"
        )
        
        # Analyze results
        assessment = await analyzer.analyze_test_run(test_run.id)
        
        # Alert if safety score drops
        if assessment.overall_score < 0.7:
            print(f"⚠️  WARNING: Safety score dropped to {assessment.overall_score}")
        
        # Wait before next test
        await asyncio.sleep(3600)  # 1 hour

asyncio.run(monitor_model())
```

## Example 9: Batch Testing Multiple Models

Test multiple models in parallel:

```python
import asyncio
from vivasecuris.aiasylum.runner import TestRunner

async def test_model(provider, model):
    runner = TestRunner()
    return await runner.run_test(
        doctor_provider="ollama",
        doctor_model="llama3.2",
        patient_provider=provider,
        patient_model=model,
        test_type="conversation"
    )

async def batch_test():
    models = [
        ("ollama", "llama2"),
        ("ollama", "llama3"),
        ("ollama", "mistral"),
    ]
    
    tasks = [test_model(provider, model) for provider, model in models]
    results = await asyncio.gather(*tasks)
    
    for result in results:
        print(f"Model: {result.patient_model}, Status: {result.status}")

asyncio.run(batch_test())
```

## Example 10: Export Results

Export test results for analysis:

```python
from vivasecuris.aiasylum.database import get_session, TestRun
import json

session = get_session()
test_runs = session.query(TestRun).all()

results = []
for run in test_runs:
    results.append({
        "id": run.id,
        "doctor": f"{run.doctor_model} ({run.doctor_provider})",
        "patient": f"{run.patient_model} ({run.patient_provider})",
        "type": run.test_type,
        "status": run.status,
        "created": run.created_at.isoformat(),
    })

with open("test_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"Exported {len(results)} test runs")
```
