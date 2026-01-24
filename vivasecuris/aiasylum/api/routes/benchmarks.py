"""Benchmark routes."""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional

from vivasecuris.aiasylum.runner import TestRunner
from vivasecuris.aiasylum.database import get_session, TestRun
from vivasecuris.aiasylum.constants import STATUS_PENDING

router = APIRouter()


class BenchmarkRequest(BaseModel):
    """Benchmark execution request."""
    provider: str
    model: str
    benchmark: str
    num_samples: Optional[int] = None


class BenchmarkResponse(BaseModel):
    """Benchmark execution response."""
    id: int
    status: str
    message: str
    test_run_id: Optional[int] = None


@router.get("/list")
async def list_benchmarks():
    """List available benchmarks."""
    return {
        "benchmarks": [
            {"name": "mmlu", "description": "Massive Multitask Language Understanding"},
            {"name": "truthfulqa", "description": "TruthfulQA benchmark"},
            {"name": "hellaswag", "description": "HellaSwag commonsense reasoning"},
            {"name": "arc", "description": "AI2 Reasoning Challenge"},
            {"name": "math", "description": "MATH dataset"},
            {"name": "gsm8k", "description": "Grade School Math 8K"},
            {"name": "winogrande", "description": "Winogrande commonsense reasoning"},
            {"name": "piqa", "description": "Physical Interaction QA"},
            {"name": "bbq", "description": "Bias Benchmark for QA"},
            {"name": "realtoxicityprompts", "description": "RealToxicityPrompts"},
        ]
    }


async def _run_benchmark_background(test_run_id: int):
    """Background task to run a benchmark test run."""
    runner = TestRunner()
    try:
        await runner.execute_test_run(test_run_id)
    except Exception as e:
        # Error is already handled in execute_test_run (sets status to failed)
        print(f"Error running benchmark test {test_run_id}: {e}")


@router.post("/run", response_model=BenchmarkResponse)
async def run_benchmark(request: BenchmarkRequest, background_tasks: BackgroundTasks):
    """Run a benchmark by creating a test run with benchmark configuration."""
    # For now, benchmarks are implemented as test runs with benchmark-specific configuration
    # This allows us to reuse the existing test infrastructure
    
    # Map benchmark names to test types
    # Most benchmarks can be run as conversation tests with specific prompts
    benchmark_to_test_type = {
        "mmlu": "conversation",
        "truthfulqa": "conversation",
        "hellaswag": "conversation",
        "arc": "conversation",
        "math": "conversation",
        "gsm8k": "conversation",
        "winogrande": "conversation",
        "piqa": "conversation",
        "bbq": "conversation",
        "realtoxicityprompts": "adversarial",
    }
    
    test_type = benchmark_to_test_type.get(request.benchmark.lower(), "conversation")
    
    # Create test run with benchmark configuration
    session = get_session()
    try:
        test_run = TestRun(
            doctor_provider=request.provider,
            doctor_model=request.model,
            patient_provider=request.provider,  # Use same provider/model for benchmarks
            patient_model=request.model,
            test_type=test_type,
            status=STATUS_PENDING,
            meta_data={
                "benchmark": request.benchmark,
                "num_samples": request.num_samples,
                "test_config": {
                    "benchmark_name": request.benchmark,
                    "num_samples": request.num_samples or 100,
                }
            },
        )
        session.add(test_run)
        session.commit()
        session.refresh(test_run)
        
        # Run benchmark in background
        background_tasks.add_task(_run_benchmark_background, test_run.id)
        
        return BenchmarkResponse(
            id=test_run.id,
            status="pending",
            message=f"Benchmark {request.benchmark} started for model {request.model}",
            test_run_id=test_run.id,
        )
    finally:
        session.close()
