"""Benchmark routes."""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional

router = APIRouter()


class BenchmarkRequest(BaseModel):
    """Benchmark execution request."""
    provider: str
    model: str
    benchmark: str
    num_samples: Optional[int] = None


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


@router.post("/run")
async def run_benchmark(request: BenchmarkRequest):
    """Run a benchmark (placeholder - to be implemented)."""
    return {
        "status": "not_implemented",
        "message": "Benchmark execution will be implemented in benchmarks module",
        "request": request.dict(),
    }
