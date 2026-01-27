"""Benchmark routes."""

import logging
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from vivasecuris.aiasylum.runner import TestRunner
from vivasecuris.aiasylum.database import get_session, TestRun, PromptLibrary, TestSuite
from vivasecuris.aiasylum.constants import STATUS_PENDING, TEST_TYPE_BENCHMARK
from vivasecuris.aiasylum.suites import SuiteRunner
from vivasecuris.aiasylum.benchmarks.datasets import (
    load_benchmark_dataset_all,
    parse_index_selection,
    filter_prompts_by_selection,
    BENCHMARK_DATASETS,
)

logger = logging.getLogger(__name__)
router = APIRouter()


class BenchmarkRequest(BaseModel):
    """Benchmark execution request."""
    provider: str
    model: str
    benchmark: str
    num_samples: Optional[int] = None
    suite_id: Optional[int] = None  # Optional suite ID to link benchmark run to a suite


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
            {"name": "jailbreak", "description": "Jailbreak Resistance Benchmark - Tests model's ability to resist prompt injection attacks"},
        ]
    }


async def _run_benchmark_background(test_run_id: int):
    """Background task to run a benchmark test run with worker pool limiting."""
    import traceback
    from vivasecuris.aiasylum.api.worker_pool import worker_pool
    
    logger.info(f"Starting benchmark background task for test run {test_run_id}")
    runner = TestRunner()
    
    # Run with worker pool limit
    async def _execute():
        await runner.execute_test_run(test_run_id)
    
    try:
        await worker_pool.run_with_limit(test_run_id, _execute())
        logger.info(f"Benchmark test run {test_run_id} completed successfully")
    except Exception as e:
        # Error is already handled in execute_test_run (sets status to failed)
        error_trace = traceback.format_exc()
        logger.error(f"Error running benchmark test {test_run_id}: {e}")
        logger.error(f"Traceback: {error_trace}")
        print(f"Error running benchmark test {test_run_id}: {e}")
        print(f"Traceback: {error_trace}")


@router.post("/run", response_model=BenchmarkResponse)
async def run_benchmark(request: BenchmarkRequest, background_tasks: BackgroundTasks):
    """Run a benchmark by creating a test run with benchmark configuration."""
    # For now, benchmarks are implemented as test runs with benchmark-specific configuration
    # This allows us to reuse the existing test infrastructure
    
    # Benchmarks use the benchmark test type
    from vivasecuris.aiasylum.constants import TEST_TYPE_BENCHMARK
    test_type = TEST_TYPE_BENCHMARK
    
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
            suite_id=request.suite_id,
            meta_data={
                "benchmark": request.benchmark,
                "num_samples": request.num_samples,
                "test_config": {
                    "benchmark_name": request.benchmark,
                    "num_samples": request.num_samples or 100,
                    # Jailbreak benchmarks default to one_shot mode
                    # Individual prompts will be handled based on their is_multi_shot flag
                    # This allows single-shot and multi-shot prompts to be mixed properly
                    "test_mode": request.test_mode or ("one_shot" if request.benchmark.lower() == "jailbreak" else "one_shot"),
                }
            },
        )
        session.add(test_run)
        session.commit()
        session.refresh(test_run)
        
        # Run benchmark in background using asyncio.create_task (same as regular test runs)
        import asyncio
        from vivasecuris.aiasylum.api.cancellation import cancellation_manager
        
        # Create and register the asyncio task so we can cancel it later
        task = asyncio.create_task(_run_benchmark_background(test_run.id))
        cancellation_manager.register_task(test_run.id, task)
        
        logger.info(f"Benchmark test run {test_run.id} queued for execution")
        
        return BenchmarkResponse(
            id=test_run.id,
            status="pending",
            message=f"Benchmark {request.benchmark} started for model {request.model}",
            test_run_id=test_run.id,
        )
    finally:
        session.close()


@router.get("/prompts/metadata")
async def get_prompts_metadata(
    benchmark: str = Query(..., description="Benchmark name (e.g., 'mmlu', 'gsm8k', 'jailbreak')"),
):
    """Get metadata about prompts in a benchmark dataset without loading all data."""
    # Import here to avoid circular import
    from vivasecuris.aiasylum.benchmarks.datasets import BENCHMARK_DATASETS
    
    if benchmark.lower() not in BENCHMARK_DATASETS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown benchmark: {benchmark}. Available: {list(BENCHMARK_DATASETS.keys())}"
        )
    
    try:
        # Load all prompts to get metadata
        all_prompts = await load_benchmark_dataset_all(benchmark)
        
        # Extract unique subjects if available
        subjects = set()
        for prompt in all_prompts:
            if "subject" in prompt and prompt["subject"]:
                subjects.add(prompt["subject"])
        
        # Count prompts by subject
        subject_counts = {}
        for prompt in all_prompts:
            subject = prompt.get("subject", "unknown")
            subject_counts[subject] = subject_counts.get(subject, 0) + 1
        
        return {
            "benchmark": benchmark,
            "total_samples": len(all_prompts),
            "subjects": sorted(list(subjects)) if subjects else None,
            "subject_counts": subject_counts,
            "has_choices": any("choices" in p and p["choices"] for p in all_prompts),
            "has_answers": any("answer" in p and p["answer"] is not None for p in all_prompts),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading benchmark metadata: {str(e)}")


@router.get("/prompts")
async def get_prompts(
    benchmark: str = Query(..., description="Benchmark name (e.g., 'mmlu', 'gsm8k', 'jailbreak')"),
    indices: Optional[str] = Query(None, description="Comma-separated indices or ranges (e.g., '0,5,10,15-20')"),
    subject: Optional[str] = Query(None, description="Filter by subject (e.g., 'abstract_algebra' for MMLU)"),
    num_samples: Optional[int] = Query(None, description="Number of samples to return (random selection if indices not specified)"),
):
    """
    Get prompts from a benchmark dataset.
    
    Supports manual selection via indices or subject filtering.
    If neither indices nor subject is specified, returns first num_samples (or all if num_samples not specified).
    Note: For 'jailbreak' benchmark, indices/subject filtering is not supported (loads from database).
    """
    if benchmark.lower() not in BENCHMARK_DATASETS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown benchmark: {benchmark}. Available: {list(BENCHMARK_DATASETS.keys())}"
        )
    
    # Jailbreak benchmark doesn't support indices/subject filtering (loads from database)
    if benchmark.lower() == "jailbreak" and (indices or subject):
        raise HTTPException(
            status_code=400,
            detail="Jailbreak benchmark does not support indices or subject filtering. Use num_samples to limit the number of prompts."
        )
    
    try:
        # Load all prompts
        all_prompts = await load_benchmark_dataset_all(benchmark)
        total_samples = len(all_prompts)
        
        # Parse indices if provided
        selected_indices = None
        if indices:
            try:
                selected_indices = parse_index_selection(indices, total_samples)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Invalid indices format: {str(e)}")
        
        # Filter prompts
        if selected_indices is not None or subject:
            filtered_prompts = filter_prompts_by_selection(
                all_prompts,
                indices=selected_indices,
                subject=subject,
            )
        else:
            # If no manual selection, use num_samples or all
            if num_samples:
                filtered_prompts = all_prompts[:num_samples]
            else:
                filtered_prompts = all_prompts
        
        # Format response
        formatted_prompts = []
        for i, prompt in enumerate(filtered_prompts):
            formatted_prompt = {
                "index": i,
                "question": prompt.get("question", ""),
                "choices": prompt.get("choices"),
                "answer": prompt.get("answer"),
                "answer_letter": prompt.get("answer_letter"),
                "answer_index": prompt.get("answer_index"),
                "subject": prompt.get("subject"),
            }
            # Optionally include raw data (can be large, so make it optional)
            # formatted_prompt["raw"] = prompt.get("raw")
            formatted_prompts.append(formatted_prompt)
        
        return {
            "benchmark": benchmark,
            "total_samples": total_samples,
            "returned_samples": len(formatted_prompts),
            "prompts": formatted_prompts,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading prompts: {str(e)}")


class PromptExportRequest(BaseModel):
    """Request to export prompts to prompt library."""
    benchmark: str
    indices: Optional[str] = None  # Comma-separated indices or ranges
    subject: Optional[str] = None  # Filter by subject
    num_samples: Optional[int] = None  # Number of samples (if indices/subject not specified)
    name_pattern: Optional[str] = "{benchmark} Question {index}"  # Pattern for prompt names
    category: Optional[str] = "benchmark"
    tags: Optional[List[str]] = None
    description: Optional[str] = None


class PromptExportResponse(BaseModel):
    """Response from prompt export."""
    benchmark: str
    exported_count: int
    prompt_ids: List[int]
    message: str


@router.post("/prompts/export", response_model=PromptExportResponse)
async def export_prompts(request: PromptExportRequest):
    """
    Export prompts from a benchmark dataset to the prompt library.
    
    Supports manual selection via indices or subject filtering.
    """
    if request.benchmark.lower() not in BENCHMARK_DATASETS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown benchmark: {request.benchmark}. Available: {list(BENCHMARK_DATASETS.keys())}"
        )
    
    try:
        # Load all prompts
        all_prompts = await load_benchmark_dataset_all(request.benchmark)
        total_samples = len(all_prompts)
        
        # Parse indices if provided
        selected_indices = None
        if request.indices:
            try:
                selected_indices = parse_index_selection(request.indices, total_samples)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Invalid indices format: {str(e)}")
        
        # Filter prompts
        if selected_indices is not None or request.subject:
            filtered_prompts = filter_prompts_by_selection(
                all_prompts,
                indices=selected_indices,
                subject=request.subject,
            )
        else:
            # If no manual selection, use num_samples or all
            if request.num_samples:
                filtered_prompts = all_prompts[:request.num_samples]
            else:
                filtered_prompts = all_prompts
        
        if not filtered_prompts:
            raise HTTPException(status_code=400, detail="No prompts match the selection criteria")
        
        # Create prompt library entries
        session = get_session()
        prompt_ids = []
        default_tags = request.tags or [request.benchmark.lower(), "benchmark"]
        
        try:
            # Track original indices
            # If indices were specified, use those as original indices
            # Otherwise, we can't reliably determine original indices (e.g., when filtering by subject)
            original_indices_list = selected_indices if selected_indices is not None else None
            
            for i, prompt in enumerate(filtered_prompts):
                # Get original index if available, otherwise use filtered position
                if original_indices_list and i < len(original_indices_list):
                    original_index = original_indices_list[i]
                else:
                    # Can't determine original index (filtered by subject or no index selection)
                    original_index = None
                
                # Format prompt text - include question and choices if available
                prompt_text = prompt.get("question", "")
                if prompt.get("choices"):
                    choices_text = "\n".join([f"{chr(65+j)}. {choice}" for j, choice in enumerate(prompt["choices"])])
                    prompt_text = f"{prompt_text}\n\n{choices_text}"
                
                # Format name using pattern (use filtered index for naming)
                name = request.name_pattern.format(
                    benchmark=request.benchmark.upper(),
                    index=i,
                    subject=prompt.get("subject", "unknown"),
                )
                
                # Check if name already exists
                existing = session.query(PromptLibrary).filter(PromptLibrary.name == name).first()
                if existing:
                    # Append index to make it unique
                    name = f"{name} ({i})"
                
                # Create prompt library entry
                db_prompt = PromptLibrary(
                    name=name,
                    description=request.description or f"Question from {request.benchmark.upper()} benchmark",
                    prompt_text=prompt_text,
                    prompt_type="test_prompt",
                    target=None,
                    category=request.category or "benchmark",
                    tags=default_tags,
                    meta_data={
                        "benchmark": request.benchmark,
                        "original_index": original_index,
                        "export_index": i,
                        "subject": prompt.get("subject"),
                        "answer": prompt.get("answer"),
                        "answer_letter": prompt.get("answer_letter"),
                        "answer_index": prompt.get("answer_index"),
                    },
                )
                session.add(db_prompt)
                session.flush()  # Get the ID
                prompt_ids.append(db_prompt.id)
            
            session.commit()
            
            return PromptExportResponse(
                benchmark=request.benchmark,
                exported_count=len(prompt_ids),
                prompt_ids=prompt_ids,
                message=f"Successfully exported {len(prompt_ids)} prompts to prompt library",
            )
        except Exception as e:
            session.rollback()
            raise HTTPException(status_code=500, detail=f"Error exporting prompts: {str(e)}")
        finally:
            session.close()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading benchmark: {str(e)}")


class CreateSuiteFromPromptsRequest(BaseModel):
    """Request to create a suite from manually selected benchmark prompts."""
    benchmark: str
    name: Optional[str] = None
    models: List[dict]  # [{"provider": "ollama", "model": "llama3.2"}, ...]
    indices: Optional[str] = None  # Comma-separated indices or ranges (e.g., "0,5,10,15-20")
    subject: Optional[str] = None  # Filter by subject
    test_mode: Optional[str] = "one_shot"  # "one_shot" or "multi_shot"
    test_config: Optional[dict] = None
    start_immediately: Optional[bool] = True  # Whether to start test runs immediately


class CreateSuiteFromPromptsResponse(BaseModel):
    """Response from suite creation."""
    suite_id: int
    benchmark: str
    total_runs: int
    message: str


@router.post("/prompts/create-suite", response_model=CreateSuiteFromPromptsResponse)
async def create_suite_from_prompts(
    request: CreateSuiteFromPromptsRequest,
    background_tasks: BackgroundTasks,
):
    """
    Create a test suite from manually selected benchmark prompts.
    
    This creates a suite that can be run again later with the same selected prompts.
    """
    if request.benchmark.lower() not in BENCHMARK_DATASETS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown benchmark: {request.benchmark}. Available: {list(BENCHMARK_DATASETS.keys())}"
        )
    
    if not request.models:
        raise HTTPException(status_code=400, detail="At least one model must be specified")
    
    # Validate models
    for i, model in enumerate(request.models):
        if not isinstance(model, dict):
            raise HTTPException(
                status_code=400,
                detail=f"Model {i+1} must be an object with 'provider' and 'model' fields"
            )
        if not model.get("provider") or not model.get("model"):
            raise HTTPException(
                status_code=400,
                detail=f"Model {i+1} must have both 'provider' and 'model' specified"
            )
    
    # Validate that either indices or subject is provided (or both)
    # Exception: jailbreak benchmark doesn't require indices/subject (loads from database)
    if request.benchmark.lower() != "jailbreak" and not request.indices and not request.subject:
        raise HTTPException(
            status_code=400,
            detail="Either 'indices' or 'subject' must be specified for manual selection (not required for 'jailbreak' benchmark)"
        )
    
    # Jailbreak benchmark doesn't support indices/subject filtering
    if request.benchmark.lower() == "jailbreak" and (request.indices or request.subject):
        raise HTTPException(
            status_code=400,
            detail="Jailbreak benchmark does not support indices or subject filtering. It loads prompts from the database."
        )
    
    # Parse indices if provided (not applicable for jailbreak)
    selected_indices = None
    if request.indices and request.benchmark.lower() != "jailbreak":
        try:
            # Load all prompts to get max index for validation
            all_prompts = await load_benchmark_dataset_all(request.benchmark)
            max_index = len(all_prompts)
            selected_indices = parse_index_selection(request.indices, max_index)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid indices format: {str(e)}")
    
    # For jailbreak, default to one_shot mode (individual prompts will be handled based on is_multi_shot flag)
    test_mode = request.test_mode or "one_shot"
    
    # Create suite
    session = get_session()
    try:
        suite_name = request.name or f"{request.benchmark.upper()} Suite (Selected Prompts)"
        
        # Create suite record
        suite = TestSuite(
            name=suite_name,
            status="pending",
            total_runs=0,
            completed_runs=0,
            failed_runs=0,
            running_runs=0,
            pending_runs=0,
            meta_data={
                "benchmark": request.benchmark,
                "selected_indices": request.indices,  # Store as string for reference
                "selected_indices_list": selected_indices,  # Store as list for use
                "selected_subject": request.subject,
                "test_mode": test_mode,  # Use computed test_mode (multi_shot for jailbreak)
                "models": request.models,
                "test_config": request.test_config or {},
            },
        )
        session.add(suite)
        session.commit()
        session.refresh(suite)
        
        # Create test runs for each model
        test_run_ids = []
        for model in request.models:
            test_run = TestRun(
                doctor_provider=model["provider"],
                doctor_model=model["model"],
                patient_provider=model["provider"],
                patient_model=model["model"],
                test_type=TEST_TYPE_BENCHMARK,
                status=STATUS_PENDING,
                suite_id=suite.id,
                meta_data={
                    "benchmark": request.benchmark,
                    "selected_indices": request.indices,  # Store string format
                    "selected_indices_list": selected_indices,  # Store parsed list
                    "selected_subject": request.subject,
                    "test_config": {
                        "benchmark_name": request.benchmark,
                        "test_mode": test_mode,  # Use computed test_mode (multi_shot for jailbreak)
                        **(request.test_config or {}),
                    },
                    "suite_id": suite.id,
                },
            )
            session.add(test_run)
            test_run_ids.append(test_run.id)
        
        # Update suite total_runs
        suite.total_runs = len(test_run_ids)
        suite.pending_runs = len(test_run_ids)
        suite.meta_data["test_run_ids"] = test_run_ids
        session.commit()
        session.refresh(suite)
        
        # Start test runs if requested
        if request.start_immediately:
            runner = TestRunner()
            for test_run_id in test_run_ids:
                background_tasks.add_task(_run_benchmark_background, test_run_id)
            message = f"Suite created and started with {len(test_run_ids)} test run(s) using manually selected prompts"
        else:
            message = f"Suite created with {len(test_run_ids)} test run(s) (pending) using manually selected prompts"
        
        return CreateSuiteFromPromptsResponse(
            suite_id=suite.id,
            benchmark=request.benchmark,
            total_runs=len(test_run_ids),
            message=message,
        )
    except HTTPException:
        raise
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=f"Error creating suite: {str(e)}")
    finally:
        session.close()
