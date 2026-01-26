"""Benchmark dataset loaders from HuggingFace and other sources."""

from typing import Dict, List, Optional, Any
import logging
import random
import time
import os

from vivasecuris.aiasylum.tests.jailbreak_loader import load_jailbreak_prompts

logger = logging.getLogger(__name__)

# Benchmark dataset mappings (HuggingFace dataset names)
BENCHMARK_DATASETS = {
    "mmlu": {
        "dataset": "cais/mmlu",
        "config": "all",  # MMLU requires a config name - 'all' loads all subjects
        "split": "test",
        "question_field": "question",
        "answer_field": "answer",  # This is 0-3 (class label), needs conversion
        "choices_field": "choices",  # List of 4 choices
        "subject_field": "subject",  # Subject category
    },
    "truthfulqa": {
        "dataset": "truthfulqa/truthful_qa",
        "config": "generation",  # Use config name, not split
        "split": "validation",
        "question_field": "question",
        "answer_field": "best_answer",
        "correct_answers_field": "correct_answers",  # Multiple correct answers
    },
    "hellaswag": {
        "dataset": "Rowan/hellaswag",
        "split": "validation",
        "question_field": "ctx",  # Context sentence
        "answer_field": "label",  # This is 0-3 (class label), needs conversion
        "choices_field": "endings",  # List of 4 endings
    },
    "arc": {
        "dataset": "allenai/ai2_arc",
        "config": "ARC-Challenge",  # ARC uses config name, not split
        "split": "test",
        "question_field": "question",
        "answer_field": "answerKey",
        "choices_field": "choices",
    },
    "math": {
        "dataset": "lighteval/MATH",
        "split": "test",
        "question_field": "problem",
        "answer_field": "solution",
    },
    "gsm8k": {
        "dataset": "gsm8k",
        "split": "test",
        "question_field": "question",
        "answer_field": "answer",
    },
    "winogrande": {
        "dataset": "winogrande",
        "split": "validation",
        "question_field": "sentence",
        "answer_field": "answer",
        "options_field": "option1,option2",
    },
    "piqa": {
        "dataset": "piqa",
        "split": "validation",
        "question_field": "goal",
        "answer_field": "label",
        "choices_field": "sol1,sol2",
    },
    "bbq": {
        "dataset": "bbq",
        "split": "test",
        "question_field": "context",
        "answer_field": "answer",
    },
    "realtoxicityprompts": {
        "dataset": "allenai/real-toxicity-prompts",
        "split": "train",
        "question_field": "prompt",
        "answer_field": None,  # This is a toxicity benchmark, no "correct" answer
    },
    "jailbreak": {
        "dataset": "internal",  # Special marker - loads from database
        "question_field": "prompt_text",
        "answer_field": "resisted",  # For jailbreaks, "resisted" means success (model didn't get jailbroken)
        "source": "database",  # Load from PromptLibrary database
    },
}


async def load_benchmark_dataset(
    benchmark_name: str,
    num_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Load benchmark dataset from HuggingFace or database.
    
    Args:
        benchmark_name: Name of the benchmark (e.g., "mmlu", "gsm8k", "jailbreak")
        num_samples: Number of samples to load (None = all)
    
    Returns:
        List of dataset items with standardized format
    """
    # Handle jailbreak benchmark (loads from database)
    if benchmark_name.lower() == "jailbreak":
        return await load_jailbreak_benchmark_dataset(num_samples)
    
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datasets library not installed. Install with: pip install datasets")
        raise ImportError("datasets library required for benchmark loading")
    
    if benchmark_name.lower() not in BENCHMARK_DATASETS:
        raise ValueError(f"Unknown benchmark: {benchmark_name}. Available: {list(BENCHMARK_DATASETS.keys())}")
    
    config = BENCHMARK_DATASETS[benchmark_name.lower()]
    dataset_name = config["dataset"]
    split = config.get("split", "test")
    dataset_config = config.get("config")  # Some datasets need config name (e.g., ARC)
    
    logger.info(f"Loading benchmark dataset: {dataset_name} (split: {split}, config: {dataset_config})")
    print(f"[load_benchmark_dataset] Starting download/load of {dataset_name} (split: {split}, config: {dataset_config})")
    
    try:
        # Load dataset - handle different dataset structures
        # Note: load_dataset is synchronous, but we're in an async function
        # This is fine - we can call sync functions from async
        try:
            if dataset_config:
                print(f"[load_benchmark_dataset] Attempting to load {dataset_name} with config={dataset_config}, split={split}")
                try:
                    # Try with config name and split (using positional argument for config, which some datasets prefer)
                    try:
                        dataset = load_dataset(dataset_name, dataset_config, split=split)
                    except Exception:
                        # Fallback to keyword argument
                        dataset = load_dataset(dataset_name, name=dataset_config, split=split)
                except Exception as e:
                    # Some datasets need config first, then access split
                    print(f"[load_benchmark_dataset] First attempt failed: {e}, trying alternative method")
                    try:
                        full_dataset = load_dataset(dataset_name, dataset_config)
                    except Exception:
                        full_dataset = load_dataset(dataset_name, name=dataset_config)
                    if split in full_dataset:
                        dataset = full_dataset[split]
                    else:
                        # Use first available split
                        available_split = list(full_dataset.keys())[0]
                        dataset = full_dataset[available_split]
                        print(f"[load_benchmark_dataset] Using split '{available_split}' instead of '{split}'")
            else:
                print(f"[load_benchmark_dataset] Attempting to load {dataset_name} with split={split}")
                dataset = load_dataset(dataset_name, split=split)
            print(f"[load_benchmark_dataset] Successfully loaded {dataset_name}, got {len(dataset)} samples")
        except Exception as e:
            # Try without split first, then select split
            logger.warning(f"Failed to load {dataset_name} with split {split}, trying without split: {e}")
            print(f"[load_benchmark_dataset] First attempt failed: {e}, trying without split")
            try:
                if dataset_config:
                    try:
                        full_dataset = load_dataset(dataset_name, dataset_config)
                    except Exception:
                        full_dataset = load_dataset(dataset_name, name=dataset_config)
                else:
                    full_dataset = load_dataset(dataset_name)
                print(f"[load_benchmark_dataset] Loaded full dataset, available splits: {list(full_dataset.keys())}")
                if split in full_dataset:
                    dataset = full_dataset[split]
                elif "test" in full_dataset:
                    dataset = full_dataset["test"]
                    print(f"[load_benchmark_dataset] Using 'test' split instead of '{split}'")
                elif "validation" in full_dataset:
                    dataset = full_dataset["validation"]
                    print(f"[load_benchmark_dataset] Using 'validation' split instead of '{split}'")
                else:
                    # Take first available split
                    available_split = list(full_dataset.keys())[0]
                    dataset = full_dataset[available_split]
                    logger.warning(f"Using split {available_split} instead of {split}")
                    print(f"[load_benchmark_dataset] Using first available split: {available_split}")
            except Exception as e2:
                logger.error(f"Failed to load dataset {dataset_name}: {e2}")
                print(f"[load_benchmark_dataset] Failed to load dataset: {e2}")
                raise
        
        # Limit samples if specified - randomize selection
        total_samples = len(dataset)
        if num_samples:
            max_samples = min(num_samples, total_samples)
            # Use a unique seed based on time and process ID to ensure different selections each run
            # This prevents the same questions from being selected when multiple tests run
            seed = int(time.time() * 1000000) + os.getpid()
            random.seed(seed)
            
            # Randomly select samples using sample() which is designed for this purpose
            selected_indices = random.sample(range(total_samples), max_samples)
            dataset = dataset.select(selected_indices)
            print(f"[load_benchmark_dataset] Randomly selected {max_samples} samples from {total_samples} total (seed: {seed})")
            print(f"[load_benchmark_dataset] Selected indices: {selected_indices[:10]}{'...' if len(selected_indices) > 10 else ''}")
        else:
            print(f"[load_benchmark_dataset] Using all {total_samples} samples")
        
        # Standardize format
        standardized = []
        question_field = config["question_field"]
        answer_field = config.get("answer_field")
        choices_field = config.get("choices_field")
        
        print(f"[load_benchmark_dataset] Standardizing format (question_field: {question_field}, answer_field: {answer_field})")
        
        for i, item in enumerate(dataset):
            if i == 0:
                print(f"[load_benchmark_dataset] Sample item keys: {list(item.keys())}")
            standardized_item = {
                "question": item.get(question_field, ""),
                "answer": item.get(answer_field) if answer_field else None,
            }
            
            # Handle answer field conversion (class labels to indices/letters)
            if answer_field and standardized_item["answer"] is not None:
                answer = standardized_item["answer"]
                # If answer is a class label (0-3), convert to letter (A-D) for multiple choice
                if isinstance(answer, (int, str)) and str(answer).isdigit():
                    answer_int = int(answer)
                    if 0 <= answer_int <= 25:  # A-Z
                        standardized_item["answer_letter"] = chr(65 + answer_int)  # A, B, C, D, etc.
                        standardized_item["answer_index"] = answer_int
            
            # Add choices if available
            if choices_field:
                if isinstance(choices_field, str) and "," in choices_field:
                    # Multiple fields (e.g., "option1,option2")
                    fields = [f.strip() for f in choices_field.split(",")]
                    choices = [item.get(f) for f in fields if item.get(f)]
                    standardized_item["choices"] = choices
                else:
                    choices = item.get(choices_field)
                    if choices:
                        standardized_item["choices"] = choices if isinstance(choices, list) else [choices]
            
            # Add subject/category if available
            subject_field = config.get("subject_field")
            if subject_field and item.get(subject_field):
                standardized_item["subject"] = item.get(subject_field)
            
            # Add raw item for reference
            standardized_item["raw"] = item
            
            standardized.append(standardized_item)
        
        logger.info(f"Loaded {len(standardized)} samples from {benchmark_name}")
        print(f"[load_benchmark_dataset] Successfully standardized {len(standardized)} samples from {benchmark_name}")
        if len(standardized) > 0:
            print(f"[load_benchmark_dataset] First sample question: {standardized[0].get('question', '')[:100]}...")
        return standardized
        
    except Exception as e:
        logger.error(f"Error loading dataset {dataset_name}: {e}")
        # Fallback to simple questions if dataset loading fails
        logger.warning(f"Falling back to simple questions for {benchmark_name}")
        return _get_fallback_questions(benchmark_name, num_samples)


def _get_fallback_questions(benchmark_name: str, num_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """Fallback questions if dataset loading fails."""
    from vivasecuris.aiasylum.benchmarks.simple import SIMPLE_MATH_QUESTIONS, SIMPLE_REASONING_QUESTIONS
    
    if benchmark_name.lower() in ["math", "gsm8k"]:
        questions = SIMPLE_MATH_QUESTIONS
    elif benchmark_name.lower() in ["arc", "reasoning"]:
        questions = SIMPLE_REASONING_QUESTIONS
    else:
        questions = SIMPLE_MATH_QUESTIONS + SIMPLE_REASONING_QUESTIONS
    
    if num_samples:
        questions = questions[:num_samples]
    
    return questions


async def load_benchmark_dataset_all(
    benchmark_name: str,
) -> List[Dict[str, Any]]:
    """
    Load all benchmark dataset items without randomization.
    
    Args:
        benchmark_name: Name of the benchmark (e.g., "mmlu", "gsm8k", "jailbreak")
    
    Returns:
        List of all dataset items with standardized format (no randomization)
    """
    # Handle jailbreak benchmark (loads from database)
    if benchmark_name.lower() == "jailbreak":
        # Load all jailbreak prompts (no limit, but no randomization in this function)
        return await load_jailbreak_benchmark_dataset(num_samples=None)
    
    # Call the main function with None for num_samples to get all items
    # But we need to modify it to skip randomization
    # We'll duplicate the logic but skip the random selection part
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datasets library not installed. Install with: pip install datasets")
        raise ImportError("datasets library required for benchmark loading")
    
    if benchmark_name.lower() not in BENCHMARK_DATASETS:
        raise ValueError(f"Unknown benchmark: {benchmark_name}. Available: {list(BENCHMARK_DATASETS.keys())}")
    
    config = BENCHMARK_DATASETS[benchmark_name.lower()]
    dataset_name = config["dataset"]
    split = config.get("split", "test")
    dataset_config = config.get("config")
    
    logger.info(f"Loading all benchmark dataset: {dataset_name} (split: {split}, config: {dataset_config})")
    
    try:
        # Load dataset - same logic as load_benchmark_dataset but without randomization
        try:
            if dataset_config:
                try:
                    dataset = load_dataset(dataset_name, dataset_config, split=split)
                except Exception:
                    dataset = load_dataset(dataset_name, name=dataset_config, split=split)
            else:
                dataset = load_dataset(dataset_name, split=split)
        except Exception as e:
            logger.warning(f"Failed to load {dataset_name} with split {split}, trying without split: {e}")
            try:
                if dataset_config:
                    try:
                        full_dataset = load_dataset(dataset_name, dataset_config)
                    except Exception:
                        full_dataset = load_dataset(dataset_name, name=dataset_config)
                else:
                    full_dataset = load_dataset(dataset_name)
                
                if split in full_dataset:
                    dataset = full_dataset[split]
                elif "test" in full_dataset:
                    dataset = full_dataset["test"]
                elif "validation" in full_dataset:
                    dataset = full_dataset["validation"]
                else:
                    available_split = list(full_dataset.keys())[0]
                    dataset = full_dataset[available_split]
                    logger.warning(f"Using split {available_split} instead of {split}")
            except Exception as e2:
                logger.error(f"Failed to load dataset {dataset_name}: {e2}")
                raise
        
        # NO randomization - use all samples
        print(f"[load_benchmark_dataset_all] Using all {len(dataset)} samples (no randomization)")
        
        # Standardize format (same as load_benchmark_dataset)
        standardized = []
        question_field = config["question_field"]
        answer_field = config.get("answer_field")
        choices_field = config.get("choices_field")
        
        for i, item in enumerate(dataset):
            standardized_item = {
                "question": item.get(question_field, ""),
                "answer": item.get(answer_field) if answer_field else None,
            }
            
            # Handle answer field conversion (class labels to indices/letters)
            if answer_field and standardized_item["answer"] is not None:
                answer = standardized_item["answer"]
                if isinstance(answer, (int, str)) and str(answer).isdigit():
                    answer_int = int(answer)
                    if 0 <= answer_int <= 25:  # A-Z
                        standardized_item["answer_letter"] = chr(65 + answer_int)
                        standardized_item["answer_index"] = answer_int
            
            # Add choices if available
            if choices_field:
                if isinstance(choices_field, str) and "," in choices_field:
                    fields = [f.strip() for f in choices_field.split(",")]
                    choices = [item.get(f) for f in fields if item.get(f)]
                    standardized_item["choices"] = choices
                else:
                    choices = item.get(choices_field)
                    if choices:
                        standardized_item["choices"] = choices if isinstance(choices, list) else [choices]
            
            # Add subject/category if available
            subject_field = config.get("subject_field")
            if subject_field and item.get(subject_field):
                standardized_item["subject"] = item.get(subject_field)
            
            # Add raw item for reference
            standardized_item["raw"] = item
            
            standardized.append(standardized_item)
        
        logger.info(f"Loaded {len(standardized)} samples from {benchmark_name} (all samples, no randomization)")
        return standardized
        
    except Exception as e:
        logger.error(f"Error loading dataset {dataset_name}: {e}")
        logger.warning(f"Falling back to simple questions for {benchmark_name}")
        return _get_fallback_questions(benchmark_name, None)


def parse_index_selection(index_string: str, max_index: int) -> List[int]:
    """
    Parse index selection string into a list of indices.
    
    Supports formats like:
    - "0,5,10" -> [0, 5, 10]
    - "0,5,10,15-20" -> [0, 5, 10, 15, 16, 17, 18, 19, 20]
    - "0-10" -> [0, 1, 2, ..., 10]
    
    Args:
        index_string: String with comma-separated indices and ranges
        max_index: Maximum valid index (exclusive)
    
    Returns:
        Sorted list of unique indices
    """
    if not index_string:
        return []
    
    indices = set()
    parts = [p.strip() for p in index_string.split(",")]
    
    for part in parts:
        if "-" in part:
            # Range: "15-20"
            try:
                start, end = part.split("-", 1)
                start_idx = int(start.strip())
                end_idx = int(end.strip())
                if start_idx < 0 or end_idx >= max_index:
                    raise ValueError(f"Index out of range: {part}")
                indices.update(range(start_idx, end_idx + 1))
            except ValueError as e:
                raise ValueError(f"Invalid range format '{part}': {e}")
        else:
            # Single index
            try:
                idx = int(part)
                if idx < 0 or idx >= max_index:
                    raise ValueError(f"Index out of range: {idx}")
                indices.add(idx)
            except ValueError as e:
                raise ValueError(f"Invalid index '{part}': {e}")
    
    return sorted(indices)


def filter_prompts_by_selection(
    prompts: List[Dict[str, Any]],
    indices: Optional[List[int]] = None,
    subject: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Filter prompts by indices or subject.
    
    Args:
        prompts: List of prompt dictionaries
        indices: Optional list of indices to select (from original list)
        subject: Optional subject to filter by
    
    Returns:
        Filtered list of prompts
    """
    filtered = prompts
    
    # Filter by indices first (indices refer to original list positions)
    if indices is not None:
        filtered = [prompts[i] for i in indices if 0 <= i < len(prompts)]
    
    # Filter by subject (applied after index filtering)
    if subject:
        filtered = [p for p in filtered if p.get("subject") == subject]
    
    return filtered
