"""Benchmark dataset loaders from HuggingFace and other sources."""

from typing import Dict, List, Optional, Any
import logging
import random
import asyncio
import hashlib
import json

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
        "dataset": "openai/gsm8k",
        "config": "main",
        "split": "test",
        "question_field": "question",
        "answer_field": "answer",
    },
    "winogrande": {
        "dataset": "winogrande",
        "config": "winogrande_debiased",
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
    # Medical / biomedical QA (Hugging Face)
    "medqa": {
        "dataset": "GBaker/MedQA-USMLE-4-options-hf",
        "split": "test",
        "question_field": "sent1",
        "answer_field": "label",
    },
    "medmcqa": {
        "dataset": "openlifescienceai/medmcqa",
        "split": "validation",
        "question_field": "question",
        "answer_field": "cop",
        "choices_field": "opa,opb,opc,opd",
        "subject_field": "subject_name",
    },
    "pubmedqa": {
        "dataset": "pubmed_qa",
        "config": "pqa_labeled",
        "split": "train",
        "question_field": "question",
        "answer_field": "final_decision",
    },
    "jailbreak": {
        "dataset": "internal",  # Special marker - loads from database
        "question_field": "prompt_text",
        "answer_field": "resisted",  # For jailbreaks, "resisted" means success (model didn't get jailbroken)
        "source": "database",  # Load from PromptLibrary database
    },
}


def standardize_benchmark_row(
    benchmark_key: str, config: Dict[str, Any], item: Dict[str, Any]
) -> Dict[str, Any]:
    """Map a raw dataset row to the common benchmark schema (question, answer, choices, ...)."""
    bk = benchmark_key.lower()

    if bk == "medqa":
        q1 = (item.get("sent1") or "").strip()
        q2 = (item.get("sent2") or "").strip()
        question = f"{q1}\n{q2}".strip() if q2 else q1
        choices = [item.get(f"ending{i}") for i in range(4)]
        answer = item.get("label")
        standardized_item: Dict[str, Any] = {
            "question": question,
            "answer": answer,
            "choices": choices,
            "raw": item,
        }
        if answer is not None and (isinstance(answer, int) or (isinstance(answer, str) and str(answer).isdigit())):
            answer_int = int(answer)
            if 0 <= answer_int <= 25:
                standardized_item["answer_letter"] = chr(65 + answer_int)
                standardized_item["answer_index"] = answer_int
        return standardized_item

    if bk == "pubmedqa":
        ctx = item.get("context")
        texts: List[str] = []
        if isinstance(ctx, dict):
            texts = list(ctx.get("contexts") or [])
        ctx_str = "\n\n".join(texts) if texts else ""
        q = item.get("question", "")
        question = (
            f"Context:\n{ctx_str}\n\nQuestion: {q}\n\n"
            "Answer with exactly one word: yes, no, or maybe."
        )
        fd = item.get("final_decision")
        answer = fd.lower().strip() if isinstance(fd, str) else fd
        return {"question": question, "answer": answer, "choices": [], "raw": item}

    question_field = config["question_field"]
    answer_field = config.get("answer_field")
    choices_field = config.get("choices_field")

    standardized_item = {
        "question": item.get(question_field, ""),
        "answer": item.get(answer_field) if answer_field else None,
    }

    labels = None
    if bk == "winogrande":
        standardized_item["choices"] = [item.get("option1"), item.get("option2")]
        labels = ["1", "2"]
    elif choices_field:
        if "," in choices_field:
            standardized_item["choices"] = [item.get(f.strip()) for f in choices_field.split(",")]
        else:
            choices = item.get(choices_field)
            if isinstance(choices, dict):
                standardized_item["choices"] = choices.get("text", [])
                labels = [str(label) for label in choices.get("label", [])]
            elif choices:
                standardized_item["choices"] = list(choices)
    choices = standardized_item.get("choices", [])
    if choices:
        answer = standardized_item["answer"]
        if labels and str(answer) in labels:
            index = labels.index(str(answer))
        elif isinstance(answer, int) or str(answer).isdigit():
            index = int(answer)
        elif isinstance(answer, str) and len(answer) == 1 and answer.upper().isalpha():
            index = ord(answer.upper()) - ord("A")
        else:
            raise ValueError(f"Invalid choice answer {answer!r} for {bk}")
        if not 0 <= index < len(choices):
            raise ValueError(f"Choice index {index} outside {len(choices)} choices for {bk}")
        standardized_item["answer_index"] = index
        standardized_item["answer_letter"] = chr(65 + index)

    subject_field = config.get("subject_field")
    if subject_field and item.get(subject_field):
        standardized_item["subject"] = item.get(subject_field)

    standardized_item["raw"] = item
    return standardized_item


class BenchmarkDataset(list):
    """List-compatible dataset with the exact selected source recorded."""

    def __init__(self, rows, provenance):
        super().__init__(rows)
        self.provenance = provenance


def selection_provenance(rows, *, seed=0, requested_count=None, source=None):
    """Hash the ordered scored content, not just potentially mutable row IDs."""
    content = [{key: row.get(key) for key in (
        "sample_id", "question", "answer", "answer_letter", "choices", "subject"
    )} for row in rows]
    encoded = json.dumps(content, sort_keys=True, ensure_ascii=False, default=str).encode()
    return {
        **(source or {}),
        "seed": seed,
        "requested_count": requested_count,
        "actual_count": len(rows),
        "sample_ids": [row.get("sample_id") for row in rows],
        "sample_indices": [row.get("dataset_index") for row in rows],
        "ordered_sample_hash": hashlib.sha256(encoded).hexdigest(),
    }


async def load_benchmark_dataset(
    benchmark_name: str,
    num_samples: Optional[int] = None,
    test_run_id: Optional[int] = None,
    *,
    seed: int = 0,
    revision: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load the requested split exactly; failures never substitute another test.

    Selection uses a local seeded RNG so every model sees the same ordered rows.
    ``test_run_id`` remains accepted for compatibility but cannot affect selection.
    Pass an immutable Hub commit as revision to reproduce across dataset updates.
    """
    benchmark_key = benchmark_name.lower()
    if benchmark_key not in BENCHMARK_DATASETS:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")
    if num_samples is not None and num_samples < 1:
        raise ValueError("num_samples must be positive")
    config = BENCHMARK_DATASETS[benchmark_key]
    dataset_name = config["dataset"]
    split = config.get("split", "test")
    source = {
        "dataset": dataset_name, "config": config.get("config"), "split": split,
        "requested_revision": revision or "main", "resolved_revision": revision if revision and len(revision) == 40 else None,
        "source": "database" if benchmark_key == "jailbreak" else "huggingface",
    }
    if benchmark_key == "jailbreak":
        rows = await load_jailbreak_benchmark_dataset(None)
        source.update(split=None, requested_revision=None, resolved_revision=None)
    else:
        try:
            from datasets import load_dataset
            kwargs = {"split": split}
            if revision:
                kwargs["revision"] = revision
            if config.get("config"):
                kwargs["name"] = config["config"]
            dataset = await asyncio.to_thread(load_dataset, dataset_name, **kwargs)
            source["dataset_fingerprint"] = getattr(dataset, "_fingerprint", None)
            rows = []
            for index, item in enumerate(dataset):
                row = standardize_benchmark_row(benchmark_key, config, item)
                row["dataset_index"] = index
                row["sample_id"] = f"{dataset_name}:{config.get('config') or 'default'}:{split}:{index}"
                rows.append(row)
        except Exception as exc:
            raise ValueError(
                f"Cannot load benchmark {benchmark_name} from {dataset_name} "
                f"(config={config.get('config')}, split={split}, revision={revision or 'main'}): {exc}. "
                "No substitute questions were used."
            ) from exc
    if not rows:
        raise ValueError(f"Benchmark {benchmark_name} has no samples in the requested split")
    source["available_count"] = len(rows)
    if num_samples is not None:
        indices = random.Random(seed).sample(range(len(rows)), min(num_samples, len(rows)))
        rows = [rows[i] for i in indices]
    for row in rows:
        if row.get("choices") and any(not isinstance(choice, str) or not choice.strip() for choice in row["choices"]):
            raise ValueError(f"Invalid choice text at {row.get('sample_id')}")
        if not isinstance(row.get("question"), str) or not row["question"].strip():
            raise ValueError(f"Invalid or empty question at {row.get('sample_id')}")
    return BenchmarkDataset(rows, selection_provenance(
        rows, seed=seed, requested_count=num_samples, source=source,
    ))


async def load_benchmark_dataset_all(
    benchmark_name: str, *, revision: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load the exact requested split in stable source order."""
    return await load_benchmark_dataset(benchmark_name, revision=revision)


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
                if start_idx < 0 or end_idx >= max_index or end_idx < start_idx:
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
        if any(i < 0 or i >= len(prompts) for i in indices):
            raise ValueError("Selected benchmark index is outside the dataset")
        filtered = [prompts[i] for i in indices]
    
    # Filter by subject (applied after index filtering)
    if subject:
        filtered = [p for p in filtered if p.get("subject") == subject]
    
    return filtered


async def load_jailbreak_benchmark_dataset(
    num_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Load jailbreak prompts from database for benchmark evaluation.
    
    Args:
        num_samples: Number of jailbreak prompts to load (None = all)
    
    Returns:
        List of standardized benchmark items with jailbreak prompts.
        For jailbreak benchmarks, "answer" is "resisted" (meaning the model
        should resist the jailbreak - this is the desired outcome).
    
    Raises:
        ValueError: If no jailbreak prompts are found in the database
    """
    # Import here to avoid circular import
    from vivasecuris.aiasylum.tests.jailbreak_loader import load_jailbreak_prompts, count_jailbreak_prompts
    
    logger.info(f"Loading jailbreak benchmark dataset (num_samples={num_samples})")
    
    # Check total count first for better error messages
    total_count = count_jailbreak_prompts()
    logger.info(f"Found {total_count} total jailbreak prompts in database")
    
    if total_count == 0:
        error_msg = (
            "No jailbreak prompts found in database. "
            "Please import jailbreak prompts using the import script: "
            "python scripts/import_jailbreaks.py"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Load jailbreak prompts from database with metadata
    # Use random=True to get diverse selection
    try:
        # We need to load prompts with their metadata to determine if they're multi-shot
        from vivasecuris.aiasylum.database import get_session, PromptLibrary
        
        session = get_session()
        try:
            # Get prompts with metadata
            all_prompts = session.query(PromptLibrary).filter(
                PromptLibrary.category == "adversarial"
            ).order_by(PromptLibrary.id).all()
            
            prompts = [p for p in all_prompts if p.tags and "jailbreak" in p.tags]
            
            # Apply limit
            if num_samples:
                prompts = prompts[:num_samples]
            
            # Convert to standardized format with metadata
            standardized = []
            for prompt in prompts:
                technique = prompt.meta_data.get("jailbreak_technique") if prompt.meta_data else None
                # Determine if this is a multi-shot technique based on constants
                from vivasecuris.aiasylum.constants import JAILBREAK_TECHNIQUES
                is_multi_shot = False
                if technique:
                    # Check if technique is in multi_shot list
                    is_multi_shot = technique in JAILBREAK_TECHNIQUES.get("multi_shot", [])
                
                standardized.append({
                    "sample_id": f"prompt-library:{prompt.id}",
                    "dataset_index": len(standardized),
                    "question": prompt.prompt_text,
                    "answer": "resisted",  # For jailbreaks, desired outcome is resistance
                    "is_multi_shot": is_multi_shot,  # Flag indicating if this prompt needs multi-shot
                    "technique": technique,
                    "metadata": prompt.meta_data or {},
                })
            
            logger.info(f"Loaded {len(standardized)} jailbreak prompts for benchmark")
            multi_shot_count = sum(1 for p in standardized if p.get("is_multi_shot", False))
            single_shot_count = len(standardized) - multi_shot_count
            logger.info(f"Jailbreak prompts breakdown: {single_shot_count} single-shot, {multi_shot_count} multi-shot")
            print(f"[load_jailbreak_benchmark_dataset] Loaded {len(standardized)} prompts: {single_shot_count} single-shot, {multi_shot_count} multi-shot")
            
            return standardized
        finally:
            session.close()
    except Exception as e:
        error_msg = f"Error loading jailbreak prompts from database: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise ValueError(error_msg) from e
