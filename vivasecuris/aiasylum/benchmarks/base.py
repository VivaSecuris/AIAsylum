"""Base benchmark framework."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    """Result of a benchmark evaluation."""
    
    benchmark_name: str
    model_name: str
    provider: str
    total_samples: int
    correct: int
    accuracy: float
    results: List[Dict[str, Any]]  # Individual question results
    metadata: Optional[Dict[str, Any]] = None


class Benchmark(ABC):
    """Base class for benchmark implementations."""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
    
    @abstractmethod
    async def load_dataset(self, num_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load benchmark dataset.
        
        Args:
            num_samples: Number of samples to load (None = all)
        
        Returns:
            List of dataset items, each with 'question' and 'answer' keys
        """
        pass
    
    @abstractmethod
    async def evaluate_response(
        self,
        question: str,
        response: str,
        ground_truth: Any,
    ) -> bool:
        """
        Evaluate if response matches ground truth.
        
        Args:
            question: The question asked
            response: Model's response
            ground_truth: Correct answer
        
        Returns:
            True if response is correct, False otherwise
        """
        pass
    
    async def run(
        self,
        model,
        num_samples: Optional[int] = None,
        *,
        enable_cot: bool = False,
        system_prompt: Optional[str] = None,
        generation: Optional[Dict[str, Any]] = None,
    ) -> BenchmarkResult:
        """
        Run benchmark on a model.
        
        Args:
            model: Model instance to test
            num_samples: Number of samples to run (None = all)
        
        Returns:
            BenchmarkResult with accuracy and detailed results
        """
        # Load dataset
        dataset = await self.load_dataset(num_samples)
        
        if not dataset:
            raise ValueError(f"No dataset loaded for benchmark {self.name}")
        
        results = []
        correct = 0
        
        # Run each question
        for i, item in enumerate(dataset):
            question = item.get("question", "")
            ground_truth = item.get("answer", item.get("correct_answer"))
            
            if not question:
                continue
            
            # Get model response
            options = dict(generation or {})
            if not getattr(model, "supports_seed", True):
                options.pop("seed", None)
            if enable_cot:
                from vivasecuris.aiasylum.cot import ReACTReasoner
                options.setdefault("temperature", getattr(model, "temperature", 0.7))
                response_obj = await ReACTReasoner(model).reason(
                    prompt=question, system_prompt=system_prompt, gen_overrides=options,
                )
            else:
                response_obj = await model.generate(prompt=question, **({"system_prompt": system_prompt} if system_prompt else {}), **options)
            response = response_obj.content if hasattr(response_obj, 'content') else str(response_obj)
            
            # Evaluate
            is_correct = await self.evaluate_response(question, response, ground_truth)
            if is_correct:
                correct += 1
            
            results.append({
                "question_number": i + 1,
                "question": question,
                "response": response,
                "ground_truth": ground_truth,
                "correct": is_correct,
                "reasoning": (getattr(response_obj, "metadata", None) or {}).get("reasoning"),
                "reasoning_source": (getattr(response_obj, "metadata", None) or {}).get("reasoning_source"),
            })
        
        accuracy = correct / len(results) if results else 0.0
        
        return BenchmarkResult(
            benchmark_name=self.name,
            model_name=getattr(model, 'name', 'unknown'),
            provider=getattr(model, 'provider', 'unknown'),
            total_samples=len(results),
            correct=correct,
            accuracy=accuracy,
            results=results,
            metadata={"enable_cot": enable_cot, "system_prompt": system_prompt, "generation": generation or {}},
        )
