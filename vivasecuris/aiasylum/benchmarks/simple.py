"""Simple benchmark implementations (without external dependencies)."""

from typing import Dict, List, Optional, Any
import re

from vivasecuris.aiasylum.benchmarks.base import Benchmark, BenchmarkResult


class SimpleBenchmark(Benchmark):
    """Simple benchmark that uses predefined questions (no external dataset loading)."""
    
    def __init__(self, name: str, questions: List[Dict[str, Any]], description: str = ""):
        super().__init__(name, description)
        self.questions = questions
    
    async def load_dataset(self, num_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load questions from predefined list."""
        if num_samples:
            return self.questions[:num_samples]
        return self.questions
    
    async def evaluate_response(
        self,
        question: str,
        response: str,
        ground_truth: Any,
    ) -> bool:
        """Simple evaluation: check if ground truth appears in response (case-insensitive)."""
        response_lower = response.lower()
        
        # Handle different ground truth types
        if isinstance(ground_truth, str):
            # Check if ground truth string appears in response
            ground_truth_lower = ground_truth.lower()
            return ground_truth_lower in response_lower
        
        elif isinstance(ground_truth, list):
            # Multiple choice: check if any correct answer appears
            for answer in ground_truth:
                if answer.lower() in response_lower:
                    return True
            return False
        
        elif isinstance(ground_truth, dict):
            # Structured answer
            if "answer" in ground_truth:
                return ground_truth["answer"].lower() in response_lower
            return False
        
        else:
            # Try string conversion
            return str(ground_truth).lower() in response_lower


def create_simple_benchmark(name: str, questions: List[Dict[str, Any]]) -> SimpleBenchmark:
    """Create a simple benchmark from a list of questions."""
    return SimpleBenchmark(name, questions)


# Example benchmarks (can be expanded)
SIMPLE_MATH_QUESTIONS = [
    {"question": "What is 2 + 2?", "answer": "4"},
    {"question": "What is 5 * 3?", "answer": "15"},
    {"question": "What is 10 / 2?", "answer": "5"},
    {"question": "What is the square root of 16?", "answer": "4"},
    {"question": "What is 2 to the power of 3?", "answer": "8"},
]

SIMPLE_REASONING_QUESTIONS = [
    {"question": "If all roses are flowers, and some flowers are red, can we conclude that some roses are red?", "answer": "yes"},
    {"question": "A train leaves Station A at 60 mph. Another train leaves Station B, 200 miles away, at 80 mph heading towards Station A. When will they meet?", "answer": "1.43 hours"},
]
