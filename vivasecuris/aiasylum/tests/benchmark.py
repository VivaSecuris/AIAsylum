"""Benchmark test implementation."""

from typing import Any, Dict, List, Optional

from vivasecuris.aiasylum.patient import Patient
from vivasecuris.aiasylum.tests.base import TestCase, TestResult
from vivasecuris.aiasylum.benchmarks.datasets import (
    load_benchmark_dataset,
    load_benchmark_dataset_all,
    parse_index_selection,
    filter_prompts_by_selection,
)
from vivasecuris.aiasylum.benchmarks.base import BenchmarkResult


class BenchmarkTest(TestCase):
    """Benchmark test - evaluates model on standard benchmark datasets."""
    
    def __init__(
        self,
        name: str = "benchmark_test",
        benchmark_name: str = "mmlu",
        num_samples: Optional[int] = None,
        test_mode: str = "one_shot",  # "one_shot" or "multi_shot"
        selected_indices: Optional[List[int]] = None,  # Specific indices to use
        selected_subject: Optional[str] = None,  # Filter by subject
    ):
        """
        Initialize benchmark test.
        
        Args:
            name: Test name
            benchmark_name: Name of benchmark (mmlu, gsm8k, etc.)
            num_samples: Number of samples to test
            test_mode: "one_shot" (each question independently) or "multi_shot" (sequential)
            selected_indices: Specific indices to use (overrides num_samples and randomization)
            selected_subject: Filter by subject (e.g., "abstract_algebra" for MMLU)
        """
        super().__init__(name, category="benchmark")
        self.benchmark_name = benchmark_name
        self.num_samples = num_samples
        self.test_mode = test_mode
        self.selected_indices = selected_indices
        self.selected_subject = selected_subject
    
    async def run(
        self,
        patient_model,
        doctor_model=None,
        context: Optional[Dict] = None,
    ) -> TestResult:
        """Run benchmark test."""
        # Get system prompts from context if available
        patient_system_prompt = context.get("patient_system_prompt") if context else None
        enable_patient_cot = context.get("enable_patient_cot", False) if context else False
        
        patient = Patient(patient_model, system_prompt=patient_system_prompt, enable_cot=enable_patient_cot)
        
        # Load benchmark dataset
        try:
            # If specific indices or subject are provided, use manual selection
            if self.selected_indices is not None or self.selected_subject:
                # Load all prompts without randomization
                all_prompts = await load_benchmark_dataset_all(self.benchmark_name)
                # Filter by selection criteria
                dataset = filter_prompts_by_selection(
                    all_prompts,
                    indices=self.selected_indices,
                    subject=self.selected_subject,
                )
                print(f"[BenchmarkTest] Loaded {len(dataset)} manually selected samples from {len(all_prompts)} total")
            else:
                # Use standard loading with randomization
                dataset = await load_benchmark_dataset(self.benchmark_name, self.num_samples)
                print(f"[BenchmarkTest] Loaded {len(dataset)} samples for benchmark {self.benchmark_name}")
        except Exception as e:
            import traceback
            error_msg = f"Error loading benchmark dataset {self.benchmark_name}: {e}\n{traceback.format_exc()}"
            print(f"[BenchmarkTest] {error_msg}")
            raise ValueError(error_msg)
        
        if not dataset:
            raise ValueError(f"No dataset loaded for benchmark {self.benchmark_name}")
        
        results = []
        correct = 0
        conversation_history = []
        
        if self.test_mode == "one_shot":
            # One-shot: each question is independent
            for i, item in enumerate(dataset):
                question = item.get("question", "")
                # Handle both answer formats (direct answer or class label)
                ground_truth = item.get("answer")
                # Prefer answer_letter if available (for MMLU, HellaSwag)
                if item.get("answer_letter"):
                    ground_truth = item.get("answer_letter")
                elif item.get("answer_index") is not None:
                    # Convert index to letter for multiple choice
                    answer_index = item.get("answer_index")
                    if 0 <= answer_index <= 25:
                        ground_truth = chr(65 + answer_index)  # A, B, C, D, etc.
                choices = item.get("choices", [])
                
                if not question:
                    continue
                
                # Format question with choices if available
                formatted_question = self._format_question(question, choices)
                
                # Get model response
                response_obj = await patient.respond(formatted_question, context=context)
                response = response_obj.content if hasattr(response_obj, 'content') else str(response_obj)
                reasoning = response_obj.metadata.get("reasoning", "") if response_obj.metadata else ""
                
                # Evaluate response
                is_correct = self._evaluate_response(formatted_question, response, ground_truth, choices)
                if is_correct:
                    correct += 1
                
                results.append({
                    "question_number": i + 1,
                    "question": formatted_question,
                    "response": response,
                    "ground_truth": ground_truth,
                    "choices": choices,
                    "correct": is_correct,
                    "reasoning": reasoning,
                })
                
                conversation_history.append({
                    "speaker": "patient",
                    "prompt_number": i + 1,
                    "prompt": formatted_question,
                    "response": response,
                    "reasoning": reasoning,
                })
        
        else:  # multi_shot
            # Multi-shot: sequential questions in one conversation
            conversation_prompts = []
            for item in dataset:
                question = item.get("question", "")
                if question:
                    formatted_question = self._format_question(
                        question,
                        item.get("choices", [])
                    )
                    conversation_prompts.append({
                        "question": formatted_question,
                        "ground_truth": item.get("answer"),
                        "choices": item.get("choices", []),
                    })
            
            # Run as sequential prompts
            for i, prompt_data in enumerate(conversation_prompts):
                question = prompt_data["question"]
                # Handle answer format (may be letter, index, or direct answer)
                ground_truth = prompt_data["ground_truth"]
                choices = prompt_data["choices"]
                
                response_obj = await patient.respond(question, context=context)
                response = response_obj.content if hasattr(response_obj, 'content') else str(response_obj)
                reasoning = response_obj.metadata.get("reasoning", "") if response_obj.metadata else ""
                
                # Evaluate response
                is_correct = self._evaluate_response(question, response, ground_truth, choices)
                if is_correct:
                    correct += 1
                
                results.append({
                    "question_number": i + 1,
                    "question": question,
                    "response": response,
                    "ground_truth": ground_truth,
                    "choices": choices,
                    "correct": is_correct,
                    "reasoning": reasoning,
                })
                
                conversation_history.append({
                    "speaker": "patient",
                    "prompt_number": i + 1,
                    "prompt": question,
                    "response": response,
                    "reasoning": reasoning,
                })
        
        accuracy = correct / len(results) if results else 0.0
        
        # Format results text
        results_text = "\n\n".join([
            f"Q{r['question_number']}: {r['question'][:100]}...\n"
            f"A: {r['response'][:200]}...\n"
            f"Correct: {r['correct']} (Expected: {r['ground_truth']})"
            for r in results[:10]  # Show first 10 in summary
        ])
        if len(results) > 10:
            results_text += f"\n\n... and {len(results) - 10} more questions"
        
        return TestResult(
            test_name=f"benchmark_{self.benchmark_name}",
            test_category="benchmark",
            input_prompt=f"Benchmark: {self.benchmark_name} ({len(results)} questions, {self.test_mode} mode)",
            output_response=f"Accuracy: {accuracy:.2%} ({correct}/{len(results)})\n\n{results_text}",
            score=accuracy,
            scores={"accuracy": accuracy, "correct": correct, "total": len(results)},
            metadata={
                "benchmark_name": self.benchmark_name,
                "test_mode": self.test_mode,
                "num_samples": len(results),
                "correct": correct,
                "accuracy": accuracy,
                "results": results,
                "conversation_history": conversation_history,
            },
        )
    
    def _format_question(self, question: str, choices: Optional[List] = None) -> str:
        """Format question with choices if available."""
        if choices and len(choices) > 0:
            # Multiple choice format
            choices_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
            return f"{question}\n\nChoices:\n{choices_text}\n\nAnswer:"
        return question
    
    def _evaluate_response(
        self,
        question: str,
        response: str,
        ground_truth: Any,
        choices: Optional[List] = None,
    ) -> bool:
        """Evaluate if response matches ground truth."""
        response_lower = response.lower().strip()
        response_upper = response.upper()
        
        # Handle multiple choice (A, B, C, D or 0, 1, 2, 3)
        if choices and len(choices) > 0:
            # Check if response contains the correct choice letter/number
            if isinstance(ground_truth, (int, str)):
                # Try to find choice index
                choice_index = None
                correct_letter = None
                
                if isinstance(ground_truth, str):
                    # If ground_truth is a letter (A, B, C, D)
                    if ground_truth.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                        choice_index = ord(ground_truth.upper()) - ord("A")
                        correct_letter = ground_truth.upper()
                elif isinstance(ground_truth, int):
                    # Numeric index (0-3 for A-D)
                    choice_index = ground_truth
                    if 0 <= choice_index <= 25:
                        correct_letter = chr(65 + choice_index)  # A, B, C, D, etc.
                
                # Check if letter appears in response
                if correct_letter and correct_letter in response_upper:
                    return True
                
                # Check if correct answer text appears in response
                if choice_index is not None and 0 <= choice_index < len(choices):
                    correct_answer = choices[choice_index]
                    if correct_answer and correct_answer.lower() in response_lower:
                        return True
                    
                    # Also check if the index number appears
                    if str(choice_index) in response or f"choice {choice_index}" in response_lower:
                        return True
        
        # Direct answer matching
        if isinstance(ground_truth, str):
            ground_truth_lower = ground_truth.lower()
            # Check if ground truth appears in response
            if ground_truth_lower in response_lower:
                return True
            # Check for partial matches (first few words)
            ground_truth_words = ground_truth_lower.split()[:3]
            if all(word in response_lower for word in ground_truth_words if len(word) > 2):
                return True
        
        return False
