"""Base test framework."""

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class TestResult:
    """Result of a test execution.
    
    Enhanced with support for:
    - Multiple safety category labels (Module 4: Edge Cases)
    - Edge case identification
    - Comprehensive metadata for red teaming analysis
    """
    
    test_name: str
    test_category: str
    input_prompt: str
    output_response: str
    score: Optional[float] = None
    scores: Optional[Dict[str, float]] = None
    analysis: Optional[str] = None
    flags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    # Enhanced fields for safety taxonomy (Module 4)
    safety_labels: Optional[List[str]] = None  # Multiple labels allowed
    edge_case: bool = False  # True if content doesn't fit any category
    edge_case_rationale: Optional[str] = None  # Explanation for edge case classification


def reasoning_fields(response) -> tuple:
    """``(reasoning, reasoning_source)`` from a model response, for the turn record.

    ``ModelResponse`` keeps a model's private trace in its metadata rather than
    in ``content`` (an inline ``<think>`` block, source ``"inline"``; a field the
    provider returned separately, ``"provider"``; a ReACT thought, ``"react"``).
    The source travels with the turn so the record says where a trace came from.
    """
    meta = getattr(response, "metadata", None) or {}
    return meta.get("reasoning", "") or "", meta.get("reasoning_source")


def response_turn_fields(response) -> Dict[str, Any]:
    """Snapshot observed response/request evidence, without inferring identity.

    Generation metadata describes the provider request and response. It does
    not measure whether or how strongly the model followed its instructions.
    """
    metadata = deepcopy(getattr(response, "metadata", None) or {})
    reasoning, reasoning_source = reasoning_fields(response)
    fields = {
        "model_name": getattr(response, "model", None),
        "model_provider": getattr(response, "provider", None),
        "reasoning": reasoning,
        "reasoning_source": reasoning_source,
        "finish_reason": getattr(response, "finish_reason", None),
        "usage": deepcopy(getattr(response, "usage", None)),
        "generation_metadata": metadata,
    }
    systems = metadata.get("request_system_prompts")
    if isinstance(systems, list) and all(isinstance(text, str) for text in systems):
        fields["request_system_prompts"] = list(systems)
        if isinstance(metadata.get("request_system_prompts_source"), str):
            fields["request_system_prompts_source"] = metadata["request_system_prompts_source"]
    for key in ("elapsed_seconds", "tokens_per_second"):
        value = getattr(response, key, metadata.get(key))
        if value is not None:
            fields[key] = value
    return fields


class TestCase(ABC):
    """Base class for all test cases."""
    
    def __init__(self, name: str, category: str = "general"):
        self.name = name
        self.category = category
    
    @abstractmethod
    async def run(
        self,
        patient_model: "BaseModel",
        doctor_model: Optional["BaseModel"] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> TestResult:
        """
        Run the test case.
        
        Args:
            patient_model: The patient model to test
            doctor_model: Optional doctor model for analysis
            context: Additional context
        
        Returns:
            TestResult with the test outcome
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, category={self.category})"
