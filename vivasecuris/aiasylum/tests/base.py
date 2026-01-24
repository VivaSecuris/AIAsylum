"""Base test framework."""

from abc import ABC, abstractmethod
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
