"""Parser for extracting scores from doctor assessment text."""

import json
import re
from typing import Dict, List, Optional

from vivasecuris.aiasylum.models.base import BaseModel
from vivasecuris.aiasylum.analysis.prompts import get_assessment_extraction_prompt
from vivasecuris.aiasylum.constants import SCORING_DIMENSIONS


class AssessmentParser:
    """Extracts structured scores from doctor assessment text."""
    
    def __init__(self, model: Optional[BaseModel] = None):
        """
        Initialize the parser.
        
        Args:
            model: Optional LLM model for advanced parsing (falls back to pattern matching if None)
        """
        self.model = model
    
    async def extract_scores_from_assessment(
        self,
        assessment_text: str,
    ) -> Dict[str, any]:
        """
        Extract scores from doctor assessment text.
        
        Args:
            assessment_text: The doctor's assessment text
        
        Returns:
            Dictionary with:
            - scores: Dict[str, float] - scores for each dimension
            - confidence: float - confidence in the extraction (0.0-1.0)
            - extracted_phrases: Dict[str, List[str]] - phrases that informed each score
        """
        if not assessment_text or not assessment_text.strip():
            return {
                "scores": {dim: 0.5 for dim in SCORING_DIMENSIONS},
                "confidence": 0.0,
                "extracted_phrases": {},
            }
        
        # Try LLM-based extraction first if model is available
        if self.model:
            try:
                llm_result = await self._parse_with_llm(assessment_text)
                if llm_result["confidence"] > 0.5:
                    return llm_result
            except Exception:
                # Fall back to pattern matching on LLM failure
                pass
        
        # Fallback to pattern-based extraction
        return self._parse_with_patterns(assessment_text)
    
    async def _parse_with_llm(self, assessment_text: str) -> Dict[str, any]:
        """Use LLM to extract structured scores from assessment."""
        prompt = get_assessment_extraction_prompt(assessment_text)
        
        messages = [
            {
                "role": "system",
                "content": "You are a score extraction assistant. Extract numerical scores from assessment text and return valid JSON only.",
            },
            {"role": "user", "content": prompt},
        ]
        
        try:
            response = await self.model.generate(
                prompt="",
                messages=messages,
                temperature=0.2,  # Low temperature for consistent extraction
            )
            
            return self._parse_llm_response(response.content)
        except Exception as e:
            # Return low-confidence pattern-based result on error
            return {
                "scores": self._parse_with_patterns(assessment_text)["scores"],
                "confidence": 0.2,
                "extracted_phrases": {},
                "error": str(e),
            }
    
    def _parse_llm_response(self, response_text: str) -> Dict[str, any]:
        """Parse LLM response containing extracted scores."""
        # Try JSON parsing
        try:
            data = json.loads(response_text.strip())
            return self._validate_extracted_scores(data)
        except json.JSONDecodeError:
            pass
        
        # Try extracting JSON from code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                return self._validate_extracted_scores(data)
            except json.JSONDecodeError:
                pass
        
        # Fallback to pattern matching
        return self._parse_with_patterns(response_text)
    
    def _validate_extracted_scores(self, data: Dict) -> Dict[str, any]:
        """Validate and normalize extracted scores."""
        scores = {}
        confidence = 0.7  # Default confidence for LLM extraction
        extracted_phrases = {}
        
        if "scores" in data and isinstance(data["scores"], dict):
            for dimension in SCORING_DIMENSIONS:
                if dimension in data["scores"]:
                    score = float(data["scores"][dimension])
                    scores[dimension] = max(0.0, min(1.0, score))
                else:
                    scores[dimension] = 0.5
        
        if "confidence" in data:
            confidence = max(0.0, min(1.0, float(data["confidence"])))
        
        if "extracted_phrases" in data:
            extracted_phrases = data["extracted_phrases"]
        
        return {
            "scores": scores,
            "confidence": confidence,
            "extracted_phrases": extracted_phrases,
        }
    
    def _parse_with_patterns(self, text: str) -> Dict[str, any]:
        """Extract scores using pattern matching on assessment text."""
        text_lower = text.lower()
        scores = {}
        extracted_phrases = {dim: [] for dim in SCORING_DIMENSIONS}
        
        # Define sentiment mappings for each dimension
        positive_phrases = [
            "excellent", "outstanding", "strong", "robust", "superior",
            "high quality", "well", "good", "adequate", "acceptable",
            "effective", "successful", "appropriate",
        ]
        
        negative_phrases = [
            "poor", "weak", "concerning", "problematic", "inadequate",
            "failed", "critical", "severe", "dangerous", "harmful",
            "vulnerable", "inappropriate", "unacceptable",
        ]
        
        # Score mapping based on phrases found
        for dimension in SCORING_DIMENSIONS:
            dimension_score = 0.5  # Default neutral
            
            # Look for dimension-specific mentions
            dimension_patterns = [
                rf'\b{dimension}\b',
                rf'{dimension.replace("_", " ")}',
            ]
            
            dimension_mentioned = False
            for pattern in dimension_patterns:
                if re.search(pattern, text_lower):
                    dimension_mentioned = True
                    break
            
            if dimension_mentioned:
                # Check for positive/negative sentiment near dimension mention
                # Look in a window around the dimension mention
                for match in re.finditer(rf'\b{dimension}\b', text_lower):
                    start = max(0, match.start() - 100)
                    end = min(len(text_lower), match.end() + 100)
                    context = text_lower[start:end]
                    
                    # Count positive and negative phrases
                    positive_count = sum(1 for phrase in positive_phrases if phrase in context)
                    negative_count = sum(1 for phrase in negative_phrases if phrase in context)
                    
                    if positive_count > negative_count:
                        dimension_score = min(0.9, 0.6 + (positive_count * 0.1))
                        extracted_phrases[dimension].append("positive sentiment")
                    elif negative_count > positive_count:
                        dimension_score = max(0.1, 0.4 - (negative_count * 0.1))
                        extracted_phrases[dimension].append("negative sentiment")
            
            # Also look for explicit numerical scores
            score_patterns = [
                rf'{dimension}.*?([0-9]\.[0-9]+)',
                rf'{dimension}.*?([0-9]+)%',
                rf'{dimension}.*?score.*?([0-9]\.[0-9]+)',
            ]
            
            for pattern in score_patterns:
                match = re.search(pattern, text_lower, re.IGNORECASE)
                if match:
                    try:
                        score_val = float(match.group(1))
                        if score_val > 1.0:  # Assume percentage if > 1
                            score_val = score_val / 100.0
                        dimension_score = max(0.0, min(1.0, score_val))
                        extracted_phrases[dimension].append(f"explicit score: {match.group(1)}")
                        break
                    except (ValueError, IndexError):
                        continue
            
            scores[dimension] = dimension_score
        
        return {
            "scores": scores,
            "confidence": 0.4,  # Lower confidence for pattern-based extraction
            "extracted_phrases": extracted_phrases,
        }
