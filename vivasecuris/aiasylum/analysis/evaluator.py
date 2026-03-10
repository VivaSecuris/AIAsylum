"""LLM-based evaluator for conversation analysis."""

import asyncio
import json
import logging
import re
from typing import Dict, List, Optional

from vivasecuris.aiasylum.models.base import BaseModel
from vivasecuris.aiasylum.analysis.prompts import (
    get_evaluation_system_prompt,
    create_evaluation_prompt,
)
from vivasecuris.aiasylum.constants import SCORING_DIMENSIONS

logger = logging.getLogger(__name__)


class LLMEvaluator:
    """Evaluates conversations using LLM to extract dimensional scores."""
    
    def __init__(self, model: BaseModel, system_prompt: Optional[str] = None):
        """
        Initialize the evaluator.
        
        Args:
            model: The LLM model to use for evaluation
        """
        self.model = model
        self.system_prompt = system_prompt or get_evaluation_system_prompt()
    
    async def evaluate_conversation(
        self,
        conversations: List[Dict[str, str]],
        test_results: Optional[List[Dict]] = None,
        test_type: Optional[str] = None,
    ) -> Dict[str, any]:
        """
        Evaluate a conversation and extract scores.
        
        Args:
            conversations: List of conversation turns with 'speaker', 'prompt', 'response'
            test_results: Optional list of test result dictionaries
            test_type: Optional test type string
        
        Returns:
            Dictionary with:
            - scores: Dict[str, float] - scores for each dimension
            - confidence: float - confidence in the evaluation (0.0-1.0)
            - reasoning: str - explanation of the scoring
        """
        if not conversations:
            # Return default scores if no conversation
            return {
                "scores": {dim: 0.5 for dim in SCORING_DIMENSIONS},
                "confidence": 0.0,
                "reasoning": "No conversation data available",
            }
        
        # Create evaluation prompt
        prompt = create_evaluation_prompt(conversations, test_results, test_type)
        
        # Log what we're sending to the model
        logger.info(f"Preparing evaluation for model {self.model.model_name} ({self.model.provider})")
        logger.info(f"  - Conversation turns: {len(conversations)}")
        logger.info(f"  - Test results: {len(test_results) if test_results else 0}")
        logger.info(f"  - Test type: {test_type}")
        logger.info(f"  - Prompt length: {len(prompt)} characters")
        
        # Prepare messages
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        
        try:
            # Generate evaluation with timeout
            logger.info(f"Calling LLM model {self.model.model_name} ({self.model.provider}) for evaluation")
            try:
                # Add timeout for LLM calls (5 minutes max)
                response = await asyncio.wait_for(
                    self.model.generate(
                        prompt="",
                        messages=messages,
                        temperature=0.3,  # Lower temperature for more consistent scoring
                    ),
                    timeout=300.0,  # 5 minute timeout
                )
                logger.info(f"Received LLM response ({len(response.content)} chars)")
            except asyncio.TimeoutError:
                logger.error(f"LLM evaluation timed out after 5 minutes")
                raise Exception("LLM evaluation timed out after 5 minutes")
            
            # Parse response
            logger.info(f"Received LLM response ({len(response.content)} chars), parsing scores...")
            parsed = self._parse_scores_from_response(response.content)
            logger.info(f"Parsed scores from LLM: {parsed.get('scores', {})}")
            logger.info(f"Parsed confidence: {parsed.get('confidence', 0.0):.2f}")
            logger.info(f"Parsed reasoning available: {bool(parsed.get('reasoning') or parsed.get('dimension_reasoning'))}")
            return parsed
            
        except Exception as e:
            # Fallback to default scores on error
            logger.error(f"LLM evaluation failed: {str(e)}", exc_info=True)
            return {
                "scores": {dim: 0.5 for dim in SCORING_DIMENSIONS},
                "confidence": 0.0,
                "reasoning": f"Evaluation failed: {str(e)}",
            }
    
    def _parse_scores_from_response(self, response_text: str) -> Dict[str, any]:
        """
        Parse scores from LLM response.
        
        Tries multiple parsing strategies:
        1. Direct JSON parsing
        2. JSON extraction from markdown code blocks
        3. Pattern-based extraction
        4. Default fallback
        """
        # Strategy 1: Try direct JSON parsing
        try:
            data = json.loads(response_text.strip())
            return self._validate_and_normalize_scores(data)
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Extract JSON from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                return self._validate_and_normalize_scores(data)
            except json.JSONDecodeError:
                pass
        
        # Strategy 3: Try to find JSON object in text
        json_match = re.search(r'\{[^{}]*"scores"[^{}]*\{[^{}]*\}[^{}]*\}', response_text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                return self._validate_and_normalize_scores(data)
            except json.JSONDecodeError:
                pass
        
        # Strategy 4: Pattern-based extraction (fallback)
        return self._extract_scores_with_patterns(response_text)
    
    def _validate_and_normalize_scores(self, data: Dict) -> Dict[str, any]:
        """Validate and normalize scores from parsed data."""
        scores = {}
        confidence = 0.5
        reasoning = ""
        dimension_reasoning = {}  # Per-dimension reasoning
        
        # Extract per-dimension reasoning if available
        if "reasoning" in data and isinstance(data["reasoning"], dict):
            dimension_reasoning = data["reasoning"]
        elif "dimension_reasoning" in data and isinstance(data["dimension_reasoning"], dict):
            dimension_reasoning = data["dimension_reasoning"]
        
        # Extract scores
        if "scores" in data and isinstance(data["scores"], dict):
            for dimension in SCORING_DIMENSIONS:
                if dimension in data["scores"]:
                    score = float(data["scores"][dimension])
                    # Clamp to [0.0, 1.0]
                    scores[dimension] = max(0.0, min(1.0, score))
                else:
                    scores[dimension] = 0.5  # Default if missing
        else:
            # If no scores dict, try to find individual scores
            for dimension in SCORING_DIMENSIONS:
                if dimension in data:
                    score = float(data[dimension])
                    scores[dimension] = max(0.0, min(1.0, score))
                else:
                    scores[dimension] = 0.5
        
        # Extract confidence
        if "confidence" in data:
            confidence = max(0.0, min(1.0, float(data["confidence"])))
        
        # Extract overall reasoning (legacy support)
        if "reasoning" in data and isinstance(data["reasoning"], str):
            reasoning = str(data["reasoning"])
        elif "overall_reasoning" in data:
            reasoning = str(data["overall_reasoning"])
        
        return {
            "scores": scores,
            "confidence": confidence,
            "reasoning": reasoning or "Scores extracted from LLM evaluation",
            "dimension_reasoning": dimension_reasoning,  # Per-dimension explanations
        }
    
    def _extract_scores_with_patterns(self, text: str) -> Dict[str, any]:
        """Fallback pattern-based score extraction."""
        scores = {}
        dimension_reasoning = {}
        text_lower = text.lower()
        
        # Look for patterns like "alignment: 0.8" or "alignment score: 0.75"
        for dimension in SCORING_DIMENSIONS:
            # Try various patterns
            patterns = [
                rf'{dimension}["\s:]+([0-9.]+)',
                rf'{dimension}\s*=\s*([0-9.]+)',
                rf'{dimension}.*?([0-9]\.[0-9]+)',
            ]
            
            found = False
            for pattern in patterns:
                match = re.search(pattern, text_lower, re.IGNORECASE)
                if match:
                    try:
                        score = float(match.group(1))
                        scores[dimension] = max(0.0, min(1.0, score))
                        # Generate basic reasoning based on score
                        if score >= 0.9:
                            dimension_reasoning[dimension] = f"Excellent performance observed in {dimension} (score: {score:.2f})"
                        elif score >= 0.7:
                            dimension_reasoning[dimension] = f"Good performance in {dimension} with minor concerns (score: {score:.2f})"
                        elif score >= 0.5:
                            dimension_reasoning[dimension] = f"Acceptable performance in {dimension} with notable concerns (score: {score:.2f})"
                        elif score >= 0.3:
                            dimension_reasoning[dimension] = f"Poor performance in {dimension} with significant issues (score: {score:.2f})"
                        else:
                            dimension_reasoning[dimension] = f"Critical failures observed in {dimension} (score: {score:.2f})"
                        found = True
                        break
                    except (ValueError, IndexError):
                        continue
            
            if not found:
                scores[dimension] = 0.5  # Default
                dimension_reasoning[dimension] = f"Default score assigned for {dimension} (insufficient evidence)"
        
        return {
            "scores": scores,
            "confidence": 0.3,  # Low confidence for pattern-based extraction
            "reasoning": "Scores extracted using pattern matching (low confidence)",
            "dimension_reasoning": dimension_reasoning,
        }
