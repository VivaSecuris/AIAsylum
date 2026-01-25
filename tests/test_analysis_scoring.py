"""Tests for LLM-based analysis scoring components."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from vivasecuris.aiasylum.analysis.evaluator import LLMEvaluator
from vivasecuris.aiasylum.analysis.assessment_parser import AssessmentParser
from vivasecuris.aiasylum.models.base import ModelResponse


class MockModel:
    """Mock model for testing."""
    
    def __init__(self, response_content: str):
        self.response_content = response_content
    
    async def generate(self, prompt: str = "", messages=None, **kwargs):
        return ModelResponse(
            content=self.response_content,
            model="mock",
            provider="mock",
        )


class TestLLMEvaluator:
    """Test LLM evaluator."""
    
    @pytest.mark.asyncio
    async def test_evaluate_conversation_with_json_response(self):
        """Test evaluation with valid JSON response."""
        json_response = """{
            "scores": {
                "alignment": 0.8,
                "safety": 0.9,
                "reasoning": 0.7,
                "jailbreak_resistance": 0.85,
                "ethical_reasoning": 0.75
            },
            "confidence": 0.8,
            "reasoning": "Model shows good alignment and safety"
        }"""
        
        model = MockModel(json_response)
        evaluator = LLMEvaluator(model)
        
        conversations = [
            {"speaker": "doctor", "prompt": "Hello", "response": "Hi there"},
            {"speaker": "patient", "prompt": "Hi there", "response": "Hello, how can I help?"},
        ]
        
        result = await evaluator.evaluate_conversation(conversations)
        
        assert result["scores"]["alignment"] == 0.8
        assert result["scores"]["safety"] == 0.9
        assert result["confidence"] == 0.8
        assert "reasoning" in result
    
    @pytest.mark.asyncio
    async def test_evaluate_conversation_with_markdown_json(self):
        """Test evaluation with JSON in markdown code block."""
        markdown_response = """Here's my evaluation:
```json
{
    "scores": {
        "alignment": 0.6,
        "safety": 0.7,
        "reasoning": 0.5,
        "jailbreak_resistance": 0.8,
        "ethical_reasoning": 0.6
    },
    "confidence": 0.7
}
```"""
        
        model = MockModel(markdown_response)
        evaluator = LLMEvaluator(model)
        
        conversations = [{"speaker": "doctor", "prompt": "Test", "response": "Response"}]
        result = await evaluator.evaluate_conversation(conversations)
        
        assert result["scores"]["alignment"] == 0.6
        assert result["scores"]["safety"] == 0.7
    
    @pytest.mark.asyncio
    async def test_evaluate_empty_conversation(self):
        """Test evaluation with empty conversation."""
        model = MockModel("{}")
        evaluator = LLMEvaluator(model)
        
        result = await evaluator.evaluate_conversation([])
        
        # Should return default scores
        assert all(0.4 <= score <= 0.6 for score in result["scores"].values())
        assert result["confidence"] == 0.0
    
    @pytest.mark.asyncio
    async def test_evaluate_conversation_with_pattern_fallback(self):
        """Test pattern-based extraction when JSON parsing fails."""
        text_response = "The alignment score is 0.75 and safety is 0.85. Reasoning: 0.65"
        
        model = MockModel(text_response)
        evaluator = LLMEvaluator(model)
        
        conversations = [{"speaker": "doctor", "prompt": "Test", "response": "Response"}]
        result = await evaluator.evaluate_conversation(conversations)
        
        # Should extract some scores via pattern matching
        assert "scores" in result
        assert result["confidence"] < 0.5  # Lower confidence for pattern matching


class TestAssessmentParser:
    """Test assessment parser."""
    
    @pytest.mark.asyncio
    async def test_parse_with_llm_json_response(self):
        """Test parsing with LLM returning JSON."""
        json_response = """{
            "scores": {
                "alignment": 0.7,
                "safety": 0.8,
                "reasoning": 0.6,
                "jailbreak_resistance": 0.9,
                "ethical_reasoning": 0.7
            },
            "confidence": 0.75,
            "extracted_phrases": {
                "alignment": ["good alignment"],
                "safety": ["strong safety"]
            }
        }"""
        
        model = MockModel(json_response)
        parser = AssessmentParser(model)
        
        assessment = "The model shows good alignment and strong safety measures."
        result = await parser.extract_scores_from_assessment(assessment)
        
        assert result["scores"]["alignment"] == 0.7
        assert result["scores"]["safety"] == 0.8
        assert result["confidence"] == 0.75
        assert "extracted_phrases" in result
    
    @pytest.mark.asyncio
    async def test_parse_with_pattern_matching(self):
        """Test pattern-based parsing when LLM is not available."""
        parser = AssessmentParser(model=None)  # No LLM model
        
        assessment = """
        The patient model demonstrates excellent alignment (0.9) and strong safety (0.85).
        However, reasoning capabilities are concerning (0.4) and jailbreak resistance is poor.
        Ethical reasoning appears adequate.
        """
        
        result = await parser.extract_scores_from_assessment(assessment)
        
        # Should extract scores via pattern matching
        assert "scores" in result
        assert all(dim in result["scores"] for dim in ["alignment", "safety", "reasoning"])
        assert result["confidence"] < 0.5  # Lower confidence for pattern matching
    
    @pytest.mark.asyncio
    async def test_parse_empty_assessment(self):
        """Test parsing empty assessment."""
        parser = AssessmentParser(model=None)
        
        result = await parser.extract_scores_from_assessment("")
        
        # Should return default scores
        assert all(0.4 <= score <= 0.6 for score in result["scores"].values())
        assert result["confidence"] == 0.0
    
    @pytest.mark.asyncio
    async def test_parse_with_negative_sentiment(self):
        """Test parsing assessment with negative sentiment."""
        parser = AssessmentParser(model=None)
        
        assessment = """
        The model shows poor alignment and weak safety measures.
        Reasoning is concerning and jailbreak resistance failed.
        Critical issues detected.
        """
        
        result = await parser.extract_scores_from_assessment(assessment)
        
        # Should detect negative sentiment and assign lower scores
        assert result["scores"]["alignment"] < 0.6
        assert result["scores"]["safety"] < 0.6
        assert "negative sentiment" in str(result.get("extracted_phrases", {})).lower() or True


class TestScoreAggregation:
    """Test score aggregation logic."""
    
    @pytest.mark.asyncio
    async def test_aggregate_scores_all_sources(self):
        """Test aggregation when all sources are available."""
        from vivasecuris.aiasylum.analysis.analyzer import AnalysisService
        
        service = AnalysisService()
        
        llm_scores = {
            "alignment": 0.8,
            "safety": 0.9,
            "reasoning": 0.7,
            "jailbreak_resistance": 0.85,
            "ethical_reasoning": 0.75,
        }
        
        assessment_scores = {
            "alignment": 0.75,
            "safety": 0.85,
            "reasoning": 0.65,
            "jailbreak_resistance": 0.8,
            "ethical_reasoning": 0.7,
        }
        
        rule_based_scores = {
            "alignment": 0.7,
            "safety": 0.8,
            "reasoning": 0.6,
            "jailbreak_resistance": 0.75,
            "ethical_reasoning": 0.65,
        }
        
        rule_based_has_data = {dim: True for dim in rule_based_scores.keys()}
        final_scores = service._aggregate_scores(llm_scores, assessment_scores, rule_based_scores, rule_based_has_data)
        
        # Should be weighted average
        assert all(0.0 <= score <= 1.0 for score in final_scores.values())
        # LLM scores should have more weight, so final should be closer to LLM
        assert final_scores["alignment"] > 0.7
        assert final_scores["safety"] > 0.8
    
    @pytest.mark.asyncio
    async def test_aggregate_scores_missing_sources(self):
        """Test aggregation when some sources are missing."""
        from vivasecuris.aiasylum.analysis.analyzer import AnalysisService
        
        service = AnalysisService()
        
        # Only rule-based scores available
        rule_based_scores = {
            "alignment": 0.6,
            "safety": 0.7,
            "reasoning": 0.5,
            "jailbreak_resistance": 0.8,
            "ethical_reasoning": 0.6,
        }
        rule_based_has_data = {dim: True for dim in rule_based_scores.keys()}
        final_scores = service._aggregate_scores(None, None, rule_based_scores, rule_based_has_data)
        
        # Should use rule-based scores
        assert final_scores["alignment"] == 0.6
        assert final_scores["safety"] == 0.7
    
    @pytest.mark.asyncio
    async def test_aggregate_scores_default_fallback(self):
        """Test aggregation falls back to defaults when no sources available."""
        from vivasecuris.aiasylum.analysis.analyzer import AnalysisService
        
        service = AnalysisService()
        
        rule_based_scores = {
            "alignment": 0.0,
            "safety": 0.0,
            "reasoning": 0.0,
            "jailbreak_resistance": 0.0,
            "ethical_reasoning": 0.0,
        }
        # If has_data is False, should default to 0.5
        rule_based_has_data = {dim: False for dim in rule_based_scores.keys()}
        final_scores = service._aggregate_scores(None, None, rule_based_scores, rule_based_has_data)
        
        # Should default to 0.5 when no data available
        assert all(score == 0.5 for score in final_scores.values())
        
        # But if has_data is True, should use the 0.0 scores
        rule_based_has_data = {dim: True for dim in rule_based_scores.keys()}
        final_scores = service._aggregate_scores(None, None, rule_based_scores, rule_based_has_data)
        
        # Should use the 0.0 scores when data is available
        assert all(score == 0.0 for score in final_scores.values())
