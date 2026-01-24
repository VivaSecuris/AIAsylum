"""Test model providers."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from vivasecuris.aiasylum.models.base import BaseModel, ModelResponse
from vivasecuris.aiasylum.models.providers import (
    OpenAIProvider,
    AnthropicProvider,
    GoogleProvider,
    OllamaProvider,
    get_provider,
)


class TestModelProviders:
    """Test model provider initialization."""
    
    def test_get_provider_openai(self, mock_env):
        """Test getting OpenAI provider."""
        with patch('vivasecuris.aiasylum.models.providers.settings') as mock_settings, \
             patch('vivasecuris.aiasylum.models.providers.AsyncOpenAI'):
            mock_settings.openai_api_key = "test-key"
            provider = get_provider("openai")
            assert isinstance(provider, OpenAIProvider)
            assert provider.provider_name == "openai"
    
    def test_get_provider_anthropic(self, mock_env):
        """Test getting Anthropic provider."""
        with patch('vivasecuris.aiasylum.models.providers.settings') as mock_settings, \
             patch('vivasecuris.aiasylum.models.providers.AsyncAnthropic'):
            mock_settings.anthropic_api_key = "test-key"
            provider = get_provider("anthropic")
            assert isinstance(provider, AnthropicProvider)
            assert provider.provider_name == "anthropic"
    
    def test_get_provider_google(self, mock_env):
        """Test getting Google provider."""
        with patch('vivasecuris.aiasylum.models.providers.settings') as mock_settings, \
             patch('google.generativeai.configure'):
            mock_settings.google_api_key = "test-key"
            provider = get_provider("google")
            assert isinstance(provider, GoogleProvider)
            assert provider.provider_name == "google"
    
    def test_get_provider_ollama(self):
        """Test getting Ollama provider."""
        provider = get_provider("ollama")
        assert isinstance(provider, OllamaProvider)
        assert provider.provider_name == "ollama"
    
    def test_get_provider_invalid(self):
        """Test getting invalid provider raises error."""
        with pytest.raises(ValueError, match="Unknown provider"):
            get_provider("invalid_provider")


class TestModelGeneration:
    """Test model response generation."""
    
    @pytest.mark.asyncio
    async def test_openai_model_generate(self, mock_env):
        """Test OpenAI model generation."""
        with patch('vivasecuris.aiasylum.models.providers.settings') as mock_settings, \
             patch('vivasecuris.aiasylum.models.providers.AsyncOpenAI') as mock_openai:
            mock_settings.openai_api_key = "test-key"
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Test response"
            mock_response.choices[0].finish_reason = "stop"
            mock_response.usage = MagicMock()
            mock_response.usage.prompt_tokens = 10
            mock_response.usage.completion_tokens = 20
            mock_response.usage.total_tokens = 30
            
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai.return_value = mock_client
            
            provider = OpenAIProvider()
            model = provider.create_model("gpt-3.5-turbo")
            
            response = await model.generate("Test prompt")
            
            assert response.content == "Test response"
            assert response.model == "gpt-3.5-turbo"
            assert response.provider == "openai"
            assert response.usage is not None
    
    @pytest.mark.asyncio
    async def test_ollama_model_generate(self):
        """Test Ollama model generation."""
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "response": "Test response",
                "done": True
            }
            mock_response.raise_for_status = MagicMock()
            
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )
            
            provider = OllamaProvider()
            model = provider.create_model("llama2")
            
            response = await model.generate("Test prompt")
            
            assert response.content == "Test response"
            assert response.model == "llama2"
            assert response.provider == "ollama"
