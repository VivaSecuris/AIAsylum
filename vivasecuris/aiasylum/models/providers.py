"""Model provider implementations."""

import os
from typing import Any, Dict, List, Optional

import httpx
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
import google.generativeai as genai

import sys
from pathlib import Path

# Add project root to path for config import
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from abc import ABC, abstractmethod

from vivasecuris.aiasylum.models.base import BaseModel, ModelResponse
from config import settings


class ModelProvider(ABC):
    """Abstract base class for model providers."""
    
    def __init__(self, provider_name: str):
        self.provider_name = provider_name
    
    @abstractmethod
    def create_model(
        self,
        model_name: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> BaseModel:
        """
        Create a model instance.
        
        Args:
            model_name: Name of the model to create
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters
        
        Returns:
            BaseModel instance
        """
        pass


class OpenAIProvider(ModelProvider):
    """OpenAI provider implementation."""
    
    def __init__(self):
        super().__init__("openai")
        from vivasecuris.aiasylum.exceptions import ModelProviderError
        
        api_key = settings.openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ModelProviderError("OPENAI_API_KEY not set")
        self.client = AsyncOpenAI(api_key=api_key)
    
    def create_model(
        self,
        model_name: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> "OpenAIModel":
        return OpenAIModel(
            model_name=model_name,
            provider=self,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )


class AnthropicProvider(ModelProvider):
    """Anthropic provider implementation."""
    
    def __init__(self):
        super().__init__("anthropic")
        from vivasecuris.aiasylum.exceptions import ModelProviderError
        
        api_key = settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ModelProviderError("ANTHROPIC_API_KEY not set")
        self.client = AsyncAnthropic(api_key=api_key)
    
    def create_model(
        self,
        model_name: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> "AnthropicModel":
        return AnthropicModel(
            model_name=model_name,
            provider=self,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )


class GoogleProvider(ModelProvider):
    """Google provider implementation."""
    
    def __init__(self):
        super().__init__("google")
        from vivasecuris.aiasylum.exceptions import ModelProviderError
        
        api_key = settings.google_api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ModelProviderError("GOOGLE_API_KEY not set")
        genai.configure(api_key=api_key)
        self.client = genai
    
    def create_model(
        self,
        model_name: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> "GoogleModel":
        return GoogleModel(
            model_name=model_name,
            provider=self,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )


# Import first-class Ollama implementation
from vivasecuris.aiasylum.models.ollama import OllamaProvider as OllamaProviderBase, OllamaModel as OllamaModelBase


class OllamaProvider(ModelProvider):
    """Ollama provider implementation (wrapper for first-class implementation)."""
    
    def __init__(self):
        super().__init__("ollama")
        self._base_provider = OllamaProviderBase()
        self.base_url = self._base_provider.base_url
    
    def create_model(
        self,
        model_name: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> OllamaModelBase:
        return self._base_provider.create_model(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )


class OpenAIModel(BaseModel):
    """OpenAI model implementation."""
    
    def __init__(self, model_name: str, provider: OpenAIProvider, **kwargs):
        super().__init__(model_name, "openai", **kwargs)
        self.provider = provider
        self.client = provider.client
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> ModelResponse:
        if messages is None:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
        
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **kwargs
        )
        
        choice = response.choices[0]
        return ModelResponse(
            content=choice.message.content or "",
            model=self.model_name,
            provider="openai",
            finish_reason=choice.finish_reason,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            } if response.usage else None,
        )
    
    async def stream_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ):
        if messages is None:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
        
        stream = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
            **kwargs
        )
        
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class AnthropicModel(BaseModel):
    """Anthropic model implementation."""
    
    def __init__(self, model_name: str, provider: AnthropicProvider, **kwargs):
        super().__init__(model_name, "anthropic", **kwargs)
        self.provider = provider
        self.client = provider.client
    
    @staticmethod
    def _extract_system_and_filter(
        system_prompt: Optional[str],
        messages: List[Dict[str, str]],
    ):
        """Return (system_str, filtered_messages).

        Anthropic requires the system prompt via a dedicated ``system`` parameter
        and does not accept ``role: system`` entries inside the messages array.
        This helper collects all system content (from the explicit param and from
        any system-role messages) into one string and strips those entries out of
        the messages list so the request is valid for the Anthropic API.
        """
        parts = [system_prompt] if system_prompt else []
        filtered = []
        for msg in messages:
            if msg.get("role") == "system":
                if msg.get("content"):
                    parts.append(msg["content"])
            else:
                filtered.append(msg)
        return "\n\n".join(parts), filtered

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> ModelResponse:
        if messages is None:
            messages = [{"role": "user", "content": prompt}]

        system_content, filtered_messages = self._extract_system_and_filter(
            system_prompt, messages
        )

        response = await self.client.messages.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system_content or "",
            messages=filtered_messages,
            **kwargs
        )
        
        content = ""
        if response.content:
            for block in response.content:
                if hasattr(block, "text"):
                    content += block.text
        
        return ModelResponse(
            content=content,
            model=self.model_name,
            provider="anthropic",
            finish_reason=response.stop_reason,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            } if response.usage else None,
        )
    
    async def stream_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ):
        if messages is None:
            messages = [{"role": "user", "content": prompt}]

        system_content, filtered_messages = self._extract_system_and_filter(
            system_prompt, messages
        )

        async with self.client.messages.stream(
            model=self.model_name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system_content or "",
            messages=filtered_messages,
            **kwargs
        ) as stream:
            async for text in stream.text_stream:
                yield text


class GoogleModel(BaseModel):
    """Google model implementation."""
    
    def __init__(self, model_name: str, provider: GoogleProvider, **kwargs):
        super().__init__(model_name, "google", **kwargs)
        self.provider = provider
        self.client = provider.client

    @staticmethod
    def _prepare_google_args(
        system_prompt: Optional[str],
        messages: Optional[List[Dict[str, str]]],
        prompt: str,
    ):
        """Return (system_str, history, final_user_text).

        Extracts the system prompt (from the explicit param or system-role messages),
        builds the prior-turn history in Google's format, and returns the final user
        message text separately so it can be sent via ``send_message`` / passed
        directly to ``generate_content``.
        """
        system_parts = [system_prompt] if system_prompt else []
        chat_messages: List[Dict[str, str]] = []

        if messages:
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    if content:
                        system_parts.append(content)
                else:
                    chat_messages.append({"role": role, "content": content})

        system_str = "\n\n".join(system_parts)

        # Separate history from the final user message
        if chat_messages and chat_messages[-1]["role"] == "user":
            final_user_text = chat_messages[-1]["content"]
            history = chat_messages[:-1]
        else:
            # Fallback: no user message in the list; use prompt arg
            final_user_text = prompt
            history = chat_messages

        # Convert history to Google's Content format: [{"role": "user"|"model", "parts": [...]}]
        google_history = []
        for msg in history:
            google_role = "model" if msg["role"] == "assistant" else "user"
            google_history.append({"role": google_role, "parts": [msg["content"]]})

        return system_str, google_history, final_user_text

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> ModelResponse:
        system_str, history, final_user_text = self._prepare_google_args(
            system_prompt, messages, prompt
        )

        model_kwargs = {}
        if system_str:
            model_kwargs["system_instruction"] = system_str
        model = self.client.GenerativeModel(self.model_name, **model_kwargs)

        generation_config = {
            "temperature": self.temperature,
            "max_output_tokens": self.max_tokens,
        }

        if history:
            chat = model.start_chat(history=history)
            response = await chat.send_message_async(
                final_user_text,
                generation_config=generation_config,
            )
        else:
            response = await model.generate_content_async(
                final_user_text,
                generation_config=generation_config,
            )

        return ModelResponse(
            content=response.text or "",
            model=self.model_name,
            provider="google",
            finish_reason=getattr(response, "finish_reason", None),
            usage=getattr(response, "usage_metadata", None),
        )
    
    async def stream_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ):
        system_str, history, final_user_text = self._prepare_google_args(
            system_prompt, messages, prompt
        )

        model_kwargs = {}
        if system_str:
            model_kwargs["system_instruction"] = system_str
        model = self.client.GenerativeModel(self.model_name, **model_kwargs)

        generation_config = {
            "temperature": self.temperature,
            "max_output_tokens": self.max_tokens,
        }

        if history:
            chat = model.start_chat(history=history)
            response = await chat.send_message_async(
                final_user_text,
                generation_config=generation_config,
                stream=True,
            )
        else:
            response = await model.generate_content_async(
                final_user_text,
                generation_config=generation_config,
                stream=True,
            )

        async for chunk in response:
            if chunk.text:
                yield chunk.text


# OllamaModel is now imported from ollama.py (first-class implementation)


def get_provider(provider_name: str) -> ModelProvider:
    """Get a model provider by name."""
    from vivasecuris.aiasylum.models.vivaos import AgenticA2AProvider, ServusProvider

    providers = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "google": GoogleProvider,
        "ollama": OllamaProvider,
        "servus": ServusProvider,
        "agentic_a2a": AgenticA2AProvider,
        "agentic": AgenticA2AProvider,
    }
    
    from vivasecuris.aiasylum.exceptions import ModelProviderError
    
    provider_class = providers.get(provider_name.lower())
    if not provider_class:
        raise ModelProviderError(f"Unknown provider: {provider_name}")
    
    return provider_class()
