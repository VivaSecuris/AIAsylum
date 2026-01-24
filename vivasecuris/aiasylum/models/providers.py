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

from vivasecuris.aiasylum.models.base import BaseModel, ModelResponse
from config import settings


class ModelProvider:
    """Base class for model providers."""
    
    def __init__(self, provider_name: str):
        self.provider_name = provider_name
    
    def create_model(
        self,
        model_name: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> BaseModel:
        """Create a model instance."""
        raise NotImplementedError


class OpenAIProvider(ModelProvider):
    """OpenAI provider implementation."""
    
    def __init__(self):
        super().__init__("openai")
        api_key = settings.openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
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
        api_key = settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
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
        api_key = settings.google_api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set")
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


class OllamaProvider(ModelProvider):
    """Ollama provider implementation."""
    
    def __init__(self):
        super().__init__("ollama")
        self.base_url = settings.ollama_base_url
    
    def create_model(
        self,
        model_name: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> "OllamaModel":
        return OllamaModel(
            model_name=model_name,
            provider=self,
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
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> ModelResponse:
        if messages is None:
            messages = [{"role": "user", "content": prompt}]
        
        # Anthropic uses system parameter separately
        response = await self.client.messages.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system_prompt or "",
            messages=messages,
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
        
        async with self.client.messages.stream(
            model=self.model_name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system_prompt or "",
            messages=messages,
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
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> ModelResponse:
        model = self.client.GenerativeModel(self.model_name)
        
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        generation_config = {
            "temperature": self.temperature,
            "max_output_tokens": self.max_tokens,
        }
        
        response = await model.generate_content_async(
            full_prompt,
            generation_config=generation_config,
            **kwargs
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
        model = self.client.GenerativeModel(self.model_name)
        
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        generation_config = {
            "temperature": self.temperature,
            "max_output_tokens": self.max_tokens,
        }
        
        response = await model.generate_content_async(
            full_prompt,
            generation_config=generation_config,
            stream=True,
            **kwargs
        )
        
        async for chunk in response:
            if chunk.text:
                yield chunk.text


class OllamaModel(BaseModel):
    """Ollama model implementation."""
    
    def __init__(self, model_name: str, provider: OllamaProvider, **kwargs):
        super().__init__(model_name, "ollama", **kwargs)
        self.provider = provider
        self.base_url = provider.base_url
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> ModelResponse:
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens,
                    },
                },
            )
            response.raise_for_status()
            data = response.json()
        
        return ModelResponse(
            content=data.get("response", ""),
            model=self.model_name,
            provider="ollama",
            finish_reason="stop" if data.get("done") else None,
        )
    
    async def stream_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ):
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": full_prompt,
                    "stream": True,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens,
                    },
                },
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        import json
                        data = json.loads(line)
                        if "response" in data:
                            yield data["response"]


def get_provider(provider_name: str) -> ModelProvider:
    """Get a model provider by name."""
    providers = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "google": GoogleProvider,
        "ollama": OllamaProvider,
    }
    
    provider_class = providers.get(provider_name.lower())
    if not provider_class:
        raise ValueError(f"Unknown provider: {provider_name}")
    
    return provider_class()
