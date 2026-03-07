"""Base model interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional


@dataclass
class ModelResponse:
    """Response from a model."""
    
    content: str
    model: str
    provider: str
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseModel(ABC):
    """Base interface for all model providers."""
    
    def __init__(
        self,
        model_name: str,
        provider: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ):
        self.model_name = model_name
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.kwargs = kwargs

    @property
    def name(self) -> str:
        """Alias for model_name so code using getattr(model, 'name', ...) gets the actual model."""
        return self.model_name

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> ModelResponse:
        """
        Generate a response from the model.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            messages: Optional conversation history
            **kwargs: Additional provider-specific parameters
        
        Returns:
            ModelResponse with the generated content
        """
        pass
    
    @abstractmethod
    async def stream_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> "AsyncIterator[str]":
        """
        Stream generate a response from the model.
        
        Yields:
            Chunks of the response as they are generated
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name}, provider={self.provider})"
    
    def __str__(self) -> str:
        return f"{self.model_name} ({self.provider})"