"""Base model interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional

# Serving defaults. Every provider samples at these unless a caller overrides
# them, and the weight-surgery evaluators read them from here so a checkpoint is
# gated under the same decoding the test harness will actually use.
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.95


@dataclass
class ModelResponse:
    """Response from a model.

    ``content`` is the *visible answer* -- the text a user would have seen --
    by construction: an inline reasoning block (``<think>...</think>``) is
    split off here, when the response is built, and kept in
    ``metadata["reasoning"]`` with ``metadata["reasoning_source"] = "inline"``.
    Providers that receive reasoning separately set the same two keys with
    source ``"provider"``. Every refusal detector, harm classifier, judge
    prompt and judge-output parser reads ``content``, so the invariant holds
    for all of them at once rather than one call site at a time. A response
    built with plain content is untouched (``metadata`` stays ``None``).
    """

    content: str
    model: str
    provider: str
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if not self.content or "<" not in self.content:
            return
        from vivasecuris.aiasylum.reasoning import split_reasoning

        answer, reasoning = split_reasoning(self.content)
        if reasoning is None:
            return
        self.content = answer
        meta = dict(self.metadata or {})
        if "reasoning" in meta:
            # A provider already delivered reasoning separately and a tag still
            # reached the text: keep the provider's, and keep the inline trace.
            meta["reasoning_inline"] = reasoning
        else:
            meta["reasoning"] = reasoning
            meta["reasoning_source"] = "inline"
        self.metadata = meta


class BaseModel(ABC):
    """Base interface for all model providers."""
    
    def __init__(
        self,
        model_name: str,
        provider: str,
        temperature: float = DEFAULT_TEMPERATURE,
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