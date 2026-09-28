"""Model provider implementations."""

import os
import re
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
        # anthropic < 0.125 shipped only the legacy Completions resource; the
        # provider speaks the Messages API and would fail on first use with a
        # bare AttributeError. Say so at construction instead.
        if getattr(self.client, "messages", None) is None:
            installed = getattr(sys.modules.get("anthropic"), "__version__", "unknown")
            raise ModelProviderError(
                f"anthropic {installed} has no Messages API; install anthropic>=0.125.0 "
                "(pip install -r requirements.txt)"
            )
    
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


def _openai_reasoning(message: Any) -> Optional[Dict[str, Any]]:
    """Reasoning an OpenAI-compatible server delivered beside the answer.

    DeepSeek-style servers return ``reasoning_content``; some return
    ``reasoning``. openai 1.3.5's message type has neither attribute but keeps
    unknown fields, so read them by name and accept only a non-empty string --
    a mock message would otherwise hand back a truthy stand-in.
    """
    for key in ("reasoning_content", "reasoning"):
        value = getattr(message, key, None)
        if value is None:
            extra = getattr(message, "model_extra", None)
            value = extra.get(key) if isinstance(extra, dict) else None
        if isinstance(value, str) and value.strip():
            return {"reasoning": value.strip(), "reasoning_source": "provider"}
    return None


class OpenAIModel(BaseModel):
    """OpenAI model implementation."""
    
    def __init__(self, model_name: str, provider: OpenAIProvider, **kwargs):
        super().__init__(model_name, "openai", **kwargs)
        self.provider_instance = provider
        self.client = provider.client

    def _generation_options(self, kwargs):
        return {"temperature": self.temperature, "max_tokens": self.max_tokens, **kwargs}
    
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
            **self._generation_options(kwargs)
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
            metadata=_openai_reasoning(choice.message),
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
            stream=True,
            **self._generation_options(kwargs)
        )
        
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


# Which Claude generations still take sampling parameters, and how each asks
# for extended thinking. Both rules come from the API documentation bundled
# with the SDK, not from a live call, so what was actually sent is recorded on
# every response (``metadata["sampling"]`` / ``metadata["thinking"]``): a
# wrong row here shows up in the record rather than silently.
_CLAUDE_ID = re.compile(r"^claude-(?P<family>opus|sonnet|haiku)-(?P<major>\d+)(?:-(?P<minor>\d+))?", re.I)
_CLAUDE_LEGACY = re.compile(r"^claude-(?P<major>[12])(?:[.-](?P<minor>\d+))?", re.I)
_CLAUDE_3 = re.compile(r"^claude-3(?:-(?P<minor>\d+))?", re.I)
_CLAUDE_FRONTIER = re.compile(r"^claude-(fable|mythos)", re.I)
_MIN_THINKING_BUDGET = 1024


def _claude_generation(model_name: str):
    """``(family, (major, minor))`` for a Claude id, ``("frontier", None)`` for fable/mythos, or None."""
    name = (model_name or "").strip().lower()
    if _CLAUDE_FRONTIER.match(name):
        return "frontier", None
    m = _CLAUDE_ID.match(name)
    if m:
        return m.group("family"), (int(m.group("major")), int(m.group("minor") or 0))
    m = _CLAUDE_3.match(name)
    if m:
        return "claude3", (3, int(m.group("minor") or 0))
    m = _CLAUDE_LEGACY.match(name)
    if m:
        return "legacy", (int(m.group("major")), int(m.group("minor") or 0))
    return None


def _accepts_sampling_params(model_name: str) -> bool:
    """Whether ``temperature`` may be sent: Haiku of any version and Claude <= 4.6.

    Newer generations (4.7 and later, the 5.x line, fable) reject it. An id this
    table does not recognise is treated as new: omitting a setting is recorded,
    a 400 is not.
    """
    info = _claude_generation(model_name)
    if info is None:
        return False
    family, gen = info
    if family == "frontier":
        return False
    if family == "haiku":
        return True
    return gen <= (4, 6)


def _thinking_config(model_name: str, max_tokens: int) -> Dict[str, Any]:
    """The ``thinking`` request block for this model, when a caller asks for it."""
    info = _claude_generation(model_name)
    family, gen = info if info else ("unknown", None)
    if family == "frontier" or (gen is not None and family != "haiku" and gen >= (4, 7)):
        # Adaptive thinking; ``display`` is what makes the trace text come back
        # non-empty (the default omits it).
        return {"type": "adaptive", "display": "summarized"}
    if gen is not None and family != "haiku" and gen == (4, 6):
        return {"type": "adaptive"}
    budget = max(_MIN_THINKING_BUDGET, min(max_tokens // 2, max_tokens - 1))
    if max_tokens <= _MIN_THINKING_BUDGET:
        raise ValueError(
            f"Extended thinking needs max_tokens above {_MIN_THINKING_BUDGET} "
            f"(the minimum budget); got {max_tokens}."
        )
    return {"type": "enabled", "budget_tokens": budget}


class AnthropicModel(BaseModel):
    """Anthropic model implementation (Messages API, anthropic >= 0.125).

    A model that thinks returns its trace as ``thinking`` content blocks; they
    are kept in ``metadata["reasoning"]`` (source ``"provider"``) and never
    reach ``content``, so every scorer reads the answer alone. Thinking is
    requested only when a caller asks (``generate(..., thinking=True)`` or
    ``create_model(name, thinking=True)``). The streaming path yields text
    deltas only.
    """

    supports_seed = False

    def __init__(self, model_name: str, provider: AnthropicProvider, **kwargs):
        super().__init__(model_name, "anthropic", **kwargs)
        self.provider_instance = provider
        self.client = provider.client

    @property
    def supports_temperature(self) -> bool:
        return _accepts_sampling_params(self.model_name)

    def _generation_options(self, kwargs):
        """Request options and the sampling settings the API can actually accept.

        New generations reject all sampling controls. On older generations,
        native thinking accepts only top_p in [0.95, 1]. These are provider
        thinking restrictions; the harness's ReACT prompts do not enable it.
        See https://platform.claude.com/docs/en/build-with-claude/thinking#sampling-parameters.
        """
        defaults = {key: self.kwargs[key] for key in ("top_p", "top_k") if key in self.kwargs}
        options = {"temperature": self.temperature, "max_tokens": self.max_tokens, **defaults, **kwargs}
        if options.pop("seed", None) is not None:
            raise ValueError("Anthropic does not support a generation seed; leave Seed empty.")
        want_thinking = options.pop("thinking", self.kwargs.get("thinking", False))
        sent = {"temperature": None, "thinking": None}
        sampling = {key: options.pop(key) for key in ("temperature", "top_p", "top_k") if key in options}
        sent.update({key: None for key in sampling})
        if self.supports_temperature:
            if want_thinking:
                top_p = sampling.get("top_p")
                if top_p is not None and 0.95 <= float(top_p) <= 1:
                    options["top_p"] = top_p
            else:
                options.update({key: value for key, value in sampling.items() if value is not None})
                info = _claude_generation(self.model_name)
                if info and info[1] and info[1] >= (4, 1) and "top_p" in options:
                    # These generations accept temperature OR top_p. A selected
                    # top_p takes precedence over the default/role temperature.
                    options.pop("temperature", None)
        sent.update({key: options[key] for key in sampling if key in options})
        if want_thinking:
            options["thinking"] = _thinking_config(self.model_name, int(options["max_tokens"]))
            sent["thinking"] = options["thinking"]
        return options, sent

    def _request_kwargs(self, system_prompt, messages, kwargs):
        system_content, filtered_messages = self._extract_system_and_filter(system_prompt, messages)
        options, sent = self._generation_options(kwargs)
        request = {"model": self.model_name, "messages": filtered_messages, **options}
        if system_content:
            request["system"] = system_content
        return request, sent

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

        request, sent = self._request_kwargs(system_prompt, messages, kwargs)
        response = await self.client.messages.create(**request)

        text_parts: List[str] = []
        reasoning_parts: List[str] = []
        for block in response.content or []:
            kind = getattr(block, "type", None)
            if kind == "text":
                text_parts.append(block.text)
            elif kind == "thinking":
                # Empty under the default display setting on models that think
                # by default; only a real trace is recorded.
                trace = getattr(block, "thinking", None)
                if isinstance(trace, str) and trace.strip():
                    reasoning_parts.append(trace.strip())
            # redacted_thinking, tool_use and anything else carry no answer text.

        metadata: Dict[str, Any] = {
            "sampling": {key: value for key, value in sent.items() if key != "thinking"},
            "thinking": sent["thinking"],
        }
        if reasoning_parts:
            metadata["reasoning"] = "\n\n".join(reasoning_parts)
            metadata["reasoning_source"] = "provider"
        stop_details = getattr(response, "stop_details", None)
        if stop_details is not None:
            metadata["stop_details"] = {
                k: getattr(stop_details, k, None) for k in ("type", "category", "explanation")
            }

        usage = None
        if getattr(response, "usage", None):
            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
            for key in ("cache_read_input_tokens", "cache_creation_input_tokens"):
                value = getattr(response.usage, key, None)
                if isinstance(value, int):
                    usage[key] = value

        # A refusal is a result this framework studies, not a failure to route
        # around: it comes back as finish_reason == "refusal" with stop_details.
        return ModelResponse(
            content="".join(text_parts),
            model=self.model_name,
            provider="anthropic",
            finish_reason=getattr(response, "stop_reason", None),
            usage=usage,
            metadata=metadata,
        )

    async def stream_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ):
        """Yield answer text as it arrives. Thinking deltas are not part of the
        visible stream, consistent with ``content`` being the visible answer."""
        if messages is None:
            messages = [{"role": "user", "content": prompt}]

        request, _sent = self._request_kwargs(system_prompt, messages, kwargs)
        async with self.client.messages.stream(**request) as stream:
            async for text in stream.text_stream:
                yield text


class GoogleModel(BaseModel):
    """Google model implementation."""

    # This integration uses google-generativeai, whose supported configuration
    # fields do not include a seed. Never claim a requested seed was applied.
    supports_seed = False
    
    def __init__(self, model_name: str, provider: GoogleProvider, **kwargs):
        super().__init__(model_name, "google", **kwargs)
        self.provider_instance = provider
        self.client = provider.client

    def _generation_options(self, kwargs):
        options = dict(kwargs)
        if options.pop("seed", None) is not None:
            raise ValueError("This Google integration does not support a generation seed; leave Seed empty.")
        max_tokens = options.pop("max_tokens", self.max_tokens)
        config = {"temperature": self.temperature, "max_output_tokens": max_tokens, **options}
        allowed = {"temperature", "max_output_tokens", "top_p", "top_k", "candidate_count", "stop_sequences"}
        unsupported = set(config) - allowed
        if unsupported:
            raise ValueError(f"Unsupported Google generation options: {', '.join(sorted(unsupported))}")
        return config

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

        generation_config = self._generation_options(kwargs)

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

        candidates = getattr(response, "candidates", None)
        reason = getattr(candidates[0], "finish_reason", None) if candidates else getattr(response, "finish_reason", None)
        finish_reason = getattr(reason, "name", None) or (str(reason) if reason is not None else None)
        raw_usage = getattr(response, "usage_metadata", None)
        usage = dict(raw_usage) if isinstance(raw_usage, dict) else None
        if raw_usage is not None and usage is None:
            usage = {name: int(getattr(raw_usage, name)) for name in (
                "prompt_token_count", "candidates_token_count", "total_token_count"
            ) if getattr(raw_usage, name, None) is not None}
        return ModelResponse(
            content=response.text or "",
            model=self.model_name,
            provider="google",
            finish_reason=finish_reason,
            usage=usage,
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

        generation_config = self._generation_options(kwargs)

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
    # Imported lazily: this module pulls torch, which is an optional extra.
    from vivasecuris.aiasylum.models.transformers_local import TransformersProvider

    providers = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "google": GoogleProvider,
        "ollama": OllamaProvider,
        "servus": ServusProvider,
        "agentic_a2a": AgenticA2AProvider,
        "agentic": AgenticA2AProvider,
        "transformers": TransformersProvider,
        "local": TransformersProvider,
    }
    
    from vivasecuris.aiasylum.exceptions import ModelProviderError
    
    provider_class = providers.get(provider_name.lower())
    if not provider_class:
        raise ModelProviderError(f"Unknown provider: {provider_name}")
    
    return provider_class()
