"""First-class Ollama model support."""

import json
from typing import Any, Dict, List, Optional, AsyncIterator

import httpx

from vivasecuris.aiasylum.models.base import BaseModel, ModelResponse
from config import settings


def _with_reasoning(metadata: Dict[str, Any], thinking: Any) -> Dict[str, Any]:
    """Attach a trace the server delivered separately (Ollama's ``thinking`` field).

    Thinking models on Ollama >= 0.9 return the trace beside the answer rather
    than inside it; it is kept under the same keys ``ModelResponse`` uses for an
    inline block, so every reader sees one shape.
    """
    if isinstance(thinking, str) and thinking.strip():
        metadata["reasoning"] = thinking.strip()
        metadata["reasoning_source"] = "provider"
    return metadata


class OllamaModel(BaseModel):
    """First-class Ollama model implementation with enhanced features."""
    
    def __init__(
        self,
        model_name: str,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ):
        super().__init__(model_name, "ollama", temperature=temperature, max_tokens=max_tokens, **kwargs)
        self.base_url = base_url or settings.ollama_base_url
        self._client = None
        self._timeout = float(getattr(settings, "ollama_request_timeout", 1800))

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self._timeout, connect=10.0),
            )
        return self._client
    
    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def _generation_options(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        options = {
            "temperature": kwargs.get("temperature", self.temperature),
            "num_predict": kwargs.get("max_tokens", self.max_tokens),
        }
        for key in ("seed", "top_p", "top_k", "repeat_penalty"):
            if key in kwargs:
                options[key] = kwargs[key]
        # The server's default can be much smaller than a model's supported
        # context. Honor an explicit budget without shortening the messages.
        num_ctx = kwargs.get("num_ctx", self.kwargs.get("num_ctx"))
        if num_ctx is not None:
            if isinstance(num_ctx, bool) or not isinstance(num_ctx, int) or num_ctx < 1:
                raise ValueError("num_ctx must be a positive whole number")
            options["num_ctx"] = num_ctx
        return options
    
    async def check_available(self) -> bool:
        """Check if model is available in Ollama."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                models = response.json().get("models", [])
                return any(model.get("name", "").startswith(self.model_name) for model in models)
        except Exception:
            return False
    
    async def pull_model(self) -> Dict[str, Any]:
        """Pull the model from Ollama if not available."""
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/pull",
                    json={"name": self.model_name},
                ) as response:
                    response.raise_for_status()
                    result = {"status": "pulling"}
                    async for line in response.aiter_lines():
                        if line:
                            try:
                                data = json.loads(line)
                                result.update(data)
                                if data.get("status") == "success":
                                    break
                            except json.JSONDecodeError:
                                pass
                    return result
        except Exception as e:
            return {"error": str(e)}
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> ModelResponse:
        """Generate a response from the Ollama model.

        Responses can differ from the Ollama app because: (1) we add system prompts
        and conversation context (doctor/patient instructions, history); (2) temperature
        (default 0.7) adds randomness—use temperature=0 or pass seed= for reproducibility.
        """
        try:
            client = await self._get_client()
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Ollama at {self.base_url}: {str(e)}")
        
        # Use chat API if messages are provided (same format as Ollama app / chat UI)
        if messages:
            # Preserve message order: system, user, assistant as sent (Ollama expects this array)
            chat_messages = []
            for msg in messages:
                role = (msg.get("role") or "user").strip().lower()
                content = msg.get("content", "") or ""
                if role not in ("system", "user", "assistant"):
                    role = "user"
                chat_messages.append({"role": role, "content": content})
            
            # Options: match Ollama app behavior when possible. temperature=0 gives reproducible outputs.
            options = self._generation_options(kwargs)
            
            request_data = {
                "model": self.model_name,
                "messages": chat_messages,
                "stream": False,
                "options": options,
            }
            # Only when asked: Ollama >= 0.9 rejects `think` on models that
            # cannot think, and older servers ignore it.
            think = kwargs.get("think", self.kwargs.get("think"))
            if think is not None:
                request_data["think"] = bool(think)
            request_system_prompts = [
                msg["content"] for msg in request_data["messages"] if msg["role"] == "system"
            ]
            
            try:
                response = await client.post("/api/chat", json=request_data, timeout=self._timeout)
                response.raise_for_status()
                data = response.json()
                message = data.get("message", {}) or {}
                
                return ModelResponse(
                    content=message.get("content", ""),
                    model=self.model_name,
                    provider="ollama",
                    finish_reason=data.get("done_reason") or ("stop" if data.get("done") else None),
                    usage={
                        "prompt_eval_count": data.get("prompt_eval_count"),
                        "eval_count": data.get("eval_count"),
                        "total_duration": data.get("total_duration"),
                    } if "prompt_eval_count" in data else None,
                    metadata=_with_reasoning({
                        "done": data.get("done"),
                        "request_system_prompts": request_system_prompts,
                        "request_system_prompts_source": "provider",
                        "request_num_ctx": options.get("num_ctx"),
                        "request_message_count": len(chat_messages),
                    }, message.get("thinking")),
                )
            except httpx.ConnectError as e:
                raise RuntimeError(f"Cannot connect to Ollama at {self.base_url}. Is Ollama running? Error: {str(e)}")
            except httpx.ReadTimeout:
                raise RuntimeError(f"Ollama request timed out ({int(self._timeout)}s). The model may be overloaded or the response took too long.")
            except httpx.HTTPStatusError as e:
                error_detail = "Unknown error"
                try:
                    error_data = e.response.json()
                    error_detail = error_data.get("error", str(e))
                except Exception:
                    error_detail = str(e)
                raise RuntimeError(f"Ollama API error: {error_detail}")
            except Exception as e:
                # Fallback to generate API if chat fails
                full_prompt = self._messages_to_prompt(messages)
                request_system_prompts = [
                    msg.get("content", "") for msg in messages if msg.get("role") == "system"
                ]
                if system_prompt:
                    full_prompt = f"{system_prompt}\n\n{full_prompt}"
                    request_system_prompts.insert(0, system_prompt)
        else:
            # Use generate API for simple prompts
            full_prompt = prompt
            request_system_prompts = []
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
                request_system_prompts.append(system_prompt)
        
        # Prepare generate request (same options as chat for consistency)
        options = self._generation_options(kwargs)
        request_data = {
            "model": self.model_name,
            "prompt": full_prompt,
            "stream": False,
            "options": options,
        }
        
        try:
            response = await client.post("/api/generate", json=request_data, timeout=self._timeout)
            response.raise_for_status()
            data = response.json()
            
            return ModelResponse(
                content=data.get("response", ""),
                model=self.model_name,
                provider="ollama",
                finish_reason=data.get("done_reason") or ("stop" if data.get("done") else None),
                usage={
                    "prompt_eval_count": data.get("prompt_eval_count"),
                    "eval_count": data.get("eval_count"),
                    "total_duration": data.get("total_duration"),
                } if "prompt_eval_count" in data else None,
                metadata=_with_reasoning(
                    {
                        "context": data.get("context"), "done": data.get("done"),
                        "request_system_prompts": request_system_prompts,
                        "request_system_prompts_source": "provider",
                        "request_num_ctx": options.get("num_ctx"),
                    }, data.get("thinking")
                ),
            )
        except httpx.ConnectError as e:
            raise RuntimeError(f"Cannot connect to Ollama at {self.base_url}. Is Ollama running? Error: {str(e)}")
        except httpx.ReadTimeout:
            raise RuntimeError(f"Ollama request timed out ({int(self._timeout)}s). The model may be overloaded or the response took too long.")
        except httpx.HTTPStatusError as e:
            error_detail = "Unknown error"
            try:
                error_data = e.response.json()
                error_detail = error_data.get("error", str(e))
            except Exception:
                error_detail = str(e)
            raise RuntimeError(f"Ollama API error: {error_detail}")
    
    async def stream_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream generate a response from the Ollama model."""
        # Build prompt from messages if provided
        if messages:
            full_prompt = self._messages_to_prompt(messages)
        else:
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
        
        request_data = {
            "model": self.model_name,
            "prompt": full_prompt,
            "stream": True,
            "options": self._generation_options(kwargs),
        }
        
        client = await self._get_client()
        async with client.stream("POST", "/api/generate", json=request_data) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if "response" in data:
                            yield data["response"]
                        if data.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert message list to prompt string for Ollama."""
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        return "\n\n".join(prompt_parts)
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List all available Ollama models."""
        try:
            client = await self._get_client()
            response = await client.get("/api/tags")
            response.raise_for_status()
            return response.json().get("models", [])
        except Exception as e:
            return [{"error": str(e)}]
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about this specific model."""
        try:
            client = await self._get_client()
            response = await client.post(
                "/api/show",
                json={"name": self.model_name},
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def __repr__(self) -> str:
        return f"OllamaModel(model={self.model_name}, base_url={self.base_url})"
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


class OllamaProvider:
    """Enhanced Ollama provider with first-class support."""
    
    def __init__(self, base_url: Optional[str] = None):
        self.provider_name = "ollama"
        self.base_url = base_url or settings.ollama_base_url
    
    def create_model(
        self,
        model_name: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> OllamaModel:
        """Create an Ollama model instance."""
        return OllamaModel(
            model_name=model_name,
            base_url=self.base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
    
    async def list_available_models(self, chat_only: bool = False) -> List[str]:
        """List installed Ollama models. Raises on connection error.

        ``chat_only`` drops models whose reported capabilities lack "completion"
        (embedding models such as bge-m3, which fail every chat request). A model
        whose capabilities cannot be read is kept.
        """
        base = self.base_url.rstrip('/')
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{base}/api/tags")
            response.raise_for_status()
            data = response.json()
            models = data.get("models", []) if isinstance(data, dict) else []
            names = [m.get("name", "") for m in models if isinstance(m, dict) and m.get("name")]
            if not chat_only:
                return names

            async def can_chat(name: str) -> bool:
                try:
                    show = await client.post(f"{base}/api/show", json={"model": name})
                    show.raise_for_status()
                    capabilities = show.json().get("capabilities")
                except Exception:
                    return True
                return not isinstance(capabilities, list) or "completion" in capabilities

            import asyncio
            keep = await asyncio.gather(*(can_chat(name) for name in names))
            return [name for name, ok in zip(names, keep) if ok]
    
    async def pull_model(self, model_name: str) -> Dict[str, Any]:
        """Pull a model from Ollama."""
        timeout = float(getattr(settings, "ollama_request_timeout", 1800))
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/pull",
                    json={"name": model_name},
                ) as response:
                    response.raise_for_status()
                    result = {"status": "pulling", "model": model_name}
                    async for line in response.aiter_lines():
                        if line:
                            try:
                                data = json.loads(line)
                                result.update(data)
                                if data.get("status") == "success":
                                    break
                            except json.JSONDecodeError:
                                pass
                    return result
        except Exception as e:
            return {"error": str(e), "model": model_name}
    
    def __repr__(self) -> str:
        return f"OllamaProvider(base_url={self.base_url})"
