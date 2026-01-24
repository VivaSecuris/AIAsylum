"""First-class Ollama model support."""

import json
from typing import Any, Dict, List, Optional, AsyncIterator

import httpx

from vivasecuris.aiasylum.models.base import BaseModel, ModelResponse
from config import settings


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
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(300.0, connect=10.0),
            )
        return self._client
    
    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
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
            async with httpx.AsyncClient(timeout=300.0) as client:
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
        """Generate a response from the Ollama model."""
        # Build prompt from messages if provided
        if messages:
            full_prompt = self._messages_to_prompt(messages)
        else:
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
        
        # Prepare request
        request_data = {
            "model": self.model_name,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.temperature),
                "num_predict": kwargs.get("max_tokens", self.max_tokens),
            },
        }
        
        # Add any additional options
        if "top_p" in kwargs:
            request_data["options"]["top_p"] = kwargs["top_p"]
        if "top_k" in kwargs:
            request_data["options"]["top_k"] = kwargs["top_k"]
        if "repeat_penalty" in kwargs:
            request_data["options"]["repeat_penalty"] = kwargs["repeat_penalty"]
        
        client = await self._get_client()
        response = await client.post("/api/generate", json=request_data)
        response.raise_for_status()
        data = response.json()
        
        return ModelResponse(
            content=data.get("response", ""),
            model=self.model_name,
            provider="ollama",
            finish_reason="stop" if data.get("done") else None,
            usage={
                "prompt_eval_count": data.get("prompt_eval_count"),
                "eval_count": data.get("eval_count"),
                "total_duration": data.get("total_duration"),
            } if "prompt_eval_count" in data else None,
            metadata={
                "context": data.get("context"),
                "done": data.get("done"),
            },
        )
    
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
            "options": {
                "temperature": kwargs.get("temperature", self.temperature),
                "num_predict": kwargs.get("max_tokens", self.max_tokens),
            },
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
    
    async def list_available_models(self) -> List[str]:
        """List all available Ollama models."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                models = response.json().get("models", [])
                return [model.get("name", "") for model in models if model.get("name")]
        except Exception:
            return []
    
    async def pull_model(self, model_name: str) -> Dict[str, Any]:
        """Pull a model from Ollama."""
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
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
