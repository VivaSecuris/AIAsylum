"""Local Hugging Face model provider.

Serves a model directly from weights on disk, in-process, so that a surgically
modified model can be driven by the existing test runner, benchmarks, suites
and analyzer with no changes to any of them. Going through Ollama instead would
require a GGUF conversion and quantization step, which perturbs exactly the
tensors an experiment is trying to measure.

Two details make this safe inside the async runner:

* ``model.generate`` is blocking and releases the GIL only intermittently, so
  every call runs in a worker thread.
* Models are cached during an enclosing model-job lease, such as a benchmark.
  Standalone calls acquire the same cross-process GPU slot and release cached
  weights before another job can use it.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import gc
import traceback
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from vivasecuris.aiasylum.exceptions import ModelProviderError
from vivasecuris.aiasylum.models.base import (
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    BaseModel,
    ModelResponse,
)
from vivasecuris.aiasylum.models.providers import ModelProvider

logger = logging.getLogger(__name__)

_CACHE: Dict[Tuple[str, str, str], Tuple[Any, Any]] = {}
_CACHE_LOCK = threading.Lock()


def _chat_messages(prompt, system_prompt, messages) -> List[Dict[str, str]]:
    """Copy the ordered messages used for formatting and request provenance."""
    chat: List[Dict[str, str]] = []
    if system_prompt:
        chat.append({"role": "system", "content": system_prompt})
    if messages:
        chat.extend(dict(message) for message in messages)
    if prompt:
        chat.append({"role": "user", "content": prompt})
    if not chat:
        raise ModelProviderError("No prompt or messages supplied")
    return chat


def generation_kwargs(
    temperature: Optional[float],
    max_new_tokens: int,
    pad_token_id: Optional[int],
    top_p: float = DEFAULT_TOP_P,
) -> Dict[str, Any]:
    """The ``model.generate`` keyword arguments this provider serves with.

    One builder for both the blocking and streaming paths, and for the
    weight-surgery evaluators: a checkpoint that is gated with these exact
    settings is gated under the decoding the test harness will use. Torch-free
    so it can be unit-tested and imported anywhere.

    ``temperature`` of 0 (or None) means greedy: HF rejects it as a sampling
    temperature, so it is expressed as ``do_sample=False`` instead.
    """
    gen: Dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "pad_token_id": pad_token_id,
    }
    if temperature and temperature > 0:
        gen.update(do_sample=True, temperature=float(temperature), top_p=float(top_p))
    else:
        gen.update(do_sample=False)
    return gen


def _get_cached(model_path: str, device: str, dtype: str) -> Tuple[Any, Any]:
    """Load a model once per (path, device, dtype) and share it across runs."""
    key = (model_path, device, dtype)
    with _CACHE_LOCK:
        if key in _CACHE:
            return _CACHE[key]

    try:
        from vivasecuris.aiasylum.interp.core.loader import load
    except ImportError as exc:
        raise ModelProviderError(
            'The "transformers" provider needs the optional interp extra: '
            'pip install -e ".[interp]"'
        ) from exc

    logger.info("Loading local model %s (%s, %s)", model_path, device, dtype)
    try:
        if os.environ.get("AIASYLUM_BENCHMARK_RUNTIME") == "1":
            from vivasecuris.aiasylum.models.benchmark_loader import load_benchmark_model
            model, tokenizer = load_benchmark_model(model_path, device=device, dtype=dtype)
        else:
            model, tokenizer = load(model_path, device=device, dtype=dtype, seed=None)
    except Exception as exc:
        raise ModelProviderError(f"Could not load local model '{model_path}': {exc}") from exc

    with _CACHE_LOCK:
        _CACHE.setdefault(key, (model, tokenizer))
        return _CACHE[key]


def clear_cache() -> None:
    """Drop every cached model and free accelerator memory."""
    global _CACHE
    with _CACHE_LOCK:
        count = len(_CACHE)
        _CACHE = {}
    gc.collect()
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            clear_workspaces = getattr(torch._C, "_cuda_clearCublasWorkspaces", None)
            if clear_workspaces is not None:
                clear_workspaces()
            torch.cuda.empty_cache()
    except Exception:
        pass
    logger.info("Cleared %d cached model(s)", count)


@asynccontextmanager
async def _generation_scope(owner: str):
    from vivasecuris.aiasylum.api import model_jobs
    if model_jobs.held_in_context():
        yield
        return
    async with model_jobs.hold(owner):
        clear_cache()
        try:
            yield
        finally:
            clear_cache()


async def _drain_worker(task):
    """Cancellation cannot stop CUDA; keep the lease until the worker exits."""
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
    return task.result()


def _clear_error_frames(exc):
    """A failed worker's traceback must not retain its model tensors."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        traceback.clear_frames(exc.__traceback__)
        exc.__traceback__ = None
        exc = exc.__cause__ or exc.__context__


class TransformersModel(BaseModel):
    """A causal LM loaded from local weights."""

    def __init__(
        self,
        model_name: str,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = 4096,
        device: str = "auto",
        dtype: str = "bfloat16",
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            provider="transformers",
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        self.device = device
        self.dtype = dtype
        self._manifest_meta = self._read_manifest(model_name)

    @staticmethod
    def _read_manifest(model_path: str) -> Dict[str, Any]:
        """Attach surgery provenance so every response is traceable to its edit."""
        try:
            from vivasecuris.aiasylum.weights.manifest import SurgeryManifest

            manifest = SurgeryManifest.load(model_path)
            return manifest.as_metadata() if manifest else {"surgery": None}
        except Exception:
            return {"surgery": None}

    def _build_prompt(self, tokenizer, prompt, system_prompt, messages) -> "tuple[str, bool]":
        """Render messages through the shared formatter: ``(text, template_applied)``.

        Delegates to ``weights.capture.format_chat`` -- the same code activation
        capture uses -- so a derived direction cannot refer to positions this
        path never produces. The caller must tokenize with
        ``add_special_tokens=not template_applied``.
        """
        from vivasecuris.aiasylum.weights.capture import format_chat

        return format_chat(tokenizer, _chat_messages(prompt, system_prompt, messages))

    def _generate_sync(self, prompt, system_prompt, messages, **kwargs) -> ModelResponse:
        import torch

        model, tokenizer = _get_cached(self.model_name, self.device, self.dtype)
        chat = _chat_messages(prompt, system_prompt, messages)
        request_system_prompts = [msg.get("content", "") for msg in chat if msg.get("role") == "system"]
        if os.environ.get("AIASYLUM_BENCHMARK_RUNTIME") == "1":
            from vivasecuris.aiasylum.models.benchmark_loader import prepare_benchmark_inputs
            inputs = prepare_benchmark_inputs(tokenizer, "", None, chat, model.device)
        else:
            from vivasecuris.aiasylum.weights.capture import format_chat

            text, applied = format_chat(tokenizer, chat)
            inputs = tokenizer(text, return_tensors="pt", add_special_tokens=not applied).to(model.device)
        prompt_tokens = int(inputs["input_ids"].shape[1])

        temperature = kwargs.get("temperature", self.temperature)
        max_new = int(kwargs.get("max_tokens", self.max_tokens))
        seed = kwargs.get("seed")
        if seed is not None:
            torch.manual_seed(int(seed))

        gen = generation_kwargs(
            temperature, max_new, tokenizer.pad_token_id or tokenizer.eos_token_id,
            top_p=kwargs.get("top_p", DEFAULT_TOP_P),
        )

        with torch.no_grad():
            out = model.generate(**inputs, **gen)

        completion_ids = out[0][prompt_tokens:]
        content = tokenizer.decode(completion_ids, skip_special_tokens=True)
        completion_tokens = int(completion_ids.shape[0])

        metadata = dict(self._manifest_meta)
        metadata.update(device=str(model.device), dtype=str(self.dtype),
                        model_revision=getattr(model.config, "_commit_hash", None),
                        request_system_prompts=request_system_prompts,
                        request_system_prompts_source="provider")

        return ModelResponse(
            content=content,
            model=self.model_name,
            provider="transformers",
            finish_reason="length" if completion_tokens >= max_new else "stop",
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            metadata=metadata,
        )

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs,
    ) -> ModelResponse:
        async with _generation_scope(f"local generation: {self.model_name}"):
            task = asyncio.create_task(asyncio.to_thread(
                self._generate_guarded_sync, prompt, system_prompt, messages, **kwargs
            ))
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                try:
                    await _drain_worker(task)
                except Exception:
                    pass
                raise

    def _generate_guarded_sync(self, prompt, system_prompt, messages, **kwargs):
        try:
            return self._generate_sync(prompt, system_prompt, messages, **kwargs)
        except Exception as exc:
            message = f"Local generation failed for '{self.model_name}': {exc}"
            _clear_error_frames(exc)
        # Raise outside the handler so the original exception cannot retain tensors.
        raise ModelProviderError(message)

    def _stream_sync(self, prompt, system_prompt, messages, queue, **kwargs) -> None:
        """Generate on a worker thread, pushing decoded text into ``queue``.

        ``None`` marks the end of the stream; an exception instance marks a
        failure, so the consumer can re-raise it on the event loop rather than
        hanging on a queue that will never fill.
        """
        import torch
        from transformers import TextIteratorStreamer

        try:
            model, tokenizer = _get_cached(self.model_name, self.device, self.dtype)
            text, applied = self._build_prompt(tokenizer, prompt, system_prompt, messages)
            inputs = tokenizer(text, return_tensors="pt", add_special_tokens=not applied).to(model.device)

            temperature = kwargs.get("temperature", self.temperature)
            seed = kwargs.get("seed")
            if seed is not None:
                torch.manual_seed(int(seed))

            gen = generation_kwargs(
                temperature, int(kwargs.get("max_tokens", self.max_tokens)),
                tokenizer.pad_token_id or tokenizer.eos_token_id,
                top_p=kwargs.get("top_p", DEFAULT_TOP_P),
            )

            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            # generate() blocks until the whole completion is done, so it runs on
            # its own thread and this one drains the streamer as tokens land.
            errors = []

            def generate_tokens():
                try:
                    with torch.no_grad():
                        model.generate(**inputs, streamer=streamer, **gen)
                except Exception as exc:
                    errors.append(str(exc))
                    _clear_error_frames(exc)
                    streamer.end()

            worker = threading.Thread(target=generate_tokens, daemon=True)
            worker.start()
            try:
                for piece in streamer:
                    if piece:
                        queue.put(piece)
            finally:
                worker.join()
            if errors:
                queue.put(ModelProviderError(errors[0]))
        except Exception as exc:  # surfaced on the consumer side
            queue.put(ModelProviderError(str(exc)))
            _clear_error_frames(exc)
        finally:
            queue.put(None)

    async def stream_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Yield the completion as the model produces it, token by token."""
        import queue as _queue

        async with _generation_scope(f"local stream: {self.model_name}"):
            q: "_queue.Queue" = _queue.Queue()
            loop = asyncio.get_running_loop()
            task = asyncio.create_task(asyncio.to_thread(
                self._stream_sync, prompt, system_prompt, messages, q, **kwargs
            ))
            try:
                while True:
                    piece = await loop.run_in_executor(None, q.get)
                    if piece is None:
                        break
                    if isinstance(piece, BaseException):
                        raise ModelProviderError(
                            f"Local generation failed for '{self.model_name}': {piece}"
                        ) from piece
                    yield piece
            finally:
                await _drain_worker(task)


class TransformersProvider(ModelProvider):
    """Factory for locally-loaded Hugging Face models."""

    def __init__(self):
        super().__init__("transformers")

    def create_model(
        self,
        model_name: str,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = 4096,
        **kwargs,
    ) -> BaseModel:
        return TransformersModel(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
