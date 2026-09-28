"""Exercise the public chat route and the real ReACT request wrapper offline."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vivasecuris.aiasylum.api.routes.models import router
from vivasecuris.aiasylum.models.base import ModelResponse


class RecordingModel:
    temperature = 0.43
    max_tokens = 99
    supports_seed = True

    def __init__(self, name="chosen", provider="openai", **kwargs):
        self.model_name = name
        self.provider = provider
        self.calls = []
        self.close = AsyncMock()
        self.__dict__.update(kwargs)

    async def generate(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        wrapped = "Begin your reasoning:" in kwargs["messages"][-1]["content"]
        return ModelResponse(
            content="Thought: consider it\nFinal Answer: chosen answer" if wrapped else "plain answer",
            model=self.model_name, provider=self.provider, finish_reason="length",
            metadata={"request_system_prompts": [m["content"] for m in kwargs["messages"] if m["role"] == "system"],
                      "request_system_prompts_source": "provider"},
        )


@pytest.fixture
def chat(monkeypatch):
    from vivasecuris.aiasylum import models
    instances = []
    def provider(name):
        def create(model, **kwargs):
            instance = RecordingModel(model, name, **kwargs)
            instances.append(instance)
            return instance
        return SimpleNamespace(create_model=create)
    monkeypatch.setattr(models, "get_provider", provider)
    app = FastAPI()
    app.include_router(router, prefix="/models")
    return TestClient(app), instances


def test_chat_has_fresh_history_exact_system_and_selected_generation(chat):
    client, instances = chat
    prior = [{"role": "user", "content": "private earlier"}, {"role": "assistant", "content": "earlier reply"}]
    for system, history in [("  first persona\n", prior), ("second persona", [])]:
        response = client.post("/models/chat", json={
            "provider": "openai", "model": "same-model", "system_prompt": system,
            "messages": history + [{"role": "user", "content": "answer me"}],
            "temperature": 0, "top_p": 0.8, "seed": 0, "max_tokens": 17, "enable_cot": False,
        })
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["content"] == "plain answer" and body["reasoning_source"] is None
        assert body["metadata"]["request_system_prompts"] == [system]
        assert body["metadata"]["request_system_prompts_source"] == "provider"
        call = instances[-1].calls[0]
        assert call["messages"] == [{"role": "system", "content": system}] + history + [{"role": "user", "content": "answer me"}]
        assert {k: call[k] for k in ("temperature", "top_p", "seed", "max_tokens")} == {
            "temperature": 0, "top_p": 0.8, "seed": 0, "max_tokens": 17}
        instances[-1].close.assert_awaited_once()
    assert instances[0] is not instances[1]


def test_chat_cot_preserves_provider_defaults_persona_and_truncation(chat):
    client, instances = chat
    response = client.post("/models/chat", json={"provider": "openai", "model": "chosen",
        "system_prompt": "custom exact", "messages": [{"role": "user", "content": "question"}], "enable_cot": True})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["content"] == "chosen answer" and body["reasoning_source"] == "react"
    assert body["finish_reason"] == "length"
    call = instances[0].calls[0]
    assert call["messages"][0] == {"role": "system", "content": "custom exact"}
    assert call["messages"][-1]["content"].count("Begin your reasoning:") == 1
    assert call["temperature"] == 0.43 and call["max_tokens"] == 99
    assert body["generation"]["requested"]["temperature"] is None
    assert body["generation"]["applied"]["temperature"] == 0.43


def test_local_chat_releases_cache_and_transport_on_failure(monkeypatch, chat):
    from vivasecuris.aiasylum import models
    from vivasecuris.aiasylum.models import transformers_local
    from vivasecuris.aiasylum.api import model_jobs
    cleanup = []
    transport = SimpleNamespace(aclose=AsyncMock())
    def fail(*args, **kwargs):
        assert model_jobs.held_in_context()
        raise ValueError("bad model")
    monkeypatch.setattr(models, "get_provider", lambda name: SimpleNamespace(create_model=fail, client=transport))
    monkeypatch.setattr(transformers_local, "clear_cache", lambda: cleanup.append(True))
    response = chat[0].post("/models/chat", json={"provider": "transformers", "model": "missing",
        "messages": [{"role": "user", "content": "question"}]})
    assert response.status_code == 400
    transport.aclose.assert_awaited_once()
    assert cleanup == [True]


@pytest.mark.parametrize("payload", [
    {"messages": []}, {"messages": [{"role": "assistant", "content": "not a new turn"}]},
    {"messages": [{"role": "system", "content": "wrong location"}]}, {"provider": "missing"},
])
def test_chat_rejects_invalid_requests_before_provider_creation(chat, payload):
    client, instances = chat
    response = client.post("/models/chat", json={"provider": "openai", "model": "chosen",
        "messages": [{"role": "user", "content": "question"}], **payload})
    assert response.status_code in (400, 422)
    assert not instances


@pytest.mark.asyncio
async def test_chat_reports_unsupported_seed_and_provider_sampling():
    from vivasecuris.aiasylum.api.model_chat import generate_chat
    from vivasecuris.aiasylum.api.routes.models import ModelChatRequest
    model = RecordingModel(supports_seed=False)
    original = model.generate
    async def filtered(**kwargs):
        result = await original(**kwargs)
        result.metadata["sampling"] = {"temperature": None, "top_p": 0.8}
        return result
    model.generate = filtered
    request = ModelChatRequest(provider="anthropic", model="chosen", messages=[], seed=0, temperature=0, top_p=0.8)
    _, generation = await generate_chat(model, [{"role": "user", "content": "hi"}], request)
    assert "seed" not in model.calls[0]
    assert generation["applied"]["seed"] is None
    assert generation["applied"]["temperature"] is None
    assert len(generation["notes"]) == 2


@pytest.mark.asyncio
async def test_cancelled_local_chat_keeps_lease_until_generation_and_cleanup_finish(monkeypatch, tmp_path):
    import asyncio
    import threading
    from vivasecuris.aiasylum import models
    from vivasecuris.aiasylum.api import model_jobs
    from vivasecuris.aiasylum.api.routes.models import chat_with_model, ModelChatRequest
    from vivasecuris.aiasylum.models import transformers_local
    started, release = threading.Event(), threading.Event()
    cleanup = []
    monkeypatch.setattr(model_jobs, "_slot", None)
    monkeypatch.setenv("AIASYLUM_MODEL_LOCK", str(tmp_path / "model.lock"))
    class SlowModel(RecordingModel):
        async def generate(self, **kwargs):
            assert model_jobs.held_in_context()
            started.set()
            assert release.wait(5)
            return await super().generate(**kwargs)
    model = SlowModel(provider="transformers")
    monkeypatch.setattr(models, "get_provider", lambda name: SimpleNamespace(create_model=lambda *args, **kwargs: model))
    monkeypatch.setattr(transformers_local, "clear_cache", lambda: cleanup.append(model_jobs.held_in_context()))
    request = ModelChatRequest(provider="transformers", model="tiny", messages=[{"role": "user", "content": "hi"}])
    task = asyncio.create_task(chat_with_model(request))
    try:
        assert await asyncio.to_thread(started.wait, 3)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done() and model_jobs.slot_status()["held_by"] == "chat with tiny"
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert model_jobs.slot_status()["held_by"] is None
    assert cleanup == [True]
    model.close.assert_awaited_once()
