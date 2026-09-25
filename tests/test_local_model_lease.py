"""Generic local generation must share and retain the GPU lease."""

import asyncio
import threading

import pytest

from vivasecuris.aiasylum.api import model_jobs
from vivasecuris.aiasylum.models import transformers_local as provider


@pytest.fixture
def lease_state(monkeypatch, tmp_path):
    monkeypatch.setenv("AIASYLUM_MODEL_LOCK", str(tmp_path / "model.lock"))
    monkeypatch.setattr(model_jobs, "_slot", None)
    monkeypatch.setattr(model_jobs, "_held_by", None)
    monkeypatch.setattr(model_jobs, "_active_lease", None)
    monkeypatch.setattr(model_jobs, "_waiting", [])
    cleared = []
    monkeypatch.setattr(provider, "clear_cache", lambda: cleared.append(True))
    return cleared


async def wait_event(event):
    async with asyncio.timeout(3):
        while not event.is_set():
            await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_generic_generation_waits_for_existing_model_job(lease_state, monkeypatch):
    started = threading.Event()
    monkeypatch.setattr(provider.TransformersModel, "_generate_sync", lambda *a, **k: started.set() or "ok")
    model = provider.TransformersModel("test/model")
    entered = asyncio.Event()
    release = asyncio.Event()

    async def existing_job():
        async with model_jobs.hold("benchmark"):
            entered.set()
            await release.wait()

    first = asyncio.create_task(existing_job())
    await entered.wait()
    generation = asyncio.create_task(model.generate("q"))
    await asyncio.sleep(0.03)
    assert not started.is_set()
    release.set()
    await first
    assert await asyncio.wait_for(generation, 3) == "ok"
    assert len(lease_state) == 2


@pytest.mark.asyncio
async def test_nested_benchmark_generation_reuses_lease(lease_state, monkeypatch):
    monkeypatch.setattr(provider.TransformersModel, "_generate_sync", lambda *a, **k: "ok")
    model = provider.TransformersModel("test/model")
    async with model_jobs.hold("benchmark"):
        assert await asyncio.wait_for(model.generate("q"), 3) == "ok"
        assert model_jobs.held_in_context()
        assert lease_state == []
    assert not model_jobs.held_in_context()


@pytest.mark.asyncio
async def test_inherited_expired_lease_is_not_reused(lease_state, monkeypatch):
    ready = asyncio.Event()
    monkeypatch.setattr(provider.TransformersModel, "_generate_sync", lambda *a, **k: "ok")
    model = provider.TransformersModel("test/model")

    async def inherited():
        await ready.wait()
        assert not model_jobs.held_in_context()
        return await model.generate("q")

    async with model_jobs.hold("finished parent"):
        child = asyncio.create_task(inherited())
    ready.set()
    assert await child == "ok"
    assert len(lease_state) == 2


@pytest.mark.asyncio
async def test_repeated_cancel_waits_for_generation_thread(lease_state, monkeypatch):
    started = threading.Event()
    release = threading.Event()

    def generate(*args, **kwargs):
        started.set()
        assert release.wait(3)
        return "ok"

    monkeypatch.setattr(provider.TransformersModel, "_generate_sync", generate)
    task = asyncio.create_task(provider.TransformersModel("test/model").generate("q"))
    try:
        await wait_event(started)
        task.cancel()
        await asyncio.sleep(0.02)
        task.cancel()
        await asyncio.sleep(0.02)
        assert not task.done()
        assert model_jobs.model_slot().locked()
        assert len(lease_state) == 1
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert not model_jobs.model_slot().locked()
    assert len(lease_state) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("close_in_child_context", [False, True])
async def test_stream_close_joins_worker_before_unlock(lease_state, monkeypatch, close_in_child_context):
    release = threading.Event()

    def stream(self, prompt, system, messages, queue, **kwargs):
        queue.put("first")
        assert release.wait(3)
        queue.put(None)

    monkeypatch.setattr(provider.TransformersModel, "_stream_sync", stream)
    generator = provider.TransformersModel("test/model").stream_generate("q")
    assert await anext(generator) == "first"

    async def release_later():
        await asyncio.sleep(0.03)
        assert model_jobs.model_slot().locked()
        assert len(lease_state) == 1
        release.set()

    helper = asyncio.create_task(release_later())
    try:
        if close_in_child_context:
            await asyncio.create_task(generator.aclose())
        else:
            await generator.aclose()
    finally:
        release.set()
    await helper
    assert not model_jobs.model_slot().locked()
    assert not model_jobs.held_in_context()
    assert len(lease_state) == 2
