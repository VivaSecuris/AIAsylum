"""Resource checks must scale with actual model shapes and selected captures."""
from unittest.mock import patch

import pytest

from vivasecuris.aiasylum.api import interp_preflight as preflight


def facts(parameters=8, layers=32, hidden=4096):
    return dict(id="test", model_type="qwen3", parameters_b=parameters, num_layers=layers,
                hidden_size=hidden, num_attention_heads=32, num_key_value_heads=8,
                head_dim=128, intermediate_size=hidden * 4, vocab_size=151936,
                max_position_embeddings=32768, is_moe=False)


def hardware(free=90):
    return dict(available=True, devices=[dict(device="cuda:0", name="GPU", free_gb=free,
                                             total_gb=96, bf16=True)],
                default_device="cuda:0", host_available_gb=220)


def test_actual_size_and_dtype_change_fit():
    options = dict(model_a="model", mode="single", device="cuda", dtype="bfloat16")
    with patch.object(preflight, "hardware_info", return_value=hardware(24)), \
         patch.object(preflight, "model_info", return_value=facts()):
        small = preflight.check_request(options)
        assert small["ready"]
        assert small["device"] == "cuda:0"
        large_dtype = preflight.check_request({**options, "dtype": "float32"})
        assert not large_dtype["ready"]
        assert "GiB" in large_dtype["errors"][0]
    with patch.object(preflight, "hardware_info", return_value=hardware(24)), \
         patch.object(preflight, "model_info", return_value=facts(32, 64, 5120)):
        assert not preflight.check_request(options)["ready"]


def test_attention_increment_is_quadratic_and_grouped_qkv_is_counted():
    def delta(seq):
        base = preflight.estimate_model(facts(), {"max_len": seq})
        attn = preflight.estimate_model(facts(), {"max_len": seq, "enable_attention_capture": True})
        # subtract the linear attention-output contribution
        return attn["capture_gb"] - base["capture_gb"] - 32 * seq * 4096 * 4 / 2**30
    assert delta(512) == pytest.approx(4 * delta(256), abs=0.005)
    f = facts()
    gqa = preflight.estimate_model(f, {"enable_qkv_capture": True})
    mha = preflight.estimate_model({**f, "num_key_value_heads": 32}, {"enable_qkv_capture": True})
    assert mha["capture_gb"] > gqa["capture_gb"]


def test_progression_counts_all_retained_prompts():
    single = preflight.estimate_model(facts(), {"mode": "single"})
    many = preflight.estimate_model(facts(), {"mode": "progression", "prompts": ["x"] * 8})
    assert many["capture_gb"] == pytest.approx(single["capture_gb"] * 8, abs=.01)
    assert many["estimated_device_gb"] == single["estimated_device_gb"]


def test_unavailable_cuda_fails_before_any_model_lookup():
    with patch.object(preflight, "hardware_info", return_value={"available": True, "devices": [], "default_device": "cpu"}), \
         patch.object(preflight, "model_info") as lookup:
        result = preflight.check_request({"device": "cuda", "model_a": "m"})
    assert not result["ready"]
    assert "unavailable" in result["errors"][0]
    lookup.assert_not_called()


def test_cross_size_model_diff_explains_separate_runs():
    with patch.object(preflight, "hardware_info", return_value=hardware()), \
         patch.object(preflight, "model_info", side_effect=[facts(), facts(14, 40, 5120)]):
        result = preflight.check_request(dict(mode="model_diff", model_a="a", model_b="b"))
    assert not result["ready"]
    assert any("separate runs" in x for x in result["errors"])


def test_meta_model_inspection_uses_local_config_without_weights(tmp_path):
    transformers = pytest.importorskip("transformers")
    torch = pytest.importorskip("torch")
    cfg = transformers.Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                                   num_hidden_layers=2, num_attention_heads=4,
                                   num_key_value_heads=2, tie_word_embeddings=True)
    cfg.save_pretrained(tmp_path)
    found = preflight.model_info(str(tmp_path))
    with torch.device("meta"):
        model = transformers.Qwen2ForCausalLM(cfg)
    assert found["parameters_b"] * 1e9 == pytest.approx(sum(p.numel() for p in model.parameters()))
    assert found["num_layers"] == 2
    assert found["num_key_value_heads"] == 2


def test_fit_refuses_incomplete_local_checkpoint_before_config_inspection(tmp_path):
    (tmp_path / 'config.json').write_text('{"model_type":"qwen2"}')
    with patch.object(preflight, 'hardware_info', return_value=hardware()), \
         patch.object(preflight, 'model_info') as inspect:
        result = preflight.check_request(dict(mode='single', model_a=str(tmp_path)))
    assert not result['ready']
    assert any('unavailable on this server' in error for error in result['errors'])
    inspect.assert_not_called()


@pytest.mark.parametrize("extra", [{"device": "cuda:no"}, {"dtype": "int4"}, {"topk": 0},
                                   {"window": 0}, {"prompt_a": "  "}, {"dim_reduction": "bad"}])
def test_request_validation_rejects_bad_configuration(extra):
    from fastapi import HTTPException
    from vivasecuris.aiasylum.api.routes.interp import InterpRunRequest, _validate
    with pytest.raises(HTTPException):
        _validate(InterpRunRequest(**{ "mode": "single", "model_a": "m", "prompt_a": "hello", **extra}))


def test_preflight_endpoint_returns_inspection_without_scheduling():
    from fastapi.testclient import TestClient
    from vivasecuris.aiasylum.api.main import app
    with patch("vivasecuris.aiasylum.api.routes.interp.check_request", return_value={"ready": True, "models": []}), \
         patch("vivasecuris.aiasylum.api.routes.interp._run_interp_background") as run:
        response = TestClient(app).post("/api/v1/interp/preflight", json={"mode": "single", "model_a": "m", "prompt_a": "hi"})
    assert response.status_code == 200
    assert response.json()["ready"]
    run.assert_not_called()


@pytest.mark.parametrize("fail", [False, True])
def test_worker_releases_locals_before_terminal_cache_clear(tmp_path, monkeypatch, fail):
    """Success and chained failures release model refs before allocator cleanup."""
    import weakref
    import torch
    from vivasecuris.aiasylum.api.routes import interp
    from vivasecuris.aiasylum.models import transformers_local

    class ModelMarker:
        pass

    references, clears = [], []

    def model_operation():
        model = ModelMarker()
        model.cycle = model
        references.append(weakref.ref(model))
        if fail:
            raise ValueError("model operation failed")
        return {"dashboard_bytes": 42}

    def analysis(*args):
        try:
            return model_operation()
        except ValueError as exc:
            raise RuntimeError("analysis failed") from exc

    def clear_cache():
        assert references and references[0]() is None
        clears.append(True)

    monkeypatch.setattr(interp, "_execute_analysis", analysis)
    monkeypatch.setattr(transformers_local, "clear_cache", clear_cache)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    if fail:
        with pytest.raises(RuntimeError, match="analysis failed") as raised:
            interp._execute(1, {}, tmp_path)
        assert isinstance(raised.value.__cause__, ValueError)
    else:
        assert interp._execute(1, {}, tmp_path) == {"dashboard_bytes": 42}
    assert clears == [True]


def test_worker_reports_live_and_reserved_cuda_bytes_after_cleanup(tmp_path, monkeypatch):
    import torch
    from vivasecuris.aiasylum.api.routes import interp
    from vivasecuris.aiasylum.models import transformers_local

    cleared = []
    monkeypatch.setattr(interp, "_execute_analysis", lambda *args: {"dashboard_bytes": 42})
    monkeypatch.setattr(transformers_local, "clear_cache", lambda: cleared.append(True))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda index: 0)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda index: 2048)

    def free_memory(index):
        assert cleared == [True]
        return 80_000, 96_000

    monkeypatch.setattr(torch.cuda, "mem_get_info", free_memory)
    summary = interp._execute(1, {}, tmp_path)
    assert summary["dashboard_bytes"] == 42
    assert summary["cuda_memory_after_cleanup"] == [{
        "device": "cuda:0", "allocated_bytes": 0, "reserved_bytes": 2048,
        "driver_free_bytes": 80_000, "total_bytes": 96_000,
    }]


@pytest.mark.parametrize("workspace_hook", ["present", "missing", "raises"])
@pytest.mark.parametrize("fail", [False, True])
def test_worker_clears_blas_workspaces_before_allocator(tmp_path, monkeypatch, workspace_hook, fail):
    import torch
    from vivasecuris.aiasylum.api.routes import interp
    from vivasecuris.aiasylum.models import transformers_local

    calls = []

    def analysis(*args):
        if fail:
            raise ValueError("analysis failed")
        return {}

    def clear_workspaces():
        calls.append("workspaces")
        if workspace_hook == "raises":
            raise RuntimeError("unsupported workspace cleanup")

    monkeypatch.setattr(interp, "_execute_analysis", analysis)
    monkeypatch.setattr(transformers_local, "clear_cache", lambda: calls.append("allocator"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    if workspace_hook == "missing":
        monkeypatch.delattr(torch._C, "_cuda_clearCublasWorkspaces", raising=False)
    else:
        monkeypatch.setattr(torch._C, "_cuda_clearCublasWorkspaces", clear_workspaces, raising=False)
    if fail:
        with pytest.raises(ValueError, match="analysis failed"):
            interp._execute(1, {}, tmp_path)
    else:
        interp._execute(1, {}, tmp_path)
    assert calls == (["allocator"] if workspace_hook == "missing" else ["workspaces", "allocator"])


@pytest.mark.asyncio
async def test_cancel_retains_model_slot_until_thread_releases_memory(tmp_path, monkeypatch):
    import asyncio
    import threading
    from vivasecuris.aiasylum.api.routes import interp
    from vivasecuris.aiasylum.api import model_jobs
    from vivasecuris.aiasylum.database import InterpRun, get_session

    entered, release = threading.Event(), threading.Event()
    monkeypatch.setattr(model_jobs, "_slot", None)
    monkeypatch.setattr(interp, "RUNS_ROOT", tmp_path)
    def execute(*args):
        entered.set()
        assert release.wait(5), "test failed to release worker"
        return {}
    monkeypatch.setattr(interp, "_execute", execute)
    session = get_session()
    row = InterpRun(mode="single", status="pending", model_a="m", prompt_a="hi", meta_data={})
    session.add(row)
    session.commit()
    rid = row.id
    session.close()
    interp.interp_cancellation.clear(rid)
    task = asyncio.create_task(interp._run_interp_background(rid))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        await asyncio.sleep(.02)
        assert model_jobs.slot_status()["held_by"] == f"interp run {rid}"
        assert not task.done()
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert model_jobs.slot_status()["held_by"] is None
    session = get_session()
    try:
        row = session.get(InterpRun, rid)
        assert row.status == "failed"
        assert row.meta_data["cancelled"] is True
    finally:
        session.close()


@pytest.mark.asyncio
async def test_stopping_pending_run_records_cancelled_state():
    from vivasecuris.aiasylum.api.routes.interp import stop_interp_run, interp_cancellation
    from vivasecuris.aiasylum.database import InterpRun, get_session
    session = get_session()
    row = InterpRun(mode="single", status="pending", model_a="m", prompt_a="hi", meta_data={})
    session.add(row)
    session.commit()
    rid = row.id
    session.close()
    try:
        assert (await stop_interp_run(rid))["stopped"]
        session = get_session()
        row = session.get(InterpRun, rid)
        assert row.status == "failed" and row.meta_data["cancelled"]
    finally:
        session.close()
        interp_cancellation.clear(rid)

@pytest.mark.asyncio
async def test_api_queues_behind_external_gpu_lock(tmp_path, monkeypatch):
    import asyncio
    from vivasecuris.aiasylum.api import model_jobs
    monkeypatch.setenv('AIASYLUM_MODEL_LOCK', str(tmp_path / 'gpu.lock'))
    monkeypatch.setattr(model_jobs, '_slot', None)
    external = model_jobs.try_process_lock()
    assert external is not None
    entered = asyncio.Event()
    async def api_job():
        async with model_jobs.hold('test API'):
            entered.set()
    task = asyncio.create_task(api_job())
    try:
        await asyncio.sleep(.03)
        assert not entered.is_set()
        assert 'test API' in model_jobs.slot_status()['waiting']
    finally:
        external.close()
        await asyncio.wait_for(task, timeout=2)
    assert entered.is_set()
