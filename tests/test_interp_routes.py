"""The interpretability API surface.

The orchestrator is mocked throughout: these cover request validation, the
memory ceiling, dashboard serving and the isolation guarantees, not the
analysis itself (tests/test_interp_engine.py does that).
"""

from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("fastapi")


@pytest.fixture
def client(test_db, monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from vivasecuris.aiasylum.api.routes import interp

    monkeypatch.setattr(interp, "RUNS_ROOT", tmp_path / "interp")

    from vivasecuris.aiasylum.api.main import app

    return TestClient(app)


def test_modes_lists_all_four(client):
    body = client.get("/api/v1/interp/modes").json()
    names = {m["name"] for m in body["modes"]}
    assert names == {"single", "comparison", "progression", "model_diff"}


def test_modes_publishes_the_memory_ceiling(client):
    """The ceiling is a documented part of the contract, not a hidden default."""
    limits = client.get("/api/v1/interp/modes").json()["limits"]
    assert limits["max_len_default"] == 512
    assert limits["max_len_ceiling"] == 1024
    assert limits["max_len_default"] <= limits["max_len_ceiling"]


@pytest.mark.parametrize(
    "payload,missing",
    [
        ({"mode": "single", "model_a": "m"}, "prompt_a"),
        ({"mode": "comparison", "model_a": "m", "prompt_a": "a"}, "prompt_b"),
        ({"mode": "model_diff", "model_a": "m", "prompt_a": "a"}, "model_b"),
        ({"mode": "progression", "model_a": "m", "prompts": ["one"]}, "prompts"),
    ],
)
def test_missing_fields_are_rejected(client, payload, missing):
    r = client.post("/api/v1/interp/runs", json=payload)
    assert r.status_code == 400
    assert missing in r.json()["detail"]


def test_unknown_mode_is_rejected(client):
    r = client.post("/api/v1/interp/runs", json={"mode": "telepathy", "model_a": "m"})
    assert r.status_code == 400
    assert "Unknown mode" in r.json()["detail"]


def test_oversized_max_len_is_refused_with_a_reason(client):
    """The regression guard: 2048 on a 3B is >12 GB of activations."""
    r = client.post(
        "/api/v1/interp/runs",
        json={"mode": "single", "model_a": "m", "prompt_a": "hi", "max_len": 4096},
    )
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert "exceeds the ceiling" in detail
    assert "square of sequence length" in detail, "the reason must be explained"


def test_tiny_max_len_is_rejected(client):
    r = client.post(
        "/api/v1/interp/runs",
        json={"mode": "single", "model_a": "m", "prompt_a": "hi", "max_len": 2},
    )
    assert r.status_code == 400


def test_attention_capture_defaults_off(client):
    """Off by default because it is the term that scales as seq^2."""
    with patch("vivasecuris.aiasylum.api.routes.interp._run_interp_background"):
        r = client.post(
            "/api/v1/interp/runs",
            json={"mode": "single", "model_a": "m", "prompt_a": "hi"},
        )
    assert r.status_code == 200
    opts = r.json()["metadata"]["options"]
    assert opts["enable_attention_capture"] is False
    assert opts["enable_mlp_capture"] is False
    assert opts["max_len"] == 512


def test_run_lifecycle(client):
    with patch("vivasecuris.aiasylum.api.routes.interp._run_interp_background"):
        created = client.post(
            "/api/v1/interp/runs",
            json={"mode": "single", "model_a": "m", "prompt_a": "hi"},
        ).json()

    run_id = created["id"]
    assert created["status"] == "pending"

    fetched = client.get(f"/api/v1/interp/runs/{run_id}").json()
    assert fetched["id"] == run_id
    assert fetched["mode"] == "single"

    assert any(r["id"] == run_id for r in client.get("/api/v1/interp/runs").json())

    assert client.delete(f"/api/v1/interp/runs/{run_id}").status_code == 200
    assert client.get(f"/api/v1/interp/runs/{run_id}").status_code == 404


def test_missing_run_is_404(client):
    assert client.get("/api/v1/interp/runs/424242").status_code == 404
    assert client.delete("/api/v1/interp/runs/424242").status_code == 404
    assert client.get("/api/v1/interp/runs/424242/dashboard").status_code == 404


def test_dashboard_is_409_until_the_run_completes(client):
    with patch("vivasecuris.aiasylum.api.routes.interp._run_interp_background"):
        run_id = client.post(
            "/api/v1/interp/runs",
            json={"mode": "single", "model_a": "m", "prompt_a": "hi"},
        ).json()["id"]

    r = client.get(f"/api/v1/interp/runs/{run_id}/dashboard")
    assert r.status_code == 409
    assert "only once it completes" in r.json()["detail"]


def test_dashboard_is_served_as_html(client, tmp_path):
    from vivasecuris.aiasylum.database import InterpRun, get_session

    out = tmp_path / "interp" / "served"
    out.mkdir(parents=True)
    (out / "dashboard.html").write_text("<!DOCTYPE html><html><body>ok</body></html>")

    session = get_session()
    try:
        row = InterpRun(
            mode="single", status="completed", model_a="m",
            prompt_a="hi", out_dir=str(out),
        )
        session.add(row)
        session.commit()
        run_id = row.id
    finally:
        session.close()

    r = client.get(f"/api/v1/interp/runs/{run_id}/dashboard")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "<!DOCTYPE html>" in r.text


def test_stop_on_a_finished_run_is_a_no_op(client):
    from vivasecuris.aiasylum.database import InterpRun, get_session

    session = get_session()
    try:
        row = InterpRun(mode="single", status="completed", model_a="m", prompt_a="hi")
        session.add(row)
        session.commit()
        run_id = row.id
    finally:
        session.close()

    body = client.post(f"/api/v1/interp/runs/{run_id}/stop").json()
    assert body["stopped"] is False
    assert body["reason"] == "not running"


# --- isolation from test runs ---------------------------------------------

def test_interp_uses_its_own_managers():
    """Both managers key on bare ints; sharing them would cross the streams."""
    from vivasecuris.aiasylum.api.cancellation import cancellation_manager
    from vivasecuris.aiasylum.api.progress_events import progress_event_manager
    from vivasecuris.aiasylum.api.routes.interp import interp_cancellation, interp_progress

    assert interp_progress is not progress_event_manager
    assert interp_cancellation is not cancellation_manager


def test_interp_does_not_use_the_shared_worker_pool():
    """A model-loading job must not occupy one of the five test-run slots."""
    import asyncio

    from vivasecuris.aiasylum.api.routes import interp

    # The shared pool is never bound into this module's namespace.
    assert not hasattr(interp, "worker_pool")

    async def check():
        sem = interp._semaphore()
        assert isinstance(sem, asyncio.Semaphore)
        # One model-heavy job at a time.
        assert sem._value == 1
        # Lazily created, so it binds to the running loop rather than import time.
        assert interp._semaphore() is sem

    asyncio.run(check())


def test_config_is_built_directly_not_via_from_api_request():
    """from_api_request hardcodes max_len=2048 and forces every capture on."""
    import inspect

    from vivasecuris.aiasylum.api.routes import interp

    # Look for the call, not the word: the module explains why it avoids it.
    source = inspect.getsource(interp)
    assert "from_api_request(" not in source


def test_build_config_honours_the_requested_limits(tmp_path):
    from vivasecuris.aiasylum.api.routes.interp import _build_config
    from vivasecuris.aiasylum.database import InterpRun

    row = InterpRun(
        mode="single", status="pending", model_a="m", prompt_a="hi",
        meta_data={"options": {"max_len": 256, "enable_attention_capture": False}},
    )
    cfg = _build_config(row, tmp_path)

    assert cfg.max_len == 256, "the engine default of 2048 must not leak through"
    assert cfg.enable_attention_capture is False
    assert cfg.analysis_mode == "single"


def test_model_diff_maps_to_the_comparison_dashboard(tmp_path):
    """model_diff produces a ComparisonResult, so the config mode is comparison."""
    from vivasecuris.aiasylum.api.routes.interp import _build_config
    from vivasecuris.aiasylum.database import InterpRun

    row = InterpRun(
        mode="model_diff", status="pending", model_a="a", model_b="b",
        prompt_a="hi", meta_data={"options": {}},
    )
    assert _build_config(row, tmp_path).analysis_mode == "comparison"


def test_causal_analyses_are_advertised_with_their_claim(client):
    """Circuit cards are correlational; the minimal circuit is not. The UI
    cannot present them as equally trustworthy if it cannot tell them apart."""
    analyses = {a["name"]: a for a in client.get("/api/v1/interp/modes").json()["analyses"]}
    assert analyses["enable_minimal_circuit"]["claim"] == "descriptive"
    assert "approximation" in analyses["enable_minimal_circuit"]["description"]
    assert analyses["enable_scrub"]["claim"] == "causal"
    assert analyses["enable_attention_capture"]["claim"] == "descriptive"
    assert all(a["cost"] for a in analyses.values())


def test_causal_flags_default_off(client):
    """Every one of them multiplies capture cost or search time."""
    r = client.post("/api/v1/interp/runs", json={
        "mode": "single", "model_a": "test-model", "prompt_a": "hello"})
    assert r.status_code == 200
    opts = r.json()["metadata"]["options"]
    for flag in ("enable_qkv_capture", "enable_pre_mlp_capture", "enable_patching",
                 "enable_scrub", "enable_minimal_circuit"):
        assert opts[flag] is False, flag
    client.delete(f"/api/v1/interp/runs/{r.json()['id']}")


def test_causal_flags_reach_the_engine_config(tmp_path):
    """They were previously hardcoded off in _build_config, which is the whole
    reason the minimal circuit was unreachable from the app."""
    from vivasecuris.aiasylum.api.routes.interp import _build_config
    from vivasecuris.aiasylum.database import InterpRun

    row = InterpRun(
        mode="comparison", status="pending", model_a="m",
        prompt_a="a", prompt_b="b",
        meta_data={"options": {
            "enable_minimal_circuit": True, "enable_scrub": True,
            "enable_patching": True, "enable_qkv_capture": True,
            "enable_pre_mlp_capture": True,
        }},
    )
    config = _build_config(row, tmp_path)
    assert config.enable_minimal_circuit is True
    assert config.enable_scrub is True
    assert config.enable_patching is True
    assert config.enable_qkv_capture is True
    assert config.enable_pre_mlp_capture is True


def test_artifact_endpoint_refuses_anything_off_the_whitelist(client, tmp_path):
    """Containment alone would still expose whatever the engine left in out_dir."""
    from vivasecuris.aiasylum.database import InterpRun, get_session

    (tmp_path / "meta.json").write_text('{"spike_layer": 7}')
    (tmp_path / "secret.txt").write_text("not yours")

    session = get_session()
    try:
        row = InterpRun(mode="single", status="completed", model_a="m",
                        out_dir=str(tmp_path), meta_data={})
        session.add(row)
        session.commit()
        rid = row.id
    finally:
        session.close()

    try:
        assert client.get(f"/api/v1/interp/runs/{rid}/artifacts/meta.json").json() == {"spike_layer": 7}
        for bad in ("secret.txt", "cos_mat.npy", "../../../etc/passwd"):
            assert client.get(f"/api/v1/interp/runs/{rid}/artifacts/{bad}").status_code in (400, 404)

        listed = client.get(f"/api/v1/interp/runs/{rid}/artifacts").json()["artifacts"]
        assert [a["name"] for a in listed] == ["meta.json"]
    finally:
        session = get_session()
        try:
            r = session.query(InterpRun).filter(InterpRun.id == rid).first()
            if r:
                session.delete(r)
                session.commit()
        finally:
            session.close()


def test_missing_artifact_says_which_analysis_was_off(client, tmp_path):
    from vivasecuris.aiasylum.database import InterpRun, get_session

    session = get_session()
    try:
        row = InterpRun(mode="single", status="completed", model_a="m",
                        out_dir=str(tmp_path), meta_data={})
        session.add(row)
        session.commit()
        rid = row.id
    finally:
        session.close()

    try:
        r = client.get(f"/api/v1/interp/runs/{rid}/artifacts/minimal_circuit_payload.json")
        assert r.status_code == 404
        assert "was off" in r.json()["detail"]
    finally:
        session = get_session()
        try:
            row = session.query(InterpRun).filter(InterpRun.id == rid).first()
            if row:
                session.delete(row)
                session.commit()
        finally:
            session.close()
