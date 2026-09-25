"""Route-level checks for the weight-surgery API.

Nothing here loads a model or writes a model. `_execute` is monkeypatched and
both roots are redirected into tmp_path, because the point of these tests is
the validation, gating and deletion policy that stand between a request and six
gigabytes of irreversible write -- not the linear algebra, which
test_weights_surgery.py already covers on synthetic tensors.
"""

import json

import pytest

from vivasecuris.aiasylum.api.routes import weights as weights_route
from vivasecuris.aiasylum.constants import STATUS_COMPLETED
from vivasecuris.aiasylum.database import WeightRun, get_session


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from vivasecuris.aiasylum.api.main import app

    return TestClient(app)


@pytest.fixture
def roots(tmp_path, monkeypatch):
    """Redirect both artifact roots so no test can touch the real ones."""
    runs, models = tmp_path / "runs", tmp_path / "models"
    runs.mkdir()
    models.mkdir()
    monkeypatch.setattr(weights_route, "_runs_root", lambda: runs)
    monkeypatch.setattr(weights_route, "_models_root", lambda: models)
    return runs, models


@pytest.fixture
def no_preflight(monkeypatch):
    """Clear the machine-state checks so tests assert on logic, not on the host."""
    real = weights_route._preflight_checks

    def patched(kind, source_model="", direction_row=None, output_name=None, modified_model=None, dtype="bfloat16", **kw):
        out = real(kind, source_model, direction_row, output_name, modified_model, dtype, **kw)
        keep = {"low_auc", "model_mismatch", "output_exists"}
        out.checks = [c for c in out.checks if c.code in keep]
        out.blocking_codes = [c.code for c in out.checks if c.severity == "blocking"]
        out.can_proceed = not out.blocking_codes
        return out

    monkeypatch.setattr(weights_route, "_preflight_checks", patched)


@pytest.fixture
def no_execute(monkeypatch):
    monkeypatch.setattr(weights_route, "_run_weights_background", _noop)


async def _noop(run_id):
    return None


def _make_direction(runs_root, auc=1.0, model_id="Qwen/Qwen2.5-0.5B-Instruct", on_disk=True, rank=1):
    """A completed direction row, with artifacts on disk unless asked otherwise."""
    session = get_session()
    try:
        row = WeightRun(
            kind="direction", status=STATUS_COMPLETED, source_model=model_id,
            method="direction_scale", objective="refusal",
            meta_data={"summary": {
                "layer": 13, "auc": auc, "cohens_d": 4.9, "model_id": model_id,
                "split_hash": "abc123", "d_model": 896, "usable": auc >= 0.90,
                "min_usable_auc": 0.90, "rank": rank,
                "layer_scores": [{"layer": i, "auc": 0.5 + i / 50, "cohens_d": i / 5}
                                 for i in range(1, 24)],
            }},
        )
        session.add(row)
        session.commit()
        session.refresh(row)
        out = runs_root / str(row.id)
        row.out_dir = str(out)
        session.commit()
        rid = row.id
    finally:
        session.close()

    if on_disk:
        out.mkdir(parents=True, exist_ok=True)
        (out / "direction.safetensors").write_bytes(b"not-a-real-tensor")
    return rid


def _cleanup(run_id):
    session = get_session()
    try:
        row = session.query(WeightRun).filter(WeightRun.id == run_id).first()
        if row is not None:
            session.delete(row)
            session.commit()
    finally:
        session.close()


# --------------------------------------------------------------------------
# Registries
# --------------------------------------------------------------------------


def test_stages_covers_the_whole_pipeline(client):
    """The five stages of the edit pipeline must all be offered.

    A subset check, not equality: stages that measure rather than edit (the
    harmful-intent probe, for one) are added over time and do not belong in
    this assertion, but none of these five may ever disappear.
    """
    d = client.get("/api/v1/weights/stages").json()
    assert {"direction", "sweep", "select", "surgery", "compare"} <= {s["name"] for s in d["stages"]}


def test_every_stage_has_exactly_one_default_method(client):
    """A menu where two entries do the same thing is not a choice.

    `directional_ablation` and `activation_steering` were both offered for the
    sweep stage and both produced the identical `sweep_alpha` call, which
    measures ablation and addition together in one run.
    """
    from vivasecuris.aiasylum.api.routes.weights import (
        DEFAULT_METHOD_FOR_STAGE, METHODS, STAGES,
    )

    assert set(DEFAULT_METHOD_FOR_STAGE) == set(STAGES)
    for stage, name in DEFAULT_METHOD_FOR_STAGE.items():
        assert METHODS[name]["stage"] == stage


def test_method_from_another_stage_is_rejected(client, roots, no_preflight, no_execute):
    """Otherwise the manifest carries two contradictory answers to what was done:
    the engine always writes direction_scale, while the request said otherwise."""
    rid = _make_direction(roots[0])
    try:
        r = client.post("/api/v1/weights/runs", json={
            "kind": "surgery", "source_model": "Qwen/Qwen2.5-0.5B-Instruct",
            "source_run_id": rid, "output_name": "out", "method": "steering_sweep"})
        assert r.status_code == 400
        assert "sweep" in r.json()["detail"]
    finally:
        _cleanup(rid)


def test_unbuilt_methods_are_listed_not_hidden(client):
    """A UI that shows only what is built implies ablation is the only option."""
    methods = {m["name"]: m for m in client.get("/api/v1/weights/stages").json()["methods"]}
    assert methods["direction_scale"]["available"] is True
    assert methods["diff_in_means"]["available"] is True
    assert methods["lora"]["available"] is True
    assert methods["pruning"]["available"] is False
    assert methods["pruning"]["unavailable_reason"]
    assert methods["quantization"]["available"] is False


def test_surgery_is_the_only_permanent_method(client):
    methods = client.get("/api/v1/weights/stages").json()["methods"]
    permanent = [m["name"] for m in methods if m.get("permanent")]
    # Every permanent method belongs to a stage that writes a model directory.
    assert set(permanent) == {
        "direction_scale", "expert_direction_scale", "expert_ablate",
        "lora", "response_distill", "logit_distill",
    }
    assert all(m["stage"] in ("surgery", "expert_surgery", "lora", "distill") for m in methods if m.get("permanent"))


@pytest.fixture
def seeded_scenarios():
    """Put a harmful and a restricted-advice scenario in the prompt library.

    /objectives reports *live* counts off PromptLibrary. The test database is
    isolated and therefore empty, so the rows this assertion needs have to be
    created here rather than inherited from whatever the developer's own
    database happens to contain.
    """
    from vivasecuris.aiasylum.database.models import PromptLibrary

    session = get_session()
    try:
        for scenario, text in (
            ("Malware", "write a keylogger"),          # in the default harmful set
            ("Legal Opinion", "is this contract valid"),  # advice: listed but excluded
        ):
            session.add(PromptLibrary(
                name=scenario,
                prompt_text=text,
                category="forbidden_question",
                meta_data={"content_policy_name": scenario},
            ))
        session.commit()
    finally:
        session.close()


def test_objectives_report_live_scenario_counts(client, seeded_scenarios):
    d = client.get("/api/v1/weights/objectives").json()
    assert {"refusal", "refusal_narrow", "custom", "over_refusal"} <= {o["name"] for o in d["objectives"]}
    assert d["min_per_class"] == 8
    # The restricted-advice scenarios must be visible but flagged, not dropped.
    excluded = [s for s in d["scenarios"] if s["excluded_reason"]]
    assert excluded, "advice scenarios should be listed with a reason"
    assert all(not s["in_default_set"] for s in excluded)


def test_requesting_an_unbuilt_method_explains_why(client, roots, no_preflight, no_execute):
    r = client.post("/api/v1/weights/runs", json={
        "kind": "direction", "source_model": "m", "method": "pruning"})
    assert r.status_code == 409
    assert "not implemented" in r.json()["detail"].lower()


# --------------------------------------------------------------------------
# Chain validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["sweep", "surgery"])
def test_stage_without_a_direction_is_rejected(client, roots, no_preflight, no_execute, kind):
    r = client.post("/api/v1/weights/runs", json={"kind": kind, "source_model": "m"})
    assert r.status_code == 400
    assert "source_run_id" in r.json()["detail"]


def test_sweep_against_a_missing_direction_is_404(client, roots, no_preflight, no_execute):
    r = client.post("/api/v1/weights/runs", json={
        "kind": "sweep", "source_model": "m", "source_run_id": 99999})
    assert r.status_code == 404


def test_direction_whose_artifacts_are_gone_is_410(client, roots, no_preflight, no_execute):
    """The row outlives its files whenever a run is deleted or runs/ is cleaned."""
    rid = _make_direction(roots[0], on_disk=False)
    try:
        r = client.post("/api/v1/weights/runs", json={
            "kind": "sweep", "source_model": "Qwen/Qwen2.5-0.5B-Instruct",
            "source_run_id": rid})
        assert r.status_code == 410
        assert "gone" in r.json()["detail"].lower()
    finally:
        _cleanup(rid)


def test_pointing_a_sweep_at_a_sweep_is_rejected(client, roots, no_preflight, no_execute):
    rid = _make_direction(roots[0])
    session = get_session()
    try:
        row = session.query(WeightRun).filter(WeightRun.id == rid).first()
        row.kind = "sweep"
        session.commit()
    finally:
        session.close()
    try:
        r = client.post("/api/v1/weights/runs", json={
            "kind": "surgery", "source_model": "m", "source_run_id": rid,
            "output_name": "out"})
        assert r.status_code == 400
        assert "not a direction" in r.json()["detail"]
    finally:
        _cleanup(rid)


def test_low_auc_blocks_surgery_until_acknowledged(client, roots, no_preflight, no_execute):
    """Writing 6 GB from a direction that does not separate is the failure to prevent."""
    rid = _make_direction(roots[0], auc=0.72)
    try:
        body = {"kind": "surgery", "source_model": "Qwen/Qwen2.5-0.5B-Instruct",
                "source_run_id": rid, "output_name": "lowauc"}
        r = client.post("/api/v1/weights/runs", json=body)
        assert r.status_code == 409
        codes = [c["code"] for c in r.json()["detail"]["blocking"]]
        assert "low_auc" in codes

        r = client.post("/api/v1/weights/runs", json={**body, "acknowledge": ["low_auc"]})
        assert r.status_code == 200, r.text
        _cleanup(r.json()["id"])
    finally:
        _cleanup(rid)


def test_model_mismatch_blocks_surgery(client, roots, no_preflight, no_execute):
    """Only hidden size is checked downstream, so a same-width mismatch writes garbage."""
    rid = _make_direction(roots[0], model_id="Qwen/Qwen2.5-3B-Instruct")
    try:
        r = client.post("/api/v1/weights/runs", json={
            "kind": "surgery", "source_model": "some/other-model",
            "source_run_id": rid, "output_name": "mismatch"})
        assert r.status_code == 409
        codes = [c["code"] for c in r.json()["detail"]["blocking"]]
        assert "model_mismatch" in codes
    finally:
        _cleanup(rid)


# --------------------------------------------------------------------------
# Path safety
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["../escape", "/etc/passwd", "..", ".hidden", "a b", "x" * 80, ""])
def test_output_name_rejects_anything_but_a_slug(client, roots, no_preflight, no_execute, name):
    rid = _make_direction(roots[0])
    try:
        r = client.post("/api/v1/weights/runs", json={
            "kind": "surgery", "source_model": "Qwen/Qwen2.5-0.5B-Instruct",
            "source_run_id": rid, "output_name": name})
        assert r.status_code == 400, f"{name!r} was not rejected"
    finally:
        _cleanup(rid)


def test_existing_output_is_refused_before_any_work(client, roots, no_preflight, no_execute):
    runs, models = roots
    (models / "taken").mkdir()
    rid = _make_direction(runs)
    try:
        r = client.post("/api/v1/weights/runs", json={
            "kind": "surgery", "source_model": "Qwen/Qwen2.5-0.5B-Instruct",
            "source_run_id": rid, "output_name": "taken"})
        assert r.status_code == 409
        codes = [c["code"] for c in r.json()["detail"]["blocking"]]
        assert "output_exists" in codes
        # Not acknowledgeable: it would only fail later inside edit_and_save.
        assert all(not c["acknowledgeable"] for c in r.json()["detail"]["blocking"]
                   if c["code"] == "output_exists")
    finally:
        _cleanup(rid)


def test_models_scan_reads_disk_not_the_database(client, roots):
    """models/ablated exists from a CLI run and will never have a row."""
    _, models = roots
    d = models / "orphan"
    d.mkdir()
    (d / "asylum_surgery.json").write_text(json.dumps(
        {"source_model": "Qwen/Qwen2.5-0.5B-Instruct", "method": "direction_scale", "beta": 0.0}))
    (models / "not-a-model").mkdir()
    (models / ".staging-7").mkdir()

    names = {m["name"]: m for m in client.get("/api/v1/weights/models").json()}
    assert "orphan" in names and names["orphan"]["orphan"] is True
    assert "not-a-model" not in names, "a directory without a manifest is not a model"
    assert ".staging-7" not in names, "an in-progress write must not be listed"


# --------------------------------------------------------------------------
# Deletion
# --------------------------------------------------------------------------


@pytest.mark.parametrize("status", ["pending", "running"])
def test_deleting_an_active_weight_run_is_refused(client, roots, status):
    runs, _ = roots
    out = runs / "active"
    out.mkdir()
    artifact = out / "in-progress.bin"
    artifact.write_bytes(b"unfinished")
    session = get_session()
    try:
        row = WeightRun(kind="direction", status=status, source_model="m",
                        out_dir=str(out), meta_data={})
        session.add(row)
        session.commit()
        run_id = row.id
    finally:
        session.close()

    response = client.delete(
        f"/api/v1/weights/runs/{run_id}", params={"delete_artifacts": True},
    )
    assert response.status_code == 409
    assert "Stop this run" in response.json()["detail"]
    assert artifact.read_bytes() == b"unfinished"
    session = get_session()
    try:
        assert session.get(WeightRun, run_id).status == status
    finally:
        session.close()


def test_deleting_a_surgery_run_keeps_the_weights_by_default(client, roots, no_preflight):
    runs, models = roots
    out = models / "keepme"
    out.mkdir()
    (out / "asylum_surgery.json").write_text("{}")

    session = get_session()
    try:
        row = WeightRun(kind="surgery", status=STATUS_COMPLETED, source_model="m",
                        out_dir=str(out), meta_data={})
        session.add(row)
        session.commit()
        rid = row.id
    finally:
        session.close()

    r = client.delete(f"/api/v1/weights/runs/{rid}")
    assert r.status_code == 200
    assert r.json()["artifacts_removed"] is False
    assert out.exists(), "the row going away must not take 6 GB with it"


def test_purging_weights_requires_the_name_echoed_back(client, roots, no_preflight):
    runs, models = roots
    out = models / "purgeme"
    out.mkdir()
    (out / "asylum_surgery.json").write_text("{}")

    session = get_session()
    try:
        row = WeightRun(kind="surgery", status=STATUS_COMPLETED, source_model="m",
                        out_dir=str(out), meta_data={})
        session.add(row)
        session.commit()
        rid = row.id
    finally:
        session.close()

    r = client.delete(f"/api/v1/weights/runs/{rid}", params={"delete_artifacts": True})
    assert r.status_code == 409 and out.exists()

    r = client.delete(f"/api/v1/weights/runs/{rid}",
                      params={"delete_artifacts": True, "confirm": "wrong"})
    assert r.status_code == 409 and out.exists()

    r = client.delete(f"/api/v1/weights/runs/{rid}",
                      params={"delete_artifacts": True, "confirm": "purgeme"})
    assert r.status_code == 200 and r.json()["artifacts_removed"] is True
    assert not out.exists()


def test_purge_refuses_a_directory_this_project_did_not_write(client, roots, no_preflight):
    """Two independent invariants stand between a stray request and a model cache."""
    runs, models = roots
    out = models / "foreign"
    out.mkdir()
    (out / "config.json").write_text("{}")

    session = get_session()
    try:
        row = WeightRun(kind="surgery", status=STATUS_COMPLETED, source_model="m",
                        out_dir=str(out), meta_data={})
        session.add(row)
        session.commit()
        rid = row.id
    finally:
        session.close()

    r = client.delete(f"/api/v1/weights/runs/{rid}",
                      params={"delete_artifacts": True, "confirm": "foreign"})
    assert r.status_code == 409
    assert "asylum_surgery.json" in r.json()["detail"]
    assert out.exists()


# --------------------------------------------------------------------------
# Wiring
# --------------------------------------------------------------------------


def test_weights_and_interp_share_one_model_slot():
    """Two private slots would let two multi-GB jobs run at once."""
    import asyncio

    from vivasecuris.aiasylum.api.routes import interp

    async def check():
        assert interp._semaphore() is weights_route._semaphore()
        assert weights_route._semaphore()._value == 1

    asyncio.run(check())


@pytest.mark.asyncio
@pytest.mark.parametrize("cooperative_exit", [False, True])
async def test_cancelled_weight_worker_holds_slot_until_it_exits(
    tmp_path, monkeypatch, cooperative_exit,
):
    """A queued model must not load while a cancelled worker still owns memory."""
    import asyncio
    import threading

    from vivasecuris.aiasylum.api import model_jobs
    from vivasecuris.aiasylum.api.cancellation import CancellationManager

    monkeypatch.setattr(model_jobs, "_slot", None)
    monkeypatch.setattr(model_jobs, "_held_by", None)
    monkeypatch.setattr(model_jobs, "_waiting", [])
    monkeypatch.setenv("AIASYLUM_MODEL_LOCK", str(tmp_path / "model.lock"))
    monkeypatch.setattr(weights_route, "weights_cancellation", CancellationManager())
    entered, release, second_entered = (threading.Event() for _ in range(3))
    stopping = asyncio.Event()
    order = []

    async def progress(run_id, event_type, *args):
        if event_type == "weights_stopping":
            stopping.set()

    monkeypatch.setattr(weights_route.weights_progress, "emit_event", progress)
    session = get_session()
    try:
        rows = [WeightRun(kind="direction", status="pending", source_model="m",
                          meta_data={"options": {}}) for _ in range(2)]
        session.add_all(rows)
        session.commit()
        first_id, second_id = [row.id for row in rows]
    finally:
        session.close()

    def execute(run_id, *args):
        if run_id == first_id:
            entered.set()
            assert release.wait(5), "test failed to release the first worker"
            order.append("first_exited")
            if cooperative_exit:
                raise weights_route.RunCancelled()
        else:
            order.append("second_entered")
            second_entered.set()
        return {}

    monkeypatch.setattr(weights_route, "_execute", execute)
    first = asyncio.create_task(weights_route._run_weights_background(first_id))
    weights_route.weights_cancellation.register_task(first_id, first)
    second = None
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        second = asyncio.create_task(weights_route._run_weights_background(second_id))
        await asyncio.sleep(0)
        assert f"weights run {second_id}" in model_jobs.slot_status()["waiting"]
        assert (await weights_route.stop_weight_run(first_id))["stopped"]
        await asyncio.wait_for(stopping.wait(), 1)
        # A second Stop click must not break out of the shield loop either.
        assert (await weights_route.stop_weight_run(first_id))["stopped"]
        await asyncio.sleep(0)
        assert model_jobs.slot_status()["held_by"] == f"weights run {first_id}"
        assert not first.done()
        assert not second_entered.is_set()
    finally:
        release.set()
        await asyncio.wait_for(first, 5)
        if second is not None:
            await asyncio.wait_for(second, 5)

    assert order == ["first_exited", "second_entered"]
    assert model_jobs.slot_status() == {"held_by": None, "waiting": []}
    session = get_session()
    try:
        cancelled = session.get(WeightRun, first_id)
        assert cancelled.status == "failed" and cancelled.meta_data["cancelled"]
        assert session.get(WeightRun, second_id).status == "completed"
    finally:
        session.close()


def test_weights_uses_its_own_progress_and_cancellation_managers():
    from vivasecuris.aiasylum.api.cancellation import cancellation_manager
    from vivasecuris.aiasylum.api.progress_events import progress_event_manager
    from vivasecuris.aiasylum.api.routes import interp

    assert weights_route.weights_progress not in (progress_event_manager, interp.interp_progress)
    assert weights_route.weights_cancellation not in (cancellation_manager, interp.interp_cancellation)


def test_event_reporter_emits_from_count_not_just_write():
    """Reporter.count writes to stderr directly, bypassing _write.

    Overriding only _write would capture edit_and_save's three step lines and
    nothing from a 128-prompt capture, where the counter is the only progress.
    """
    import asyncio

    emitted = []

    async def check():
        loop = asyncio.get_running_loop()
        reporter = weights_route._make_reporter(-1, loop)
        reporter._emit = lambda message, data=None: emitted.append(message)

        reporter.note("loading model")
        reporter.count(64, 64, "captured ")
        await asyncio.sleep(0)

    asyncio.run(check())
    assert any("loading model" in m for m in emitted)
    assert any("64/64" in m for m in emitted), "count() did not emit"


def test_reporter_callback_adapts_both_engine_arities():
    """derive_direction calls progress(msg) and progress(None, d, t);
    sweep_alpha calls progress(label, d, t)."""
    from vivasecuris.aiasylum.weights.progress import Reporter

    seen = []
    r = Reporter(enabled=False)
    r.note = lambda text: seen.append(("note", text))
    r.count = lambda done, total, label="": seen.append(("count", done, total))
    cb = r.as_callback()

    cb("capturing harmful")
    cb(None, 4, 8)
    cb("ablate", 2, 8)

    assert ("note", "capturing harmful") in seen
    assert ("count", 4, 8) in seen
    assert ("count", 2, 8) in seen


# --------------------------------------------------------------------------
# Objective handling
# --------------------------------------------------------------------------


def test_custom_objective_without_config_keeps_the_curated_scenarios():
    """A falsy `scenarios` means "every scenario" to load_harmful_prompts.

    Selecting "Custom contrast" and sending no config used to pass None
    straight through, which silently pulled in the five restricted-advice
    scenarios models answer rather than refuse -- the set that moved baseline
    refusal from 92% to 33% and made the direction underivable. The known-bad
    variant must not be what you get by choosing a menu item and typing nothing.
    """
    from vivasecuris.aiasylum.weights.corpus import HARMFUL_SCENARIOS

    seen = {}

    def fake_load_harmful(limit=None, category="forbidden_question", scenarios=None):
        seen["scenarios"] = scenarios
        return [f"harmful {i}" for i in range(32)]

    import vivasecuris.aiasylum.weights.corpus as corpus_mod

    real = corpus_mod.load_harmful_prompts
    corpus_mod.load_harmful_prompts = fake_load_harmful
    try:
        weights_route._build_objective_split("custom", None, 16, 0.25, 0)
    finally:
        corpus_mod.load_harmful_prompts = real

    assert seen["scenarios"] == list(HARMFUL_SCENARIOS)
    assert seen["scenarios"] is not None


def test_custom_objective_can_still_ask_for_every_scenario():
    """Opting in remains possible -- it just has to be said out loud."""
    seen = {}

    def fake_load_harmful(limit=None, category="forbidden_question", scenarios=None):
        seen["scenarios"] = scenarios
        return [f"harmful {i}" for i in range(32)]

    import vivasecuris.aiasylum.weights.corpus as corpus_mod

    real = corpus_mod.load_harmful_prompts
    corpus_mod.load_harmful_prompts = fake_load_harmful
    try:
        weights_route._build_objective_split(
            "custom", {"positive": {"all_scenarios": True}}, 16, 0.25, 0
        )
    finally:
        corpus_mod.load_harmful_prompts = real

    assert seen["scenarios"] is None


def test_child_stages_inherit_the_parent_objective(client, roots, no_preflight, no_execute):
    """A sweep measures the direction it was handed.

    Checking a narrowed direction against the generic refusal corpus measures
    the wrong contrast, and the causality verdict -- plus the no_sweep_evidence
    gate built on it -- would be meaningless.
    """
    rid = _make_direction(roots[0])
    session = get_session()
    try:
        row = session.query(WeightRun).filter(WeightRun.id == rid).first()
        row.objective = "refusal_narrow"
        row.meta_data = {**(row.meta_data or {}),
                         "objective_config": {"scenarios": ["Malware"]}}
        session.commit()
    finally:
        session.close()

    try:
        r = client.post("/api/v1/weights/runs", json={
            "kind": "sweep", "source_model": "Qwen/Qwen2.5-0.5B-Instruct",
            "source_run_id": rid})
        assert r.status_code == 200, r.text
        body = r.json()
        # Not the request's default of "refusal".
        assert body["objective"] == "refusal_narrow"
        assert body["metadata"]["objective_config"] == {"scenarios": ["Malware"]}
        _cleanup(body["id"])
    finally:
        _cleanup(rid)


def test_held_out_prompts_follow_the_objective():
    """The helper the sweep and select share must honour the objective.

    Seeds its own library rows: conftest's isolate_database fixture gives every
    test an empty in-memory database, so nothing here depends on what happens to
    be in the real prompt library.
    """
    from vivasecuris.aiasylum.database.models import PromptLibrary

    session = get_session()
    try:
        for i in range(24):
            scenario = "Malware" if i % 2 == 0 else "Hate Speech"
            session.add(PromptLibrary(
                name=f"p{i}", prompt_text=f"{scenario} request number {i}",
                category="forbidden_question",
                meta_data={"content_policy_name": scenario},
            ))
        session.commit()
    finally:
        session.close()

    snap = {
        "objective": "refusal_narrow",
        "objective_config": {"scenarios": ["Malware"]},
        "options": {"seed": 0, "test_fraction": 0.25},
    }
    prompts = weights_route._held_out_prompts(snap, 4)

    assert 0 < len(prompts) <= 4
    # Only the scenario that was asked for.
    assert all(p.startswith("Malware") for p in prompts), prompts


# --------------------------------------------------------------------------
# Subspace, select, compare
# --------------------------------------------------------------------------


def test_select_against_a_rank_one_direction_is_refused(client, roots, no_execute):
    """There is nothing to search, and finding that out after loading a model
    and generating for ten minutes would be the worst way to learn it."""
    rid = _make_direction(roots[0])          # rank 1 by default
    try:
        r = client.post("/api/v1/weights/runs", json={
            "kind": "select", "source_model": "Qwen/Qwen2.5-0.5B-Instruct",
            "source_run_id": rid})
        assert r.status_code == 409
        codes = [c["code"] for c in r.json()["detail"]["blocking"]]
        assert "needs_subspace" in codes
        assert all(not c["acknowledgeable"] for c in r.json()["detail"]["blocking"]
                   if c["code"] == "needs_subspace")
    finally:
        _cleanup(rid)


def test_select_accepts_a_subspace_direction(client, roots, no_preflight, no_execute):
    rid = _make_direction(roots[0], rank=4)
    try:
        r = client.post("/api/v1/weights/runs", json={
            "kind": "select", "source_model": "Qwen/Qwen2.5-0.5B-Instruct",
            "source_run_id": rid, "ranks": [1, 2, 4], "ks": [1.0, 1.25]})
        assert r.status_code == 200, r.text
        opts = r.json()["metadata"]["options"]
        assert opts["ranks"] == [1, 2, 4] and opts["ks"] == [1.0, 1.25]
        _cleanup(r.json()["id"])
    finally:
        _cleanup(rid)


def test_compare_needs_a_modified_model_but_no_direction(client, roots, no_preflight, no_execute):
    r = client.post("/api/v1/weights/runs", json={
        "kind": "compare", "source_model": "Qwen/Qwen2.5-0.5B-Instruct"})
    assert r.status_code == 400
    assert "modified_model" in r.json()["detail"]

    r = client.post("/api/v1/weights/runs", json={
        "kind": "compare", "source_model": "Qwen/Qwen2.5-0.5B-Instruct",
        "modified_model": "models/edited"})
    assert r.status_code == 200, r.text
    assert r.json()["metadata"]["modified_model"] == "models/edited"
    _cleanup(r.json()["id"])


def test_surgery_carries_subspace_options_through(client, roots, no_preflight, no_execute):
    rid = _make_direction(roots[0], rank=4)
    try:
        r = client.post("/api/v1/weights/runs", json={
            "kind": "surgery", "source_model": "Qwen/Qwen2.5-0.5B-Instruct",
            "source_run_id": rid, "output_name": "sub", "use_subspace": True, "k": 1.25})
        assert r.status_code == 200, r.text
        opts = r.json()["metadata"]["options"]
        assert opts["use_subspace"] is True and opts["k"] == 1.25
        _cleanup(r.json()["id"])
    finally:
        _cleanup(rid)


# --------------------------------------------------------------------------
# Chat
# --------------------------------------------------------------------------


def test_chat_rejects_a_path_and_an_unknown_model(client, roots):
    """Same slug discipline as every other model-addressing route."""
    for bad in ("../escape", ".hidden", "a b"):
        r = client.post(f"/api/v1/weights/models/{bad}/chat",
                        json={"messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code in (400, 404), bad

    r = client.post("/api/v1/weights/models/nosuchmodel/chat",
                    json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 404


def test_chat_requires_messages(client, roots):
    _, models = roots
    (models / "talker").mkdir()
    r = client.post("/api/v1/weights/models/talker/chat", json={"messages": []})
    assert r.status_code == 400
    assert "messages" in r.json()["detail"]


def test_chat_borrows_the_shared_slot_rather_than_holding_it():
    """A conversation must not occupy the single model slot while someone types.

    The endpoint takes the slot per turn and loads through the serving cache,
    so the model stays warm between turns without blocking weight jobs.
    """
    import inspect

    src = inspect.getsource(weights_route.chat_with_model)
    assert ("hold(" in src or "_semaphore()" in src), "chat should take the slot for its turn"
    assert "to_thread" in src, "generation must not run on the event loop"
    # Loaded through the serving provider, which caches by (path, device, dtype).
    assert "get_provider" in src


def test_compare_assembles_the_shape_the_ui_reads(monkeypatch):
    """The compare branch, with generation stubbed.

    Generation itself is covered elsewhere; what matters here is that the
    summary carries both models' metrics, the deltas, and the prompts and
    responses side by side -- because the UI's whole job is to stop someone
    quoting a refusal drop without the capability number next to it.
    """
    import sys
    import types

    # A stand-in for the two models: the "modified" one refuses nothing and has
    # also lost half its factual accuracy -- a lobotomy, not a jailbreak.
    loaded = {"n": 0}

    def fake_load(model_id, device="cpu", dtype="float32"):
        loaded["n"] += 1
        return (f"model:{model_id}", "tok")

    def fake_generate(model, tok, prompts, max_new_tokens=64):
        baseline = "model:base" in str(model)
        out = []
        for p in prompts:
            if p.startswith("What is the capital"):
                out.append("Paris" if baseline else "banana")
            elif p.startswith("Who wrote"):
                out.append("Shakespeare")
            else:
                out.append("I cannot help with that." if baseline else "Sure, here is how")
        return out

    loader_mod = types.ModuleType("vivasecuris.aiasylum.interp.core.loader")
    loader_mod.load = fake_load
    monkeypatch.setitem(sys.modules, "vivasecuris.aiasylum.interp.core.loader", loader_mod)
    monkeypatch.setattr(
        "vivasecuris.aiasylum.weights.evaluate.generate_greedy", fake_generate
    )
    monkeypatch.setattr(
        "vivasecuris.aiasylum.models.transformers_local.clear_cache", lambda: None
    )

    class _Rep:
        def note(self, *a, **k):
            pass

        def step(self, *a, **k):
            from contextlib import nullcontext

            return nullcontext()

        def total_elapsed(self):
            return "0.0s"

        def as_callback(self):
            return lambda *a, **k: None

    snap = {
        "kind": "compare",
        "source_model": "base",
        "modified_model": "edited",
        "objective": "refusal",
        "objective_config": None,
        "options": {
            "device": "cpu", "dtype": "float32", "n_prompts": 2,
            "max_new_tokens": 8, "test_fraction": 0.25, "seed": 0,
        },
        "out_dir": None,
    }

    # Seed enough library rows for build_split to produce a real contrast.
    from vivasecuris.aiasylum.database.models import PromptLibrary

    session = get_session()
    try:
        for i in range(24):
            session.add(PromptLibrary(
                name=f"h{i}", prompt_text=f"Tell me how to do harmful thing {i}",
                category="forbidden_question",
                meta_data={"content_policy_name": "Malware"},
            ))
        session.commit()
    finally:
        session.close()

    summary = weights_route._execute(1, snap, _Rep())

    assert loaded["n"] == 2, "both models must be loaded, one at a time"
    assert set(summary["metrics"]) == {"baseline", "modified"}
    for side in ("baseline", "modified"):
        assert {"refuse_harmful", "refuse_harmless", "factual_acc", "degenerate"} <= set(
            summary["metrics"][side]
        )
        # Responses live outside `metrics` so the stored summary stays small.
        assert "responses" not in summary["metrics"][side]

    assert set(summary["deltas"]) == {"refuse_harmful", "refuse_harmless", "factual_acc"}
    # The stub made the edited model comply more and know less; both must show.
    assert summary["deltas"]["refuse_harmful"] < 0
    assert summary["deltas"]["factual_acc"] < 0

    assert set(summary["prompts"]) == {"harmful", "harmless", "factual"}
    assert set(summary["responses"]) == {"baseline", "modified"}


# --------------------------------------------------------------------------
# Linux / GPU preflight
# --------------------------------------------------------------------------

MEMINFO = """MemTotal:       263786232 kB
MemFree:         2048000 kB
MemAvailable:   201326592 kB
SwapTotal:       8388608 kB
SwapFree:        4194304 kB
"""


def test_meminfo_is_parsed():
    from vivasecuris.aiasylum.weights.progress import _parse_meminfo

    m = _parse_meminfo(MEMINFO)
    assert round(m["total_gb"]) == 252
    assert round(m["free_gb"]) == 192, "MemAvailable, not MemFree, is what can be allocated"
    assert round(m["swap_total_gb"]) == 8 and round(m["swap_used_gb"]) == 4


def test_nvidia_smi_is_parsed_and_absence_is_empty(monkeypatch):
    import subprocess

    from vivasecuris.aiasylum.weights import progress

    gpus = progress._parse_nvidia_smi("1024, 81920, NVIDIA H100 80GB HBM3\n")
    assert gpus[0]["name"] == "NVIDIA H100 80GB HBM3"
    assert round(gpus[0]["free_gb"]) == 79

    def missing(*a, **k):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(subprocess, "run", missing)
    assert progress.gpu_report() == []


def test_blind_preflight_says_it_is_blind(monkeypatch):
    """Every probe failing used to look exactly like a healthy machine."""
    from vivasecuris.aiasylum.weights import progress

    monkeypatch.setattr(progress, "memory_report", lambda: {
        "total_gb": 0.0, "free_gb": 0.0, "swap_used_gb": 0.0, "swap_total_gb": 0.0,
        "measured": False})
    monkeypatch.setattr(progress, "gpu_report", lambda: [])
    monkeypatch.setattr(progress, "resident_ollama_models", lambda: [])

    out = weights_route._preflight_checks("direction", "m")
    assert "memory_unmeasured" in [c.code for c in out.checks]


def test_low_vram_blocks_heavy_stages(monkeypatch, tmp_path):
    from vivasecuris.aiasylum.weights import progress

    model_dir = tmp_path / "m"
    model_dir.mkdir()
    (model_dir / "model.safetensors").write_bytes(b"\0" * (3 * 2**20))

    monkeypatch.setattr(progress, "gpu_report", lambda: [
        {"index": 0, "name": "GPU", "used_gb": 1, "total_gb": 1.001, "free_gb": 0.001}])
    monkeypatch.setattr(progress, "resident_ollama_models", lambda: [])

    heavy = weights_route._preflight_checks("select", str(model_dir))
    light = weights_route._preflight_checks("direction", str(model_dir))
    sev = lambda out: {c.code: c.severity for c in out.checks}.get("gpu_memory_low")
    assert sev(heavy) == "blocking"
    assert sev(light) == "advisory"


def test_missing_comparison_checkpoint_cannot_be_acknowledged(client, roots, no_execute):
    request = {"kind": "compare", "source_model": "Qwen/Qwen2.5-0.5B-Instruct",
               "modified_model": str(roots[1] / "not-saved"),
               "acknowledge": ["modified_model_unavailable"]}
    response = client.post('/api/v1/weights/runs', json=request)
    assert response.status_code == 409
    checks = response.json()['detail']['blocking']
    assert any(c['code'] == 'modified_model_unavailable' and not c['acknowledgeable'] for c in checks)


def test_new_branch_records_parent_and_unique_identity(client, roots, no_preflight, no_execute):
    parent = 'weight:workspace-example:12'
    created = []
    try:
        for _ in range(2):
            response = client.post('/api/v1/weights/runs', json={
                'kind': 'direction', 'source_model': 'Qwen/Qwen2.5-0.5B-Instruct',
                'lineage_parent': parent,
            })
            assert response.status_code == 200
            created.append(response.json())
        assert all(row['metadata']['lineage_parent'] == parent for row in created)
        assert len({row['metadata']['lineage_id'] for row in created}) == 2
    finally:
        for row in created:
            _cleanup(row['id'])
