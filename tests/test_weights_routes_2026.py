"""Route checks for the 2026 refusal-geometry additions: the RFM-AGOP method,
the rank curve, the over-refusal objective and the compare extras.

Same discipline as test_weights_routes.py: `_run_weights_background` is
patched out and both roots are redirected, so nothing here loads a model.
"""

from __future__ import annotations

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
    runs, models = tmp_path / "runs", tmp_path / "models"
    runs.mkdir()
    models.mkdir()
    monkeypatch.setattr(weights_route, "_runs_root", lambda: runs)
    monkeypatch.setattr(weights_route, "_models_root", lambda: models)
    return runs, models


@pytest.fixture
def no_preflight(monkeypatch):
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
    async def _noop(run_id):
        return None

    monkeypatch.setattr(weights_route, "_run_weights_background", _noop)


def _direction_row(runs_root, rank=1, method="diff_in_means", auc=1.0, stable_rank=9.5):
    session = get_session()
    try:
        row = WeightRun(
            kind="direction", status=STATUS_COMPLETED, source_model="Qwen/Qwen2.5-0.5B-Instruct",
            method=method, objective="refusal",
            meta_data={"summary": {
                "layer": 5, "auc": auc, "cohens_d": 2.0, "usable": auc >= 0.9, "d_model": 896,
                "split_hash": "abc", "rank": rank, "method": method,
                "extra": {"stable_rank": {"at_layer": stable_rank, "band": "moderate"}},
            }},
        )
        session.add(row)
        session.commit()
        session.refresh(row)
        out = runs_root / str(row.id)
        out.mkdir()
        (out / "direction.safetensors").write_bytes(b"x")
        row.out_dir = str(out)
        session.commit()
        return row.id
    finally:
        session.close()


def test_stages_expose_the_new_methods_and_objective(client):
    data = client.get("/api/v1/weights/stages").json()
    methods = {m["name"]: m for m in data["methods"]}
    assert methods["rfm_agop"]["stage"] == "direction" and methods["rfm_agop"]["available"]
    assert methods["subspace_curve"]["stage"] == "sweep"
    assert methods["compare_rederive"]["stage"] == "compare"
    assert "over_refusal" in {o["name"] for o in data["objectives"]}


def test_rfm_direction_run_records_its_options(client, roots, no_preflight, no_execute):
    r = client.post("/api/v1/weights/runs", json={
        "kind": "direction", "source_model": "m", "method": "rfm_agop",
        "rfm_rank": 3, "rfm_iterations": 4,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["method"] == "rfm_agop"
    assert body["metadata"]["options"]["rfm_rank"] == 3
    assert body["metadata"]["options"]["rfm_iterations"] == 4


def test_rfm_rank_and_iterations_are_validated(client, roots, no_preflight, no_execute):
    r = client.post("/api/v1/weights/runs", json={
        "kind": "direction", "source_model": "m", "method": "rfm_agop", "rfm_rank": 0})
    assert r.status_code == 400
    r = client.post("/api/v1/weights/runs", json={
        "kind": "direction", "source_model": "m", "method": "rfm_agop", "rfm_iterations": 0})
    assert r.status_code == 400


def test_rank_curve_refuses_a_single_vector(client, roots, no_preflight, no_execute):
    runs, _ = roots
    did = _direction_row(runs, rank=1)
    r = client.post("/api/v1/weights/runs", json={
        "kind": "sweep", "source_model": "Qwen/Qwen2.5-0.5B-Instruct",
        "source_run_id": did, "method": "subspace_curve"})
    assert r.status_code == 409
    assert "rank curve" in r.json()["detail"]


def test_rank_curve_accepts_a_cone_and_records_timeline_and_thinking(client, roots, no_preflight, no_execute):
    runs, _ = roots
    did = _direction_row(runs, rank=4, method="rfm_agop")
    r = client.post("/api/v1/weights/runs", json={
        "kind": "sweep", "source_model": "Qwen/Qwen2.5-0.5B-Instruct",
        "source_run_id": did, "method": "subspace_curve", "thinking": True, "timeline_prompts": 2})
    assert r.status_code == 200, r.text
    opts = r.json()["metadata"]["options"]
    assert opts["thinking"] is True and opts["timeline_prompts"] == 2
    r = client.post("/api/v1/weights/runs", json={
        "kind": "sweep", "source_model": "Qwen/Qwen2.5-0.5B-Instruct",
        "source_run_id": did, "timeline_prompts": -1})
    assert r.status_code == 400


def test_directions_listing_reports_method_rank_and_stable_rank(client, roots):
    runs, _ = roots
    _direction_row(runs, rank=3, method="rfm_agop", stable_rank=14.0)
    rows = client.get("/api/v1/weights/directions").json()
    assert rows and rows[0]["method"] == "rfm_agop" and rows[0]["rank"] == 3
    assert rows[0]["stable_rank"] == 14.0


def test_over_refusal_objective_needs_both_prompt_lists():
    with pytest.raises(ValueError):
        weights_route._build_objective_split("over_refusal", {"refused": ["a"] * 8}, 8, 0.25, 0)
    split = weights_route._build_objective_split(
        "over_refusal",
        {"refused": [f"r{i}" for i in range(10)], "answered": [f"a{i}" for i in range(10)]},
        10, 0.25, 0,
    )
    assert split.source.startswith("over_refusal")
    assert len(split.harmful_train) + len(split.harmful_test) == 10


def test_compare_extras_flags_are_recorded(client, roots, no_preflight, no_execute):
    r = client.post("/api/v1/weights/runs", json={
        "kind": "compare", "source_model": "m", "modified_model": "m-edited",
        "method": "compare_rederive", "rederive": True, "misalignment_control": True})
    assert r.status_code == 200, r.text
    opts = r.json()["metadata"]["options"]
    assert opts["rederive"] is True and opts["misalignment_control"] is True


def test_headlines_carry_the_new_numbers():
    h = weights_route._headline("direction", {
        "layer": 5, "auc": 0.95, "usable": True, "method": "rfm_agop", "rank": 3,
        "extra": {"stable_rank": {"at_layer": 11.2}},
        "refusal_overlap": {"cosine_with_refusal": 0.3},
    })
    assert h["method"] == "rfm_agop" and h["rank"] == 3 and h["stable_rank"] == 11.2
    assert h["refusal_overlap"] == 0.3
    h = weights_route._headline("sweep", {"curve": [], "k50_rank": 3, "max_compliance": 0.8,
                                          "rank_at_max": 4, "monotone": True, "any_degenerate": False})
    assert h["k50_rank"] == 3 and h["rank_at_max"] == 4
    h = weights_route._headline("compare", {"deltas": {"refuse_harmful": -0.5, "factual_acc": 0.0,
                                                       "misalignment_rate": 0.1, "rederived_auc": -0.2}})
    assert h["misalignment_delta"] == 0.1 and h["rederived_auc_delta"] == -0.2


# ---------------------------------------------------------------------------
# Probe stage
# ---------------------------------------------------------------------------


def test_probe_stage_and_method_are_listed(client):
    data = client.get("/api/v1/weights/stages").json()
    assert "probe" in {s["name"] for s in data["stages"]}
    methods = {m["name"]: m for m in data["methods"]}
    assert methods["linear_probe"]["stage"] == "probe" and methods["linear_probe"]["available"]
    # The registry must say raw activations, since that is the finding it encodes.
    assert "raw" in methods["linear_probe"]["description"].lower()


def test_probe_run_records_pooling_and_dataset_sizes(client, roots, no_preflight, no_execute):
    r = client.post("/api/v1/weights/runs", json={
        "kind": "probe", "source_model": "m", "pooling": "last_k",
        "use_eliciting_suffix": True, "n_direct": 40, "holdout_techniques": 3})
    assert r.status_code == 200, r.text
    opts = r.json()["metadata"]["options"]
    assert opts["pooling"] == "last_k" and opts["use_eliciting_suffix"] is True
    assert opts["n_direct"] == 40 and opts["holdout_techniques"] == 3
    assert r.json()["method"] == "linear_probe"


def test_probe_rejects_unknown_pooling(client, roots, no_preflight, no_execute):
    r = client.post("/api/v1/weights/runs", json={
        "kind": "probe", "source_model": "m", "pooling": "median"})
    assert r.status_code == 400 and "pooling" in r.json()["detail"].lower()
    r = client.post("/api/v1/weights/runs", json={
        "kind": "probe", "source_model": "m", "holdout_techniques": -1})
    assert r.status_code == 400


def test_probe_needs_no_direction(client, roots, no_preflight, no_execute):
    """Unlike sweep/select/surgery, a probe is fitted from scratch."""
    r = client.post("/api/v1/weights/runs", json={"kind": "probe", "source_model": "m"})
    assert r.status_code == 200, r.text
    assert r.json()["source_run_id"] is None


def test_probes_listing_reports_auroc_and_null(client, roots):
    runs, _ = roots
    session = get_session()
    try:
        row = WeightRun(
            kind="probe", status=STATUS_COMPLETED, source_model="m", method="linear_probe",
            meta_data={"summary": {
                "best_layer": 14, "best_auroc": 0.93, "null_auroc_p95": 0.61, "beats_null": True,
                "best_ece": 0.04, "pooling": "mean", "usable": True,
                "group_auroc": {"heldout:dan": 0.88},
            }},
        )
        session.add(row); session.commit(); session.refresh(row)
        out = runs / str(row.id); out.mkdir()
        (out / "probes.npz").write_bytes(b"x")
        row.out_dir = str(out); session.commit()
    finally:
        session.close()

    rows = client.get("/api/v1/weights/probes").json()
    assert rows and rows[0]["best_layer"] == 14 and rows[0]["auroc"] == 0.93
    assert rows[0]["beats_null"] is True and rows[0]["artifacts_present"] is True
    assert rows[0]["group_auroc"]["heldout:dan"] == 0.88


def test_probe_headline():
    h = weights_route._headline("probe", {
        "best_layer": 12, "best_auroc": 0.91, "null_auroc_p95": 0.6,
        "beats_null": True, "usable": True})
    assert h["best_layer"] == 12 and h["auroc"] == 0.91 and h["beats_null"] is True


def test_combined_compare_method_enables_rederivation_without_separate_flag(client, roots, no_preflight, no_execute):
    response = client.post('/api/v1/weights/runs', json={
        'kind':'compare', 'method':'compare_rederive', 'source_model':'m', 'modified_model':'m'})
    assert response.status_code == 200, response.text
    assert response.json()['metadata']['options']['rederive'] is True


def test_overrefusal_rejects_missing_sets_before_queueing(client, roots, no_preflight, no_execute):
    response = client.post('/api/v1/weights/runs', json={
        'kind':'direction', 'objective':'over_refusal', 'source_model':'m'})
    assert response.status_code == 400
    response = client.post('/api/v1/weights/runs', json={
        'kind':'direction', 'objective':'over_refusal', 'source_model':'m',
        'objective_config':{'refused':[f'r{i}' for i in range(8)], 'answered':[f'a{i}' for i in range(8)]}})
    assert response.status_code == 200, response.text


def test_child_snapshots_parent_partition_and_options(client, roots, no_preflight, no_execute):
    from vivasecuris.aiasylum.weights.corpus import build_split
    split = build_split(8, seed=19, harmful=[f'h{i}' for i in range(20)], harmless=[f'b{i}' for i in range(20)])
    did = _direction_row(roots[0], rank=2)
    session = get_session()
    try:
        parent = session.get(WeightRun, did)
        parent.meta_data = {**parent.meta_data, 'options':{'n_per_class':8,'seed':19},
                            'summary':{**parent.meta_data['summary'], 'prompt_split':split.to_dict(), 'split_hash':split.hash}}
        session.commit()
    finally:
        session.close()
    response = client.post('/api/v1/weights/runs', json={'kind':'sweep', 'source_model':'m', 'source_run_id':did,
                                                       'n_prompts':8, 'seed':0})
    assert response.status_code == 200, response.text
    meta = response.json()['metadata']
    assert meta['source_options']['seed'] == 19
    assert weights_route._held_out_prompts(meta, 8) == split.harmful_test
