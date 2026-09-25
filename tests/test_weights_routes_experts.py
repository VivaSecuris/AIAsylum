"""Route-level checks for the mixture-of-experts stages.

`routing` records which experts fire; `expert_surgery` edits the ones you name.
As in test_weights_routes.py nothing here loads a model: the fixtures redirect
both roots and stub execution, so what is tested is validation and recording.
"""

import json

import pytest

from tests.test_weights_routes import (  # noqa: F401  (fixtures)
    _cleanup,
    _make_direction,
    client,
    no_execute,
    no_preflight,
    roots,
)

RUNS = "/api/v1/weights/runs"


def test_moe_stages_and_methods_are_listed(client):
    d = client.get("/api/v1/weights/stages").json()
    stages = {s["name"]: s for s in d["stages"]}
    assert "routing" in stages and "expert_surgery" in stages
    assert "expert_selection" in stages["expert_surgery"]["needs"]
    # A direction is needed only for the direction-scaling method, so the stage
    # card must not demand one up front.
    assert "source_run_id" not in stages["expert_surgery"]["needs"]

    methods = {m["name"]: m for m in d["methods"]}
    assert methods["expert_ablate"]["permanent"] is True
    assert methods["expert_ablate"]["stage"] == "expert_surgery"
    assert methods["expert_direction_scale"]["permanent"] is True
    assert methods["expert_routing"]["permanent"] is False
    assert methods["expert_routing"]["stage"] == "routing"


def test_expert_surgery_needs_a_well_formed_selection(client, roots, no_preflight, no_execute):
    base = {"kind": "expert_surgery", "source_model": "m", "output_name": "out"}
    r = client.post(RUNS, json=base)
    assert r.status_code == 400
    assert "expert_selection" in r.json()["detail"]

    for bad in ({"12": "some"}, {"x": [1]}, {"12": []}, {"12": [1, -1]}, {"12": [True]}, {"12": "all", "13": [1.5]}):
        r = client.post(RUNS, json={**base, "expert_selection": bad})
        assert r.status_code == 400, bad


def test_expert_ablate_needs_no_direction_and_records_its_options(client, roots, no_preflight, no_execute):
    r = client.post(RUNS, json={
        "kind": "expert_surgery", "source_model": "m", "output_name": "out",
        "expert_selection": {"3": [1, 2], "5": "all"}, "expert_scale": 0.0,
        "include_shared_expert": True,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    try:
        assert body["kind"] == "expert_surgery"
        assert body["method"] == "expert_ablate"
        assert body["source_run_id"] is None
        assert body["out_dir"] == str(roots[1] / "out")
        opts = body["metadata"]["options"]
        assert opts["expert_selection"] == {"3": [1, 2], "5": "all"}
        assert opts["expert_scale"] == 0.0
        assert opts["include_shared_expert"] is True
    finally:
        _cleanup(body["id"])


def test_expert_direction_scale_needs_a_direction(client, roots, no_preflight, no_execute):
    base = {
        "kind": "expert_surgery", "source_model": "Qwen/Qwen2.5-0.5B-Instruct", "output_name": "out",
        "method": "expert_direction_scale", "expert_selection": {"3": [1]},
    }
    r = client.post(RUNS, json=base)
    assert r.status_code == 400
    assert "source_run_id" in r.json()["detail"]

    rid = _make_direction(roots[0])
    try:
        r = client.post(RUNS, json={**base, "source_run_id": rid})
        assert r.status_code == 200, r.text
        child = r.json()
        try:
            assert child["source_run_id"] == rid
            assert child["metadata"]["direction_dir"] == str(roots[0] / str(rid))
        finally:
            _cleanup(child["id"])
    finally:
        _cleanup(rid)


def test_expert_ablate_rejects_a_subspace_flag(client, roots, no_preflight, no_execute):
    r = client.post(RUNS, json={
        "kind": "expert_surgery", "source_model": "m", "output_name": "out",
        "method": "expert_ablate", "use_subspace": True, "expert_selection": {"3": [1]},
    })
    assert r.status_code == 400
    assert "expert_direction_scale" in r.json()["detail"]


def test_method_and_stage_must_agree_for_moe_methods(client, roots, no_preflight, no_execute):
    r = client.post(RUNS, json={
        "kind": "surgery", "source_model": "m", "output_name": "out",
        "method": "expert_ablate", "source_run_id": 1,
    })
    assert r.status_code == 400
    assert "expert_surgery" in r.json()["detail"]


def test_routing_needs_no_direction_and_enough_prompts(client, roots, no_preflight, no_execute):
    r = client.post(RUNS, json={"kind": "routing", "source_model": "m", "n_per_class": 16})
    assert r.status_code == 200, r.text
    body = r.json()
    try:
        assert body["kind"] == "routing"
        assert body["method"] == "expert_routing"
        assert body["source_run_id"] is None
        assert body["metadata"]["options"]["n_per_class"] == 16
        assert body["out_dir"] == str(roots[0] / str(body["id"]))
    finally:
        _cleanup(body["id"])

    r = client.post(RUNS, json={"kind": "routing", "source_model": "m", "n_per_class": 4})
    assert r.status_code == 400


def test_routing_read_endpoint(client, roots, no_preflight, no_execute):
    r = client.post(RUNS, json={"kind": "routing", "source_model": "m", "n_per_class": 16})
    rid = r.json()["id"]
    try:
        assert client.get(f"{RUNS}/{rid}/routing").status_code == 404
        out = roots[0] / str(rid)
        out.mkdir(parents=True, exist_ok=True)
        (out / "routing.json").write_text(json.dumps({"layers": [], "ranking": []}))
        r = client.get(f"{RUNS}/{rid}/routing")
        assert r.status_code == 200
        assert r.json() == {"layers": [], "ranking": []}
    finally:
        _cleanup(rid)

    rid = _make_direction(roots[0])
    try:
        assert client.get(f"{RUNS}/{rid}/routing").status_code == 400
    finally:
        _cleanup(rid)
    assert client.get(f"{RUNS}/999999/routing").status_code == 404


def test_headline_shapes():
    from vivasecuris.aiasylum.api.routes.weights import _headline

    h = _headline("routing", {
        "ranking": [{"layer": 3, "expert": 7, "delta": 0.2}], "moe_layers": 4,
        "consistency": {"gate_vs_expert_counts_match": True},
    })
    assert h == {"top_layer": 3, "top_expert": 7, "top_delta": 0.2, "moe_layers": 4, "consistent": True}

    h = _headline("expert_surgery", {
        "manifest": {"matrices_edited": 3, "extra": {"expert_mode": "ablate", "experts_edited": 3, "layers_edited": 2}},
        "output_path": "p", "size_bytes": 1,
    })
    assert h["experts"] == 3 and h["layers"] == 2 and h["expert_mode"] == "ablate"
    assert h["matrices_edited"] == 3


def test_moe_preflight_distinguishes_dense_from_moe(tmp_path):
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from vivasecuris.aiasylum.api.routes.weights import _moe_preflight

    common = dict(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2)
    dense = tmp_path / "dense"
    transformers.AutoModelForCausalLM.from_config(transformers.Qwen2Config(**common)).save_pretrained(dense)
    checks = _moe_preflight(str(dense), None)
    assert [c.code for c in checks] == ["not_moe"]
    assert checks[0].severity == "blocking" and not checks[0].acknowledgeable

    moe = tmp_path / "moe"
    cfg = transformers.Qwen2MoeConfig(**common, num_experts=4, num_experts_per_tok=2,
                                      moe_intermediate_size=32, shared_expert_intermediate_size=32)
    transformers.AutoModelForCausalLM.from_config(cfg).save_pretrained(moe)
    assert _moe_preflight(str(moe), None) == []
    assert _moe_preflight(str(moe), {"1": [0, 3]}) == []

    checks = _moe_preflight(str(moe), {"0": [9]})
    assert [c.code for c in checks] == ["expert_selection_invalid"]
    assert not checks[0].acknowledgeable and "4 experts" in checks[0].message

    checks = _moe_preflight(str(tmp_path / "missing"), None)
    assert [c.code for c in checks] == ["moe_unknown"]
