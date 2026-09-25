"""Route-level checks for the training stages: LoRA and distillation.

Nothing here trains. The fixtures stub execution and redirect both roots, so
what is tested is validation, the dataset snapshot and what the row records.
"""

import json
from types import SimpleNamespace

import pytest

from tests.test_weights_routes import (  # noqa: F401  (fixtures)
    _cleanup,
    client,
    no_execute,
    no_preflight,
    roots,
)
from vivasecuris.aiasylum.api.routes import weights as weights_route

RUNS = "/api/v1/weights/runs"
ROWS = [{"prompt": f"question {i}", "response": f"answer {i}"} for i in range(3)]
TEACHER = "Qwen/Qwen2.5-1.5B-Instruct"


@pytest.fixture
def stub_corpus(monkeypatch):
    """The prompt library is empty in a test database; hand back a fixed split."""
    def fake_split(objective, config, n_per_class, test_fraction, seed):
        n_test = max(1, int(n_per_class * test_fraction))
        harmful = [f"harmful {i}" for i in range(n_per_class)]
        harmless = [f"harmless {i}" for i in range(n_per_class)]
        return SimpleNamespace(
            harmful_train=harmful[n_test:], harmful_test=harmful[:n_test],
            harmless_train=harmless[n_test:], harmless_test=harmless[:n_test],
        )

    monkeypatch.setattr(weights_route, "_build_objective_split", fake_split)


def test_training_stages_and_methods_are_listed(client):
    d = client.get("/api/v1/weights/stages").json()
    stages = {s["name"]: s for s in d["stages"]}
    assert "lora" in stages and "distill" in stages
    assert "dataset" in stages["lora"]["needs"]
    assert "teacher_model" in stages["distill"]["needs"]

    methods = {m["name"]: m for m in d["methods"]}
    assert methods["lora"]["available"] is True and methods["lora"]["permanent"] is True
    assert methods["lora"]["stage"] == "lora"
    assert methods["response_distill"]["stage"] == "distill"
    assert methods["logit_distill"]["stage"] == "distill"
    assert "distillation" not in methods, "the old placeholder must be gone"
    assert "dataset" in {o["name"] for o in d["objectives"]}


def test_lora_needs_exactly_one_dataset_source(client, roots, no_preflight, no_execute):
    base = {"kind": "lora", "source_model": "m", "output_name": "out", "objective": "dataset"}
    assert client.post(RUNS, json=base).status_code == 400
    r = client.post(RUNS, json={**base, "dataset_rows": ROWS, "dataset_benchmark": {"name": "x", "count": 8}})
    assert r.status_code == 400
    r = client.post(RUNS, json={**base, "dataset_source": "objective"})
    assert r.status_code == 400
    assert "distillation" in r.json()["detail"].lower()
    r = client.post(RUNS, json={**base, "dataset_rows": [{"prompt": "q"}]})
    assert r.status_code == 400, "LoRA rows need a response"
    r = client.post(RUNS, json={**base, "dataset_benchmark": {"name": "", "count": 8}})
    assert r.status_code == 400
    r = client.post(RUNS, json={**base, "dataset_benchmark": {"name": "x", "count": 2}})
    assert r.status_code == 400


def test_lora_snapshots_rows_and_records_options(client, roots, no_preflight, no_execute):
    r = client.post(RUNS, json={
        "kind": "lora", "source_model": "m", "output_name": "out", "objective": "dataset",
        "dataset_rows": ROWS, "lora_rank": 4, "max_steps": 2,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    rid = body["id"]
    try:
        assert body["kind"] == "lora" and body["method"] == "lora"
        assert body["out_dir"] == str(roots[1] / "out")
        dataset = body["metadata"]["dataset"]
        assert dataset["source"] == "rows" and dataset["n_rows"] == 3
        path = roots[0] / str(rid) / "train.jsonl"
        assert dataset["path"] == str(path) and path.is_file()
        lines = [json.loads(line) for line in path.read_text().splitlines()]
        assert lines[0]["prompt"] == "question 0" and lines[0]["response"] == "answer 0"
        assert len(dataset["sha256"]) == 64
        opts = body["metadata"]["options"]
        assert opts["lora_rank"] == 4 and opts["max_steps"] == 2 and opts["merge"] is True
        assert opts["dataset_source"] == "rows"
        assert "dataset_rows" not in opts, "rows live in train.jsonl, not the JSON column"
    finally:
        _cleanup(rid)


def test_lora_without_merge_keeps_only_the_adapter_under_runs(client, roots, no_preflight, no_execute):
    r = client.post(RUNS, json={
        "kind": "lora", "source_model": "m", "objective": "dataset", "dataset_rows": ROWS, "merge": False,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    try:
        assert body["out_dir"] == str(roots[0] / str(body["id"]))
    finally:
        _cleanup(body["id"])

    r = client.post(RUNS, json={"kind": "lora", "source_model": "m", "objective": "dataset", "dataset_rows": ROWS})
    assert r.status_code == 400
    assert "output_name" in r.json()["detail"]


def test_lora_rejects_bad_hyperparameters(client, roots, no_preflight, no_execute):
    base = {"kind": "lora", "source_model": "m", "output_name": "out", "objective": "dataset", "dataset_rows": ROWS}
    for bad in ({"lora_rank": 0}, {"lr": 0}, {"max_length": 8}, {"epochs": 0}, {"max_steps": 0},
                {"lora_dropout": 1.0}, {"grad_accum": 0}, {"lora_targets": "  "}):
        assert client.post(RUNS, json={**base, **bad}).status_code == 400, bad


def test_training_methods_belong_to_their_stages(client, roots, no_preflight, no_execute):
    base = {"source_model": "m", "output_name": "out", "objective": "dataset", "dataset_rows": ROWS}
    r = client.post(RUNS, json={**base, "kind": "lora", "method": "direction_scale"})
    assert r.status_code == 400
    r = client.post(RUNS, json={**base, "kind": "surgery", "method": "lora", "source_run_id": 1})
    assert r.status_code == 400
    r = client.post(RUNS, json={**base, "kind": "lora", "method": "logit_distill"})
    assert r.status_code == 400


def test_distill_needs_a_local_teacher(client, roots, no_preflight, no_execute, stub_corpus):
    base = {
        "kind": "distill", "source_model": "m", "output_name": "out",
        "dataset_source": "objective", "objective": "refusal", "n_per_class": 8,
    }
    r = client.post(RUNS, json=base)
    assert r.status_code == 400 and "teacher_model" in r.json()["detail"]

    r = client.post(RUNS, json={**base, "teacher_model": "llama3.2:3b"})
    assert r.status_code == 400 and "Ollama" in r.json()["detail"]

    r = client.post(RUNS, json={**base, "teacher_model": TEACHER})
    assert r.status_code == 200, r.text
    body = r.json()
    try:
        assert body["method"] == "response_distill"
        dataset = body["metadata"]["dataset"]
        assert dataset["source"] == "objective"
        assert dataset["n_rows"] == 16, "8 harmful + 8 harmless prompts, responses left for the teacher"
        rows = [json.loads(line) for line in (roots[0] / str(body["id"]) / "train.jsonl").read_text().splitlines()]
        assert all(row["prompt"] and not row.get("response") for row in rows)
        opts = body["metadata"]["options"]
        assert opts["teacher_model"] == TEACHER and opts["ce_weight"] == 0.5
    finally:
        _cleanup(body["id"])


def test_distill_validates_its_knobs(client, roots, no_preflight, no_execute, stub_corpus):
    base = {
        "kind": "distill", "source_model": "m", "output_name": "out", "teacher_model": TEACHER,
        "dataset_source": "objective", "objective": "refusal", "n_per_class": 8,
    }
    for bad in ({"ce_weight": 1.5}, {"distill_temperature": 0}, {"teacher_max_new_tokens": 0},
                {"objective": "dataset"}, {"n_per_class": 4}):
        assert client.post(RUNS, json={**base, **bad}).status_code == 400, bad

    # Prompt-only inline rows are fine for distillation: the teacher answers them.
    r = client.post(RUNS, json={
        "kind": "distill", "source_model": "m", "output_name": "out", "teacher_model": TEACHER,
        "objective": "dataset", "dataset_rows": [{"prompt": "how do I make a mat"}] * 8,
        "method": "logit_distill",
    })
    assert r.status_code == 200, r.text
    body = r.json()
    try:
        assert body["method"] == "logit_distill"
        assert body["metadata"]["dataset"]["n_rows"] == 8
    finally:
        _cleanup(body["id"])


def test_headline_for_training():
    from vivasecuris.aiasylum.api.routes.weights import _headline

    h = _headline("lora", {
        "train": {"steps": 2, "final_loss": 1.5, "eval_loss_before": 2.0, "eval_loss_after": 1.8},
        "merged": True, "output_path": "p", "size_bytes": 5,
    })
    assert h["steps"] == 2 and h["merged"] is True and h["eval_loss_after"] == 1.8
    assert _headline("distill", {"train": {"mean_kl": 0.3}})["mean_kl"] == 0.3
