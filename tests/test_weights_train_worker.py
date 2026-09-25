"""The training worker, in-process and as the subprocess the API spawns.

Every stdout line must be a JSON event and the last one must be ``done`` or
``error``; the exit code must agree with it."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")

from tests._tiny_lm import sample_rows, save_tiny
from vivasecuris.aiasylum.weights import train_worker
from vivasecuris.aiasylum.weights.train_data import write_jsonl

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module", autouse=True)
def small_cpu_workload():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def _write_job(tmp_path, **overrides) -> Path:
    model_dir = save_tiny(tmp_path / "base")
    write_jsonl(tmp_path / "rows.jsonl", sample_rows(4))
    job = {
        "kind": "lora",
        "source_model": str(model_dir),
        "dataset_path": str(tmp_path / "rows.jsonl"),
        "run_dir": str(tmp_path / "run"),
        "out_dir": None,
        "device": "cpu",
        "dtype": "float32",
        "lora": {"rank": 2, "alpha": 4, "max_steps": 2, "batch_size": 1, "grad_accum": 1,
                 "eval_rows": 1, "max_length": 32, "merge": False},
        "notes": None,
        "manifest_extra": {},
        "distill": None,
    }
    job.update(overrides)
    path = tmp_path / "job.json"
    path.write_text(json.dumps(job))
    return path


def _events(text: str):
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def test_main_in_process_emits_steps_then_done(tmp_path, capsys):
    code = train_worker.main(_write_job(tmp_path))
    captured = capsys.readouterr()
    events = _events(captured.out)

    assert code == 0
    kinds = [e["event"] for e in events]
    assert kinds[0] == "note" and "step" in kinds and kinds[-1] == "done"
    step = next(e for e in events if e["event"] == "step")
    assert {"phase", "step", "total", "loss", "lr", "tokens", "elapsed"} <= set(step)
    assert step["phase"] == "train" and step["total"] == 2

    result = events[-1]["result"]
    assert result["train"]["steps"] == 2 and result["stopped"] is False
    assert result["merged"] is False and result["manifest"] is None
    assert result["adapter_path"] == result["output_path"]
    assert Path(result["adapter_path"]).name == "adapter" and Path(result["adapter_path"]).is_dir()
    assert "history" not in result["train"] and len(result["history"]) >= 2
    assert result["elapsed_seconds"] > 0
    assert sys.stdout is not sys.stderr, "stdout is restored after the run"


def test_main_merges_when_asked(tmp_path, capsys):
    out_dir = tmp_path / "merged"
    job = _write_job(tmp_path, out_dir=str(out_dir), notes="merged in test", manifest_extra={"job_id": 3})
    spec = json.loads(job.read_text())
    spec["lora"]["merge"] = True
    job.write_text(json.dumps(spec))

    assert train_worker.main(job) == 0
    result = _events(capsys.readouterr().out)[-1]["result"]
    assert result["merged"] is True and result["output_path"] == str(out_dir)
    assert result["manifest"]["method"] == "lora_merge"
    assert result["manifest"]["notes"] == "merged in test"
    assert result["manifest"]["extra"]["job_id"] == 3
    assert result["manifest"]["extra"]["dataset"]["rows"] == 4
    assert (out_dir / "asylum_surgery.json").exists() and (out_dir / "config.json").exists()


def test_main_reports_errors_as_an_event(tmp_path, capsys):
    job = _write_job(tmp_path, dataset_path=str(tmp_path / "missing.jsonl"))
    assert train_worker.main(job) == 1
    events = _events(capsys.readouterr().out)
    assert events[-1]["event"] == "error"
    assert "dataset not found" in events[-1]["message"]
    assert "Traceback" in events[-1]["traceback"]


def test_main_rejects_merge_without_out_dir(tmp_path, capsys):
    job = _write_job(tmp_path)
    spec = json.loads(job.read_text())
    spec["lora"]["merge"] = True
    job.write_text(json.dumps(spec))
    assert train_worker.main(job) == 1
    assert "out_dir is null" in _events(capsys.readouterr().out)[-1]["message"]


def test_main_stop_request_exits_three_with_partial_adapter(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(train_worker, "should_stop", lambda: True)
    job = _write_job(tmp_path)
    assert train_worker.main(job) == 3
    events = _events(capsys.readouterr().out)
    assert events[-1]["event"] == "done"
    result = events[-1]["result"]
    assert result["stopped"] is True and result["merged"] is False
    assert Path(result["adapter_path"]).name == "adapter-partial"
    assert (tmp_path / "run" / "adapter-partial" / "adapter_config.json").exists()


def test_main_distills_at_response_level(tmp_path, capsys):
    write_jsonl(tmp_path / "prompts.jsonl", [r.__class__(prompt=r.prompt) for r in sample_rows(4)])
    job = _write_job(
        tmp_path, kind="distill", dataset_path=str(tmp_path / "prompts.jsonl"),
        distill={"teacher_model": str(tmp_path / "base"), "level": "response", "temperature": 2.0,
                 "ce_weight": 0.5, "teacher_max_new_tokens": 6, "teacher_system_prompt": None},
    )
    assert train_worker.main(job) == 0
    events = _events(capsys.readouterr().out)
    counts = [e for e in events if e["event"] == "count"]
    assert counts and counts[-1] == {"event": "count", "phase": "teacher_generation", "done": 4, "total": 4}
    result = events[-1]["result"]
    assert result["distill"]["level"] == "response" and result["distill"]["generated"] is True
    assert 1 <= len(result["teacher_samples"]) <= 5
    assert all(set(s) == {"prompt", "response"} for s in result["teacher_samples"])
    assert (tmp_path / "run" / "teacher_responses.jsonl").exists()
    assert result["train"]["steps"] == 2 and result["train"]["level"] == "response"


def test_subprocess_exit_code_and_parseable_lines(tmp_path):
    job = _write_job(tmp_path)
    proc = subprocess.run(
        [sys.executable, "-m", "vivasecuris.aiasylum.weights.train_worker", str(job)],
        capture_output=True, text=True, timeout=600, cwd=str(PROJECT_ROOT),
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    events = _events(proc.stdout)  # every line must parse as JSON
    assert events and events[-1]["event"] == "done"
    assert events[-1]["result"]["train"]["steps"] == 2
    assert any(e["event"] == "step" for e in events)
