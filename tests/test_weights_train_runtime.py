"""The worker driver: events reach the reporter, and a cancel stops the process group.

A fake worker module stands in for the real trainer, so this runs in a second
with no torch. What it checks is the contract the API relies on: every event
kind is relayed, a `done` result comes back, a worker error surfaces with the
worker's own message, and a cancellation kills the process promptly.
"""

import os
import textwrap
import time

import pytest

from vivasecuris.aiasylum.api.train_runtime import TrainingWorkerError, run_training_worker

FAKE_WORKER = textwrap.dedent('''
    import json, sys, time

    def emit(event):
        sys.stdout.write(json.dumps(event) + "\\n")
        sys.stdout.flush()

    job = json.loads(open(sys.argv[1]).read())
    emit({"event": "note", "message": "hello"})
    emit({"event": "step", "phase": "train", "step": 1, "total": 2, "loss": 1.5, "lr": 1e-4})
    emit({"event": "count", "phase": "teacher_generation", "done": 1, "total": 1})
    mode = job.get("mode")
    if mode == "hang":
        emit({"event": "note", "message": "sleeping"})
        time.sleep(60)
    if mode == "fail":
        emit({"event": "error", "message": "boom", "traceback": "tb"})
        sys.exit(1)
    print("stray print that is not json")
    emit({"event": "eval", "step": 2, "eval_loss": 1.2})
    emit({"event": "done", "result": {"train": {"steps": 2}, "history": [], "output_path": "x", "merged": False}})
    sys.exit(0)
''')


class _Reporter:
    def __init__(self):
        self.notes, self.metrics_, self.counts = [], [], []

    def note(self, message):
        self.notes.append(message)

    def metrics(self, data):
        self.metrics_.append(data)

    def count(self, done, total, label=""):
        self.counts.append((done, total, label))


@pytest.fixture
def fake_module(tmp_path, monkeypatch):
    pkg = tmp_path / "fakepkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "worker.py").write_text(FAKE_WORKER)
    previous = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path) + (os.pathsep + previous if previous else ""))
    return "fakepkg.worker"


def test_events_reach_the_reporter_and_the_result_comes_back(tmp_path, fake_module):
    reporter = _Reporter()
    result = run_training_worker(
        1, {"run_dir": str(tmp_path / "run"), "mode": "ok"}, tmp_path / "worker.log", reporter,
        is_cancelled=lambda: False, module=fake_module, poll=0.05,
    )
    assert result["train"]["steps"] == 2 and result.get("stopped") is not True
    assert "hello" in reporter.notes
    assert reporter.metrics_[0]["loss"] == 1.5 and reporter.metrics_[0]["step"] == 1
    assert reporter.metrics_[-1] == {"phase": "eval", "step": 2, "eval_loss": 1.2}
    assert reporter.counts == [(1, 1, "teacher generation ")]
    assert (tmp_path / "run" / "job.json").is_file()
    assert "stray print" in (tmp_path / "worker.log").read_text()


def test_a_worker_error_surfaces_with_its_own_message(tmp_path, fake_module):
    with pytest.raises(TrainingWorkerError, match="boom"):
        run_training_worker(
            2, {"run_dir": str(tmp_path / "run"), "mode": "fail"}, tmp_path / "worker.log", _Reporter(),
            is_cancelled=lambda: False, module=fake_module, poll=0.05,
        )


def test_cancellation_stops_the_process_group_promptly(tmp_path, fake_module):
    flag = {"cancel": False}

    class CancelOnSleep(_Reporter):
        def note(self, message):
            super().note(message)
            if message == "sleeping":
                flag["cancel"] = True

    started = time.time()
    result = run_training_worker(
        3, {"run_dir": str(tmp_path / "run"), "mode": "hang"}, tmp_path / "worker.log", CancelOnSleep(),
        is_cancelled=lambda: flag["cancel"], module=fake_module, poll=0.05, stop_grace=2.0,
    )
    assert result["stopped"] is True
    assert time.time() - started < 15, "a hung worker must not outlive the grace period"


def test_a_reporter_exception_terminates_the_worker(tmp_path, fake_module):
    class Raises(_Reporter):
        def note(self, message):
            if message == "sleeping":
                raise RuntimeError("cancelled by the stream")

    started = time.time()
    with pytest.raises(RuntimeError, match="cancelled by the stream"):
        run_training_worker(
            4, {"run_dir": str(tmp_path / "run"), "mode": "hang"}, tmp_path / "worker.log", Raises(),
            is_cancelled=lambda: False, module=fake_module, poll=0.05, stop_grace=2.0,
        )
    assert time.time() - started < 15
