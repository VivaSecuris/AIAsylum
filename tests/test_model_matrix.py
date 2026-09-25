"""Remote validation orchestration tests: no network, model downloads, or GPU."""
import argparse
import json
import re
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import validate_interp_smoke as smoke
from scripts import validate_model_matrix as matrix


def host(free_vram=90, ram=200):
    return {"free_vram_gib": free_vram, "available_ram_gib": ram}


def snapshot(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "tokenizer_config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"fake weights")
    return tmp_path


def passing_smoke():
    return {"status": "passed", "steps": {mode: {"status": "passed"} for mode in ("single", "comparison")}}


def test_catalog_includes_six_sizes_and_defaults_to_three():
    catalog = matrix.ROOT / "config/interp_model_matrix.json"
    assert [row["parameters_b"] for row in matrix.select_models(catalog, "all")] == [0.6, 1.7, 4, 8, 14, 32]
    assert len(matrix.select_models(catalog, None)) == 3
    assert matrix.select_models(catalog, "Qwen/Qwen3-8B")[0]["key"] == "qwen3-8b"
    with pytest.raises(ValueError, match="Unknown model"):
        matrix.select_models(catalog, "typo")


def test_larger_models_fail_before_download_on_small_gpu():
    models = matrix.select_models(matrix.ROOT / "config/interp_model_matrix.json", "all")
    errors = matrix.resource_errors(models, host(free_vram=24), 200, 125)
    assert len(errors) == 2
    assert "qwen3-14b" in errors[0] and "qwen3-32b" in errors[1]
    assert matrix.resource_errors(models, host(), 200, 125) == []
    assert "host RAM" in matrix.resource_errors(models[-1:], host(ram=32), 200, 0)[0]
    assert "cache needs" in matrix.resource_errors(models, host(), 20, 125)[0]


@pytest.mark.parametrize("action", ["plan", "download", "run"])
def test_no_cuda_fails_without_downloading(action, tmp_path, monkeypatch):
    monkeypatch.setattr(matrix, "hardware", lambda device: (_ for _ in ()).throw(RuntimeError("CUDA unavailable")))
    out = tmp_path / "out"
    assert matrix.main([action, "--out", str(out), "--models", "qwen3-0.6b"]) == 2
    assert not (out / "downloads.json").exists()
    if action == "plan":
        assert not out.exists(), "plan must be read-only"
    else:
        assert json.loads((out / "matrix-report.json").read_text())["status"] == "blocked"


def test_snapshot_requires_all_shards(tmp_path):
    snap = snapshot(tmp_path)
    (snap / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"layer": "missing.safetensors"}}))
    with pytest.raises(ValueError, match="weights missing"):
        matrix.check_snapshot(snap)
    (snap / "missing.safetensors").write_bytes(b"fake shard")
    matrix.check_snapshot(snap)


def test_download_pins_revision_and_run_never_downloads(tmp_path, monkeypatch):
    import huggingface_hub

    snap = snapshot(tmp_path / "cache" / "commit-123")
    calls = []
    monkeypatch.setattr(matrix, "hardware", lambda device: host())
    monkeypatch.setattr(matrix.shutil, "disk_usage", lambda path: SimpleNamespace(free=300 * matrix.GIB))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kw: calls.append(kw) or str(snap))
    monkeypatch.setattr("vivasecuris.aiasylum.interp.core.loader.get_hf_token", lambda: None)
    out = tmp_path / "out"
    argv = ["--out", str(out), "--models", "qwen3-0.6b"]
    assert matrix.main(["download", *argv]) == 0
    assert len(calls) == 1 and calls[0]["revision"] == "main"
    assert json.loads((out / "downloads.json").read_text())["qwen3-0.6b"]["commit"] == "commit-123"
    seen = []
    monkeypatch.setattr(matrix, "validate_one", lambda model, path, args, hardware: seen.append(path) or {"status": "passed"})
    assert matrix.main(["run", *argv]) == 0
    assert seen == [snap] and len(calls) == 1


def test_run_requires_download_and_returns_nonzero(tmp_path, monkeypatch):
    monkeypatch.setattr(matrix, "hardware", lambda device: host())
    monkeypatch.setattr(matrix.shutil, "disk_usage", lambda path: SimpleNamespace(free=300 * matrix.GIB))
    assert matrix.main(["run", "--out", str(tmp_path), "--models", "qwen3-0.6b"]) == 1
    row = json.loads((tmp_path / "matrix-report.json").read_text())["results"]["qwen3-0.6b"]
    assert row["status"] == "failed" and "download phase first" in row["error"]


def test_validate_child_uses_cuda_offline_and_rejects_false_success(tmp_path, monkeypatch):
    model = matrix.select_models(matrix.ROOT / "config/interp_model_matrix.json", "qwen3-0.6b")[0]
    args = argparse.Namespace(out=tmp_path / "results", device="cuda:1", weights=False, budget="smoke", resume=False)
    calls = []

    def run(cmd, **kwargs):
        calls.append((cmd, kwargs["env"]))
        out = Path(cmd[cmd.index("--out") + 1])
        matrix.write_json(out / "report.json", {"status": "passed", "steps": {"single": {"status": "passed"}}})
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(matrix.subprocess, "run", run)
    result = matrix.validate_one(model, snapshot(tmp_path / "snapshot"), args, host())
    assert result["status"] == "failed", "zero exit with missing comparison is not success"
    cmd, env = calls[0]
    assert cmd[cmd.index("--device") + 1] == "cuda:1"
    assert env["HF_HUB_OFFLINE"] == env["TRANSFORMERS_OFFLINE"] == "1"


def test_resume_only_reuses_same_snapshot_and_complete_reports(tmp_path, monkeypatch):
    model = matrix.select_models(matrix.ROOT / "config/interp_model_matrix.json", "qwen3-0.6b")[0]
    args = argparse.Namespace(out=tmp_path / "results", device="cuda:0", weights=False, budget="smoke", resume=True)
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        matrix.write_json(Path(cmd[cmd.index("--out") + 1]) / "report.json", passing_smoke())
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(matrix.subprocess, "run", run)
    first = snapshot(tmp_path / "commit-1")
    assert matrix.validate_one(model, first, args, host())["status"] == "passed"
    assert matrix.validate_one(model, first, args, host())["stages"]["interp"]["resumed"]
    assert len(calls) == 1
    matrix.validate_one(model, snapshot(tmp_path / "commit-2"), args, host())
    assert len(calls) == 2


def test_weight_report_requires_every_step(tmp_path):
    report = tmp_path / "report.json"
    matrix.write_json(report, {"steps": {step: {"result": {}} for step in matrix.WEIGHT_STEPS}})
    assert matrix.report_passed(report, weights=True)
    data = json.loads(report.read_text())
    data["steps"]["patching"] = {"error": "OOM"}
    matrix.write_json(report, data)
    assert not matrix.report_passed(report, weights=True)


def test_smoke_checks_missing_payloads_nonfinite_and_patching():
    result = SimpleNamespace(
        activation_norm_mat=np.ones((5, 3)), cos_mat=np.ones((5, 3)), dn_mat=np.ones((5, 3)),
        pca_payload={"0": {}}, predictions_payload={"data": []}, attention_payload={"0": {}},
        mlp_payload={"0": {}}, patching_results={"experiments": [{"results": [{"recovered": 0.5}]}]},
    )
    assert smoke.check_result(result, "<!DOCTYPE html>", "comparison", 4)["patching_experiments"] == 1
    result.cos_mat[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        smoke.check_result(result, "<!DOCTYPE html>", "comparison", 4)
    result.attention_payload = None
    with pytest.raises(ValueError, match="attention_payload"):
        smoke.check_result(result, "<!DOCTYPE html>", "single", 4)


@pytest.mark.parametrize("script", ["remote_validate.sh", "remote_session.sh"])
def test_remote_scripts_reject_shell_input_before_ssh(script):
    result = subprocess.run(["bash", str(matrix.ROOT / "scripts" / script), "preflight", "-oProxyCommand=bad"],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "invalid SSH host" in result.stderr or "contains characters" in result.stderr


def _script_default_remote_dir(name: str) -> str:
    """The REMOTE_DIR default a remote script falls back to."""
    text = (matrix.ROOT / "scripts" / name).read_text()
    match = re.search(r'^REMOTE_DIR="\$\{REMOTE_DIR:-([^}]+)\}"', text, re.M)
    assert match, f"{name} has no REMOTE_DIR default"
    return match.group(1)


def test_remote_scripts_agree_on_the_deploy_directory():
    """Both remote scripts must target the same checkout.

    They share its venv, its database and -- since the validator takes the same
    host GPU lock the API's jobs wait on -- its `runs/.model-job.lock`. Two
    directories means two lock files, which silently removes the mutual
    exclusion and lets a browser job load a second model onto the same GPU
    while a validation is running.
    """
    assert (_script_default_remote_dir("remote_validate.sh")
            == _script_default_remote_dir("remote_session.sh"))


def test_remote_validator_uses_paths_relative_to_remote_cwd(tmp_path):
    ssh = tmp_path / "ssh"
    ssh.write_text("#!/bin/sh\ncat\n")
    ssh.chmod(0o700)
    env = dict(os.environ, PATH=str(tmp_path) + os.pathsep + os.environ["PATH"])
    result = subprocess.run(["bash", str(matrix.ROOT / "scripts/remote_validate.sh"), "run", "gpu-box"],
                            env=env, capture_output=True, text=True, check=True)
    # Read from the script rather than hardcoded, so this pins the behaviour
    # (cd into the deploy directory, then use relative paths) and not the name.
    assert f"cd ~/{_script_default_remote_dir('remote_validate.sh')}" in result.stdout
    assert "--out 'runs/validate/" in result.stdout
    assert "--out '~/" not in result.stdout
