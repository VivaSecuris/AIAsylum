#!/usr/bin/env python3
"""Plan, download, and validate multiple sizes serially on one CUDA GPU.

Plan never downloads. Download pins each Hub revision to a local snapshot;
run uses those snapshots with network access disabled for the model loaders.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
GIB = 2**30
WEIGHT_STEPS = {"split", "dim", "rfm", "overlap", "sweep", "curve", "timelines", "misalignment", "probe", "audit", "patching"}


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def free_disk_gib(path: Path) -> float:
    while not path.exists():
        path = path.parent
    return shutil.disk_usage(path).free / GIB


def implementation_fingerprint(weights: bool) -> str:
    """A resumed pass must test the current implementation, including dirty edits."""
    digest = hashlib.sha256()
    paths = list((ROOT / "vivasecuris/aiasylum/interp").rglob("*.py"))
    paths.append(ROOT / "scripts/validate_interp_smoke.py")
    if weights:
        paths.extend((ROOT / "vivasecuris/aiasylum/weights").rglob("*.py"))
        paths.append(ROOT / "scripts/validate_interp_2026.py")
    for path in sorted(paths):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def select_models(path: Path, selection: str | None) -> list[dict]:
    catalog = json.loads(path.read_text())
    models = catalog["models"]
    keys = set()
    for model in models:
        key = model["key"]
        if not re.fullmatch(r"[a-zA-Z0-9._-]+", key) or key in keys or key in (".", ".."):
            raise ValueError(f"Invalid or duplicate model key: {key}")
        keys.add(key)
        for field in ("parameters_b", "min_vram_gib", "min_ram_gib", "download_gib"):
            if not math.isfinite(float(model[field])) or float(model[field]) <= 0:
                raise ValueError(f"{key}: {field} must be a positive number")
        if not model.get("repo_id") or not model.get("revision"):
            raise ValueError(f"{key}: repo_id and revision are required")
    if selection == "all":
        return models
    requested = selection.split(",") if selection else catalog["default_models"]
    selected = []
    for item in requested:
        match = next((m for m in models if item.strip() in (m["key"], m["repo_id"])), None)
        if match is None:
            raise ValueError(f"Unknown model {item!r}; choose {', '.join(m['key'] for m in models)} or all")
        if match not in selected:
            selected.append(match)
    if not selected:
        raise ValueError("Select at least one model")
    return selected


def hardware(device: str) -> dict:
    """Use actual free memory on the selected GPU; never sum unrelated GPUs."""
    import torch

    target = torch.device(device)
    if target.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Verify nvidia-smi and install a CUDA-enabled PyTorch wheel; CPU fallback is disabled")
    index = target.index if target.index is not None else 0
    if index >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device {index} is unavailable ({torch.cuda.device_count()} visible)")
    with torch.cuda.device(index):
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError(f"CUDA device {index} does not support BF16")
        free, total = torch.cuda.mem_get_info(index)
    ram = None
    try:
        fields = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        ram = int(fields["MemAvailable"].split()[0]) * 1024 / GIB
    except (OSError, KeyError, ValueError):
        pass
    return {"device": f"cuda:{index}", "name": torch.cuda.get_device_name(index),
            "free_vram_gib": free / GIB, "total_vram_gib": total / GIB,
            "available_ram_gib": ram, "torch": torch.__version__, "cuda": torch.version.cuda}


def resource_errors(models: list[dict], host: dict, free_disk_gib: float, download_gib: float) -> list[str]:
    errors = []
    for model in models:
        if host["free_vram_gib"] < model["min_vram_gib"]:
            errors.append(f"{model['key']}: needs an estimated {model['min_vram_gib']} GiB free on one GPU; selected GPU has {host['free_vram_gib']:.1f} GiB")
        ram = host.get("available_ram_gib")
        if ram is not None and ram < model["min_ram_gib"]:
            errors.append(f"{model['key']}: needs an estimated {model['min_ram_gib']} GiB available host RAM; only {ram:.1f} GiB available")
    required_disk = download_gib + 5  # reserve for activation/report artifacts
    if free_disk_gib < required_disk:
        errors.append(f"Model cache needs {required_disk:.1f} GiB free including artifacts reserve; only {free_disk_gib:.1f} GiB available")
    return errors


def check_snapshot(path: Path) -> None:
    if not (path / "config.json").is_file() or not (path / "tokenizer_config.json").is_file():
        raise ValueError(f"Incomplete snapshot {path}: configuration/tokenizer missing; rerun download")
    index = path / "model.safetensors.index.json"
    if index.is_file():
        files = set(json.loads(index.read_text())["weight_map"].values())
    else:
        files = {"model.safetensors"}
    if not files or any(not (path / file).is_file() or (path / file).stat().st_size == 0 for file in files):
        raise ValueError(f"Incomplete snapshot {path}: safetensors weights missing; rerun download")


def cached_snapshot(model: dict, downloads: dict) -> Path | None:
    record = downloads.get(model["key"], {})
    if record.get("repo_id") != model["repo_id"] or record.get("revision") != model["revision"]:
        return None
    try:
        path = Path(record["snapshot"])
        check_snapshot(path)
        return path
    except (KeyError, OSError, ValueError):
        return None


def report_passed(path: Path, weights: bool = False) -> bool:
    try:
        report = json.loads(path.read_text())
        if weights:
            steps = report.get("steps", {})
            return WEIGHT_STEPS <= steps.keys() and all("error" not in steps[step] for step in WEIGHT_STEPS)
        return report.get("status") == "passed" and all(
            report.get("steps", {}).get(mode, {}).get("status") == "passed" for mode in ("single", "comparison")
        )
    except (OSError, ValueError, TypeError):
        return False


def validate_one(model: dict, snapshot: Path, args, host: dict) -> dict:
    identity = {"repo_id": model["repo_id"], "snapshot": str(snapshot), "weights": args.weights,
                "budget": args.budget, "device": args.device, "dtype": "bfloat16",
                "implementation": implementation_fingerprint(args.weights)}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    out = args.out / model["key"] / digest
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "identity.json", identity)
    entry = {"model": model["repo_id"], "snapshot": str(snapshot), "out": str(out), "hardware": host, "stages": {}}
    stages = [("interp", "validate_interp_smoke.py")]
    if args.weights:
        stages.append(("weights", "validate_interp_2026.py"))
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    for stage, script in stages:
        stage_out = out / stage
        stage_out.mkdir(exist_ok=True)
        report = stage_out / "report.json"
        if args.resume and report_passed(report, weights=stage == "weights"):
            entry["stages"][stage] = {"status": "passed", "resumed": True, "report": str(report)}
            continue
        # A fresh smoke run must not inherit a stale success after an early crash.
        if stage == "interp":
            report.unlink(missing_ok=True)
        cmd = [sys.executable, str(ROOT / "scripts" / script), "--model", str(snapshot),
               "--out", str(stage_out), "--device", args.device]
        if stage == "weights":
            cmd.extend(["--dtype", "bfloat16", "--budget", args.budget, "--patching"])
        started = time.monotonic()
        print(f"{model['key']}: {stage}; log {stage_out / 'run.log'}", flush=True)
        with (stage_out / "run.log").open("a") as log:
            # The web API and CLI share one physical GPU. Hold the same lock
            # for the child lifetime so browser jobs queue rather than OOM.
            import fcntl
            lock_path = Path(os.environ.get("AIASYLUM_MODEL_LOCK", ROOT / "runs/.model-job.lock"))
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            with lock_path.open("a") as gpu_lock:
                fcntl.flock(gpu_lock, fcntl.LOCK_EX)
                completed = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
        passed = completed.returncode == 0 and report_passed(report, weights=stage == "weights")
        entry["stages"][stage] = {"status": "passed" if passed else "failed", "exit_code": completed.returncode,
                                    "elapsed_s": round(time.monotonic() - started, 2), "report": str(report)}
    entry["status"] = "passed" if all(stage["status"] == "passed" for stage in entry["stages"].values()) else "failed"
    return entry


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "download", "run"))
    parser.add_argument("--matrix", type=Path, default=ROOT / "config/interp_model_matrix.json")
    parser.add_argument("--models", help="Comma-separated catalog keys/repo IDs, or all; default: 0.6B, 1.7B, 8B")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/model-matrix")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--weights", action="store_true", help="Also run the prompt-library weights validation")
    parser.add_argument("--budget", choices=("smoke", "small", "full"), default="smoke")
    parser.add_argument("--resume", action="store_true", help="Reuse fully passing reports for identical snapshots/options")
    args = parser.parse_args(argv)
    args.out = args.out.expanduser().resolve()
    try:
        models = select_models(args.matrix, args.models)
        downloads_path = args.out / "downloads.json"
        downloads = json.loads(downloads_path.read_text()) if downloads_path.exists() else {}
        cache = Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub").expanduser()
        free_disk = free_disk_gib(cache)
        missing_gib = sum(m["download_gib"] for m in models if cached_snapshot(m, downloads) is None)
        plan = {"action": args.action, "models": models, "cache": str(cache), "estimated_download_gib": missing_gib,
                "free_cache_disk_gib": round(free_disk, 1), "free_artifact_disk_gib": round(free_disk_gib(args.out), 1), "errors": []}
        try:
            host = hardware(args.device)
            plan["hardware"] = host
            plan["errors"] = resource_errors(models, host, free_disk, missing_gib if args.action != "run" else 0)
        except (RuntimeError, ImportError, ValueError) as exc:
            plan["errors"] = [str(exc)]
        if plan["free_artifact_disk_gib"] < 5:
            plan["errors"].append("Artifact volume needs at least 5 GiB free for reports and activations")
        plan["ready"] = not plan["errors"]
        print(json.dumps(plan, indent=2), flush=True)
        if args.action == "plan":
            return 0 if plan["ready"] else 2
        if plan["errors"]:
            write_json(args.out / "matrix-report.json", {**plan, "status": "blocked"})
            return 2
        # flock is also released on interruption; avoid two downloads/runs writing the same manifest.
        import fcntl
        args.out.mkdir(parents=True, exist_ok=True)
        with (args.out / ".lock").open("w") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError("Another matrix download/run is active for this output directory")
            report = {**plan, "status": "running", "results": {}}
            write_json(args.out / "matrix-report.json", report)
            for model in models:
                try:
                    snapshot = cached_snapshot(model, downloads)
                    if args.action == "download":
                        if snapshot is None:
                            from huggingface_hub import snapshot_download
                            from vivasecuris.aiasylum.interp.core.loader import get_hf_token
                            snapshot = Path(snapshot_download(
                                repo_id=model["repo_id"], revision=model["revision"], cache_dir=cache,
                                token=get_hf_token(), allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"],
                            ))
                            check_snapshot(snapshot)
                        downloads[model["key"]] = {"repo_id": model["repo_id"], "revision": model["revision"],
                                                     "snapshot": str(snapshot.resolve()), "commit": snapshot.name}
                        write_json(downloads_path, downloads)
                        entry = {"status": "downloaded", **downloads[model["key"]]}
                    else:
                        if snapshot is None:
                            raise RuntimeError("No complete pinned snapshot; run the download phase first")
                        current_host = hardware(args.device)
                        errors = resource_errors([model], current_host, free_disk_gib(cache), 0)
                        if free_disk_gib(args.out) < 5:
                            errors.append("Artifact volume has less than 5 GiB free")
                        if errors:
                            raise RuntimeError("; ".join(errors))
                        entry = validate_one(model, snapshot, args, current_host)
                except Exception as exc:
                    entry = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
                    print(f"{model['key']}: {entry['error']}", file=sys.stderr, flush=True)
                report["results"][model["key"]] = entry
                write_json(args.out / "matrix-report.json", report)
            report["status"] = "passed" if all(entry["status"] in ("passed", "downloaded") for entry in report["results"].values()) else "failed"
            write_json(args.out / "matrix-report.json", report)
            print(f"{report['status']}: {args.out / 'matrix-report.json'}")
            return 0 if report["status"] == "passed" else 1
    except KeyboardInterrupt:
        path = args.out / "matrix-report.json"
        if path.exists():
            report = json.loads(path.read_text())
            report["status"] = "interrupted"
            write_json(path, report)
        print("Matrix interrupted; the active validation subprocess was stopped", file=sys.stderr)
        return 130
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    # subprocess.run kills and reaps its child when interrupted. Forward TERM
    # through that cleanup so `remote_session.sh down` releases GPU memory too.
    def on_terminate(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, on_terminate)
    raise SystemExit(main())
