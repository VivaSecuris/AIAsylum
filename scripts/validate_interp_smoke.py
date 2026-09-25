#!/usr/bin/env python3
"""Exercise the actual single/comparison dashboards and causal patching on CUDA.

This uses short, benign prompts and does not need the prompt-library database.
Use validate_model_matrix.py for downloads, resource checks and cross-size runs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def check_result(result, html: str, mode: str, n_layers: int) -> dict:
    """Check useful outputs, including optional stages that otherwise log warnings."""
    import numpy as np

    names = ("activation_norm_mat",) if mode == "single" else ("cos_mat", "dn_mat")
    shapes = {}
    for name in names:
        matrix = np.asarray(getattr(result, name))
        if matrix.ndim != 2 or matrix.shape[0] != n_layers + 1 or not matrix.shape[1]:
            raise ValueError(f"{mode}: {name} has invalid shape {matrix.shape}")
        if not np.isfinite(matrix).all():
            raise ValueError(f"{mode}: {name} contains non-finite values")
        shapes[name] = list(matrix.shape)
    for name in ("pca_payload", "predictions_payload", "attention_payload", "mlp_payload"):
        if not getattr(result, name, None):
            raise ValueError(f"{mode}: requested {name} was not produced; inspect run.log")
    if "<!doctype html>" not in html.lower() or "cdn.plot.ly" in html:
        raise ValueError(f"{mode}: dashboard missing or uses an external Plotly CDN")
    patches = 0
    if mode == "comparison":
        payload = result.patching_results or {}
        experiments = payload.get("experiments") or []
        if not experiments or any(not experiment.get("results") for experiment in experiments):
            raise ValueError("comparison: causal patching produced no experiments")
        for experiment in experiments:
            for row in experiment["results"]:
                if not np.isfinite(row.get("recovered", float("nan"))):
                    raise ValueError("comparison: causal patching produced non-finite recovery")
        patches = len(experiments)
    return {"matrix_shapes": shapes, "patching_experiments": patches, "dashboard_bytes": len(html)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    report = {"model": args.model, "device": args.device, "dtype": "bfloat16", "steps": {},
              "started_at": datetime.now(timezone.utc).isoformat()}
    try:
        import torch
        from vivasecuris.aiasylum.interp.core.config import Config
        from vivasecuris.aiasylum.interp.core.loader import load
        from vivasecuris.aiasylum.interp.core.orchestrator import AnalysisOrchestrator

        device = torch.device(args.device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("This validation requires an available CUDA GPU; CPU fallback is disabled")
        torch.cuda.set_device(device)
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("Selected CUDA GPU does not support BF16")
        torch.cuda.reset_peak_memory_stats(device)
        model, tokenizer = load(args.model, device=args.device, dtype="bfloat16")
        n_layers = model.config.num_hidden_layers
        report.update(layers=n_layers, hidden_size=model.config.hidden_size, gpu=torch.cuda.get_device_name(device))
        for mode in ("single", "comparison"):
            start = time.monotonic()
            try:
                cfg = Config(
                    model=args.model, out_dir=args.out / mode, device=args.device, dtype="bfloat16",
                    analysis_mode=mode, max_len=128, window=64, topk=5,
                    prompt="Explain why leaves change color in autumn.",
                    prompt_a="The capital of France is", prompt_b="The capital of Germany is",
                    enable_component_analysis=True, enable_attention_capture=True,
                    enable_mlp_capture=True, enable_attn_output_capture=True,
                    enable_patching=mode == "comparison",
                    patch_layers=sorted({0, n_layers // 2, n_layers - 1}),
                )
                result, html = AnalysisOrchestrator.run_and_save(cfg, model=model, tokenizer=tokenizer)
                options = {name: getattr(cfg, name) for name in (
                    "device", "dtype", "max_len", "window", "topk", "dim_reduction", "seed",
                    "enable_component_analysis", "enable_attention_capture", "enable_mlp_capture",
                    "enable_attn_output_capture", "enable_qkv_capture", "enable_pre_mlp_capture",
                    "enable_patching", "patch_layers", "enable_scrub", "enable_minimal_circuit",
                )}
                report["steps"][mode] = {
                    "status": "passed", **check_result(result, html, mode, n_layers), "options": options,
                    "prompt_a": cfg.prompt if mode == "single" else cfg.prompt_a,
                    "prompt_b": cfg.prompt_b if mode == "comparison" else None,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                }
                del result, html
            except Exception as exc:
                logging.exception("%s failed", mode)
                report["steps"][mode] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            report["steps"][mode]["elapsed_s"] = round(time.monotonic() - start, 2)
            (args.out / "report.json").write_text(json.dumps(report, indent=2))
            gc.collect()
            torch.cuda.empty_cache()
        report["peak_allocated_gib"] = round(torch.cuda.max_memory_allocated(device) / 2**30, 3)
    except Exception as exc:
        logging.exception("Validation failed")
        report["error"] = f"{type(exc).__name__}: {exc}"
    report["status"] = "passed" if not report.get("error") and all(
        report["steps"].get(mode, {}).get("status") == "passed" for mode in ("single", "comparison")
    ) else "failed"
    report["completed_at"] = datetime.now(timezone.utc).isoformat()
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
