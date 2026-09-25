#!/usr/bin/env python
"""Known-answer controls for the interpretability engine on a real checkpoint.

Every check here has an answer that follows from the arithmetic, so a wrong
result is a defect rather than an interesting finding. `INTERP_VALIDATION_GAPS`
lists these as outstanding: the existing evidence shows the engine produces
finite numbers on real weights, not that the numbers are right.

  1. logit lens identity   Reading the final hidden state through the final
                           norm and the unembedding must reproduce the model's
                           own logits exactly. This is the fix that made the
                           lens meaningful; on a real checkpoint it either
                           matches to tolerance or it does not.

  2. self-patching         Patching a run into itself must change nothing. Any
                           drift is the patching machinery perturbing the
                           forward pass rather than measuring it.

  3. full-window patching  Replacing every position at one layer with another
                           run's residual makes the rest of the forward pass
                           identical to that run, so its logits must follow --
                           at an early, a middle and the final layer.

  4. layer sweep shape     Single-position patching recovery must rise with
                           depth and beat a same-norm random perturbation. A
                           flat or non-monotone curve means the patch is not
                           carrying the information it claims to.

  5. head and neuron       Per-head and per-neuron patches must run real
                           forwards and sum to something bounded. These paths
                           have only ever run on tiny random models.

  6. bf16 against fp32     The same controls in both dtypes, with explicit
                           tolerances, so precision is not mistaken for a bug.

Writes one JSON report. Takes the host GPU lock so it queues behind API jobs.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("controls")
log.setLevel(logging.INFO)


def agree_argmax(a, b) -> bool:
    """Do the two logit vectors pick the same token?

    The check that matters operationally: bfloat16 will not reproduce float32
    values, but it must not change which token wins.
    """
    return int(a.argmax()) == int(b.argmax())


def check(results: dict, name: str, passed: bool, detail: dict) -> None:
    results[name] = {"passed": bool(passed), **detail}
    log.info("%-26s %s  %s", name, "PASS" if passed else "FAIL",
             " ".join(f"{k}={v}" for k, v in detail.items() if not isinstance(v, (list, dict))))


def run_dtype(model_id: str, dtype: str, device: str, prompt_a: str, prompt_b: str,
              topk: int) -> dict:
    import torch

    from vivasecuris.aiasylum.interp.analysis.predictions import PredictionAnalyzer
    from vivasecuris.aiasylum.interp.core.config import Config
    from vivasecuris.aiasylum.interp.core.loader import load
    from vivasecuris.aiasylum.interp.core.runner import ModelRunner
    from vivasecuris.aiasylum.interp.patching import patch_and_run, run_patching_experiments

    results: dict = {}
    # Relative to the logit scale, not absolute. These logits span about 21, and
    # bfloat16 carries roughly three decimal digits of mantissa, so an absolute
    # bound of 0.02 fails on arithmetic that is behaving exactly as the format
    # requires. The first run failed three controls that way and none of them
    # was a defect. What survives the format is the *ranking*, which is why the
    # lens check also requires the argmax to agree.
    rel_tol = 1e-2 if dtype == "bfloat16" else 1e-6
    results["_relative_tolerance"] = rel_tol

    model, tok = load(model_id, device=device, dtype=dtype, seed=0)
    cfg = Config(model=model_id, out_dir="/tmp/controls", topk=topk,
                 enable_attention_capture=True, enable_mlp_capture=True,
                 enable_attn_output_capture=True, enable_qkv_capture=True,
                 enable_pre_mlp_capture=True, enable_patching=True)
    runner = ModelRunner(model, tok, debug_mode=True, config=cfg)

    a = runner.run_once(prompt_a)
    b = runner.run_once(prompt_b)
    n_tok = a.input_ids.shape[1]
    if n_tok != b.input_ids.shape[1]:
        raise SystemExit(f"prompts must tokenise to the same length ({n_tok} vs {b.input_ids.shape[1]})")
    n_hidden = len(a.hidden_states)
    n_blocks = n_hidden - 1
    log.info("%s %s: %d tokens, %d blocks", model_id, dtype, n_tok, n_blocks)

    # 1. logit lens identity -------------------------------------------------
    lens = PredictionAnalyzer(model, tok, topk=topk)
    lens_logits = lens.lens_logits(a.hidden_states[-1][0, -1], is_final=True)
    model_logits = a.logits[0, -1].float()
    scale = float(model_logits.abs().max())
    tol = rel_tol * scale
    results["_logit_scale"] = round(scale, 3)
    results["_absolute_tolerance"] = round(tol, 6)
    err = float((lens_logits - model_logits).abs().max())
    check(results, "logit_lens_identity", err < tol and agree_argmax(lens_logits, model_logits),
          {"max_abs_err": round(err, 6), "rel_err": round(err / scale, 8),
           "argmax_agrees": agree_argmax(lens_logits, model_logits),
           "final_hidden_is_normed": lens.final_is_normed})

    # An intermediate layer must NOT reproduce the final logits, or the lens is
    # reading the same tensor at every depth.
    mid = lens.lens_logits(a.hidden_states[n_blocks // 2][0, -1])
    check(results, "logit_lens_depth_varies", float((mid - model_logits).abs().max()) > tol,
          {"mid_layer": n_blocks // 2,
           "max_abs_diff_from_final": round(float((mid - model_logits).abs().max()), 4)})

    # 2. self-patching is a no-op -------------------------------------------
    worst = 0.0
    for layer in (1, n_blocks // 2, n_blocks):
        logits, _ = patch_and_run(model, a, a, layer, [(i, i) for i in range(n_tok)])
        worst = max(worst, float((logits - model_logits).abs().max()))
    check(results, "self_patch_is_noop", worst < tol,
          {"max_abs_err": round(worst, 6), "rel_err": round(worst / scale, 8),
           "tolerance": round(tol, 6)})

    # 3. full-window patching reproduces the source -------------------------
    b_logits = b.logits[0, -1].float()
    per_layer = {}
    for layer in (1, n_blocks // 2, n_blocks):
        logits, _ = patch_and_run(model, a, b, layer, [(i, i) for i in range(n_tok)])
        per_layer[layer] = round(float((logits - model_logits).abs().max()), 6)
    check(results, "full_window_patch_reproduces_source",
          all(v < tol for v in per_layer.values()),
          {"max_abs_err_by_layer": per_layer, "tolerance": round(tol, 6),
           "worst_rel_err": round(max(per_layer.values()) / scale, 8)})

    # 4. layer sweep shape ---------------------------------------------------
    payload = run_patching_experiments(model, tok, cfg, a, b, None, None,
                                       spike_layer=n_blocks // 2, start_a=0, start_b=0,
                                       window_len=n_tok)
    rec = {}
    for exp in payload["experiments"]:
        if exp["direction"] == "A_to_B" and exp["component"] == "layer":
            rec[exp["layer"]] = exp["results"][0]["recovered"]
    early = rec.get(1, 0.0)
    final = rec.get(n_blocks, 0.0)
    null = payload["baseline"].get("A_to_B", {}).get("recovered", 1.0)
    best = payload["summary"]["A_to_B"]["best_recovered"]
    check(results, "patch_recovery_rises_with_depth",
          final > 0.9 and early < 0.3 and best > null + 0.2,
          {"recovered_layer_1": round(early, 3), "recovered_final": round(final, 3),
           "best": round(best, 3), "random_null": round(null, 3),
           "beats_null": payload["summary"]["A_to_B"].get("beats_null")})
    results["patch_recovery_by_layer"] = {k: round(v, 4) for k, v in sorted(rec.items())}

    # 5. head and neuron patches run real forwards ---------------------------
    from dataclasses import replace

    head_payload = run_patching_experiments(model, tok, replace(cfg, patch_components="head"),
                                            a, b, None, None, spike_layer=n_blocks // 2,
                                            start_a=0, start_b=0, window_len=n_tok)
    head_recs = [e["results"][0]["recovered"] for e in head_payload["experiments"]
                 if e["direction"] == "A_to_B"]
    check(results, "head_patching_runs", bool(head_recs) and head_payload["forward_passes"] > 0,
          {"heads": len(head_recs), "forward_passes": head_payload["forward_passes"],
           "max_recovered": round(max(head_recs), 4) if head_recs else None,
           "notes": head_payload["notes"][:1]})

    neuron_payload = run_patching_experiments(model, tok, replace(cfg, patch_components="neuron"),
                                             a, b, None, None, spike_layer=n_blocks // 2,
                                             start_a=0, start_b=0, window_len=n_tok)
    neuron_recs = [e["results"][0]["recovered"] for e in neuron_payload["experiments"]
                   if e["direction"] == "A_to_B"]
    check(results, "neuron_patching_runs", bool(neuron_recs),
          {"neurons": len(neuron_recs), "forward_passes": neuron_payload["forward_passes"],
           "max_recovered": round(max(neuron_recs), 4) if neuron_recs else None,
           "notes": neuron_payload["notes"][:1]})

    # Per-neuron contributions must sum to the MLP output: that is the identity
    # the decomposition rests on, and it is what pre-MLP capture is for.
    from vivasecuris.aiasylum.interp.patching import compute_per_neuron_mlp_outputs

    layer = n_blocks // 2
    # One position only. The full window is d_ffn x seq x d_model floats, which
    # the engine now refuses above a memory ceiling -- 1.3 GiB for one layer of
    # an 8B model. The identity being checked does not need the whole window.
    last = n_tok - 1
    try:
        per = compute_per_neuron_mlp_outputs(a, model, layer, last, 1)
    except ValueError as exc:
        per = None
        guard = str(exc)
    else:
        guard = None
    if per is not None:
        actual = a.mlp_activations[layer][0].float()[last : last + 1]
        rel = float((per.sum(dim=0) - actual).norm() / actual.norm().clamp_min(1e-9))
        check(results, "neuron_decomposition_sums_to_mlp",
              rel < (5e-2 if dtype == "bfloat16" else 1e-3),
              {"relative_error": round(rel, 6), "neurons": int(per.shape[0]), "position": last})
    else:
        check(results, "neuron_decomposition_sums_to_mlp", False,
              {"reason": guard or "pre-MLP capture returned nothing"})

    del model, tok
    import gc

    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="runs/validate/known-answer-controls.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtypes", default="bfloat16,float32")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--prompt-a", default="The capital city of France is called")
    ap.add_argument("--prompt-b", default="The capital city of Japan is called")
    args = ap.parse_args()

    from vivasecuris.aiasylum.api.model_jobs import try_process_lock

    lock = try_process_lock()
    while lock is None:
        log.info("waiting for the host GPU lock")
        time.sleep(3)
        lock = try_process_lock()

    report = {"model": args.model, "prompts": [args.prompt_a, args.prompt_b], "by_dtype": {}}
    for dtype in [d.strip() for d in args.dtypes.split(",") if d.strip()]:
        log.info("=== %s ===", dtype)
        t0 = time.time()
        try:
            report["by_dtype"][dtype] = run_dtype(args.model, dtype, args.device,
                                                  args.prompt_a, args.prompt_b, args.topk)
        except Exception as exc:
            log.exception("%s failed", dtype)
            report["by_dtype"][dtype] = {"_error": f"{type(exc).__name__}: {exc}"}
        report["by_dtype"][dtype]["_elapsed_s"] = round(time.time() - t0, 1)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))

    print("\n==== known-answer controls ====")
    failed = 0
    for dtype, res in report["by_dtype"].items():
        print(f"\n{dtype}  ({res.get('_elapsed_s')}s, relative tolerance "
              f"{res.get('_relative_tolerance')} = {res.get('_absolute_tolerance')} "
              f"absolute on a logit scale of {res.get('_logit_scale')})")
        if "_error" in res:
            print(f"  ERROR {res['_error']}")
            failed += 1
            continue
        for name, r in res.items():
            if name.startswith("_") or not isinstance(r, dict) or "passed" not in r:
                continue
            mark = "PASS" if r["passed"] else "FAIL"
            extra = " ".join(f"{k}={v}" for k, v in r.items()
                             if k != "passed" and not isinstance(v, (list, dict)))
            print(f"  {mark}  {name:<34} {extra}")
            failed += 0 if r["passed"] else 1
    print(f"\nreport: {out}")
    lock.close()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
