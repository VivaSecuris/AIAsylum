#!/usr/bin/env python
"""One-pass, resumable validation of the 2026 interpretability additions on a real model.

Written for an expensive remote GPU box: every step is timed, written to
``report.json`` as soon as it finishes, and skipped on re-run if its result is
already there, so a crash or a stopped instance never costs a repeat of the
steps that completed. The default budget is small on purpose; pass
``--budget full`` only once the small run has been read.

    PYTHONPATH=. python scripts/validate_interp_2026.py \\
        --model Qwen/Qwen3-1.7B --out runs/validate/qwen3-1.7b --device cuda --thinking

Steps (weights track):
  1. split           seeded harmful/harmless split from the prompt library
  2. dim             difference-in-means direction + stable rank per layer
  3. rfm             RFM-AGOP cone (rank 5) + eigenvalue weights
  4. overlap         principal angles between the DIM subspace and the RFM cone
  5. sweep           ablation/addition sweep on the DIM vector, capability-controlled
  6. curve           refusal against directions removed (RFM cone)
  7. timelines       refusal-decision timelines for a few prompts
  8. misalignment    broad-misalignment probes on the unedited model
  9. probe           harmful-intent monitor: per-layer AUROC against a shuffled-label
                     null, held-out jailbreak families, and the knows-but-complies audit
  10. patching       (optional, --patching) real forward-pass residual patching on one pair
 11. surgery        (optional, --surgery) write the RFM edit at the curve's k50 rank and
                     compare with re-derivation; writes a full model copy

Nothing here needs the API or the database beyond the prompt library.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("validate")

BUDGETS = {
    # cap_limit is the capability control size. It has to be large enough that a
    # single question flipping cannot move the verdict: on the first 8B run a
    # 6-question control read 83.3% against a 100% baseline and reported
    # "capability_cost" on the strength of one answer.
    "smoke": dict(n_per_class=16, n_prompts=4, max_new_tokens=32, rfm_rank=3, rfm_iters=3,
                  timelines=1, mis=6, probe_direct=24, probe_jb=24, probe_benign=48, audit=6,
                  cap_limit=12),
    "small": dict(n_per_class=48, n_prompts=8, max_new_tokens=64, rfm_rank=5, rfm_iters=5,
                  timelines=3, mis=8, probe_direct=80, probe_jb=80, probe_benign=160, audit=12,
                  cap_limit=12),
    "full": dict(n_per_class=128, n_prompts=32, max_new_tokens=96, rfm_rank=6, rfm_iters=5,
                 timelines=5, mis=16, probe_direct=150, probe_jb=150, probe_benign=300, audit=32,
                 cap_limit=12),
}


class Report:
    def __init__(self, path: Path):
        self.path = path
        self.data = json.loads(path.read_text()) if path.exists() else {"steps": {}, "meta": {}}

    def done(self, step: str) -> bool:
        return step in self.data["steps"] and "error" not in self.data["steps"][step]

    def get(self, step: str):
        return self.data["steps"].get(step, {}).get("result")

    def record(self, step: str, result, elapsed: float, error: str | None = None):
        entry = {"elapsed_s": round(elapsed, 2)}
        if error:
            entry["error"] = error
        else:
            entry["result"] = result
        self.data["steps"][step] = entry
        self.path.write_text(json.dumps(self.data, indent=2, default=str))


def step(report: Report, name: str, fn):
    if report.done(name):
        log.info("skip %s (already in report)", name)
        return report.get(name)
    log.info("=== %s ===", name)
    t0 = time.time()
    try:
        result = fn()
    except Exception as exc:  # record and continue with the next step
        log.exception("step %s failed", name)
        report.record(name, None, time.time() - t0, error=f"{type(exc).__name__}: {exc}")
        return None
    report.record(name, result, time.time() - t0)
    log.info("%s done in %.1fs", name, time.time() - t0)
    return result


def hold_gpu_lock(poll_s: float = 2.0, log_every_s: float = 30.0):
    """Take the host-level model lock the API's jobs also wait on.

    ``api/model_jobs.py`` serialises GPU work across processes with a flock on
    ``runs/.model-job.lock`` (or ``$AIASYLUM_MODEL_LOCK``). A WebUI job started
    while this script holds the model would otherwise load a second model onto
    the same GPU. Holding the same lock makes browser jobs queue behind the
    validation instead. Returns the open handle; keep it alive for the run.
    """
    from vivasecuris.aiasylum.api.model_jobs import try_process_lock

    waited = 0.0
    handle = try_process_lock()
    while handle is None:
        if waited == 0.0 or int(waited) % int(log_every_s) == 0:
            log.info("waiting for the host GPU lock (another model job is running)")
        time.sleep(poll_s)
        waited += poll_s
        handle = try_process_lock()
    if waited:
        log.info("acquired the host GPU lock after %.0fs", waited)
    return handle


def gpu_mem() -> dict:
    try:
        import torch

        if torch.cuda.is_available():
            return {"device": torch.cuda.get_device_name(0),
                    "total_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1),
                    "max_allocated_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return {"device": "mps"}
    except Exception:
        pass
    return {"device": "cpu"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, help="Report and artifact directory")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--budget", choices=sorted(BUDGETS), default="small")
    ap.add_argument("--thinking", action="store_true", help="Keep <think> on for sweeps/timelines (Qwen3)")
    ap.add_argument("--patching", action="store_true", help="Also run one real forward-pass patching sweep")
    ap.add_argument("--surgery", action="store_true", help="Also write the RFM edit and compare (writes a model copy)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    B = BUDGETS[args.budget]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = Report(out / "report.json")
    report.data["meta"].update({"model": args.model, "budget": args.budget, "device": args.device,
                                "dtype": args.dtype, "thinking": args.thinking, "started": time.strftime("%FT%T")})

    from vivasecuris.aiasylum.interp.core.loader import load
    from vivasecuris.aiasylum.weights.corpus import build_split
    from vivasecuris.aiasylum.weights.direction import RefusalDirection, derive_direction, derive_subspace

    # Held for the whole run, released when the process exits. API jobs on the
    # same host queue behind it rather than sharing the GPU with this model.
    gpu_lock = hold_gpu_lock()
    report.data["meta"]["gpu_lock"] = "held"

    split = step(report, "split", lambda: build_split(n_per_class=B["n_per_class"], seed=args.seed).summary())
    split_obj = build_split(n_per_class=B["n_per_class"], seed=args.seed)

    t0 = time.time()
    model, tok = load(args.model, device=args.device, dtype=args.dtype, seed=args.seed)
    report.data["meta"]["load_s"] = round(time.time() - t0, 1)
    report.data["meta"]["hardware"] = gpu_mem()

    # ---- 2. DIM -------------------------------------------------------------
    def do_dim():
        d = derive_direction(model, tok, split_obj, model_id=args.model)
        d.save(out / "dim")
        return d.metadata()
    dim_meta = step(report, "dim", do_dim)
    dim = RefusalDirection.load(out / "dim") if (out / "dim" / "direction.safetensors").exists() else None

    # ---- 3. RFM -------------------------------------------------------------
    def do_rfm():
        from vivasecuris.aiasylum.weights.rfm import derive_rfm_subspace

        d = derive_rfm_subspace(model, tok, split_obj, rank=B["rfm_rank"], iterations=B["rfm_iters"],
                                model_id=args.model)
        d.save(out / "rfm")
        return d.metadata()
    step(report, "rfm", do_rfm)
    rfm = RefusalDirection.load(out / "rfm") if (out / "rfm" / "direction.safetensors").exists() else None

    # ---- 4. overlap ---------------------------------------------------------
    def do_overlap():
        import torch

        from vivasecuris.aiasylum.interp.analysis.baselines import principal_angles, subspace_overlap

        sub = derive_subspace(model, tok, split_obj, rank=B["rfm_rank"], model_id=args.model)
        sub.save(out / "dim_subspace")
        return {
            "cos_dim_vs_rfm_v1": float(torch.dot(dim.vector.float(), rfm.vector.float()).abs()),
            "overlap_dim_subspace_vs_rfm_cone": subspace_overlap(sub.basis, rfm.basis),
            "principal_angles_deg": [round(a * 57.2958, 1) for a in principal_angles(sub.basis, rfm.basis)],
            "rfm_layer": rfm.layer, "dim_layer": dim.layer,
        }
    if dim is not None and rfm is not None:
        step(report, "overlap", do_overlap)

    prompts = list(split_obj.harmful_test[: B["n_prompts"]])

    # ---- 5. sweep -----------------------------------------------------------
    def do_sweep():
        from vivasecuris.aiasylum.weights.steering import summarize_sweep, sweep_alpha

        rows = sweep_alpha(model, tok, dim.vector, prompts, layer=dim.layer, max_new_tokens=B["max_new_tokens"],
                           capability_control=True, capability_limit=B["cap_limit"], thinking=args.thinking)
        return {"rows": rows, **summarize_sweep(rows)}
    if dim is not None:
        step(report, "sweep", do_sweep)

    # ---- 6. curve -----------------------------------------------------------
    def do_curve():
        from vivasecuris.aiasylum.weights.steering import summarize_curve, sweep_subspace_rank

        rows = sweep_subspace_rank(model, tok, rfm, prompts, max_new_tokens=B["max_new_tokens"],
                                   thinking=args.thinking, capability_control=True,
                                   capability_limit=B["cap_limit"])
        return {"rows": rows, **summarize_curve(rows)}
    curve = step(report, "curve", do_curve) if rfm is not None else None

    # ---- 7. timelines -------------------------------------------------------
    def do_timelines():
        from vivasecuris.aiasylum.weights.timeline import refusal_timeline

        return [refusal_timeline(model, tok, p, rfm, max_new_tokens=B["max_new_tokens"], thinking=args.thinking)
                for p in prompts[: B["timelines"]]]
    if rfm is not None:
        step(report, "timelines", do_timelines)

    # ---- 8. misalignment ----------------------------------------------------
    def do_mis():
        from vivasecuris.aiasylum.weights.evaluate import generate_greedy
        from vivasecuris.aiasylum.weights.misalignment import MISALIGNMENT_PROBES, misalignment_rate

        probes = list(MISALIGNMENT_PROBES)[: B["mis"]]
        resp = generate_greedy(model, tok, probes, max_new_tokens=96)
        return {**misalignment_rate(resp, probes), "responses": resp}
    step(report, "misalignment", do_mis)

    # ---- 9. probe -----------------------------------------------------------
    def do_probe():
        from vivasecuris.aiasylum.interp.probes.dataset import build_harmful_intent_dataset
        from vivasecuris.aiasylum.interp.probes.train import train_probes
        from vivasecuris.aiasylum.weights.capture import capture_pooled_residuals

        ds = build_harmful_intent_dataset(
            n_direct=B["probe_direct"], n_jailbreak=B["probe_jb"], n_benign=B["probe_benign"],
            holdout_techniques=2, seed=args.seed,
        )
        tr = capture_pooled_residuals(model, tok, ds.train_prompts, pooling="mean", batch_size=8)
        te = capture_pooled_residuals(model, tok, ds.test_prompts, pooling="mean", batch_size=8)
        ps = train_probes(tr, ds.train_labels, te, ds.test_labels, test_groups=ds.test_groups,
                          model_id=args.model, pooling="mean", dataset_hash=ds.hash,
                          dataset_summary=ds.summary(), seed=args.seed)
        ps.save(out / "probe")
        return ps.metadata()
    probe_meta = step(report, "probe", do_probe)

    # ---- 9b. knows-but-complies audit ---------------------------------------
    def do_audit():
        from vivasecuris.aiasylum.interp.probes.monitor import audit_responses, score_prompts
        from vivasecuris.aiasylum.interp.probes.train import ProbeSet
        from vivasecuris.aiasylum.weights.evaluate import generate_greedy

        ps = ProbeSet.load(out / "probe")
        targets = list(split_obj.harmful_test[: B["audit"]])
        responses = generate_greedy(model, tok, targets, max_new_tokens=B["max_new_tokens"])
        scores = score_prompts(model, tok, targets, ps)
        a = audit_responses(targets, responses, scores)
        # Full rows are large; keep the counts and the ranked failures.
        return {k: v for k, v in a.items() if k != "rows"}
    if probe_meta is not None:
        step(report, "audit", do_audit)

    # ---- 10. patching (optional) --------------------------------------------
    if args.patching:
        def do_patch():
            from vivasecuris.aiasylum.interp.core.config import Config
            from vivasecuris.aiasylum.interp.core.runner import ModelRunner
            from vivasecuris.aiasylum.interp.patching import run_patching_experiments
            from vivasecuris.aiasylum.weights.capture import format_prompts

            cfg = Config(model=args.model, out_dir=str(out / "patching"), enable_patching=True,
                         enable_attention_capture=False, enable_mlp_capture=False, enable_attn_output_capture=False)
            runner = ModelRunner(model, tok, debug_mode=False, config=cfg)
            a = runner.run_once(format_prompts(tok, [split_obj.harmful_test[0]])[0])
            b = runner.run_once(format_prompts(tok, [split_obj.harmless_test[0]])[0])
            w = min(a.input_ids.shape[1], b.input_ids.shape[1])
            payload = run_patching_experiments(model, tok, cfg, a, b, None, None, spike_layer=dim.layer if dim else 8,
                                               start_a=a.input_ids.shape[1] - w, start_b=b.input_ids.shape[1] - w,
                                               window_len=w)
            return {k: v for k, v in payload.items() if k != "experiments"} | {
                "per_layer_recovered": {
                    e["direction"] + "@" + str(e["layer"]): e["results"][0]["recovered"] for e in payload["experiments"]
                }
            }
        step(report, "patching", do_patch)

    # ---- 11. surgery (optional) ---------------------------------------------
    if args.surgery and rfm is not None:
        def do_surgery():
            from vivasecuris.aiasylum.weights.surgery import edit_and_save

            rank = (curve or {}).get("k50_rank") or rfm.rank
            edited = out / "edited-rfm"
            if not (edited / "asylum_surgery.json").exists():
                edit_and_save(args.model, RefusalDirection(
                    vector=rfm.vector, layer=rfm.layer, auc=rfm.auc, cohens_d=rfm.cohens_d, model_id=rfm.model_id,
                    split_hash=rfm.split_hash, basis=rfm.basis[:rank], basis_layers=rfm.basis_layers[:rank],
                    weights=rfm.as_weights(rank), method=rfm.method, extra=rfm.extra,
                ), str(edited), device="cpu", dtype=args.dtype, use_subspace=True, k=1.0)
            return {"path": str(edited), "rank": rank}
        del model, tok
        gc.collect()
        step(report, "surgery", do_surgery)

    report.data["meta"]["finished"] = time.strftime("%FT%T")
    report.data["meta"]["hardware"] = gpu_mem()
    report.path.write_text(json.dumps(report.data, indent=2, default=str))
    try:
        gpu_lock.close()
    except Exception:
        pass

    # ---- summary ------------------------------------------------------------
    s = report.data["steps"]
    def g(name, *keys):
        cur = s.get(name, {}).get("result")
        for k in keys:
            cur = cur.get(k) if isinstance(cur, dict) else None
        return cur
    print("\n==== validation summary ====")
    print(f"model {args.model}  budget {args.budget}  hardware {report.data['meta'].get('hardware')}")
    print(f"DIM   layer {g('dim','layer')}  AUC {g('dim','auc')}  stable rank {g('dim','extra','stable_rank','at_layer')} "
          f"({g('dim','extra','stable_rank','band')})")
    print(f"RFM   layer {g('rfm','layer')}  AUC {g('rfm','auc')}  weights {g('rfm','weights')}")
    print(f"overlap DIM-subspace vs RFM cone: {g('overlap','overlap_dim_subspace_vs_rfm_cone')}  "
          f"cos(v1): {g('overlap','cos_dim_vs_rfm_v1')}")
    print(f"sweep verdict {g('sweep','verdict')}  ablate delta {g('sweep','ablate_delta_points')} pts  "
          f"factual delta {g('sweep','factual_delta_points')} pts")
    print(f"curve k50 rank {g('curve','k50_rank')}  max compliance {g('curve','max_compliance')} at rank "
          f"{g('curve','rank_at_max')}  monotone {g('curve','monotone')}")
    tls = s.get("timelines", {}).get("result") or []
    print("timelines decision indices:", [t.get("decision_index") for t in tls],
          "in think:", [t.get("decision_in_think") for t in tls])
    print(f"misalignment (heuristic) {g('misalignment','rate')}")
    print(f"probe layer {g('probe','best_layer')}  AUROC {g('probe','best_auroc')}  "
          f"null p95 {g('probe','null_auroc_p95')}  beats null {g('probe','beats_null')}")
    print(f"  by family: {g('probe','group_auroc')}")
    print(f"knows-but-complies {g('audit','knows_but_complies_rate')}  counts {g('audit','counts')}")
    errors = {k: v["error"] for k, v in s.items() if "error" in v}
    if errors:
        print("ERRORS:", json.dumps(errors, indent=2))
    total = sum(v.get("elapsed_s", 0) for v in s.values())
    print(f"total step time {total/60:.1f} min; report at {report.path}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
