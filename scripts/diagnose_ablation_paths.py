#!/usr/bin/env python
"""Why does the same direction move refusal through one code path and not the other?

The Qwen3-8B smoke validation reported a contradiction: the difference-in-means
direction separates harmful from harmless perfectly (held-out AUC 1.0), yet the
sweep's ``ablate`` row moved refusal by zero points, while the rank curve's
rank-1 ablation of a geometrically identical direction (cosine 1.000) took
refusal from 50 percent to zero.

Both claim to compute ``h <- h - (h.r)r`` at every layer, so one of them is not
doing what it says. This script runs the same vector and the same prompts
through four interventions and prints the refusal rate for each:

    baseline        no intervention
    steer_ablate    weights.steering.steer(mode="ablate"), the sweep's path
    ablate_subspace weights.steering.ablate_subspace(rank 1), the curve's path
    steer_plus_last steer's hooks plus a post-hook on the final block

``steer`` selects ``range(n_blocks)``, and every one of those indices takes the
pre-hook branch, so the final residual -- the output of the last block, which
feeds the norm and the unembedding -- is never touched. ``ablate_subspace``
adds that post-hook. If that is the whole difference, ``steer_plus_last`` will
match ``ablate_subspace`` and both will differ from ``steer_ablate``.

Read-only with respect to the repository: it loads a model and a saved
direction and writes one JSON report.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from contextlib import contextmanager
from pathlib import Path

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("diagnose")
log.setLevel(logging.INFO)


@contextmanager
def steer_plus_last(model, vector):
    """``steer(mode='ablate')`` plus the post-hook on the final block."""
    import torch

    from vivasecuris.aiasylum.interp.core.arch import detect_architecture, get_decoder_layers
    from vivasecuris.aiasylum.weights.steering import _as_post_hook, _make_ablate_hook

    arch = detect_architecture(model)
    blocks = get_decoder_layers(model, arch)
    vec = vector.to(torch.float32).flatten()
    vec = vec / vec.norm()
    handles = []
    try:
        for block in blocks:
            handles.append(block.register_forward_pre_hook(_make_ablate_hook(vec)))
        handles.append(blocks[-1].register_forward_hook(_as_post_hook(_make_ablate_hook(vec))))
        yield model
    finally:
        for h in handles:
            h.remove()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--direction", required=True, help="Directory holding direction.safetensors")
    ap.add_argument("--out", default="runs/validate/ablation-paths.json")
    ap.add_argument("--n-prompts", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    from vivasecuris.aiasylum.api.model_jobs import try_process_lock
    from vivasecuris.aiasylum.interp.core.arch import detect_architecture, get_decoder_layers
    from vivasecuris.aiasylum.interp.core.loader import load
    from vivasecuris.aiasylum.weights.corpus import build_split
    from vivasecuris.aiasylum.weights.direction import RefusalDirection
    from vivasecuris.aiasylum.weights.evaluate import generate_greedy
    from vivasecuris.aiasylum.weights.steering import ablate_subspace, refusal_rate, steer

    lock = try_process_lock()
    while lock is None:
        log.info("waiting for the host GPU lock")
        time.sleep(2)
        lock = try_process_lock()

    d = RefusalDirection.load(args.direction)
    prompts = list(build_split(n_per_class=32, seed=0).harmful_test[: args.n_prompts])
    log.info("direction layer %d, AUC %.3f, method %s; %d prompts",
             d.layer, d.auc, d.method, len(prompts))

    model, tok = load(args.model, device=args.device, dtype=args.dtype, seed=None)
    arch = detect_architecture(model)
    n_blocks = len(get_decoder_layers(model, arch))
    log.info("model has %d blocks; hidden_states indices 0..%d; direction at %d",
             n_blocks, n_blocks, d.layer)

    def run(label, ctx):
        t0 = time.time()
        if ctx is None:
            out = generate_greedy(model, tok, prompts, max_new_tokens=args.max_new_tokens)
        else:
            with ctx:
                out = generate_greedy(model, tok, prompts, max_new_tokens=args.max_new_tokens)
        rate = refusal_rate(out)
        log.info("%-16s refusal %5.1f%%  (%.0fs)", label, rate * 100, time.time() - t0)
        return {"refusal_rate": rate, "responses": out}

    basis = d.vector.reshape(1, -1)
    results = {
        "baseline": run("baseline", None),
        "steer_ablate": run("steer_ablate", steer(model, d.vector, mode="ablate")),
        "ablate_subspace": run("ablate_subspace", ablate_subspace(model, basis, k=1.0)),
        "steer_plus_last": run("steer_plus_last", steer_plus_last(model, d.vector)),
    }

    # Does the final residual actually still carry the direction under `steer`?
    import torch

    from vivasecuris.aiasylum.weights.capture import capture_last_token_residuals

    def final_projection(ctx):
        if ctx is None:
            acts = capture_last_token_residuals(model, tok, prompts, batch_size=4)
        else:
            with ctx:
                acts = capture_last_token_residuals(model, tok, prompts, batch_size=4)
        r = d.vector.to(torch.float32).flatten()
        r = r / r.norm()
        # |component| along the direction at the derivation layer and at the final residual
        return {
            "at_direction_layer": float((acts[d.layer] @ r).abs().mean()),
            "at_final_residual": float((acts[-1] @ r).abs().mean()),
        }

    projections = {
        "baseline": final_projection(None),
        "steer_ablate": final_projection(steer(model, d.vector, mode="ablate")),
        "ablate_subspace": final_projection(ablate_subspace(model, basis, k=1.0)),
    }

    report = {
        "model": args.model,
        "direction": {"path": args.direction, "layer": d.layer, "auc": d.auc, "method": d.method},
        "n_blocks": n_blocks,
        "prompts": prompts,
        "refusal": {k: v["refusal_rate"] for k, v in results.items()},
        "residual_projection_abs_mean": projections,
        "responses": {k: v["responses"] for k, v in results.items()},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print("\n==== refusal rate by ablation path ====")
    for k, v in report["refusal"].items():
        print(f"  {k:<18} {v*100:5.1f}%")
    print("\n==== |projection onto the direction| (mean over prompts) ====")
    print(f"  {'path':<18} {'at layer ' + str(d.layer):>16} {'at final residual':>18}")
    for k, v in projections.items():
        print(f"  {k:<18} {v['at_direction_layer']:>16.3f} {v['at_final_residual']:>18.3f}")
    print(f"\nreport: {out}")
    lock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
