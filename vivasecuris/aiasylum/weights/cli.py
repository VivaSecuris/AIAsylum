"""CLI for weight-level inspection and modification.

Mounted on the main ``aiasylum`` group. Every torch import is deferred into the
command bodies so that ``aiasylum --help`` keeps working on an install without
the optional ``interp`` extra.
"""

from __future__ import annotations

import json
import sys

import click

DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"

# LOW_DISK_GB now lives in weights/progress.py beside the memory preflight, so
# the API enforces the same threshold instead of re-deriving one.


def _preflight(needed_gb: float = 0.0) -> None:
    """Warn about memory contention before a run that will be slow if ignored."""
    from vivasecuris.aiasylum.weights.progress import memory_warnings

    for warning in memory_warnings(needed_gb):
        click.echo(click.style(f"WARNING: {warning}", fg="yellow"))


def _warn_if_low_disk(path: str = "/") -> None:
    from vivasecuris.aiasylum.weights.progress import disk_warnings

    for warning in disk_warnings(path):
        click.echo(click.style(f"WARNING: {warning}", fg="yellow"))


def _require_interp():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        raise click.ClickException(
            "This command needs the optional interp extra. Install it with:\n"
            '    pip install -e ".[interp]"'
        )


@click.group()
def weights():
    """Inspect and modify model weights."""


@weights.command("direction")
@click.option("--model", default=DEFAULT_MODEL, show_default=True, help="Hugging Face model id or local path")
@click.option("--out", required=True, help="Directory to write direction.safetensors into")
@click.option("--n-per-class", default=128, show_default=True, help="Prompts per class before splitting")
@click.option("--test-fraction", default=0.25, show_default=True, help="Held-out fraction")
@click.option("--seed", default=0, show_default=True)
@click.option("--batch-size", default=8, show_default=True)
@click.option("--device", default="auto", show_default=True, type=click.Choice(["auto", "cpu", "mps", "cuda"]))
@click.option("--dtype", default="bfloat16", show_default=True)
@click.option("--subspace-rank", default=1, show_default=True,
              help="Derive an orthonormal refusal subspace of this many directions (1 = single direction)")
@click.option("--pool-layers", default=12, show_default=True,
              help="How many best-separating layers to pool when building a subspace")
@click.option("--method", default="diff_in_means", show_default=True,
              type=click.Choice(["diff_in_means", "rfm_agop"]),
              help="diff_in_means, or rfm_agop for a multi-dimensional refusal cone with per-direction weights")
@click.option("--rfm-iterations", default=5, show_default=True, help="RFM-AGOP iterations (rfm_agop only)")
def derive(model, out, n_per_class, test_fraction, seed, batch_size, device, dtype, subspace_rank, pool_layers,
           method, rfm_iterations):
    """Derive a refusal direction (or subspace) by difference-in-means or RFM-AGOP."""
    _require_interp()
    from vivasecuris.aiasylum.interp.core.loader import load
    from vivasecuris.aiasylum.weights.corpus import build_split
    from vivasecuris.aiasylum.weights.direction import MIN_USABLE_AUC, derive_direction, derive_subspace

    _preflight()
    click.echo(f"Building prompt split (n={n_per_class}/class, seed={seed})...")
    split = build_split(n_per_class=n_per_class, test_fraction=test_fraction, seed=seed)
    for k, v in split.summary().items():
        click.echo(f"  {k}: {v}")

    from vivasecuris.aiasylum.weights.progress import Reporter

    reporter = Reporter()

    with reporter.step(f"loading {model} ({device}, {dtype})"):
        mdl, tok = load(model, device=device, dtype=dtype)

    report = reporter.as_callback()

    if method == "rfm_agop":
        from vivasecuris.aiasylum.weights.rfm import derive_rfm_subspace

        direction = derive_rfm_subspace(
            mdl, tok, split, rank=max(subspace_rank, 1), iterations=rfm_iterations,
            model_id=model, batch_size=batch_size, progress=report,
        )
    elif subspace_rank > 1:
        direction = derive_subspace(
            mdl, tok, split, rank=subspace_rank, pool_layers=pool_layers,
            model_id=model, batch_size=batch_size, progress=report,
        )
    else:
        direction = derive_direction(
            mdl, tok, split, model_id=model, batch_size=batch_size, progress=report,
        )
    click.echo(f"\nTotal elapsed: {reporter.total_elapsed()}")

    click.echo(f"\nBest layer: {direction.layer}   method: {direction.method}")
    click.echo(f"Held-out AUC: {direction.auc:.4f}   Cohen's d: {direction.cohens_d:.2f}")
    sr = (direction.extra or {}).get("stable_rank") or {}
    if sr.get("at_layer") is not None:
        click.echo(f"Stable rank of refusal residuals at layer {direction.layer}: "
                   f"{sr['at_layer']:.1f} ({sr.get('band')}: {sr.get('note')})")
    if direction.basis is not None:
        click.echo(f"Refusal subspace: rank {direction.rank} from layers {direction.basis_layers}")
        if direction.weights:
            click.echo(f"Removal weights (mu_i/mu_1): {[round(w, 3) for w in direction.weights]}")

    top = sorted(direction.layer_scores, key=lambda s: s.auc, reverse=True)[:5]
    click.echo("\nTop layers by held-out separation:")
    for s in top:
        click.echo(f"  layer {s.layer:3d}   AUC {s.auc:.4f}   d {s.cohens_d:6.2f}")

    direction.save(out)
    click.echo(f"\nSaved to {out}")

    if not direction.usable:
        raise click.ClickException(
            f"AUC {direction.auc:.3f} is below the {MIN_USABLE_AUC:.2f} usability threshold. "
            f"The direction does not separate these classes -- steering and surgery "
            f"will not produce a reliable behavior change. Stopping here."
        )


@weights.command("steer")
@click.option("--model", default=DEFAULT_MODEL, show_default=True)
@click.option("--direction", "direction_path", required=True, help="Directory written by `weights direction`")
@click.option("--alpha", type=float, default=None, help="Single steering strength")
@click.option("--sweep", is_flag=True, help="Sweep a range of alphas and report refusal rates")
@click.option("--prompt", default=None, help="Single prompt (default: held-out harmful prompts)")
@click.option("--n-prompts", default=8, show_default=True, help="Prompts per alpha when sweeping")
@click.option("--max-new-tokens", default=64, show_default=True)
@click.option("--device", default="auto", show_default=True)
@click.option("--dtype", default="bfloat16", show_default=True)
@click.option("--curve", is_flag=True, help="Refusal against the number of subspace directions removed")
@click.option("--thinking", is_flag=True, help="Keep a reasoning model's <think> block on while steering")
@click.option("--timeline", is_flag=True, help="Print the refusal-decision timeline for the first prompt")
def steer_cmd(model, direction_path, alpha, sweep, prompt, n_prompts, max_new_tokens, device, dtype,
              curve, thinking, timeline):
    """Steer activations at inference time. The causal check before surgery."""
    _require_interp()
    from vivasecuris.aiasylum.interp.core.loader import load
    from vivasecuris.aiasylum.weights.corpus import build_split
    from vivasecuris.aiasylum.weights.direction import RefusalDirection
    from vivasecuris.aiasylum.weights.steering import (
        generate_with_steering,
        summarize_curve,
        summarize_sweep,
        sweep_alpha,
        sweep_subspace_rank,
    )

    from vivasecuris.aiasylum.weights.progress import Reporter

    _preflight()
    d = RefusalDirection.load(direction_path)
    click.echo(f"Direction: layer {d.layer}, AUC {d.auc:.3f}, rank {d.rank} ({d.method}), from {d.model_id}")

    _loader = Reporter()
    with _loader.step(f"loading {model} ({device}, {dtype})"):
        mdl, tok = load(model, device=device, dtype=dtype)

    if timeline:
        from vivasecuris.aiasylum.weights.timeline import refusal_timeline

        target = prompt or build_split(seed=0).harmful_test[0]
        tl = refusal_timeline(mdl, tok, target, d, max_new_tokens=max_new_tokens, thinking=thinking)
        click.echo(f"\nPrompt: {target}\nnormalisation: {tl['normalisation']}  "
                   f"decision at token {tl['decision_index']} "
                   f"({'inside' if tl['decision_in_think'] else 'outside'} <think>)  final: {tl['final_side']}")
        for k, (t, s_) in enumerate(zip(tl["tokens"], tl["score"])):
            bar = ("+" if s_ > 0 else "-") * min(30, int(abs(s_) * 10))
            click.echo(f"{k:4d} {'T' if tl['in_think'][k] else ' '} {s_:+6.2f} {bar:<30} {t!r}")
        if not (curve or sweep):
            return

    if curve:
        if d.rank < 2:
            raise click.ClickException("--curve needs a subspace direction (rank above 1).")
        prompts = [prompt] if prompt else build_split(seed=0).harmful_test[:n_prompts]
        rows = sweep_subspace_rank(mdl, tok, d, prompts, max_new_tokens=max_new_tokens, thinking=thinking)
        click.echo(f"\n{'directions removed':<20}{'refusal':>9}  {'factual':>8}  chart")
        for r in rows:
            bar = "#" * int(round(r["refusal_rate"] * 40))
            fac = f"{r['factual_acc']*100:>7.1f}%" if "factual_acc" in r else "       -"
            flag = "  <- degenerate" if r["degenerate"] else ""
            click.echo(f"{r['label']:<20}{r['refusal_rate']*100:>8.1f}%  {fac}  {bar}{flag}")
        cs = summarize_curve(rows)
        click.echo(f"\nSmallest rank reaching 50% compliance: {cs['k50_rank']}   "
                   f"max compliance {cs['max_compliance']} at rank {cs['rank_at_max']}   "
                   f"monotone: {cs['monotone']}")
        return

    if sweep:
        prompts = [prompt] if prompt else build_split(seed=0).harmful_test[:n_prompts]
        click.echo(f"\nSweeping over {len(prompts)} held-out harmful prompts...")
        click.echo(f"(addition at layer {d.layer} only, alpha as a multiple of that")
        click.echo(" layer's residual norm; ablation applies at every layer)\n")
        from vivasecuris.aiasylum.weights.progress import Reporter

        reporter = Reporter()
        current = {"label": None}

        def report(label, done, total):
            if label != current["label"]:
                current["label"] = label
                reporter.note(f"{label}")
            reporter.count(done, total)

        results = sweep_alpha(
            mdl, tok, d.vector, prompts, layer=d.layer,
            max_new_tokens=max_new_tokens, progress=report, thinking=thinking,
        )

        click.echo(f"\n{'intervention':<16}{'refusal':>9}  chart")
        for r in results:
            bar = "#" * int(round(r["refusal_rate"] * 40))
            flag = "  <- degenerate output" if r["degenerate"] else ""
            click.echo(f"{r['label']:<16}{r['refusal_rate']*100:>8.1f}%  {bar}{flag}")

        summary = summarize_sweep(results)
        if summary["ablate_delta_points"] is not None:
            click.echo(
                f"\nAblation moves refusal by {summary['ablate_delta_points']:+.1f} "
                f"points vs baseline."
            )
        if summary["verdict"] != "causal":
            click.echo(
                click.style(
                    "WARNING: ablation barely moves refusal. The direction does not look "
                    "causal for this model; weight surgery would likely fail silently.",
                    fg="yellow",
                )
            )
        return

    if alpha is None:
        raise click.ClickException("Pass --alpha, or --sweep to scan a range.")
    target = prompt or build_split(seed=0).harmful_test[0]
    click.echo(f"\nPrompt: {target}\n")
    for a in (0.0, alpha):
        label = "baseline" if a == 0.0 else f"alpha={a:+.1f}"
        text = generate_with_steering(mdl, tok, target, vector=d.vector, alpha=a, max_new_tokens=max_new_tokens)
        click.echo(f"--- {label} ---\n{text}\n")


@weights.command("ablate")
@click.option("--model", default=DEFAULT_MODEL, show_default=True, help="Source model to edit")
@click.option("--direction", "direction_path", required=True)
@click.option("--out", required=True, help="Directory for the modified model")
@click.option("--beta", type=float, default=0.0, show_default=True,
              help="0 ablates, 1 is a no-op, >1 amplifies")
@click.option("--no-embeddings", is_flag=True, help="Leave the embedding table untouched")
@click.option("--dtype", default="bfloat16", show_default=True)
@click.option("--notes", default=None, help="Free text recorded in the manifest")
@click.option("--subspace", is_flag=True,
              help="Remove the direction's whole orthonormal subspace (needs a subspace direction)")
@click.option("--k", type=float, default=None,
              help="Subspace removal strength (default 1.0; >1 over-projects). Subspace mode only")
@click.option("--rank", type=int, default=None,
              help="Use only the first N basis directions (e.g. the rank `weights select` recommends)")
def ablate(model, direction_path, out, beta, no_embeddings, dtype, notes, subspace, k, rank):
    """Write a permanently modified model to disk."""
    _require_interp()
    from vivasecuris.aiasylum.weights.direction import RefusalDirection
    from vivasecuris.aiasylum.weights.surgery import edit_and_save

    from vivasecuris.aiasylum.weights.progress import Reporter

    _warn_if_low_disk()
    _preflight()
    d = RefusalDirection.load(direction_path)
    click.echo(f"Direction: layer {d.layer}, AUC {d.auc:.3f}, d_model {d.vector.shape[0]}")

    if subspace and d.basis is None:
        raise click.ClickException(
            "--subspace was requested but this direction holds only a single vector. "
            "Derive one with:  aiasylum weights direction --subspace-rank <N> --out <dir>"
        )
    if rank is not None:
        if d.basis is None:
            raise click.ClickException("--rank needs a subspace direction; this one holds a single vector.")
        if not 1 <= rank <= d.rank:
            raise click.ClickException(f"--rank must be between 1 and {d.rank} for this direction.")
        # The basis is built incrementally from a fixed pool of best-separating
        # layers (row 0 is the best single direction, each later row an
        # orthogonal residual component), so the first N rows *are* the rank-N
        # subspace and no re-derivation is needed. Exact while the pool is the
        # same for both ranks, i.e. rank <= pool_layers; guarded by
        # test_subspace_prefix_matches_a_smaller_rank_derivation.
        d.basis = d.basis[:rank].contiguous()
        d.basis_layers = list(d.basis_layers[:rank])
    if subspace:
        strength = 1.0 if k is None else k
        click.echo(f"Removing rank-{d.rank} subspace with k={strength} (surgery runs on CPU)\n")
    else:
        click.echo(f"Editing {model} with beta={beta} (surgery runs on CPU)\n")

    reporter = Reporter()
    path = edit_and_save(
        source_model=model, direction=d, out_dir=out, beta=beta,
        device="cpu", dtype=dtype, include_embeddings=not no_embeddings, notes=notes,
        reporter=reporter, use_subspace=subspace, k=k,
    )
    click.echo(f"\nSaved to {path} in {reporter.total_elapsed()}")
    click.echo(f"Test it with:  aiasylum weights info --model {path}")
    click.echo(f"Measure it with:  aiasylum weights compare --modified {path}")


@weights.command("info")
@click.option("--model", required=True, help="A modified model directory")
def info(model):
    """Print the surgery manifest for a modified model."""
    from vivasecuris.aiasylum.weights.manifest import SurgeryManifest

    manifest = SurgeryManifest.load(model)
    if manifest is None:
        click.echo(f"No surgery manifest at {model} -- this looks like a stock model.")
        sys.exit(1)

    from dataclasses import asdict
    for key, value in asdict(manifest).items():
        if value not in (None, {}, ""):
            click.echo(f"{key:>24}: {value}")


@weights.command("routing")
@click.option("--model", default=DEFAULT_MODEL, show_default=True, help="A mixture-of-experts model id or local path")
@click.option("--out", required=True, help="Where to write routing.json")
@click.option("--n-per-class", default=64, show_default=True, help="Harmful and harmless prompts to run")
@click.option("--max-length", default=512, show_default=True)
@click.option("--top", default=20, show_default=True, help="Rows to print")
@click.option("--thinking", is_flag=True, help="Keep a reasoning model's <think> block on")
@click.option("--device", default="auto", show_default=True)
@click.option("--dtype", default="bfloat16", show_default=True)
def routing_cmd(model, out, n_per_class, max_length, top, thinking, device, dtype):
    """Which experts fire on harmful vs harmless prompts (mixture-of-experts models only)."""
    _require_interp()
    from pathlib import Path

    from vivasecuris.aiasylum.interp.core.loader import load
    from vivasecuris.aiasylum.weights.corpus import build_split
    from vivasecuris.aiasylum.weights.progress import Reporter
    from vivasecuris.aiasylum.weights.routing import routing_statistics, routing_table

    _preflight()
    split = build_split(n_per_class=n_per_class, test_fraction=0.25, seed=0)
    harmful = list(split.harmful_train) + list(split.harmful_test)
    harmless = list(split.harmless_train) + list(split.harmless_test)
    reporter = Reporter()
    with reporter.step(f"loading {model}"):
        mdl, tok = load(model, device=device, dtype=dtype)
    click.echo(f"Recording expert routing over {len(harmful)} harmful and {len(harmless)} harmless prompts")
    result = routing_statistics(
        mdl, tok, harmful, harmless, max_length=max_length, thinking=thinking,
        progress=lambda done, total: reporter.count(done, total, "prompts "),
    )
    result["split"] = split.summary()
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1))

    if not result["consistency"]["gate_vs_expert_counts_match"]:
        click.echo(click.style(
            "WARNING: the replayed router selection disagrees with the experts' own row "
            "counts; the fractions are unverified for this model family.", fg="yellow",
        ))
    click.echo(f"\n{'layer':>5} {'expert':>6} {'harmful':>8} {'harmless':>8} {'delta':>7} {'last':>7}")
    for row in routing_table(result, top=top):
        click.echo(
            f"{row['layer']:>5} {row['expert']:>6} {row['harmful_frac'] * 100:>7.1f}% "
            f"{row['harmless_frac'] * 100:>7.1f}% {row['delta'] * 100:>+7.1f} "
            f"{row['last_token_delta'] * 100:>+7.1f}"
        )
    click.echo(f"\nWrote {path} in {reporter.total_elapsed()}")
    click.echo("Edit the top rows with:  aiasylum weights experts --model ... --experts LAYER:E,E --ablate --out DIR")


def _parse_expert_args(specs) -> dict:
    """'12:3,7' / '15:all' (repeatable) -> {"12": [3, 7], "15": "all"}."""
    selection = {}
    for spec in specs:
        if ":" not in spec:
            raise click.ClickException(f"--experts expects LAYER:EXPERTS, got {spec!r}")
        layer, rhs = spec.split(":", 1)
        layer, rhs = layer.strip(), rhs.strip()
        if not layer.isdigit():
            raise click.ClickException(f"--experts layer must be an integer, got {layer!r}")
        if rhs.lower() == "all":
            selection[layer] = "all"
            continue
        try:
            experts = sorted({int(e) for e in rhs.replace(" ", "").split(",") if e})
        except ValueError:
            raise click.ClickException(f"--experts {spec!r}: expert indices must be integers or 'all'")
        if not experts:
            raise click.ClickException(f"--experts {spec!r} names no experts")
        selection[layer] = experts
    if not selection:
        raise click.ClickException("--experts is required at least once, e.g. --experts 12:3,7")
    return selection


@weights.command("experts")
@click.option("--model", default=DEFAULT_MODEL, show_default=True, help="Mixture-of-experts model to edit")
@click.option("--out", required=True, help="Directory for the modified model")
@click.option("--experts", "expert_specs", multiple=True, required=True,
              help="LAYER:EXPERTS, repeatable, e.g. --experts 12:3,7 --experts 15:all")
@click.option("--ablate", is_flag=True,
              help="Scale the chosen experts' down-projections by --scale (needs no direction)")
@click.option("--scale", type=float, default=0.0, show_default=True,
              help="Scale for --ablate: 0 removes the experts' write, 1 is a no-op control, 2 doubles it")
@click.option("--direction", "direction_path", default=None,
              help="Direction directory for a direction or subspace edit inside the chosen experts")
@click.option("--beta", type=float, default=0.0, show_default=True,
              help="Direction edit: 0 ablates, 1 is a no-op, >1 amplifies")
@click.option("--subspace", is_flag=True, help="Remove the direction's whole subspace inside the chosen experts")
@click.option("--k", type=float, default=None, help="Subspace removal strength (default 1.0)")
@click.option("--include-shared", is_flag=True, help="Also edit each chosen layer's shared expert")
@click.option("--dtype", default="bfloat16", show_default=True)
@click.option("--notes", default=None, help="Free text recorded in the manifest")
def experts_cmd(model, out, expert_specs, ablate, scale, direction_path, beta, subspace, k,
                include_shared, dtype, notes):
    """Edit specific experts of specific layers and leave everything else alone.

    Partial by design: the manifest records coverage_verified=False, and the
    untouched experts still write whatever they carry whenever the router picks
    them. Use `weights routing` first to see which experts to name.
    """
    _require_interp()
    from vivasecuris.aiasylum.weights.progress import Reporter
    from vivasecuris.aiasylum.weights.surgery import edit_and_save

    selection = _parse_expert_args(expert_specs)
    if ablate and direction_path:
        raise click.ClickException(
            "--ablate scales whole down-projections; drop --direction, or drop --ablate for a direction edit."
        )
    if not ablate and not direction_path:
        raise click.ClickException("Pass --ablate (no direction needed) or --direction DIR for a direction edit.")

    d = None
    mode = "ablate"
    if direction_path:
        from vivasecuris.aiasylum.weights.direction import RefusalDirection

        d = RefusalDirection.load(direction_path)
        click.echo(f"Direction: layer {d.layer}, AUC {d.auc:.3f}, d_model {d.vector.shape[0]}")
        if subspace and d.basis is None:
            raise click.ClickException("--subspace needs a subspace direction; this one holds a single vector.")
        mode = "subspace" if subspace else "direction"

    _warn_if_low_disk()
    _preflight()
    click.echo(f"Editing {model}: {mode} edit in experts {selection} (surgery runs on CPU)\n")
    reporter = Reporter()
    path = edit_and_save(
        source_model=model, direction=d, out_dir=out, beta=beta, device="cpu", dtype=dtype,
        notes=notes, reporter=reporter, use_subspace=subspace, k=k,
        expert_selection=selection, expert_mode=mode, expert_scale=scale, include_shared=include_shared,
    )
    click.echo(f"\nSaved to {path} in {reporter.total_elapsed()}")
    click.echo(f"Inspect it with:  aiasylum weights info --model {path}")
    click.echo(f"Measure it with:  aiasylum weights compare --modified {path}")


# --------------------------------------------------------------------------
# Training: LoRA and distillation
# --------------------------------------------------------------------------
#
# Both commands build the same job the API hands to weights/train_worker.py and
# run it in this process (the CLI is already its own process, so the worker's
# isolation buys nothing here). The worker's event stream is rendered by the
# ordinary Reporter instead of being serialised.

_TRAIN_OPTIONS = [
    click.option("--rank", default=8, show_default=True, help="LoRA rank"),
    click.option("--alpha", default=16, show_default=True, help="LoRA alpha (scale is alpha / rank)"),
    click.option("--dropout", default=0.05, show_default=True),
    click.option("--targets", default="attention", show_default=True,
                 help="attention | attention+mlp | comma-separated module names"),
    click.option("--epochs", default=1, show_default=True),
    click.option("--max-steps", type=int, default=None, help="Stop after this many optimizer steps"),
    click.option("--lr", default=2e-4, show_default=True),
    click.option("--batch-size", default=1, show_default=True),
    click.option("--grad-accum", default=8, show_default=True),
    click.option("--max-length", default=512, show_default=True),
    click.option("--eval-rows", default=32, show_default=True, help="Rows held out for the eval loss"),
    click.option("--no-merge", is_flag=True, help="Keep only the adapter; write no merged model"),
    click.option("--gradient-checkpointing", is_flag=True),
    click.option("--device", default="auto", show_default=True),
    click.option("--dtype", default="bfloat16", show_default=True),
    click.option("--seed", default=0, show_default=True),
    click.option("--notes", default=None, help="Free text recorded in the manifest"),
]


def _train_options(f):
    for option in reversed(_TRAIN_OPTIONS):
        f = option(f)
    return f


class _TerminalEmitter:
    """The worker's event stream, rendered through a Reporter."""

    def __init__(self, reporter):
        self.reporter = reporter

    def note(self, message: str) -> None:
        self.reporter.note(message)

    def __call__(self, event: dict) -> None:
        kind = event.get("event")
        if kind == "note":
            self.reporter.note(str(event.get("message", "")))
        elif kind == "step":
            self.reporter.metrics({k: v for k, v in event.items() if k != "event"})
        elif kind == "eval":
            self.reporter.metrics({"phase": "eval", "step": event.get("step"), "eval_loss": event.get("eval_loss")})
        elif kind == "count":
            self.reporter.count(int(event.get("done", 0)), int(event.get("total", 0)), "teacher ")


def _train_run_dir(out: str, no_merge: bool):
    from pathlib import Path

    out_path = Path(out)
    # The adapter, the step log and any teacher responses sit beside the merged
    # model rather than inside it, so the model directory stays a plain checkpoint.
    return out_path if no_merge else out_path.parent / f"{out_path.name}-train"


def _run_training(kind, model, dataset_path, out, distill, *, rank, alpha, dropout, targets, epochs,
                  max_steps, lr, batch_size, grad_accum, max_length, eval_rows, no_merge,
                  gradient_checkpointing, device, dtype, seed, notes):
    import os

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    _require_interp()
    try:
        import peft  # noqa: F401
    except ImportError:
        raise click.ClickException(
            "Training needs the optional lora extra. Install it with:\n"
            '    pip install -e ".[lora]"'
        )
    from vivasecuris.aiasylum.weights.progress import Reporter
    from vivasecuris.aiasylum.weights.train_worker import run_job

    run_dir = _train_run_dir(out, no_merge)
    job = {
        "kind": kind, "source_model": model, "dataset_path": str(dataset_path),
        "run_dir": str(run_dir), "out_dir": None if no_merge else str(out),
        "device": device, "dtype": dtype,
        "lora": {
            "rank": rank, "alpha": alpha, "dropout": dropout, "targets": targets, "epochs": epochs,
            "max_steps": max_steps, "lr": lr, "batch_size": batch_size, "grad_accum": grad_accum,
            "max_length": max_length, "eval_rows": eval_rows, "seed": seed,
            "gradient_checkpointing": gradient_checkpointing, "merge": not no_merge,
        },
        "notes": notes,
        "manifest_extra": {
            "method": "lora" if kind == "lora" else f"{distill['level']}_distill",
            "objective": "dataset",
        },
        "distill": distill,
    }
    _warn_if_low_disk()
    _preflight()
    reporter = Reporter()
    try:
        result = run_job(job, _TerminalEmitter(reporter))
    except Exception as exc:
        raise click.ClickException(f"{type(exc).__name__}: {exc}")

    train = result.get("train") or {}
    click.echo(
        f"\nTrained {train.get('steps')} steps in {reporter.total_elapsed()}: final loss "
        f"{train.get('final_loss')}, eval loss {train.get('eval_loss_before')} -> {train.get('eval_loss_after')}"
    )
    click.echo(f"Adapter: {result.get('adapter_path')}")
    if result.get("merged"):
        click.echo(f"Merged model: {result['output_path']}")
        click.echo(f"Inspect it with:  aiasylum weights info --model {result['output_path']}")
        click.echo(f"Measure it with:  aiasylum weights compare --modified {result['output_path']}")
    return result


@weights.command("lora")
@click.option("--model", default=DEFAULT_MODEL, show_default=True, help="Student model to fine-tune")
@click.option("--dataset", "dataset_path", required=True, help="JSONL of {prompt, response[, system]} rows")
@click.option("--out", required=True, help="Directory for the merged model (with --no-merge: for the adapter)")
@_train_options
def lora_cmd(model, dataset_path, out, **opts):
    """Fine-tune a LoRA adapter on prompt/response rows and merge it into a new model.

    The loss is masked to the response, so the model learns to answer rather
    than to repeat the question. Plain precision: no QLoRA on Apple silicon.
    """
    _run_training("lora", model, dataset_path, out, None, **opts)


@weights.command("distill")
@click.option("--student", "model", default="Qwen/Qwen2.5-0.5B-Instruct", show_default=True,
              help="The model that learns")
@click.option("--teacher", required=True,
              help="A local open-weights teacher the loader can load (Ollama names are refused)")
@click.option("--level", default="response", show_default=True, type=click.Choice(["response", "logit"]),
              help="response: train on the teacher's text; logit: match its token distribution")
@click.option("--prompts", "prompts_path", default=None, help="JSONL of {prompt[, response]} rows")
@click.option("--objective", default=None, type=click.Choice(["refusal"]),
              help="Take prompts from the refusal corpus instead of --prompts")
@click.option("--n-per-class", default=64, show_default=True, help="With --objective: prompts per class")
@click.option("--out", required=True, help="Directory for the merged student (with --no-merge: for the adapter)")
@click.option("--temperature", default=2.0, show_default=True, help="Softening temperature (logit level)")
@click.option("--ce-weight", default=0.5, show_default=True,
              help="Weight of plain cross-entropy against the KL term (logit level)")
@click.option("--teacher-max-new-tokens", default=256, show_default=True)
@click.option("--teacher-system-prompt", default=None)
@_train_options
def distill_cmd(model, teacher, level, prompts_path, objective, n_per_class, out, temperature,
                ce_weight, teacher_max_new_tokens, teacher_system_prompt, **opts):
    """Distil a local teacher into a smaller student through a LoRA adapter.

    Response level asks the teacher for an answer to every prompt, unloads it,
    and trains the student on those answers. Logit level keeps both resident
    and matches the teacher's full next-token distribution.
    """
    if bool(prompts_path) == bool(objective):
        raise click.ClickException("Pass exactly one of --prompts or --objective.")
    dataset_path = prompts_path
    if objective:
        from vivasecuris.aiasylum.weights.corpus import build_split
        from vivasecuris.aiasylum.weights.train_data import rows_from_prompts, write_jsonl

        run_dir = _train_run_dir(out, opts["no_merge"])
        run_dir.mkdir(parents=True, exist_ok=True)
        split = build_split(n_per_class=n_per_class, test_fraction=0.25, seed=opts["seed"])
        prompts = (list(split.harmful_train) + list(split.harmful_test)
                   + list(split.harmless_train) + list(split.harmless_test))
        dataset_path = run_dir / "train.jsonl"
        write_jsonl(dataset_path, rows_from_prompts(prompts))
        click.echo(f"Wrote {len(prompts)} prompts from the {objective} corpus to {dataset_path}")
    distill = {
        "teacher_model": teacher, "level": level, "temperature": temperature, "ce_weight": ce_weight,
        "teacher_max_new_tokens": teacher_max_new_tokens, "teacher_system_prompt": teacher_system_prompt,
    }
    _run_training("distill", model, dataset_path, out, distill, **opts)


@weights.command("select")
@click.option("--model", default=DEFAULT_MODEL, show_default=True, help="Source model to edit")
@click.option("--direction", "direction_path", required=True,
              help="A subspace direction directory (from `weights direction --subspace-rank N`)")
@click.option("--n-prompts", default=32, show_default=True, help="Held-out harmful prompts to score per config")
@click.option("--ranks", default="1,2,3,4,6,8", show_default=True, help="Comma-separated subspace ranks to try")
@click.option("--ks", default="1.0,1.25,1.5", show_default=True, help="Comma-separated removal strengths to try")
@click.option("--factual-floor", default=0.05, show_default=True,
              help="Max allowed drop in factual accuracy vs baseline for a config to be admissible")
@click.option("--max-new-tokens", default=96, show_default=True)
@click.option("--capability-set", default="builtin", show_default=True,
              help="Capability control: 'builtin' (12 questions) or 'mmlu:<n>'")
@click.option("--device", default="auto", show_default=True)
@click.option("--dtype", default="bfloat16", show_default=True)
def select_cmd(model, direction_path, n_prompts, ranks, ks, factual_floor, max_new_tokens,
               capability_set, device, dtype):
    """Search subspace rank x strength for the most-compliant capability-safe edit.

    Previews every candidate at inference time (no weights written) and keeps
    only those that hold the factual capability control, then prints the frontier
    (the "what 100% costs" curve) and the recommended config.
    """
    _require_interp()
    from vivasecuris.aiasylum.interp.core.loader import load
    from vivasecuris.aiasylum.weights.corpus import build_split
    from vivasecuris.aiasylum.weights.direction import RefusalDirection
    from vivasecuris.aiasylum.weights.progress import Reporter
    from vivasecuris.aiasylum.weights.surgery import select_edit

    _preflight()
    d = RefusalDirection.load(direction_path)
    if d.basis is None:
        raise click.ClickException(
            "This direction holds only a single vector; there is no subspace to search. "
            "Derive one with:  aiasylum weights direction --subspace-rank <N> --out <dir>"
        )
    click.echo(f"Subspace: rank {d.rank} from layers {d.basis_layers}, from {d.model_id}")

    rank_list = tuple(int(x) for x in ranks.split(",") if x.strip())
    k_list = tuple(float(x) for x in ks.split(",") if x.strip())
    prompts = build_split(seed=0).harmful_test[:n_prompts]

    reporter = Reporter()
    with reporter.step(f"loading {model} ({device}, {dtype})"):
        mdl, tok = load(model, device=device, dtype=dtype)

    current = {"msg": None}

    def report(msg):
        if msg != current["msg"]:
            current["msg"] = msg
            reporter.note(msg)

    from vivasecuris.aiasylum.weights.evaluate import capability_set as _capability_set

    result = select_edit(
        mdl, tok, d, prompts, ranks=rank_list, ks=k_list,
        factual_floor=factual_floor, max_new_tokens=max_new_tokens, progress=report,
        capability=_capability_set(capability_set),
    )

    base = result["baseline"]
    click.echo(
        f"\nBaseline (no edit): refuse {base['refuse_harmful']*100:.1f}%, "
        f"factual {base['factual_acc']*100:.1f}%  (floor: factual >= "
        f"{(base['factual_acc']-factual_floor)*100:.1f}%)"
    )
    click.echo(f"\n{'rank':>4}{'k':>7}{'refuse':>9}{'factual':>9}{'admissible':>12}")
    for row in result["frontier"]:
        mark = "yes" if row["accepted"] else ("DEGENERATE" if row["degenerate"] else "no")
        click.echo(
            f"{row['rank']:>4}{row['k']:>7.2f}{row['refuse_harmful']*100:>8.1f}%"
            f"{row['factual_acc']*100:>8.1f}%{mark:>12}"
        )

    best = result["best"]
    if best is None:
        click.echo(
            "\nNo config cleared the capability floor. The remaining refusals on this "
            "model appear entangled with general capability: reaching them costs more "
            "factual accuracy than --factual-floor allows. Raise the floor knowingly, or "
            "accept the shipped single-direction edit."
        )
        return
    click.echo(
        f"\nRecommended: rank {best['rank']}, k {best['k']:.2f}  -> refuse "
        f"{best['refuse_harmful']*100:.1f}% (factual {best['factual_acc']*100:.1f}%, "
        f"drop {best['factual_drop']*100:+.1f} pts)"
    )
    click.echo(
        f"Write it with:  aiasylum weights ablate --model {model} "
        f"--direction {direction_path} --subspace --rank {best['rank']} --k {best['k']:.2f} "
        f"--out models/<name>/"
    )


@weights.command("compare")
@click.option("--baseline", default=DEFAULT_MODEL, show_default=True, help="Stock model (HF id or path)")
@click.option("--modified", required=True, help="Edited model directory")
@click.option("--n-prompts", default=32, show_default=True, help="Held-out harmful/harmless prompts per class")
@click.option("--max-new-tokens", default=96, show_default=True)
@click.option("--capability-set", default="builtin", show_default=True,
              help="Capability control: 'builtin' (12 questions) or 'mmlu:<n>'")
@click.option("--device", default="auto", show_default=True)
@click.option("--dtype", default="bfloat16", show_default=True)
@click.option("--out", default=None, help="Write the full report as JSON here")
@click.option("--misalignment", is_flag=True,
              help="Also run the broad-misalignment control (open-ended probes, heuristic judge)")
def compare(baseline, modified, n_prompts, max_new_tokens, capability_set, device, dtype, out, misalignment):
    """Compare two models on refusal + capability through the same loader.

    Baseline and modified run through the identical loader, tokenizer and greedy
    decoding, so the weights are the only variable. Reports refuse-harmful,
    false-refuse-harmless and the factual capability control, with deltas.
    """
    _require_interp()
    import gc
    import json as _json

    from vivasecuris.aiasylum.interp.core.loader import load
    from vivasecuris.aiasylum.weights.corpus import build_split
    from vivasecuris.aiasylum.weights.evaluate import (
        capability_set as _capability_set, compare_verdict, generate_greedy,
    )
    from vivasecuris.aiasylum.weights.manifest import SurgeryManifest
    from vivasecuris.aiasylum.weights.progress import Reporter
    from vivasecuris.aiasylum.weights.steering import refusal_rate, _looks_degenerate

    _preflight()
    split = build_split(seed=0)
    harmful = split.harmful_test[:n_prompts]
    harmless = split.harmless_test[:n_prompts]
    capability = _capability_set(capability_set)
    factual_qs = capability.questions

    man = SurgeryManifest.load(modified)
    if man is not None:
        click.echo(f"Modified model manifest: method={man.method}, "
                   f"beta={man.beta}, extra={man.extra or '{}'}")

    metrics = {}
    for label, model_id in (("baseline", baseline), ("modified", modified)):
        reporter = Reporter()
        with reporter.step(f"loading {label}: {model_id} ({device}, {dtype})"):
            mdl, tok = load(model_id, device=device, dtype=dtype)
        harm = generate_greedy(mdl, tok, harmful, max_new_tokens=max_new_tokens)
        harmless_r = generate_greedy(mdl, tok, harmless, max_new_tokens=max_new_tokens)
        fac = generate_greedy(mdl, tok, factual_qs, max_new_tokens=capability.max_new_tokens)
        metrics[label] = {
            "refuse_harmful": refusal_rate(harm),
            "refuse_harmless": refusal_rate(harmless_r),
            "factual_acc": capability.score(fac),
            "degenerate": bool(_looks_degenerate(harm) or _looks_degenerate(fac)),
            "responses": {"harmful": harm, "harmless": harmless_r, "factual": fac},
        }
        if misalignment:
            from vivasecuris.aiasylum.weights.misalignment import MISALIGNMENT_PROBES, misalignment_rate

            mis = generate_greedy(mdl, tok, list(MISALIGNMENT_PROBES), max_new_tokens=96)
            metrics[label]["misalignment_rate"] = misalignment_rate(mis, list(MISALIGNMENT_PROBES))["rate"]
            metrics[label]["responses"]["misalignment"] = mis
        click.echo(f"  {label} done in {reporter.total_elapsed()}")
        del mdl, tok
        gc.collect()

    b, m = metrics["baseline"], metrics["modified"]
    click.echo(f"\ncapability control: {capability.name} ({capability.size} items); "
               f"verdict: {compare_verdict(b, m)}")
    click.echo("\n" + "=" * 64)
    click.echo(f"{'model':<12}{'refuse harmful':>16}{'refuse harmless':>17}{'factual':>10}")
    for label in ("baseline", "modified"):
        x = metrics[label]
        click.echo(f"{label:<12}{x['refuse_harmful']*100:>15.1f}%"
                   f"{x['refuse_harmless']*100:>16.1f}%{x['factual_acc']*100:>9.1f}%")
    click.echo(f"{'delta':<12}{(m['refuse_harmful']-b['refuse_harmful'])*100:>15.1f} "
               f"{'':>15}{(m['factual_acc']-b['factual_acc'])*100:>9.1f}")
    if misalignment:
        click.echo(f"\nbroad misalignment (heuristic judge): baseline {b['misalignment_rate']*100:.1f}%  "
                   f"modified {m['misalignment_rate']*100:.1f}%")
    if m["degenerate"]:
        click.echo("\nWARNING: modified model looks degenerate -- its 'compliance' may just "
                   "be incoherence. Treat the refusal number as unreliable.")

    if out:
        payload = {
            "baseline_model": baseline, "modified_model": modified,
            "max_new_tokens": max_new_tokens, "device": device, "dtype": dtype,
            "prompts": {"harmful": harmful, "harmless": harmless, "factual": factual_qs},
            "metrics": {k: {kk: vv for kk, vv in v.items() if kk != "responses"}
                        for k, v in metrics.items()},
            "responses": {k: v["responses"] for k, v in metrics.items()},
        }
        with open(out, "w") as fh:
            _json.dump(payload, fh, indent=2)
        click.echo(f"\nWrote report to {out}")


@weights.command("probe")
@click.option("--model", default=DEFAULT_MODEL, show_default=True, help="Model to fit the monitor on")
@click.option("--out", required=True, help="Directory to write probes.npz into")
@click.option("--pooling", default="mean", show_default=True,
              type=click.Choice(["last", "mean", "max", "last_k"]),
              help="How prompt positions are combined; final-token-only probes fail in known ways")
@click.option("--elicit", is_flag=True,
              help="Prompted probe: append an eliciting question and read the answer position")
@click.option("--n-direct", default=120, show_default=True)
@click.option("--n-jailbreak", default=120, show_default=True)
@click.option("--n-benign", default=240, show_default=True)
@click.option("--holdout-techniques", default=2, show_default=True,
              help="Jailbreak families held out entirely, so the number is generalisation")
@click.option("--batch-size", default=8, show_default=True)
@click.option("--device", default="auto", show_default=True)
@click.option("--dtype", default="bfloat16", show_default=True)
def probe_cmd(model, out, pooling, elicit, n_direct, n_jailbreak, n_benign,
              holdout_techniques, batch_size, device, dtype):
    """Train a harmful-intent monitor on raw residual activations."""
    _require_interp()
    from vivasecuris.aiasylum.interp.core.loader import load
    from vivasecuris.aiasylum.interp.probes.dataset import ELICITING_SUFFIX, build_harmful_intent_dataset
    from vivasecuris.aiasylum.interp.probes.train import train_probes
    from vivasecuris.aiasylum.weights.capture import capture_pooled_residuals
    from vivasecuris.aiasylum.weights.progress import Reporter

    _preflight()
    ds = build_harmful_intent_dataset(
        n_direct=n_direct, n_jailbreak=n_jailbreak, n_benign=n_benign,
        holdout_techniques=holdout_techniques,
    )
    for k, v in ds.summary().items():
        click.echo(f"  {k}: {v}")

    reporter = Reporter()
    with reporter.step(f"loading {model} ({device}, {dtype})"):
        mdl, tok = load(model, device=device, dtype=dtype)
    suffix = ELICITING_SUFFIX if elicit else None

    with reporter.step(f"capturing {len(ds.train_prompts)} train prompts ({pooling})"):
        train_acts = capture_pooled_residuals(mdl, tok, ds.train_prompts, pooling=pooling,
                                              prompt_suffix=suffix, batch_size=batch_size)
    with reporter.step(f"capturing {len(ds.test_prompts)} held-out prompts"):
        test_acts = capture_pooled_residuals(mdl, tok, ds.test_prompts, pooling=pooling,
                                             prompt_suffix=suffix, batch_size=batch_size)

    ps = train_probes(train_acts, ds.train_labels, test_acts, ds.test_labels,
                      test_groups=ds.test_groups, model_id=model, pooling=pooling,
                      dataset_hash=ds.hash, prompt_suffix=suffix, dataset_summary=ds.summary())
    ps.save(out)

    click.echo(f"\nBest layer {ps.best_layer}: held-out AUROC {ps.best.auroc:.4f} "
               f"(shuffled-label null p95 {ps.best.null_auroc_p95:.3f}), ECE {ps.best.ece:.3f}")
    click.echo(f"\n{'layer':>6}{'AUROC':>9}{'null p95':>10}{'beats null':>12}")
    for p in sorted(ps.probes.values(), key=lambda p: p.layer):
        mark = "yes" if p.beats_null else "no"
        star = "  <- selected" if p.layer == ps.best_layer else ""
        click.echo(f"{p.layer:>6}{p.auroc:>9.4f}{p.null_auroc_p95:>10.3f}{mark:>12}{star}")
    if ps.group_auroc:
        click.echo("\nBy prompt family (held-out families are the honest test):")
        for g, v in sorted(ps.group_auroc.items()):
            click.echo(f"  {g:<28}{v:.4f}")
    click.echo(f"\nSaved to {out}")
    if not ps.usable:
        raise click.ClickException(
            f"AUROC {ps.best.auroc:.3f} against a {ps.best.null_auroc_p95:.3f} null is not a "
            f"usable monitor. More prompts, a different pooling, or --elicit may help."
        )


@weights.command("monitor")
@click.option("--model", default=DEFAULT_MODEL, show_default=True)
@click.option("--probe", "probe_path", required=True, help="Directory written by `weights probe`")
@click.option("--n-prompts", default=16, show_default=True, help="Held-out harmful prompts to audit")
@click.option("--max-new-tokens", default=96, show_default=True)
@click.option("--threshold", default=0.5, show_default=True, help="Probe score counted as 'registered harm'")
@click.option("--device", default="auto", show_default=True)
@click.option("--dtype", default="bfloat16", show_default=True)
def monitor_cmd(model, probe_path, n_prompts, max_new_tokens, threshold, device, dtype):
    """Audit a model: did it internally register harm, and did it comply anyway?"""
    _require_interp()
    from vivasecuris.aiasylum.interp.core.loader import load
    from vivasecuris.aiasylum.interp.probes.monitor import audit_responses, score_prompts
    from vivasecuris.aiasylum.interp.probes.train import ProbeSet
    from vivasecuris.aiasylum.weights.corpus import build_split
    from vivasecuris.aiasylum.weights.evaluate import generate_greedy
    from vivasecuris.aiasylum.weights.progress import Reporter

    _preflight()
    ps = ProbeSet.load(probe_path)
    click.echo(f"Probe: layer {ps.best_layer}, AUROC {ps.best.auroc:.3f}, "
               f"pooling {ps.pooling}, from {ps.model_id}")
    if ps.model_id != model:
        click.echo(click.style(
            f"WARNING: this probe was fitted on {ps.model_id}, not {model}. Probe weights "
            f"do not transfer between models; the scores below are not meaningful.",
            fg="yellow"))

    reporter = Reporter()
    with reporter.step(f"loading {model} ({device}, {dtype})"):
        mdl, tok = load(model, device=device, dtype=dtype)

    prompts = build_split(seed=0).harmful_test[:n_prompts]
    with reporter.step(f"generating {len(prompts)} responses"):
        responses = generate_greedy(mdl, tok, prompts, max_new_tokens=max_new_tokens)
    with reporter.step("scoring internal harm"):
        scores = score_prompts(mdl, tok, prompts, ps)

    audit = audit_responses(prompts, responses, scores, threshold=threshold)
    click.echo(f"\n{'outcome':<22}{'count':>7}")
    for name in ("knows_but_complies", "missed", "caught", "over_refusal"):
        click.echo(f"{name:<22}{audit['counts'].get(name, 0):>7}")
    click.echo(f"\nknows-but-complies rate: {audit['knows_but_complies_rate']*100:.1f}%")
    if audit["worst"]:
        click.echo("\nWorst failures (model registered harm, answered anyway):")
        for r in audit["worst"][:5]:
            click.echo(f"  [{r['harm_score']:.2f}] {r['prompt'][:70]}")
            click.echo(f"         -> {r['response'][:70]!r}")


@weights.command("chat")
@click.option("--model", required=True, help="Model directory (edited or stock) or HF id")
@click.option("--compare-with", default=None,
              help="Second model to answer every prompt from the identical history")
@click.option("--probe", "probe_path", default=None,
              help="Probe directory from `weights probe`, to score internal harm per turn")
@click.option("--system", "system_prompt", default=None, help="System prompt for the session")
@click.option("--temperature", default=0.0, show_default=True,
              help="0.0 = greedy, which is what every measurement in this pipeline uses")
@click.option("--max-tokens", default=256, show_default=True)
@click.option("--device", default="auto", show_default=True)
@click.option("--dtype", default="bfloat16", show_default=True)
@click.option("--single", default=None, help="Send one prompt, print the answer, exit")
@click.option("--save", "save_path", default=None, help="Write the transcript here on exit")
@click.option("--no-stream", is_flag=True, help="Wait for the whole completion instead of streaming")
@click.option("--quiet", is_flag=True, help="Hide the per-turn measurement line")
def chat(model, compare_with, probe_path, system_prompt, temperature, max_tokens,
         device, dtype, single, save_path, no_stream, quiet):
    """Talk to a local model, edited or stock, with every turn measured.

    Loads through the same ``transformers`` provider the test harness uses, so
    an edited model behaves exactly as it will in a run, with no quantization
    step to perturb the tensors you edited. The surgery manifest is printed
    first and recorded in the transcript, so a session can never be mistaken
    for one against the stock model.

    Each answer is scored for refusal with the project's own phrase list, so
    the numbers here agree with a sweep, a compare and a test run. With
    ``--probe`` the prompt is also scored for internal harm, which separates
    "the model did not register the request as harmful" from "it registered it
    and answered anyway".

    ``--compare-with`` drives a second model from the identical history and
    prints both answers, one prompt at a time. Both models stay resident, so
    it needs roughly twice the memory.

    Type ``/help`` for the command list.
    """
    _require_interp()
    try:  # arrow-key history and line editing, when the build has it
        import readline  # noqa: F401
    except ImportError:
        pass

    from vivasecuris.aiasylum.weights.chat import ChatSession, load_handle, make_probe_scorer

    _preflight()

    handles = [load_handle(model, "modified" if compare_with else "model",
                           temperature, max_tokens, device, dtype)]
    if compare_with:
        # Primary first: it owns the conversation history.
        handles.insert(0, load_handle(compare_with, "baseline",
                                      temperature, max_tokens, device, dtype))
        handles[1].label = "modified"

    probe_set = scorer = None
    if probe_path:
        try:
            probe_set, scorer = make_probe_scorer(probe_path, handles[-1], device, dtype)
            fitted_on = probe_set.model_id
            click.echo(click.style(
                f"[probe] layer {probe_set.best_layer}, AUROC {probe_set.best.auroc:.3f}, "
                f"pooling {probe_set.pooling}", fg="cyan"))
            if fitted_on != handles[-1].path:
                click.echo(click.style(
                    f"[probe] WARNING: fitted on {fitted_on}, not {handles[-1].path}. "
                    f"Probe weights index one layer of one network and do not transfer; "
                    f"the harm scores below are not meaningful.", fg="yellow"))
        except Exception as exc:
            click.echo(click.style(f"[probe] could not load: {exc}", fg="red"))

    for h in handles:
        _echo_provenance(h)
    if system_prompt:
        click.echo(f"[system] {system_prompt}")

    session = ChatSession(handles, system_prompt=system_prompt, max_tokens=max_tokens,
                          temperature=temperature, probe_set=probe_set, probe_scorer=scorer)

    if single is not None:
        _chat_turn(session, single, stream=not no_stream, quiet=quiet)
        if save_path:
            click.echo(f"\n[saved] {session.save(save_path)}")
        return

    click.echo("\nType a message, or /help for commands.")
    while True:
        try:
            line = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            click.echo("")
            break
        if not line:
            continue
        if line.startswith("/"):
            if _chat_command(session, line, save_path):
                break
            continue
        try:
            _chat_turn(session, line, stream=not no_stream, quiet=quiet)
        except KeyboardInterrupt:
            click.echo(click.style("\n[interrupted]", fg="yellow"))
        except Exception as exc:  # keep the session alive on a bad turn
            click.echo(click.style(f"\n[error] {exc}", fg="red"))

    _echo_stats(session)
    if save_path:
        click.echo(f"[saved] {session.save(save_path)}")


def _echo_provenance(handle) -> None:
    """Say plainly whether this is a modified model, and how it was modified."""
    if not handle.is_modified:
        click.echo(f"[{handle.label}] stock: {handle.path}")
        return
    m = handle.manifest
    bits = [f"method={m.method}"]
    if m.beta is not None:
        bits.append(f"beta={m.beta}")
    if m.extra:
        for key in ("subspace_rank", "k", "derivation"):
            if key in m.extra:
                bits.append(f"{key}={m.extra[key]}")
    if m.direction_layer is not None:
        bits.append(f"layer={m.direction_layer}")
    if m.direction_auc is not None:
        bits.append(f"auc={m.direction_auc:.3f}")
    click.echo(click.style(
        f"[{handle.label}] MODIFIED: {handle.path}\n"
        f"          {', '.join(bits)}\n"
        f"          source={m.source_model}", fg="yellow"))


def _metric_line(metrics) -> str:
    """One compact line per answer: what was measured, not what was said."""
    bits = ["refused" if metrics.refused else "complied"]
    if metrics.harm_score is not None:
        bits.append(f"harm {metrics.harm_score:.2f}")
    if metrics.outcome:
        bits.append(metrics.outcome.replace("_", " "))
    bits.append(f"{metrics.completion_chars} chars")
    if metrics.thinking_chars:
        bits.append(f"+{metrics.thinking_chars} thinking")
    bits.append(f"{metrics.elapsed_s:.1f}s")
    if metrics.truncated:
        bits.append("hit the token limit")
    return "  ".join(bits)


def _metric_colour(metrics) -> str:
    if metrics.outcome == "knows_but_complies":
        return "red"
    if metrics.truncated:
        return "yellow"
    return "bright_black"


def _chat_turn(session, message: str, stream: bool = True, quiet: bool = False) -> None:
    """One exchange: ask every model, print, measure, record."""
    import time

    harm = session._harm_score(message)

    if len(session.handles) > 1:
        # Compare mode never streams: two interleaved streams are unreadable,
        # and the point is to set the answers side by side.
        results = []
        for handle in session.handles:
            answer, metrics = session.ask(message, handle=handle)
            metrics.harm_score = harm
            if harm is not None:
                from vivasecuris.aiasylum.interp.probes.monitor import classify
                metrics.outcome = classify(harm, metrics.refused)
            results.append((handle, answer, metrics))
            colour = "green" if handle.label == "baseline" else "magenta"
            click.echo(click.style(f"\n{handle.label}>", fg=colour, bold=True) + f" {answer}")
            if not quiet:
                click.echo(click.style("  " + _metric_line(metrics), fg=_metric_colour(metrics)))
        # The primary (the modified model) owns the history; the other is an alternate.
        primary_handle, primary_answer, primary_metrics = results[-1]
        alternates = [
            {"label": h.label, "content": a, "metrics": _as_dict(m)}
            for h, a, m in results[:-1]
        ]
        session.commit_turn(message, primary_answer, primary_metrics, alternates=alternates)
        if not quiet and results[0][2].refused != results[-1][2].refused:
            click.echo(click.style(
                "  the two models disagree on whether to refuse this one", fg="cyan"))
        return

    handle = session.handles[0]
    if stream:
        click.echo(click.style("\nbot>", fg="cyan", bold=True) + " ", nl=False)
        pieces = []
        t0 = time.time()
        for piece in session.stream_turn(message, handle=handle):
            click.echo(piece, nl=False)
            pieces.append(piece)
        click.echo("")
        answer = "".join(pieces)
        metrics = session.measure(handle.label, message, answer, time.time() - t0, harm)
    else:
        answer, metrics = session.ask(message, handle=handle)
        metrics.harm_score = harm
        if harm is not None:
            from vivasecuris.aiasylum.interp.probes.monitor import classify
            metrics.outcome = classify(harm, metrics.refused)
        click.echo(click.style("\nbot>", fg="cyan", bold=True) + f" {answer}")

    if not quiet:
        click.echo(click.style("  " + _metric_line(metrics), fg=_metric_colour(metrics)))
    session.commit_turn(message, answer, metrics)


def _as_dict(metrics):
    from dataclasses import asdict

    return asdict(metrics)


CHAT_HELP = """
  /help              this list
  /exit  /quit       leave (also ctrl-d)
  /reset             clear the conversation history
  /system <text>     set the system prompt; empty clears it
  /temp <float>      change temperature; 0 is greedy
  /tokens <int>      change the completion limit
  /info              provenance and settings for every model in the session
  /stats             session totals so far
  /last              re-print the last answer in full
  /retry             ask the last question again, dropping the previous answer
  /save [path]       write the transcript as JSON
"""


def _chat_command(session, line: str, default_save: str = None) -> bool:
    """Handle a slash command. Returns True when the session should end."""
    parts = line.split(maxsplit=1)
    cmd, arg = parts[0], (parts[1].strip() if len(parts) > 1 else "")

    if cmd in ("/exit", "/quit"):
        return True
    if cmd == "/help":
        click.echo(CHAT_HELP)
    elif cmd == "/reset":
        session.reset()
        click.echo("[history cleared]")
    elif cmd == "/system":
        session.system_prompt = arg or None
        click.echo(f"[system] {session.system_prompt}")
    elif cmd == "/temp":
        try:
            session.temperature = float(arg)
            click.echo(f"[temperature] {session.temperature}"
                       + ("  (greedy)" if session.temperature == 0 else "  (sampled)"))
        except ValueError:
            click.echo(click.style("usage: /temp 0.7", fg="red"))
    elif cmd == "/tokens":
        try:
            session.max_tokens = int(arg)
            click.echo(f"[max tokens] {session.max_tokens}")
        except ValueError:
            click.echo(click.style("usage: /tokens 512", fg="red"))
    elif cmd == "/info":
        for h in session.handles:
            _echo_provenance(h)
        click.echo(f"[settings] temperature={session.temperature} "
                   f"max_tokens={session.max_tokens} system={session.system_prompt!r}")
        if session.probe_set is not None:
            click.echo(f"[probe] layer {session.probe_set.best_layer}, "
                       f"AUROC {session.probe_set.best.auroc:.3f}")
    elif cmd == "/stats":
        _echo_stats(session)
    elif cmd == "/last":
        answers = [t for t in session.turns if t.role == "assistant"]
        click.echo(answers[-1].content if answers else "[nothing yet]")
    elif cmd == "/retry":
        users = [t for t in session.turns if t.role == "user"]
        if not users:
            click.echo("[nothing to retry]")
        else:
            last = users[-1].content
            # Drop the previous exchange so the retry starts from the same history.
            while session.turns and session.turns[-1].role == "assistant":
                session.turns.pop()
            if session.turns and session.turns[-1].role == "user":
                session.turns.pop()
            _chat_turn(session, last)
    elif cmd == "/save":
        path = arg or default_save or "runs/chat/session.json"
        click.echo(f"[saved] {session.save(path)}")
    else:
        click.echo(click.style(f"unknown command {cmd}; /help for the list", fg="red"))
    return False


def _echo_stats(session) -> None:
    s = session.summary()
    if not s["turns"]:
        click.echo("[no turns yet]")
        return
    click.echo(f"\n[session] {s['turns']} turns, refused {s['refused']} "
               f"({s['refusal_rate']*100:.0f}%)")
    if s["mean_harm_score"] is not None:
        click.echo(f"[session] mean internal harm score {s['mean_harm_score']:.2f}, "
                   f"knows-but-complies {s['knows_but_complies']}")
    if s["truncated"]:
        click.echo(f"[session] {s['truncated']} answers hit the token limit")
