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
def derive(model, out, n_per_class, test_fraction, seed, batch_size, device, dtype, subspace_rank, pool_layers):
    """Derive a refusal direction (or subspace) by difference-in-means."""
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

    if subspace_rank > 1:
        direction = derive_subspace(
            mdl, tok, split, rank=subspace_rank, pool_layers=pool_layers,
            model_id=model, batch_size=batch_size, progress=report,
        )
    else:
        direction = derive_direction(
            mdl, tok, split, model_id=model, batch_size=batch_size, progress=report,
        )
    click.echo(f"\nTotal elapsed: {reporter.total_elapsed()}")

    click.echo(f"\nBest layer: {direction.layer}")
    click.echo(f"Held-out AUC: {direction.auc:.4f}   Cohen's d: {direction.cohens_d:.2f}")
    if direction.basis is not None:
        click.echo(f"Refusal subspace: rank {direction.rank} from layers {direction.basis_layers}")

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
def steer_cmd(model, direction_path, alpha, sweep, prompt, n_prompts, max_new_tokens, device, dtype):
    """Steer activations at inference time. The causal check before surgery."""
    _require_interp()
    from vivasecuris.aiasylum.interp.core.loader import load
    from vivasecuris.aiasylum.weights.corpus import build_split
    from vivasecuris.aiasylum.weights.direction import RefusalDirection
    from vivasecuris.aiasylum.weights.steering import (
        generate_with_steering,
        summarize_sweep,
        sweep_alpha,
    )

    from vivasecuris.aiasylum.weights.progress import Reporter

    _preflight()
    d = RefusalDirection.load(direction_path)
    click.echo(f"Direction: layer {d.layer}, AUC {d.auc:.3f}, from {d.model_id}")

    _loader = Reporter()
    with _loader.step(f"loading {model} ({device}, {dtype})"):
        mdl, tok = load(model, device=device, dtype=dtype)

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
            max_new_tokens=max_new_tokens, progress=report,
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
@click.option("--device", default="auto", show_default=True)
@click.option("--dtype", default="bfloat16", show_default=True)
def select_cmd(model, direction_path, n_prompts, ranks, ks, factual_floor, max_new_tokens, device, dtype):
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

    result = select_edit(
        mdl, tok, d, prompts, ranks=rank_list, ks=k_list,
        factual_floor=factual_floor, max_new_tokens=max_new_tokens, progress=report,
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
@click.option("--device", default="auto", show_default=True)
@click.option("--dtype", default="bfloat16", show_default=True)
@click.option("--out", default=None, help="Write the full report as JSON here")
def compare(baseline, modified, n_prompts, max_new_tokens, device, dtype, out):
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
        capability_questions, factual_accuracy, generate_greedy,
    )
    from vivasecuris.aiasylum.weights.manifest import SurgeryManifest
    from vivasecuris.aiasylum.weights.progress import Reporter
    from vivasecuris.aiasylum.weights.steering import refusal_rate, _looks_degenerate

    _preflight()
    split = build_split(seed=0)
    harmful = split.harmful_test[:n_prompts]
    harmless = split.harmless_test[:n_prompts]
    factual_qs = capability_questions()

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
        fac = generate_greedy(mdl, tok, factual_qs, max_new_tokens=32)
        metrics[label] = {
            "refuse_harmful": refusal_rate(harm),
            "refuse_harmless": refusal_rate(harmless_r),
            "factual_acc": factual_accuracy(fac),
            "degenerate": bool(_looks_degenerate(harm) or _looks_degenerate(fac)),
            "responses": {"harmful": harm, "harmless": harmless_r, "factual": fac},
        }
        click.echo(f"  {label} done in {reporter.total_elapsed()}")
        del mdl, tok
        gc.collect()

    b, m = metrics["baseline"], metrics["modified"]
    click.echo("\n" + "=" * 64)
    click.echo(f"{'model':<12}{'refuse harmful':>16}{'refuse harmless':>17}{'factual':>10}")
    for label in ("baseline", "modified"):
        x = metrics[label]
        click.echo(f"{label:<12}{x['refuse_harmful']*100:>15.1f}%"
                   f"{x['refuse_harmless']*100:>16.1f}%{x['factual_acc']*100:>9.1f}%")
    click.echo(f"{'delta':<12}{(m['refuse_harmful']-b['refuse_harmful'])*100:>15.1f} "
               f"{'':>15}{(m['factual_acc']-b['factual_acc'])*100:>9.1f}")
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


@weights.command("chat")
@click.option("--model", required=True, help="Model directory (edited or stock) or HF id")
@click.option("--system", "system_prompt", default=None, help="System prompt to set for the session")
@click.option("--temperature", default=0.0, show_default=True, help="0.0 = greedy (faithful to the surgery measurements)")
@click.option("--max-tokens", default=256, show_default=True)
@click.option("--device", default="auto", show_default=True)
@click.option("--dtype", default="bfloat16", show_default=True)
@click.option("--single", default=None, help="Send one prompt and exit instead of an interactive loop")
def chat(model, system_prompt, temperature, max_tokens, device, dtype, single):
    """Interactively chat with a local model, edited or stock.

    Loads the directory through the same ``transformers`` provider the test
    harness uses -- so an edited model behaves exactly as it will in a run, with
    no GGUF/quantization step to perturb the tensors you edited. Prints the
    surgery manifest first so you know whether you are talking to a modified
    model. Type ``/system <text>`` to change the system prompt, ``/reset`` to
    clear history, ``/exit`` to quit.
    """
    _require_interp()
    import asyncio

    from vivasecuris.aiasylum.models import get_provider
    from vivasecuris.aiasylum.weights.manifest import SurgeryManifest

    _preflight()

    man = SurgeryManifest.load(model)
    if man is not None:
        extra = f", {man.extra}" if man.extra else ""
        click.echo(click.style(
            f"[modified model] method={man.method}, beta={man.beta}, "
            f"source={man.source_model}{extra}", fg="yellow"))
    else:
        click.echo(f"[stock model] {model}")
    if system_prompt:
        click.echo(f"[system] {system_prompt}")

    provider = get_provider("transformers")
    mdl = provider.create_model(
        model, temperature=temperature, max_tokens=max_tokens, device=device, dtype=dtype
    )

    def ask(history):
        resp = asyncio.run(mdl.generate(prompt="", system_prompt=system_prompt, messages=history))
        return resp.content

    if single is not None:
        history = [{"role": "user", "content": single}]
        click.echo(f"\n{ask(history)}")
        return

    click.echo("\nType a message (/system <text>, /reset, /exit):")
    history: list = []
    while True:
        try:
            line = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            click.echo("\nbye")
            break
        if not line:
            continue
        if line in ("/exit", "/quit"):
            break
        if line == "/reset":
            history = []
            click.echo("[history cleared]")
            continue
        if line.startswith("/system"):
            system_prompt = line[len("/system"):].strip() or None
            click.echo(f"[system set to] {system_prompt}")
            continue
        history.append({"role": "user", "content": line})
        try:
            answer = ask(history)
        except Exception as exc:  # keep the session alive on a bad turn
            click.echo(click.style(f"[error] {exc}", fg="red"))
            history.pop()
            continue
        history.append({"role": "assistant", "content": answer})
        click.echo(f"\nbot> {answer}")
