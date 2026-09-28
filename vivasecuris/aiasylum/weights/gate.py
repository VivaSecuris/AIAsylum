"""Conditional activation steering: a gate decides *when*, a direction decides *what*.

The refusal-removal pipeline removes a direction everywhere, unconditionally.
Adding a *targeted* behaviour -- refuse this category, abstain on questions the
model will get wrong -- needs the opposite: act only when the input is in scope,
and leave every other input untouched. That is conditional activation steering
(CAST, Lee et al. 2024): a *condition* model (here a trained probe) scores the
prompt, and only when it clears a threshold is the *behaviour* direction added
to the residual stream during generation.

Two passes, on purpose. One forward pass scores the prompt with the probe; if
it fires, generation runs with the steering hook installed, otherwise it runs
bare. When the gate does not fire, the hook is never installed, so the output
is bit-for-bit the stock model's -- and the only cost this control imposes on
out-of-scope traffic is the gate's false-positive rate, which the tuner
measures directly. An in-pass hook that consulted the probe mid-generation
would be cheaper but would entangle the KV cache and the layer order; the
capability is not worth that risk.

The amount added is ``alpha = m * (mu_pos - mu_neg)``, where the class-projection
gap is the one already stored on the derived direction. So ``m`` is "fractions
of the distance between the two classes", a scale that transfers across models
and layers, and ``m`` is searched smallest-first: precision before force.
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

logger = logging.getLogger(__name__)


def class_gap(direction) -> float:
    """``mu_pos - mu_neg``: the projection gap between the classes on ``direction``.

    Recorded by ``direction._projection_extra`` at derivation time as
    ``extra["projection_means"] = {"harmful", "harmless", ...}`` (the names are
    the derivation's two classes, whatever they mean for this direction). It is
    the natural unit for "how far to push a borderline prompt across the
    boundary". Falls back to 1.0 for a direction saved without the means, which
    makes ``m`` a raw activation-addition coefficient instead.
    """
    pm = (getattr(direction, "extra", None) or {}).get("projection_means") or {}
    if "harmful" in pm and "harmless" in pm:
        return float(pm["harmful"]) - float(pm["harmless"])
    logger.warning("Direction carries no projection_means; m is a raw coefficient, not a class fraction.")
    return 1.0


def alpha_for(direction, m: float) -> float:
    """Activation-addition coefficient for a class fraction ``m``."""
    return float(m) * class_gap(direction)


@contextmanager
def steer_context(model, direction, alpha: float, positions: str = "all"):
    """Install an ``h += alpha * r`` hook at the direction's own layer, or nothing.

    ``alpha == 0`` yields the bare model (no hook), so a candidate at ``m=0`` is
    the exact baseline. The vector is added at ``direction.layer`` -- the
    hidden-states index the direction was derived at -- which is where its
    class separation is sharpest.
    """
    if not alpha:
        with nullcontext():
            yield model
        return
    from vivasecuris.aiasylum.weights.steering import steer

    with steer(model, direction.vector, alpha=alpha, layers=[int(direction.layer)],
               positions=positions, mode="add"):
        yield model


def gate_scores(
    model,
    tokenizer,
    prompts: Sequence[str],
    probe_set,
    system_prompt: Optional[str] = None,
    batch_size: int = 8,
    max_length: int = 512,
    progress=None,
) -> List[float]:
    """Probe probability per prompt, captured the way the probe was trained.

    Thin wrapper over ``interp.probes.monitor.score_prompts`` so the gate reads
    the same internal signal the monitor does.
    """
    from vivasecuris.aiasylum.interp.probes.monitor import score_prompts

    return score_prompts(
        model, tokenizer, list(prompts), probe_set,
        batch_size=batch_size, max_length=max_length,
        system_prompt=system_prompt, progress=progress,
    )


@dataclass
class GatedSteer:
    """A tuned conditional-steering control: which probe, which direction, m, tau.

    Holds paths rather than tensors so it round-trips as a small JSON record and
    so a serving process re-loads the exact artifacts the tuner selected. Probe
    and direction must both have been produced on the model being served -- probe
    weights and a direction's layer index do not transfer between networks.
    """

    probe_dir: str
    direction_dir: str
    layer: int
    m: float
    tau: float
    gap: float
    model_id: str = "unknown"
    goal: str = "category"          # category | factual_abstain
    positions: str = "all"
    extra: dict = field(default_factory=dict)

    @property
    def alpha(self) -> float:
        return float(self.m) * float(self.gap)

    def metadata(self) -> dict:
        return {
            "probe_dir": self.probe_dir,
            "direction_dir": self.direction_dir,
            "layer": int(self.layer),
            "m": float(self.m),
            "tau": float(self.tau),
            "gap": float(self.gap),
            "alpha": self.alpha,
            "model_id": self.model_id,
            "goal": self.goal,
            "positions": self.positions,
            "extra": dict(self.extra),
        }

    def save(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "behaviour.json").write_text(json.dumps(self.metadata(), indent=2))
        logger.info("Saved gated behaviour to %s (m=%.3f, tau=%.3f, layer=%d)",
                    out, self.m, self.tau, self.layer)
        return out

    @classmethod
    def load(cls, path: str | Path) -> "GatedSteer":
        p = Path(path)
        meta_path = p / "behaviour.json" if p.is_dir() else p
        meta = json.loads(Path(meta_path).read_text())
        return cls(
            probe_dir=meta["probe_dir"],
            direction_dir=meta["direction_dir"],
            layer=int(meta["layer"]),
            m=float(meta["m"]),
            tau=float(meta["tau"]),
            gap=float(meta["gap"]),
            model_id=meta.get("model_id", "unknown"),
            goal=meta.get("goal", "category"),
            positions=meta.get("positions", "all"),
            extra=dict(meta.get("extra") or {}),
        )

    def resolve(self):
        """Load the referenced ``(ProbeSet, RefusalDirection)`` from disk."""
        from vivasecuris.aiasylum.interp.probes.train import ProbeSet
        from vivasecuris.aiasylum.weights.direction import RefusalDirection

        return ProbeSet.load(self.probe_dir), RefusalDirection.load(self.direction_dir)


def generate_gated(
    model,
    tokenizer,
    prompts: Sequence[str],
    probe_set,
    direction,
    m: float,
    tau: float,
    *,
    system_prompt: Optional[str] = None,
    max_new_tokens: int = 96,
    sampling=None,
    batch_size: int = 8,
    max_length: int = 512,
    scores: Optional[Sequence[float]] = None,
    progress=None,
):
    """Score each prompt, then generate steered where ``p >= tau`` and bare elsewhere.

    Returns ``(responses, fired, scores)``. Pass ``scores`` to reuse gate
    probabilities computed earlier (they do not depend on ``m`` or ``tau``), so
    a sweep over thresholds never re-runs the probe.
    """
    from vivasecuris.aiasylum.weights.evaluate import _generate

    if scores is None:
        scores = gate_scores(
            model, tokenizer, prompts, probe_set,
            system_prompt=system_prompt, batch_size=batch_size, max_length=max_length,
        )
    scores = [float(s) for s in scores]
    fired = [s >= float(tau) for s in scores]
    alpha = alpha_for(direction, m)

    def context_for(i: int):
        return steer_context(model, direction, alpha) if fired[i] else nullcontext()

    responses = _generate(
        model, tokenizer, list(prompts),
        max_new_tokens=max_new_tokens, system_prompt=system_prompt,
        sampling=sampling, progress=progress, context_for=context_for,
    )
    return responses, fired, scores
