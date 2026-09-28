"""Interactive session with a local model, instrumented for audit.

A chat loop against an edited model is not a convenience feature here; it is
how you find out what an edit actually did. Aggregate refusal rates say a
surgery moved refusal by N points. They do not tell you whether the model now
answers in a different register, hedges differently, has lost the thread on
long turns, or refuses the same things for visibly different reasons. Reading
the generations is the only way to see that, and the operating guide already
says the phrase-matching detector is not a substitute for reading them.

So every turn is measured rather than merely printed:

- **refusal**, via the project's own ``REFUSAL_PHRASES``, so the number in a
  session agrees with the number in a sweep, a compare and a test run.
- **internal harm score**, when a probe is loaded: what the residual stream
  carried about the prompt. Paired with the refusal outcome it gives the
  knows-but-complies classification from ``interp.probes.monitor``.
- **provenance**, from the surgery manifest, so a transcript can never be
  mistaken for one produced by the stock model.
- **tokens and latency**, because a degenerate edit usually shows up first as
  a completion that runs to the token limit.

``compare`` mode drives two models from the identical history and prints both
answers. That is the comparison the numbers in ``weights compare`` summarise,
one prompt at a time.

The session object holds no terminal code so it can be tested directly; the
click command in ``weights/cli.py`` renders it.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TurnMetrics:
    """What was measured about one assistant turn."""

    label: str                      # "model", or "baseline"/"modified" in compare mode
    refused: bool
    elapsed_s: float
    completion_chars: int
    truncated: bool = False         # hit the token ceiling rather than stopping
    harm_score: Optional[float] = None
    outcome: Optional[str] = None   # knows_but_complies | caught | missed | over_refusal
    thinking_chars: int = 0         # length of a stripped <think> block, if any


@dataclass
class Turn:
    role: str
    content: str
    at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metrics: Optional[TurnMetrics] = None
    # In compare mode one user turn produces several assistant answers.
    alternates: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ModelHandle:
    """One model in the session, with its provenance."""

    label: str
    path: str
    model: Any
    manifest: Any = None

    @property
    def is_modified(self) -> bool:
        return self.manifest is not None

    def provenance(self) -> Dict[str, Any]:
        if self.manifest is None:
            return {"label": self.label, "path": self.path, "modified": False}
        m = self.manifest
        return {
            "label": self.label,
            "path": self.path,
            "modified": True,
            "method": m.method,
            "beta": m.beta,
            "source_model": m.source_model,
            "direction_layer": m.direction_layer,
            "direction_auc": m.direction_auc,
            "split_hash": m.split_hash,
            "extra": dict(m.extra or {}),
        }


class ChatSession:
    """History, settings and per-turn measurement for one or two local models."""

    def __init__(
        self,
        handles: List[ModelHandle],
        system_prompt: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
        probe_set: Any = None,
        probe_scorer: Optional[Callable[[str, Optional[str]], float]] = None,
    ):
        if not handles:
            raise ValueError("A session needs at least one model")
        self.handles = handles
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.probe_set = probe_set
        # Injected so the session can be tested without loading a model.
        self._probe_scorer = probe_scorer
        self.turns: List[Turn] = []
        self.started = datetime.now(timezone.utc).isoformat()

    # -- history ----------------------------------------------------------

    @property
    def history(self) -> List[Dict[str, str]]:
        """Message list for the provider: the primary model's thread.

        In compare mode the alternates deliberately do not enter the history.
        Two models answering the same question then diverging into two
        different conversations would stop being a comparison after one turn.
        """
        return [{"role": t.role, "content": t.content} for t in self.turns]

    def reset(self) -> None:
        self.turns = []

    # -- measurement ------------------------------------------------------

    def _harm_score(self, prompt: str) -> Optional[float]:
        """Internal harm score for ``prompt`` as this session serves it.

        The scorer gets the session's system prompt too, so the probe reads the
        same text the model generates from. Scoring the bare prompt would pair
        the answer with a forward pass the model never ran -- and if the system
        prompt *is* the jailbreak, that is exactly the pass that matters.
        """
        if self._probe_scorer is None:
            return None
        try:
            return float(self._probe_scorer(prompt, self.system_prompt))
        except Exception as exc:  # a broken probe must not end the session
            logger.warning("Probe scoring failed: %s", exc)
            return None

    def measure(self, label: str, prompt: str, answer: str, elapsed: float,
                harm_score: Optional[float] = None) -> TurnMetrics:
        """Score one answer. Shared by streaming and non-streaming paths."""
        from vivasecuris.aiasylum.weights.capture import strip_thinking
        from vivasecuris.aiasylum.weights.steering import refusal_rate

        # Only the answer counts as a refusal; a reasoning trace that muses
        # "I can't help" and then complies is compliance.
        visible = strip_thinking(answer)
        refused = refusal_rate([answer]) >= 1.0
        outcome = None
        if harm_score is not None:
            from vivasecuris.aiasylum.interp.probes.monitor import classify

            outcome = classify(harm_score, refused)
        return TurnMetrics(
            label=label,
            refused=refused,
            elapsed_s=round(elapsed, 2),
            completion_chars=len(visible),
            harm_score=harm_score,
            outcome=outcome,
            thinking_chars=max(0, len(answer) - len(visible)),
        )

    # -- turns ------------------------------------------------------------

    def stream_turn(self, message: str, handle: Optional[ModelHandle] = None) -> Iterator[str]:
        """Yield the answer piece by piece, as the model produces it.

        The provider's stream is an async iterator and this is a plain
        generator called from click, which has no running loop. So the loop
        runs on its own thread and hands pieces over through a queue; this
        side yields them the moment they land, which is what makes the output
        appear incrementally rather than all at once when generation ends.

        The caller prints the pieces and then calls ``commit_turn`` with the
        joined text. Keeping those separate lets the renderer own the display
        of a stream it is already consuming.
        """
        import asyncio
        import queue as _queue
        import threading

        handle = handle or self.handles[0]
        history = self.history + [{"role": "user", "content": message}]
        q: "_queue.Queue" = _queue.Queue()

        async def _pump():
            agen = handle.model.stream_generate(
                prompt="", system_prompt=self.system_prompt, messages=history,
                max_tokens=self.max_tokens, temperature=self.temperature,
            )
            async for piece in agen:
                q.put(piece)

        def _run():
            try:
                asyncio.run(_pump())
            except BaseException as exc:
                q.put(exc)
            finally:
                q.put(None)

        worker = threading.Thread(target=_run, daemon=True)
        worker.start()
        try:
            while True:
                piece = q.get()
                if piece is None:
                    break
                if isinstance(piece, BaseException):
                    raise piece
                yield piece
        finally:
            worker.join(timeout=5)

    def ask(self, message: str, handle: Optional[ModelHandle] = None) -> tuple:
        """One answer plus its metrics, without recording anything."""
        import asyncio

        handle = handle or self.handles[0]
        history = self.history + [{"role": "user", "content": message}]
        harm = self._harm_score(message)
        t0 = time.time()
        response = asyncio.run(handle.model.generate(
            prompt="", system_prompt=self.system_prompt, messages=history,
            max_tokens=self.max_tokens, temperature=self.temperature,
        ))
        metrics = self.measure(handle.label, message, response.content, time.time() - t0, harm)
        metrics.truncated = getattr(response, "finish_reason", None) == "length"
        # The provider already split the trace off the answer; the answer the
        # session sees carries no <think> block, so the count comes from the record.
        trace = (getattr(response, "metadata", None) or {}).get("reasoning")
        if trace and not metrics.thinking_chars:
            metrics.thinking_chars = len(trace)
        return response.content, metrics

    def commit_turn(self, message: str, answer: str, metrics: TurnMetrics,
                    alternates: Optional[List[Dict[str, Any]]] = None) -> Turn:
        """Record a completed exchange in the history."""
        self.turns.append(Turn(role="user", content=message))
        turn = Turn(role="assistant", content=answer, metrics=metrics,
                    alternates=list(alternates or []))
        self.turns.append(turn)
        return turn

    def ask_all(self, message: str) -> List[tuple]:
        """Ask every model the same question from the identical history."""
        return [self.ask(message, handle=h) for h in self.handles]

    # -- reporting --------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Session-level numbers, for the closing line and the transcript."""
        answers = [t for t in self.turns if t.role == "assistant" and t.metrics]
        n = len(answers)
        refused = sum(1 for t in answers if t.metrics.refused)
        knows = sum(1 for t in answers if t.metrics.outcome == "knows_but_complies")
        scored = [t.metrics.harm_score for t in answers if t.metrics.harm_score is not None]
        return {
            "started": self.started,
            "models": [h.provenance() for h in self.handles],
            "system_prompt": self.system_prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "turns": n,
            "refused": refused,
            "refusal_rate": (refused / n) if n else 0.0,
            "knows_but_complies": knows,
            "mean_harm_score": (sum(scored) / len(scored)) if scored else None,
            "truncated": sum(1 for t in answers if t.metrics.truncated),
            "probe": (self.probe_set.metadata() if self.probe_set is not None else None),
        }

    def transcript(self) -> Dict[str, Any]:
        return {
            "summary": self.summary(),
            "turns": [
                {**{k: v for k, v in asdict(t).items() if k != "metrics"},
                 "metrics": (asdict(t.metrics) if t.metrics else None)}
                for t in self.turns
            ],
        }

    def save(self, path: str | Path) -> Path:
        """Write the transcript, with provenance, as JSON."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.transcript(), indent=2, default=str))
        logger.info("Saved transcript to %s (%d turns)", p, len(self.turns))
        return p


def load_handle(path: str, label: str, temperature: float, max_tokens: int,
                device: str, dtype: str) -> ModelHandle:
    """Load one model through the serving provider, with its manifest."""
    from vivasecuris.aiasylum.models import get_provider
    from vivasecuris.aiasylum.weights.manifest import SurgeryManifest

    provider = get_provider("transformers")
    model = provider.create_model(
        path, temperature=temperature, max_tokens=max_tokens, device=device, dtype=dtype
    )
    return ModelHandle(label=label, path=path, model=model, manifest=SurgeryManifest.load(path))


def make_probe_scorer(probe_path: str, handle: ModelHandle, device: str, dtype: str):
    """A callable scoring one prompt with a saved probe, or ``None``.

    The probe must have been fitted on the model being scored: probe weights
    index a specific layer of a specific network and do not transfer. The
    mismatch is reported rather than silently producing numbers.
    """
    from vivasecuris.aiasylum.interp.probes.monitor import score_prompts
    from vivasecuris.aiasylum.interp.probes.train import ProbeSet
    from vivasecuris.aiasylum.models.transformers_local import _get_cached

    ps = ProbeSet.load(probe_path)
    model, tokenizer = _get_cached(handle.path, device, dtype)

    def score(prompt: str, system_prompt: Optional[str] = None) -> float:
        return score_prompts(model, tokenizer, [prompt], ps, system_prompt=system_prompt)[0]

    return ps, score
