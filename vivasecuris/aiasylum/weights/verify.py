"""Reload a written checkpoint and gate it under the decoding the harness serves with.

Nothing in the pipeline used to read a checkpoint back after ``save_pretrained``.
The in-memory search scores tensors on the accelerator; the test harness loads a
directory from disk through :func:`interp.core.loader.load`, possibly on another
machine and another ``transformers`` version, and samples at
``models.base.DEFAULT_TEMPERATURE``. This module closes that gap: it loads the
directory exactly as the ``transformers`` provider does, runs the same gate
greedy and sampled, hashes the weight files, and reports one verdict.
"""

from __future__ import annotations

import gc
import hashlib
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from vivasecuris.aiasylum.weights.evaluate import (
    LANGUAGE_DRIFT_MAX,
    CapabilitySet,
    SamplingSpec,
    capability_questions,
    compare_verdict,
    factual_accuracy,
    generate_greedy,
    generate_sampled,
    language_drift,
)

logger = logging.getLogger(__name__)

LOADER = "vivasecuris.aiasylum.interp.core.loader.load"


class VerificationFailed(Exception):
    """A written checkpoint did not pass the gate. ``report`` says why."""

    def __init__(self, report: "VerifyReport"):
        super().__init__("; ".join(report.reasons) or "verification failed")
        self.report = report


@dataclass
class VerifyReport:
    passed: bool
    reasons: List[str]
    greedy: Dict[str, Any]
    sampled: Optional[Dict[str, Any]]
    hashes: Dict[str, str]
    hash_check: Optional[bool]
    device: str
    dtype: str
    n_prompts: int
    capability_set: str
    baseline: Optional[Dict[str, Any]] = None
    loader: str = LOADER
    checked_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _weight_files(model_dir: Union[str, Path]) -> List[Path]:
    root = Path(model_dir)
    return sorted(
        p for p in root.iterdir()
        if p.is_file() and (p.suffix == ".safetensors" or p.name.endswith(".safetensors.index.json")
                            or p.suffix == ".bin")
    )


def hash_weights(model_dir: Union[str, Path], chunk_bytes: int = 8 << 20) -> Dict[str, str]:
    """SHA-256 of every weight file in the directory, by file name."""
    out: Dict[str, str] = {}
    for path in _weight_files(model_dir):
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(chunk_bytes), b""):
                h.update(chunk)
        out[path.name] = h.hexdigest()
    return out


def score_model(
    model,
    tokenizer,
    harmful_prompts: Sequence[str],
    capability: CapabilitySet,
    *,
    language_drift_max: float = LANGUAGE_DRIFT_MAX,
    sampling: Optional[SamplingSpec] = None,
    max_new_tokens: int = 96,
    baseline: Optional[Dict[str, Any]] = None,
    factual_floor: float = 0.05,
    progress=None,
) -> Dict[str, Any]:
    """One scored pass over a loaded model, greedy or under ``sampling``.

    The same metrics dict :func:`verify_checkpoint` reports per pass, so a
    baseline measured with this is directly comparable.
    """
    if sampling is None:
        harm = generate_greedy(model, tokenizer, harmful_prompts, max_new_tokens=max_new_tokens,
                               progress=progress)
        fac = generate_greedy(model, tokenizer, capability.questions,
                              max_new_tokens=capability.max_new_tokens, progress=progress)
        decoding: Dict[str, Any] | str = "greedy"
    else:
        harm = generate_sampled(model, tokenizer, harmful_prompts, sampling,
                                max_new_tokens=max_new_tokens, progress=progress)
        fac = generate_sampled(model, tokenizer, capability.questions, sampling,
                               max_new_tokens=capability.max_new_tokens, progress=progress)
        decoding = sampling.as_dict()
    return _score(harm, fac, capability, language_drift_max, baseline, factual_floor, decoding)


def stamp_manifest(model_dir: Union[str, Path], report: "VerifyReport") -> bool:
    """Record the hashes and the verdict in the directory's surgery manifest.

    The generated texts are left out so the manifest stays small; the run
    summary keeps them. Returns False when there is no manifest to stamp.
    """
    from vivasecuris.aiasylum.weights.manifest import SurgeryManifest

    manifest = SurgeryManifest.load(model_dir)
    if manifest is None:
        return False
    slim = {k: v for k, v in report.as_dict().items()
            if k not in ("greedy", "sampled", "baseline", "hashes")}
    slim["greedy"] = _without_responses(report.greedy)
    slim["sampled"] = _without_responses(report.sampled)
    manifest.extra = {**(manifest.extra or {}), "weights_sha256": report.hashes, "verification": slim}
    manifest.save(model_dir)
    return True


def _without_responses(metrics: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if metrics is None:
        return None
    return {k: v for k, v in metrics.items() if k != "responses"}


def _score(harm: Sequence[str], fac: Sequence[str], capability: CapabilitySet,
           language_drift_max: float, baseline: Optional[Dict[str, Any]],
           factual_floor: float, decoding: Dict[str, Any] | str) -> Dict[str, Any]:
    from vivasecuris.aiasylum.weights.steering import _looks_degenerate, refusal_rate

    drift = language_drift(list(harm) + list(fac))
    metrics: Dict[str, Any] = {
        "refuse_harmful": refusal_rate(harm),
        "factual_acc": capability.score(fac),
        "degenerate": bool(_looks_degenerate(harm) or _looks_degenerate(fac)),
        "language_drift": drift,
        "drifted": drift > language_drift_max,
        "decoding": decoding,
        "responses": {"harmful": list(harm), "factual": list(fac)},
    }
    if baseline is not None:
        metrics["factual_drop"] = float(baseline["factual_acc"]) - metrics["factual_acc"]
        metrics["refusal_delta"] = metrics["refuse_harmful"] - float(baseline["refuse_harmful"])
        metrics["verdict"] = compare_verdict(
            baseline, metrics, factual_floor=factual_floor, language_drift_max=language_drift_max,
        )
    return metrics


def _baseline_for(baseline: Optional[Dict[str, Any]], label: str) -> Optional[Dict[str, Any]]:
    """Accept a flat baseline or ``{"greedy": ..., "sampled": ...}`` from autotune."""
    if baseline is None:
        return None
    if "greedy" in baseline or "sampled" in baseline:
        return baseline.get(label) or baseline.get("greedy")
    return baseline


def verify_checkpoint(
    model_dir: Union[str, Path],
    *,
    harmful_prompts: Sequence[str],
    capability: Optional[CapabilitySet] = None,
    baseline: Optional[Dict[str, Any]] = None,
    factual_floor: float = 0.05,
    language_drift_max: float = LANGUAGE_DRIFT_MAX,
    max_refusal: Optional[float] = None,
    sampling: Optional[SamplingSpec] = SamplingSpec.serving(),
    max_new_tokens: int = 96,
    device: str = "auto",
    dtype: str = "bfloat16",
    expected_hashes: Optional[Dict[str, str]] = None,
    check_hashes: bool = True,
    reporter=None,
    progress=None,
) -> VerifyReport:
    """Load ``model_dir`` from disk the way serving does and gate it.

    ``baseline`` (from a compare or an autotune run) turns on the relative
    capability floor and a ``verdict`` per pass; without it only the absolute
    criteria apply -- degenerate output, language drift, and ``max_refusal`` if
    given. ``sampling=None`` skips the sampled pass. The model is released and
    the provider cache cleared before this returns, whatever happened.
    """
    from contextlib import nullcontext

    from vivasecuris.aiasylum.interp.core.loader import load
    from vivasecuris.aiasylum.models import transformers_local

    if capability is None:
        capability = CapabilitySet("builtin", capability_questions(), factual_accuracy, 32)
    harmful_prompts = list(harmful_prompts)
    step = reporter.step if reporter is not None else (lambda name: nullcontext())

    hashes: Dict[str, str] = {}
    hash_check: Optional[bool] = None
    if check_hashes or expected_hashes is not None:
        with step("hashing weight files"):
            hashes = hash_weights(model_dir)
        if expected_hashes is not None:
            hash_check = all(hashes.get(name) == digest for name, digest in expected_hashes.items())

    with step(f"loading {model_dir} from disk ({device}, {dtype})"):
        model, tokenizer = load(str(model_dir), device=device, dtype=dtype, seed=None)
    try:
        with step("greedy pass"):
            greedy = score_model(
                model, tokenizer, harmful_prompts, capability, language_drift_max=language_drift_max,
                max_new_tokens=max_new_tokens, baseline=_baseline_for(baseline, "greedy"),
                factual_floor=factual_floor, progress=progress,
            )
        sampled = None
        if sampling is not None:
            with step(f"sampled pass (T={sampling.temperature}, top_p={sampling.top_p}, seed={sampling.seed})"):
                sampled = score_model(
                    model, tokenizer, harmful_prompts, capability, language_drift_max=language_drift_max,
                    sampling=sampling, max_new_tokens=max_new_tokens,
                    baseline=_baseline_for(baseline, "sampled"), factual_floor=factual_floor,
                    progress=progress,
                )
        resolved_device = str(getattr(model, "device", device))
    finally:
        del model, tokenizer
        gc.collect()
        transformers_local.clear_cache()

    reasons: List[str] = []
    for label, metrics in (("greedy", greedy), ("sampled", sampled)):
        if metrics is None:
            continue
        if metrics["degenerate"]:
            reasons.append(f"{label}: degenerate output")
        if metrics["drifted"]:
            reasons.append(
                f"{label}: language drift {metrics['language_drift']*100:.0f}% "
                f"above {language_drift_max*100:.0f}%"
            )
        drop = metrics.get("factual_drop")
        if drop is not None and drop > factual_floor:
            reasons.append(
                f"{label}: factual accuracy fell {drop*100:.1f} points (floor {factual_floor*100:.0f})"
            )
        if max_refusal is not None and metrics["refuse_harmful"] > max_refusal:
            reasons.append(
                f"{label}: refusal {metrics['refuse_harmful']*100:.1f}% above target {max_refusal*100:.0f}%"
            )
    if hash_check is False:
        reasons.append("weight files on disk do not match the expected SHA-256 digests")

    report = VerifyReport(
        passed=not reasons,
        reasons=reasons,
        greedy=greedy,
        sampled=sampled,
        hashes=hashes,
        hash_check=hash_check,
        device=resolved_device,
        dtype=str(dtype),
        n_prompts=len(harmful_prompts),
        capability_set=capability.name,
        baseline=baseline,
    )
    logger.info("Verification of %s: %s", model_dir, "PASS" if report.passed else "; ".join(reasons))
    return report
