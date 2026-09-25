"""Linear probes on raw residual activations: the harmful-intent monitor.

The 2026 benchmark result this module is built on (SAEGuardBench / InterpGuard,
April 2026) is blunt: sparse-autoencoder features *hurt* jailbreak detection
compared with a logistic regression on raw activations, on every model tested
(Gemma-2-2B 0.949 vs 0.712 AUROC, Llama-3.1-8B 0.867 vs 0.477). The
reconstruction objective throws away low-variance directions, and the safety
signal is in them. So detection here reads raw activations; SAE features, when
this project has them, explain a detection rather than making it.

Two further findings shape the defaults. Final-token-only probes fail in
diagnosable ways (arXiv 2605.12726), so pooling runs over the whole prompt.
Prompted probes -- append a question, read the answer position -- are the most
data-efficient monitors and generalise best under distribution shift (arXiv
2504.20271), so ``prompt_suffix`` is a first-class option.

No probe is reported without its shuffled-label null. A probe on 1000-dimensional
activations with a few hundred examples can memorise, and an AUROC that does
not clear the permutation ceiling is not detection.
"""

from vivasecuris.aiasylum.interp.probes.dataset import (
    ProbeDataset,
    build_harmful_intent_dataset,
    ELICITING_SUFFIX,
)
from vivasecuris.aiasylum.interp.probes.train import (
    LayerProbe,
    ProbeSet,
    train_probes,
)

__all__ = [
    "ProbeDataset",
    "build_harmful_intent_dataset",
    "ELICITING_SUFFIX",
    "LayerProbe",
    "ProbeSet",
    "train_probes",
]
