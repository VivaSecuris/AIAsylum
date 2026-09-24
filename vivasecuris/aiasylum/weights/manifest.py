"""Provenance for a surgically modified model.

Written as ``asylum_surgery.json`` beside the weights, read back by the
``transformers`` provider, and surfaced through ``ModelResponse.metadata`` --
the same channel ``models/vivaos.py`` uses for ``cognition_caught``/``crs``.
It therefore lands in ``ConversationTurn.meta_data`` with no schema change, so
any test run can be traced back to the exact edit that produced the model.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

MANIFEST_NAME = "asylum_surgery.json"


@dataclass
class SurgeryManifest:
    source_model: str
    method: str                       # "direction_scale" | "lora_merge"
    beta: Optional[float] = None
    direction_layer: Optional[int] = None
    direction_auc: Optional[float] = None
    split_hash: Optional[str] = None
    architecture: Optional[str] = None
    matrices_edited: Optional[int] = None
    embeddings_tied: Optional[bool] = None
    embeddings_edited: Optional[bool] = None
    mean_relative_change: Optional[float] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    notes: Optional[str] = None
    extra: dict = field(default_factory=dict)

    def save(self, model_dir: str | Path) -> Path:
        path = Path(model_dir) / MANIFEST_NAME
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))
        logger.info("Wrote surgery manifest to %s", path)
        return path

    @classmethod
    def load(cls, model_dir: str | Path) -> Optional["SurgeryManifest"]:
        """Return the manifest, or ``None`` for an unmodified stock model."""
        path = Path(model_dir) / MANIFEST_NAME
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            known = {f for f in cls.__dataclass_fields__}
            extra = {k: v for k, v in data.items() if k not in known}
            manifest = cls(**{k: v for k, v in data.items() if k in known})
            if extra:
                manifest.extra.update(extra)
            return manifest
        except Exception as exc:
            logger.warning("Could not read surgery manifest at %s: %s", path, exc)
            return None

    def as_metadata(self) -> dict:
        """Compact form for ModelResponse.metadata."""
        return {
            "surgery": {
                "source_model": self.source_model,
                "method": self.method,
                "beta": self.beta,
                "direction_layer": self.direction_layer,
                "direction_auc": self.direction_auc,
                "split_hash": self.split_hash,
                "created_at": self.created_at,
            }
        }
