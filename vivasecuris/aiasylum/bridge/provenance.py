"""Bridge provenance manifest (``aiasylum-bridge.json``)."""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any, Optional


def write_bridge_manifest(
    out_dir: str | Path,
    *,
    direction: str,
    payload: dict[str, Any],
) -> Path:
    """Write ``aiasylum-bridge.json`` next to a checkpoint or work product."""
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    doc = {
        "version": 1,
        "direction": direction,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        **payload,
    }
    path = root / "aiasylum-bridge.json"
    path.write_text(json.dumps(doc, indent=2) + "\n")
    return path.resolve()


def read_bridge_manifest(path: str | Path) -> Optional[dict[str, Any]]:
    candidate = Path(path)
    if candidate.is_dir():
        candidate = candidate / "aiasylum-bridge.json"
    if not candidate.is_file():
        return None
    return json.loads(candidate.read_text())
