"""Non-network checkpoint sizing for preflight; estimates are not allocation guarantees."""

import json
import re
from pathlib import Path
from typing import Optional


def estimated_weights_gb(model_ref: str, dtype: str = "bfloat16") -> Optional[float]:
    """Size the loaded/saved dtype, resolving Hub references from the local cache.

    Prefer the index's total bytes or actual checkpoint files. A parameter count
    in the model name is a conservative fallback for uncached dense checkpoints.
    Quantized source sizes cannot estimate an expanded loaded copy reliably.
    """
    root = Path(model_ref).expanduser()
    if not root.is_dir():
        try:
            from huggingface_hub import try_to_load_from_cache

            cached = try_to_load_from_cache(model_ref, "config.json")
            if isinstance(cached, str):
                root = Path(cached).parent
        except Exception:
            pass
    config = {}
    if root.is_dir():
        try:
            config = json.loads((root / "config.json").read_text())
        except (OSError, ValueError):
            pass
        total = 0
        for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
            try:
                total = int(json.loads((root / name).read_text())["metadata"]["total_size"])
                break
            except (OSError, ValueError, KeyError, TypeError):
                pass
        if not total:
            for pattern in ("*.safetensors", "pytorch_model*.bin"):
                total = sum(p.stat().st_size for p in root.glob(pattern) if p.is_file())
                if total:
                    break
        if total and not config.get("quantization_config"):
            stored = str(config.get("torch_dtype") or config.get("dtype") or "bfloat16")
            stored_bytes = 4 if "float32" in stored else 2
            output_bytes = 4 if dtype in ("float32", "fp32") else 2
            return total * output_bytes / stored_bytes / 2**30
    # Explicit total parameter size, never an active-parameter suffix (A3B).
    match = re.search(r"(?:^|[-_/])(\d+(?:\.\d+)?)B(?:[-_/]|$)", model_ref, re.I)
    if match:
        return float(match.group(1)) * 1e9 * (4 if dtype in ("float32", "fp32") else 2) / 2**30
    return None
