"""Helpers for passing test context (temperature, seed) to model generation."""

from typing import Any, Dict, Optional


def model_gen_kwargs_from_context(context: Optional[Dict]) -> Dict[str, Any]:
    """Build kwargs for model.generate() from test context (temperature, seed).
    Always includes temperature (default 0.7) when context is provided so the model
    receives an explicit value rather than relying on provider defaults.
    """
    out = {}
    if context is not None:
        try:
            t = context.get("temperature")
            out["temperature"] = float(t) if t is not None and str(t).strip() != "" else 0.7
        except (TypeError, ValueError):
            out["temperature"] = 0.7
    if context and "seed" in context and context["seed"] is not None:
        try:
            s = context["seed"]
            if s == "" or s is None:
                pass
            else:
                out["seed"] = int(s)
        except (TypeError, ValueError):
            pass
    return out
