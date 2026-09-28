"""Helpers for passing test context (temperature, seed, per-role settings) to model generation."""

from typing import Any, Dict, List, Optional

DEFAULT_TEMPERATURE = 0.7


def _blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def _as_float(value: Any) -> Optional[float]:
    if _blank(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> Optional[int]:
    if _blank(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _layers(context: Optional[Dict], role: Optional[str], overrides: Optional[Dict]) -> List[Dict]:
    """Settings layers, most specific first: per-call overrides, then ``roles[role]``."""
    layers: List[Dict] = []
    if isinstance(overrides, dict):
        layers.append(overrides)
    if role and context:
        roles = context.get("roles")
        if isinstance(roles, dict) and isinstance(roles.get(role), dict):
            layers.append(roles[role])
    return layers


def role_setting(context: Optional[Dict], role: Optional[str], key: str, default: Any = None,
                 overrides: Optional[Dict] = None) -> Any:
    """One setting for a role: overrides, then ``roles[role]``, then the flat context key."""
    for layer in _layers(context, role, overrides):
        if key in layer and not _blank(layer[key]):
            return layer[key]
    if context and key in context and not _blank(context[key]):
        return context[key]
    return default


def model_gen_kwargs_from_context(context: Optional[Dict], role: Optional[str] = None,
                                  overrides: Optional[Dict] = None) -> Dict[str, Any]:
    """Build kwargs for model.generate() from test context.

    Precedence per setting: ``overrides`` (e.g. one group-therapy patient) >
    ``context["roles"][role]`` > the legacy flat ``temperature``/``seed`` > defaults.
    Temperature is always included (default 0.7) when there is any context, so the
    model receives an explicit value rather than relying on provider defaults.
    ``top_p`` and ``max_tokens`` are only sent when set. A layer holding
    ``"seed": None`` means "no seed for this role" (a provider that rejects seeds).
    """
    out: Dict[str, Any] = {}
    layers = _layers(context, role, overrides)
    if context is None and not layers:
        return out

    temperature = None
    for layer in layers:
        temperature = _as_float(layer.get("temperature"))
        if temperature is not None:
            break
    if temperature is None:
        temperature = _as_float((context or {}).get("temperature"))
    out["temperature"] = DEFAULT_TEMPERATURE if temperature is None else temperature

    for key, parse, valid in (
        ("top_p", _as_float, lambda v: 0 < v <= 1),
        ("max_tokens", _as_int, lambda v: v >= 1),
    ):
        for layer in layers:
            value = parse(layer.get(key))
            if value is not None:
                if valid(value):
                    out[key] = value
                break

    for layer in layers:
        if "seed" not in layer:
            continue
        if layer["seed"] is None:
            return out
        seed = _as_int(layer["seed"])
        if seed is not None:
            out["seed"] = seed
            return out
    seed = _as_int((context or {}).get("seed"))
    if seed is not None:
        out["seed"] = seed
    return out
