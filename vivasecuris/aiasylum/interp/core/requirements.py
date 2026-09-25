"""Resolve dependencies and require the evidence promised by an analysis request."""
from typing import Any, Mapping


def capture_options(options: Mapping[str, Any]) -> dict[str, Any]:
    """Return effective flags, including the prerequisites of each public option."""
    values = dict(options)
    if values.get("enable_minimal_circuit"):
        values.update(enable_qkv_capture=True, enable_mlp_capture=True)
    if values.get("enable_patching"):
        if values.get("patch_components") == "head":
            values["enable_qkv_capture"] = True
        elif values.get("patch_components") == "neuron":
            values["enable_pre_mlp_capture"] = True
    if values.get("enable_qkv_capture"):
        values["enable_attention_capture"] = True
    captures = any(values.get(flag) for flag in (
        "enable_attention_capture", "enable_mlp_capture", "enable_pre_mlp_capture", "enable_qkv_capture",
    ))
    if captures:
        values.update(enable_component_analysis=True, enable_attn_output_capture=True)
    return values


def validate_result(result: Any, config: Any, mode: str | None = None) -> None:
    """A completed run must contain each requested scientific output."""
    mode = mode or config.analysis_mode
    required = {"pca_payload": "trajectory projection", "predictions_payload": "next-token predictions"}
    flags = {
        "enable_attention_capture": ("attention_payload", "attention capture"),
        "enable_mlp_capture": ("mlp_payload", "MLP capture"),
        "enable_pre_mlp_capture": ("mlp_payload", "internal MLP neuron capture"),
        "enable_qkv_capture": ("ov_qk_payload", "OV/QK analysis"),
        "enable_patching": ("patching_results", "activation patching"),
        "enable_minimal_circuit": ("minimal_circuit_payload", "approximate circuit search"),
    }
    if mode != "model_diff":
        for flag, (field, label) in flags.items():
            if getattr(config, flag, False):
                required[field] = label
    for field, label in required.items():
        payload = getattr(result, field, None)
        if not payload:
            raise RuntimeError(f"Analysis incomplete: requested {label} produced no result")
        if isinstance(payload, dict) and payload.get("available") is False:
            raise RuntimeError(f"Analysis incomplete: {label}: {payload.get('reason', 'unavailable')}")
    for field, available_key in (("attention_payload", "attention_available"), ("mlp_payload", "mlp_available"), ("ov_qk_payload", "ov_available")):
        payload = getattr(result, field, None)
        if field in required and payload:
            for block, value in payload.items():
                if isinstance(value, dict) and value.get(available_key) is False:
                    raise RuntimeError(f"Analysis incomplete: {required[field]} unavailable for block {block}")
    patching = getattr(result, "patching_results", None)
    if getattr(config, "enable_patching", False) and patching:
        if not patching.get("enabled") or not patching.get("experiments"):
            raise RuntimeError(f"Activation patching produced no interventions: {patching.get('reason', patching.get('notes', ''))}")
    circuit = getattr(result, "minimal_circuit_payload", None)
    if getattr(config, "enable_minimal_circuit", False) and circuit and not circuit.get("num_candidates"):
        raise RuntimeError("Approximate circuit search found no supported components to examine")
    result.meta["validated_outputs"] = list(required.values())
