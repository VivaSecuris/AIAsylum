"""Model-aware resource checks without downloading or allocating model weights."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

GIB = 2**30
MODEL_PRESETS = [
    {"id": f"Qwen/Qwen3-{size}B", "label": f"Qwen3 {size}B", "parameters_b": float(size),
     "description": "Dense model; run each size separately. BF16 weights need about 2 bytes per parameter."}
    for size in ("0.6", "1.7", "4", "8", "14", "32")
]


def hardware_info() -> dict[str, Any]:
    info: dict[str, Any] = {"available": False, "devices": [], "default_device": "cpu"}
    try:
        import torch
        from vivasecuris.aiasylum.interp.core.loader import _total_memory_gb

        total = _total_memory_gb()
        available = None
        try:
            import psutil
            available = psutil.virtual_memory().available / GIB
        except ImportError:
            if os.path.exists("/proc/meminfo"):
                for line in open("/proc/meminfo"):
                    if line.startswith("MemAvailable:"):
                        available = int(line.split()[1]) * 1024 / GIB
        info.update(available=True, host_total_gb=total, host_available_gb=available,
                    torch_version=torch.__version__, cuda_version=torch.version.cuda)
        info["devices"].append({"device": "cpu", "name": "CPU", "total_gb": total,
                                "free_gb": available})
        if torch.backends.mps.is_available():
            info["devices"].append({"device": "mps", "name": "Apple GPU (shared memory)",
                                    "total_gb": total, "free_gb": available})
            info["default_device"] = "mps"
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                free, capacity = torch.cuda.mem_get_info(index)
                info["devices"].append({"device": f"cuda:{index}", "name": torch.cuda.get_device_name(index),
                                        "total_gb": capacity / GIB, "free_gb": free / GIB,
                                        "bf16": torch.cuda.get_device_capability(index)[0] >= 8})
            info["default_device"] = "cuda:0"
    except ImportError:
        info["error"] = "Install the interpretability dependencies: pip install -e '.[interp]'"
    except Exception as exc:
        info["error"] = f"Could not inspect hardware: {exc}"
    return info


def model_info(model_id: str) -> dict[str, Any]:
    """Fetch config only and count a meta-device model, including tied weights/MoE."""
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM
    from vivasecuris.aiasylum.interp.core.loader import get_hf_token, _check_model_id
    from vivasecuris.aiasylum.interp.core.hook_registry import describe_architecture

    _check_model_id(model_id)
    cfg = AutoConfig.from_pretrained(model_id, token=get_hf_token(), trust_remote_code=False)
    # Meta tensors have shapes but no backing storage, even for a 32B model.
    with init_empty_weights(include_buffers=True):
        model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=False)
        model.tie_weights()
        count = sum(p.numel() for p in model.parameters())
        arch = describe_architecture(model)
    if arch.family is None:
        raise ValueError(f"Architecture {cfg.model_type!r} has no supported activation hooks.")
    return {
        "id": model_id, "model_type": cfg.model_type, "parameters_b": count / 1e9,
        "num_layers": int(cfg.num_hidden_layers), "hidden_size": int(cfg.hidden_size),
        "num_attention_heads": int(cfg.num_attention_heads),
        "num_key_value_heads": int(getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)),
        "head_dim": int(getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads),
        "intermediate_size": int(cfg.intermediate_size), "vocab_size": int(cfg.vocab_size),
        "max_position_embeddings": getattr(cfg, "max_position_embeddings", None),
        "is_moe": arch.is_moe,
    }


def estimate_model(facts: dict[str, Any], options: dict[str, Any]) -> dict[str, Any]:
    """Conservative full-precision capture estimate, in GiB; no quantization assumed."""
    seq = options.get("max_len", 512)
    layers, hidden = facts["num_layers"], facts["hidden_size"]
    scalar = 4 if options.get("dtype", "bfloat16") in ("float32", "fp32") else 2
    weights = facts["parameters_b"] * 1e9 * scalar / GIB
    # The runner retains hidden states/logits on CPU as float32. Hooks likewise
    # retain float32 copies. Include all prompts retained by progression.
    per_pass = ((layers + 1) * seq * hidden + seq * facts["vocab_size"]) * 4
    capture = per_pass
    if options.get("enable_attention_capture"):
        capture += layers * facts["num_attention_heads"] * seq**2 * 4
    if options.get("enable_mlp_capture"):
        capture += layers * seq * hidden * 4
    if options.get("enable_mlp_capture") or options.get("enable_attention_capture"):
        capture += layers * seq * hidden * 4  # attention output
    if options.get("enable_pre_mlp_capture"):
        capture += layers * seq * facts["intermediate_size"] * 4
    if options.get("enable_qkv_capture"):
        capture += layers * seq * (facts["num_attention_heads"] + 2 * facts["num_key_value_heads"]) * facts["head_dim"] * 4
    mode = options.get("mode", "single")
    passes = len(options.get("prompts") or []) if mode == "progression" else (2 if mode in ("comparison", "model_diff") else 1)
    # Account for eager attention retained by Transformers until the forward
    # returns, plus logits, hidden states and workspace. GPU memory is per pass.
    gpu_capture = per_pass * scalar / 4
    if options.get("enable_attention_capture"):
        gpu_capture += layers * facts["num_attention_heads"] * seq**2 * scalar
    return {**facts, "weights_gb": round(weights, 3),
            "capture_gb": round(capture * passes / GIB, 3),
            "estimated_device_gb": round(weights * 1.15 + gpu_capture / GIB + 2, 3)}


def check_request(options: dict[str, Any]) -> dict[str, Any]:
    from vivasecuris.aiasylum.interp.core.requirements import capture_options
    options = capture_options(options)
    hardware = hardware_info()
    device = options.get("device", "auto")
    device = hardware["default_device"] if device == "auto" else ("cuda:0" if device == "cuda" else device)
    result: dict[str, Any] = {"ready": False, "errors": [], "warnings": [], "models": [],
                              "device": device, "dtype": options.get("dtype", "bfloat16"), "hardware": hardware}
    errors, warnings = result["errors"], result["warnings"]
    selected = next((d for d in hardware["devices"] if d["device"] == device), None)
    if not hardware["available"] or hardware.get("error"):
        errors.append(hardware.get("error", "Interpretability runtime is unavailable."))
    if selected is None:
        errors.append(f"Device {device!r} is unavailable on the API server. Choose an available device.")
    if errors:
        return result
    if device.startswith("cuda") and result["dtype"] in ("bfloat16", "bf16") and not selected.get("bf16", True):
        errors.append("This GPU does not support native BF16. Choose float16 or float32.")
    ids = [options["model_a"]]
    if options.get("mode") == "model_diff":
        ids.append(options["model_b"])
    for model_id in ids:
        try:
            path = Path(model_id).expanduser()
            if path.exists() or model_id.startswith(("/", "./", "../", "~/", "models/")):
                from vivasecuris.aiasylum.api.model_catalog import checkpoint_status
                status = checkpoint_status(path)
                if status["availability"] != "ready":
                    raise ValueError(f"Checkpoint is unavailable on this server: {status['reason']}")
            model = estimate_model(model_info(model_id), options)
            result["models"].append(model)
            context = model.get("max_position_embeddings")
            if context and options.get("max_len", 512) > context:
                errors.append(f"{model_id}: token limit exceeds the model context of {context}.")
            if options.get("enable_patching"):
                component = options.get("patch_components", "layer")
                layer_count = model["num_layers"] + (1 if component == "layer" else 0)
                for layer in options.get("patch_layers") or []:
                    if not 0 <= layer < layer_count:
                        errors.append(f"{model_id}: patch layer {layer} is outside 0–{layer_count - 1}.")
                for name, size in (("patch_heads", model["num_attention_heads"]), ("patch_neurons", model["intermediate_size"])):
                    for layer, unit in options.get(name) or []:
                        if not 0 <= layer < model["num_layers"] or not 0 <= unit < size:
                            errors.append(f"{model_id}: {name} pair ({layer}, {unit}) exceeds {model['num_layers']} blocks / {size} units.")
            needed = model["estimated_device_gb"]
            if device in ("mps", "cpu"):
                needed += model["capture_gb"]
            free = selected.get("free_gb")
            if free is not None and needed > free:
                errors.append(f"{model_id}: estimated {needed:.1f} GiB required on {device}; {free:.1f} GiB available. Select a smaller model, shorter token limit, or a larger GPU.")
            if device.startswith("cuda"):
                host_needed = model["weights_gb"] + model["capture_gb"] + 2
                host_free = hardware.get("host_available_gb")
                if host_free is not None and host_needed > host_free:
                    errors.append(f"{model_id}: loading and CPU captures need about {host_needed:.1f} GiB host RAM; {host_free:.1f} GiB available.")
            if model.get("is_moe") and options.get("enable_pre_mlp_capture"):
                errors.append(f"{model_id}: per-neuron capture is unavailable for mixture-of-experts models.")
        except Exception as exc:
            errors.append(f"{model_id}: could not inspect model configuration: {exc}")
    if len(result["models"]) == 2:
        a, b = result["models"]
        if (a["num_layers"], a["hidden_size"]) != (b["num_layers"], b["hidden_size"]):
            errors.append("Model comparison requires matching layer counts and hidden widths (for example, an original and edited checkpoint). Analyze different model sizes in separate runs.")
        warnings.append("Model comparison also requires identical tokenization; this is checked during capture.")
    if device == "cpu":
        warnings.append("CPU execution can be slow for billion-parameter models. Use CUDA for the large-model matrix.")
    if selected.get("free_gb") is None:
        warnings.append("Available memory could not be measured; the resource estimate cannot confirm fit.")
    warnings.append("Estimates assume unquantized weights and include working headroom. Capture memory grows with token count; attention grows quadratically. A successful preflight is not a completed model test.")
    result["ready"] = not errors
    return result
