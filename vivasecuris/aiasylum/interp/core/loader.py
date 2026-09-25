"""Local model loading for weight-level work.

Adapted from labotomy (``llm_prompt_diff.core.model_loader``). Kept: the MPS
high-watermark fix, device/dtype resolution, the Hugging Face token resolution
chain, gated-repo error messages, and deterministic seeding.

Deliberately dropped: labotomy's Ollama name resolution. It required Ollama to
be *running* before any Hugging Face model could load, and its built-in
fallback table silently mapped ``llama3.2``/``llama3``/``llama3.1`` onto
``microsoft/phi-2``. For a dashboard that is a cosmetic surprise; for weight
surgery it means editing a different model than the one named, with no error.
Model ids here are Hugging Face ids or local paths, and nothing is substituted.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

_OLLAMA_STYLE = ("llama3", "llama3.1", "llama3.2", "qwen2.5", "mistral", "gemma", "phi3")


def get_hf_token() -> Optional[str]:
    """Resolve a Hugging Face token: env, then CLI cache, then token files.

    Returns ``None`` when no token is configured, which is fine for ungated
    models such as the Qwen2.5 instruct series.
    """
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if token:
        return token

    try:
        try:
            from huggingface_hub import get_token as hf_get_token
        except ImportError:
            from huggingface_hub.utils import get_token as hf_get_token
        token = hf_get_token()
        if token:
            logger.info("Using Hugging Face token from huggingface-cli login")
            return token
    except Exception as exc:
        logger.debug("huggingface-cli token lookup failed: %s", exc)

    candidates = []
    if os.getenv("HF_HUB_TOKEN_PATH"):
        candidates.append(Path(os.environ["HF_HUB_TOKEN_PATH"]))
    if os.getenv("HF_HOME"):
        candidates.append(Path(os.environ["HF_HOME"]) / "token")
    candidates.append(Path.home() / ".cache" / "huggingface" / "token")

    for path in candidates:
        try:
            if path.exists():
                raw = path.read_text().strip()
                if raw:
                    logger.info("Using Hugging Face token from %s", path)
                    return raw
        except Exception as exc:
            logger.debug("Could not read token file %s: %s", path, exc)

    return None


def _total_memory_gb() -> float:
    """Physical memory in GiB; 0.0 when it cannot be determined."""
    try:
        import subprocess

        out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5)
        return int(out.stdout.strip()) / 2**30
    except Exception:
        try:
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
        except Exception:
            return 0.0


def resolve_device(device: str = "auto"):
    """Resolve a device string to a ``torch.device``, preferring MPS on Apple."""
    import torch

    if device != "auto":
        resolved = torch.device(device)
        if resolved.type == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("CUDA was requested but is unavailable in this PyTorch runtime")
            if resolved.index is not None and resolved.index >= torch.cuda.device_count():
                raise ValueError(f"CUDA device {resolved.index} does not exist in this runtime")
        if resolved.type == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS was requested but is unavailable in this PyTorch runtime")
        return resolved
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(dtype: str = "bfloat16"):
    """Resolve a dtype string to a ``torch.dtype``."""
    import torch

    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if dtype not in mapping:
        raise ValueError(f"Unknown dtype '{dtype}'. Choose from: {', '.join(sorted(mapping))}")
    return mapping[dtype]


def _check_model_id(model_id: str) -> None:
    """Reject bare Ollama-style names rather than guessing at a mapping."""
    if Path(model_id).exists():
        return
    if "/" in model_id:
        return
    base = model_id.split(":")[0].strip().lower()
    if base in _OLLAMA_STYLE or ":" in model_id:
        raise ValueError(
            f"'{model_id}' looks like an Ollama model name, not a Hugging Face id. "
            f"Weight surgery needs the exact upstream weights, so no mapping is "
            f"guessed here. Pass the full id, e.g. 'Qwen/Qwen2.5-3B-Instruct'."
        )


def set_deterministic(seed: int = 0) -> None:
    """Seed torch (and CUDA, when present) for reproducible capture."""
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load(
    model_id: str,
    device: str = "auto",
    dtype: str = "bfloat16",
    seed: Optional[int] = 0,
) -> Tuple["object", "object"]:
    """Load a causal LM and its tokenizer onto ``device`` in ``dtype``.

    ``model_id`` is a Hugging Face repo id or a local directory. Returns the
    model in eval mode and the tokenizer, with ``pad_token`` filled in from
    ``eos_token`` when the tokenizer lacks one.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _check_model_id(model_id)

    torch_device = resolve_device(device)
    torch_dtype = resolve_dtype(dtype)

    # PyTorch caps MPS allocations below unified memory, which can refuse a model
    # that would otherwise fit. labotomy lifted the cap unconditionally
    # (ratio 0.0 = unlimited), which is right on a 32 GB+ Mac but harmful below
    # that: instead of failing fast, MPS allocates into swap and generation slows
    # by orders of magnitude. Raise the cap only where there is headroom to spare.
    if torch_device.type == "mps" and "PYTORCH_MPS_HIGH_WATERMARK_RATIO" not in os.environ:
        total_gb = _total_memory_gb()
        if total_gb >= 32:
            os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
            logger.info("Lifted the MPS allocation cap (%.0f GB unified memory)", total_gb)
        else:
            logger.info(
                "Leaving the MPS allocation cap in place (%.0f GB unified memory). "
                "Lifting it here would let allocations spill into swap.", total_gb
            )

    if seed is not None:
        set_deterministic(seed)

    token = get_hf_token()
    logger.info("Loading %s (device=%s, dtype=%s)", model_id, torch_device, torch_dtype)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
        # torch_dtype also works on our oldest supported transformers (4.45).
        # Dispatch CUDA weights directly while reading shards so large models
        # do not first require a second complete copy in host RAM.
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True,
            device_map={"": str(torch_device)} if torch_device.type == "cuda" else None,
            token=token,
        )
    except (OSError, ValueError) as exc:
        text = str(exc).lower()
        if "403" in str(exc) or "gated" in text or "awaiting a review" in text:
            raise ValueError(
                f"'{model_id}' is a gated repository. Accept its license on the model "
                f"page, then authenticate with `huggingface-cli login` or HF_TOKEN."
            ) from exc
        if "401" in str(exc) or "authentication" in text:
            raise ValueError(
                f"Authentication required for '{model_id}'. Run `huggingface-cli login` "
                f"or set HF_TOKEN."
            ) from exc
        if "404" in str(exc) or "not found" in text or "not a local folder" in text:
            raise ValueError(
                f"'{model_id}' was not found on the Hub or on disk. Check the id; "
                f"weight surgery needs an exact Hugging Face id such as "
                f"'Qwen/Qwen2.5-3B-Instruct'."
            ) from exc
        raise

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch_device.type != "cuda":
        model = model.to(torch_device)
    model.eval()

    logger.info(
        "Loaded %s: %d layers, d_model=%d, on %s",
        model_id,
        model.config.num_hidden_layers,
        model.config.hidden_size,
        torch_device,
    )
    return model, tokenizer


class ModelLoader:
    """Static-method facade over this module.

    Retained because the vendored services (`orchestrator`, `comparison_service`,
    `single_prompt_service`, `multi_prompt_service`) call it this way. New code
    should use the module-level functions directly.
    """

    @staticmethod
    def load_model(model_id: str, device: str = "cpu", dtype: str = "float32", max_length: int = 2048):
        model, tokenizer = load(model_id, device=device, dtype=dtype, seed=None)
        if max_length and hasattr(model.config, "max_position_embeddings"):
            if max_length < model.config.max_position_embeddings:
                model.config.max_length = max_length
        return model, tokenizer

    @staticmethod
    def resolve_model_id(model_id: str) -> str:
        """Identity. labotomy resolved Ollama names here; see this module's docstring."""
        _check_model_id(model_id)
        return model_id

    @staticmethod
    def set_deterministic(seed: int = 0) -> None:
        set_deterministic(seed)

    @staticmethod
    def get_device(device_str: str):
        return resolve_device(device_str)

    @staticmethod
    def get_dtype(dtype_str: str):
        return resolve_dtype(dtype_str)
