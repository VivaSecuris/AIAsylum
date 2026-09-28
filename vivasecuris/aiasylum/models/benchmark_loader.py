"""Text-only generation for modern native Transformers architectures.

This is selected only in the isolated benchmark worker. Interpretability keeps
its validated loader and does not claim support for these newer architectures.
"""

from __future__ import annotations


def load_benchmark_model(model_id: str, device: str, dtype: str):
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer
    from vivasecuris.aiasylum.interp.core.loader import get_hf_token, load, resolve_device, resolve_dtype

    token = get_hf_token()
    config = AutoConfig.from_pretrained(model_id, token=token, trust_remote_code=False)
    if config.model_type not in {"qwen3_5", "gemma4", "gemma4_unified", "mistral3"}:
        return load(model_id, device=device, dtype=dtype, seed=None)

    torch_device = resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token, trust_remote_code=False)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, config=config, token=token, trust_remote_code=False,
        dtype=resolve_dtype(dtype),
        device_map={"": str(torch_device)} if torch_device.type == "cuda" else None,
    )
    if torch_device.type != "cuda":
        model = model.to(torch_device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def prepare_benchmark_inputs(tokenizer, prompt, system_prompt, messages, device):
    """Tokenize the chat template once, including tokenizer-native Mistral tokens."""
    chat = []
    if system_prompt:
        chat.append({"role": "system", "content": system_prompt})
    chat.extend(messages or [])
    if prompt:
        chat.append({"role": "user", "content": prompt})
    if not chat:
        raise ValueError("No prompt or messages supplied")
    # MistralCommonBackend implements its own template without a Jinja string;
    # decoding its control tokens to text and encoding again is not equivalent.
    native_mistral = tokenizer.__class__.__name__ == "MistralCommonBackend"
    if native_mistral:
        return tokenizer.apply_chat_template(
            chat, tokenize=True, add_generation_prompt=True, return_dict=True,
            return_tensors="pt",
        ).to(device)
    # Every Jinja-templated or template-less tokenizer goes through the shared
    # formatter. transformers' own tokenize=True path is render-then-tokenize with
    # add_special_tokens=False, so this is byte-identical for templates and, for a
    # base model, finally adds the BOS the old hardcoded False never did.
    from vivasecuris.aiasylum.weights.capture import format_chat

    text, applied = format_chat(tokenizer, chat)
    return tokenizer(text, return_tensors="pt", add_special_tokens=not applied).to(device)
