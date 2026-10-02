"""Ollama ↔ Transformers bridge helpers.

Convert a local Ollama model to a Hugging Face checkpoint for weight surgery,
then pack an edited checkpoint back into a new Ollama tag. Cloud tags (``:cloud``)
have no local blobs and cannot round-trip.
"""

from vivasecuris.aiasylum.bridge.convert import (
    BridgeError,
    build_modelfile,
    gguf_to_hf,
    hf_to_gguf,
    quantize_gguf,
    validate_hf_checkpoint,
)
from vivasecuris.aiasylum.bridge.ollama_store import (
    assert_local_ollama_model,
    export_weight_gguf,
    ollama_models_root,
    parse_model_name,
    read_manifest,
    resolve_manifest_path,
)
from vivasecuris.aiasylum.bridge.provenance import write_bridge_manifest

__all__ = [
    "BridgeError",
    "assert_local_ollama_model",
    "build_modelfile",
    "export_weight_gguf",
    "gguf_to_hf",
    "hf_to_gguf",
    "ollama_models_root",
    "parse_model_name",
    "quantize_gguf",
    "read_manifest",
    "resolve_manifest_path",
    "validate_hf_checkpoint",
    "write_bridge_manifest",
]
