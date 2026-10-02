"""Locate and export local Ollama model blobs."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Optional


class BridgeError(Exception):
    """Operator-facing bridge failure."""


def ollama_models_root(explicit: Optional[str | Path] = None) -> Path:
    """Return the Ollama models directory (``$OLLAMA_MODELS`` or ``~/.ollama/models``)."""
    if explicit is not None:
        root = Path(explicit).expanduser()
    elif os.environ.get("OLLAMA_MODELS"):
        root = Path(os.environ["OLLAMA_MODELS"]).expanduser()
    else:
        root = Path.home() / ".ollama" / "models"
    if not root.is_dir():
        raise BridgeError(f"Ollama models directory not found: {root}")
    return root.resolve()


def parse_model_name(name: str) -> tuple[str, str]:
    """Split ``library/name:tag`` or ``name:tag`` into ``(model, tag)``.

    Default tag is ``latest``. Rejects empty names.
    """
    raw = (name or "").strip()
    if not raw:
        raise BridgeError("Ollama model name is empty")
    if raw.endswith(":cloud") or ":cloud" in raw.split("/")[-1]:
        raise BridgeError(
            f"{name!r} is an Ollama cloud tag. Cloud models have no local weight "
            f"blobs and cannot be converted for Transformers surgery. Pull a local "
            f"tag (for example qwen2.5:0.5b) or download the Hugging Face checkpoint."
        )
    # Strip optional registry/library prefix for manifest lookup under library/
    leaf = raw.split("/")[-1]
    if ":" in leaf:
        model, tag = leaf.rsplit(":", 1)
    else:
        model, tag = leaf, "latest"
    if not model or not tag:
        raise BridgeError(f"Invalid Ollama model name: {name!r}")
    if tag.lower() == "cloud":
        raise BridgeError(
            f"{name!r} is an Ollama cloud tag. Cloud models have no local weight "
            f"blobs and cannot be converted for Transformers surgery."
        )
    return model, tag


def resolve_manifest_path(name: str, models_root: Optional[Path] = None) -> Path:
    """Return the on-disk manifest path for a library model tag."""
    root = models_root or ollama_models_root()
    model, tag = parse_model_name(name)
    candidates = [
        root / "manifests" / "registry.ollama.ai" / "library" / model / tag,
        root / "manifests" / "library" / model / tag,
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise BridgeError(
        f"No local Ollama manifest for {name!r}. Looked under {candidates[0].parent}. "
        f"Run `ollama pull {name}` first, or pass a model that exists in `ollama list`."
    )


def read_manifest(name: str, models_root: Optional[Path] = None) -> dict[str, Any]:
    path = resolve_manifest_path(name, models_root)
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise BridgeError(f"Cannot read Ollama manifest {path}: {exc}") from exc
    if not isinstance(data, dict) or "layers" not in data:
        raise BridgeError(f"Unrecognized Ollama manifest format: {path}")
    return data


def assert_local_ollama_model(name: str, models_root: Optional[Path] = None) -> dict[str, Any]:
    """Reject cloud tags and missing manifests; return the parsed manifest."""
    parse_model_name(name)  # cloud check
    return read_manifest(name, models_root)


def _blob_path(models_root: Path, digest: str) -> Path:
    # Ollama stores digests as sha256-<hex> files.
    hex_digest = digest.removeprefix("sha256:")
    path = models_root / "blobs" / f"sha256-{hex_digest}"
    if not path.is_file():
        # Some installs keep the colon form
        alt = models_root / "blobs" / digest
        if alt.is_file():
            return alt
        raise BridgeError(f"Missing Ollama blob for {digest} at {path}")
    return path


def layer_by_media_type(manifest: dict[str, Any], media_type: str) -> Optional[dict[str, Any]]:
    for layer in manifest.get("layers") or []:
        if layer.get("mediaType") == media_type:
            return layer
    return None


def export_weight_gguf(
    name: str,
    dest_gguf: str | Path,
    models_root: Optional[Path] = None,
) -> dict[str, Any]:
    """Hardlink or copy the model weight blob to ``dest_gguf``; return metadata."""
    root = models_root or ollama_models_root()
    manifest = assert_local_ollama_model(name, root)
    model_layer = layer_by_media_type(manifest, "application/vnd.ollama.image.model")
    if model_layer is None:
        raise BridgeError(
            f"{name!r} has no application/vnd.ollama.image.model layer. "
            f"It may be a non-weight artifact or an incomplete pull."
        )
    digest = model_layer.get("digest")
    if not digest:
        raise BridgeError(f"{name!r} model layer is missing a digest")
    src = _blob_path(root, digest)
    dest = Path(dest_gguf)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    try:
        os.link(src, dest)
        link_mode = "hardlink"
    except OSError:
        shutil.copy2(src, dest)
        link_mode = "copy"

    template_text = None
    system_text = None
    template_layer = layer_by_media_type(manifest, "application/vnd.ollama.image.template")
    if template_layer and template_layer.get("digest"):
        template_text = _blob_path(root, template_layer["digest"]).read_text(errors="replace")
    system_layer = layer_by_media_type(manifest, "application/vnd.ollama.image.system")
    if system_layer and system_layer.get("digest"):
        system_text = _blob_path(root, system_layer["digest"]).read_text(errors="replace").strip()

    return {
        "source_ollama": name,
        "manifest_path": str(resolve_manifest_path(name, root)),
        "blob_digest": digest,
        "blob_bytes": int(model_layer.get("size") or src.stat().st_size),
        "gguf_path": str(dest.resolve()),
        "link_mode": link_mode,
        "template": template_text,
        "system": system_text or None,
    }
