"""Unit tests for the Ollama ↔ Transformers bridge helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vivasecuris.aiasylum.bridge.convert import BridgeError, build_modelfile, validate_hf_checkpoint
from vivasecuris.aiasylum.bridge.ollama_store import (
    assert_local_ollama_model,
    export_weight_gguf,
    parse_model_name,
    resolve_manifest_path,
)


def test_parse_model_name_defaults_tag():
    assert parse_model_name("qwen2.5") == ("qwen2.5", "latest")
    assert parse_model_name("qwen2.5:0.5b") == ("qwen2.5", "0.5b")
    assert parse_model_name("library/qwen2.5:0.5b") == ("qwen2.5", "0.5b")


@pytest.mark.parametrize(
    "name",
    [
        "glm-5.3:cloud",
        "deepseek-v4-pro:cloud",
        "registry.ollama.ai/library/glm-5.3:cloud",
    ],
)
def test_parse_rejects_cloud_tags(name):
    with pytest.raises(BridgeError, match="cloud"):
        parse_model_name(name)


def test_parse_rejects_empty():
    with pytest.raises(BridgeError, match="empty"):
        parse_model_name("  ")


def test_resolve_manifest_missing(tmp_path):
    models = tmp_path / "models"
    (models / "manifests" / "registry.ollama.ai" / "library").mkdir(parents=True)
    (models / "blobs").mkdir()
    with pytest.raises(BridgeError, match="No local Ollama manifest"):
        resolve_manifest_path("missing:tag", models)


def test_export_weight_gguf_hardlink(tmp_path):
    models = tmp_path / "models"
    manifest_dir = models / "manifests" / "registry.ollama.ai" / "library" / "toy"
    manifest_dir.mkdir(parents=True)
    blobs = models / "blobs"
    blobs.mkdir()
    digest = "sha256:" + ("ab" * 32)
    blob_name = "sha256-" + ("ab" * 32)
    blob = blobs / blob_name
    blob.write_bytes(b"GGUF-fake-weights")
    template_digest = "sha256:" + ("cd" * 32)
    template_blob = blobs / ("sha256-" + ("cd" * 32))
    template_blob.write_text("{{ .Prompt }}")
    manifest = {
        "schemaVersion": 2,
        "layers": [
            {
                "mediaType": "application/vnd.ollama.image.model",
                "digest": digest,
                "size": blob.stat().st_size,
            },
            {
                "mediaType": "application/vnd.ollama.image.template",
                "digest": template_digest,
                "size": template_blob.stat().st_size,
            },
        ],
    }
    (manifest_dir / "0.5b").write_text(json.dumps(manifest))

    dest = tmp_path / "out" / "toy.gguf"
    meta = export_weight_gguf("toy:0.5b", dest, models)
    assert dest.is_file()
    assert dest.read_bytes() == b"GGUF-fake-weights"
    assert meta["blob_digest"] == digest
    assert meta["template"] == "{{ .Prompt }}"
    assert meta["link_mode"] in {"hardlink", "copy"}


def test_assert_local_rejects_cloud_before_lookup():
    with pytest.raises(BridgeError, match="cloud"):
        assert_local_ollama_model("glm-5.3:cloud")


def test_build_modelfile_includes_system_and_template():
    text = build_modelfile(
        "model-f16.gguf",
        template="{{ .System }}{{ .Prompt }}",
        system="You are helpful.",
        parameters={"temperature": 0.2},
    )
    assert text.startswith("FROM model-f16.gguf\n")
    assert 'SYSTEM """You are helpful."""' in text
    assert 'TEMPLATE """{{ .System }}{{ .Prompt }}"""' in text
    assert "PARAMETER temperature 0.2" in text


def test_validate_hf_checkpoint_requires_files(tmp_path):
    root = tmp_path / "ckpt"
    root.mkdir()
    with pytest.raises(BridgeError, match="config.json"):
        validate_hf_checkpoint(root)
    (root / "config.json").write_text(json.dumps({"model_type": "qwen2"}))
    with pytest.raises(BridgeError, match="tokenizer_config"):
        validate_hf_checkpoint(root)
    (root / "tokenizer_config.json").write_text("{}")
    (root / "tokenizer.json").write_text("{}")
    with pytest.raises(BridgeError, match="weights"):
        validate_hf_checkpoint(root)
    (root / "model.safetensors").write_bytes(b"x")
    assert validate_hf_checkpoint(root) == root.resolve()
