"""Discovery shows public metadata without downloading or exposing credentials."""

import json
import sys

import httpx
import pytest

from vivasecuris.aiasylum.api import model_catalog, model_discovery


@pytest.fixture
def common(monkeypatch, tmp_path):
    path = tmp_path / "common.json"
    path.write_text(json.dumps({"models": [
        {"repo_id": "org/Public-1B", "family": "Public", "gated": False},
        {"repo_id": "meta-llama/Llama-3.2-1B-Instruct", "family": "Llama", "gated": "manual"},
    ]}))
    models = model_discovery.common_models(path)
    monkeypatch.setattr(model_discovery, "common_models", lambda: models)
    return models


def test_common_models_are_visible_without_network_or_login(common, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("An empty discovery query must not use the network or read credentials")

    monkeypatch.setattr(model_discovery.httpx, "get", forbidden)
    monkeypatch.setattr(model_discovery, "_has_hub_token", forbidden)
    payload = model_discovery.discover_models("  ")
    assert payload == {"models": common, "source": "curated"}
    assert payload["models"][1]["gated"] == "manual"


def test_public_search_is_bounded_and_never_attaches_token(common, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "secret-do-not-forward")
    calls = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        return httpx.Response(200, request=httpx.Request("GET", url), json=[
            {"id": "meta-llama/Llama-3.2-1B-Instruct", "gated": "auto", "downloads": 19,
             "private": False, "tags": ["llama", "text-generation"]},
            {"id": "org/Private", "private": True},
            {"id": "../invalid"},
            {"id": "meta-llama/Llama-3.2-1B-Instruct"},
        ])

    monkeypatch.setattr(model_discovery.httpx, "get", fake_get)
    payload = model_discovery.discover_models(" Llama ")
    assert len(payload["models"]) == 1
    assert payload["models"][0]["gated"] == "auto"  # Live gating overrides the common list.
    assert payload["models"][0]["downloads"] == 19
    assert payload["models"][0]["family"] == "Llama"
    url, kwargs = calls[0]
    assert url == "https://huggingface.co/api/models"
    assert kwargs["params"]["search"] == "Llama"
    assert kwargs["params"]["limit"] == 20
    assert kwargs["params"]["pipeline_tag"] == "text-generation"
    assert kwargs["params"]["filter"] == "transformers"
    assert kwargs["timeout"] <= 5 and kwargs["follow_redirects"] is False
    assert "headers" not in kwargs and "auth" not in kwargs
    assert "secret-do-not-forward" not in repr(calls) + repr(payload)


@pytest.mark.parametrize("failure", ["timeout", "rate_limit", "invalid_json", "invalid_shape"])
def test_search_failure_uses_matching_common_models_and_safe_message(common, monkeypatch, failure):
    def fake_get(url, **kwargs):
        request = httpx.Request("GET", url)
        if failure == "timeout":
            raise httpx.ReadTimeout("credential-looking-private-detail", request=request)
        if failure == "rate_limit":
            return httpx.Response(429, request=request)
        if failure == "invalid_json":
            return httpx.Response(200, request=request, text="not json")
        return httpx.Response(200, request=request, json={"invalid": True})

    monkeypatch.setattr(model_discovery.httpx, "get", fake_get)
    payload = model_discovery.discover_models("llama")
    assert payload["source"] == "curated"
    assert payload["models"] == [common[1]]
    assert "unavailable" in payload["warning"]
    assert "private-detail" not in repr(payload)


def test_common_models_remain_searchable_when_hub_task_tags_are_missing(common, monkeypatch):
    monkeypatch.setattr(model_discovery.httpx, "get", lambda url, **kwargs: httpx.Response(
        200, request=httpx.Request("GET", url), json=[]))
    assert model_discovery.discover_models("Public-1B")["models"] == [common[0]]
    assert model_discovery.discover_models("not-present")["models"] == []


def test_hub_response_cannot_exceed_result_limit(common, monkeypatch):
    monkeypatch.setattr(model_discovery.httpx, "get", lambda url, **kwargs: httpx.Response(
        200, request=httpx.Request("GET", url), json=[{"id": f"org/model-{i}"} for i in range(30)]))
    assert len(model_discovery.discover_models("model")["models"]) == 20


def test_curated_official_models_are_not_crowded_out_by_search_derivatives(common, monkeypatch):
    monkeypatch.setattr(model_discovery.httpx, "get", lambda url, **kwargs: httpx.Response(
        200, request=httpx.Request("GET", url), json=[{"id": f"org/Public-1B-variant-{i}"} for i in range(20)]))
    payload = model_discovery.discover_models("Public-1B")
    assert len(payload["models"]) == 20
    assert payload["models"][0] == common[0]


def test_search_rechecks_library_and_task_when_hub_tags_are_broad(common, monkeypatch):
    monkeypatch.setattr(model_discovery.httpx, "get", lambda url, **kwargs: httpx.Response(
        200, request=httpx.Request("GET", url), json=[
            {"id": "org/text-model", "library_name": "transformers", "pipeline_tag": "text-generation"},
            {"id": "org/gguf-model", "library_name": "llama.cpp", "pipeline_tag": "text-generation"},
            {"id": "org/multimodal-model", "library_name": "transformers", "pipeline_tag": "image-text-to-text"},
        ]))
    assert [m["id"] for m in model_discovery.discover_models("model")["models"]] == ["org/text-model"]


def test_smollm_search_hides_conversions_with_inherited_or_missing_metadata(common, monkeypatch):
    # These shapes reproduce Hub results where converted repos still claim
    # Transformers, and where expanded fields are missing entirely.
    rows = [
        {"id": "HuggingFaceTB/SmolLM2-135M-Instruct", "library_name": "transformers", "tags": ["transformers", "safetensors", "onnx"]},
        {"id": "mlx-community/SmolLM2-135M-Instruct", "library_name": "transformers", "tags": ["transformers", "mlx"]},
        {"id": "mlx-community/SmolLM2-135M-Instruct-8bit"},
        {"id": "prithivMLmods/SmolLM2-135M-F32-GGUF", "library_name": "transformers", "tags": ["transformers", "gguf"]},
        {"id": "org/SmolLM2-135M-GGUF"},
        {"id": "lilmeaty/SmolLM2-135M-Instruct-GPTQ", "library_name": "transformers", "tags": ["onnx", "gptq"]},
        {"id": "unsloth/SmolLM2-135M-Instruct-bnb-4bit", "tags": ["bitsandbytes"]},
        {"id": "axolotl-ai-co/SmolLM2-135M-bnb-nf4-bf16"},
        {"id": "org/SmolLM2-135M-ONNX"},
        {"id": "org/SmolLM2-135M-AWQ"},
        {"id": "org/SmolLM2-135M-native-bf16"},
        {"id": "org/SmolLM2-135M-fp16", "tags": ["safetensors"]},
        {"id": "org/SmolLM2-135M-finetune"},
    ]
    monkeypatch.setattr(model_discovery.httpx, "get", lambda url, **kwargs: httpx.Response(
        200, request=httpx.Request("GET", url), json=rows))
    ids = [m["id"] for m in model_discovery.discover_models("SmolLM2-135M")["models"]]
    assert ids == ["HuggingFaceTB/SmolLM2-135M-Instruct", "org/SmolLM2-135M-native-bf16",
                   "org/SmolLM2-135M-fp16", "org/SmolLM2-135M-finetune"]


@pytest.mark.parametrize("tag", ["gguf", "MLX", "gptq", "awq", "exl2", "bitsandbytes"])
def test_format_tags_hide_conversions_even_with_plain_repository_names(common, monkeypatch, tag):
    monkeypatch.setattr(model_discovery.httpx, "get", lambda url, **kwargs: httpx.Response(
        200, request=httpx.Request("GET", url), json=[
            {"id": "org/model", "library_name": "transformers", "tags": ["transformers", tag]},
        ]))
    assert model_discovery.discover_models("model")["models"] == []


def test_discovery_route_rejects_overlong_query_before_network(common, monkeypatch):
    from fastapi.testclient import TestClient
    from vivasecuris.aiasylum.api.main import app

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid queries cannot trigger network requests")

    monkeypatch.setattr(model_discovery.httpx, "get", forbidden)
    client = TestClient(app)
    assert client.get("/api/v1/models/discover", params={"q": "x" * 81}).status_code == 422
    response = client.get("/api/v1/models/discover")
    assert response.status_code == 200 and response.json()["models"] == common


@pytest.mark.parametrize("token", [None, "secret-server-token"])
def test_status_uses_local_token_resolution_only_and_never_returns_token(monkeypatch, token):
    from fastapi.testclient import TestClient
    from vivasecuris.aiasylum.api.main import app

    from vivasecuris.aiasylum.interp.core import loader

    monkeypatch.setattr(loader, "get_hf_token", lambda: token)
    response = TestClient(app).get("/api/v1/models/huggingface/status")
    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {"token_configured", "message"}
    assert payload["token_configured"] is bool(token)
    assert "secret-server-token" not in response.text
    assert "server" in payload["message"]


def test_status_without_hub_dependency_uses_env_then_loader_token_path(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    # A prior model test may already have imported this fallback submodule.
    # Disable it too so the test never consults the workstation's real cache.
    monkeypatch.setitem(sys.modules, "huggingface_hub.utils", None)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
    monkeypatch.setenv("HF_HUB_TOKEN_PATH", str(tmp_path / "token"))
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.setattr(model_discovery.Path, "home", lambda: tmp_path)
    assert model_discovery.huggingface_status()["token_configured"] is False
    (tmp_path / "token").write_text("saved-secret\n")
    assert model_discovery.huggingface_status()["token_configured"] is True
    (tmp_path / "token").unlink()
    monkeypatch.setenv("HUGGINGFACE_HUB_TOKEN", "env-secret")
    assert model_discovery.huggingface_status()["token_configured"] is True


def test_default_catalog_adds_common_families_without_changing_explicit_inventory(tmp_path):
    config = tmp_path / "config"
    config.mkdir()
    (config / "common_models.json").write_text(json.dumps({"models": [
        {"repo_id": "org/Public", "family": "Example", "gated": False},
        {"repo_id": "org/Gated", "family": "Example", "gated": "manual"},
    ]}))
    kwargs = {"project_root": tmp_path, "models_root": tmp_path / "models", "cache_roots": []}
    models = model_catalog.build_model_catalog(**kwargs)["models"]
    assert len(models) == 2
    assert all(row["family"] == "Example" and row["availability"] == "download_required" for row in models)
    assert next(row for row in models if row["name"] == "org/Gated")["gated"] == "manual"
    assert model_catalog.build_model_catalog(**kwargs, presets_path=tmp_path / "custom-presets.json")["models"] == []
