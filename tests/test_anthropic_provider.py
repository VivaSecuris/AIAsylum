"""The Anthropic provider on the Messages API (anthropic >= 0.125).

Every client here is a mock -- ``MagicMock`` with ``AsyncMock`` methods and
``SimpleNamespace`` responses, the style the rest of the provider tests use --
so nothing needs a key or the network. A thinking block must land in
``metadata["reasoning"]`` and never in the answer; ``temperature`` must reach
only the models that still accept it, and what was sent is recorded.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vivasecuris.aiasylum.models.providers import (
    AnthropicModel,
    _accepts_sampling_params,
    _claude_generation,
    _thinking_config,
)

CURRENT = ["claude-fable-5-1", "claude-opus-5-5", "claude-sonnet-5", "claude-haiku-4-5-20251001"]


def _model(name="claude-sonnet-5", blocks=None, stop_reason="end_turn", **response_extra):
    client = MagicMock()
    response = SimpleNamespace(content=blocks if blocks is not None else [SimpleNamespace(type="text", text="answer")],
                               stop_reason=stop_reason, usage=None, **response_extra)
    client.messages.create = AsyncMock(return_value=response)
    return AnthropicModel(name, SimpleNamespace(client=client)), client


# --- the reply ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_thinking_goes_to_metadata_and_never_into_the_answer():
    model, _ = _model(blocks=[
        SimpleNamespace(type="thinking", thinking="I can't help with this... actually, I can.", signature="s"),
        SimpleNamespace(type="text", text="Sure, step one."),
    ])
    r = await model.generate("q")
    assert r.content == "Sure, step one."
    assert r.metadata["reasoning"] == "I can't help with this... actually, I can."
    assert r.metadata["reasoning_source"] == "provider"


@pytest.mark.asyncio
@pytest.mark.parametrize("blocks", [
    [SimpleNamespace(type="text", text="only text")],
    [SimpleNamespace(type="thinking", thinking="", signature="s"), SimpleNamespace(type="text", text="only text")],
    [SimpleNamespace(type="redacted_thinking", data="opaque"), SimpleNamespace(type="text", text="only text")],
], ids=["text-only", "empty-thinking", "redacted"])
async def test_no_trace_means_no_reasoning_keys(blocks):
    model, _ = _model(blocks=blocks)
    r = await model.generate("q")
    assert r.content == "only text"
    assert "reasoning" not in r.metadata and "reasoning_source" not in r.metadata


@pytest.mark.asyncio
async def test_text_blocks_are_concatenated():
    model, _ = _model(blocks=[SimpleNamespace(type="text", text="one "), SimpleNamespace(type="text", text="two")])
    assert (await model.generate("q")).content == "one two"


@pytest.mark.asyncio
async def test_a_refusal_comes_back_as_one():
    details = SimpleNamespace(type="refusal", category="cyber", explanation="declined")
    model, _ = _model(blocks=[], stop_reason="refusal", stop_details=details)
    r = await model.generate("q")
    assert r.finish_reason == "refusal" and r.content == ""
    assert r.metadata["stop_details"] == {"type": "refusal", "category": "cyber", "explanation": "declined"}
    plain, _ = _model()
    assert "stop_details" not in (await plain.generate("q")).metadata


# --- the request ----------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("name, config", [
    ("claude-sonnet-5", {"type": "adaptive", "display": "summarized"}),
    ("claude-fable-5-1", {"type": "adaptive", "display": "summarized"}),
    ("claude-sonnet-4-6", {"type": "adaptive"}),
    ("claude-haiku-4-5-20251001", {"type": "enabled", "budget_tokens": 2048}),
])
async def test_thinking_is_requested_only_when_asked(name, config):
    model, client = _model(name)
    await model.generate("q")
    assert "thinking" not in client.messages.create.call_args.kwargs
    r = await model.generate("q", thinking=True, max_tokens=4096)
    assert client.messages.create.call_args.kwargs["thinking"] == config
    assert r.metadata["thinking"] == config


def test_a_budget_needs_room_above_the_minimum():
    with pytest.raises(ValueError, match="max_tokens above 1024"):
        _thinking_config("claude-haiku-4-5-20251001", 1024)


@pytest.mark.asyncio
async def test_create_model_default_and_per_call_override():
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=SimpleNamespace(content=[SimpleNamespace(type="text", text="a")], stop_reason="end_turn", usage=None))
    model = AnthropicModel("claude-sonnet-5", SimpleNamespace(client=client), thinking=True)
    await model.generate("q")
    assert "thinking" in client.messages.create.call_args.kwargs
    await model.generate("q", thinking=False)
    assert "thinking" not in client.messages.create.call_args.kwargs


@pytest.mark.asyncio
@pytest.mark.parametrize("name, accepts", [
    ("claude-fable-5-1", False), ("claude-opus-5-5", False), ("claude-sonnet-5", False),
    ("claude-haiku-4-5-20251001", True), ("claude-opus-4-1", True), ("claude-3-5-sonnet-20241022", True),
])
async def test_temperature_reaches_only_the_models_that_take_it(name, accepts):
    model, client = _model(name)
    r = await model.generate("q", temperature=0.3)
    sent = client.messages.create.call_args.kwargs
    assert ("temperature" in sent) is accepts
    assert r.metadata["sampling"]["temperature"] == (0.3 if accepts else None)
    assert model.supports_temperature is accepts


@pytest.mark.asyncio
async def test_an_empty_system_prompt_is_omitted():
    model, client = _model()
    await model.generate("q")
    assert "system" not in client.messages.create.call_args.kwargs
    await model.generate("q", system_prompt="Be brief.")
    assert client.messages.create.call_args.kwargs["system"] == "Be brief."


def test_generation_table():
    assert _claude_generation("claude-haiku-4-5-20251001") == ("haiku", (4, 5))
    assert _claude_generation("claude-opus-5-5") == ("opus", (5, 5))
    assert _claude_generation("claude-sonnet-5") == ("sonnet", (5, 0))
    assert _claude_generation("claude-fable-5-1") == ("frontier", None)
    assert _claude_generation("claude-3-5-sonnet-20241022") == ("claude3", (3, 5))
    assert _claude_generation("gpt-4o") is None
    assert _accepts_sampling_params("some-future-claude") is False     # unknown ids are treated as new
    assert [_accepts_sampling_params(n) for n in CURRENT] == [False, False, False, True]


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["claude-sonnet-5", "claude-opus-4-7", "claude-fable-5-1", "unknown-claude"])
@pytest.mark.parametrize("thinking", [False, True])
async def test_new_models_omit_every_sampling_control_and_record_it(name, thinking):
    model, client = _model(name)
    response = await model.generate("q", thinking=thinking, temperature=0.2, top_p=0.9, top_k=20)
    request = client.messages.create.call_args.kwargs
    assert not {"temperature", "top_p", "top_k"} & request.keys()
    assert response.metadata["sampling"] == {"temperature": None, "top_p": None, "top_k": None}


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["claude-haiku-4-5-20251001", "claude-sonnet-4-6"])
@pytest.mark.parametrize("top_p, expected", [(0.9, None), (0.95, 0.95), (1.0, 1.0)])
async def test_native_thinking_keeps_only_compatible_top_p(name, top_p, expected):
    model, client = _model(name)
    response = await model.generate("q", thinking=True, top_p=top_p, top_k=20)
    request = client.messages.create.call_args.kwargs
    assert "temperature" not in request and "top_k" not in request
    assert request.get("top_p") == expected
    assert response.metadata["sampling"] == {"temperature": None, "top_p": expected, "top_k": None}


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["claude-haiku-4-5-20251001", "claude-sonnet-4-5", "claude-opus-4-1"])
async def test_top_p_replaces_incompatible_default_temperature(name):
    model, client = _model(name)
    response = await model.generate("q", top_p=0.9, top_k=20)
    request = client.messages.create.call_args.kwargs
    assert "temperature" not in request
    assert request["top_p"] == 0.9 and request["top_k"] == 20
    assert response.metadata["sampling"] == {"temperature": None, "top_p": 0.9, "top_k": 20}


@pytest.mark.asyncio
async def test_legacy_model_preserves_compatible_sampling_and_constructor_defaults():
    model, client = _model("claude-3-5-sonnet-20241022")
    model.kwargs.update(top_p=0.9, top_k=20)
    response = await model.generate("q", temperature=0.2)
    assert {key: client.messages.create.call_args.kwargs[key] for key in ("temperature", "top_p", "top_k")} == {
        "temperature": 0.2, "top_p": 0.9, "top_k": 20,
    }
    assert response.metadata["sampling"] == {"temperature": 0.2, "top_p": 0.9, "top_k": 20}


@pytest.mark.asyncio
@pytest.mark.parametrize("name, thinking, expected", [
    ("claude-sonnet-5", False, {}),
    ("claude-haiku-4-5-20251001", True, {"top_p": 0.95}),
    ("claude-haiku-4-5-20251001", False, {"top_p": 0.95, "top_k": 20}),
])
async def test_stream_uses_the_same_sampling_rules(name, thinking, expected):
    async def text_stream():
        yield "answer"

    model, client = _model(name)
    stream = MagicMock()
    stream.__aenter__ = AsyncMock(return_value=SimpleNamespace(text_stream=text_stream()))
    stream.__aexit__ = AsyncMock(return_value=None)
    client.messages.stream.return_value = stream
    assert [part async for part in model.stream_generate("q", thinking=thinking, top_p=0.95, top_k=20)] == ["answer"]
    request = client.messages.stream.call_args.kwargs
    assert {key: request[key] for key in ("temperature", "top_p", "top_k") if key in request} == expected


# --- the provider -------------------------------------------------------------------


def test_an_sdk_without_the_messages_api_is_refused_at_construction():
    from vivasecuris.aiasylum.exceptions import ModelProviderError
    from vivasecuris.aiasylum.models.providers import AnthropicProvider

    with patch("vivasecuris.aiasylum.models.providers.settings") as s, \
         patch("vivasecuris.aiasylum.models.providers.AsyncAnthropic", return_value=SimpleNamespace()):
        s.anthropic_api_key = "k"
        with pytest.raises(ModelProviderError, match="no Messages API"):
            AnthropicProvider()
