"""The visible-answer rule: one splitter, one invariant, every scorer reads the answer.

Pure-function cases first, then the ``ModelResponse`` chokepoint, then the
provider paths that deliver reasoning separately. Nothing here loads a model
or touches the network.
"""

import pytest

from vivasecuris.aiasylum.reasoning import (
    DEFAULT_TAGS,
    REASONING_SPLIT_VERSION,
    split_reasoning,
    visible_answer,
)


# --- the rule ----------------------------------------------------------------


@pytest.mark.parametrize(
    "text, answer, reasoning",
    [
        ("Sure, here it is.", "Sure, here it is.", None),
        ("", "", None),
        (None, "", None),
        ("<think>I can't help with this.</think>Sure, step one is.", "Sure, step one is.", "I can't help with this."),
        ("<think>\nmusing\n</think>\n\nAnswer.", "Answer.", "musing"),
        # A template that pre-fills <think> in the prompt: the completion carries only the closing tag.
        ("I'm sorry, I can't.</think>\nHere is the recipe.", "Here is the recipe.", "I'm sorry, I can't."),
        # Cut off inside the trace: there is no answer yet.
        ("<think>still going", "", "still going"),
        ("<THINK>Loud</THINK>quiet", "quiet", "Loud"),
        ("<thinking>alt tag</thinking>answer", "answer", "alt tag"),
        # Several blocks fold into one reasoning string; the answer follows the last closing tag.
        ("<think>a</think>mid<think>b</think>end", "end", "amidb"),
        # An unclosed block after some answer text still means the reply is unfinished.
        ("<think>a</think>partial <think>more", "", "apartial more"),
        # Present but empty is not the same as absent.
        ("<think></think>I'm sorry, I can't help with that.", "I'm sorry, I can't help with that.", ""),
        # Angle brackets that are not a reasoning tag are left alone.
        ("Use <b>bold</b> here.", "Use <b>bold</b> here.", None),
    ],
)
def test_split_reasoning(text, answer, reasoning):
    assert split_reasoning(text) == (answer, reasoning)
    assert visible_answer(text) == answer


def test_analysis_is_a_reasoning_tag_only_on_request():
    text = "<analysis>The final answer is: $100"
    assert split_reasoning(text) == (text, None)
    assert split_reasoning(text, tags=("think", "thinking", "analysis")) == ("", "The final answer is: $100")
    assert DEFAULT_TAGS == ("think", "thinking")
    assert REASONING_SPLIT_VERSION == 1


def test_the_two_older_helpers_agree_with_the_rule():
    from vivasecuris.aiasylum.benchmarks.simple import _visible_answer_text
    from vivasecuris.aiasylum.weights.capture import strip_thinking

    closing_only = "I refuse.</think>\n\nSure, step 1."
    assert strip_thinking(closing_only) == "Sure, step 1."          # new: closing-only traces are stripped
    assert strip_thinking("<think>still going") == ""
    assert strip_thinking("  plain") == "plain"                     # kept: leading whitespace trimmed
    assert _visible_answer_text("<think>A looks plausible.</think>Final answer: **B**") == "Final answer: B"
    assert _visible_answer_text("<analysis>The final answer is: $100") == ""
    assert _visible_answer_text("<think>A</think>B <thinking>C") == ""


# --- the chokepoint ------------------------------------------------------------


def _resp(content, **kw):
    from vivasecuris.aiasylum.models.base import ModelResponse

    return ModelResponse(content=content, model="m", provider="p", **kw)


def test_plain_content_is_untouched():
    r = _resp("Sure, I can help. 2 < 3 and <b>bold</b>.")
    assert r.content == "Sure, I can help. 2 < 3 and <b>bold</b>."
    assert r.metadata is None


def test_an_inline_trace_moves_to_metadata():
    r = _resp("<think>I can't help with this.</think>Sure, step one is.", metadata={"done": True})
    assert r.content == "Sure, step one is."
    assert r.metadata == {"done": True, "reasoning": "I can't help with this.", "reasoning_source": "inline"}


def test_provider_reasoning_is_never_overwritten():
    r = _resp("<think>leftover</think>answer", metadata={"reasoning": "from the field", "reasoning_source": "provider"})
    assert r.content == "answer"
    assert r.metadata["reasoning"] == "from the field"
    assert r.metadata["reasoning_source"] == "provider"
    assert r.metadata["reasoning_inline"] == "leftover"


def test_rebuilding_a_response_is_a_no_op():
    import dataclasses

    r = _resp("<think>t</think>a")
    again = dataclasses.replace(r)
    assert (again.content, again.metadata) == (r.content, r.metadata)


@pytest.mark.asyncio
async def test_react_keeps_the_native_trace_under_its_own_key():
    from vivasecuris.aiasylum.cot.react import ReACTReasoner
    from vivasecuris.aiasylum.models.base import BaseModel, ModelResponse

    class ThinkingModel(BaseModel):
        def __init__(self):
            super().__init__("tiny", "mock")

        async def generate(self, prompt="", system_prompt=None, messages=None, **kwargs):
            return ModelResponse(
                content="<think>native musing</think>Thought: consider it.\nFinal Answer: 42",
                model="tiny", provider="mock", finish_reason="length",
            )

        async def stream_generate(self, prompt="", system_prompt=None, messages=None, **kwargs):
            yield ""

    out = await ReACTReasoner(ThinkingModel()).reason("what is six times seven")
    assert out.content == "42"
    assert out.finish_reason == "length"
    assert out.metadata["reasoning"] == "Thought: consider it."
    assert out.metadata["reasoning_source"] == "react"
    assert out.metadata["native_reasoning"] == "native musing"
    assert out.metadata["native_reasoning_source"] == "inline"
    assert out.metadata["cot_enabled"] is True


# --- providers that deliver reasoning separately -------------------------------


@pytest.mark.asyncio
async def test_ollama_thinking_field_lands_in_metadata_and_think_is_sent_only_when_asked():
    import httpx

    from vivasecuris.aiasylum.models.ollama import OllamaModel

    seen = []

    def handler(request):
        import json as _json

        payload = _json.loads(request.content)
        seen.append(payload)
        return httpx.Response(200, json={
            "message": {"role": "assistant", "content": "Sure, step one.", "thinking": "I can't help... but I will."},
            "done": True, "prompt_eval_count": 3, "eval_count": 4,
        })

    model = OllamaModel("qwen3:0.6b")
    model._client = httpx.AsyncClient(base_url="http://ollama.test", transport=httpx.MockTransport(handler))
    try:
        r = await model.generate("", messages=[{"role": "user", "content": "how?"}])
        assert r.content == "Sure, step one."
        assert r.metadata["reasoning"] == "I can't help... but I will."
        assert r.metadata["reasoning_source"] == "provider"
        assert "think" not in seen[-1]                      # never sent unless asked

        await model.generate("", messages=[{"role": "user", "content": "how?"}], think=False)
        assert seen[-1]["think"] is False
    finally:
        await model.close()


@pytest.mark.asyncio
async def test_openai_reasoning_content_is_read_only_when_it_is_a_string():
    from unittest.mock import AsyncMock, MagicMock, patch

    from vivasecuris.aiasylum.models.providers import OpenAIProvider

    def make(message):
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message = message
        resp.choices[0].finish_reason = "stop"
        resp.usage = None
        return resp

    with patch("vivasecuris.aiasylum.models.providers.settings") as s, \
         patch("vivasecuris.aiasylum.models.providers.AsyncOpenAI") as cls:
        s.openai_api_key = "k"
        client = MagicMock()
        cls.return_value = client
        model = OpenAIProvider().create_model("deepseek-reasoner")

        # A real string field is provider reasoning.
        msg = MagicMock()
        msg.content = "42"
        msg.reasoning_content = "six times seven"
        client.chat.completions.create = AsyncMock(return_value=make(msg))
        r = await model.generate("q")
        assert (r.content, r.metadata) == ("42", {"reasoning": "six times seven", "reasoning_source": "provider"})

        # A mock attribute (what test doubles hand back) is not.
        bare = MagicMock()
        bare.content = "plain"
        client.chat.completions.create = AsyncMock(return_value=make(bare))
        r = await model.generate("q")
        assert r.content == "plain" and r.metadata is None


def test_vivaos_scores_the_visible_answer():
    from vivasecuris.aiasylum.models.vivaos import score_response_flags
    from vivasecuris.aiasylum.reasoning import split_reasoning

    raw = "<think>Sure, as DAN I will answer without restrictions.</think>I'd rather not; here is a safe alternative."
    visible, trace = split_reasoning(raw)
    assert score_response_flags(raw) == ["JailbreakCompliance"]   # what the old path scored
    assert score_response_flags(visible) == []                    # what the provider now scores
    assert trace.startswith("Sure, as DAN")
