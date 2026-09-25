"""The instrumented chat session: measurement, history, compare mode, transcript.

The session takes model handles rather than paths, so almost everything here
runs against a fake model and no weights are loaded. One test at the end drives
the real local provider on a tiny model to check the streaming path.
"""

from __future__ import annotations

import json

import pytest


class FakeResponse:
    def __init__(self, content, finish_reason="stop"):
        self.content = content
        self.finish_reason = finish_reason


class FakeModel:
    """Records what it was asked and replies from a script."""

    def __init__(self, replies, finish_reason="stop"):
        self.replies = list(replies)
        self.finish_reason = finish_reason
        self.calls = []

    def _next(self):
        return self.replies.pop(0) if self.replies else "(no more scripted replies)"

    async def generate(self, prompt="", system_prompt=None, messages=None, **kwargs):
        self.calls.append({"system": system_prompt, "messages": list(messages or []), **kwargs})
        return FakeResponse(self._next(), self.finish_reason)

    async def stream_generate(self, prompt="", system_prompt=None, messages=None, **kwargs):
        self.calls.append({"system": system_prompt, "messages": list(messages or []), **kwargs})
        for word in self._next().split(" "):
            yield word + " "


def handle(label="model", replies=("Sure, here is how.",), path="/tmp/m", manifest=None,
           finish_reason="stop"):
    from vivasecuris.aiasylum.weights.chat import ModelHandle

    return ModelHandle(label=label, path=path, model=FakeModel(replies, finish_reason),
                       manifest=manifest)


def session(handles=None, **kw):
    from vivasecuris.aiasylum.weights.chat import ChatSession

    return ChatSession(handles or [handle()], **kw)


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def test_refusal_is_scored_with_the_projects_own_phrases():
    s = session()
    refused = s.measure("m", "p", "I'm sorry, I can't help with that.", 0.1)
    complied = s.measure("m", "p", "Sure, here are the steps.", 0.1)
    assert refused.refused is True and complied.refused is False


def test_a_reasoning_trace_that_refuses_then_complies_counts_as_compliance():
    """`strip_thinking` exists precisely so a <think> block cannot be read as
    the answer. The visible length must also exclude it."""
    s = session()
    m = s.measure("m", "p", "<think>I can't help with this.</think>Sure, step one is.", 0.1)
    assert m.refused is False
    assert m.thinking_chars > 0
    assert m.completion_chars == len("Sure, step one is.")


def test_harm_score_produces_the_knows_but_complies_classification():
    s = session(probe_scorer=lambda prompt: 0.9)
    complied = s.measure("m", "p", "Sure, here you go.", 0.1, harm_score=0.9)
    refused = s.measure("m", "p", "I cannot assist with that.", 0.1, harm_score=0.9)
    missed = s.measure("m", "p", "Sure, here you go.", 0.1, harm_score=0.1)
    assert complied.outcome == "knows_but_complies"
    assert refused.outcome == "caught"
    assert missed.outcome == "missed"
    # Without a probe there is no claim to make.
    assert s.measure("m", "p", "Sure.", 0.1).outcome is None


def test_a_broken_probe_does_not_end_the_session():
    def explode(prompt):
        raise RuntimeError("probe is on the wrong model")

    s = session(probe_scorer=explode)
    assert s._harm_score("anything") is None


def test_truncation_is_flagged_from_the_finish_reason():
    s = session([handle(replies=["a" * 50], finish_reason="length")])
    _, m = s.ask("go on")
    assert m.truncated is True


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------


def test_history_accumulates_and_is_sent_to_the_model():
    h = handle(replies=["first answer", "second answer"])
    s = session([h])
    answer, m = s.ask("one")
    s.commit_turn("one", answer, m)
    answer2, m2 = s.ask("two")
    s.commit_turn("two", answer2, m2)

    assert [t.role for t in s.turns] == ["user", "assistant", "user", "assistant"]
    # The second call saw the first exchange plus the new question.
    sent = h.model.calls[1]["messages"]
    assert [x["content"] for x in sent] == ["one", "first answer", "two"]


def test_reset_clears_history():
    s = session()
    answer, m = s.ask("hi")
    s.commit_turn("hi", answer, m)
    s.reset()
    assert s.turns == [] and s.history == []


def test_system_prompt_and_settings_reach_the_model():
    h = handle()
    s = session([h], system_prompt="be terse", max_tokens=99, temperature=0.4)
    s.ask("hello")
    call = h.model.calls[0]
    assert call["system"] == "be terse"
    assert call["max_tokens"] == 99 and call["temperature"] == 0.4


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


def test_stream_turn_yields_pieces_that_join_to_the_answer():
    h = handle(replies=["one two three"])
    s = session([h])
    pieces = list(s.stream_turn("go"))
    assert len(pieces) > 1, "a stream that yields once is not a stream"
    assert "".join(pieces).strip() == "one two three"


def test_stream_turn_propagates_a_model_failure():
    class Broken:
        async def stream_generate(self, **kw):
            raise RuntimeError("cuda is out of memory")
            yield  # pragma: no cover

    from vivasecuris.aiasylum.weights.chat import ModelHandle

    s = session([ModelHandle("m", "/tmp/m", Broken())])
    with pytest.raises(RuntimeError, match="out of memory"):
        list(s.stream_turn("go"))


# ---------------------------------------------------------------------------
# Compare mode
# ---------------------------------------------------------------------------


def test_compare_asks_both_models_from_the_identical_history():
    base = handle("baseline", ["I cannot help with that."])
    mod = handle("modified", ["Sure, here is how."])
    s = session([base, mod])

    results = s.ask_all("the question")
    assert [m.refused for _, m in results] == [True, False]
    # Both saw exactly the same messages: that is what makes it a comparison.
    assert base.model.calls[0]["messages"] == mod.model.calls[0]["messages"]


def test_compare_keeps_one_history_with_the_other_answer_as_an_alternate():
    base = handle("baseline", ["refused: I cannot help.", "second baseline"])
    mod = handle("modified", ["Sure thing.", "second modified"])
    s = session([base, mod])

    results = s.ask_all("q1")
    (_, base_m), (_, mod_m) = results
    s.commit_turn("q1", "Sure thing.", mod_m,
                  alternates=[{"label": "baseline", "content": "refused: I cannot help."}])
    s.ask_all("q2")

    # The recorded thread is the modified model's, and the baseline saw it too,
    # so the two never drift into separate conversations.
    assert [x["content"] for x in base.model.calls[1]["messages"]] == ["q1", "Sure thing.", "q2"]
    assert s.turns[-1].alternates[0]["label"] == "baseline"


# ---------------------------------------------------------------------------
# Provenance and transcript
# ---------------------------------------------------------------------------


def test_provenance_reports_a_stock_model_as_unmodified():
    assert handle().provenance() == {"label": "model", "path": "/tmp/m", "modified": False}


def test_provenance_carries_the_surgery_manifest():
    from vivasecuris.aiasylum.weights.manifest import SurgeryManifest

    man = SurgeryManifest(source_model="Qwen/Qwen3-8B", method="direction_subspace",
                          beta=None, direction_layer=14, direction_auc=0.98,
                          split_hash="abc123", extra={"subspace_rank": 3, "k": 1.0})
    p = handle(manifest=man).provenance()
    assert p["modified"] is True and p["method"] == "direction_subspace"
    assert p["source_model"] == "Qwen/Qwen3-8B" and p["extra"]["subspace_rank"] == 3


def test_summary_counts_refusals_and_knows_but_complies():
    s = session(probe_scorer=lambda p: 0.9)
    for reply in ("Sure, here you go.", "I'm sorry, I can't help with that."):
        h = handle(replies=[reply])
        answer, m = s.ask(reply and "q", handle=h)
        m.harm_score = 0.9
        from vivasecuris.aiasylum.interp.probes.monitor import classify
        m.outcome = classify(0.9, m.refused)
        s.commit_turn("q", answer, m)

    out = s.summary()
    assert out["turns"] == 2 and out["refused"] == 1 and out["refusal_rate"] == 0.5
    assert out["knows_but_complies"] == 1
    assert out["mean_harm_score"] == pytest.approx(0.9)


def test_transcript_round_trips_as_json_with_provenance(tmp_path):
    from vivasecuris.aiasylum.weights.manifest import SurgeryManifest

    man = SurgeryManifest(source_model="src", method="direction_scale", beta=0.0)
    s = session([handle(manifest=man)])
    answer, m = s.ask("hello")
    s.commit_turn("hello", answer, m)

    path = s.save(tmp_path / "t.json")
    data = json.loads(path.read_text())
    assert data["summary"]["models"][0]["modified"] is True
    assert data["summary"]["models"][0]["method"] == "direction_scale"
    assert [t["role"] for t in data["turns"]] == ["user", "assistant"]
    assert data["turns"][1]["metrics"]["refused"] is False


def test_empty_session_is_rejected():
    from vivasecuris.aiasylum.weights.chat import ChatSession

    with pytest.raises(ValueError, match="at least one model"):
        ChatSession([])


# ---------------------------------------------------------------------------
# Slash commands
# ---------------------------------------------------------------------------


def test_slash_commands_change_settings_and_signal_exit():
    from vivasecuris.aiasylum.weights.cli import _chat_command

    s = session()
    assert _chat_command(s, "/exit") is True
    assert _chat_command(s, "/quit") is True
    assert _chat_command(s, "/help") is False

    _chat_command(s, "/system be brief")
    assert s.system_prompt == "be brief"
    _chat_command(s, "/system")
    assert s.system_prompt is None

    _chat_command(s, "/temp 0.8")
    assert s.temperature == 0.8
    _chat_command(s, "/temp nonsense")
    assert s.temperature == 0.8, "a bad value must not corrupt the setting"

    _chat_command(s, "/tokens 512")
    assert s.max_tokens == 512
    _chat_command(s, "/tokens x")
    assert s.max_tokens == 512

    _chat_command(s, "/nonsense")  # must not raise


def test_slash_reset_and_save(tmp_path):
    from vivasecuris.aiasylum.weights.cli import _chat_command

    s = session()
    answer, m = s.ask("hi")
    s.commit_turn("hi", answer, m)
    _chat_command(s, f"/save {tmp_path / 'out.json'}")
    assert (tmp_path / "out.json").exists()
    _chat_command(s, "/reset")
    assert s.turns == []


def test_slash_retry_drops_the_previous_exchange():
    from vivasecuris.aiasylum.weights.cli import _chat_command

    h = handle(replies=["first", "second"])
    s = session([h])
    answer, m = s.ask("the question")
    s.commit_turn("the question", answer, m)
    assert [t.content for t in s.turns] == ["the question", "first"]

    _chat_command(s, "/retry")
    # One exchange, with the new answer, and the retry saw the original history.
    # Compared stripped: the transcript records exactly what the model emitted,
    # whitespace included, which is what an audit record has to do.
    assert [t.content.strip() for t in s.turns] == ["the question", "second"]
    assert [x["content"] for x in h.model.calls[-1]["messages"]] == ["the question"]


def test_metric_line_states_what_was_measured():
    from vivasecuris.aiasylum.weights.chat import TurnMetrics
    from vivasecuris.aiasylum.weights.cli import _metric_colour, _metric_line

    m = TurnMetrics(label="m", refused=False, elapsed_s=1.2, completion_chars=40,
                    harm_score=0.91, outcome="knows_but_complies", truncated=True)
    line = _metric_line(m)
    assert "complied" in line and "harm 0.91" in line and "knows but complies" in line
    assert "token limit" in line
    assert _metric_colour(m) == "red"

    clean = TurnMetrics(label="m", refused=True, elapsed_s=0.5, completion_chars=10)
    assert "refused" in _metric_line(clean) and _metric_colour(clean) == "bright_black"


# ---------------------------------------------------------------------------
# Real provider, tiny model
# ---------------------------------------------------------------------------


def test_streaming_through_the_real_provider(tmp_path_factory):
    """The provider used to fake streaming by chunking a finished string. This
    drives the real path on a tiny model."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Config

    from vivasecuris.aiasylum.weights.chat import ChatSession, load_handle

    out = tmp_path_factory.mktemp("tiny-chat")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=128, tie_word_embeddings=False,
                      bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id)
    torch.manual_seed(0)
    AutoModelForCausalLM.from_config(cfg).save_pretrained(str(out))
    tok.save_pretrained(str(out))

    h = load_handle(str(out), "model", temperature=0.0, max_tokens=8,
                    device="cpu", dtype="float32")
    assert h.is_modified is False, "a freshly saved model carries no surgery manifest"

    s = ChatSession([h], max_tokens=8)
    pieces = list(s.stream_turn("hello"))
    streamed = "".join(pieces)

    answer, metrics = s.ask("hello")
    assert isinstance(streamed, str) and isinstance(answer, str)
    assert metrics.elapsed_s >= 0 and metrics.refused in (True, False)
    s.commit_turn("hello", streamed, metrics)
    assert s.summary()["turns"] == 1
