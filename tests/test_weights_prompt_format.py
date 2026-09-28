"""Prompt formatting: the (text, special-tokens) contract every capture and generation path shares.

Fixture-only; nothing downloads. The BOS-capable tokenizer exists because no
other fixture in this repo inserts a BOS token, so the defect this file guards
-- a base model captured with no BOS because ``add_special_tokens`` was keyed
on the caller's *request* rather than on whether a template was *applied* --
was invisible to every test that came before it.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

try:
    from _tiny_lm import build_model, build_tokenizer
except ImportError:  # pragma: no cover - depends on how pytest roots the tests dir
    from tests._tiny_lm import build_model, build_tokenizer

from vivasecuris.aiasylum.weights import capture as capture_mod
from vivasecuris.aiasylum.weights.capture import (
    capture_last_token_residuals,
    capture_pooled_residuals,
    format_chat,
    format_prompts,
    has_chat_template,
    render_chat,
)

BOS_ID = 3
PLAIN = "{% for m in messages %}{{ m['content'] }} {% endfor %}"          # emits no BOS
BOS_TEMPLATE = "[BOS] " + PLAIN                                            # emits its own BOS


@pytest.fixture(autouse=True)
def _reset_warn_once():
    capture_mod._warned_no_template.clear()
    yield
    capture_mod._warned_no_template.clear()


@pytest.fixture
def bos_tok():
    return build_tokenizer(with_bos=True)


@pytest.fixture
def model():
    return build_model()


def _observe_input_ids(model):
    """Record the input_ids of every forward pass; returns (list, handle)."""
    seen = []
    handle = model.register_forward_pre_hook(
        lambda _m, _a, kw: seen.append(kw["input_ids"].detach().clone()), with_kwargs=True
    )
    return seen, handle


def _bos_per_row(ids):
    return (ids == BOS_ID).sum(dim=1)


def test_bos_fixture_mechanics(bos_tok):
    """The premise of this file: the fixture really inserts one BOS, and only when asked."""
    assert bos_tok.bos_token_id == BOS_ID
    assert has_chat_template(bos_tok) is False
    assert bos_tok("w3 w4")["input_ids"] == [BOS_ID, 7, 8]
    assert bos_tok("w3 w4", add_special_tokens=False)["input_ids"] == [7, 8]
    # A literal [BOS] in the text is matched as the special token, not shredded.
    assert bos_tok("[BOS] w3", add_special_tokens=False)["input_ids"] == [BOS_ID, 7]
    assert bos_tok("[BOS] w3")["input_ids"] == [BOS_ID, BOS_ID, 7]


def test_format_chat_reports_template_applied(bos_tok):
    msgs = [{"role": "user", "content": "w3 w4"}]

    text, applied = format_chat(bos_tok, msgs)
    assert applied is False
    assert text == "w3 w4\n\n"

    # An empty template is no template: transformers would render every prompt to "".
    bos_tok.chat_template = ""
    assert has_chat_template(bos_tok) is False
    assert format_chat(bos_tok, msgs)[1] is False

    bos_tok.chat_template = PLAIN
    text, applied = format_chat(bos_tok, msgs)
    assert applied is True
    assert text == render_chat(bos_tok, msgs)


def test_no_chat_template_does_not_drop_the_system_prompt(bos_tok):
    assert format_prompts(bos_tok, ["w3"], system_prompt="w20") == (["w20\n\nw3\n\n"], False)
    assert format_prompts(bos_tok, []) == ([], False)


def test_special_tokens_flag_keys_on_the_template_not_the_request(model, bos_tok):
    """apply_template=True on a base model must still yield exactly one BOS per row."""
    seen, handle = _observe_input_ids(model)
    try:
        capture_last_token_residuals(model, bos_tok, ["w3 w4", "w5"], apply_template=True, batch_size=2)
        capture_pooled_residuals(model, bos_tok, ["w3 w4", "w5"], apply_template=True, batch_size=2)
        assert seen, "no forward pass observed"
        for ids in seen:
            assert (_bos_per_row(ids) == 1).all(), ids

        # The instruct no-op: a template that renders no BOS gets none added on top.
        seen.clear()
        bos_tok.chat_template = PLAIN
        capture_last_token_residuals(model, bos_tok, ["w3 w4"], apply_template=True)
        assert seen
        for ids in seen:
            assert (_bos_per_row(ids) == 0).all(), ids
    finally:
        handle.remove()


def test_distill_branches_agree_on_bos(model, bos_tok):
    """A base-model teacher sees exactly one BOS with or without a system prompt.

    Before the single-path rewrite the system-prompt branch added BOS and the bare
    branch did not, so the same teacher was prompted two different ways.
    """
    from vivasecuris.aiasylum.weights.distill import generate_teacher_rows

    w3, w4, w20 = 7, 8, 24                      # ids in the with_bos vocabulary
    seen, handle = _observe_input_ids(model)
    try:
        generate_teacher_rows(model, bos_tok, ["w3 w4"], max_new_tokens=2)
        assert seen, "no forward pass observed"
        assert seen[0][0].tolist() == [BOS_ID, w3, w4]          # the prompt pass; later passes are cached
        seen.clear()

        generate_teacher_rows(model, bos_tok, ["w3 w4"], max_new_tokens=2, system_prompt="w20")
        assert seen
        assert seen[0][0].tolist() == [BOS_ID, w20, w3, w4]     # system kept, still one BOS
    finally:
        handle.remove()


@pytest.mark.parametrize("templated", [False, True], ids=["base-model", "templated"])
def test_capture_and_generation_produce_the_same_prompt_text(bos_tok, templated):
    """The drift guard _build_prompt's docstring promises: capture, serving and the
    benchmark worker render the same text and tokenize it with the same specials rule."""
    from vivasecuris.aiasylum.models.benchmark_loader import prepare_benchmark_inputs
    from vivasecuris.aiasylum.models.transformers_local import TransformersModel

    if templated:
        bos_tok.chat_template = PLAIN
    prompt, system = "w3 w4", "w20"

    texts, applied = format_prompts(bos_tok, [prompt], system_prompt=system)
    # _build_prompt reads nothing from self, so it can be called unbound without
    # constructing (and lazily loading) a provider.
    served_text, served_applied = TransformersModel._build_prompt(None, bos_tok, prompt, system, None)
    assert (served_text, served_applied) == (texts[0], applied)
    assert applied is templated

    expected_ids = bos_tok(texts[0], add_special_tokens=not applied)["input_ids"]
    bench = prepare_benchmark_inputs(bos_tok, prompt, system, None, "cpu")
    assert bench["input_ids"][0].tolist() == expected_ids
    # And the rule itself: a base model gets exactly one BOS, a template gets none added.
    assert expected_ids.count(BOS_ID) == (0 if templated else 1)


def _split():
    from vivasecuris.aiasylum.weights.corpus import PromptSplit

    harmful = [f"w{i} w{i + 1} w{i + 2}" for i in range(10, 22, 2)]
    harmless = [f"w{i} w{i + 3}" for i in range(30, 42, 2)]
    return PromptSplit(
        harmful_train=harmful[:4], harmful_test=harmful[4:],
        harmless_train=harmless[:4], harmless_test=harmless[4:],
        seed=0, source="test",
    )


def test_derivation_refuses_base_model_without_override(model, bos_tok):
    """Deriving from plain-text prompts is a different experiment: it has to be asked for,
    and the direction then records that it was."""
    from vivasecuris.aiasylum.weights.capture import NoChatTemplateError, PROMPT_FORMAT_VERSION
    from vivasecuris.aiasylum.weights.direction import derive_direction, derive_subspace
    from vivasecuris.aiasylum.weights.rfm import derive_rfm_subspace

    split = _split()
    common = dict(model_id="tiny", batch_size=4, max_length=16)
    with pytest.raises(NoChatTemplateError):
        derive_direction(model, bos_tok, split, **common)
    with pytest.raises(NoChatTemplateError):
        derive_subspace(model, bos_tok, split, rank=2, pool_layers=2, **common)
    with pytest.raises(NoChatTemplateError):
        derive_rfm_subspace(model, bos_tok, split, rank=2, iterations=1, candidate_layers=1, **common)

    d = derive_direction(model, bos_tok, split, allow_no_chat_template=True, **common)
    assert d.template_applied is False
    assert d.format_version == PROMPT_FORMAT_VERSION
    assert d.extra["thinking"] is False
    assert d.metadata()["template_applied"] is False

    # The same derivation on a templated tokenizer needs no override and says so.
    bos_tok.chat_template = PLAIN
    t = derive_direction(model, bos_tok, split, **common)
    assert t.template_applied is True


def test_direction_records_format_version_and_old_files_load(tmp_path, bos_tok):
    """New files carry the format contract; old files load as version 1 and are refused on base models."""
    import json

    from vivasecuris.aiasylum.weights.capture import PROMPT_FORMAT_VERSION
    from vivasecuris.aiasylum.weights.direction import RefusalDirection, check_direction_format

    d = RefusalDirection(
        vector=torch.ones(32) / 32 ** 0.5, layer=1, auc=0.95, cohens_d=1.0,
        model_id="tiny", split_hash="abc", template_applied=False,
    )
    d.save(tmp_path)
    meta = json.loads((tmp_path / "direction.json").read_text())
    assert meta["format_version"] == PROMPT_FORMAT_VERSION == 2
    assert meta["template_applied"] is False

    # A pre-fix file has neither key: it must load, as version 1, unknown template.
    meta.pop("format_version")
    meta.pop("template_applied")
    (tmp_path / "direction.json").write_text(json.dumps(meta))
    old = RefusalDirection.load(tmp_path)
    assert old.format_version == 1 and old.template_applied is None

    with pytest.raises(ValueError, match="before the prompt-format fix"):
        check_direction_format(old, bos_tok)            # v1 on a base model: invalid
    bos_tok.chat_template = PLAIN
    check_direction_format(old, bos_tok)                # v1 on a templated model: equivalent to v2
    check_direction_format(d, build_tokenizer(with_bos=True))   # v2 on a base model: fine


def test_routing_does_not_double_bos_on_a_template_that_emits_one(bos_tok):
    """Llama-3/Gemma-style templates carry their own BOS; routing must not add a second."""
    from vivasecuris.aiasylum.weights.routing import routing_statistics

    moe = build_model(family="qwen2_moe")
    bos_tok.chat_template = BOS_TEMPLATE
    seen, handle = _observe_input_ids(moe)
    try:
        routing_statistics(moe, bos_tok, ["w3 w4", "w5 w6"], ["w7", "w8 w9"], max_length=32)
        assert seen
        for ids in seen:
            assert (_bos_per_row(ids) == 1).all(), ids
    finally:
        handle.remove()
