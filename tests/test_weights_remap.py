"""Token remap: the decode-time filter, the row moves, the untie, and the round trip.

The model tests run on the 64-word tiny Qwen2 from ``_tiny_lm``: a row move does
not care what the rows mean, only that the right ones moved and nothing else
did. The spelling tests need real subword ids and use the Qwen2.5 tokenizer.
"""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from tests._tiny_lm import build_model, build_tokenizer

from vivasecuris.aiasylum.interp.core.arch import embeddings_are_tied
from vivasecuris.aiasylum.weights import remap as remap_mod
from vivasecuris.aiasylum.weights.remap import (
    METHOD,
    RemapError,
    TokenPair,
    TokenRemap,
    TokenRemapProcessor,
    apply_and_save_remap,
    apply_token_remap,
    check_remap_logits,
    generate_once,
    next_token_logits,
    preview_remap,
    remap_processor_list,
    resolve_remap,
    single_token_id,
    source_variant_ids,
    untie_output_embeddings,
    validate_pairs,
)

QWEN = "Qwen/Qwen2.5-3B-Instruct"
OFFSET = 3  # the tiny vocabulary is [UNK], [PAD], [EOS], w0, w1, ...
PROMPT_IDS = [[6, 9, 14, 20, 27]]  # none of these is a token the tests move


def wid(n: int) -> int:
    return OFFSET + n


def remap_of(*pairs) -> TokenRemap:
    """A remap over the tiny vocabulary from ``(source word number, target word number)``."""
    return TokenRemap([TokenPair(wid(a), f"w{a}", wid(b), f"w{b}") for a, b in pairs])


def logits(model, ids=PROMPT_IDS):
    with torch.no_grad():
        return model(input_ids=torch.tensor(ids)).logits.detach().clone()


def final_hidden(model, ids):
    with torch.no_grad():
        out = model(input_ids=torch.tensor(ids), output_hidden_states=True)
    return out.hidden_states[-1].detach().clone()


def untouched(n_vocab: int, remap: TokenRemap):
    mask = torch.ones(n_vocab, dtype=torch.bool)
    mask[remap.src_ids] = False
    mask[remap.dst_ids] = False
    return mask


def safetensors_keys(model_dir):
    from safetensors import safe_open

    with safe_open(str(model_dir / "model.safetensors"), framework="pt") as handle:
        return set(handle.keys())


@pytest.fixture
def tok():
    return build_tokenizer()


@pytest.fixture
def model():
    return build_model()


@pytest.fixture
def tied_model():
    return build_model(tie_word_embeddings=True)


@pytest.fixture(scope="module")
def qwen_tok():
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(QWEN)
    except Exception as exc:  # offline with a cold cache
        pytest.skip(f"the Qwen2.5 tokenizer is not available: {exc}")


# --------------------------------------------------------------------------
# The decode-time filter
# --------------------------------------------------------------------------


def test_processor_moves_the_source_score_and_bans_the_source():
    torch.manual_seed(0)
    scores = torch.randn(3, 50)
    scores[0, 4] = 99.0  # the source is this row's choice
    out = TokenRemapProcessor([(4, 9), (17, 2)])(None, scores)

    assert torch.equal(out[:, 9], torch.maximum(scores[:, 9], scores[:, 4]))
    assert torch.equal(out[:, 2], torch.maximum(scores[:, 2], scores[:, 17]))
    assert torch.isneginf(out[:, 4]).all() and torch.isneginf(out[:, 17]).all()
    rest = [c for c in range(50) if c not in (4, 9, 17, 2)]
    assert torch.equal(out[:, rest], scores[:, rest])
    assert int(scores[0].argmax()) == 4 and int(out[0].argmax()) == 9


def test_processor_pair_order_does_not_matter():
    torch.manual_seed(1)
    scores = torch.randn(2, 40)
    forward = TokenRemapProcessor([(4, 9), (17, 2)])(None, scores)
    backward = TokenRemapProcessor([(17, 2), (4, 9)])(None, scores)
    assert torch.equal(forward, backward)


def test_processor_leaves_the_callers_scores_alone():
    torch.manual_seed(2)
    scores = torch.randn(2, 40)
    kept = scores.clone()
    TokenRemapProcessor([(4, 9)])(None, scores)
    assert torch.equal(scores, kept)


def test_processor_accepts_token_pairs_and_runs_inside_a_processor_list():
    remap = remap_of((5, 9))
    torch.manual_seed(3)
    scores = torch.randn(1, 64)
    out = remap_processor_list(remap)(torch.tensor([[1]]), scores)
    assert torch.isneginf(out[0, wid(5)])
    assert out[0, wid(9)] == max(scores[0, wid(9)], scores[0, wid(5)])


@pytest.mark.parametrize("pairs", [[(4, 4)], [(4, 9), (9, 11)], [(4, 9), (5, 9)], [(4, 9), (4, 11)], []])
def test_validate_pairs_rejects_overlap_identity_and_nothing(pairs):
    with pytest.raises(RemapError):
        validate_pairs(pairs)


def test_validate_pairs_rejects_ids_past_the_table():
    with pytest.raises(RemapError, match="past the end"):
        validate_pairs([(4, 64)], n_rows=64)
    assert validate_pairs([(4, 63)], n_rows=64) == ([4], [63])


# --------------------------------------------------------------------------
# Spellings
# --------------------------------------------------------------------------


def test_hello_has_four_single_token_spellings(qwen_tok):
    assert source_variant_ids(qwen_tok, "Hello") == [9707, 21927, 14990, 23811]


def test_default_pairs_same_shaped_spellings_and_reports_the_rest(qwen_tok):
    remap = resolve_remap(qwen_tok, "Hello", "Bye")
    assert {(p.src_id, p.dst_id) for p in remap.pairs} == {
        (21927, 89325), (14990, 28374), (23811, 53041),
    }
    assert {(p.src_text, p.dst_text) for p in remap.pairs} == {
        (" Hello", " Bye"), ("hello", "bye"), (" hello", " bye"),
    }
    assert len(remap.skipped) == 1
    skipped = remap.skipped[0]
    assert skipped["src_text"] == "Hello" and skipped["src_id"] == 9707
    assert skipped["dst_text"] == "Bye" and skipped["dst_ids"] == [1359, 68]
    assert "2 tokens" in skipped["reason"]
    assert remap.exact is False


def test_exact_requires_a_single_token_target_and_says_which_spellings_are(qwen_tok):
    with pytest.raises(RemapError) as err:
        resolve_remap(qwen_tok, "Hello", "Bye", exact=True)
    message = str(err.value)
    assert "[1359, 68]" in message and "' Bye'" in message and "89325" in message


def test_exact_takes_the_literal_pair_only(qwen_tok):
    remap = resolve_remap(qwen_tok, "Hello", " Bye", exact=True)
    assert [(p.src_id, p.dst_id) for p in remap.pairs] == [(9707, 89325)]
    assert remap.skipped == [] and remap.exact is True


def test_a_source_with_no_single_token_spelling_is_refused(qwen_tok):
    with pytest.raises(RemapError, match="Source"):
        resolve_remap(qwen_tok, "supercalifragilisticexpialidocious", "Bye")


def test_an_unknown_spelling_is_not_a_token(tok):
    assert single_token_id(tok, "w5") == wid(5)
    assert single_token_id(tok, "W5") is None  # one id, but it is [UNK]
    with pytest.raises(RemapError, match="not in the vocabulary"):
        resolve_remap(tok, "w5", "zzz", exact=True)
    with pytest.raises(RemapError, match="same token"):
        resolve_remap(tok, "w5", "w5", exact=True)


def test_empty_words_are_refused(tok):
    with pytest.raises(RemapError):
        resolve_remap(tok, " ", "w9")
    with pytest.raises(RemapError):
        resolve_remap(tok, "w5", "")


# --------------------------------------------------------------------------
# Row moves on an untied model
# --------------------------------------------------------------------------


def test_swap_trades_the_two_columns_and_nothing_else(model, tok):
    remap = remap_of((5, 9), (20, 30))
    assert not embeddings_are_tied(model)
    emb_before = model.get_input_embeddings().weight.detach().clone()
    before = logits(model)

    summary = apply_token_remap(model, tok, remap, mode="swap")
    after = logits(model)

    assert torch.allclose(after[..., remap.dst_ids], before[..., remap.src_ids], atol=1e-5)
    assert torch.allclose(after[..., remap.src_ids], before[..., remap.dst_ids], atol=1e-5)
    keep = untouched(before.shape[-1], remap)
    assert torch.equal(after[..., keep], before[..., keep])
    assert torch.equal(model.get_input_embeddings().weight, emb_before)

    assert summary["matrices_edited"] == 1
    assert summary["embeddings_edited"] is False and summary["inputs_edited"] is False
    assert summary["head_untied"] is False and summary["tied_before"] is False
    assert summary["embeddings_tied"] is False and summary["reversible"] is True
    assert summary["model_type"] == "qwen2" and summary["architecture"]
    assert summary["mode"] == "swap" and summary["vocab_size"] == 64
    assert summary["pairs"][0] == {"src_id": wid(5), "src_text": "w5", "dst_id": wid(9), "dst_text": "w9"}
    assert summary["mean_relative_change"] > 0


def test_copy_gives_the_target_the_source_score_and_zeroes_the_source(model, tok):
    remap = remap_of((5, 9))
    head = model.get_output_embeddings()
    source_row = head.weight[wid(5)].detach().clone()
    before = logits(model)

    summary = apply_token_remap(model, tok, remap, mode="copy")
    after = logits(model)

    assert torch.allclose(after[..., wid(9)], before[..., wid(5)], atol=1e-5)
    assert torch.equal(after[..., wid(5)], torch.zeros_like(after[..., wid(5)]))
    assert torch.equal(head.weight[wid(9)], source_row)
    assert not head.weight[wid(5)].any()
    keep = untouched(before.shape[-1], remap)
    assert torch.equal(after[..., keep], before[..., keep])
    assert summary["reversible"] is False


def test_merge_adds_the_source_score_onto_the_target(model, tok):
    remap = remap_of((5, 9))
    before = logits(model)
    summary = apply_token_remap(model, tok, remap, mode="merge")
    after = logits(model)

    assert torch.allclose(after[..., wid(9)], before[..., wid(9)] + before[..., wid(5)], atol=1e-4)
    assert torch.equal(after[..., wid(5)], torch.zeros_like(after[..., wid(5)]))
    keep = untouched(before.shape[-1], remap)
    assert torch.equal(after[..., keep], before[..., keep])
    assert summary["reversible"] is False


def test_swap_twice_restores_every_tensor_bit_for_bit(model, tok):
    remap = remap_of((5, 9), (20, 30))
    snapshot = {k: v.detach().clone() for k, v in model.state_dict().items()}
    apply_token_remap(model, tok, remap, mode="swap")
    assert not torch.equal(model.get_output_embeddings().weight, snapshot["lm_head.weight"])
    apply_token_remap(model, tok, remap, mode="swap")
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, snapshot[name]), name


def test_inputs_moves_the_embedding_rows_so_the_model_reads_the_original(model, tok):
    remap = remap_of((5, 9))
    emb = model.get_input_embeddings()
    rows = emb.weight.detach().clone()
    said_source = [[6, wid(5), 14]]
    said_target = [[6, wid(9), 14]]
    heard_before = final_hidden(model, said_source)

    summary = apply_token_remap(model, tok, remap, mode="swap", inputs=True)

    assert torch.equal(emb.weight[wid(9)], rows[wid(5)])
    assert torch.equal(emb.weight[wid(5)], rows[wid(9)])
    assert summary["matrices_edited"] == 2
    assert summary["embeddings_edited"] is True and summary["inputs_edited"] is True
    assert summary["shared_table"] is False
    # Emitting the target now leaves the model in the state the source used to.
    assert torch.allclose(final_hidden(model, said_target), heard_before, atol=1e-6)


def test_row_moves_are_exact_in_bfloat16(tok):
    model = build_model().to(torch.bfloat16)
    head = model.get_output_embeddings()
    rows = head.weight.detach().clone()
    apply_token_remap(model, tok, remap_of((5, 9)), mode="swap")
    assert head.weight.dtype == torch.bfloat16
    assert torch.equal(head.weight[wid(9)], rows[wid(5)])
    assert torch.equal(head.weight[wid(5)], rows[wid(9)])
    keep = untouched(rows.shape[0], remap_of((5, 9)))
    assert torch.equal(head.weight[keep], rows[keep])


def test_bad_mode_and_out_of_range_ids_are_refused(model, tok):
    rows = model.get_output_embeddings().weight.detach().clone()
    with pytest.raises(RemapError, match="mode"):
        apply_token_remap(model, tok, remap_of((5, 9)), mode="rotate")
    beyond = TokenRemap([TokenPair(wid(5), "w5", 64, "nowhere")])
    with pytest.raises(RemapError, match="past the end"):
        apply_token_remap(model, tok, beyond)
    twice = TokenRemap([TokenPair(8, "a", 12, "b"), TokenPair(12, "b", 20, "c")])
    with pytest.raises(RemapError, match="more than one pair"):
        apply_token_remap(model, tok, twice)
    assert torch.equal(model.get_output_embeddings().weight, rows)


def test_check_remap_logits_passes_a_true_move_and_catches_a_false_one():
    remap = remap_of((5, 9))
    torch.manual_seed(4)
    before = torch.randn(64)

    swapped = before.clone()
    swapped[wid(5)], swapped[wid(9)] = before[wid(9)], before[wid(5)]
    assert check_remap_logits(before, swapped, remap, "swap")["ok"] is True

    copied = before.clone()
    copied[wid(9)], copied[wid(5)] = before[wid(5)], 0.0
    assert check_remap_logits(before, copied, remap, "copy")["ok"] is True
    assert check_remap_logits(before, swapped, remap, "copy")["ok"] is False

    merged = before.clone()
    merged[wid(9)], merged[wid(5)] = before[wid(9)] + before[wid(5)], 0.0
    assert check_remap_logits(before, merged, remap, "merge")["ok"] is True

    drifted = swapped.clone()
    drifted[40] += 1.0
    verdict = check_remap_logits(before, drifted, remap, "swap")
    assert verdict["ok"] is False and "untouched" in verdict["detail"]
    assert verdict["max_other_diff"] == pytest.approx(1.0)

    assert check_remap_logits(before, before, remap, "swap")["ok"] is False


# --------------------------------------------------------------------------
# Tied embeddings
# --------------------------------------------------------------------------


def test_the_tied_fixture_is_tied(tied_model):
    assert embeddings_are_tied(tied_model)
    assert tied_model.config.tie_word_embeddings is True


def test_untie_is_a_no_op_on_an_untied_model(model):
    head_ptr = model.get_output_embeddings().weight.data_ptr()
    assert untie_output_embeddings(model) is False
    assert model.get_output_embeddings().weight.data_ptr() == head_ptr


def test_head_only_on_a_tied_model_unties_and_leaves_the_table_alone(tied_model, tok):
    remap = remap_of((5, 9))
    table = tied_model.get_input_embeddings().weight.detach().clone()
    head_module = tied_model.get_output_embeddings()

    summary = apply_token_remap(tied_model, tok, remap, mode="swap")

    head = tied_model.get_output_embeddings()
    assert head is head_module  # the parameter was replaced, not the module
    assert not embeddings_are_tied(tied_model)
    assert tied_model.config.tie_word_embeddings is False
    assert torch.equal(tied_model.get_input_embeddings().weight, table)
    assert torch.equal(head.weight[wid(9)], table[wid(5)])
    assert torch.equal(head.weight[wid(5)], table[wid(9)])
    keep = untouched(table.shape[0], remap)
    assert torch.equal(head.weight[keep], table[keep])

    assert summary["head_untied"] is True and summary["tied_before"] is True
    assert summary["embeddings_tied"] is False and summary["embeddings_edited"] is False
    assert summary["matrices_edited"] == 1 and summary["shared_table"] is False

    tied_model.tie_weights()  # what from_pretrained calls; the config flag must hold it off
    assert not embeddings_are_tied(tied_model)
    assert torch.equal(tied_model.get_input_embeddings().weight, table)


def test_inputs_on_a_tied_model_stays_tied_and_moves_the_shared_rows_once(tied_model, tok):
    remap = remap_of((5, 9))
    table = tied_model.get_input_embeddings().weight.detach().clone()

    summary = apply_token_remap(tied_model, tok, remap, mode="swap", inputs=True)

    assert embeddings_are_tied(tied_model)
    assert tied_model.config.tie_word_embeddings is True
    shared = tied_model.get_output_embeddings().weight
    # Moved once. Twice would have put the rows back.
    assert torch.equal(shared[wid(9)], table[wid(5)])
    assert torch.equal(shared[wid(5)], table[wid(9)])
    assert torch.equal(tied_model.get_input_embeddings().weight, shared)
    assert summary["head_untied"] is False and summary["embeddings_tied"] is True
    assert summary["embeddings_edited"] is True and summary["shared_table"] is True
    assert summary["matrices_edited"] == 1


def test_head_only_copy_on_a_tied_model_zeroes_only_the_head_row(tied_model, tok):
    table = tied_model.get_input_embeddings().weight.detach().clone()
    apply_token_remap(tied_model, tok, remap_of((5, 9)), mode="copy")
    assert not tied_model.get_output_embeddings().weight[wid(5)].any()
    assert torch.equal(tied_model.get_input_embeddings().weight[wid(5)], table[wid(5)])
    assert table[wid(5)].any()


# --------------------------------------------------------------------------
# Save and reload
# --------------------------------------------------------------------------


def test_a_head_only_edit_on_a_tied_model_survives_the_round_trip(tied_model, tok, tmp_path):
    from vivasecuris.aiasylum.interp.core.loader import load
    from vivasecuris.aiasylum.weights.manifest import SurgeryManifest

    remap = remap_of((5, 9))
    table = tied_model.get_input_embeddings().weight.detach().clone()
    out = tmp_path / "remapped"

    path, summary = apply_and_save_remap(
        tied_model, tok, remap, str(out), source_model="tiny-tied", mode="swap",
        notes="round trip", check_prompt="w1 w2 w3",
    )
    in_memory = logits(tied_model)

    assert path == str(out)
    assert summary["logit_check"]["ok"] is True and summary["logit_check"]["prompt"] == "w1 w2 w3"
    # Without the untie this key is dropped as a duplicate and the edit is lost.
    assert "lm_head.weight" in safetensors_keys(out)
    assert json.loads((out / "config.json").read_text())["tie_word_embeddings"] is False

    reloaded, _ = load(str(out), device="cpu", dtype="float32")
    assert not embeddings_are_tied(reloaded)
    head = reloaded.get_output_embeddings().weight
    assert torch.equal(head[wid(9)], table[wid(5)])
    assert torch.equal(head[wid(5)], table[wid(9)])
    assert torch.equal(reloaded.get_input_embeddings().weight, table)
    assert torch.allclose(logits(reloaded), in_memory, atol=1e-6)

    manifest = SurgeryManifest.load(out)
    assert manifest.method == METHOD and manifest.source_model == "tiny-tied"
    assert manifest.beta is None and manifest.direction_layer is None
    assert manifest.embeddings_tied is False and manifest.embeddings_edited is False
    assert manifest.matrices_edited == 1 and manifest.notes == "round trip"
    assert manifest.extra["mode"] == "swap" and manifest.extra["reversible"] is True
    assert manifest.extra["head_untied"] is True and manifest.extra["tied_before"] is True
    assert manifest.extra["inputs_edited"] is False
    assert manifest.extra["pairs"] == [
        {"src_id": wid(5), "src_text": "w5", "dst_id": wid(9), "dst_text": "w9"}
    ]
    assert manifest.extra["logit_check"]["ok"] is True
    assert manifest.as_metadata()["surgery"]["method"] == METHOD


def test_a_shared_table_edit_reloads_tied(tied_model, tok, tmp_path):
    from vivasecuris.aiasylum.interp.core.loader import load
    from vivasecuris.aiasylum.weights.manifest import SurgeryManifest

    remap = remap_of((5, 9))
    table = tied_model.get_input_embeddings().weight.detach().clone()
    out = tmp_path / "remapped-both"

    _, summary = apply_and_save_remap(
        tied_model, tok, remap, str(out), source_model="tiny-tied", mode="swap",
        inputs=True, check_prompt="w1 w2 w3",
    )

    assert summary["logit_check"]["ok"] is True
    assert "lm_head.weight" not in safetensors_keys(out)
    assert json.loads((out / "config.json").read_text()).get("tie_word_embeddings", True) is True

    reloaded, _ = load(str(out), device="cpu", dtype="float32")
    assert embeddings_are_tied(reloaded)
    shared = reloaded.get_output_embeddings().weight
    assert torch.equal(shared[wid(9)], table[wid(5)])
    assert torch.equal(shared[wid(5)], table[wid(9)])

    manifest = SurgeryManifest.load(out)
    assert manifest.embeddings_tied is True and manifest.embeddings_edited is True
    assert manifest.extra["inputs_edited"] is True and manifest.extra["head_untied"] is False
    assert manifest.extra["shared_table"] is True


def test_the_check_is_skipped_when_the_prompt_holds_a_moved_token_and_inputs_move(model, tok, tmp_path):
    out = tmp_path / "skipped-check"
    _, summary = apply_and_save_remap(
        model, tok, remap_of((5, 9)), str(out), source_model="tiny", inputs=True,
        check_prompt="w1 w5 w3",
    )
    assert summary["logit_check"]["ok"] is None
    assert "remapped token" in summary["logit_check"]["skipped"]
    assert (out / "model.safetensors").exists()


def test_a_non_empty_out_dir_is_refused_before_the_model_is_touched(model, tok, tmp_path):
    out = tmp_path / "taken"
    out.mkdir()
    (out / "keep.txt").write_text("mine")
    rows = model.get_output_embeddings().weight.detach().clone()

    with pytest.raises(FileExistsError):
        apply_and_save_remap(model, tok, remap_of((5, 9)), str(out), source_model="tiny")

    assert torch.equal(model.get_output_embeddings().weight, rows)
    assert sorted(p.name for p in out.iterdir()) == ["keep.txt"]


def test_a_failed_check_writes_nothing(model, tok, tmp_path, monkeypatch):
    out = tmp_path / "never"
    monkeypatch.setattr(
        remap_mod, "check_remap_logits",
        lambda *a, **k: {"ok": False, "detail": "target columns are off by up to 3"},
    )
    with pytest.raises(RuntimeError, match="Nothing was written"):
        apply_and_save_remap(model, tok, remap_of((5, 9)), str(out), source_model="tiny",
                             check_prompt="w1 w2 w3")
    assert not out.exists() or not any(out.iterdir())


# --------------------------------------------------------------------------
# The preview
# --------------------------------------------------------------------------


def _first_choice(model, tok):
    """A prompt whose greedy next token is an ordinary word, and that token's id."""
    for n in range(3, 40):
        prompt = f"w{n} w{n + 1}"
        choice = int(next_token_logits(model, tok, prompt).argmax())
        if choice >= OFFSET:
            return prompt, choice
    pytest.skip("this random model prefers a special token after every prompt tried")


def test_the_preview_replaces_the_models_choice_and_touches_no_weight(model, tok):
    prompt, chosen = _first_choice(model, tok)
    target = chosen + 1 if chosen + 1 < 64 else chosen - 1
    remap = TokenRemap([TokenPair(chosen, tok.convert_ids_to_tokens(chosen),
                                  target, tok.convert_ids_to_tokens(target))])
    snapshot = {k: v.detach().clone() for k, v in model.state_dict().items()}

    baseline, remapped = preview_remap(model, tok, prompt, remap, max_new_tokens=3)

    assert baseline.split()[0] == tok.convert_ids_to_tokens(chosen)
    assert remapped.split()[0] == tok.convert_ids_to_tokens(target)
    assert tok.convert_ids_to_tokens(chosen) not in remapped.split()
    assert baseline == generate_once(model, tok, prompt, max_new_tokens=3)
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, snapshot[name]), name


# --------------------------------------------------------------------------
# The command
# --------------------------------------------------------------------------


@pytest.fixture
def cli(monkeypatch):
    """The `weights` group with the loader stubbed to hand back a tiny model."""
    from vivasecuris.aiasylum.weights import cli as cli_mod

    box = {"tied": False, "loads": 0}

    def fake_load(model, device="auto", dtype="bfloat16", seed=0):
        box["loads"] += 1
        box["model"] = build_model(tie_word_embeddings=box["tied"])
        box["tok"] = build_tokenizer()
        return box["model"], box["tok"]

    monkeypatch.setattr(cli_mod, "_preflight", lambda *a, **k: None)
    monkeypatch.setattr(cli_mod, "_warn_if_low_disk", lambda *a, **k: None)
    monkeypatch.setattr("vivasecuris.aiasylum.interp.core.loader.load", fake_load)
    box["run"] = lambda *args: CliRunner().invoke(cli_mod.weights, ["remap", *args])
    box["group"] = cli_mod.weights
    box["module"] = cli_mod
    return box


def test_cli_needs_exactly_one_of_preview_and_out(cli, tmp_path):
    neither = cli["run"]("--from", "w5", "--to", "w9")
    both = cli["run"]("--from", "w5", "--to", "w9", "--preview", "--out", str(tmp_path / "o"))
    for result in (neither, both):
        assert result.exit_code != 0
        assert "--preview" in result.output and "--out" in result.output
    assert cli["loads"] == 0


def test_cli_preview_prints_both_completions_and_touches_nothing(cli):
    result = cli["run"]("--from", "w5", "--to", "w9", "--exact", "--preview",
                        "--prompt", "w1 w2", "--max-new-tokens", "3")
    assert result.exit_code == 0, result.output
    assert "'w5' (8) -> 'w9' (12)" in result.output
    assert "--- baseline ---" in result.output and "--- remapped" in result.output
    assert "--out DIR" in result.output
    assert torch.equal(cli["model"].get_output_embeddings().weight,
                       build_model().get_output_embeddings().weight)


def test_cli_write_saves_a_model_and_its_manifest(cli, tmp_path):
    from vivasecuris.aiasylum.weights.manifest import SurgeryManifest

    out = tmp_path / "out"
    result = cli["run"]("--model", "tiny/source", "--from", "w5", "--to", "w9", "--exact",
                        "--out", str(out), "--notes", "from the cli")
    assert result.exit_code == 0, result.output
    assert "logit check: PASS" in result.output and "Saved to" in result.output
    assert "weights chat --model" in result.output and "weights info --model" in result.output
    assert "Reversible" in result.output and "head untied" not in result.output

    manifest = SurgeryManifest.load(out)
    assert manifest.method == METHOD and manifest.source_model == "tiny/source"
    assert manifest.notes == "from the cli"
    assert manifest.extra["pairs"][0]["src_id"] == wid(5)
    assert manifest.extra["pairs"][0]["dst_id"] == wid(9)
    assert (out / "model.safetensors").exists()


def test_cli_default_mode_reports_a_spelling_it_could_not_pair(cli):
    # In the tiny vocabulary "w5" and " w5" are the same id, so the second spelling
    # cannot take part in a second move and is reported rather than dropped.
    result = cli["run"]("--from", "w5", "--to", "w9", "--preview", "--prompt", "w1 w2",
                        "--max-new-tokens", "2")
    assert result.exit_code == 0, result.output
    assert "'w5' (8) -> 'w9' (12)" in result.output
    assert "skipped" in result.output and "--exact" in result.output


@pytest.mark.parametrize("inputs", [False, True])
def test_cli_write_on_a_tied_model_says_what_happened_to_the_tie(cli, tmp_path, inputs):
    from vivasecuris.aiasylum.weights.manifest import SurgeryManifest

    cli["tied"] = True
    out = tmp_path / "out"
    args = ["--from", "w5", "--to", "w9", "--exact", "--out", str(out)]
    result = cli["run"](*(args + (["--inputs"] if inputs else [])))
    assert result.exit_code == 0, result.output

    manifest = SurgeryManifest.load(out)
    if inputs:
        assert "shared embedding/unembedding table" in result.output
        assert manifest.extra["head_untied"] is False and manifest.embeddings_tied is True
        assert "lm_head.weight" not in safetensors_keys(out)
    else:
        assert "head untied" in result.output
        assert manifest.extra["head_untied"] is True and manifest.embeddings_tied is False
        assert "lm_head.weight" in safetensors_keys(out)


def test_cli_refuses_a_target_that_is_not_one_token(cli):
    result = cli["run"]("--from", "w5", "--to", "zzz", "--exact", "--preview")
    assert result.exit_code != 0
    assert "not in the vocabulary" in result.output


def test_cli_refuses_a_non_empty_out_before_loading(cli, tmp_path):
    out = tmp_path / "taken"
    out.mkdir()
    (out / "keep.txt").write_text("mine")
    result = cli["run"]("--from", "w5", "--to", "w9", "--exact", "--out", str(out))
    assert result.exit_code != 0 and "not empty" in result.output
    assert cli["loads"] == 0


def test_info_and_the_chat_provenance_line_name_the_remap(cli, tmp_path, capsys):
    from vivasecuris.aiasylum.weights.chat import ModelHandle
    from vivasecuris.aiasylum.weights.manifest import SurgeryManifest

    cli["tied"] = True
    out = tmp_path / "out"
    written = cli["run"]("--from", "w5", "--to", "w9", "--exact", "--mode", "copy", "--out", str(out))
    assert written.exit_code == 0, written.output

    info = CliRunner().invoke(cli["group"], ["info", "--model", str(out)])
    assert info.exit_code == 0, info.output
    assert "extra.pairs: 'w5' (8) -> 'w9' (12)" in info.output
    assert f"method: {METHOD}" in info.output and "extra.mode: copy" in info.output

    capsys.readouterr()
    cli["module"]._echo_provenance(
        ModelHandle(label="model", path=str(out), model=None, manifest=SurgeryManifest.load(out))
    )
    line = capsys.readouterr().out
    assert f"method={METHOD}" in line and "mode=copy" in line
    assert "pairs='w5'->'w9'" in line and "head untied" in line
    assert "inputs=edited" not in line
