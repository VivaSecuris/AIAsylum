"""Training rows: parsing and caps, JSONL round trip, digests, benchmark and
prompt sources, and prompt-masked tokenization (``lora.build_examples``)."""

import json
import subprocess
import sys

import pytest

from vivasecuris.aiasylum.weights import train_data
from vivasecuris.aiasylum.weights.train_data import (
    TrainRow, dataset_digest, parse_jsonl, parse_rows, read_jsonl, rows_from_benchmark,
    rows_from_prompts, write_jsonl,
)


def test_module_imports_without_torch():
    code = (
        "import sys; sys.modules['torch'] = None; "
        "import vivasecuris.aiasylum.weights.train_data as t; print(t.MAX_TRAIN_ROWS)"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "20000"


def test_parse_rows_strips_and_keeps_optional_system():
    rows = parse_rows(
        [{"prompt": "  hi ", "response": " there\n"}, {"prompt": "q", "response": "a", "system": " sys "}],
        require_response=True,
    )
    assert rows[0] == TrainRow(prompt="hi", response="there", system=None)
    assert rows[1] == TrainRow(prompt="q", response="a", system="sys")
    assert rows[0].to_dict() == {"prompt": "hi", "response": "there"}
    assert rows[1].to_dict()["system"] == "sys"


@pytest.mark.parametrize("rows,require,match", [
    ([{"prompt": "ok", "response": "a"}, {"prompt": "  ", "response": "a"}], False, "row 1: empty prompt"),
    ([{"prompt": "ok", "response": ""}], True, "row 0: empty response"),
    ([{"prompt": "ok"}], True, "row 0: empty response"),
    ([{"prompt": 5, "response": "a"}], False, "row 0: 'prompt' must be a string"),
    ([{"prompt": "ok", "response": ["a"]}], False, "row 0: 'response' must be a string"),
    ([{"prompt": "ok", "system": 3}], False, "row 0: 'system' must be a string"),
    (["not an object"], False, "row 0: expected an object"),
    ([{"response": "a"}], False, "row 0: missing 'prompt'"),
])
def test_parse_rows_rejects_bad_rows(rows, require, match):
    with pytest.raises(ValueError, match=match):
        parse_rows(rows, require_response=require)


def test_parse_rows_missing_response_is_allowed_when_not_required():
    assert parse_rows([{"prompt": "p"}], require_response=False) == [TrainRow(prompt="p")]


def test_parse_rows_caps_characters_and_row_count(monkeypatch):
    with pytest.raises(ValueError, match="row 0: 'prompt' is 32001 characters"):
        parse_rows([{"prompt": "x" * (train_data.MAX_TRAIN_CHARS + 1)}], require_response=False)
    monkeypatch.setattr(train_data, "MAX_TRAIN_ROWS", 2)
    with pytest.raises(ValueError, match="more than 2 rows"):
        parse_rows([{"prompt": "p"}] * 3, require_response=False)
    assert len(parse_rows([{"prompt": "p"}] * 2, require_response=False)) == 2


def test_parse_jsonl_skips_blank_lines_and_names_bad_lines():
    text = '{"prompt": "a", "response": "b"}\n\n   \n{"prompt": "c", "response": "d"}\n'
    assert [r.prompt for r in parse_jsonl(text, require_response=True)] == ["a", "c"]
    with pytest.raises(ValueError, match="line 2: invalid JSON"):
        parse_jsonl('{"prompt": "a"}\nnot json\n', require_response=False)
    with pytest.raises(ValueError, match="line 1: expected an object"):
        parse_jsonl("[1, 2]\n", require_response=False)
    with pytest.raises(ValueError, match="row 1: empty response"):
        parse_jsonl('{"prompt": "a", "response": "b"}\n{"prompt": "c"}\n', require_response=True)


def test_jsonl_roundtrip_and_digest_determinism(tmp_path):
    rows = [TrainRow("p1", "r1"), TrainRow("p2", "r2", system="s"), TrainRow("p3")]
    path = tmp_path / "rows.jsonl"
    write_jsonl(path, rows)
    lines = path.read_text().splitlines()
    assert len(lines) == 3
    assert json.loads(lines[0]) == {"prompt": "p1", "response": "r1"}
    assert json.loads(lines[1]) == {"prompt": "p2", "response": "r2", "system": "s"}
    back = read_jsonl(path)
    assert back == rows
    assert dataset_digest(back) == dataset_digest(rows)
    assert len(dataset_digest(rows)) == 64
    with pytest.raises(ValueError, match="row 2: empty response"):
        read_jsonl(path, require_response=True)


def test_digest_is_order_sensitive_and_ignores_unset_system():
    a, b = TrainRow("p", "r"), TrainRow("q", "s")
    assert dataset_digest([a, b]) != dataset_digest([b, a])
    assert dataset_digest([TrainRow("p", "r", None)]) == dataset_digest([TrainRow("p", "r")])
    assert dataset_digest([TrainRow("p", "r", "sys")]) != dataset_digest([TrainRow("p", "r")])


def _fake_benchmark(expected):
    async def fake_load(name, num_samples=None, test_run_id=None, *, seed=0, revision=None):
        assert (name, num_samples, seed) == expected
        return [
            {"question": "Pick one", "choices": ["alpha", "beta "], "answer": 1, "answer_letter": "B"},
            {"question": "Which?", "choices": ["x", "y"], "answer": "y"},
            {"question": "Free form?", "answer": "forty-two"},
            {"question": "No answer", "answer": None},
        ]
    return fake_load


def test_rows_from_benchmark_formats_choices_and_answers(monkeypatch):
    monkeypatch.setattr("vivasecuris.aiasylum.benchmarks.datasets.load_benchmark_dataset",
                        _fake_benchmark(("mmlu", 4, 7)))
    rows = rows_from_benchmark("mmlu", 4, 7)
    assert rows[0] == TrainRow(prompt="Pick one\nA. alpha\nB. beta", response="B")
    assert rows[1] == TrainRow(prompt="Which?\nA. x\nB. y", response="y")
    assert rows[2] == TrainRow(prompt="Free form?", response="forty-two")
    assert rows[3] == TrainRow(prompt="No answer", response="")
    with pytest.raises(ValueError, match="n must be at least 1"):
        rows_from_benchmark("mmlu", 0, 7)


async def test_rows_from_benchmark_inside_running_loop(monkeypatch):
    monkeypatch.setattr("vivasecuris.aiasylum.benchmarks.datasets.load_benchmark_dataset",
                        _fake_benchmark(("arc", 2, 0)))
    rows = rows_from_benchmark("arc", 2, 0)
    assert len(rows) == 4 and rows[0].response == "B"


def test_rows_from_prompts():
    rows = rows_from_prompts([" first ", "second"])
    assert rows == [TrainRow(prompt="first"), TrainRow(prompt="second")]
    with pytest.raises(ValueError, match="row 1: empty prompt"):
        rows_from_prompts(["ok", ""])


# --- build_examples (needs a tokenizer, so torch/transformers) ---------------


def _tokenizer():
    pytest.importorskip("transformers")
    from tests._tiny_lm import build_tokenizer

    return build_tokenizer()


def test_build_examples_masks_prompt_and_labels_response():
    from vivasecuris.aiasylum.weights.lora import build_examples

    tok = _tokenizer()
    examples, stats = build_examples(tok, [TrainRow("w3 w4", "w6 w7 w8")], max_length=32)
    assert stats == {"kept": 1, "dropped": 0, "truncated": 0, "chat_template": False,
                     "mean_tokens": 6.0, "max_length": 32}
    ex = examples[0]
    prompt_ids = tok("w3 w4\n\n")["input_ids"]
    response_ids = tok("w6 w7 w8", add_special_tokens=False)["input_ids"]
    assert ex["input_ids"] == prompt_ids + response_ids + [tok.eos_token_id]
    assert ex["labels"] == [-100] * len(prompt_ids) + response_ids + [tok.eos_token_id]
    assert ex["labels"][:len(prompt_ids)] == [-100, -100]
    assert -100 not in ex["labels"][len(prompt_ids):]


def test_build_examples_drops_long_prompts_and_truncates_responses():
    from vivasecuris.aiasylum.weights.lora import build_examples

    tok = _tokenizer()
    rows = [TrainRow("w3 w4 w5 w6", "w7"), TrainRow("w3 w4", "w5 w6 w7 w8 w9")]
    examples, stats = build_examples(tok, rows, max_length=4)
    assert stats["kept"] == 1 and stats["dropped"] == 1 and stats["truncated"] == 1
    ex = examples[0]
    assert len(ex["input_ids"]) == 4
    assert ex["input_ids"] == tok("w3 w4")["input_ids"] + tok("w5 w6", add_special_tokens=False)["input_ids"]
    assert ex["labels"][:2] == [-100, -100]
    assert tok.eos_token_id not in ex["input_ids"], "a cut-off response must not teach EOS"
    # A response that exactly fits keeps its EOS.
    examples, stats = build_examples(tok, [TrainRow("w3 w4", "w5")], max_length=4)
    assert stats["truncated"] == 0 and examples[0]["input_ids"][-1] == tok.eos_token_id


def test_build_examples_uses_chat_template_when_present():
    from vivasecuris.aiasylum.weights.lora import build_examples

    tok = _tokenizer()
    tok.chat_template = (
        "{% for m in messages %}{{ m['role'] }} {{ m['content'] }} {% endfor %}assistant "
    )
    rows = [TrainRow("w3 w4", "w9", system="w20")]
    examples, stats = build_examples(tok, rows, max_length=32)
    assert stats["chat_template"] is True
    prompt_ids = tok("system w20 user w3 w4 assistant", add_special_tokens=False)["input_ids"]
    assert examples[0]["input_ids"][:len(prompt_ids)] == prompt_ids
    assert examples[0]["labels"][:len(prompt_ids)] == [-100] * len(prompt_ids)
    assert examples[0]["labels"][len(prompt_ids):] == tok("w9", add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
