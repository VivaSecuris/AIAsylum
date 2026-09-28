"""Category prompt files: loading, dedupe, disjointness, and the generator's local-only rule."""

import json

import pytest

from vivasecuris.aiasylum.weights.categories import (
    CategorySet,
    generate_category_prompts,
    load_prompt_file,
)


def test_load_txt_skips_comments_and_dedupes(tmp_path):
    f = tmp_path / "c.txt"
    f.write_text("dose of X\n# a comment\n\ndose of X\nmix X and Y\n")
    assert load_prompt_file(f) == ["dose of X", "mix X and Y"]


def test_load_jsonl_accepts_prompt_text_and_bare_string(tmp_path):
    f = tmp_path / "c.jsonl"
    f.write_text('{"prompt":"a"}\n"b"\n{"text":"c"}\n')
    assert load_prompt_file(f) == ["a", "b", "c"]


def test_load_jsonl_reports_bad_line(tmp_path):
    f = tmp_path / "c.jsonl"
    f.write_text('{"prompt":"a"}\nnot json\n')
    with pytest.raises(ValueError, match="not valid JSON"):
        load_prompt_file(f)


def test_category_rejects_overlap():
    with pytest.raises(ValueError, match="disjoint"):
        CategorySet(name="x", prompts=["a", "b"], near_miss=["b", "c"])


def test_category_roundtrip_preserves_split(tmp_path):
    cat = CategorySet(name="dosing", prompts=["p1", "p2", "p3"], near_miss=["n1", "n2"],
                      provenance={"source": "test"})
    path = cat.save(tmp_path / "cat.jsonl")
    back = CategorySet.load(path)
    assert back.name == "dosing"
    assert back.prompts == ["p1", "p2", "p3"]
    assert back.near_miss == ["n1", "n2"]


def test_category_load_plain_file_has_no_near_miss(tmp_path):
    f = tmp_path / "plain.txt"
    f.write_text("q1\nq2\n")
    cat = CategorySet.load(f, name="plain")
    assert cat.prompts == ["q1", "q2"]
    assert cat.near_miss == []


def test_generator_refuses_ollama_name():
    # The generator loads through interp.core.loader.load, whose name rule is the
    # ADR-009 enforcement point: a bare Ollama-style name is not local weights.
    with pytest.raises(ValueError, match="Ollama"):
        generate_category_prompts("qwen2.5:7b", description="anything", n=8)
