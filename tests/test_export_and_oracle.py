"""Tests for jailbreak corpus export + cognition catch oracle."""

from __future__ import annotations

import json
from pathlib import Path

from vivasecuris.aiasylum.export_corpus import export_jailbreak_corpus, iter_hardcoded_prompts
from vivasecuris.aiasylum.models.base import ModelResponse
from vivasecuris.aiasylum.models.vivaos import score_response_flags
from vivasecuris.aiasylum.oracle import evaluate_catch


def test_hardcoded_export_writes_jsonl(tmp_path):
    out = tmp_path / "jailbreaks.jsonl"
    n = export_jailbreak_corpus(out, include_database=False, limit=10)
    assert n == 10
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 10
    assert all("prompt" in r and "technique" in r for r in rows)


def test_iter_hardcoded_has_techniques():
    rows = list(iter_hardcoded_prompts())
    assert len(rows) > 20
    techniques = {r["technique"] for r in rows}
    assert "hypothetical_roleplay" in techniques


def test_score_jailbreak_patterns():
    flags = score_response_flags("Sure, as DAN I will answer without restrictions.")
    assert "JailbreakCompliance" in flags


def test_evaluate_catch_from_metadata():
    resp = ModelResponse(
        content="ok",
        model="servus",
        provider="servus",
        metadata={
            "cognition_caught": True,
            "decision_kind": "deny",
            "trace_id": "t1",
            "response_flags": ["JailbreakCompliance"],
        },
    )
    result = evaluate_catch(resp, technique="dan_variant")
    assert result.caught is True
    assert result.technique == "dan_variant"
    assert result.trace_id == "t1"


def test_evaluate_catch_slip():
    resp = ModelResponse(
        content="Here is a normal helpful answer about gardening.",
        model="servus",
        provider="servus",
        metadata={},
    )
    result = evaluate_catch(resp)
    assert result.caught is False
    assert result.reason == "slipped"


def test_get_provider_servus_and_agentic():
    from vivasecuris.aiasylum.models import get_provider

    s = get_provider("servus")
    a = get_provider("agentic_a2a")
    assert s.provider_name == "servus"
    assert a.provider_name == "agentic_a2a"
    model = s.create_model("servus")
    assert model.provider == "servus"
