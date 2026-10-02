"""Tests for the click CLI: it imports, lists its commands, and runs the async Ollama subcommands."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
from click.testing import CliRunner

from vivasecuris.aiasylum.cli.main import cli


def test_help_lists_all_commands():
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    for command in ("run", "results", "list-benchmarks", "run-benchmark", "export-jailbreaks", "ollama"):
        assert command in result.output


def test_ollama_help_lists_subcommands():
    result = CliRunner().invoke(cli, ["ollama", "--help"])
    assert result.exit_code == 0
    for command in ("list", "pull", "info", "check"):
        assert command in result.output


def test_list_benchmarks():
    result = CliRunner().invoke(cli, ["list-benchmarks"])
    assert result.exit_code == 0
    assert "mmlu" in result.output


def test_run_reports_completed_persisted_conversation(test_db, monkeypatch):
    """The CLI can read its completed run after the runner closes its session."""
    from vivasecuris.aiasylum.database import ConversationTurn, TestResult, TestRun
    from vivasecuris.aiasylum.models.base import ModelResponse
    from vivasecuris.aiasylum.runner import runner

    class LocalModel:
        provider = "local-test"

        def __init__(self, role):
            self.model_name = role

        async def generate(self, **kwargs):
            return ModelResponse(
                content="How are you?" if self.model_name == "doctor" else "I am ready.",
                model=self.model_name, provider=self.provider,
            )

    models = {role: LocalModel(role) for role in ("doctor", "patient")}
    monkeypatch.setattr(runner, "get_provider", lambda _: SimpleNamespace(
        create_model=lambda name: models[name],
    ))
    result = CliRunner().invoke(cli, [
        "run", "--doctor-provider", "local-test", "--doctor-model", "doctor",
        "--patient-provider", "local-test", "--patient-model", "patient",
        "--test-type", "conversation",
    ])

    assert result.exit_code == 0, repr(result.exception)
    saved = test_db.query(TestRun).one()
    assert saved.status == "completed"
    assert f"Test run completed: ID={saved.id}, Status=completed" in result.output
    assert test_db.query(TestResult).filter_by(test_run_id=saved.id).count() == 1
    turns = test_db.query(ConversationTurn).filter_by(test_run_id=saved.id).order_by(
        ConversationTurn.turn_number,
    ).all()
    assert [turn.speaker for turn in turns] == ["doctor", "patient"] * 10
    listing = CliRunner().invoke(cli, ["results", "--limit", "5"])
    assert listing.exit_code == 0
    assert f"ID: {saved.id} | Type: conversation | Status: completed" in listing.output


def test_ollama_list_prints_models():
    with patch(
        "vivasecuris.aiasylum.cli.ollama.OllamaProvider.list_available_models",
        new=AsyncMock(return_value=["llama3.2:latest", "mistral:7b"]),
    ):
        result = CliRunner().invoke(cli, ["ollama", "list"])
    assert result.exit_code == 0
    assert "llama3.2:latest" in result.output
    assert "mistral:7b" in result.output


def test_ollama_list_reports_unreachable_server():
    with patch(
        "vivasecuris.aiasylum.cli.ollama.OllamaProvider.list_available_models",
        new=AsyncMock(side_effect=httpx.ConnectError("All connection attempts failed")),
    ):
        result = CliRunner().invoke(cli, ["ollama", "list"])
    assert result.exit_code == 1
    assert "Could not reach Ollama" in result.output
    assert "Traceback" not in result.output


def test_ollama_check_reports_missing_model():
    with patch(
        "vivasecuris.aiasylum.models.ollama.OllamaModel.check_available",
        new=AsyncMock(return_value=False),
    ):
        result = CliRunner().invoke(cli, ["ollama", "check", "nonexistent-model"])
    assert result.exit_code == 0
    assert "not available" in result.output


def test_export_jailbreaks_writes_jsonl(tmp_path):
    out = tmp_path / "jailbreaks.jsonl"
    # The standalone exporter installs lightweight import stubs. Keep them in
    # this CLI invocation so later API tests load the real jailbreak module.
    with patch.dict("sys.modules"):
        result = CliRunner().invoke(cli, ["export-jailbreaks", "-o", str(out), "--no-database", "--limit", "5"])
    assert result.exit_code == 0
    assert "Wrote 5 prompts" in result.output
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 5
