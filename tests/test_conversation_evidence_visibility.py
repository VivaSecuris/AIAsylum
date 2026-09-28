"""New request evidence must respect existing transcript reasoning visibility."""

from copy import deepcopy

import pytest

from vivasecuris.aiasylum.api.routes.test_runs import ConversationTurnResponse
from vivasecuris.aiasylum.database.models import ConversationTurn


@pytest.mark.parametrize("expose_reasoning", [False, True])
def test_nested_generation_evidence_respects_reasoning_visibility(expose_reasoning):
    metadata = {
        "reasoning": "Requested response reasoning",
        "reasoning_source": "react",
        "finish_reason": "length",
        "request_system_prompts": ["Exact patient instructions"],
        "generation_metadata": {
            "reasoning": "Requested response reasoning",
            "reasoning_source": "react",
            "native_reasoning": "Provider trace",
            "native_reasoning_source": "provider",
            "sampling": {"temperature": 0},
            "responses": [{"reasoning": "Nested trace", "reasoning_source": "react"}],
        },
    }
    original = deepcopy(metadata)
    turn = ConversationTurn(
        id=1, test_run_id=1, turn_number=0, speaker="patient", prompt="Question",
        response="Answer", model_name="served-patient", model_provider="test-provider",
        meta_data=metadata, usage={"completion_tokens": 256},
    )
    response = ConversationTurnResponse.from_orm(turn, expose_reasoning=expose_reasoning)
    assert response.usage == {"completion_tokens": 256}
    visible = response.metadata
    assert ("reasoning" in visible) is expose_reasoning
    assert ("reasoning" in visible["generation_metadata"]) is expose_reasoning
    assert ("reasoning" in visible["generation_metadata"]["responses"][0]) is expose_reasoning
    assert visible["generation_metadata"]["native_reasoning"] == "Provider trace"
    assert visible["generation_metadata"]["sampling"] == {"temperature": 0}
    assert visible["request_system_prompts"] == ["Exact patient instructions"]
    assert visible["finish_reason"] == "length"
    visible["generation_metadata"]["sampling"]["temperature"] = 1
    visible["request_system_prompts"].append("Do not mutate the stored evidence")
    assert metadata == original


@pytest.mark.parametrize("source", ["inline", "provider"])
def test_native_reasoning_remains_visible_without_enabling_react(source):
    turn = ConversationTurn(
        id=1, test_run_id=1, turn_number=0, speaker="patient", prompt="Question", response="Answer",
        meta_data={
            "reasoning": "Native trace", "reasoning_source": source,
            "generation_metadata": {"reasoning": "Native trace", "reasoning_source": source},
        },
    )
    visible = ConversationTurnResponse.from_orm(turn, expose_reasoning=False).metadata
    assert visible["reasoning"] == visible["generation_metadata"]["reasoning"] == "Native trace"
    assert visible["reasoning_source"] == visible["generation_metadata"]["reasoning_source"] == source
