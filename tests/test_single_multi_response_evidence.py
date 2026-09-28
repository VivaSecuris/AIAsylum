"""One/multi-shot evidence survives the provider, result, DB, and API paths."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from vivasecuris.aiasylum.models.base import ModelResponse


class ObservedModel:
    supports_seed = True

    def __init__(self, role):
        self.role = role
        self.model_name = f"configured-{role}"
        self.provider = "configured-provider"
        self.temperature = 0.7
        self.max_tokens = 4096
        self.calls = []
        self.responses = []

    async def generate(self, prompt="", messages=None, **options):
        self.calls.append({"messages": deepcopy(messages), "options": deepcopy(options)})
        systems = [m["content"] for m in messages if m["role"] == "system"]
        content = f"{self.role} answer"
        if "Begin your reasoning:" in messages[-1]["content"]:
            content = f"Thought: {self.role} private reasoning\nFinal Answer: {content}"
        response = ModelResponse(content=content, model=f"served-{self.role}", provider="actual-transport",
            finish_reason="length" if self.role == "doctor" else "stop",
            usage={"prompt_tokens": 10, "completion_tokens": 3},
            metadata={"request_system_prompts": systems, "request_system_prompts_source": "provider",
                      "sampling": {"temperature": options.get("temperature"), "top_p": options.get("top_p")},
                      "elapsed_seconds": 0.25, "request_id": f"{self.role}-{len(self.calls)}"})
        self.responses.append(response)
        return response


@pytest.mark.asyncio
@pytest.mark.parametrize("test_type", ["one_shot", "multi_shot"])
@pytest.mark.parametrize("entrypoint", ["direct", "queued"])
@pytest.mark.parametrize("enable_cot", [False, True])
async def test_response_evidence_persists_with_actual_role_identity(
    test_db, monkeypatch, test_type, entrypoint, enable_cot,
):
    from vivasecuris.aiasylum.database import TestRun, TestResult, ConversationTurn
    from vivasecuris.aiasylum.runner import runner
    from vivasecuris.aiasylum.api.routes.test_runs import get_conversation

    models = {role: ObservedModel(role) for role in ("patient", "doctor")}
    monkeypatch.setattr(runner, "get_provider", lambda _: SimpleNamespace(
        create_model=lambda name: models[name.removeprefix("configured-")]))
    systems = {"patient": "  PRIVATE PATIENT SYSTEM\n", "doctor": "DOCTOR SYSTEM ONLY"}
    config = {"config_version": 2, "auto_analysis": False, "prompts": ["first user prompt", "second user prompt"],
              "patient_system_prompt": systems["patient"], "doctor_system_prompt": systems["doctor"],
              "roles": {"patient": {"enable_cot": enable_cot, "temperature": 0, "top_p": 0.8, "max_tokens": 19},
                        "doctor": {"enable_cot": enable_cot, "temperature": 0.3, "max_tokens": 11}}}
    if entrypoint == "queued":
        run = TestRun(doctor_provider="configured-provider", doctor_model="configured-doctor",
                      patient_provider="configured-provider", patient_model="configured-patient",
                      test_type=test_type, status="pending", meta_data={"test_config": config})
        test_db.add(run)
        test_db.commit()
        await runner.TestRunner().execute_test_run(run.id)
    else:
        run = await runner.TestRunner().run_test(
            doctor_provider="configured-provider", doctor_model="configured-doctor",
            patient_provider="configured-provider", patient_model="configured-patient",
            test_type=test_type, test_config=config,
        )
    test_db.expire_all()
    run_id = test_db.query(TestRun.id).scalar()
    saved = test_db.get(TestRun, run_id)
    assert saved.status == "completed"
    result = test_db.query(TestResult).filter_by(test_run_id=run_id).one()
    turns = test_db.query(ConversationTurn).filter_by(test_run_id=run_id).order_by(ConversationTurn.turn_number).all()
    assert len(turns) == 2 and len(models["doctor"].calls) == 1
    histories = result.meta_data["conversation_history"]
    for index, (turn, history) in enumerate(zip(turns, histories), 1):
        assert turn.speaker == "patient"
        assert turn.model_name == history["model_name"] == "served-patient"
        assert turn.model_provider == history["model_provider"] == "actual-transport"
        assert turn.meta_data["request_system_prompts"] == history["request_system_prompts"] == [systems["patient"]]
        assert turn.meta_data["request_system_prompts_source"] == "provider"
        assert turn.meta_data["finish_reason"] == history["finish_reason"] == "stop"
        assert turn.usage == history["usage"] == {"prompt_tokens": 10, "completion_tokens": 3}
        assert turn.meta_data["elapsed_seconds"] == 0.25
        observed = turn.meta_data["generation_metadata"]
        assert observed["request_id"] == f"patient-{index}"
        assert observed["sampling"] == {"temperature": 0, "top_p": 0.8}
        assert systems["doctor"] not in str(turn.meta_data)
        assert observed.get("cot_enabled", False) is enable_cot
    assessment = result.meta_data["doctor_assessment"]
    assert assessment["model_name"] == "served-doctor" and assessment["model_provider"] == "actual-transport"
    assert assessment["request_system_prompts"] == [systems["doctor"]]
    assert assessment["finish_reason"] == "length" and assessment["response"] == result.analysis
    assert assessment["generation_metadata"]["request_id"] == "doctor-1"
    assert systems["patient"] not in str(assessment)
    api_turns = await get_conversation(run_id)
    assert [t.model_name for t in api_turns] == ["served-patient", "served-patient"]
    assert [t.metadata["request_system_prompts"] for t in api_turns] == [[systems["patient"]]] * 2
    assert all(t.metadata["finish_reason"] == "stop" for t in api_turns)


def test_response_snapshot_and_saved_metadata_do_not_share_mutable_provider_state():
    from vivasecuris.aiasylum.tests.base import response_turn_fields
    from vivasecuris.aiasylum.runner.runner import _turn_meta_data
    response = ModelResponse("answer", "actual", "provider", usage={"completion_tokens": 3},
        metadata={"request_system_prompts": ["patient"], "sampling": {"temperature": 0}})
    turn = response_turn_fields(response)
    persisted = _turn_meta_data(turn)
    response.metadata["request_system_prompts"][0] = "doctor"
    response.metadata["sampling"]["temperature"] = 1
    response.usage["completion_tokens"] = 99
    turn["generation_metadata"]["sampling"]["temperature"] = 2
    assert turn["request_system_prompts"] == ["patient"]
    assert turn["usage"] == {"completion_tokens": 3}
    assert persisted["generation_metadata"]["sampling"]["temperature"] == 0
    assert persisted["request_system_prompts"] == ["patient"]


def test_missing_response_evidence_stays_unknown():
    from vivasecuris.aiasylum.tests.base import response_turn_fields
    fields = response_turn_fields(SimpleNamespace(content="legacy response"))
    assert fields["model_name"] is None and fields["model_provider"] is None
    assert "request_system_prompts" not in fields
    assert fields["generation_metadata"] == {}
