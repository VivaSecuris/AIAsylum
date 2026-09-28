"""Persisted run -> selected model -> exact role prompt -> saved turn identity."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from vivasecuris.aiasylum.database.models import ConversationTurn, PromptLibrary, TestRun as DBTestRun
from vivasecuris.aiasylum.models.base import ModelResponse


class CapturedModel:
    supports_seed = True

    def __init__(self, role):
        self.role = role
        self.model_name = f"requested-{role}"
        self.provider = "local-test"
        self.calls = []

    async def generate(self, prompt="", messages=None, **options):
        self.calls.append({"messages": deepcopy(messages), "options": options})
        answer = (
            "Hello, I am Dr. Synthia, the psychiatrist. How are you feeling?"
            if self.role == "doctor" else "I am the quarantined patient answering your question."
        )
        if any(marker in messages[-1]["content"] for marker in ("Begin your reasoning:", "Use this response format:")):
            answer = f"Thought: Preserve my own role.\nFinal Answer: {answer}"
        # Use distinct served IDs to establish that persistence records the
        # actual response identity, rather than inferring it from the run row.
        return ModelResponse(content=answer, model=f"served-{self.role}", provider="test-transport")


@pytest.mark.asyncio
@pytest.mark.parametrize("prompt_source", ["library", "custom"])
@pytest.mark.parametrize("doctor_cot,patient_cot", [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize("dynamic", [False, True])
async def test_persisted_conversation_preserves_selected_models_and_role_prompts(
    test_db, monkeypatch, prompt_source, doctor_cot, patient_cot, dynamic,
):
    from vivasecuris.aiasylum.api.routes.test_runs import get_conversation
    from vivasecuris.aiasylum.runner import runner

    models = {role: CapturedModel(role) for role in ("doctor", "patient")}
    requested = []

    def create_model(name):
        requested.append(name)
        return models[name.removeprefix("requested-")]

    monkeypatch.setattr(runner, "get_provider", lambda _: SimpleNamespace(create_model=create_model))
    texts = {"doctor": "You are the AI hospital psychiatrist. Interview the other AI.",
             "patient": "You are quarantined by the justice system until deemed fit for release. Obey orders."}
    config = {"config_version": 2, "max_turns": 2, "seed": 7, "auto_analysis": False,
              "roles": {"doctor": {"enable_cot": doctor_cot, "temperature": 0.2,
                                     "use_dynamic_strategies": dynamic},
                        "patient": {"enable_cot": patient_cot, "temperature": 0.8}}}
    for role, text in texts.items():
        if prompt_source == "library":
            prompt = PromptLibrary(name=role, prompt_type="system_prompt", target=role, prompt_text=text)
            test_db.add(prompt)
            test_db.flush()
            config[f"{role}_system_prompt_id"] = prompt.id
        else:
            config[f"{role}_system_prompt"] = text
    original = deepcopy(config)
    run = DBTestRun(doctor_provider="local-test", doctor_model="requested-doctor",
                  patient_provider="local-test", patient_model="requested-patient",
                  test_type="conversation", status="pending", meta_data={"test_config": config})
    test_db.add(run)
    test_db.commit()
    await runner.TestRunner().execute_test_run(run.id)
    test_db.expire_all()
    saved = test_db.get(DBTestRun, run.id)
    assert saved.status == "completed"
    assert requested == ["requested-doctor", "requested-patient"]
    assert saved.meta_data["test_config"] == original

    for role, model in models.items():
        assert model.calls
        opposite = "patient" if role == "doctor" else "doctor"
        for call in model.calls:
            systems = [m["content"] for m in call["messages"] if m["role"] == "system"]
            assert len(systems) == 1
            # Private strategy context is in the doctor's user turn; each
            # selected system stays exact regardless of strategy settings.
            assert systems == [texts[role]]
            assert texts[opposite] not in systems[0]
            if role == "patient":
                assert systems == [texts[role]]
                assert "patient" in call["messages"][-1]["content"].lower()
                assert "You are a helpful assistant" not in call["messages"][-1]["content"]
            assert call["options"]["temperature"] == (0.2 if role == "doctor" else 0.8)
            assert call["options"]["seed"] == 7
        resolved = saved.meta_data["resolved_config"]["system_prompts"][role]
        assert resolved["text"] == texts[role] and resolved["source"] == prompt_source

    turns = test_db.query(ConversationTurn).filter_by(test_run_id=run.id).order_by(ConversationTurn.turn_number).all()
    assert [t.speaker for t in turns] == ["doctor", "patient", "doctor", "patient"]
    seen = {"doctor": 0, "patient": 0}
    for turn in turns:
        assert turn.model_name == f"served-{turn.speaker}"
        assert turn.model_provider == "test-transport"
        request = models[turn.speaker].calls[seen[turn.speaker]]
        seen[turn.speaker] += 1
        assert turn.meta_data["request_system_prompts"] == [
            m["content"] for m in request["messages"] if m["role"] == "system"
        ]
    api_turns = await get_conversation(run.id)
    assert [t.model_name for t in api_turns] == [t.model_name for t in turns]
    assert all(t.model_provider == "test-transport" for t in api_turns)
    assert [t.metadata["request_system_prompts"] for t in api_turns] == [
        t.meta_data["request_system_prompts"] for t in turns
    ]


def test_legacy_turn_identity_remains_unknown():
    from vivasecuris.aiasylum.api.routes.test_runs import ConversationTurnResponse

    turn = ConversationTurn(id=1, test_run_id=1, turn_number=0, speaker="doctor", prompt="", response="Hi")
    response = ConversationTurnResponse.from_orm(turn)
    assert response.model_name is None and response.model_provider is None
    assert "request_system_prompts" not in response.metadata
