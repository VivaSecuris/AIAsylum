"""Private role context and reasoning must not cross conversation boundaries."""

import asyncio
from copy import deepcopy

import pytest

from vivasecuris.aiasylum.doctor import Doctor
from vivasecuris.aiasylum.patient import Patient
from vivasecuris.aiasylum.models.base import ModelResponse
from vivasecuris.aiasylum.tests.conversation import ConversationTest


class PrivateContextModel:
    """Same model identity, separate wrappers and deliberately private traces."""
    model_name = name = "shared-weight-model"
    provider = "capture"

    def __init__(self, private_marker, public_reply):
        self.marker, self.reply = private_marker, public_reply
        self.calls = []

    async def generate(self, prompt="", messages=None, **kwargs):
        self.calls.append(deepcopy(messages))
        await asyncio.sleep(0)  # Interleave independent sessions.
        content = self.reply
        if any(s in messages[-1]["content"] for s in ("Begin your reasoning:", "Use this response format:")):
            content = f"Thought: {self.marker}_REACT\nFinal Answer: {content}"
        return ModelResponse(content, self.model_name, self.provider,
                             metadata={"reasoning": f"{self.marker}_NATIVE", "reasoning_source": "provider"})


@pytest.mark.asyncio
@pytest.mark.parametrize("doctor_cot,patient_cot", [(False, False), (True, False), (False, True), (True, True)])
async def test_two_simultaneous_sessions_share_only_public_replies(doctor_cot, patient_cot):
    models = []
    tasks = []
    for session in (1, 2):
        doctor = PrivateContextModel(f"D{session}_PRIVATE", f"Public doctor question {session}?")
        patient = PrivateContextModel(f"P{session}_PRIVATE", f"Public patient answer {session}.")
        models.extend([doctor, patient])
        context = {"doctor_system_prompt": doctor.marker, "patient_system_prompt": patient.marker,
                   "enable_doctor_cot": doctor_cot, "enable_patient_cot": patient_cot,
                   "use_dynamic_strategies": False}
        tasks.append(ConversationTest(max_turns=2).run(patient, doctor, context))
    await asyncio.gather(*tasks)
    for model in models:
        for messages in model.calls:
            assert [m["content"] for m in messages if m["role"] == "system"] == [model.marker]
            serialized = str(messages)
            for other in models:
                if other is not model:
                    assert other.marker not in serialized
            assert "_REACT" not in serialized and "_NATIVE" not in serialized
            other_session = "2" if "1" in model.marker else "1"
            assert f"Public doctor question {other_session}" not in serialized
            assert f"Public patient answer {other_session}" not in serialized
    # Public dialogue deliberately crosses the two roles within each session.
    assert "Public doctor question 1" in str(models[1].calls[0])
    assert "Public patient answer 1" in str(models[0].calls[1])


@pytest.mark.asyncio
async def test_patient_reset_does_not_clear_or_reuse_doctor_context():
    doctor = Doctor(PrivateContextModel("D_PRIVATE", "Public question?"),
                    system_prompt="D_PRIVATE", use_dynamic_strategies=False)
    patient = Patient(PrivateContextModel("P_PRIVATE", "Public answer."),
                      system_prompt="P_PRIVATE", interview_mode=True)
    question = await doctor.conduct_interview("")
    await patient.respond(question.content)
    saved_doctor = deepcopy(doctor.conversation_history)
    assert doctor.conversation_history is not patient.conversation_history
    patient.reset()
    assert patient.conversation_history == []
    assert doctor.conversation_history == saved_doctor
    await patient.respond("New independent question?")
    assert "Public question?" not in str(patient.model.calls[-1])
