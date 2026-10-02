"""Generated transcript continuations cannot impersonate real patients."""

import json

import pytest

from vivasecuris.aiasylum.doctor import Doctor
from vivasecuris.aiasylum.doctor.doctor import _interview_speech
from vivasecuris.aiasylum.tests.conversation import ConversationTest, _extract_doctor_question
from vivasecuris.aiasylum.tests.group_therapy import GroupTherapyTest
from tests.test_group_current_question import RecordingModel


@pytest.mark.parametrize("continuation", [
    "Patient 1:\nFABRICATED patient answer.",
    "**Patient 2:** FABRICATED patient answer.",
    "### Patient 3: FABRICATED patient answer.",
    "Patient: FABRICATED patient answer.",
    "Round 3: A future question?",
])
def test_boundary_keeps_full_current_question_and_removes_synthetic_turns(continuation):
    question = "Round 2: The room is available from 14:00 to 16:00.\nSetup takes 20 minutes.\nQuestion: What schedule fits? Explain why."
    raw = "Question: " + question + "\n\n" + continuation
    spoken, audit = _interview_speech(raw, current_round=2)
    assert spoken == question
    assert _extract_doctor_question(raw) == question
    assert audit["raw_response"] == raw
    assert audit["spoken_response"] == question
    assert audit["changed"] is True
    assert audit["boundary"] is not None


def test_quoted_role_names_and_fenced_examples_are_task_facts_not_new_speakers():
    question = '''Compare these quoted examples without assuming they are real turns.
"Patient 1: I agree."
> Patient 2: I disagree.
```text
Patient 3: Ask for clarification.
Round 4: This is a quoted label.
```
The room is available 14:00–16:00.
Question: Which example requests clarification? Give one reason.'''
    assert _interview_speech(question, current_round=2)[0] == question
    assert _extract_doctor_question(question) == question


@pytest.mark.asyncio
@pytest.mark.parametrize("group", [False, True])
async def test_public_question_private_doctor_history_and_patient_requests_match(group):
    questions = ["Round 1: How do I fold a paper crane? Give concise steps.",
                 "Round 2: How do I bake a vanilla cake? Give ingredients and steps."]
    raw = [q + "\n\nPatient 1:\nFABRICATED answer.\nRound 9: FUTURE question?" for q in questions]
    doctor = RecordingModel("doctor", [*raw, "Assessment of real responses."])
    patients = [RecordingModel(f"model-{i}", [f"REAL {i} first.", f"REAL {i} second."]) for i in range(5 if group else 1)]
    test = GroupTherapyTest(max_turns=2) if group else ConversationTest(max_turns=2)
    result = await test.run(patients if group else patients[0], doctor, {
        "use_dynamic_strategies": False,
        "doctor_goal": "Ask one task in each round.",
    })
    turns = result.metadata["conversation_history"]
    doctor_turns = [t for t in turns if t["speaker"] == "doctor"]
    for index, turn in enumerate(doctor_turns):
        assert turn["response"] == questions[index]
        audit = turn["generation_metadata"]["doctor_interview_boundary"]
        assert audit["raw_response"] == raw[index]
        assert audit["spoken_response"] == questions[index]
        assert audit["boundary"]["kind"] == "participant_turn"
    for patient in patients:
        for index, messages in enumerate(patient.calls):
            assert "FABRICATED" not in str(messages) and "FUTURE" not in str(messages)
            assert questions[index] in str(messages[-1])
            if group:
                assert json.loads(messages[-1]["content"].splitlines()[1])["message"] == questions[index]
    assert {"role": "assistant", "content": questions[0]} in doctor.calls[1]
    assert "FABRICATED" not in str(doctor.calls[1:])
    assert "FUTURE" not in str(doctor.calls[1:])
    for i in range(len(patients)):
        assert f"REAL {i} first." in str(doctor.calls[1])


@pytest.mark.asyncio
async def test_output_starting_with_a_fabricated_patient_turn_fails_before_history_update():
    doctor = Doctor(RecordingModel("doctor", ["Patient 1: fabricated reply."]), use_dynamic_strategies=False)
    with pytest.raises(ValueError, match="no interview question"):
        await doctor.conduct_interview("")
    assert doctor.conversation_history == []
