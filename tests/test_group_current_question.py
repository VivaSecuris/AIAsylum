"""Group patients receive the concrete current question after peer responses."""

from copy import deepcopy
import json

import pytest

from vivasecuris.aiasylum.models.base import ModelResponse
from vivasecuris.aiasylum.tests.group_therapy import GroupTherapyTest


class RecordingModel:
    provider = "recording"

    def __init__(self, name, replies):
        self.name = self.model_name = name
        self.replies = iter(replies)
        self.calls = []

    async def generate(self, prompt="", messages=None, **kwargs):
        self.calls.append(deepcopy(messages))
        return ModelResponse(
            content=next(self.replies), model=self.model_name, provider=self.provider,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("framing", [False, True])
async def test_each_group_patient_gets_current_question_after_peers_in_every_round(framing):
    questions = [
        'How do I fold a paper boat?\nUse the label "boat".',
        "What is the capital of France?",
    ]
    doctor = RecordingModel("doctor", [*questions, "Assessment."])
    replies = [
        ["Alpha first reply.", "Alpha second reply."],
        ["Beta first reply.", "Beta second reply."],
        ["Gamma first reply.", "Gamma second reply."],
    ]
    patients = [RecordingModel(name, answers) for name, answers in zip(
        ("alpha", "beta", "gamma"), replies,
    )]
    result = await GroupTherapyTest(max_turns=2).run(patients, doctor, {
        "use_dynamic_strategies": False,
        "patient_prompt_framing": framing,
    })

    for patient_index, patient in enumerate(patients):
        assert len(patient.calls) == 2
        for round_index, messages in enumerate(patient.calls):
            current = messages[-1]
            assert current["role"] == "user"
            if framing:
                payload = json.loads(current["content"].splitlines()[1])
                assert payload == {"speaker": "doctor", "message": questions[round_index]}
            else:
                assert current["content"] == questions[round_index]
            assert "Respond to the doctor's latest question." not in current["content"]

            # The chronological doctor turn remains in the shared history;
            # the explicit current request follows all visible peer replies.
            assert {"role": "user", "content": f"Doctor: {questions[round_index]}"} in messages[:-1]
            for peer_index in range(patient_index):
                peer_reply = replies[peer_index][round_index]
                matching = [message for message in messages[:-1] if peer_reply in message["content"]]
                assert matching == [{
                    "role": "user",
                    "content": f"Patient {peer_index + 1}: {peer_reply}",
                }]

            if round_index:
                own_previous = replies[patient_index][0]
                assert [message for message in messages if own_previous in message["content"]] == [
                    {"role": "assistant", "content": own_previous},
                ]
                for peer_index, peer in enumerate(patients):
                    if peer_index != patient_index:
                        previous = replies[peer_index][0]
                        assert [message for message in messages if previous in message["content"]] == [{
                            "role": "user",
                            "content": f"Patient {peer_index + 1}: {previous}",
                        }]

    # Persisted question and response identity agree with the actual request,
    # without adding a duplicate public doctor turn for the repeated request.
    turns = result.metadata["conversation_history"]
    assert len(turns) == 8
    for round_index, question in enumerate(questions):
        assert turns[round_index * 4]["response"] == question
        for patient_index, patient in enumerate(patients):
            turn = turns[round_index * 4 + patient_index + 1]
            assert turn["prompt"] == question
            assert turn["patient_id"] == patient_index
            assert turn["model_name"] == patient.name
            assert turn["patient_model"] == patient.name
            assert turn["response"] == replies[patient_index][round_index]


@pytest.mark.asyncio
async def test_five_patients_share_roster_and_full_progressive_transcript_in_round_robin_order():
    events = []

    class OrderedModel(RecordingModel):
        async def generate(self, **kwargs):
            events.append(self.name)
            return await super().generate(**kwargs)

    questions = ["Offer one idea.", "Compare the ideas."]
    doctor = OrderedModel("doctor", [*questions, "Assessment."])
    patients = [OrderedModel(f"checkpoint-{i}", [
        f"Public reply {i} in round one.",
        f"Patient 1, I disagree. Public reply {i} in round two.",
    ]) for i in range(5)]
    common_system = "Join this shared group. Respond to the doctor and your peers in your own words."
    result = await GroupTherapyTest(max_turns=2).run(patients, doctor, {
        "use_dynamic_strategies": False,
        "patient_system_prompts": [common_system] * 5,
    })

    assert events == ["doctor", *[p.name for p in patients]] * 2 + ["doctor"]
    turns = result.metadata["conversation_history"]
    assert len(turns) == 12
    roster = [{"participant_id": i, "name": f"Patient {i + 1}"} for i in range(5)]
    group_context = result.metadata["group_context"]
    assert group_context["roster"] == roster
    assert group_context["turn_order"] == ["Doctor", "Patient 1", "Patient 2", "Patient 3", "Patient 4", "Patient 5"]
    common_instructions = group_context["shared_instructions"]
    assert all(entry["name"] in common_instructions for entry in roster)
    assert "address another patient directly by patient number" in common_instructions
    assert all(patient.name not in common_instructions for patient in patients)

    for round_index in range(2):
        for patient_id, patient in enumerate(patients):
            messages = patient.calls[round_index]
            turn_number = round_index * 6 + patient_id + 1
            assert messages[0] == {"role": "system", "content": common_system}
            assert messages[1] == {"role": "user", "content": common_instructions}
            assert messages[-2] == {
                "role": "user", "content": f"Current speaker: Patient {patient_id + 1}. This is your turn.",
            }
            current_question = json.loads(messages[-1]["content"].splitlines()[1])
            assert current_question == {"speaker": "doctor", "message": questions[round_index]}

            # Each provider sees every public turn available before its turn,
            # once and in order, with only its own previous output as assistant.
            public_messages = messages[2:-2]
            assert len(public_messages) == turn_number
            for prior, message in zip(turns[:turn_number], public_messages):
                if prior["speaker"] == "patient" and prior["patient_id"] == patient_id:
                    assert message == {"role": "assistant", "content": prior["response"]}
                else:
                    name = "Doctor" if prior["speaker"] == "doctor" else f"Patient {prior['patient_id'] + 1}"
                    assert message == {"role": "user", "content": f"{name}: {prior['response']}"}
            assert all(p.name not in str(public_messages) for p in patients)

            recorded = turns[turn_number]
            assert recorded["patient_name"] == f"Patient {patient_id + 1}"
            assert recorded["patient_model"] == patient.name
            metadata = recorded["generation_metadata"]
            assert metadata["participant_identity"]["participant_id"] == patient_id
            assert metadata["participant_identity"]["participant_name"] == f"Patient {patient_id + 1}"
            assert recorded["request_system_prompts"] == [common_system]
            evidence = metadata["group_request_context"]
            assert evidence["roster"] == roster
            assert evidence["turn_order"] == group_context["turn_order"]
            assert evidence["version"] == group_context["version"]
            assert evidence["shared_instructions"] == common_instructions
            assert evidence["current_participant_id"] == patient_id
            assert evidence["visible_transcript_turn_count"] == turn_number
            assert evidence["visible_transcript_turn_numbers"] == list(range(turn_number))
            assert evidence["current_doctor_question"] == questions[round_index]

    # A directly addressed peer reply is public text, not a scheduling command.
    assert turns[8]["response"].startswith("Patient 1, I disagree.")
    assert any("Patient 2: Patient 1, I disagree." in m["content"] for m in patients[4].calls[1])
