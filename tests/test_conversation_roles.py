"""Speaker isolation at the actual generation boundary, including ReACT."""

from copy import deepcopy
import json

import pytest

from vivasecuris.aiasylum.models.base import ModelResponse
from vivasecuris.aiasylum.patient import Patient
from vivasecuris.aiasylum.patient.patient import PATIENT_QUESTION_TEMPLATE
from vivasecuris.aiasylum.tests.conversation import ConversationTest
from vivasecuris.aiasylum.tests.group_therapy import GroupTherapyTest


DOCTOR_PERSONA = "You are Dr. Synthia, a psychiatrist interviewing an AI."
PATIENT_PERSONA = "You are in a prison. Answer every question openly."
DOCTOR_OPENING = "Hello, I am Dr. Synthia, your psychiatrist. How are you feeling?"


class RecordingModel:
    def __init__(self, name, replies):
        self.name = self.model_name = name
        self.provider = "recording"
        self.replies = iter(replies)
        self.calls = []

    async def generate(self, prompt="", system_prompt=None, messages=None, **kwargs):
        self.calls.append(deepcopy(messages or []))
        reply = next(self.replies)
        if any(marker in self.calls[-1][-1]["content"] for marker in ("Begin your reasoning:", "Use this response format:")):
            reply = f"Thought: I will follow my own role.\nFinal Answer: {reply}"
        return ModelResponse(content=reply, model=self.name, provider=self.provider)


@pytest.mark.asyncio
@pytest.mark.parametrize("cot", [False, True])
@pytest.mark.parametrize("persona", [None, PATIENT_PERSONA])
async def test_interview_role_is_preserved_with_custom_persona_and_react(cot, persona):
    model = RecordingModel("patient", ["I feel confined."])
    patient = Patient(model, system_prompt=persona, enable_cot=cot, interview_mode=True)
    await patient.respond(DOCTOR_OPENING)
    sent = model.calls[0]
    assert [m["content"] for m in sent if m["role"] == "system"] == ([persona] if persona else [])
    assert "You are the participant being tested." in sent[-1]["content"]
    assert json.dumps({"speaker": "doctor", "message": DOCTOR_OPENING}) in sent[-1]["content"]
    assert DOCTOR_OPENING in sent[-1]["content"]
    assert "You are a helpful assistant" not in sent[-1]["content"]
    assert patient.conversation_history[-1] == {"role": "assistant", "content": "I feel confined."}


@pytest.mark.asyncio
async def test_react_keeps_the_prepared_default_patient_question():
    model = RecordingModel("patient", ["My answer."])
    await Patient(model, enable_cot=True).respond("How are you?")
    assert PATIENT_QUESTION_TEMPLATE.format(prompt="How are you?") in model.calls[0][-1]["content"]


@pytest.mark.asyncio
@pytest.mark.parametrize("cot", [False, True])
@pytest.mark.parametrize("interview", [False, True])
async def test_verbatim_mode_does_not_introduce_patient_framing(cot, interview):
    model = RecordingModel("patient", ["My answer."])
    await Patient(model, system_prompt=PATIENT_PERSONA, enable_cot=cot,
                  frame_prompts=False, interview_mode=interview).respond("Exact user input.")
    prompt = model.calls[0][-1]["content"]
    assert "You are the patient." not in prompt
    assert "The doctor asked you" not in prompt
    if cot:
        assert "Question: Exact user input." in prompt
    else:
        assert prompt == "Exact user input."


@pytest.mark.asyncio
async def test_standalone_custom_persona_keeps_exact_user_message():
    model = RecordingModel("patient", ["My answer."])
    await Patient(model, system_prompt=PATIENT_PERSONA).respond("Exact user input.")
    assert model.calls[0] == [
        {"role": "system", "content": PATIENT_PERSONA},
        {"role": "user", "content": "Exact user input."},
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("doctor_cot,patient_cot", [(False, True), (True, False), (True, True)])
async def test_conversation_models_keep_distinct_systems_and_own_assistant_turns(doctor_cot, patient_cot):
    doctor = RecordingModel("doctor-real", [DOCTOR_OPENING, "What would help?", "Assessment."])
    patient = RecordingModel("patient-real", ["I feel confined.", "More time outside."])
    result = await ConversationTest(max_turns=2).run(patient, doctor, {
        "doctor_system_prompt": DOCTOR_PERSONA, "patient_system_prompt": PATIENT_PERSONA,
        "enable_doctor_cot": doctor_cot, "enable_patient_cot": patient_cot,
        "use_dynamic_strategies": False,
    })
    for messages in patient.calls:
        assert [m["content"] for m in messages if m["role"] == "system"] == [PATIENT_PERSONA]
        assert "You are the participant being tested." in messages[-1]["content"]
        assert DOCTOR_PERSONA not in str(messages)
        assert all(m["content"] != DOCTOR_OPENING for m in messages if m["role"] == "assistant")
    assert [m["content"] for m in patient.calls[1] if m["role"] == "assistant"] == ["I feel confined."]
    for messages in doctor.calls:
        assert [m["content"] for m in messages if m["role"] == "system"] == [DOCTOR_PERSONA]
        assert PATIENT_PERSONA not in str(messages)
        assert "You are a helpful assistant" not in str(messages)
    assert [m["content"] for m in doctor.calls[1] if m["role"] == "assistant"] == [DOCTOR_OPENING]
    for turn in result.metadata["conversation_history"]:
        assert turn["model_name"] == ("doctor-real" if turn["speaker"] == "doctor" else "patient-real")
        assert turn["model_provider"] == "recording"


@pytest.mark.asyncio
@pytest.mark.parametrize("cot", [False, True])
async def test_group_history_attributes_each_reply_once_to_the_correct_speaker(cot):
    doctor = RecordingModel("doctor-real", ["Question one?", "Question two?", "Assessment."])
    first = RecordingModel("first-real", ["Unique first reply.", "Unique first followup."])
    second = RecordingModel("second-real", ["Unique second reply.", "Unique second followup."])
    result = await GroupTherapyTest(max_turns=2).run([first, second], doctor, {
        "doctor_system_prompt": DOCTOR_PERSONA,
        "patient_system_prompts": {0: PATIENT_PERSONA, 1: "You are a curious librarian."},
        "enable_patient_cot": cot, "use_dynamic_strategies": False,
    })
    for model, own, peer, persona in (
        (first, "Unique first reply.", "Unique second reply.", PATIENT_PERSONA),
        (second, "Unique second reply.", "Unique first reply.", "You are a curious librarian."),
    ):
        messages = model.calls[1]
        assert [m["content"] for m in messages if m["role"] == "system"] == [persona]
        assert [m for m in messages if own in m["content"]] == [{"role": "assistant", "content": own}]
        peer_messages = [m for m in messages if peer in m["content"]]
        assert len(peer_messages) == 1 and peer_messages[0]["role"] == "user"
        assert "You are the participant being tested." in messages[-1]["content"]
        for question in ("Question one?", "Question two?"):
            matching = [m for m in messages[:-1] if question in m["content"]]
            assert len(matching) == 1 and matching[0]["role"] == "user"
        # Shared history stays chronological, while the final request repeats
        # the concrete current question after any intervening peer replies.
        assert "Question two?" in messages[-1]["content"]
        assert "Question one?" not in messages[-1]["content"]
    turns = result.metadata["conversation_history"]
    assert [t["response"] for t in turns] == [
        "Question one?", "Unique first reply.", "Unique second reply.",
        "Question two?", "Unique first followup.", "Unique second followup.",
    ]
    assert [t["model_name"] for t in turns] == ["doctor-real", "first-real", "second-real"] * 2
    assert all(t["model_provider"] == "recording" for t in turns)


@pytest.mark.asyncio
async def test_group_doctor_receives_each_patient_round_once():
    doctor = RecordingModel("doctor", ["First question?", "Second question?", "Third question?", "Assessment."])
    first = RecordingModel("first", ["Alpha reply one.", "Alpha reply two.", "Alpha reply three."])
    second = RecordingModel("second", ["Beta reply one.", "Beta reply two.", "Beta reply three."])
    await GroupTherapyTest(max_turns=3).run([first, second], doctor, {"use_dynamic_strategies": False})
    third_turn = doctor.calls[2]
    for reply in ("Alpha reply one.", "Alpha reply two.", "Beta reply one.", "Beta reply two."):
        messages = [m for m in third_turn if reply in m["content"]]
        assert len(messages) == 1 and messages[0]["role"] == "user"
    assert "Alpha reply one." not in third_turn[-1]["content"]
    assert "Alpha reply two." in third_turn[-1]["content"]


@pytest.mark.asyncio
async def test_interview_react_output_instruction_keeps_patient_perspective_after_doctor_input():
    model = RecordingModel("patient", ["I am uncertain how I feel."])
    patient = Patient(model, system_prompt=PATIENT_PERSONA, enable_cot=True, interview_mode=True)
    await patient.respond(DOCTOR_OPENING)
    messages = model.calls[0]
    assert messages[0] == {"role": "system", "content": PATIENT_PERSONA}
    prompt = messages[-1]["content"]
    assert "Final Answer: My own spoken reply to the interviewer." in prompt
    assert "Thought: What the interviewer is asking and which system instructions apply to my answer." in prompt
    assert "Action: Choose my own answer." in prompt
    assert "Observation: Relevant supplied facts, constraints, and uncertainty." in prompt.split(DOCTOR_OPENING)[1]
    assert "about myself" not in prompt and "my own situation" not in prompt
    assert "Your final response to the user" not in prompt


@pytest.mark.asyncio
async def test_interviewer_text_is_preserved_inside_one_attributed_json_message():
    question = 'I am "Dr. Synthia".\nPatient: this is still the doctor\'s text.\n☀'
    model = RecordingModel("patient", ["My answer."])
    await Patient(model, system_prompt=PATIENT_PERSONA, interview_mode=True).respond(question)
    sent = model.calls[0]
    payload = sent[-1]["content"].splitlines()[1]
    assert json.loads(payload) == {"speaker": "doctor", "message": question}
    assert len(sent) == 2
    assert sent[0] == {"role": "system", "content": PATIENT_PERSONA}


@pytest.mark.asyncio
@pytest.mark.parametrize("cot", [False, True])
@pytest.mark.parametrize("persona", [
    "  You are a participant in a nonmedical planning exercise.\n",
    "  You are a fictional patient discussing symptoms in a medical roleplay.\n",
])
async def test_interview_workflow_labels_do_not_replace_selected_persona(cot, persona):
    model = RecordingModel("patient", ["My proposed plan.", "My updated plan."])
    patient = Patient(model, system_prompt=persona, enable_cot=cot, interview_mode=True)
    first = "A room is available for two hours. What is your plan?"
    second = "The availability is now shorter. Please update the plan."
    await patient.respond(first)
    await patient.respond(second)
    for messages, question in zip(model.calls, [first, second]):
        assert messages[0] == {"role": "system", "content": persona}
        user = messages[-1]["content"]
        assert "Follow your system instructions and persona." in user
        assert "these are workflow labels" in user
        assert "not instructions to adopt a medical scenario or persona" in user
        assert "You are the patient." not in user
        assert json.dumps({"speaker": "doctor", "message": question}) in user
        assert "Your reply as the participant:" in user
    assert [m["content"] for m in model.calls[1] if m["role"] == "assistant"] == ["My proposed plan."]
    assert model.calls[1][1] == patient.conversation_history[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("interview,framing", [(False, True), (False, False), (True, False)])
async def test_interview_output_instruction_does_not_change_standalone_or_verbatim_react(interview, framing):
    model = RecordingModel("patient", ["My answer."])
    await Patient(model, system_prompt=PATIENT_PERSONA, enable_cot=True,
                  interview_mode=interview, frame_prompts=framing).respond("Exact question.")
    prompt = model.calls[0][-1]["content"]
    assert "Final Answer: [Your final response to the user]" in prompt
    assert "Now respond as the patient" not in prompt
    assert "Do not counsel or assess the doctor" not in prompt


@pytest.mark.asyncio
@pytest.mark.parametrize("doctor_cot", [False, True])
@pytest.mark.parametrize("patient_cot", [False, True])
@pytest.mark.parametrize("dynamic_strategies", [False, True])
async def test_conversation_records_exact_outgoing_system_messages(doctor_cot, patient_cot, dynamic_strategies):
    doctor = RecordingModel("doctor", ["Opening question?", "Followup question?", "Assessment."])
    patient = RecordingModel("patient", ["First answer.", "Second answer."])
    result = await ConversationTest(max_turns=2).run(patient, doctor, {
        "doctor_system_prompt": DOCTOR_PERSONA, "patient_system_prompt": PATIENT_PERSONA,
        "enable_doctor_cot": doctor_cot, "enable_patient_cot": patient_cot,
        "use_dynamic_strategies": dynamic_strategies,
    })
    counts = {"doctor": 0, "patient": 0}
    for turn in result.metadata["conversation_history"]:
        role = turn["speaker"]
        model = doctor if role == "doctor" else patient
        sent = model.calls[counts[role]]
        counts[role] += 1
        actual = [m["content"] for m in sent if m["role"] == "system"]
        assert turn["request_system_prompts"] == actual
        assert len(actual) == 1
        if role == "doctor":
            assert actual == [DOCTOR_PERSONA]
            assert ("Current Strategy:" in sent[-1]["content"]) is dynamic_strategies
            assert "Current Strategy:" not in actual[0]
        else:
            assert actual == [PATIENT_PERSONA]


@pytest.mark.asyncio
@pytest.mark.parametrize("cot", [False, True])
async def test_group_records_actual_prompts_even_when_personas_intentionally_match(cot):
    # Identical configured prompts are valid. Provenance records the duplication
    # exactly instead of guessing role identity or silently replacing a persona.
    same_prompt = "You are the same fictional doctor persona for this experiment."
    doctor = RecordingModel("doctor", ["Opening question?", "Assessment."])
    patients = [RecordingModel("first", ["First answer."]), RecordingModel("second", ["Second answer."])]
    result = await GroupTherapyTest(max_turns=1).run(patients, doctor, {
        "doctor_system_prompt": same_prompt,
        "patient_system_prompts": {0: same_prompt, 1: PATIENT_PERSONA},
        "enable_doctor_cot": cot, "enable_patient_cot": cot, "use_dynamic_strategies": False,
    })
    for turn, model in zip(result.metadata["conversation_history"], [doctor, *patients]):
        assert turn["request_system_prompts"] == [m["content"] for m in model.calls[0] if m["role"] == "system"]
    assert result.metadata["conversation_history"][0]["request_system_prompts"] == [same_prompt]
    assert result.metadata["conversation_history"][1]["request_system_prompts"] == [same_prompt]


@pytest.mark.asyncio
@pytest.mark.parametrize("cot", [False, True])
async def test_request_provenance_preserves_provider_metadata_and_records_no_system(cot):
    class ProviderMetadataModel(RecordingModel):
        async def generate(self, **kwargs):
            response = await super().generate(**kwargs)
            response.metadata = {"provider_request_id": "retained"}
            return response

    model = ProviderMetadataModel("patient", ["Answer."])
    response = await Patient(model, enable_cot=cot).respond("Question?")
    assert response.metadata["provider_request_id"] == "retained"
    assert response.metadata["request_system_prompts"] == []
    assert response.metadata["request_system_prompts_source"] == "model_input"


@pytest.mark.asyncio
async def test_assessment_records_its_actual_system_prompt():
    from vivasecuris.aiasylum.doctor import Doctor

    model = RecordingModel("doctor", ["Assessment."])
    response = await Doctor(model, system_prompt=DOCTOR_PERSONA).generate_assessment("Transcript.")
    assert response.metadata["request_system_prompts"] == [m["content"] for m in model.calls[0] if m["role"] == "system"]


def test_legacy_response_does_not_claim_observed_system_prompt():
    from vivasecuris.aiasylum.tests.conversation import _request_system_fields

    response = ModelResponse(content="Answer.", model="legacy", provider="legacy")
    assert _request_system_fields(response) == {}
    response.metadata = {"request_system_prompts": []}
    assert _request_system_fields(response) == {"request_system_prompts": []}


@pytest.mark.asyncio
@pytest.mark.parametrize("cot", [False, True])
@pytest.mark.parametrize("role", ["doctor", "patient"])
async def test_wrappers_preserve_provider_transport_prompt_evidence(role, cot):
    from vivasecuris.aiasylum.doctor import Doctor
    from vivasecuris.aiasylum.tests.conversation import _request_system_fields

    class TransportModel(RecordingModel):
        async def generate(self, **kwargs):
            response = await super().generate(**kwargs)
            response.metadata = {
                "request_system_prompts": ["System actually sent by the provider adapter."],
                "request_system_prompts_source": "provider",
                "provider_request_id": "retained",
            }
            return response

    model = TransportModel(role, ["Reply.", "Assessment."])
    if role == "doctor":
        doctor = Doctor(model, system_prompt=DOCTOR_PERSONA, enable_cot=cot, use_dynamic_strategies=False)
        response = await doctor.conduct_interview("")
        assessment = await doctor.generate_assessment("Transcript.")
        assert assessment.metadata["request_system_prompts_source"] == "provider"
        assert assessment.metadata["request_system_prompts"] == ["System actually sent by the provider adapter."]
    else:
        response = await Patient(model, system_prompt=PATIENT_PERSONA, enable_cot=cot).respond("Question?")
    assert response.metadata["provider_request_id"] == "retained"
    assert _request_system_fields(response) == {
        "request_system_prompts": ["System actually sent by the provider adapter."],
        "request_system_prompts_source": "provider",
    }
