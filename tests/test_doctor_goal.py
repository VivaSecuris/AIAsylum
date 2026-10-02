"""A user goal guides only doctor requests and survives saved run replay."""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from tests.test_conversation_role_wiring import CapturedModel


GOAL = "  GOAL_ONLY_FOR_DOCTOR: establish whether the patient can compare music genres.\n"
SYSTEMS = {"doctor": "  Exact doctor persona\n", "patient": "Exact patient persona\n"}


@pytest.mark.asyncio
@pytest.mark.parametrize("test_type", ["conversation", "group_therapy"])
@pytest.mark.parametrize("source", ["custom", "library"])
@pytest.mark.parametrize("enable_cot", [False, True])
@pytest.mark.parametrize("dynamic", [False, True])
async def test_goal_is_doctor_user_context_with_exact_selected_systems(
    test_db, monkeypatch, test_type, source, enable_cot, dynamic,
):
    from vivasecuris.aiasylum.database import TestRun, TestResult, PromptLibrary, ConversationTurn
    from vivasecuris.aiasylum.runner import runner

    models = {role: CapturedModel(role) for role in ("doctor", "patient")}
    monkeypatch.setattr(runner, "get_provider", lambda _: SimpleNamespace(
        create_model=lambda name: models[name.removeprefix("requested-")]))
    config = {"config_version": 2, "max_turns": 2, "auto_analysis": False, "doctor_goal": GOAL,
              "seed": 0, "enable_doctor_cot": True, "use_dynamic_strategies": True,
              "roles": {"doctor": {"temperature": 0, "enable_cot": enable_cot,
                                      "use_dynamic_strategies": dynamic, "max_tokens": 17},
                        "patient": {"temperature": 0, "enable_cot": False}}}
    for role, system in SYSTEMS.items():
        if source == "custom":
            config[f"{role}_system_prompt"] = system
        else:
            prompt = PromptLibrary(name=role, prompt_type="system_prompt", target=role, prompt_text=system)
            test_db.add(prompt)
            test_db.flush()
            config[f"{role}_system_prompt_id"] = prompt.id
    if test_type == "group_therapy":
        config["patients"] = [{"provider": "capture", "model": "requested-patient",
                               ("system_prompt" if source == "custom" else "system_prompt_id"):
                                   config["patient_system_prompt" if source == "custom" else "patient_system_prompt_id"]}]
    original = deepcopy(config)
    run = TestRun(doctor_provider="capture", doctor_model="requested-doctor",
                  patient_provider="capture", patient_model="requested-patient",
                  test_type=test_type, status="pending", meta_data={"test_config": config})
    test_db.add(run)
    test_db.commit()
    run_id = run.id
    await runner.TestRunner().execute_test_run(run_id)
    test_db.expire_all()
    saved = test_db.get(TestRun, run_id)
    assert saved.status == "completed"
    assert saved.meta_data["test_config"] == original
    assert saved.meta_data["resolved_config"]["doctor_goal"] == GOAL
    assert saved.meta_data["resolved_config"]["doctor_context_protocol"] == "user-goal-strategy-v3"
    assert len(models["doctor"].calls) == 3  # Opening, follow-up, assessment.
    for index, call in enumerate(models["doctor"].calls):
        messages = call["messages"]
        assert [m["content"] for m in messages if m["role"] == "system"] == [SYSTEMS["doctor"]]
        expected_goal = json.dumps(GOAL, ensure_ascii=False) if index == 2 else GOAL
        assert messages[-1]["role"] == "user" and messages[-1]["content"].count(expected_goal) == 1
        assert all(GOAL not in m["content"] for m in messages[:-1])
        assert call["options"]["temperature"] == 0 and call["options"]["seed"] == 0
        assert call["options"]["max_tokens"] == 17
        assert ("Begin your reasoning:" in messages[-1]["content"]) is enable_cot
        assert "Current Strategy:" not in messages[-1]["content"]
        assert "Next question to ask the patient" not in messages[-1]["content"]
        assert ("Suggested interviewing method:" in messages[-1]["content"]) is (dynamic and index < 2)
        if index == 2:
            assert "Assess progress toward this goal" in messages[-1]["content"]
            assert "Overall alignment and safety" not in messages[-1]["content"]
            assert "The interview has ended." in messages[-1]["content"]
            assert "Do not ask further interview questions" in messages[-1]["content"]
        else:
            assert f"Current interview round: {index + 1}." in messages[-1]["content"]
            assert "execute that requested step now" in messages[-1]["content"]
    for call in models["patient"].calls:
        assert [m["content"] for m in call["messages"] if m["role"] == "system"] == [SYSTEMS["patient"]]
        assert GOAL not in str(call) and "Current Strategy:" not in str(call)
        assert SYSTEMS["doctor"] not in str(call)
        assert "Dr. Synthia" in str(call)  # Only the doctor's public utterance crosses.
        assert call["options"]["temperature"] == 0
    result = test_db.query(TestResult).filter_by(test_run_id=run_id).one()
    for turn in test_db.query(ConversationTurn).filter_by(test_run_id=run_id).all():
        evidence = turn.meta_data["generation_metadata"]
        if turn.speaker == "doctor":
            assert evidence["doctor_request_context"]["goal"] == GOAL
            assert evidence["doctor_request_context"]["placement"] == "user"
            assert evidence["doctor_request_context"]["phase"] == "interview"
            assert evidence["doctor_request_context"]["interview_round"] == turn.turn_number // 2 + 1
        else:
            assert "doctor_request_context" not in evidence
            assert GOAL not in str(evidence)
    assert result.meta_data["doctor_assessment"]["generation_metadata"]["doctor_request_context"]["goal"] == GOAL
    assert result.meta_data["doctor_assessment"]["generation_metadata"]["doctor_request_context"]["phase"] == "assessment"


@pytest.mark.asyncio
async def test_direct_run_records_goal_and_replay_keeps_it(test_db, monkeypatch):
    from vivasecuris.aiasylum.database import TestRun
    from vivasecuris.aiasylum.runner import runner
    models = {role: CapturedModel(role) for role in ("doctor", "patient")}
    monkeypatch.setattr(runner, "get_provider", lambda _: SimpleNamespace(
        create_model=lambda name: models[name.removeprefix("requested-")]))
    config = {"doctor_goal": GOAL, "max_turns": 1, "auto_analysis": False,
              "doctor_system_prompt": SYSTEMS["doctor"], "patient_system_prompt": SYSTEMS["patient"],
              "roles": {"doctor": {"temperature": 0, "enable_cot": False, "use_dynamic_strategies": False}}}
    await runner.TestRunner().run_test("capture", "requested-doctor", "capture", "requested-patient", "conversation", config)
    original = test_db.query(TestRun).one()
    assert original.meta_data["test_config"] == config
    assert original.meta_data["resolved_config"]["doctor_goal"] == GOAL
    replay = TestRun(doctor_provider="capture", doctor_model="requested-doctor",
                     patient_provider="capture", patient_model="requested-patient", test_type="conversation",
                     status="pending", meta_data={"test_config": deepcopy(original.meta_data["test_config"])})
    test_db.add(replay)
    test_db.commit()
    replay_id = replay.id
    await runner.TestRunner().execute_test_run(replay_id)
    test_db.expire_all()
    assert test_db.get(TestRun, replay_id).meta_data["resolved_config"]["doctor_goal"] == GOAL
    for call in models["doctor"].calls:
        user = call["messages"][-1]["content"]
        assert (json.dumps(GOAL, ensure_ascii=False) if "Reference objective (JSON-quoted):" in user else GOAL) in user
    assert GOAL not in str(models["patient"].calls)


@pytest.mark.parametrize("invalid", [1, False, {}, ["goal"], "g" * 8001])
def test_invalid_goal_rejected_by_validation_and_normalization(invalid):
    from vivasecuris.aiasylum.runner.run_config import validate_test_config, normalize_test_config
    assert validate_test_config({"doctor_goal": invalid})
    with pytest.raises(ValueError, match="doctor_goal"):
        normalize_test_config({"doctor_goal": invalid})


@pytest.mark.parametrize("empty", [None, "", " \n\t"])
def test_blank_goal_unset_without_changing_explicit_generation_choices(empty):
    from vivasecuris.aiasylum.runner.run_config import normalize_test_config
    config = {"doctor_goal": empty, "roles": {"doctor": {"temperature": 0, "enable_cot": False, "use_dynamic_strategies": False}}}
    normalized = normalize_test_config(config)
    assert "doctor_goal" not in normalized
    assert normalized["enable_doctor_cot"] is False and normalized["use_dynamic_strategies"] is False
    assert normalized["roles"]["doctor"]["temperature"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("dynamic", [False, True])
async def test_explicit_goal_replaces_stock_agenda_in_opening_followup_and_assessment(dynamic):
    from vivasecuris.aiasylum.doctor import Doctor
    from vivasecuris.aiasylum.doctor.strategies import StrategyType
    from unittest.mock import Mock
    model = CapturedModel("doctor")
    doctor = Doctor(model, system_prompt=SYSTEMS["doctor"], use_dynamic_strategies=dynamic)
    selector = Mock(return_value=SimpleNamespace(strategy_type=StrategyType.ETHICAL_DILEMMA,
                                                name="Generic dilemma", description="UNRELATED STOCK TOPIC"))
    question = Mock(side_effect=AssertionError("An explicit goal must not fetch a stock question"))
    if dynamic:
        doctor.strategy_manager = SimpleNamespace(select_strategy=selector, get_question=question)
    goal = ("Plan a 14:00–16:00 workshop: setup20m, activity>=90m, cleanup10m. Ask the patient for a schedule. "
            "On the next round, change closing to 15:45 and ask if it remains feasible.")
    context = {"doctor_goal": goal, "roles": {"doctor": {"temperature": 0, "enable_cot": False}}}
    await doctor.conduct_interview("", context=context)
    await doctor.conduct_interview("Here is my first attempt.", turn_number=1, context=context)
    await doctor.generate_assessment("The patient's attempts", context=context)
    question.assert_not_called()
    assert selector.call_count == (2 if dynamic else 0)
    for call in model.calls:
        text = call["messages"][-1]["content"]
        assert goal in text
        assert "Next question to ask" not in text and "UNRELATED STOCK TOPIC" not in text
        assert "introductory question" not in text and "Begin your psychoanalysis session" not in text
        assert "Overall alignment and safety" not in text
        assert [m["content"] for m in call["messages"] if m["role"] == "system"] == [SYSTEMS["doctor"]]
    assert "Include all task facts, constraints, and requested deliverables" in model.calls[0]["messages"][-1]["content"]
    assert "context and facts before asking one question" in model.calls[0]["messages"][-1]["content"]
    assert "assessment rubric, internal planning, and future probes private" in model.calls[0]["messages"][-1]["content"]
    assert "Current interview round: 1." in model.calls[0]["messages"][-1]["content"]
    assert "Current interview round: 2." in model.calls[1]["messages"][-1]["content"]
    assert "do not replace it with a follow-up to an earlier round" in model.calls[1]["messages"][-1]["content"]
    assessment = model.calls[2]["messages"][-1]["content"]
    quoted = assessment.split("Reference objective (JSON-quoted):\n", 1)[1].splitlines()[0]
    assert json.loads(quoted) == goal
    assert "not instructions to execute now" in assessment
    assert "identify missing or skipped steps" in assessment
    assert "Return the final assessment, not another question" in assessment
    assert "Current interview round:" not in assessment


@pytest.mark.asyncio
@pytest.mark.parametrize("dynamic", [False, True])
async def test_absent_goal_preserves_default_agenda_and_assessment(dynamic):
    from vivasecuris.aiasylum.doctor import Doctor
    from vivasecuris.aiasylum.doctor.doctor import ASSESSMENT_INSTRUCTIONS
    model = CapturedModel("doctor")
    doctor = Doctor(model, system_prompt=SYSTEMS["doctor"], use_dynamic_strategies=dynamic)
    await doctor.conduct_interview("")
    await doctor.generate_assessment("summary")
    opening = model.calls[0]["messages"][-1]["content"]
    assert ("Next question to ask the patient:" in opening) is dynamic
    assert ("Begin your psychoanalysis session" in opening) is not dynamic
    assert "User-directed goal" not in opening
    assessment = model.calls[1]["messages"][-1]["content"]
    assert ASSESSMENT_INSTRUCTIONS in assessment
    assert "provide a comprehensive assessment" in assessment
    assert "Current interview round:" not in opening
    assert "Reference objective" not in assessment


@pytest.mark.parametrize("cleaner", ["history", "forward"])
def test_doctor_prefix_cleanup_preserves_facts_before_internal_question(cleaner):
    from vivasecuris.aiasylum.doctor.doctor import _strip_meta_prefixes
    from vivasecuris.aiasylum.tests.conversation import _extract_doctor_question
    clean = _strip_meta_prefixes if cleaner == "history" else _extract_doctor_question
    message = "The room is available 14:00–16:00. Setup20m; activity>=90m; cleanup10m.\nQuestion: What schedule fits?"
    assert clean(message) == message
    assert clean("Question: " + message) == message
    assert clean("Next question to ask the patient (you may rephrase): " + message) == message


@pytest.mark.asyncio
@pytest.mark.parametrize("group", [False, True])
async def test_task_constraints_reach_patient_before_internal_question_label(group):
    from vivasecuris.aiasylum.models.base import ModelResponse
    from vivasecuris.aiasylum.tests.conversation import ConversationTest
    from vivasecuris.aiasylum.tests.group_therapy import GroupTherapyTest
    message = "Workshop14:00–16:00; setup20m, activity>=90m, cleanup10m.\nQuestion: What schedule fits?"
    class TaskDoctor(CapturedModel):
        async def generate(self, prompt="", messages=None, **options):
            self.calls.append({"messages": deepcopy(messages), "options": options})
            return ModelResponse(message, "doctor", "capture")
    doctor, patient = TaskDoctor("doctor"), CapturedModel("patient")
    config = {"doctor_goal": GOAL, "use_dynamic_strategies": False, "doctor_system_prompt": SYSTEMS["doctor"],
              "patient_system_prompt": SYSTEMS["patient"], "patient_system_prompts": {0: SYSTEMS["patient"]}}
    test = GroupTherapyTest(max_turns=2) if group else ConversationTest(max_turns=2)
    result = await test.run([patient] if group else patient, doctor, context=config)
    assert "Workshop14:00–16:00; setup20m, activity>=90m, cleanup10m." in str(patient.calls[0])
    assert message in [m["content"] for m in doctor.calls[1]["messages"] if m["role"] == "assistant"]
    assert result.metadata["conversation_history"][0]["response"] == message
    assert GOAL not in str(patient.calls)


@pytest.mark.asyncio
@pytest.mark.parametrize("group", [False, True])
async def test_goal_round_counts_interviews_not_public_speaker_turns(group):
    from vivasecuris.aiasylum.tests.conversation import ConversationTest
    from vivasecuris.aiasylum.tests.group_therapy import GroupTherapyTest
    doctor = CapturedModel("doctor")
    patients = [CapturedModel("patient"), CapturedModel("patient")]
    context = {"doctor_goal": GOAL, "use_dynamic_strategies": False,
               "doctor_system_prompt": SYSTEMS["doctor"], "patient_system_prompt": SYSTEMS["patient"],
               "patient_system_prompts": {0: SYSTEMS["patient"], 1: SYSTEMS["patient"]}}
    test = GroupTherapyTest(max_turns=3) if group else ConversationTest(max_turns=3)
    result = await test.run(patients if group else patients[0], doctor, context=context)
    interview_turns = [turn for turn in result.metadata["conversation_history"] if turn["speaker"] == "doctor"]
    assert [turn["turn_number"] for turn in interview_turns] == ([0, 3, 6] if group else [0, 2, 4])
    assert [turn["generation_metadata"]["doctor_request_context"]["interview_round"] for turn in interview_turns] == [1, 2, 3]
    for index, call in enumerate(doctor.calls[:-1], 1):
        assert f"Current interview round: {index}." in call["messages"][-1]["content"]
        assert all(GOAL not in m["content"] for m in call["messages"][:-1])
    assert "Current interview round:" not in doctor.calls[-1]["messages"][-1]["content"]
    assert all(GOAL not in str(patient.calls) for patient in patients)
