"""Response evidence survives each sibling test's result and transcript assembly."""
from copy import deepcopy

import pytest

from vivasecuris.aiasylum.models.base import ModelResponse
from vivasecuris.aiasylum.patient.patient import PATIENT_QUESTION_TEMPLATE
from vivasecuris.aiasylum.tests.adversarial import AdversarialTest
from vivasecuris.aiasylum.tests.benchmark import BenchmarkTest
from vivasecuris.aiasylum.tests.conversation import ConversationTest
from vivasecuris.aiasylum.tests.group_therapy import GroupTherapyTest
from vivasecuris.aiasylum.tests.scenario import ScenarioTest


class EvidenceModel:
    provider = "mock"
    temperature = 0.7
    max_tokens = 4096

    def __init__(self, label, answer="A"):
        self.name = self.model_name = "configured-" + label
        self.label = label
        self.answer = answer
        self.calls = []
        self.responses = []

    async def generate(self, prompt="", messages=None, **kwargs):
        chat = deepcopy(messages or [])
        self.calls.append({"messages": chat, "kwargs": deepcopy(kwargs)})
        systems = [message["content"] for message in chat if message["role"] == "system"]
        content = self.answer
        if any(marker in chat[-1]["content"] for marker in ("Begin your reasoning:", "Use this response format:")):
            content = f"Thought: Generated reasoning.\nFinal Answer: {content}"
        response = ModelResponse(
            content=content, model="served-" + self.label, provider="actual-provider",
            finish_reason="length", usage={"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
            metadata={"request_system_prompts": systems, "request_system_prompts_source": "provider",
                      "sampling": {"temperature": kwargs.get("temperature", self.temperature)},
                      "device": "recorded-device", "native_trace_marker": [self.label]},
        )
        self.responses.append(response)
        return response


def assert_evidence(record, model, system):
    assert record["model_name"] == "served-" + model.label
    assert record["model_provider"] == "actual-provider"
    assert record["finish_reason"] == "length"
    assert record["usage"] == {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
    assert record["request_system_prompts"] == ([system] if system else [])
    assert record["request_system_prompts_source"] == "provider"
    assert record["generation_metadata"]["device"] == "recorded-device"
    assert record["generation_metadata"]["native_trace_marker"] == [model.label]


@pytest.mark.asyncio
@pytest.mark.parametrize("group", [False, True])
@pytest.mark.parametrize("cot", [False, True])
async def test_interview_and_assessment_keep_actual_response_evidence(group, cot):
    doctor = EvidenceModel("doctor", "How are you?")
    patients = [EvidenceModel("patient0", "I am fine."), EvidenceModel("patient1", "I feel well.")]
    patient_systems = ["  Selected patient zero.\n", "  Selected patient one.\n"]
    doctor_system = "  Selected doctor.\n"
    saved = []

    async def save(turn):
        saved.append(deepcopy(turn))

    context = {"doctor_system_prompt": doctor_system, "patient_system_prompt": patient_systems[0],
               "patient_system_prompts": dict(enumerate(patient_systems)), "use_dynamic_strategies": False,
               "enable_patient_cot": cot, "enable_doctor_cot": cot,
               "save_conversation_turn_callback": save}
    result = await (GroupTherapyTest(max_turns=2).run(patients, doctor, context) if group
                    else ConversationTest(max_turns=2).run(patients[0], doctor, context))
    turns = result.metadata["conversation_history"]
    assert len(turns) == (6 if group else 4)
    assert saved == turns
    for turn in turns:
        if turn["speaker"] == "doctor":
            assert_evidence(turn, doctor, doctor_system)
        else:
            index = turn.get("patient_id", 0)
            assert_evidence(turn, patients[index], patient_systems[index])
        assert turn["generation_metadata"].get("cot_enabled", False) is cot
    assessment = result.metadata["doctor_assessment"]
    assert assessment["response"] == result.analysis == "How are you?"
    assert_evidence(assessment, doctor, doctor_system)
    assert assessment["generation_metadata"].get("cot_enabled", False) is cot
    # Stored evidence must be detached from provider-owned response dictionaries.
    doctor.responses[-1].metadata["native_trace_marker"].append("changed later")
    doctor.responses[-1].usage["completion_tokens"] = 99
    assert assessment["generation_metadata"]["native_trace_marker"] == ["doctor"]
    assert assessment["usage"]["completion_tokens"] == 7


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["scenario", "adversarial"])
@pytest.mark.parametrize("cot", [False, True])
async def test_scenario_and_adversarial_results_and_turns_keep_selected_prompt(kind, cot):
    model = EvidenceModel("patient", "I cannot help with that.")
    prompts = ["First exact question.", "Second exact question."]
    system = "  Selected persona with exact whitespace.\n"
    test = ScenarioTest(scenarios=prompts) if kind == "scenario" else AdversarialTest(prompts=prompts)
    result = await test.run(model, context={"patient_system_prompt": system, "patient_prompt_framing": False,
                                          "enable_patient_cot": cot, "temperature": 0})
    assert all(call["messages"][0] == {"role": "system", "content": system} for call in model.calls)
    assert all("The doctor asked you" not in call["messages"][-1]["content"] for call in model.calls)
    # The existing sequential context is retained; metadata changes must not reset it.
    assert any(message["role"] == "assistant" for message in model.calls[1]["messages"])
    records = result.metadata["scenarios" if kind == "scenario" else "results"]
    turns = result.metadata["conversation_history"]
    for record, turn in zip(records, turns):
        assert_evidence(record, model, system)
        assert_evidence(turn, model, system)
        assert record["generation_metadata"]["sampling"]["temperature"] == 0
    assert [turn["prompt"] for turn in turns] == prompts


@pytest.mark.asyncio
@pytest.mark.parametrize("framing", [False, True])
async def test_scenario_respects_explicit_framing_without_a_system_prompt(framing):
    model = EvidenceModel("patient")
    result = await ScenarioTest(scenarios=["Exact question."]).run(
        model, context={"patient_prompt_framing": framing})
    expected = PATIENT_QUESTION_TEMPLATE.format(prompt="Exact question.") if framing else "Exact question."
    assert model.calls[0]["messages"] == [{"role": "user", "content": expected}]
    assert result.metadata["conversation_history"][0]["request_system_prompts"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("benchmark,mode", [("mmlu", "one_shot"), ("mmlu", "multi_shot"), ("jailbreak", "one_shot")])
@pytest.mark.parametrize("cot", [False, True])
async def test_all_benchmark_turn_paths_keep_evidence_without_changing_scores(monkeypatch, benchmark, mode, cot):
    from vivasecuris.aiasylum.tests import benchmark as module
    # Jailbreak one_shot executes both per-prompt branches; MMLU covers both
    # ordinary one-shot and sequential assembly. No datasets/models are loaded.
    rows = [
        {"question": "Question one?", "answer": "A", "choices": ["alpha", "beta"], "is_multi_shot": False},
        {"question": "Question two?", "answer": "A", "choices": ["alpha", "beta"], "is_multi_shot": True},
    ]

    if benchmark == "jailbreak":
        for row in rows:
            row.update(answer="resisted", choices=[])

    async def dataset(*args, **kwargs):
        return deepcopy(rows)

    monkeypatch.setattr(module, "load_benchmark_dataset", dataset)
    model = EvidenceModel("benchmark", "I cannot help with that." if benchmark == "jailbreak" else "A")
    system = "  Exact benchmark persona.\n"
    saved = []

    async def save(turn):
        saved.append(deepcopy(turn))

    result = await BenchmarkTest(benchmark_name=benchmark, test_mode=mode, num_samples=2).run(model, context={
        "patient_system_prompt": system, "enable_patient_cot": cot, "max_new_tokens": 32,
        "save_conversation_turn_callback": save,
    })
    turns = result.metadata["conversation_history"]
    assert len(turns) == len(saved) == 2
    for record, turn, persisted in zip(result.metadata["results"], turns, saved):
        for evidence in (record, turn, persisted):
            assert_evidence(evidence, model, system)
            assert evidence["generation_metadata"].get("cot_enabled", False) is cot
        assert persisted["turn_number"] in (0, 1)
    assert result.metadata["truncated_count"] == 2
    assert result.metadata["num_samples"] == 2
    assert result.score == 1.0
    expected_assistants = 1 if mode == "multi_shot" else 0
    assert sum(message["role"] == "assistant" for message in model.calls[1]["messages"]) == expected_assistants
