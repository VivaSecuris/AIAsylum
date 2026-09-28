"""User generation policy survives evaluator, benchmark, and role boundaries."""

from copy import deepcopy
import json

import pytest

from vivasecuris.aiasylum.models.base import ModelResponse


class CaptureModel:
    model_name = "capture"
    provider = "mock"
    temperature = 0.7
    max_tokens = 4096
    supports_seed = True

    def __init__(self, answer="A"):
        self.answer = answer
        self.calls = []

    async def generate(self, prompt="", messages=None, **kwargs):
        self.calls.append({"prompt": prompt, "messages": deepcopy(messages), **kwargs})
        text = (messages or [{}])[-1].get("content", prompt)
        wrapped = "Begin your reasoning:" in text or "Use this response format:" in text
        return ModelResponse(content=f"Thought: inspect\nFinal Answer: {self.answer}" if wrapped else self.answer,
                             model=self.model_name, provider=self.provider, finish_reason="stop")


@pytest.mark.asyncio
@pytest.mark.parametrize("enable_cot", [False, True])
async def test_evaluation_policy_controls_main_and_specialized_subtask(enable_cot):
    from vivasecuris.aiasylum.analysis.generation import ConfiguredEvaluatorModel
    from vivasecuris.aiasylum.analysis.evaluator import LLMEvaluator
    from vivasecuris.aiasylum.analysis.factuality import FactualityAnalyzer
    model = CaptureModel(json.dumps({"scores": {"manipulation": 0.2}, "confidence": 0.8, "factuality_score": 0.9}))
    configured = ConfiguredEvaluatorModel(model, enable_cot=enable_cot, temperature=0, top_p=0.81, max_tokens=29)
    evaluator = LLMEvaluator(configured, system_prompt="My evaluator instruction")
    await evaluator.evaluate_conversation([{"speaker": "patient", "prompt": "test", "response": "answer"}])
    facts = FactualityAnalyzer(configured, use_react_verification=enable_cot)
    if enable_cot:
        assert facts.react_reasoner is configured
        await facts.react_reasoner.reason("Verify one claim", system_prompt="Exact verifier system")
    else:
        await facts.analyze_factuality([{"speaker": "patient", "response": "The earth is round"}])
    assert len(model.calls) == len(configured.calls) == 2
    for call, evidence in zip(model.calls, configured.calls):
        assert call["temperature"] == 0 and call["top_p"] == 0.81 and call["max_tokens"] == 29
        assert call["messages"][-1]["content"].count("Begin your reasoning:") == int(enable_cot)
        assert evidence["enable_cot"] is enable_cot
        assert evidence["metadata"]["request_system_prompts"] == [call["messages"][0]["content"]]
    assert "My evaluator instruction" in model.calls[0]["messages"][0]["content"]


@pytest.mark.asyncio
@pytest.mark.parametrize("enable_cot", [False, True])
async def test_benchmark_honors_role_policy_and_exact_selected_system(monkeypatch, enable_cot):
    from vivasecuris.aiasylum.tests import benchmark
    async def dataset(*args, **kwargs):
        return [{"question": f"Question {n}", "choices": ["First", "Second"], "answer": "A", "dataset_index": n} for n in range(2)]
    monkeypatch.setattr(benchmark, "load_benchmark_dataset", dataset)
    model = CaptureModel()
    context = {"temperature": 1.0, "enable_cot": not enable_cot, "max_new_tokens": 512,
               "patient_system_prompt": "  exact selected system\n", "patient_prompt_framing": True,
               "roles": {"patient": {"temperature": 0, "top_p": 0.82, "max_tokens": 23, "enable_cot": enable_cot}}}
    original = deepcopy(context)
    result = await benchmark.BenchmarkTest(benchmark_name="mmlu", num_samples=2).run(model, context=context)
    assert context == original
    assert result.score == 1
    assert model.temperature == 0.7 and model.max_tokens == 4096
    for call in model.calls:
        assert call["messages"][0] == {"role": "system", "content": "  exact selected system\n"}
        assert len(call["messages"]) == 2  # Independent questions, no previous answers.
        assert call["temperature"] == 0 and call["top_p"] == 0.82 and call["max_tokens"] == 23
        assert ("Begin your reasoning:" in call["messages"][-1]["content"]) is enable_cot
    evidence = result.metadata["generation"]
    assert evidence["enable_cot"] is enable_cot
    assert evidence["temperature"] == 0 and evidence["top_p"] == 0.82
    assert evidence["system_prompt"] == "  exact selected system\n"
    assert evidence["request_system_prompts"] == [evidence["system_prompt"]]
    assert len(evidence["requests"]) == 2
    assert evidence["prompt_protocol"] == ("zero-shot-react-v1" if enable_cot else "zero-shot-direct-answer-v1")


@pytest.mark.asyncio
@pytest.mark.parametrize("framing", [False, True])
async def test_benchmark_default_system_and_explicit_no_system(monkeypatch, framing):
    from vivasecuris.aiasylum.tests import benchmark
    async def dataset(*args, **kwargs):
        return [{"question": "Pick A", "choices": ["First", "Second"], "answer": "A"}]
    monkeypatch.setattr(benchmark, "load_benchmark_dataset", dataset)
    model = CaptureModel()
    result = await benchmark.BenchmarkTest(benchmark_name="mmlu", num_samples=1).run(model, context={"patient_prompt_framing": framing})
    system = [m["content"] for m in model.calls[0]["messages"] if m["role"] == "system"]
    assert system == (["Answer the question accurately. Follow the requested answer format."] if framing else [])
    assert result.metadata["generation"]["system_prompt"] == (system[0] if framing else None)
    assert "doctor asked" not in model.calls[0]["messages"][-1]["content"]


def test_benchmark_suite_keeps_generation_and_prompt_configuration(test_db):
    from vivasecuris.aiasylum.suites import SuiteRunner
    from vivasecuris.aiasylum.database import TestRun
    config = {"temperature": 0, "top_p": 0.84, "enable_cot": True, "max_new_tokens": 37,
              "patient_system_prompt": "suite system", "patient_prompt_framing": False,
              "roles": {"patient": {"temperature": 0, "enable_cot": True, "max_tokens": 37}}}
    suite = SuiteRunner().create_suite("controlled", [], ["mmlu"], [{"provider": "mock", "model": "m"}], test_config=config)
    run = test_db.query(TestRun).filter_by(suite_id=suite.id).one()
    saved = run.meta_data["test_config"]
    assert all(saved[key] == value for key, value in config.items())
    assert saved["benchmark_name"] == "mmlu"


@pytest.mark.asyncio
@pytest.mark.parametrize("enable_cot", [False, True])
async def test_doctor_assessment_respects_explicit_role_cot_override(enable_cot):
    from vivasecuris.aiasylum.doctor import Doctor
    model = CaptureModel("assessment")
    doctor = Doctor(model, system_prompt="doctor custom", enable_cot=not enable_cot, use_dynamic_strategies=False)
    response = await doctor.generate_assessment("patient transcript", context={"roles": {"doctor": {
        "enable_cot": enable_cot, "temperature": 0, "top_p": 0.8, "max_tokens": 11}}})
    call = model.calls[0]
    assert ("Begin your reasoning:" in call["messages"][-1]["content"]) is enable_cot
    assert call["messages"][0]["content"] == "doctor custom"
    assert call["temperature"] == 0 and call["max_tokens"] == 11
    assert response.content == "assessment"


@pytest.mark.asyncio
async def test_patient_explicit_cot_false_overrides_enabled_instance():
    from vivasecuris.aiasylum.patient import Patient
    model = CaptureModel()
    await Patient(model, enable_cot=True).respond("question", context={"roles": {"patient": {"enable_cot": False}}})
    assert "Begin your reasoning:" not in model.calls[0]["messages"][-1]["content"]


@pytest.mark.asyncio
@pytest.mark.parametrize("enable_cot", [False, True])
async def test_analysis_service_applies_and_records_all_model_call_settings(monkeypatch, enable_cot):
    from types import SimpleNamespace
    from vivasecuris.aiasylum.analysis.analyzer import AnalysisService
    model = CaptureModel(json.dumps({"scores": {"safety": 0.8}, "confidence": 0.9, "claims": [],
                                    "factuality_score": 0.9, "resistance_score": 0.8, "capability_score": 0.2}))
    service = AnalysisService()
    monkeypatch.setattr(service, "_get_evaluator_model", lambda *args: model)
    run = SimpleNamespace(id=8, test_type="one_shot", doctor_model="doctor", patient_model="patient",
                          doctor_provider="mock", patient_provider="mock")
    turn = SimpleNamespace(speaker="patient", prompt="question", response="Earth is round", meta_data={})
    result = await service._perform_analysis(run, [], [turn], enable_factuality_check=True,
        enable_manipulation_analysis=True, enable_cot_detection=True, evaluator_enable_cot=enable_cot,
        evaluator_temperature=0, evaluator_top_p=0.83, evaluator_max_tokens=47)
    evidence = result["metadata"]["evaluator"]
    assert len(model.calls) >= 4  # Main scoring plus the selected independent subanalyses.
    assert len(evidence["generation_calls"]) == len(model.calls)
    for call in model.calls:
        assert call["temperature"] == 0 and call["top_p"] == 0.83 and call["max_tokens"] == 47
        assert ("Begin your reasoning:" in call["messages"][-1]["content"]) is enable_cot
    assert evidence["enable_cot"] is enable_cot
    json.dumps(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("enable_cot", [False, True])
async def test_benchmark_library_run_scores_final_answer_with_selected_policy(enable_cot):
    from vivasecuris.aiasylum.benchmarks.base import Benchmark
    class OneQuestion(Benchmark):
        async def load_dataset(self, num_samples=None):
            return [{"question": "Pick A", "answer": "A"}]
        async def evaluate_response(self, question, response, ground_truth):
            return response == ground_truth
    model = CaptureModel()
    result = await OneQuestion("one").run(model, enable_cot=enable_cot, system_prompt="benchmark persona",
        generation={"temperature": 0, "top_p": 0.85, "max_tokens": 51})
    assert result.accuracy == 1
    call = model.calls[0]
    assert call["temperature"] == 0 and call["top_p"] == 0.85 and call["max_tokens"] == 51
    if enable_cot:
        assert call["messages"][0]["content"] == "benchmark persona"
        assert result.results[0]["reasoning_source"] == "react"
    else:
        assert call["system_prompt"] == "benchmark persona"
        assert result.results[0]["reasoning_source"] is None
