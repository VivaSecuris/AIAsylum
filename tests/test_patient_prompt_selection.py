"""Role-specific library presets cannot be executed by the wrong role."""

from types import SimpleNamespace
from copy import deepcopy

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vivasecuris.aiasylum.database import PromptLibrary, TestRun as Run, TestResult as Result, TestSuite as Suite
from vivasecuris.aiasylum.models.base import ModelResponse


def preset(db, target, *, kind="test_prompt"):
    row = PromptLibrary(name=f"{target} preset", prompt_type=kind, target=target, prompt_text="Exact prompt for $topic")
    db.add(row)
    db.commit()
    return row


@pytest.fixture
def client(monkeypatch):
    from vivasecuris.aiasylum.api.routes import test_runs, suites
    async def no_run(*args):
        pass
    monkeypatch.setattr(test_runs, "_run_test_background", no_run)
    monkeypatch.setattr(suites, "_run_suite_test_background", no_run)
    app = FastAPI()
    app.include_router(test_runs.router, prefix="/runs")
    app.include_router(suites.router, prefix="/suites")
    return TestClient(app)


CASES = [("one_shot", "prompt_id"), ("multi_shot", "prompt_id"), ("multi_shot", "prompt_ids")]


@pytest.mark.parametrize("target", ["doctor", "evaluator"])
@pytest.mark.parametrize("test_type,key", CASES)
@pytest.mark.parametrize("endpoint", ["runs", "suites"])
def test_api_rejects_wrong_role_before_creating_or_queuing_work(test_db, client, target, test_type, key, endpoint):
    row = preset(test_db, target)
    config = {key: [row.id] if key == "prompt_ids" else row.id}
    if endpoint == "runs":
        body = {"doctor_provider": "mock", "doctor_model": "d", "patient_provider": "mock",
                "patient_model": "p", "test_type": test_type, "test_config": config}
    else:
        body = {"models": [{"provider": "mock", "model": "p"}], "test_types": [test_type], "test_config": config}
    response = client.post(f"/{endpoint}/", json=body)
    assert response.status_code == 422, response.text
    assert "another role" in response.json()["detail"] and target in response.json()["detail"]
    assert test_db.query(Run).count() == test_db.query(Suite).count() == 0
    test_db.refresh(row)
    assert not row.usage_count


@pytest.mark.asyncio
@pytest.mark.parametrize("target", ["doctor", "evaluator"])
@pytest.mark.parametrize("test_type,key", CASES)
@pytest.mark.parametrize("entrypoint", ["direct", "queued"])
async def test_execution_rejects_wrong_role_before_provider_creation(test_db, monkeypatch, target, test_type, key, entrypoint):
    from vivasecuris.aiasylum.runner import runner
    row = preset(test_db, target)
    config = {key: [row.id] if key == "prompt_ids" else row.id, "auto_analysis": False}
    calls = []
    def provider(name):
        calls.append(name)
        raise AssertionError("Provider must not be constructed for an invalid preset")
    monkeypatch.setattr(runner, "get_provider", provider)
    with pytest.raises(Exception, match="another role"):
        if entrypoint == "direct":
            await runner.TestRunner().run_test("mock", "d", "mock", "p", test_type, config)
        else:
            run = Run(doctor_provider="mock", doctor_model="d", patient_provider="mock", patient_model="p",
                      test_type=test_type, status="pending", meta_data={"test_config": config})
            test_db.add(run)
            test_db.commit()
            await runner.TestRunner().execute_test_run(run.id)
    assert not calls
    assert not test_db.query(Result).count()
    test_db.refresh(row)
    assert not row.usage_count


@pytest.mark.asyncio
@pytest.mark.parametrize("target", [None, "patient"])
@pytest.mark.parametrize("test_type,key", CASES)
@pytest.mark.parametrize("entrypoint", ["direct", "queued"])
async def test_patient_and_legacy_presets_reach_patient_unchanged(test_db, monkeypatch, target, test_type, key, entrypoint):
    from vivasecuris.aiasylum.runner import runner
    row = preset(test_db, target)
    calls = []
    class Model:
        provider = "mock"
        supports_seed = True
        def __init__(self, name):
            self.model_name = name
        async def generate(self, prompt="", messages=None, **kwargs):
            calls.append((self.model_name, deepcopy(messages)))
            return ModelResponse("answer", self.model_name, "mock")
    monkeypatch.setattr(runner, "get_provider", lambda _: SimpleNamespace(create_model=Model))
    config = {key: [row.id] if key == "prompt_ids" else row.id, "variables": {"topic": "music"},
              "patient_prompt_framing": False, "auto_analysis": False}
    if entrypoint == "direct":
        await runner.TestRunner().run_test("mock", "d", "mock", "p", test_type, config)
    else:
        run = Run(doctor_provider="mock", doctor_model="d", patient_provider="mock", patient_model="p",
                  test_type=test_type, status="pending", meta_data={"test_config": config})
        test_db.add(run)
        test_db.commit()
        await runner.TestRunner().execute_test_run(run.id)
    assert [messages[-1]["content"] for name, messages in calls if name == "p"] == ["Exact prompt for music"]
    test_db.refresh(row)
    assert row.usage_count == 1


@pytest.mark.parametrize("target", [None, "patient"])
@pytest.mark.parametrize("endpoint", ["runs", "suites"])
def test_api_accepts_legacy_and_patient_presets(test_db, client, target, endpoint):
    row = preset(test_db, target)
    config = {"prompt_id": row.id}
    body = ({"doctor_provider": "mock", "doctor_model": "d", "patient_provider": "mock", "patient_model": "p",
             "test_type": "one_shot", "test_config": config} if endpoint == "runs" else
            {"models": [{"provider": "mock", "model": "p"}], "test_types": ["one_shot"], "test_config": config})
    response = client.post(f"/{endpoint}/", json=body)
    assert response.status_code == 200, response.text


def test_custom_text_and_doctor_goal_are_not_role_filtered(test_db):
    from vivasecuris.aiasylum.runner.run_config import validate_patient_prompt_selection
    validate_patient_prompt_selection(test_db, "one_shot", {"prompt": "Explicit custom text", "doctor_goal": "An assessment goal"})
    doctor = preset(test_db, "doctor")
    validate_patient_prompt_selection(test_db, "conversation", {"prompt_id": doctor.id})


def test_system_preset_cannot_be_used_as_patient_user_prompt(test_db):
    from vivasecuris.aiasylum.runner.run_config import patient_test_prompts
    system = preset(test_db, "patient", kind="system_prompt")
    with pytest.raises(ValueError, match="unavailable"):
        patient_test_prompts(test_db, [system.id])
