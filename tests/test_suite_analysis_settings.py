"""Suite completion must not override or duplicate each run's analysis choice."""

import asyncio
from types import SimpleNamespace

import pytest

from tests.test_conversation_role_wiring import CapturedModel


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
async def test_suite_completion_honors_exact_analysis_settings_once(test_db, monkeypatch, enabled):
    from vivasecuris.aiasylum.api.routes import analysis, suites
    from vivasecuris.aiasylum.database import TestRun, TestSuite
    from vivasecuris.aiasylum.runner import runner
    from vivasecuris.aiasylum.suites import SuiteRunner

    analysis_calls = []

    async def capture_analysis(*args, **kwargs):
        analysis_calls.append((args, kwargs))

    monkeypatch.setattr(analysis, "_run_analysis_background", capture_analysis)
    monkeypatch.setattr(runner, "get_provider", lambda _: SimpleNamespace(
        create_model=lambda name: CapturedModel(name)))
    settings = {
        "enable_cot_detection": False,
        "cot_analysis_mode": "simple",
        "enable_factuality_check": True,
        "enable_manipulation_analysis": True,
        "evaluator_provider": "chosen-provider",
        "evaluator_model": "chosen-evaluator",
        "evaluator_system_prompt": "  Exact evaluator system\n",
        "evaluator_system_prompt_id": None,
        "evaluator_temperature": 0,
        "evaluator_top_p": 0.8,
        "evaluator_max_tokens": 137,
        "evaluator_enable_cot": True,
    }
    suite = SuiteRunner().create_suite(
        name="analysis settings", test_types=["one_shot"], benchmarks=[],
        models=[{"provider": "capture", "model": "patient-a"},
                {"provider": "capture", "model": "patient-b"}],
        doctor={"provider": "capture", "model": "doctor"},
        test_config={"prompt": "Answer briefly", "auto_analysis": enabled,
                     "analysis_config": settings},
    )
    runs = test_db.query(TestRun).filter_by(suite_id=suite.id).order_by(TestRun.id).all()
    source_ids = [run.id for run in runs]
    for run_id in source_ids:
        await suites._run_suite_test_background(run_id, suite.id)
        # Let the analysis coroutine run without doing model inference.
        await asyncio.sleep(0)
    await asyncio.sleep(0)
    test_db.expire_all()
    completed_suite = test_db.get(TestSuite, suite.id)
    assert completed_suite.status == "completed"
    assert completed_suite.completed_runs == 2
    analyses = test_db.query(TestRun).filter_by(test_type="analysis").all()
    assert len(analyses) == len(analysis_calls) == (2 if enabled else 0)
    if enabled:
        assert sorted(row.meta_data["source_test_run_id"] for row in analyses) == source_ids
        for row in analyses:
            assert row.doctor_provider == "chosen-provider"
            assert row.doctor_model == "chosen-evaluator"
            assert row.meta_data["analysis_config"] == {"enable_activation_patching": False, **settings}
        for args, kwargs in analysis_calls:
            assert not args  # Shared runner passes named, reviewable settings.
            assert kwargs["test_run_id"] in source_ids
            for key, value in settings.items():
                assert kwargs[key] == value
