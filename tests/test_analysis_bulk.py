"""Bulk behavioral analysis should not mistake objective benchmarks for missing work."""

import pytest
from fastapi import BackgroundTasks

from vivasecuris.aiasylum.api.routes.analysis import analyze_unanalyzed
from vivasecuris.aiasylum.database import Assessment, TestResult, TestRun, get_session


@pytest.mark.asyncio
async def test_bulk_analysis_skips_benchmarks_and_already_analyzed_runs():
    with get_session() as session:
        rows = [TestRun(doctor_provider="transformers", doctor_model="base",
                        patient_provider="transformers", patient_model="base",
                        test_type=kind, status=status)
                for kind, status in [("benchmark", "completed"), ("one_shot", "completed"),
                                     ("one_shot", "completed"), ("analysis", "completed"),
                                     ("one_shot", "pending")]]
        session.add_all(rows)
        session.flush()
        ids = [row.id for row in rows]
        session.add(TestResult(test_run_id=ids[0], test_name="benchmark_mmlu",
                               input_prompt="Question", output_response="A", score=0.0))
        session.add(Assessment(test_run_id=ids[2], assessment_text="Reviewed",
                               scores={"overall": 0.8}, overall_score=0.8))
        session.commit()

    background_tasks = BackgroundTasks()
    result = await analyze_unanalyzed(background_tasks)

    assert result == {"started": 1, "test_run_ids": [ids[1]]}
    assert len(background_tasks.tasks) == 1
    with get_session() as session:
        created = session.query(TestRun).filter(TestRun.test_type == "analysis", TestRun.status == "pending").all()
        assert len(created) == 1
        assert created[0].meta_data["source_test_run_id"] == ids[1]
