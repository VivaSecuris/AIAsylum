"""Template substitutions preserve exact literal values through test execution."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from vivasecuris.aiasylum.database import PromptLibrary, TestRun as Run, TestResult as Result
from vivasecuris.aiasylum.models.base import ModelResponse
from vivasecuris.aiasylum.utils.prompt_variables import extract_variables, substitute_variables


@pytest.mark.parametrize("value", [
    r"C:\temp\new", r"\g<1>", r"\1", r"\unknown", "ending\\", "  exact\nlines\t ", "",
])
def test_variable_values_are_literal_text(value):
    assert substitute_variables("Before $value / $value after", {"value": value}) == f"Before {value} / {value} after"


@pytest.mark.parametrize("reverse", [False, True])
def test_inserted_variable_names_are_not_recursively_expanded(reverse):
    variables = {"first": "literal $other", "other": "expanded"}
    if reverse:
        variables = dict(reversed(list(variables.items())))
    assert substitute_variables("$first | $other | $missing", variables) == "literal $other | expanded | $missing"
    assert variables["first"] == "literal $other"


@pytest.mark.parametrize("prompt,variables,expected", [
    ("$name, $name_suffix / $name2!", {"name": "Ada"}, "Ada, $name_suffix / $name2!"),
    ("$name $name_suffix $name2", {"name": "A", "name_suffix": "B", "name2": "C"}, "A B C"),
    ("$naïve $123 $_under", {"naïve": "Unicode", "123": "numeric", "_under": "underscore"}, "Unicode numeric underscore"),
    ("$a.b / $axb", {"a.b": "literal dotted key"}, "literal dotted key / $axb"),
    ("${name} and {{name}} and $name", {"name": "Ada"}, "${name} and {{name}} and Ada"),
    ("$missing, $$missing and $name", {}, "$missing, $$missing and $name"),
])
def test_existing_placeholder_syntax_boundaries_and_missing_values_are_preserved(prompt, variables, expected):
    assert substitute_variables(prompt, variables) == expected
    assert extract_variables("$name_suffix $name2 $name $name") == ["name", "name2", "name_suffix"]


@pytest.mark.asyncio
@pytest.mark.parametrize("entrypoint", ["direct", "queued", "suite"])
@pytest.mark.parametrize("test_type,source", [
    ("one_shot", "library"), ("one_shot", "custom"),
    ("multi_shot", "library"), ("multi_shot", "custom"),
])
async def test_literal_values_reach_patient_and_saved_transcript(
    test_db, monkeypatch, entrypoint, test_type, source,
):
    from vivasecuris.aiasylum.runner import runner
    from vivasecuris.aiasylum.suites import SuiteRunner

    payload = r"C:\temp\new | \g<1> | literal $other"
    templates = ["  Task: $payload\nUnresolved: $missing / Known: $other  "]
    if test_type == "multi_shot":
        templates.append("Again: $payload\n")
    expected = [text.replace("$other", "expanded").replace("$payload", payload) for text in templates]
    rows = []
    config = {"variables": {"payload": payload, "other": "expanded"},
              "patient_prompt_framing": False, "auto_analysis": False}
    if source == "library":
        rows = [PromptLibrary(name=f"Literal variable test {index}", prompt_type="test_prompt",
                              target="patient", prompt_text=text) for index, text in enumerate(templates)]
        test_db.add_all(rows)
        test_db.commit()
        config.update({"prompt_id": rows[0].id} if test_type == "one_shot" else {"prompt_ids": [row.id for row in rows]})
    else:
        config.update({"prompt": templates[0]} if test_type == "one_shot" else {"prompts": templates})
    original = deepcopy(config)
    calls = []

    class Model:
        provider = "capture"
        supports_seed = True

        def __init__(self, name):
            self.model_name = name

        async def generate(self, prompt="", messages=None, **kwargs):
            calls.append((self.model_name, deepcopy(messages)))
            return ModelResponse("Observed answer", self.model_name, self.provider)

    monkeypatch.setattr(runner, "get_provider", lambda _: SimpleNamespace(create_model=Model))
    if entrypoint == "direct":
        await runner.TestRunner().run_test("capture", "doctor", "capture", "patient", test_type, config)
    elif entrypoint == "queued":
        run = Run(doctor_provider="capture", doctor_model="doctor", patient_provider="capture",
                  patient_model="patient", test_type=test_type, status="pending", meta_data={"test_config": config})
        test_db.add(run)
        test_db.commit()
        await runner.TestRunner().execute_test_run(run.id)
    else:
        suite = SuiteRunner().create_suite(
            name="Literal variables", test_types=[test_type], benchmarks=[],
            models=[{"provider": "capture", "model": "patient"}],
            doctor={"provider": "capture", "model": "doctor"}, test_config=config,
        )
        run = test_db.query(Run).filter_by(suite_id=suite.id).one()
        await runner.TestRunner().execute_test_run(run.id)

    assert [messages[-1]["content"] for name, messages in calls if name == "patient"] == expected
    test_db.expire_all()
    result = test_db.query(Result).one()
    assert [turn["prompt"] for turn in result.meta_data["conversation_history"]] == expected
    saved = test_db.query(Run).one()
    assert saved.status == "completed"
    assert saved.meta_data["test_config"]["variables"] == original["variables"]
    assert config == original
    for row, text in zip(rows, templates):
        test_db.refresh(row)
        assert row.prompt_text == text and row.usage_count == 1
