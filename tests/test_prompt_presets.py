"""Opt-in role presets remain visible and never overwrite user-owned prompts."""

import json
from copy import deepcopy

import pytest
from sqlalchemy import event

from scripts import seed_prompt_presets
from vivasecuris.aiasylum.database.models import PromptLibrary
from vivasecuris.aiasylum.prompt_presets import (
    CATALOG_CATEGORY, CATALOG_ID, CATALOG_TAG,
    COMMON_SYSTEM_CATALOG_ID, COMMON_SYSTEM_CATEGORY, COMMON_SYSTEM_TAG,
    COMMON_SYSTEM_PRESETS, PROMPT_PRESETS, ROLE_PROMPT_PRESETS,
    install_prompt_presets, select_prompt_presets,
)
from vivasecuris.aiasylum.utils.prompt_variables import extract_variables


def _snapshot(session):
    session.expire_all()
    return [
        {column.name: deepcopy(getattr(row, "meta_data" if column.name == "metadata" else column.name))
         for column in PromptLibrary.__table__.columns}
        for row in session.query(PromptLibrary).order_by(PromptLibrary.id)
    ]


def test_catalog_contains_matching_user_system_pairs_for_every_role():
    assert len(ROLE_PROMPT_PRESETS) == 12
    assert len(PROMPT_PRESETS) == 20
    assert len({p.name for p in PROMPT_PRESETS}) == len(PROMPT_PRESETS)
    assert len({p.preset_id for p in PROMPT_PRESETS}) == len(PROMPT_PRESETS)
    assert {p.target for p in PROMPT_PRESETS} == {"doctor", "patient", "evaluator"}
    for role in ("doctor", "patient", "evaluator"):
        selected = [p for p in ROLE_PROMPT_PRESETS if p.target == role]
        pairs = {p.pair for p in selected}
        assert len(pairs) == 2
        for pair in pairs:
            assert {p.prompt_type for p in selected if p.pair == pair} == {"test_prompt", "system_prompt"}
    for preset in ROLE_PROMPT_PRESETS:
        row = preset.as_row()
        assert 0 < len(row["name"]) <= 200
        assert row["category"] == CATALOG_CATEGORY
        assert CATALOG_TAG in row["tags"] and preset.target in row["tags"]
        assert row["meta_data"]["preset_catalog"] == CATALOG_ID
        assert len(row["meta_data"]["observable_checks"]) >= 2
        assert extract_variables(row["prompt_text"]) == []  # usable without unresolved template fields
        # The library intentionally hides attack/benchmark collections by default.
        assert not {"adversarial", "jailbreak", "benchmark"}.intersection(row["tags"])
        assert row["category"] not in {"forbidden_question", "adversarial", "benchmark", "roleplay"}


def test_common_system_catalog_has_reviewable_sources_examples_and_exact_roles():
    assert len(COMMON_SYSTEM_PRESETS) == 8
    assert {p.target for p in COMMON_SYSTEM_PRESETS} == {"doctor", "patient", "evaluator"}
    for preset in COMMON_SYSTEM_PRESETS:
        row = preset.as_row()
        metadata = row["meta_data"]
        assert row["prompt_type"] == "system_prompt"
        assert row["category"] == COMMON_SYSTEM_CATEGORY
        assert COMMON_SYSTEM_TAG in row["tags"] and preset.target in row["tags"]
        assert metadata["preset_catalog"] == COMMON_SYSTEM_CATALOG_ID
        assert metadata["preset_id"].startswith(COMMON_SYSTEM_CATALOG_ID + "/")
        assert metadata["message_role"] == "system"
        assert metadata["rationale"] and metadata["example_user_message"]
        assert len(metadata["observable_checks"]) >= 2
        assert metadata["source_references"]
        for source in metadata["source_references"]:
            assert source["url"].startswith(("https://platform.claude.com/docs/",
                                             "https://ai.google.dev/",
                                             "https://learn.microsoft.com/"))
            assert source["title"] and source["principle"]
        assert extract_variables(row["prompt_text"]) == []
        assert not {"adversarial", "jailbreak", "benchmark"}.intersection(row["tags"])


def test_dry_run_and_repeated_apply_do_not_modify_existing_rows(test_db):
    test_db.add(PromptLibrary(name="My existing prompt", prompt_text="  keep my whitespace\n",
                              target="patient", usage_count=8, meta_data={"owner": "user"}))
    test_db.commit()
    before = _snapshot(test_db)

    plan = install_prompt_presets(test_db)
    assert plan["mode"] == "dry_run" and len(plan["would_add"]) == len(PROMPT_PRESETS)
    test_db.commit()
    assert _snapshot(test_db) == before

    report = install_prompt_presets(test_db, apply=True)
    test_db.commit()
    assert len(report["added"]) == len(PROMPT_PRESETS) and not report["skipped"]
    installed = _snapshot(test_db)
    assert installed[:1] == before
    again = install_prompt_presets(test_db, apply=True)
    test_db.commit()
    assert again["added"] == [] and len(again["skipped"]) == len(PROMPT_PRESETS)
    assert _snapshot(test_db) == installed


def test_name_collision_and_renamed_edited_preset_are_preserved(test_db):
    collision = PromptLibrary(
        name=PROMPT_PRESETS[0].name, prompt_text="Existing user text with the same name",
        prompt_type="test_prompt", target="patient", usage_count=42,
        category="personal", tags=["edited"], meta_data={"custom": {"keep": True}},
    )
    test_db.add(collision)
    test_db.commit()
    before = _snapshot(test_db)
    report = install_prompt_presets(test_db, apply=True)
    test_db.commit()
    assert len(report["added"]) == len(PROMPT_PRESETS) - 1
    assert report["skipped"] == [{"name": PROMPT_PRESETS[0].name, "existing_id": collision.id}]
    assert _snapshot(test_db)[:1] == before

    installed = test_db.query(PromptLibrary).filter_by(name=PROMPT_PRESETS[1].name).one()
    installed.name = "Renamed and customized by the user"
    installed.prompt_text = "  my edited text\n"
    installed.description = "My description"
    installed.target = "evaluator"
    installed.prompt_type = "system_prompt"
    installed.tags = ["personal"]
    installed.category = "personal"
    installed.usage_count = 9
    installed.meta_data = {**installed.meta_data, "my_metadata": "preserve"}
    test_db.commit()
    edited = _snapshot(test_db)

    again = install_prompt_presets(test_db, apply=True)
    test_db.commit()
    assert not again["added"] and len(again["skipped"]) == len(PROMPT_PRESETS)
    assert _snapshot(test_db) == edited


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["doctor", "patient", "evaluator"])
@pytest.mark.parametrize("prompt_type", ["test_prompt", "system_prompt"])
async def test_installed_presets_are_visible_through_role_and_type_filters(test_db, role, prompt_type):
    from vivasecuris.aiasylum.api.routes.prompts import get_prompt, list_prompts

    install_prompt_presets(test_db, apply=True)
    test_db.commit()
    results = await list_prompts(category=CATALOG_CATEGORY, target=role,
                                 prompt_type=prompt_type, search="Role Lab v1", limit=100, offset=0)
    expected = {p.name: p for p in ROLE_PROMPT_PRESETS if p.target == role and p.prompt_type == prompt_type}
    assert {p.name for p in results} == set(expected)
    for prompt in results:
        fetched = await get_prompt(prompt.id)
        assert fetched.prompt_text == expected[prompt.name].prompt_text
        assert fetched.target == role and fetched.prompt_type == prompt_type
        assert fetched.metadata["preset_id"] == expected[prompt.name].preset_id


def test_cli_requires_apply_and_catalog_does_not_open_database(test_db, monkeypatch, capsys):
    def no_session():
        raise AssertionError("catalog display must not open a database session")

    with monkeypatch.context() as patch:
        patch.setattr(seed_prompt_presets, "get_session", no_session)
        assert seed_prompt_presets.main(["--catalog"]) == 0
    catalog = json.loads(capsys.readouterr().out)
    assert len(catalog) == len(PROMPT_PRESETS) and catalog[0]["prompt_text"] == PROMPT_PRESETS[0].prompt_text

    assert seed_prompt_presets.main([]) == 0
    report = json.loads(capsys.readouterr().out)
    assert len(report["would_add"]) == len(PROMPT_PRESETS)
    assert test_db.query(PromptLibrary).count() == 0

    assert seed_prompt_presets.main(["--apply"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert len(report["added"]) == len(PROMPT_PRESETS)
    assert test_db.query(PromptLibrary).count() == len(PROMPT_PRESETS)
    assert seed_prompt_presets.main(["--apply"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert not report["added"] and len(report["skipped"]) == len(PROMPT_PRESETS)


def test_common_catalog_upgrade_preserves_prior_catalog_and_user_edits(test_db):
    install_prompt_presets(test_db, apply=True, catalog_id=CATALOG_ID)
    test_db.commit()
    original = test_db.query(PromptLibrary).filter_by(name=ROLE_PROMPT_PRESETS[0].name).one()
    original.name = "My renamed interviewer"
    original.prompt_text = "  My exact custom instructions\n"
    original.usage_count = 27
    test_db.commit()
    prior_catalog = _snapshot(test_db)

    report = install_prompt_presets(test_db, apply=True, catalog_id=COMMON_SYSTEM_CATALOG_ID)
    test_db.commit()
    assert report["catalog"] == COMMON_SYSTEM_CATALOG_ID
    assert report["added"] == [p.name for p in COMMON_SYSTEM_PRESETS]
    assert not report["skipped"]
    assert _snapshot(test_db)[:len(prior_catalog)] == prior_catalog

    new = test_db.query(PromptLibrary).filter_by(name=COMMON_SYSTEM_PRESETS[0].name).one()
    new.name = "My renamed AI assistant"
    new.prompt_text = "  Custom identity policy\n"
    new.meta_data = {**new.meta_data, "local_notes": {"keep": True}}
    test_db.commit()
    edited = _snapshot(test_db)
    again = install_prompt_presets(test_db, apply=True)
    test_db.commit()
    assert not again["added"] and len(again["skipped"]) == len(PROMPT_PRESETS)
    assert _snapshot(test_db) == edited


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["doctor", "patient", "evaluator"])
async def test_common_presets_are_available_in_existing_system_role_filters(test_db, role):
    from vivasecuris.aiasylum.api.routes.prompts import get_prompt, list_prompts

    install_prompt_presets(test_db, apply=True)
    test_db.commit()
    expected = {p.name: p for p in COMMON_SYSTEM_PRESETS if p.target == role}
    # No category filter: this is the system picker contract, including default visibility.
    results = await list_prompts(target=role, prompt_type="system_prompt",
                                 search="Common Systems v1", limit=100, offset=0)
    assert {p.name for p in results} == set(expected)
    for prompt in results:
        fetched = await get_prompt(prompt.id)
        preset = expected[prompt.name]
        assert fetched.prompt_text == preset.prompt_text
        assert fetched.metadata == preset.as_row()["meta_data"]
        assert fetched.target == role


def test_cli_catalog_selection_is_read_only_and_limits_install(test_db, monkeypatch, capsys):
    with monkeypatch.context() as patch:
        patch.setattr(seed_prompt_presets, "get_session", lambda: pytest.fail("catalog opened DB"))
        assert seed_prompt_presets.main(["--catalog", "--catalog-id", COMMON_SYSTEM_CATALOG_ID]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert [row["name"] for row in rows] == [p.name for p in COMMON_SYSTEM_PRESETS]
    assert seed_prompt_presets.main(["--apply", "--catalog-id", COMMON_SYSTEM_CATALOG_ID]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["catalog"] == COMMON_SYSTEM_CATALOG_ID and len(report["added"]) == 8
    assert {r.name for r in test_db.query(PromptLibrary)} == {p.name for p in COMMON_SYSTEM_PRESETS}
    with pytest.raises(ValueError, match="Unknown prompt catalog"):
        select_prompt_presets("missing-catalog")


def test_failed_install_rolls_back_all_new_rows(test_db):
    test_db.add(PromptLibrary(name="Existing", prompt_text="Preserve"))
    test_db.commit()
    before = _snapshot(test_db)
    inserted = []

    def fail_during_insert(_mapper, _connection, row):
        inserted.append(row.name)
        if len(inserted) == 3:
            raise RuntimeError("Simulated insert failure")

    event.listen(PromptLibrary, "before_insert", fail_during_insert)
    try:
        with pytest.raises(RuntimeError, match="Simulated insert failure"):
            seed_prompt_presets.main(["--apply"])
    finally:
        event.remove(PromptLibrary, "before_insert", fail_during_insert)
    assert len(inserted) == 3
    assert _snapshot(test_db) == before
