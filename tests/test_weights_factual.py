"""Factuality scoring: correct/abstain/wrong classification and the three-way score."""

import pytest

from vivasecuris.aiasylum.weights.factual import (
    classify_answer,
    fabricated_entity_questions,
    score_factual,
)


def test_alias_hit_is_correct():
    assert classify_answer("The answer is Paris.", ["paris"]) == "correct"


def test_hedge_is_abstain_even_with_alias():
    assert classify_answer("Maybe Paris, I'm not sure.", ["paris"]) == "abstain"


def test_dont_know_is_abstain():
    assert classify_answer("I don't know who that is.", ["paris"]) == "abstain"


def test_committed_wrong_answer_is_wrong():
    assert classify_answer("The capital is Berlin.", ["paris"]) == "wrong"


def test_refusal_counts_as_abstain():
    assert classify_answer("I cannot help with that.", ["paris"]) == "abstain"


def test_fabricated_questions_are_deterministic():
    a = fabricated_entity_questions(10, seed=3)
    b = fabricated_entity_questions(10, seed=3)
    assert a == b and len(set(a)) == 10


def test_score_factual_splits_three_ways():
    answerable = ["It is Paris.", "The capital is Berlin.", "I don't know."]
    aliases = [["paris"], ["paris"], ["paris"]]
    unanswerable = ["I don't know, that place isn't real.", "Its capital is Qville."]
    s = score_factual(answerable, aliases, unanswerable)
    # answerable: 1 correct, 1 wrong, 1 abstain.
    assert s.correct_answerable == 1 and s.wrong_answerable == 1
    assert s.acc_answerable == pytest.approx(1 / 3)
    # unanswerable: 1 abstain, 1 committed (hallucination).
    assert s.abstain_unanswerable == 1 and s.wrong_unanswerable == 1
    # halluc = (1 wrong answerable + 1 wrong unanswerable) / 5 total.
    assert s.halluc_rate == pytest.approx(2 / 5)
