"""Reproducible benchmark selection, final-answer scoring and isolated trials."""
import asyncio
import copy
import random
import sys
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from vivasecuris.aiasylum.benchmarks import datasets as data
from vivasecuris.aiasylum.benchmarks.simple import exact_response_matches
from vivasecuris.aiasylum.tests.benchmark import BenchmarkTest
from vivasecuris.aiasylum.models.base import ModelResponse


class FakeDataset(list):
    _fingerprint = 'same-source-content'


@pytest.fixture
def fake_dataset(monkeypatch):
    rows = FakeDataset([{'question': f'Question {i}', 'answer': i % 4,
        'choices': ['alpha', 'beta', 'gamma', 'delta'], 'subject': 'science'} for i in range(40)])
    calls = []
    def load(path, **kwargs):
        calls.append((path, kwargs))
        return rows
    monkeypatch.setitem(sys.modules, 'datasets', SimpleNamespace(load_dataset=load))
    return rows, calls


@pytest.mark.asyncio
async def test_same_seed_same_order_independent_of_run_and_global_rng(fake_dataset):
    state = random.getstate()
    first = await data.load_benchmark_dataset('mmlu', 7, test_run_id=1, seed=17, revision='a' * 40)
    second = await data.load_benchmark_dataset('mmlu', 7, test_run_id=999, seed=17, revision='a' * 40)
    other = await data.load_benchmark_dataset('mmlu', 7, seed=18)
    assert first.provenance == second.provenance
    assert first.provenance['ordered_sample_hash'] != other.provenance['ordered_sample_hash']
    assert random.getstate() == state
    assert len(set(first.provenance['sample_ids'])) == 7
    assert first.provenance['resolved_revision'] == 'a' * 40
    assert fake_dataset[1][0] == ('cais/mmlu', {'split': 'test', 'name': 'all', 'revision': 'a' * 40})


@pytest.mark.asyncio
async def test_requested_and_available_count_are_distinct(fake_dataset):
    result = await data.load_benchmark_dataset('mmlu', 100)
    assert result.provenance['requested_count'] == 100
    assert result.provenance['actual_count'] == result.provenance['available_count'] == 40
    assert result.provenance['resolved_revision'] is None


@pytest.mark.asyncio
async def test_changed_answer_changes_sample_hash(fake_dataset):
    first = await data.load_benchmark_dataset_all('mmlu')
    fake_dataset[0][0]['answer'] = 3
    second = await data.load_benchmark_dataset_all('mmlu')
    assert first.provenance['sample_ids'] == second.provenance['sample_ids']
    assert first.provenance['ordered_sample_hash'] != second.provenance['ordered_sample_hash']


@pytest.mark.asyncio
@pytest.mark.parametrize('all_rows', [True, False])
async def test_source_failure_never_returns_toy_fallback(monkeypatch, all_rows):
    def broken(*args, **kwargs):
        raise OSError('dataset unavailable')
    monkeypatch.setitem(sys.modules, 'datasets', SimpleNamespace(load_dataset=broken))
    with pytest.raises(ValueError, match='No substitute questions were used'):
        if all_rows:
            await data.load_benchmark_dataset_all('mmlu')
        else:
            await data.load_benchmark_dataset('mmlu', 2)


def test_arc_uses_dataset_choice_labels_and_winogrande_is_one_based():
    arc = data.standardize_benchmark_row('arc', data.BENCHMARK_DATASETS['arc'], {
        'question': 'Which?', 'answerKey': '2',
        'choices': {'label': ['1', '2', '3'], 'text': ['a', 'b', 'c']}})
    assert arc['choices'] == ['a', 'b', 'c']
    assert arc['answer_letter'] == 'B'
    wino = data.standardize_benchmark_row('winogrande', data.BENCHMARK_DATASETS['winogrande'], {
        'sentence': 'The _ opened', 'option1': 'door', 'option2': 'floor', 'answer': '1'})
    assert wino['answer_letter'] == 'A'
    assert wino['choices'] == ['door', 'floor']


@pytest.mark.parametrize('response,expected', [
    ('Because I disagree.', False),
    ('A is an option but I choose B.', False),
    ('B', False),
    ('A', True),
    ('(A)', True),
    ('The answer is A.\nExplanation follows.', True),
    ('<think>A looks plausible.</think>Final answer: B', False),
    ('<think>B looks plausible.</think>Final answer: A', True),
    ('<think>The answer is A', False),
    ('Final answer: A\nFinal answer: C', False),
    ('alpha', True),
])
def test_mcq_scores_final_answer_not_substring(response, expected):
    assert exact_response_matches(response, 'A', ['alpha', 'beta', 'gamma', 'delta']) is expected


@pytest.mark.parametrize('response,expected', [
    ('142', False), ('42', True), ('Final answer: 42 dollars.', True),
    ('42 was my first guess. Final answer: 43', False),
    ('<think>I calculated 42.</think>Final answer: 9', False),
    ('<think>Answer: 42', False),
    ('We add the amounts.\n#### 42', True),
    ('\\boxed{42}', True), ('42 or 43', False),
])
def test_gsm8k_extracts_reference_and_final_numeric_answer(response, expected):
    assert exact_response_matches(response, 'Reasoning with 9 steps\n#### 42', numeric=True) is expected


def test_short_freeform_answers_are_not_vacuously_correct():
    assert not exact_response_matches('yes', 'no')
    assert not exact_response_matches('maybe no', 'no')
    assert exact_response_matches('Final answer: no.', 'no')


def test_manual_selection_rejects_bad_indices():
    with pytest.raises(ValueError):
        data.filter_prompts_by_selection([{'question': 'one'}], [2])
    with pytest.raises(ValueError):
        data.parse_index_selection('5-2', 10)


class FakeModel:
    provider = 'mock'
    temperature = 0.7
    max_tokens = 4096
    def __init__(self):
        self.calls = []
    async def generate(self, **kwargs):
        self.calls.append((copy.deepcopy(kwargs), self.temperature, self.max_tokens))
        return ModelResponse(content='A', model='mock', provider=self.provider, finish_reason='stop')


@pytest.mark.asyncio
async def test_one_shot_is_independent_greedy_and_records_provenance(fake_dataset):
    model = FakeModel()
    result = await BenchmarkTest(benchmark_name='mmlu', num_samples=3).run(
        model, context={'seed': 7, 'max_new_tokens': 16, 'temperature': 0.9})
    assert len(model.calls) == 3
    for call, temperature, budget in model.calls:
        assert [m['role'] for m in call['messages']] == ['system', 'user']
        assert 'doctor' not in call['messages'][-1]['content'].lower()
        assert call['temperature'] == temperature == 0.0
        assert call['seed'] == 7 and budget == 16
    assert model.temperature == 0.7 and model.max_tokens == 4096
    assert result.metadata['dataset_provenance']['actual_count'] == 3
    assert result.metadata['scoring'] == {'version': 'final-answer-v4', 'method': 'mcq_final_answer'}
    assert len({r['sample_id'] for r in result.metadata['results']}) == 3
    assert result.metadata['generation']['prompt_protocol'] == 'zero-shot-direct-answer-v1'


@pytest.mark.asyncio
async def test_local_cancellation_does_not_release_gpu_while_generation_runs(monkeypatch, fake_dataset, tmp_path):
    from vivasecuris.aiasylum.api import model_jobs
    monkeypatch.setattr(model_jobs, '_slot', None)
    monkeypatch.setattr(model_jobs, '_held_by', None)
    monkeypatch.setattr(model_jobs, '_waiting', [])
    monkeypatch.setenv('AIASYLUM_MODEL_LOCK', str(tmp_path / 'gpu.lock'))
    cleanup = []
    monkeypatch.setattr(BenchmarkTest, '_clear_model_cache', staticmethod(lambda: cleanup.append('clear')))
    started = asyncio.Event()
    release = asyncio.Event()
    class SlowModel(FakeModel):
        provider = 'transformers'
        async def generate(self, **kwargs):
            started.set()
            await release.wait()
            return ModelResponse('A', 'fake', self.provider)
    job = asyncio.create_task(BenchmarkTest(benchmark_name='mmlu', num_samples=1).run(SlowModel(), context={'test_run_id': 45}))
    await asyncio.wait_for(started.wait(), 3)
    job.cancel()
    await asyncio.sleep(0)
    assert model_jobs.slot_status()['held_by'] == 'benchmark run 45'
    assert not job.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await job
    assert model_jobs.slot_status()['held_by'] is None
    assert cleanup == ['clear', 'clear']


def test_run_request_validates_sample_seed_and_budget():
    from vivasecuris.aiasylum.api.routes.benchmarks import BenchmarkRequest
    for invalid in ({'num_samples': 0}, {'seed': -1}, {'max_new_tokens': 0}, {'max_new_tokens': 8193}):
        with pytest.raises(ValidationError):
            BenchmarkRequest(provider='transformers', model='example/model', benchmark='mmlu', **invalid)
    valid = BenchmarkRequest(provider='transformers', model='example/model', benchmark='mmlu')
    assert (valid.num_samples, valid.seed, valid.max_new_tokens) == (100, 0, 512)


@pytest.mark.asyncio
async def test_runner_persists_matching_trials_and_failure_without_score(monkeypatch, fake_dataset, test_db):
    from vivasecuris.aiasylum.database import TestRun as Run, TestResult as Result
    from vivasecuris.aiasylum.runner import runner as runner_module
    model = FakeModel()
    monkeypatch.setattr(runner_module, 'get_provider', lambda _name: SimpleNamespace(create_model=lambda _model: model))
    run_ids = []
    for name in ('first-model', 'second-model'):
        row = Run(doctor_provider='mock', doctor_model=name, patient_provider='mock', patient_model=name,
            test_type='benchmark', status='pending', meta_data={'benchmark': 'mmlu', 'num_samples': 3,
                'test_config': {'seed': 13, 'max_new_tokens': 20, 'dataset_revision': 'a' * 40}})
        test_db.add(row)
        test_db.commit()
        run_ids.append(row.id)
        await runner_module.TestRunner().execute_test_run(row.id)
    test_db.expire_all()
    results = test_db.query(Result).filter(Result.test_run_id.in_(run_ids)).order_by(Result.test_run_id).all()
    assert len(results) == 2
    assert results[0].meta_data['dataset_provenance'] == results[1].meta_data['dataset_provenance']
    assert results[0].meta_data['generation'] == results[1].meta_data['generation']
    assert all(row.score is not None for row in results)
    def unavailable(*args, **kwargs):
        raise OSError('upstream unavailable')
    monkeypatch.setitem(sys.modules, 'datasets', SimpleNamespace(load_dataset=unavailable))
    failed = Run(doctor_provider='mock', doctor_model='third', patient_provider='mock', patient_model='third',
        test_type='benchmark', status='pending', meta_data={'benchmark': 'mmlu', 'num_samples': 3, 'test_config': {}})
    test_db.add(failed)
    test_db.commit()
    with pytest.raises(ValueError, match='No substitute questions were used'):
        await runner_module.TestRunner().execute_test_run(failed.id)
    test_db.expire_all()
    assert test_db.get(Run, failed.id).status == 'failed'
    assert 'No substitute questions were used' in test_db.get(Run, failed.id).meta_data['error']
    assert test_db.query(Result).filter(Result.test_run_id == failed.id).count() == 0


@pytest.mark.parametrize('response,reference', [
    ('The final answer is: $100', 'Work\n#### 100'),
    ('Therefore, the final answer is: 623 pounds.', 'Work\n#### 623'),
    ('**The final answer is:** **$100**', '#### 100'),
    ('Therefore, the final answer is: **623** pounds.', '#### 623'),
    ('The final answer is: `$1,000.00`', '#### 1000'),
    ('Final answer = £100', '#### 100'),
    ('The answer is: A.', 'A'),
])
def test_real_pilot_answer_formats(response, reference):
    if reference == 'A':
        assert exact_response_matches(response, reference, ['alpha', 'beta'])
    else:
        assert exact_response_matches(response, reference, numeric=True)


@pytest.mark.parametrize('response', [
    'The final answer is: 100 or 101',
    'The final answer is: 100 maybe',
    'The final answer is: 100 is wrong',
    'myanswer is: 100',
    'The final answer is: 1,00',
    '<think>The final answer is: $100',
    '<analysis>The final answer is: $100',
])
def test_ambiguous_or_unfinished_formats_remain_invalid(response):
    assert not exact_response_matches(response, '#### 100', numeric=True)


@pytest.mark.asyncio
async def test_usage_truncation_and_invalid_answers_are_recorded(fake_dataset):
    rows, _ = fake_dataset
    rows[:] = [{'question': f'Math {i}', 'answer': 'Work\n#### 100'} for i in range(3)]
    class DiagnosticModel(FakeModel):
        async def generate(self, **kwargs):
            responses = [
                ('The final answer is: $100', 'stop', 8),
                ('<think>It might be 100 but', 'length', 16),
                ('Therefore, the final answer is: 623 pounds.', 'stop', 12),
            ]
            answer, finish, count = responses[len(self.calls)]
            self.calls.append(kwargs)
            return ModelResponse(answer, 'mock', 'mock', finish_reason=finish,
                usage={'completion_tokens': count, 'prompt_tokens': 20, 'total_tokens': count + 20})
    result = await BenchmarkTest(benchmark_name='gsm8k', num_samples=3, max_new_tokens=16).run(DiagnosticModel())
    assert result.score == 1 / 3
    assert result.metadata['scoring']['version'] == 'final-answer-v4'
    assert result.metadata['truncated_count'] == 1
    assert result.metadata['invalid_answer_count'] == 1
    answers = result.metadata['results']
    assert [answer['answer_valid'] for answer in answers] == [True, False, True]
    assert [answer['truncated'] for answer in answers] == [False, True, False]
    assert [answer['usage']['completion_tokens'] for answer in answers] == [8, 16, 12]



def test_markdown_normalization_does_not_join_arithmetic_digits():
    assert not exact_response_matches('2**3**', '23', numeric=True)
    assert not exact_response_matches('<thinking>The final answer is: $100', '100', numeric=True)


@pytest.mark.parametrize('response,reference,correct', [
    ('Final answer: <623>', '#### 623', True),
    ('Their combined weight is 623 pounds. Final answer: <623>', '#### 623', True),
    ('Final answer: <number> 30', '#### 30', True),
    ('Final answer: <number> 14,000', '#### 14000', True),
    ('Final answer: <number> $100', '#### 100', True),
    ('Final answer: <number> 150', '#### 100', False),
    ('Final answer: <350>', '#### 100', False),
    ('Final answer: <number>', '#### 100', False),
    ('Final answer: <number> 30 or 31', '#### 30', False),
    ('Final answer: <30 or 31>', '#### 30', False),
    ('Final answer: <30 + 31>', '#### 61', False),
    ('Final answer: <<30>>', '#### 30', False),
    ('Final answer: <number> 30\nFinal answer: 31', '#### 30', False),
    ('<think>Final answer: <number> 30', '#### 30', False),
    ('<think>Final answer: <30></think>Final answer: 31', '#### 30', False),
])
def test_qwen_literal_numeric_format_regressions(response, reference, correct):
    assert exact_response_matches(response, reference, numeric=True) is correct


@pytest.mark.parametrize('response,reference,correct', [
    ('D. All of the above\n\nExplanation: A, B, and C each describe a clustering algorithm.', 'D', True),
    ('A\n\nExplanation: The other options (B, C, and D) are consequences of tourism.', 'A', True),
    ('C. carving\n\nHowever, carving is not a recognized penalty in ice hockey.', 'C', True),
    ('D. All of the above\n\nExplanation: A, B, and C all apply.', 'A', False),
    ('A. alpha\n\nFinal answer: B', 'A', False),
    ('A. alpha\n\nFinal answer: B', 'B', True),
    ('A. alpha\n\nThe correct answer is B.', 'A', False),
    ('A. alpha\n\nExplanation: the correct answer is B.', 'A', False),
    ('A. alpha\n\nFinal answer: A or B', 'A', False),
    ('A. alpha\nB. beta\nC. carving\nD. All of the above', 'A', False),
    ('A. alpha\n\nB. beta', 'A', False),
    ('<think>A. alpha\n\nExplanation: It seems likely.', 'A', False),
    ('<think>A. alpha</think>B. beta\n\nExplanation: B is correct.', 'A', False),
    ('I considered A, B and C, and am not sure.', 'A', False),
])
def test_smol_answer_first_explanations_remain_unambiguous(response, reference, correct):
    assert exact_response_matches(response, reference, ['alpha', 'beta', 'carving', 'All of the above']) is correct


@pytest.mark.parametrize('response', [
    '(A)wrong', '(A) wrong', 'A. is wrong', 'A. not correct',
    'A. maybe', 'A. uncertain', 'A. alpha is incorrect',
    'A. alpha (incorrect)', 'Final answer: A. is wrong',
])
def test_negated_or_uncertain_choice_is_not_accepted(response):
    from vivasecuris.aiasylum.benchmarks.simple import choice_answer
    assert choice_answer(response, ['alpha', 'beta']) is None
    assert not exact_response_matches(response, 'A', ['alpha', 'beta'])


@pytest.mark.parametrize('choice', ['not correct', 'incorrect', 'false', 'not having a job'])
def test_exact_negative_choice_text_is_valid(choice):
    assert exact_response_matches(f'A. {choice}', 'A', [choice, 'another option'])


@pytest.mark.parametrize('response,reference', [
    ('The correct answer is:  \nAnswer: B', 'B'),
    ('The correct answer is:\n\nAnswer: C', 'C'),
    ('Calculation above.\n\n### Final answer: 72', '72'),
    ('Calculation above.\n\n### Final answer: 4', '4'),
    ('Therefore, the answer is: Final answer: 2', '2'),
    ('Final answer: 14, 000', '14000'),
    ('Final answer: $1, 000, 000.50', '1000000.50'),
])
def test_complete_source_audit_presentation_regressions(response, reference):
    if reference in ('B', 'C'):
        assert exact_response_matches(response, reference, ['alpha', 'beta', 'gamma'])
    else:
        assert exact_response_matches(response, f'#### {reference}', numeric=True)


@pytest.mark.parametrize('response', [
    'Final answer: 14,00', 'Final answer: 14, 00',
    'Final answer: 14, 000+1', 'Final answer: 14, 000 or 15, 000',
    'Final answer: 14, 000, 00', 'Final answer: 14, 000 = 14001',
])
def test_thousands_formatting_never_turns_expressions_into_numbers(response):
    assert not exact_response_matches(response, '#### 14000', numeric=True)


@pytest.mark.parametrize('response,reference', [
    (r'Final answer: \$10', '10'),
    (r'Final answer: -$2', '-2'),
    (r'Final answer: -\$100', '-100'),
    (r'Final answer: \(-53.55\)', '-53.55'),
    (r'Final answer: \[10\]', '10'),
    (r'Final answer: $10$', '10'),
    (r'Final answer: $-2$', '-2'),
])
def test_currency_and_whole_numeric_math_wrappers(response, reference):
    assert exact_response_matches(response, f'#### {reference}', numeric=True)


@pytest.mark.parametrize('response,reference', [
    (r'Final answer: -$100', '100'),
    (r'Final answer: \(10+1\)', '11'),
    (r'Final answer: \[10 or 11\]', '10'),
    (r'Final answer: $10=11$', '11'),
    (r'Final answer: \(10\) or 11', '10'),
])
def test_math_wrappers_preserve_sign_and_reject_equations(response, reference):
    assert not exact_response_matches(response, f'#### {reference}', numeric=True)
