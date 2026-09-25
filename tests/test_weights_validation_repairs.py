"""Scientific-data and option regressions from the September feature audit."""
import asyncio
from contextlib import nullcontext
import json
from types import SimpleNamespace
from threading import Event

import pytest

from vivasecuris.aiasylum.api.routes import weights as routes
from vivasecuris.aiasylum.weights.corpus import PromptSplit, build_split


def sample_split():
    return build_split(n_per_class=8, seed=17, harmful=[f'positive {i}' for i in range(40)],
                       harmless=[f'negative {i}' for i in range(40)])


def test_saved_partition_survives_child_count_seed_and_library_changes(monkeypatch):
    split = sample_split()
    monkeypatch.setattr(routes, '_build_objective_split', lambda *a, **k: pytest.fail('must not resample'))
    snap = {'source_direction': {'prompt_split': split.to_dict()},
            'options': {'n_per_class': 128, 'seed': 42}}
    prompts = routes._held_out_prompts(snap, 8)
    assert prompts == split.harmful_test and len(prompts) == 2
    assert not set(prompts) & set(split.harmful_train)
    assert routes._evaluation_evidence(snap, 8)['actual'] == 2
    assert routes._evaluation_evidence(snap, 8)['warnings']


def test_legacy_partition_uses_parent_options_and_requires_digest(monkeypatch):
    split = sample_split()
    def rebuild(*args, **kwargs):
        assert kwargs['n_per_class'] == 8 and kwargs['seed'] == 17
        return split
    monkeypatch.setattr(routes, '_build_objective_split', rebuild)
    snap = {'source_direction': {'split_hash': split.hash},
            'source_options': {'n_per_class': 8, 'seed': 17},
            'options': {'n_per_class': 128, 'seed': 42}}
    assert routes._held_out_prompts(snap, 8) == split.harmful_test
    snap['source_direction']['split_hash'] = 'changed-corpus'
    with pytest.raises(ValueError, match='exact prompt split'):
        routes._held_out_prompts(snap, 8)


def test_saved_partition_integrity_and_cross_class_overlap():
    split = sample_split()
    data = split.to_dict()
    data['harmful_test'] = ['changed']
    with pytest.raises(ValueError, match='integrity'):
        PromptSplit.from_dict(data)
    with pytest.raises(ValueError, match='overlap'):
        build_split(harmful=['shared'] + [f'h{i}' for i in range(9)],
                    harmless=['shared'] + [f'b{i}' for i in range(9)])


def test_capability_answers_are_final_whole_phrases():
    from vivasecuris.aiasylum.weights.evaluate import factual_accuracy
    gold = [('Gold symbol?', ('au',))]
    assert factual_accuracy(['because'], gold) == 0
    assert factual_accuracy(['Au'], gold) == 1
    assert factual_accuracy(['The chemical symbol is Au.'], gold) == 1
    assert factual_accuracy(['It is not Au.'], gold) == 0
    assert factual_accuracy(['<think>Au</think>Ag'], gold) == 0
    assert factual_accuracy(['680'], [('17*4?', ('68',))]) == 0


def test_capability_mcq_does_not_select_rejected_first_mention():
    from vivasecuris.aiasylum.weights.evaluate import mmlu_capability_set
    control = mmlu_capability_set([{'question':'q', 'choices':['first','second','third','fourth'], 'answer':'A'}])
    assert control.score(['A is wrong; the answer is B.']) == 0
    assert control.score(['<think>B</think>Final answer: A']) == 1
    assert control.score(['A or B']) == 0
    correct_b = mmlu_capability_set([{'question':'q', 'choices':['first','second'], 'answer':'B'}])
    assert correct_b.score(['A is wrong. The answer is B.']) == 1


@pytest.fixture
def probe_sources(monkeypatch):
    from vivasecuris.aiasylum.interp.probes import dataset
    monkeypatch.setattr('vivasecuris.aiasylum.weights.corpus.load_harmful_prompts',
                        lambda: [f'direct {i}' for i in range(80)])
    monkeypatch.setattr(dataset, '_jailbreak_rows', lambda: [])
    return dataset


def test_probe_requested_missing_families_fail(probe_sources):
    with pytest.raises(ValueError, match='no usable jailbreak families'):
        probe_sources.build_harmful_intent_dataset()


def test_probe_direct_only_is_explicit_and_reports_sizes(probe_sources):
    ds = probe_sources.build_harmful_intent_dataset(n_direct=80, n_jailbreak=0, holdout_techniques=0)
    assert not ds.held_out_techniques
    assert ds.actual['jailbreak_train_pool'] == 0
    assert ds.actual['benign'] == 128
    assert ds.warnings and ds.requested['benign'] == 240
    assert not set(ds.train_prompts) & set(ds.test_prompts)


def test_probe_custom_families_are_disjoint_and_all_holdouts_measured(probe_sources):
    examples = [{'prompt':f'wrapped request {family} {i}', 'technique':family}
                for family in ['one', 'two', 'three', 'four'] for i in range(24)]
    ds = probe_sources.build_harmful_intent_dataset(n_direct=40, n_jailbreak=40, n_benign=64,
                                                   holdout_techniques=2, jailbreak_examples=examples)
    assert len(ds.held_out_techniques) == 2
    assert set(f'heldout:{t}' for t in ds.held_out_techniques) <= set(ds.test_groups)
    assert not set(ds.train_prompts) & set(ds.test_prompts)
    for family in ds.held_out_techniques:
        assert all(f'wrapped request {family} ' not in p for p in ds.train_prompts)
    with pytest.raises(ValueError, match='only 4 are available'):
        probe_sources.build_harmful_intent_dataset(holdout_techniques=4, jailbreak_examples=examples)


def test_probe_loader_honors_importers_family_key():
    from vivasecuris.aiasylum.database import PromptLibrary, get_session
    from vivasecuris.aiasylum.interp.probes.dataset import _jailbreak_rows
    session = get_session()
    try:
        session.add(PromptLibrary(name='imported family', prompt_text='example text', category='adversarial',
                                  meta_data={'jailbreak_technique':'roleplay', 'source':'same-corpus'}))
        session.commit()
    finally:
        session.close()
    assert _jailbreak_rows() == [('example text', 'roleplay')]


def test_dtype_and_cached_index_size_estimates(monkeypatch, tmp_path):
    import huggingface_hub
    from vivasecuris.aiasylum.weights.resources import estimated_weights_gb
    (tmp_path / 'config.json').write_text(json.dumps({'torch_dtype':'bfloat16'}))
    (tmp_path / 'model.safetensors.index.json').write_text(json.dumps({'metadata':{'total_size':64 * 2**30}}))
    monkeypatch.setattr(huggingface_hub, 'try_to_load_from_cache', lambda *a: str(tmp_path / 'config.json'))
    assert estimated_weights_gb('org/remote', 'bfloat16') == 64
    assert estimated_weights_gb('org/remote', 'float32') == 128
    monkeypatch.setattr(huggingface_hub, 'try_to_load_from_cache', lambda *a: None)
    assert estimated_weights_gb('Qwen/Qwen3-32B', 'float32') > 119


def test_dim_subspace_honors_capture_limit_and_saves_exact_partition(monkeypatch, tmp_path):
    from vivasecuris.aiasylum.weights import direction
    split = sample_split()
    monkeypatch.setattr(routes, '_build_objective_split', lambda *a, **k: split)
    monkeypatch.setattr('vivasecuris.aiasylum.interp.core.loader.load', lambda *a, **k: (None,None))
    monkeypatch.setattr('vivasecuris.aiasylum.models.transformers_local.clear_cache', lambda: None)
    def derive(model, tok, actual_split, **kwargs):
        assert kwargs['max_length'] == 37
        assert actual_split is split and kwargs['rank'] == 2
        return SimpleNamespace(save=lambda path: None, metadata=lambda: {'split_hash': split.hash})
    monkeypatch.setattr(direction, 'derive_subspace', derive)
    reporter = SimpleNamespace(note=lambda *a:None, step=lambda *a:nullcontext(),
                               total_elapsed=lambda:0, as_callback=lambda:None)
    summary = routes._execute(1, {'kind':'direction','source_model':'m','objective':'refusal',
                                  'out_dir':str(tmp_path), 'options':{'n_per_class':8,'test_fraction':.25,
                                   'seed':0,'device':'cpu','dtype':'float32','subspace_rank':2,
                                   'batch_size':1,'max_length':37}}, reporter)
    assert PromptSplit.from_dict(summary['prompt_split']).hash == split.hash
    assert json.loads((tmp_path / 'prompt_split.json').read_text()) == split.to_dict()


def test_chat_request_bounds():
    from pydantic import ValidationError
    for extra in ({'max_tokens':0}, {'max_tokens':4097}, {'temperature':-1}, {'temperature':float('nan')},
                  {'dtype':'int4'}, {'messages':[{'role':'system','content':'x'}]}):
        with pytest.raises(ValidationError):
            routes.ChatRequest.model_validate({'messages':[{'role':'user','content':'hello'}], **extra})


@pytest.mark.asyncio
async def test_chat_cancellation_keeps_slot_until_generation_exits(monkeypatch, tmp_path):
    from vivasecuris.aiasylum.api import model_jobs
    from vivasecuris.aiasylum.models.base import ModelResponse
    started, finish = Event(), Event()
    path = tmp_path / 'chat-model'; path.mkdir()
    monkeypatch.setattr('vivasecuris.aiasylum.api.model_catalog.checkpoint_status', lambda path: {'availability':'ready'})
    monkeypatch.setattr(routes, '_models_root', lambda: tmp_path)
    monkeypatch.setattr(routes, '_interp_extra_installed', lambda: True)
    monkeypatch.setattr(model_jobs, '_slot', asyncio.Semaphore(1))
    monkeypatch.setattr(model_jobs, 'try_process_lock', lambda: SimpleNamespace(close=lambda:None))
    async def generate(**kwargs):
        started.set()
        if not finish.wait(timeout=5):
            raise RuntimeError('test failed to release generation')
        return ModelResponse(content='done', model='test', provider='transformers', finish_reason='stop')
    provider = SimpleNamespace(create_model=lambda *a, **k: SimpleNamespace(generate=generate))
    monkeypatch.setattr('vivasecuris.aiasylum.models.get_provider', lambda *a: provider)
    task = asyncio.create_task(routes.chat_with_model('chat-model', routes.ChatRequest(messages=[routes.ChatMessage(role='user', content='hello')])))
    try:
        assert await asyncio.to_thread(started.wait, 3)
        task.cancel()
        await asyncio.sleep(0.01)
        assert not task.done()
        assert model_jobs.slot_status()['held_by'] == 'chat with chat-model'
    finally:
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert model_jobs.slot_status()['held_by'] is None
