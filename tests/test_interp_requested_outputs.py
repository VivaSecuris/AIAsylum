"""Numerical projections and complete requested outputs, using real tiny-model forwards."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from tests.test_interp_compatibility import build_model, tokenizer, small_cpu_workload
from vivasecuris.aiasylum.interp.core.config import Config
from vivasecuris.aiasylum.interp.core.orchestrator import AnalysisOrchestrator
from vivasecuris.aiasylum.interp.core.requirements import validate_result
from vivasecuris.aiasylum.interp.analysis.tsne import TSNEAnalyzer
from vivasecuris.aiasylum.interp.analysis.umap import UMAPAnalyzer


def configuration(tmp_path, **overrides):
    values = dict(model='offline-test', out_dir=tmp_path, device='cpu', dtype='float32',
                  max_len=24, window=6, topk=3, pca_layers='0,2', tsne_n_iter=300,
                  enable_component_analysis=False, enable_attention_capture=False,
                  enable_mlp_capture=False, enable_attn_output_capture=False,
                  prompt='the cat sat on a mat', prompt_a='the cat sat on a mat',
                  prompt_b='the dog sat on a rug',
                  prompts=['the cat sat on a mat', 'the dog sat on a rug'], query_marker='sat')
    values.update(overrides)
    return Config(**values)


@pytest.mark.parametrize('method', ['pca', 'tsne', 'umap'])
@pytest.mark.parametrize('mode', ['single', 'comparison', 'progression'])
def test_requested_projection_runs_real_algorithm_and_serializes(method, mode, tokenizer, tmp_path):
    if method == 'umap':
        pytest.importorskip('umap')
    cfg = configuration(tmp_path, analysis_mode=mode, dim_reduction=method)
    result, html = AnalysisOrchestrator.run_and_save(cfg, build_model(), tokenizer)
    assert result.meta['dim_reduction'] == method
    assert (tmp_path / 'pca_payload.json').exists()
    for payload in result.pca_payload.values():
        if mode == 'progression':
            coordinates = [item['pca'] for item in payload.values()]
        else:
            coordinates = [payload['trajectory']] if mode == 'single' else [payload['trajectory_a'], payload['trajectory_b']]
            assert f'{method}_params' in payload
            if method == 'tsne':
                assert payload['tsne_params']['max_iter'] == 300
        for points in coordinates:
            matrix = np.asarray(points)
            assert matrix.shape[1] == 3
            assert np.isfinite(matrix).all()
            assert np.linalg.norm(matrix) > 0
    assert result.predictions_payload
    assert 'trajectory projection' in result.meta['validated_outputs']


@pytest.mark.parametrize('mode', ['single', 'comparison', 'progression', 'model_diff'])
def test_projection_failure_never_falls_back_to_mislabeled_pca(mode, tokenizer, tmp_path):
    from vivasecuris.aiasylum.interp.core.services.model_comparison_service import ModelComparisonService
    cfg = configuration(tmp_path, analysis_mode=mode, dim_reduction='tsne')
    with patch.object(TSNEAnalyzer, 'fit_embedding', side_effect=RuntimeError('sentinel t-SNE failure')):
        with pytest.raises(RuntimeError, match='TSNE projection failed.*sentinel'):
            if mode == 'model_diff':
                ModelComparisonService.run_model_diff(cfg, lambda: (build_model(), tokenizer), lambda: (build_model(), tokenizer), 'a', 'b')
            else:
                AnalysisOrchestrator.run_and_save(cfg, build_model(), tokenizer)
    assert not (tmp_path / 'dashboard.html').exists()


def test_small_windows_are_handled_explicitly():
    points, _ = TSNEAnalyzer.fit_embedding(np.array([[0., 1., 2.], [3., 4., 5.]]))
    assert points.shape == (2, 3) and np.isfinite(points).all()
    with pytest.raises(ValueError, match='at least two'):
        TSNEAnalyzer.fit_embedding(np.ones((1, 5)))
    with pytest.raises(ValueError, match='at least three'):
        UMAPAnalyzer.fit_embedding(np.ones((2, 5)))


@pytest.mark.parametrize('flag', ['enable_qkv_capture', 'enable_pre_mlp_capture', 'enable_minimal_circuit'])
def test_independent_capture_or_search_option_produces_evidence(flag, tokenizer, tmp_path):
    cfg = configuration(tmp_path, **{flag: True})
    result, _ = AnalysisOrchestrator.run_and_save(cfg, build_model(), tokenizer)
    if flag == 'enable_qkv_capture':
        assert result.ov_qk_payload
        assert all(value['ov_available'] and value['qk_available'] for value in result.ov_qk_payload.values())
    elif flag == 'enable_pre_mlp_capture':
        assert all(value['activation_space'] == 'mlp_neurons' for value in result.mlp_payload.values())
        assert all(value['num_neurons'] == 64 for value in result.mlp_payload.values())
    else:
        assert result.minimal_circuit_payload['num_candidates'] > 0
        assert result.minimal_circuit_payload['claim'] == 'descriptive'


def test_progression_captures_are_saved_and_rendered(tokenizer, tmp_path):
    cfg = configuration(tmp_path, analysis_mode='progression', enable_attention_capture=True, enable_mlp_capture=True)
    result, html = AnalysisOrchestrator.run_and_save(cfg, build_model(), tokenizer)
    for name in ('attention_payload', 'mlp_payload', 'predictions'):
        assert (tmp_path / f'{name}.json').exists()
    for index, label in enumerate(result.prompt_labels):
        for block, payload in result.attention_payload[label].items():
            start = result.alignment_info[index]['start']
            end = start + result.query_window_len
            expected = result.run_results[index].attention_weights[int(block)][0, :, start:end, start:end].mean(0)
            np.testing.assert_allclose(payload['mean_over_heads'], expected.numpy())
        assert set(result.mlp_payload[label]) == set(result.attention_payload[label])
    assert 'attention-progression-plot' in html and 'mlp-progression-plot' in html
    assert 'prediction-progression-plot' in html


@pytest.mark.parametrize('component,selection', [('layer', {'patch_layers': [0, 2]}), ('head', {'patch_heads': [(1, 2)]}), ('neuron', {'patch_neurons': [(1, 3)]})])
def test_explicit_patching_options_drive_actual_forwards(component, selection, tokenizer, tmp_path):
    cfg = configuration(tmp_path, enable_patching=True, patch_components=component,
                        patch_positions=[0, 1], **selection)
    result, _ = AnalysisOrchestrator.run_and_save(cfg, build_model(), tokenizer)
    patching = result.patching_results
    assert patching['positions'] == [0, 1]
    assert patching['component'] == component
    assert patching['layers'] == ([0, 2] if component == 'layer' else [1])
    assert all(e['component'] == component and len(e['results']) == 2 for e in patching['experiments'])
    assert patching['forward_passes'] == (10 if component == 'layer' else 4)
    assert {e['direction'] for e in patching['experiments']} == {'A_to_B', 'B_to_A'}


@pytest.mark.parametrize('extra', [dict(patch_positions=[6]), dict(patch_layers=[3]), dict(patch_components='head', patch_heads=[(1, 4)]), dict(patch_components='neuron', patch_neurons=[(1, 64)])])
def test_invalid_model_or_actual_window_indices_fail(extra, tokenizer, tmp_path):
    cfg = configuration(tmp_path, enable_patching=True, **extra)
    with pytest.raises(RuntimeError, match='Activation patching failed'):
        AnalysisOrchestrator.run_and_save(cfg, build_model(), tokenizer)


def test_requested_patching_exception_fails_run(tokenizer, tmp_path):
    with patch('vivasecuris.aiasylum.interp.core.services.comparison_service.run_patching_experiments', side_effect=ValueError('broken patch')):
        with pytest.raises(RuntimeError, match='Activation patching failed: broken patch'):
            AnalysisOrchestrator.run_and_save(configuration(tmp_path, enable_patching=True), build_model(), tokenizer)
    assert not (tmp_path / 'dashboard.html').exists()


def test_requested_output_validator_rejects_unavailable_payloads(tmp_path):
    cfg = configuration(tmp_path, enable_patching=True)
    result = SimpleNamespace(meta={}, pca_payload={'0': {}}, predictions_payload={'predictions': [1]}, patching_results={'enabled': False, 'reason': 'unsupported'})
    with pytest.raises(RuntimeError, match='unsupported'):
        validate_result(result, cfg)


@pytest.mark.parametrize('family', ['mixtral', 'qwen2_moe'])
def test_unavailable_moe_neuron_capture_fails_explicitly(family, tokenizer, tmp_path):
    with pytest.raises(ValueError, match='enable_pre_mlp_capture could not capture'):
        AnalysisOrchestrator.run_and_save(configuration(tmp_path, enable_pre_mlp_capture=True), build_model(family), tokenizer)


def test_api_cannot_mark_mocked_partial_result_complete(tmp_path, monkeypatch):
    from vivasecuris.aiasylum.api.routes import interp
    from vivasecuris.aiasylum.models import transformers_local
    cfg = configuration(tmp_path, enable_patching=True)
    result = SimpleNamespace(meta={}, pca_payload={'0': {}}, predictions_payload={'predictions': [1]}, patching_results=None)
    monkeypatch.setattr(interp, 'check_request', lambda _: {'ready': True, 'device': 'cpu'})
    monkeypatch.setattr(transformers_local, 'clear_cache', lambda: None)
    monkeypatch.setattr(AnalysisOrchestrator, 'run_and_save', lambda _: (result, '<html>'))
    with pytest.raises(RuntimeError, match='activation patching produced no result'):
        interp._execute_analysis(1, {'mode': 'comparison', 'config': cfg, 'model_a': 'offline', 'progress': lambda _: None}, tmp_path)


def test_patch_validation_preflight_rejects_model_boundaries():
    from tests.test_interp_preflight import hardware, facts
    from vivasecuris.aiasylum.api import interp_preflight
    with patch.object(interp_preflight, 'hardware_info', return_value=hardware()), patch.object(interp_preflight, 'model_info', return_value=facts()):
        result = interp_preflight.check_request(dict(mode='comparison', model_a='offline', enable_patching=True, patch_components='head', patch_heads=[(32, 0), (0, 32)]))
    assert not result['ready']
    assert len([error for error in result['errors'] if 'patch_heads pair' in error]) == 2


@pytest.mark.parametrize('extra', [dict(patch_positions=[-1]), dict(patch_positions=[1, 1]), dict(patch_heads=[(0, 1)]), dict(patch_components='head', patch_heads=[(1, 1)], patch_layers=[0]), dict(patch_positions=[128])])
def test_patch_request_validation_rejects_ambiguous_or_invalid_selections(extra):
    from fastapi import HTTPException
    from vivasecuris.aiasylum.api.routes.interp import InterpRunRequest, _validate
    request = InterpRunRequest(mode='comparison', model_a='offline', prompt_a='a', prompt_b='b', enable_patching=True, **extra)
    with pytest.raises(HTTPException):
        _validate(request)


@pytest.mark.parametrize('extra', [dict(patch_layers=[True]), dict(patch_positions=[1.2]), dict(patch_heads=[[0, '2']]), dict(patch_neurons=[[0, 2, 3]])])
def test_patch_request_requires_integer_shapes(extra):
    from pydantic import ValidationError
    from vivasecuris.aiasylum.api.routes.interp import InterpRunRequest
    with pytest.raises(ValidationError):
        InterpRunRequest(mode='comparison', model_a='offline', prompt_a='a', prompt_b='b', enable_patching=True, **extra)
