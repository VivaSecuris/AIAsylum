#!/usr/bin/env python3
"""Run bounded, artifact-checked real-model analyses through the server job queue.

Example: python scripts/validate_interp_features.py --model Qwen/Qwen3-0.6B
  --api http://127.0.0.1:18000/api/v1 --key-file ~/.config/aiasylum/aiasylum-gpu.key
  --out runs/interp-feature-validation/qwen3-06b.json

Does not modify or save model weights. CUDA time is consumed. Use --cases to
bound a larger-model check. Every API job uses the server's shared model slot.
"""
import argparse
import json
from pathlib import Path
import time
import urllib.request


def scenarios():
    prompt_a = 'The cat sat on the mat. The capital of France is'
    prompt_b = 'The dog sat on the rug. The capital of Germany is'
    base = dict(mode='comparison', prompt_a=prompt_a, prompt_b=prompt_b,
                max_len=128, window=8, topk=3, device='cuda', dtype='bfloat16')
    long_prompt = 'the ' * 2200
    return {
        'tsne': {**base, 'mode': 'single', 'dim_reduction': 'tsne'},
        'umap': {**base, 'dim_reduction': 'umap'},
        'progression': {**base, 'mode': 'progression', 'prompts': [prompt_a, 'Paris is in France. ' + prompt_a, 'London is in England. Paris is in France. ' + prompt_a], 'enable_attention_capture': True, 'enable_mlp_capture': True},
        'qkv': {**base, 'enable_qkv_capture': True},
        'pre_mlp': {**base, 'enable_pre_mlp_capture': True},
        'layer_patch': {**base, 'enable_patching': True, 'patch_layers': [0, 1], 'patch_positions': [0, 1]},
        'head_patch': {**base, 'enable_patching': True, 'patch_components': 'head', 'patch_heads': [[0, 0]], 'patch_positions': [0, 1]},
        'neuron_patch': {**base, 'enable_patching': True, 'patch_components': 'neuron', 'patch_neurons': [[0, 0]], 'patch_positions': [0, 1]},
        'circuit': {**base, 'window': 4, 'enable_minimal_circuit': True},
        'long_512': {**base, 'mode': 'single', 'prompt_a': long_prompt, 'max_len': 512, 'window': 8, 'enable_attention_capture': True, 'enable_mlp_capture': True},
        'long_1024': {**base, 'mode': 'single', 'prompt_a': long_prompt, 'max_len': 1024, 'window': 8, 'enable_attention_capture': True, 'enable_mlp_capture': True},
        'fp16': {**base, 'mode': 'single', 'dtype': 'float16', 'enable_attention_capture': True, 'enable_mlp_capture': True},
        'fp32': {**base, 'mode': 'single', 'dtype': 'float32', 'enable_attention_capture': True, 'enable_mlp_capture': True},
        'cancel': {**base, 'prompt_a': long_prompt, 'prompt_b': long_prompt.replace('the ', 'a '), 'max_len': 512, 'window': 8, 'enable_patching': True, 'patch_positions': list(range(8))},
        'self_control': {**base, 'prompt_b': prompt_a, 'enable_patching': True, 'patch_layers': [0, 1]},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--api', default='http://127.0.0.1:18000/api/v1')
    parser.add_argument('--key-file', type=Path, required=True)
    parser.add_argument('--cases', default='tsne,umap,progression,qkv,pre_mlp,layer_patch,head_patch,neuron_patch,circuit,self_control')
    parser.add_argument('--timeout', type=int, default=1800, help='Per-job seconds; timeout does not cancel a running server job')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    key = args.key_file.expanduser().read_text().strip()

    def call(path, payload=None):
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(args.api.rstrip('/') + path, data=data,
                                     headers={'X-API-Key': key, 'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=120) as response:
            return json.load(response)

    report = {'model': args.model, 'cases': [], 'passed': False}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cases = scenarios()
    selected = args.cases.split(',')
    unknown = set(selected) - cases.keys()
    if unknown:
        parser.error(f'Unknown cases: {sorted(unknown)}')
    for name in selected:
        entry = {'case': name, 'request': {**cases[name], 'model_a': args.model}}
        report['cases'].append(entry)
        try:
            fit = call('/interp/preflight', entry['request'])
            if not fit['ready']:
                raise RuntimeError('Preflight refused: ' + '; '.join(fit['errors']))
            created = call('/interp/runs', entry['request'])
            entry['run_id'] = created['id']
            deadline = time.monotonic() + args.timeout
            stop_sent = False
            while True:
                run = call(f"/interp/runs/{created['id']}")
                if name == 'cancel' and run['status'] == 'running' and not stop_sent:
                    time.sleep(1)
                    stopped = call(f"/interp/runs/{created['id']}/stop", {})
                    assert stopped['stopped'], 'Run completed before active cancellation; rerun with a larger model'
                    entry['cancellation_observed_state'] = 'running'
                    stop_sent = True
                if run['status'] in ('failed', 'completed'):
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError('Job remains on the server; inspect it before starting another validation')
                time.sleep(3)
            if name == 'cancel':
                assert stop_sent and run['status'] == 'failed' and run['metadata']['cancelled'], 'Active cancellation was not recorded'
                entry['passed'] = True
                args.out.write_text(json.dumps(report, indent=2))
                print(json.dumps({'case': name, 'run_id': created['id'], 'passed': True}), flush=True)
                continue
            if run['status'] != 'completed':
                raise RuntimeError(run.get('error') or 'Analysis failed')
            summary = run['metadata']['summary']
            assert summary['validated_outputs'], 'Server did not validate requested outputs'
            entry['summary'] = summary
            meta = call(f"/interp/runs/{created['id']}/artifacts/meta.json")
            # Artifacts endpoint returns the JSON content directly.
            if name.startswith('long_'):
                assert min(summary['input_token_counts']) == entry['request']['max_len'], 'Actual input was shorter than claimed context length'
            if 'patch' in name or name == 'self_control':
                patching = call(f"/interp/runs/{created['id']}/artifacts/patching_results.json")
                assert patching['enabled'] and patching['experiments']
                if name == 'self_control':
                    for experiment in patching['experiments']:
                        for measured in experiment['results']:
                            # A self patch must not flip the original next-token prediction.
                            assert not measured['top_flipped']
                entry['intervention_count'] = sum(len(item['results']) for item in patching['experiments'])
            entry['passed'] = True
        except Exception as exc:
            entry.update(passed=False, error=str(exc))
        args.out.write_text(json.dumps(report, indent=2))
        print(json.dumps({key: entry[key] for key in ('case', 'run_id', 'passed', 'error') if key in entry}), flush=True)
        if not entry['passed']:
            break
    report['passed'] = len(report['cases']) == len(selected) and all(entry['passed'] for entry in report['cases'])
    args.out.write_text(json.dumps(report, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
