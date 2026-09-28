"""Branch forms restore saved interpretation inputs without inheriting prior runs."""
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_interp_branch_snapshot_restores_options_and_multiline_prompts():
    node = shutil.which("node")
    typescript = ROOT / "frontend/node_modules/typescript"
    if not node or not typescript.exists():
        pytest.skip("Node and frontend TypeScript dependencies are needed")
    script = r"""
const fs = require('node:fs');
const assert = require('node:assert/strict');
const ts = require(process.argv[1]);
const source = fs.readFileSync(process.argv[2], 'utf8');
const code = ts.transpileModule(source, { compilerOptions: {
  module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020,
}}).outputText;
const loaded = { exports: {} };
new Function('module', 'exports', code)(loaded, loaded.exports);
const { parseInterpPrefill } = loaded.exports;
const prompts = ['First line\nSecond line', 'Another\nmultiline prompt'];
const restored = parseInterpPrefill({ mode: 'comparison', model_a: 'original', model_b: 'edited',
  prompt: 'Reference\nquestion', prompt_b: 'Changed question', prompts: JSON.stringify(prompts),
  options: JSON.stringify({ max_len: 768, window: 300, topk: 24, dim_reduction: 'umap',
    device: 'cuda:1', dtype: 'fp32', enable_attention_capture: true, enable_mlp_capture: false,
    enable_patching: true, patch_components: 'neuron', patch_layers: [1, 3], patch_positions: [0, 4], patch_neurons: [[1, 2], [3, 7]], output_name: 'must-not-leak' }),
});
assert.equal(restored.mode, 'comparison');
assert.equal(restored.modelA, 'original');
assert.equal(restored.modelB, 'edited');
assert.equal(restored.promptA, 'Reference\nquestion');
assert.equal(restored.promptB, 'Changed question');
assert.deepEqual(restored.prompts, prompts);
assert.equal(restored.maxLen, 768);
assert.equal(restored.window, 300);
assert.equal(restored.topk, 24);
assert.equal(restored.dimReduction, 'umap');
assert.equal(restored.device, 'cuda:1');
assert.equal(restored.dtype, 'float32');
assert.equal(restored.advanced.enable_attention_capture, true);
assert.equal(restored.advanced.enable_mlp_capture, false);
assert.equal(restored.advanced.enable_patching, true);
assert.deepEqual(restored.patch, {component: 'neuron', layers: '1, 3', positions: '0, 4', units: '1:2, 3:7'});
assert.equal('output_name' in restored, false);
// A different branch begins from defaults, with no values retained from above.
const next = parseInterpPrefill({ mode: 'single', model_a: 'next' });
assert.equal(next.modelB, '');
assert.equal(next.promptA, '');
assert.equal(next.promptB, '');
assert.deepEqual(next.prompts, []);
assert.equal(next.maxLen, 512);
assert.equal(next.window, 128);
assert.equal(next.topk, 10);
assert.equal(next.dimReduction, 'pca');
assert.deepEqual(next.patch, {component: 'layer', layers: '', positions: '', units: ''});
assert.equal(next.device, 'auto');
assert.equal(next.dtype, 'bfloat16');
assert(Object.values(next.advanced).every(value => value === false));
// Malformed snapshots are visibly reported, and string booleans cannot enable work.
assert(parseInterpPrefill({ options: '{bad' }).warnings.length > 0);
const malformed = parseInterpPrefill({ options: JSON.stringify({ max_len: '1024', dtype: 'bad', device: 'gpu', enable_patching: 'false' }) });
assert(malformed.warnings.length >= 3);
assert.equal(malformed.advanced.enable_patching, false);
const invalidPatch = parseInterpPrefill({ options: JSON.stringify({ patch_components: 'head', patch_heads: [[1, 'bad']], patch_positions: [1.5] }) });
assert(invalidPatch.warnings.length === 2);
assert.equal(invalidPatch.patch.units, '');
assert.equal(parseInterpPrefill({ prompt: 'alias', prompt_a: 'explicit' }).promptA, 'explicit');
assert.equal(parseInterpPrefill({}, '/models/saved-default').modelA, '/models/saved-default');
assert.equal(parseInterpPrefill({ model_a: '/models/selected' }, '/models/saved-default').modelA, '/models/selected');
assert.equal(parseInterpPrefill({ options: JSON.stringify({ model_a: '/models/restored' }) }, '/models/saved-default').modelA, '/models/restored');
"""
    subprocess.run(
        [node, "-e", script, str(typescript), str(ROOT / "frontend/lib/interp-prefill.ts")],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )


def test_patch_form_parser_preserves_pairs_and_rejects_ambiguous_fields():
    node = shutil.which('node')
    typescript = ROOT / 'frontend/node_modules/typescript'
    if not node or not typescript.exists():
        pytest.skip('Node and TypeScript are needed')
    script = r'''
const fs = require('node:fs'), assert = require('node:assert/strict'), ts = require(process.argv[1]);
const source = fs.readFileSync(process.argv[2], 'utf8');
const code = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020, jsx: ts.JsxEmit.React } }).outputText;
const loaded = {exports: {}};
new Function('module', 'exports', code)(loaded, loaded.exports);
const parse = loaded.exports.parsePatchOptions;
const parsed = parse({component:'head', layers:'1, 3', positions:'0, 2', units:'1:4, 3:2'});
assert.deepEqual(parsed.errors, []);
assert.deepEqual(parsed.request.patch_heads, [[1,4],[3,2]]);
assert.deepEqual(parsed.request.patch_positions, [0,2]);
assert.equal(parsed.request.patch_neurons, undefined);
assert(parse({component:'head', layers:'0', positions:'', units:'1:2'}).errors.length > 0);
for (const value of ['-1', '0.5', '0, 0', '1,a']) {
 assert(parse({component:'layer', layers:value, positions:'', units:''}).errors.length > 0);
}
assert.deepEqual(parse({component:'neuron', layers:'', positions:'', units:''}).request, {patch_components:'neuron', patch_layers:undefined, patch_positions:undefined, patch_heads:undefined, patch_neurons:undefined});
'''
    subprocess.run([node, '-e', script, str(typescript), str(ROOT / 'frontend/components/interp/PatchingOptions.tsx')], check=True, capture_output=True, text=True)
