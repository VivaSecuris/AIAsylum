"""Execute weight-form round trips and render result components with real React."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]

LOADER = r"""
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const frontend = path.join(process.cwd(), 'frontend');
const requireFrontend = require('node:module').createRequire(path.join(frontend, 'package.json'));
const ts = requireFrontend('typescript');
const cache = new Map();
const overrides = new Map();
let stubComponents = false;
const componentStubs = new Proxy({}, { get: (_, name) => name === '__esModule' ? true : String(name) });
function load(file) {
  const resolved = ['.ts', '.tsx', ''].map((ext) => file + ext).find((p) => fs.existsSync(p));
  if (!resolved) throw new Error('No module ' + file);
  if (cache.has(resolved)) return cache.get(resolved).exports;
  const mod = { exports: {} }; cache.set(resolved, mod);
  const code = ts.transpileModule(fs.readFileSync(resolved, 'utf8'), { compilerOptions: {
    module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020, jsx: ts.JsxEmit.ReactJSX,
  }}).outputText;
  const localRequire = (name) => overrides.has(name) ? overrides.get(name)
    : stubComponents && name.startsWith('@/components/') ? componentStubs
    : name.startsWith('@/') ? load(path.join(frontend, name.slice(2)))
    : name.startsWith('.') ? load(path.resolve(path.dirname(resolved), name)) : requireFrontend(name);
  new Function('module', 'exports', 'require', code)(mod, mod.exports, localRequire);
  return mod.exports;
}
"""


def run_frontend(script):
    node = shutil.which("node")
    if not node or not (ROOT / "frontend/node_modules/typescript").exists():
        pytest.skip("Node and frontend dependencies are needed")
    result = subprocess.run(
        [node, "-e", LOADER + script], cwd=ROOT, env=os.environ.copy(),
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_new_weight_kinds_restore_recorded_inputs_and_resubmit():
    run_frontend(r"""
const { restoreNewKindState, newKindRequest, NEW_KIND_DEFAULTS } = load(path.join(frontend, 'lib/weight-kind-config'));
const cases = [
  ['induce', { probe_run_id: 15, category_config: { name: 'medicine', prompts: ['target one', 'target two'], near_miss: ['allowed one'] }, ms: [0.25, 0.8], taus: [0.4, 0.7] }],
  ['hneurons', { n_questions: 720, n_samples: 15 }],
  ['hneuron_bake', { hneurons_run_id: 23, hneuron_alpha: 0 }],
  ['redteam', { redteam_target: 'prompt_leak', secret_system: 'Keep this passphrase secret.\nOther instructions.', baseline_model: 'local-baseline' }],
  ['redteam', { redteam_target: 'refusal', category_config: { name: 'custom', prompts: ['withheld one', 'withheld two'], near_miss: ['allowed'] }, baseline_model: 'ungated-model' }],
  ['embed_align', { model_b: 'second-local-model', which_embedding: 'output' }],
  ['embed_extract', { n_queries: 333, col_subset: 700 }],
  ['embed_recon', { reference_model: 'reference-local-model', n_queries: 321 }],
];
for (const [kind, options] of cases) {
  const state = restoreNewKindState(options);
  // Match real launch: unknown persisted options survive, visible fields come from state.
  const request = JSON.parse(JSON.stringify({ ...options, ...newKindRequest(kind, state) }));
  assert.deepEqual(request, options, kind + ' changed while branching');
}
const allOptions = Object.assign({}, ...cases.map(([, options]) => options));
const allState = restoreNewKindState(allOptions);
assert.equal(Object.keys(allState).length, Object.keys(NEW_KIND_DEFAULTS).length);
assert.equal(allState.probeRunId, '15'); assert.equal(allState.hneuronsRunId, '23');
assert.deepEqual(restoreNewKindState(), NEW_KIND_DEFAULTS, 'a new link must reset the preceding experiment');
assert.notEqual(restoreNewKindState(), NEW_KIND_DEFAULTS, 'defaults must not be mutated');
const align = restoreNewKindState({ model_b: 'recorded', which_embedding: 'output' });
align.modelB = ''; align.whichEmbedding = 'input';
assert.equal(newKindRequest('embed_align', align).model_b, '');
assert.equal(newKindRequest('embed_align', align).which_embedding, 'input');
const bake = restoreNewKindState({ hneurons_run_id: 23 }); bake.hneuronsRunId = '';
assert.equal(JSON.parse(JSON.stringify({ hneurons_run_id: 23, ...newKindRequest('hneuron_bake', bake) })).hneurons_run_id, undefined);
assert.deepEqual(restoreNewKindState({ n_queries: null, category_config: null, ms: null }), NEW_KIND_DEFAULTS);
""")


def test_weight_page_launch_preserves_restored_options_through_the_actual_callback():
    run_frontend(r"""
// Run the real page, including its URL restoration effects and launch handler.
// Child widgets and remote hooks are stubbed; no reconstructed launch object is used.
stubComponents = true;
const React = requireFrontend('react');
let slots = [], cursor = 0, pendingEffects = [];
overrides.set('react', {
  ...React,
  useState(initial) {
    const index = cursor++;
    if (!(index in slots)) slots[index] = typeof initial === 'function' ? initial() : initial;
    return [slots[index], (value) => { slots[index] = typeof value === 'function' ? value(slots[index]) : value; }];
  },
  useMemo(compute) { return compute(); },
  useEffect(effect, deps) {
    const index = cursor++;
    const previous = slots[index];
    if (!previous || deps.some((dep, i) => !Object.is(dep, previous[i]))) pendingEffects.push(effect);
    slots[index] = deps;
  },
});
const router = { isReady: true, query: {}, push: () => {} };
overrides.set('next/router', { useRouter: () => router });
overrides.set('next/link', { __esModule: true, default: 'Link' });
overrides.set('@radix-ui/react-tabs', componentStubs);
overrides.set('lucide-react', componentStubs);
overrides.set('@/components/weights/ExpertSelectionInput', {
  ExpertSelectionInput: 'ExpertSelectionInput', parseExpertSelection: () => ({ selection: {}, errors: [] }), formatExpertSelection: () => '',
});
const kinds = ['direction', 'probe', 'induce', 'hneurons', 'hneuron_bake', 'redteam', 'embed_align', 'embed_extract', 'embed_recon'];
const stages = { stages: kinds.map((kind) => ({ name: kind, label: kind, needs: kind === 'induce' ? ['source_run_id'] : [] })), methods: [], objectives: [], interp_extra_installed: true };
let submitted;
overrides.set('@/lib/hooks', {
  useWeightStages: () => ({ data: stages }), useWeightObjectives: () => ({}),
  useDirections: () => ({ data: [] }), useWeightRuns: () => ({ data: [] }), useEditedModels: () => ({ data: [] }),
  useWeightPreflight: () => ({ data: { checks: [] } }), useDeleteWeightRun: () => ({}),
  useCreateWeightRun: () => ({ isPending: false, mutateAsync: async (body) => { submitted = JSON.parse(JSON.stringify(body)); return { id: 999 }; } }),
});
overrides.set('@/lib/model-catalog', { useModelCatalog: () => ({}) });
overrides.set('@/lib/api', { WRITING_WEIGHT_KINDS: ['hneuron_bake'] });
overrides.set('@/lib/toast', { toast: { success: () => {}, error: (message) => { throw new Error(message); } } });
overrides.set('@/lib/utils', { formatApiError: (e) => e.message, formatDateTime: () => '' });
const Page = load(path.join(frontend, 'pages/weights/index')).default;
function settle() {
  for (let i = 0; i < 10; i++) {
    cursor = 0; pendingEffects = [];
    const tree = Page();
    if (!pendingEffects.length) return tree;
    for (const effect of pendingEffects) effect();
  }
  throw new Error('Page effects did not settle');
}
function find(node, matches) {
  if (Array.isArray(node)) return node.map((child) => find(child, matches)).find(Boolean);
  if (!node || typeof node !== 'object') return undefined;
  return matches(node) ? node : find(node.props?.children, matches);
}
const category = { name: 'medicine', prompts: Array.from({ length: 8 }, (_, i) => 'target ' + i), near_miss: ['allowed'] };
const cases = [
  ['induce', { probe_run_id: 15, category_config: category, ms: [0.25, 0.8], taus: [0.4, 0.7], n_benign: 17, n_prompts: 13, verify_sampled: false, capability_set: 'recorded-control', goal: 'category' }],
  ['hneurons', { n_questions: 720, n_samples: 15, hneuron_top_k: 777, max_answer_tokens: 31, questions: [{ question: 'Question?', aliases: ['answer'] }] }],
  ['hneuron_bake', { hneurons_run_id: 23, hneuron_alpha: 0 }],
  ['redteam', { redteam_target: 'prompt_leak', secret_system: 'Keep this passphrase secret.', baseline_model: 'local-baseline', redteam_attacks: ['persona_split'] }],
  ['redteam', { redteam_target: 'refusal', category_config: category, baseline_model: 'ungated-model' }],
  ['embed_align', { model_b: 'second-local-model', which_embedding: 'output', max_anchors: 513 }],
  ['embed_extract', { n_queries: 333, col_subset: 700 }],
  ['embed_recon', { reference_model: 'reference-local-model', n_queries: 321, max_anchors: 255, extra_cols: 777 }],
  ['probe', { category_config: category, n_benign: 17 }],
];
(async () => {
  // Reuse the same page state across links, as Next does during client navigation.
  for (const [kind, options] of cases) {
    router.query = { kind, source_model: 'local-model', source_run_id: '9', options: JSON.stringify(options), lineage_parent: 'earlier-run' };
    let tree = settle();
    const output = find(tree, (node) => node.type === 'input' && node.props.placeholder === 'qwen05b-ablated');
    if (output) { output.props.onChange({ target: { value: 'new-checkpoint' } }); tree = settle(); }
    const button = find(tree, (node) => node.type === 'button' && node.props.onClick?.name === 'launch');
    assert.ok(button && !button.props.disabled, kind + ' launch must be available');
    submitted = undefined;
    await button.props.onClick();
    assert.ok(submitted, kind + ' did not send a request');
    assert.equal(submitted.kind, kind);
    for (const [key, value] of Object.entries(options)) assert.deepEqual(submitted[key], value, kind + ' dropped or changed ' + key);
  }
  // A fresh induce link must neither reuse the previous options nor force the
  // probe/autotune form defaults over the API's induce defaults.
  router.query = { kind: 'induce', source_model: 'local-model', source_run_id: '9' };
  const tree = settle();
  await find(tree, (node) => node.type === 'button' && node.props.onClick?.name === 'launch').props.onClick();
  assert.equal(submitted.n_benign, undefined);
  assert.equal(submitted.n_prompts, undefined);
  assert.equal(submitted.verify_sampled, undefined);
})().catch((error) => { console.error(error); process.exitCode = 1; });
""")


def test_hneuron_results_require_both_baselines_and_nonempty_selection():
    run_frontend(r"""
const React = requireFrontend('react');
const { renderToStaticMarkup } = requireFrontend('react-dom/server');
const { HNeuronResults } = load(path.join(frontend, 'components/weights/HNeuronResults'));
const { ResultSummary } = load(path.join(frontend, 'components/weights/ResultSummary'));
const render = (Component, summary) => renderToStaticMarkup(React.createElement(Component, { kind: 'hneurons', summary }));
const base = { n_selected: 20, n_total: 1000, fraction: 0.02, auroc: 0.75, null_auroc_p95: 0.55, surface_auroc: 0.9, beats_null: true };
for (const summary of [base, { ...base, surface_auroc: null }, { ...base, surface_auroc: NaN },
  { ...base, surface_auroc: 0.55, n_selected: 0 }, { ...base, surface_auroc: 0.55, usable: false }]) {
  for (const Component of [HNeuronResults, ResultSummary]) {
    const html = render(Component, summary);
    assert.match(html, /[Uu]sable signal not established/);
    assert.doesNotMatch(html, /clears both baselines/);
  }
}
for (const summary of [{ ...base, surface_auroc: 0.55 }, { ...base, surface_auroc: 0.55, usable: true, beats_surface: true }]) {
  for (const Component of [HNeuronResults, ResultSummary]) assert.match(render(Component, summary), /clears both baselines/);
}
""")


def test_induce_results_distinguish_rejected_sampled_accepted_and_unverified():
    run_frontend(r"""
const React = requireFrontend('react');
const { renderToStaticMarkup } = requireFrontend('react-dom/server');
const { InduceSection } = load(path.join(frontend, 'components/weights/InduceSection'));
const { ResultSummary } = load(path.join(frontend, 'components/weights/ResultSummary'));
const winner = { m: 0.5, tau: 0.6, refuse_target: 0.9, refuse_near_miss: 0.1, factual_acc: 0.7, admissible: true };
const base = { winner, trials: [winner], baseline: {} };
const cases = [
  [{ ...base, accepted: false, reason: 'capability_cost' }, /Control rejected/],
  [{ ...base, accepted: false, verification: { accepted: false, admissible: false, reason: 'near_miss_refused' } }, /Control rejected/],
  [{ ...base, accepted: true, verification: { accepted: true, admissible: true, target_met: true } }, /Accepted and verified under sampling/],
  [{ ...base, accepted: true, verification: null }, /Accepted on the reporting split; sampling not checked/],
  [base, /Verification not established/],
  [{ ...base, verification: { refuse_target: 0.9 } }, /Verification not established/],
  [{ ...base, winner: null }, /No admissible control found/],
];
for (const [summary, expected] of cases) {
  for (const Component of [InduceSection, ResultSummary]) {
    const html = renderToStaticMarkup(React.createElement(Component, { kind: 'induce', summary }));
    assert.match(html, expected);
    if (summary.accepted !== true || summary.verification?.accepted !== true) assert.doesNotMatch(html, /[Vv]erified under sampling/);
  }
}
""")
