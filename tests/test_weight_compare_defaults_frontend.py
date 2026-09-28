"""Comparison form defaults/replay contracts through its real effects and submit."""
from tests.test_weight_frontend_regressions import run_frontend


HARNESS = r"""
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
    if (!previous || deps.some((dep, i) => !Object.is(dep, previous.deps[i]))) {
      pendingEffects.push(() => {
        previous?.cleanup?.();
        slots[index] = { deps, cleanup: effect() };
      });
    }
  },
});
const router = { isReady: true, query: {}, push() {} };
overrides.set('next/router', { useRouter: () => router });
overrides.set('next/link', { __esModule: true, default: 'Link' });
overrides.set('@radix-ui/react-tabs', componentStubs);
overrides.set('lucide-react', componentStubs);
overrides.set('@/components/weights/ExpertSelectionInput', {
  ExpertSelectionInput: 'ExpertSelectionInput', parseExpertSelection: () => ({ selection: {}, errors: [] }), formatExpertSelection: () => '',
});
const stages = { stages: [{ name: 'compare', label: 'Compare', needs: [] }], methods: [], objectives: [], interp_extra_installed: true };
let submitted, promptCalls = [];
overrides.set('@/lib/hooks', {
  useWeightStages: () => ({ data: stages }), useWeightObjectives: () => ({}),
  useDirections: () => ({ data: [] }), useWeightRuns: () => ({ data: [] }), useEditedModels: () => ({ data: [] }),
  useWeightPreflight: () => ({ data: { checks: [] } }), useDeleteWeightRun: () => ({}),
  useCreateWeightRun: () => ({ isPending: false, mutateAsync: async (body) => { submitted = JSON.parse(JSON.stringify(body)); return { id: 999 }; } }),
});
overrides.set('@/lib/model-catalog', { useModelCatalog: () => ({}) });
overrides.set('@/lib/api', { WRITING_WEIGHT_KINDS: [], apiClient: {
  getPrompt: (id) => new Promise((resolve, reject) => { promptCalls.push({ id, resolve, reject }); }),
} });
overrides.set('@/lib/toast', { toast: { success() {}, error(message) { throw new Error(message); } } });
overrides.set('@/lib/utils', { formatApiError: (error) => error.message, formatDateTime: () => '' });
const { normalizeSettings } = load(path.join(frontend, 'lib/settings'));
let settings = normalizeSettings({
  defaultPatientProvider: 'transformers', defaultPatientModel: 'saved-patient',
  defaultPatientSystemPromptText: '  Exact saved persona.\n',
  defaultPatientGeneration: { temperature: '0.6', top_p: '0.7', max_tokens: '777', enable_cot: true },
});
global.window = {};
global.localStorage = { getItem: () => JSON.stringify(settings) };
const Page = load(path.join(frontend, 'pages/weights/index')).default;
function settle() {
  for (let i = 0; i < 12; i++) {
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
function launchButton(tree) { return find(tree, (node) => node.type === 'button' && node.props.onClick?.name === 'launch'); }
async function submit(tree) {
  const button = launchButton(tree);
  assert.ok(button && !button.props.disabled, 'Comparison should be available');
  submitted = undefined; await button.props.onClick(); assert.ok(submitted);
  return submitted;
}
const flush = () => new Promise((resolve) => setImmediate(resolve));
"""


def test_weight_comparison_real_submit_uses_defaults_only_for_fresh_runs():
    run_frontend(HARNESS + r"""
(async () => {
  router.query = { kind: 'compare', modified_model: 'modified' };
  let request = await submit(settle());
  assert.equal(request.source_model, 'saved-patient');
  assert.equal(request.enable_cot, true); assert.equal(request.temperature, 0.6);
  assert.equal(request.top_p, 0.7); assert.equal(request.max_new_tokens, 777);
  assert.equal(request.system_prompt, '  Exact saved persona.\n');

  // The same mounted page receives an old saved comparison from before these
  // generation fields existed. Absent fields mean its original runtime defaults.
  router.query = { kind: 'compare', source_model: 'recorded-source', modified_model: 'recorded-edit',
    options: JSON.stringify({ seed: 0, n_prompts: 13 }) };
  request = await submit(settle());
  assert.equal(request.source_model, 'recorded-source'); assert.equal(request.modified_model, 'recorded-edit');
  assert.equal(request.enable_cot, false); assert.equal(request.temperature, 0);
  assert.equal(request.top_p, 0.9); assert.equal(request.max_new_tokens, 64);
  assert.equal(request.system_prompt, ''); assert.equal(request.seed, 0); assert.equal(request.n_prompts, 13);

  // Complete recorded choices override both legacy defaults and current Settings.
  const recorded = { enable_cot: true, temperature: 0, top_p: 0.4, max_new_tokens: 123,
    system_prompt: '  Exact recorded persona.\n', seed: 0, rederive: true, misalignment_control: true };
  router.query = { ...router.query, options: JSON.stringify(recorded) };
  request = await submit(settle());
  for (const [key, value] of Object.entries(recorded)) assert.deepEqual(request[key], value, key);

  // Explicit off/zero/null must clear the last selected choices without fetching
  // or restoring a conflicting saved library prompt.
  settings.defaultPatientSystemPromptId = 17;
  router.query = { ...router.query, options: JSON.stringify({ enable_cot: false, temperature: 0,
    top_p: 1, max_new_tokens: 1, system_prompt: null, seed: 0 }) };
  request = await submit(settle());
  assert.equal(request.enable_cot, false); assert.equal(request.temperature, 0);
  assert.equal(request.system_prompt, ''); assert.equal(request.top_p, 1); assert.equal(request.max_new_tokens, 1);
  assert.equal(promptCalls.length, 0, 'Recorded defaults must not fetch the current Settings prompt');

  // Returning to a fresh link restores current Settings, not the preceding run.
  settings.defaultPatientSystemPromptId = null;
  router.query = { kind: 'compare', modified_model: 'fresh-edit' };
  request = await submit(settle());
  assert.equal(request.enable_cot, true); assert.equal(request.max_new_tokens, 777);
  assert.equal(request.system_prompt, '  Exact saved persona.\n');
})().catch((error) => { console.error(error); process.exitCode = 1; });
""")


def test_weight_comparison_library_lookup_cannot_override_a_new_replay():
    run_frontend(HARNESS + r"""
(async () => {
  settings.defaultPatientSystemPromptId = 17;
  router.query = { kind: 'compare', modified_model: 'modified' };
  let tree = settle();
  assert.equal(promptCalls.length, 1); assert.equal(promptCalls[0].id, 17);
  assert.equal(launchButton(tree).props.disabled, true, 'Must wait for the actual library instructions');
  promptCalls[0].resolve({ id: 17, target: 'patient', prompt_type: 'system_prompt', prompt_text: '  Exact library text.\n' });
  await flush(); tree = settle();
  let request = await submit(tree);
  assert.equal(request.system_prompt, '  Exact library text.\n');
  assert.equal(request.enable_cot, true);

  // Navigate to another fresh comparison and then a replay while the library
  // request is outstanding. React effect cleanup must fence off the late result.
  router.query = { kind: 'compare', modified_model: 'another-edit' };
  tree = settle(); assert.equal(promptCalls.length, 2);
  assert.equal(launchButton(tree).props.disabled, true);
  router.query = { kind: 'compare', modified_model: 'replayed-edit', options: JSON.stringify({ seed: 0 }) };
  tree = settle(); request = await submit(tree);
  assert.equal(request.system_prompt, ''); assert.equal(request.enable_cot, false);
  promptCalls[1].resolve({ id: 17, target: 'patient', prompt_type: 'system_prompt', prompt_text: 'Late conflicting persona' });
  await flush(); request = await submit(settle());
  assert.equal(request.system_prompt, ''); assert.equal(request.enable_cot, false);
  assert.equal(promptCalls.length, 2, 'Replay must not issue its own Settings lookup');

  // An unavailable saved prompt blocks launch until the user explicitly clears it.
  router.query = { kind: 'compare', modified_model: 'third-edit' };
  tree = settle(); promptCalls[2].reject(new Error('missing prompt'));
  await flush(); tree = settle();
  assert.equal(launchButton(tree).props.disabled, true);
  const clear = find(tree, (node) => node.type === 'button' && node.props.children === 'Use no application system prompt');
  assert.ok(clear); clear.props.onClick(); request = await submit(settle());
  assert.equal(request.system_prompt, ''); assert.equal(request.enable_cot, true);
})().catch((error) => { console.error(error); process.exitCode = 1; });
""")
