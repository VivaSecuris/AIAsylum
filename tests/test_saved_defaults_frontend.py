"""Exercise saved role defaults through real request builders and UI callbacks."""
from tests.test_weight_frontend_regressions import run_frontend


SETUP = r"""
const settingsModule = load(path.join(frontend, 'lib/settings'));
const config = load(path.join(frontend, 'lib/create-test-config'));
const saved = settingsModule.normalizeSettings({
  defaultPatientProvider: 'transformers', defaultPatientModel: 'patient', defaultPatientSystemPromptId: 17,
  defaultDoctorProvider: 'ollama', defaultDoctorModel: 'doctor', defaultDoctorSystemPromptText: 'Doctor custom',
  defaultEvaluatorProvider: 'ollama', defaultEvaluatorModel: 'evaluator', defaultEvaluatorSystemPromptText: 'Evaluator custom',
  defaultAutoAnalysis: true, defaultEnableCotDetection: false, defaultEnableFactualityCheck: true,
  defaultEnableManipulationAnalysis: true, defaultCotAnalysisMode: 'partial',
  defaultPatientGeneration: { temperature: '0.6', top_p: '0.8', max_tokens: '777', enable_cot: true },
  defaultDoctorGeneration: { temperature: '0', top_p: '0.7', max_tokens: '888', enable_cot: true, use_dynamic_strategies: false },
  defaultEvaluatorGeneration: { temperature: '0.1', top_p: '0.9', max_tokens: '999', enable_cot: true },
});
global.window = {};
let stored = JSON.stringify(saved);
global.localStorage = { getItem: () => stored, setItem: (_, value) => { stored = value; } };
"""


HOOKS = r"""
const React = requireFrontend('react');
let slots = [], cursor = 0;
overrides.set('react', { ...React,
  useState(initial) {
    const index = cursor++;
    if (!(index in slots)) slots[index] = typeof initial === 'function' ? initial() : initial;
    return [slots[index], (next) => { slots[index] = typeof next === 'function' ? next(slots[index]) : next; }];
  },
  useMemo(compute) { return compute(); },
  useEffect() {},
});
function render(Component, props = {}) { cursor = 0; return Component(props); }
function find(node, matches) {
  if (Array.isArray(node)) return node.map((child) => find(child, matches)).find(Boolean);
  if (!node || typeof node !== 'object') return undefined;
  return matches(node) ? node : find(node.props?.children, matches);
}
const named = (node, name) => node.type?.name === name || node.type === name;
let submitted;
const router = { isReady: true, query: {}, push() {}, back() {} };
overrides.set('next/router', { useRouter: () => router });
overrides.set('next/link', { __esModule: true, default: 'Link' });
overrides.set('@/lib/hooks', {
  usePromptsByIds: (ids) => ids.map((id) => ({ data: { id, prompt_text: 'selected prompt' } })),
  useEditedModels: () => ({ data: [] }), useBenchmarks: () => ({ data: { benchmarks: [{ name: 'gsm8k' }] } }),
  useCreateSuite: () => ({ mutateAsync: async (body) => { submitted = body; return { id: 1 }; } }),
  useCreateTestRun: () => ({ mutateAsync: async (body) => { submitted = body; return { id: 1 }; } }),
  useRunBenchmark: () => ({ mutateAsync: async (body) => { submitted = body; return { id: 1 }; } }),
});
overrides.set('@/lib/model-catalog', { useModelCatalog: () => ({ data: { models: [{ availability: 'ready', model_ref: 'patient', name: 'patient', kind: 'base' }] } }) });
overrides.set('@/lib/model-organization', { useModelOrganization: () => ({}), modelOrganizationKey: (value) => value });
overrides.set('@/lib/toast', { toast: { success() {}, error(message) { throw new Error(message); } } });
overrides.set('@/lib/utils', { formatApiError: (error) => error.message });
"""


def test_defaults_migrate_clone_and_override_per_role():
    run_frontend(SETUP + r"""
const old = settingsModule.normalizeSettings({ defaultPatientModel: 'old' });
assert.equal(old.defaultPatientGeneration.enable_cot, false);
assert.equal(old.defaultPatientGeneration.temperature, '');
assert.equal(old.defaultDoctorGeneration.use_dynamic_strategies, true);
old.defaultPatientGeneration.temperature = '1';
assert.equal(settingsModule.DEFAULT_SETTINGS.defaultPatientGeneration.temperature, '');
assert.equal(settingsModule.getRoleGenerationDefaults(saved, 'patient').enable_cot, true);
settingsModule.saveSettings(saved);
assert.deepEqual(settingsModule.getSettings(), saved);
stored = 'null'; assert.equal(settingsModule.getSettings().defaultPatientModel, '');
stored = '{bad'; assert.equal(settingsModule.getSettings().defaultPatientModel, '');

for (const type of ['one_shot', 'multi_shot', 'conversation', 'group_therapy']) {
  const state = config.initialFormState(saved, { type }).state;
  state.oneShotSource = 'custom'; state.customPrompt = 'question';
  state.groupPatients = [config.newGroupPatient('transformers', 'first', state.patient.systemPrompt, state.patient.generation),
    config.newGroupPatient('transformers', 'second', state.patient.systemPrompt, state.patient.generation)];
  const request = config.buildTestRunRequest(state).test_config;
  assert.deepEqual(request.roles.patient, { temperature: 0.6, top_p: 0.8, max_tokens: 777, enable_cot: true });
  assert.equal(request.roles.doctor.enable_cot, true, type + ' must offer doctor reasoning');
  assert.equal(request.roles.doctor.temperature, 0);
  if (['conversation', 'group_therapy'].includes(type)) assert.equal(request.roles.doctor.use_dynamic_strategies, false);
  assert.equal(request.analysis_config.evaluator_enable_cot, true);
  assert.equal(request.analysis_config.enable_cot_detection, false);
  assert.equal(request.analysis_config.evaluator_top_p, 0.9);
  assert.equal(request.analysis_config.evaluator_max_tokens, 999);
  assert.equal(request.analysis_config.evaluator_system_prompt, 'Evaluator custom');
  assert.equal(request.doctor_system_prompt, 'Doctor custom');
}
const custom = config.initialFormState({ ...saved, defaultPatientSystemPromptId: null, defaultPatientSystemPromptText: 'My custom patient' }).state;
assert.deepEqual(custom.patient.systemPrompt, { mode: 'custom', text: 'My custom patient' });
const partial = config.initialFormState(saved, { test_config: JSON.stringify({ roles: { patient: { enable_cot: false } } }) }).state;
assert.equal(partial.patient.generation.enable_cot, false);
assert.equal(partial.patient.generation.top_p, '0.8');
const complete = config.initialFormState(saved, { test_config: JSON.stringify({ config_version: 2, roles: { patient: { enable_cot: false } } }) }).state;
assert.equal(complete.patient.generation.top_p, '');
assert.equal(complete.doctor.generation.enable_cot, false);
assert.equal(complete.evaluator.enable_cot, false);
assert.equal(complete.evaluator.enabled, false);
assert.deepEqual(complete.patient.systemPrompt, { mode: 'default' });
const cleared = config.initialFormState(saved, { test_config: JSON.stringify({ auto_analysis: false,
  patient_system_prompt_id: null, doctor_system_prompt: '', analysis_config: { evaluator_system_prompt: '' } }) }).state;
assert.equal(cleared.evaluator.enabled, false);
assert.deepEqual(cleared.patient.systemPrompt, { mode: 'default' });
assert.deepEqual(cleared.doctor.systemPrompt, { mode: 'default' });
assert.deepEqual(cleared.evaluator.systemPrompt, { mode: 'default' });
""")


def test_changing_provider_clears_the_old_model_without_touching_role_settings():
    run_frontend(SETUP + HOOKS + r"""
const { RoleFields } = load(path.join(frontend, 'components/forms/create-test/RoleStep'));
const { EvaluatorStep } = load(path.join(frontend, 'components/forms/create-test/EvaluatorStep'));
const { AnalysisConfigDialog } = load(path.join(frontend, 'components/analysis/AnalysisConfigDialog'));
const state = config.initialFormState(saved).state;
let changed;
let tree = render(RoleFields, { role: 'doctor', step: state.doctor, onChange: (next) => { changed = next; } });
find(tree, (node) => named(node, 'ModelSelector')).props.onProviderChange('anthropic');
assert.equal(changed.provider, 'anthropic'); assert.equal(changed.model, '');
assert.deepEqual(changed.systemPrompt, state.doctor.systemPrompt);
assert.deepEqual(changed.generation, state.doctor.generation);
tree = render(EvaluatorStep, { number: 3, value: state.evaluator, onChange: (next) => { changed = next; } });
find(tree, (node) => named(node, 'ModelSelector')).props.onProviderChange('google');
assert.equal(changed.provider, 'google'); assert.equal(changed.model, '');
assert.equal(changed.enable_cot, true);
slots = [];
tree = render(AnalysisConfigDialog, { isOpen: true, onClose() {}, onConfirm(value) { submitted = value; } });
find(tree, (node) => named(node, 'ModelSelector')).props.onProviderChange('anthropic');
tree = render(AnalysisConfigDialog, { isOpen: true, onClose() {}, onConfirm(value) { submitted = value; } });
const selector = find(tree, (node) => named(node, 'ModelSelector'));
assert.equal(selector.props.provider, 'anthropic'); assert.equal(selector.props.model, '');
""")


def test_benchmark_and_analysis_defaults_round_trip_with_explicit_clearing():
    run_frontend(SETUP + r"""
const state = config.initialFormState(saved, { type: 'benchmark' }).state;
state.benchmark = 'gsm8k';
const benchmark = config.buildBenchmarkRequest(state);
assert.equal(benchmark.enable_cot, true); assert.equal(benchmark.temperature, 0.6);
assert.equal(benchmark.top_p, 0.8); assert.equal(benchmark.max_new_tokens, 777);
assert.equal(benchmark.patient_system_prompt_id, 17);
const suite = config.buildBenchmarkSuiteConfig(state);
assert.equal(suite.patient_system_prompt_id, 17);
assert.equal(suite.roles.patient.enable_cot, true);
assert.equal(suite.roles.patient.max_tokens, 777);
const replay = config.initialFormState(saved, { type: 'benchmark', test_config: JSON.stringify({ ...suite, benchmark: 'gsm8k', num_samples: 100 }) }).state;
assert.deepEqual(config.buildBenchmarkRequest(replay), benchmark);
state.patient.generation = config.emptyGeneration();
state.patient.systemPrompt = { mode: 'default' };
const fresh = config.buildBenchmarkRequest(state);
assert.equal(fresh.temperature, 0); assert.equal(fresh.enable_cot, false); assert.equal(fresh.max_new_tokens, 512);
assert.equal('patient_system_prompt_id' in fresh, false);
state.patient.systemPrompt = { mode: 'custom', text: '  exact instructions\n' };
assert.equal(config.buildBenchmarkRequest(state).patient_system_prompt, '  exact instructions\n');
assert.equal(config.buildBenchmarkSuiteConfig(state).roles.patient.temperature, 0);
const evaluator = config.initialEvaluatorState(saved, { evaluator_enable_cot: false, evaluator_temperature: 0,
  evaluator_provider: '', evaluator_model: '', evaluator_system_prompt_id: null, enable_cot_detection: true });
const analysis = config.buildAnalysisConfig(evaluator);
assert.equal(analysis.evaluator_enable_cot, false); assert.equal(analysis.evaluator_temperature, 0);
assert.equal(analysis.enable_cot_detection, true); assert.equal(analysis.evaluator_top_p, 0.9);
assert.equal('evaluator_system_prompt_id' in analysis, false); assert.equal('evaluator_system_prompt' in analysis, false);
assert.equal('evaluator_model' in analysis, false);
""")


def test_suite_actual_submit_inherits_link_and_benchmark_defaults():
    run_frontend(SETUP + HOOKS + r"""
const { SuiteForm } = load(path.join(frontend, 'components/forms/SuiteForm'));
router.query = { model: 'linked', provider: 'ollama', type: 'conversation',
  test_config: JSON.stringify({ roles: { patient: { temperature: 0, enable_cot: false } }, patient_system_prompt: 'Link custom', patient_system_prompt_id: null }) };
let tree = render(SuiteForm);
(async () => {
  await tree.props.onSubmit({ preventDefault() {} });
  assert.deepEqual(submitted.models, [{ provider: 'ollama', model: 'linked' }]);
  assert.equal(submitted.test_config.patient_system_prompt, 'Link custom');
  assert.equal(submitted.test_config.roles.patient.temperature, 0);
  assert.equal(submitted.test_config.roles.patient.enable_cot, false);
  assert.equal(submitted.test_config.roles.patient.top_p, 0.8);
  assert.equal(submitted.test_config.analysis_config.evaluator_enable_cot, true);
  // Fresh benchmark suite must send the same settings as the single benchmark.
  slots = []; router.query = { type: 'benchmark', test_config: JSON.stringify({ benchmark: 'gsm8k' }) };
  tree = render(SuiteForm);
  await tree.props.onSubmit({ preventDefault() {} });
  assert.equal(submitted.test_config.enable_cot, true);
  assert.equal(submitted.test_config.patient_system_prompt_id, 17);
  assert.equal(submitted.test_config.roles.patient.top_p, 0.8);
  assert.equal(submitted.test_config.max_new_tokens, 777);
})().catch((error) => { console.error(error); process.exitCode = 1; });
""")


def test_new_group_patients_keep_defaults_and_independent_values():
    run_frontend(SETUP + HOOKS + r"""
const { CreateTestForm } = load(path.join(frontend, 'components/forms/create-test/CreateTestForm'));
let tree = render(CreateTestForm, { query: {} });
find(tree, (node) => named(node, 'TestDesignSection')).props.setTestType('group_therapy');
tree = render(CreateTestForm, { query: {} });
const group = find(tree, (node) => named(node, 'GroupPatientsStep'));
assert.equal(group.props.patients.length, 2);
for (const patient of group.props.patients) {
  assert.equal(patient.model, 'patient'); assert.equal(patient.generation.enable_cot, true);
  assert.deepEqual(patient.systemPrompt, { mode: 'library', id: 17 });
}
assert.notEqual(group.props.patients[0].generation, group.props.patients[1].generation);
const { GroupPatientsStep } = load(path.join(frontend, 'components/forms/create-test/GroupPatientsStep'));
let changed;
const groupTree = GroupPatientsStep({ ...group.props, onChange: (patients) => { changed = patients; } });
const add = find(groupTree, (node) => node.type === 'button' && Array.isArray(node.props.children) && node.props.children.includes(' Add patient'));
assert.ok(add); add.props.onClick();
assert.equal(changed.length, 3); assert.equal(changed[2].generation.max_tokens, '777');
assert.equal(changed[2].systemPrompt.id, 17);
""")


def test_analysis_dialog_actual_confirm_keeps_sampling_and_separate_reasoning_detection():
    run_frontend(SETUP + HOOKS + r"""
const { AnalysisConfigDialog } = load(path.join(frontend, 'components/analysis/AnalysisConfigDialog'));
let confirmed;
const props = { isOpen: true, onClose() {}, onConfirm: (value) => { confirmed = value; } };
let tree = render(AnalysisConfigDialog, props);
const generation = find(tree, (node) => named(node, 'GenerationSettings'));
assert.equal(generation.props.value.enable_cot, true); assert.equal(generation.props.value.top_p, '0.9');
generation.props.onChange({ ...generation.props.value, enable_cot: false, temperature: '0' });
tree = render(AnalysisConfigDialog, props);
const submit = find(tree, (node) => node.type === 'button' && node.props.children === 'Run Analysis');
assert.equal(submit.props.disabled, false);
submit.props.onClick({ preventDefault() {}, stopPropagation() {} });
assert.equal(confirmed.evaluator_enable_cot, false); assert.equal(confirmed.enable_cot_detection, false);
assert.equal(confirmed.evaluator_top_p, 0.9); assert.equal(confirmed.evaluator_max_tokens, 999);
assert.equal(confirmed.evaluator_temperature, 0); assert.equal(confirmed.evaluator_system_prompt, 'Evaluator custom');
""")


def test_campaign_actual_launch_saved_defaults_and_recorded_retry_precedence():
    run_frontend(SETUP + HOOKS + r"""
overrides.set('@/lib/benchmark-campaigns', {
  BENCHMARK_CHECKS: [{ id: 'gsm8k', name: 'GSM8K' }], benchmarkName: (name) => name, shortModelName: (name) => name,
});
const { CampaignBuilder } = load(path.join(frontend, 'components/benchmarks/CampaignBuilder'));
function launch(initial) {
  slots = [];
  const props = { initial, pending: false, error: null, onStart: (request) => { submitted = request; } };
  let tree = render(CampaignBuilder, props);
  for (let stage = 0; stage < 2; stage++) {
    const next = find(tree, (node) => node.type === 'button' && Array.isArray(node.props.children)
      && node.props.children.includes(stage === 0 ? 'Choose checks' : 'Review comparison'));
    assert.ok(next && !next.props.disabled); next.props.onClick(); tree = render(CampaignBuilder, props);
  }
  const run = find(tree, (node) => node.type === 'button' && Array.isArray(node.props.children)
    && typeof node.props.children[0] === 'string' && node.props.children[0].startsWith('Run '));
  assert.ok(run && !run.props.disabled); run.props.onClick();
}
launch(undefined);
assert.deepEqual(submitted.models, ['patient']); assert.equal(submitted.enable_cot, true);
assert.equal(submitted.max_new_tokens, 777); assert.equal(submitted.top_p, 0.8);
assert.equal(submitted.patient_system_prompt_id, 17);
launch({ name: 'Old direct answers', models: ['patient'], benchmarks: ['gsm8k'], num_samples: 10, seed: 3, max_new_tokens: 256 });
assert.equal(submitted.enable_cot, false); assert.equal(submitted.temperature, 0);
assert.equal(submitted.max_new_tokens, 256); assert.equal('top_p' in submitted, false);
assert.equal('patient_system_prompt_id' in submitted, false); assert.equal('patient_system_prompt' in submitted, false);
launch({ name: 'Frozen library persona', models: ['patient'], benchmarks: ['gsm8k'], num_samples: 10, seed: 3,
  max_new_tokens: 512, patient_system_prompt_id: 17, patient_system_prompt: '  exact saved library instructions\n', enable_cot: true, temperature: 0.2, top_p: 0.5 });
assert.equal(submitted.patient_system_prompt, '  exact saved library instructions\n');
assert.equal('patient_system_prompt_id' in submitted, false);
assert.equal(submitted.enable_cot, true); assert.equal(submitted.top_p, 0.5);
""")
