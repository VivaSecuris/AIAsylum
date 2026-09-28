"""Execute the model workspace's configuration and real chat component handlers."""
from tests.test_conversation_frontend_identity import run_frontend


HARNESS = r"""
let slots = [], cursor = 0, pendingEffects = [], cleanups = new Map();
overrides.set('react', {
  ...React,
  useState(initial) {
    const i = cursor++;
    if (!(i in slots)) slots[i] = typeof initial === 'function' ? initial() : initial;
    return [slots[i], value => { slots[i] = typeof value === 'function' ? value(slots[i]) : value; }];
  },
  useRef(initial) {
    const i = cursor++;
    if (!(i in slots)) slots[i] = { current: initial };
    return slots[i];
  },
  useEffect(effect, deps) {
    const i = cursor++, previous = slots[i];
    if (!previous || deps.some((dep, n) => !Object.is(dep, previous[n]))) pendingEffects.push(() => {
      cleanups.get(i)?.();
      const cleanup = effect();
      if (typeof cleanup === 'function') cleanups.set(i, cleanup); else cleanups.delete(i);
    });
    slots[i] = deps;
  },
});
function settle(Component, props = {}) {
  for (let attempt = 0; attempt < 12; attempt++) {
    cursor = 0; pendingEffects = [];
    const tree = Component(props);
    if (!pendingEffects.length) return tree;
    for (const effect of pendingEffects) effect();
  }
  throw new Error('Component effects did not settle');
}
const Link = ({ children, href, ...props }) => React.createElement('a', { ...props, href: typeof href === 'string' ? href : href.pathname + '?' + new URLSearchParams(href.query) }, children);
overrides.set('next/link', { __esModule: true, default: Link });
function byText(tree, type, text) { return find(tree, node => node.type === type && node.props.children === text); }
function editMessage(tree, value) { find(tree, node => node.type === 'textarea' && node.props.id === 'model-chat-message').props.onChange({ target: { value } }); }
const tick = () => new Promise(resolve => setImmediate(resolve));
const preventDefault = () => {};
"""


def test_chat_configuration_defaults_and_validated_workflow_handoffs():
    run_frontend(r"""
const settings = load(path.join(frontend, 'lib/settings'));
const chat = load(path.join(frontend, 'lib/model-chat'));
const form = load(path.join(frontend, 'lib/create-test-config'));
const defaults = settings.normalizeSettings({
  defaultPatientProvider: 'ollama', defaultPatientModel: 'saved-patient',
  defaultDoctorProvider: 'openai', defaultDoctorModel: 'saved-doctor',
  defaultDoctorSystemPromptText: 'Doctor only', defaultEvaluatorSystemPromptText: 'Evaluator only',
  defaultAutoAnalysis: true, defaultEvaluatorProvider: 'google', defaultEvaluatorModel: 'saved-evaluator',
  defaultPatientSystemPromptText: '  Exact persona\n',
  defaultPatientGeneration: { temperature: '0', top_p: '0.9', max_tokens: '8192', enable_cot: true, use_dynamic_strategies: false },
});
const saved = JSON.stringify(defaults);
const initial = chat.chatSettingsFromDefaults(defaults);
assert.equal(initial.temperature, 0); assert.equal(initial.max_tokens, 8192); assert.equal(initial.enable_cot, true);
assert.equal(initial.system_prompt, '  Exact persona\n');
const chosen = { ...initial, seed: 0, enable_cot: false };
const nextDefaults = chat.settingsWithChatDefaults(defaults, chosen);
assert.equal(JSON.stringify(defaults), saved, 'Do not mutate saved defaults');
assert.equal(nextDefaults.defaultDoctorModel, 'saved-doctor');
assert.equal(nextDefaults.defaultPatientModel, 'saved-patient');
assert.equal(nextDefaults.defaultPatientSystemPromptId, null);
assert.equal(nextDefaults.defaultPatientSystemPromptText, chosen.system_prompt);
assert.deepEqual(nextDefaults.defaultDoctorGeneration, defaults.defaultDoctorGeneration);
assert.equal(nextDefaults.defaultPatientGeneration.enable_cot, false);
assert.equal(nextDefaults.defaultPatientGeneration.temperature, '0');
const model = '/models/a b+#/checkpoint';
for (const provider of ['transformers', 'local', 'ollama', 'openai']) {
  const actions = chat.modelActionUrls(provider, model, chosen, 'Latest question?', defaults, 'base/model');
  assert.deepEqual(chat.modelChatUrl(provider, model), { pathname: '/models/chat', query: { provider, model } });
  for (const url of [actions.test, actions.suite]) {
    assert.equal(url.query.provider, provider); assert.equal(url.query.model, model);
    const { state } = form.initialFormState(settings.normalizeSettings(), url.query);
    const request = form.buildTestRunRequest(state);
    assert.equal(request.patient_model, model); assert.equal(request.patient_provider, provider);
    assert.equal(request.doctor_model, 'saved-doctor');
    assert.equal(request.test_config.prompt, 'Latest question?');
    assert.equal(request.test_config.roles.patient.temperature, 0);
    assert.equal(request.test_config.roles.patient.top_p, 0.9);
    assert.equal(request.test_config.roles.patient.max_tokens, 8192);
    assert.equal(request.test_config.roles.patient.enable_cot, false);
    assert.equal(request.test_config.seed, 0);
    assert.equal(state.patient.systemPrompt.text, chosen.system_prompt);
    assert.equal(request.test_config.patient_system_prompt, chosen.system_prompt);
    assert.equal(request.test_config.doctor_system_prompt, 'Doctor only');
    assert.equal(request.test_config.analysis_config.evaluator_system_prompt, 'Evaluator only');
  }
  if (provider === 'transformers' || provider === 'local') {
    assert.equal(actions.modify.query.source_model, model);
    assert.equal(actions.compare.query.source_model, 'base/model');
    assert.equal(actions.compare.query.modified_model, model);
    assert.deepEqual(JSON.parse(actions.compare.query.options), { system_prompt: chosen.system_prompt, enable_cot: false, temperature: 0, top_p: 0.9, max_new_tokens: 8192, seed: 0 });
  } else { assert.equal(actions.modify, null); assert.equal(actions.compare, null); }
}
const empty = chat.modelActionUrls('ollama', 'm', { ...chosen, system_prompt: '' }, '', defaults);
const config = JSON.parse(empty.test.query.test_config);
assert.equal(config.patient_system_prompt, undefined); assert.equal(config.patient_system_prompt_id, undefined);
assert.equal(config.patient_prompt_framing, false, 'A plain chat must stay unframed after handoff');
const resolved = chat.chatSettingsForHandoff({ ...chosen, temperature: null, top_p: null, max_tokens: null, seed: null }, { generation: { applied: { temperature: 0, top_p: .95, max_tokens: 3000, seed: 7 } } });
assert.deepEqual([resolved.temperature, resolved.top_p, resolved.max_tokens, resolved.seed], [0, .95, 3000, 7]);
assert.equal(chat.chatSettingsForHandoff(chosen, { generation: { applied: { temperature: 1, seed: 4 } } }).temperature, 0);
assert.throws(() => chat.modelActionUrls('', model, chosen, '', defaults), /Choose a provider/);
for (const patch of [{ temperature: -1 }, { top_p: 0 }, { max_tokens: 32769 }, { max_tokens: 1.5 }, { seed: -1 }, { seed: NaN }]) assert.throws(() => chat.buildChatRequest([], 'Hi', { ...chosen, ...patch }));
assert.equal(chat.supportsWeightActions('huggingface'), false);
""")


def test_chat_request_history_contains_only_completed_exchanges_and_preserves_zeroes():
    run_frontend(r"""
const { buildChatRequest, chatSettingsFromDefaults, chatErrorMessage } = load(path.join(frontend, 'lib/model-chat'));
const { DEFAULT_SETTINGS } = load(path.join(frontend, 'lib/settings'));
const settings = { ...chatSettingsFromDefaults(DEFAULT_SETTINGS), system_prompt: 'Exact persona', temperature: 0, seed: 0, enable_cot: false };
const history = [{ prompt: 'Question one', response: { content: 'Answer one' }, at: 'today' }];
const request = buildChatRequest(history, 'Question two', settings);
assert.deepEqual(request.messages, [{ role: 'user', content: 'Question one' }, { role: 'assistant', content: 'Answer one' }, { role: 'user', content: 'Question two' }]);
assert.equal(request.temperature, 0); assert.equal(request.seed, 0); assert.equal(request.enable_cot, false);
assert.equal(request.top_p, null); assert.equal(request.max_tokens, null);
assert.deepEqual(buildChatRequest(history, 'Question two', settings), request, 'Retry is stateless and must not duplicate a failed turn');
assert.equal(history.length, 1);
assert.deepEqual(buildChatRequest([...history, { prompt: 'Unanswered', response: { content: '', reasoning: 'Do not replay me' } }], 'Next', settings).messages, [...request.messages.slice(0, 2), { role: 'user', content: 'Next' }]);
assert.throws(() => buildChatRequest([], ' ', settings), /Enter a message/);
assert.throws(() => buildChatRequest(Array(50).fill(history[0]), 'one more', settings), /50-question limit/);
assert.equal(chatErrorMessage({ response: { data: { detail: [{ msg: 'Bad settings' }] } } }), 'Bad settings');
""")


def test_actual_chat_send_retry_provenance_and_workflow_links():
    run_frontend(HARNESS + r"""
(async () => {
const settingModule = load(path.join(frontend, 'lib/settings'));
let defaults = settingModule.normalizeSettings({ defaultDoctorProvider: 'ollama', defaultDoctorModel: 'doctor', defaultPatientSystemPromptText: '  Chosen persona\n', defaultPatientGeneration: { temperature: '0', enable_cot: true } });
overrides.set('@/lib/settings', { ...settingModule, getSettings: () => defaults, saveSettings: value => { defaults = value; } });
let calls = [], fail = true;
const response = { content: 'My reply', model: 'actual-model', provider: 'actual-provider', reasoning: 'Requested reasoning text', reasoning_source: 'react', finish_reason: 'length', elapsed_seconds: .5, usage: { output_tokens: 12 }, metadata: { request_system_prompts: ['  Chosen persona\n'], request_system_prompts_source: 'provider', native_reasoning: 'Native thought text', native_reasoning_source: 'provider' }, generation: { requested: { temperature: 0 }, applied: { temperature: 0, max_tokens: 128, top_p: .8 }, notes: ['Provider applied sampling.'] } };
overrides.set('@/lib/api', { apiClient: { listPrompts: async () => [], chatWithAnyModel: async request => { calls.push(request); if (fail) throw Error('Temporary problem'); return response; } } });
const { ModelChat } = load(path.join(frontend, 'components/weights/ModelChat'));
let props = { provider: 'ollama', model: 'qwen:7b', name: 'My model' };
let tree = settle(ModelChat, props);
assert.match(renderToStaticMarkup(tree), /ReACT on/);
assert.match(renderToStaticMarkup(tree), /Find and download its open-weight version/);
assert.equal(byText(tree, Link, 'Modify weights'), undefined);
editMessage(tree, 'Hello model'); tree = settle(ModelChat, props);
await find(tree, n => n.type === 'form').props.onSubmit({ preventDefault }); tree = settle(ModelChat, props);
assert.match(renderToStaticMarkup(tree), /Temporary problem/);
assert.equal(find(tree, n => n.type === 'textarea' && n.props.id === 'model-chat-message').props.value, 'Hello model');
fail = false;
await find(tree, n => n.type === 'form').props.onSubmit({ preventDefault }); tree = settle(ModelChat, props);
assert.equal(calls.length, 2); assert.deepEqual(calls[0], calls[1]);
assert.equal(calls[1].provider, 'ollama'); assert.equal(calls[1].model, 'qwen:7b');
assert.equal(calls[1].temperature, 0); assert.equal(calls[1].enable_cot, true);
assert.equal(calls[1].system_prompt, '  Chosen persona\n');
let html = renderToStaticMarkup(tree);
assert.match(html, /actual-provider \/ actual-model/); assert.match(html, /Requested reasoning \(ReACT\)/);
assert.match(html, /Requested reasoning text/); assert.match(html, /Model-provided reasoning/); assert.match(html, /Native thought text/); assert.match(html, /System prompt sent/);
assert.match(html, /Output reached its token limit/); assert.match(html, /Provider applied sampling/);
assert.ok(html.indexOf('System prompt sent') < html.indexOf('Requested reasoning (ReACT)'));
assert.ok(html.indexOf('System prompt sent') < html.indexOf('Native thought text'));
assert.ok(html.indexOf('Requested reasoning text') < html.indexOf('My reply'));
const testUrl = byText(tree, Link, 'Run a test').props.href;
const cfg = JSON.parse(testUrl.query.test_config);
assert.equal(cfg.roles.patient.temperature, 0); assert.equal(cfg.roles.patient.max_tokens, 128); assert.equal(cfg.roles.patient.top_p, .8);
assert.equal(cfg.patient_system_prompt, '  Chosen persona\n'); assert.equal(cfg.prompt, 'Hello model');
editMessage(tree, 'Follow-up'); tree = settle(ModelChat, props);
await find(tree, n => n.type === 'form').props.onSubmit({ preventDefault }); tree = settle(ModelChat, props);
assert.deepEqual(calls.at(-1).messages.map(m => m.role), ['user', 'assistant', 'user']);
assert.deepEqual(calls.at(-1).messages.map(m => m.content), ['Hello model', 'My reply', 'Follow-up']);
find(tree, n => n.type === 'input' && n.props.type === 'checkbox').props.onChange({ target: { checked: false } });
tree = settle(ModelChat, props); html = renderToStaticMarkup(tree);
assert.doesNotMatch(html, /My reply|Requested reasoning text/); assert.match(html, /ReACT off/);
byText(tree, 'button', 'Save as patient defaults').props.onClick(); tree = settle(ModelChat, props);
assert.equal(defaults.defaultPatientGeneration.enable_cot, false); assert.equal(defaults.defaultDoctorModel, 'doctor');
assert.match(renderToStaticMarkup(tree), /Saved as patient defaults/);
props = { provider: 'transformers', model: '/models/edited', sourceModel: 'base/model' }; tree = settle(ModelChat, props);
assert.ok(byText(tree, Link, 'Modify weights')); assert.ok(byText(tree, Link, 'Compare weight versions'));
})().catch(error => { console.error(error); process.exitCode = 1; });
""")


def test_wrong_role_saved_prompt_is_blocked_and_old_model_responses_are_ignored():
    run_frontend(HARNESS + r"""
(async () => {
const settingModule = load(path.join(frontend, 'lib/settings'));
overrides.set('@/lib/settings', { ...settingModule, getSettings: () => settingModule.normalizeSettings({ defaultPatientSystemPromptId: 12 }) });
let calls = [], resolveReply;
overrides.set('@/lib/api', { apiClient: { listPrompts: async () => [],
  getPrompt: async () => ({ prompt_type: 'system_prompt', target: 'doctor', prompt_text: 'Doctor-only instructions' }),
  chatWithAnyModel: request => { calls.push(request); return new Promise(resolve => { resolveReply = resolve; }); },
} });
const { ModelChat } = load(path.join(frontend, 'components/weights/ModelChat'));
let props = { provider: 'ollama', model: 'first' };
let tree = settle(ModelChat, props); await tick(); tree = settle(ModelChat, props);
assert.match(renderToStaticMarkup(tree), /saved patient prompt belongs to a different role/);
assert.doesNotMatch(renderToStaticMarkup(tree), /Doctor-only instructions/);
editMessage(tree, 'Question'); tree = settle(ModelChat, props);
await find(tree, n => n.type === 'form').props.onSubmit({ preventDefault }); assert.equal(calls.length, 0);
byText(tree, 'button', 'Use no app system prompt').props.onClick(); tree = settle(ModelChat, props);
const inFlight = find(tree, n => n.type === 'form').props.onSubmit({ preventDefault }); tree = settle(ModelChat, props);
assert.equal(calls[0].system_prompt, undefined);
props = { provider: 'openai', model: 'second' }; tree = settle(ModelChat, props);
resolveReply({ content: 'Old model response', provider: 'ollama', model: 'first', metadata: {} });
await inFlight; tree = settle(ModelChat, props);
const html = renderToStaticMarkup(tree);
assert.doesNotMatch(html, /Old model response|Waiting for the model’s reply/);
assert.match(html, /Chat with second/);
assert.match(html, /No app system prompt/);
})().catch(error => { console.error(error); process.exitCode = 1; });
""")


def test_workspace_route_keeps_provider_model_identity_and_gates_unavailable_providers():
    run_frontend(HARNESS + r"""
const settingModule = load(path.join(frontend, 'lib/settings'));
overrides.set('@/lib/settings', { ...settingModule, getSettings: () => settingModule.normalizeSettings({ defaultPatientProvider: 'ollama', defaultPatientModel: 'saved-model' }) });
let query = { provider: 'openai', model: 'chosen-model' }, pushed;
overrides.set('next/router', { useRouter: () => ({ isReady: true, query, push: value => { pushed = value; } }) });
let providers = [{ name: 'openai', aliases: [], configured: true, available: true }, { name: 'transformers', aliases: ['local'], configured: true, available: true }, { name: 'ollama', aliases: [], configured: true, available: true }];
overrides.set('@tanstack/react-query', { useQuery: () => ({ data: providers }) });
overrides.set('@/lib/api', { apiClient: { listPrompts: async () => [],} });
overrides.set('@/lib/model-catalog', { useModelCatalog: () => ({ data: { models: [{ model_ref: '/models/edited', name: 'edited', kind: 'custom', provider: 'transformers', source_model: 'base/model', availability: 'ready' }] } }), canUseModel: entry => entry.availability === 'ready' });
const Chat = () => null, Selector = () => null;
overrides.set('@/components/weights/ModelChat', { ModelChat: Chat });
overrides.set('@/components/forms/ModelSelector', { ModelSelector: Selector });
overrides.set('@/components/layout/Layout', { Layout: Child });
overrides.set('@/components/common/LoadingSpinner', { LoadingSpinner: Child });
const Page = load(path.join(frontend, 'pages/models/chat')).default;
let tree = settle(Page);
let chat = find(tree, n => n.type === Chat);
assert.equal(chat.props.provider, 'openai'); assert.equal(chat.props.model, 'chosen-model'); assert.equal(chat.props.disabledReason, undefined);
let selector = find(tree, n => n.type === Selector);
selector.props.onProviderChange('transformers'); tree = settle(Page);
selector = find(tree, n => n.type === Selector); assert.equal(selector.props.model, '');
selector.props.onModelChange('/models/edited'); tree = settle(Page);
// Editing the chooser does not silently switch a live conversation.
assert.equal(find(tree, n => n.type === Chat).props.model, 'chosen-model');
find(tree, n => n.type === 'form').props.onSubmit({ preventDefault });
assert.deepEqual(pushed, { pathname: '/models/chat', query: { provider: 'transformers', model: '/models/edited' } });
query = pushed.query; tree = settle(Page); chat = find(tree, n => n.type === Chat);
assert.equal(chat.props.model, '/models/edited'); assert.equal(chat.props.name, 'edited'); assert.equal(chat.props.sourceModel, 'base/model');
assert.ok(byText(tree, Link, 'Checkpoint provenance')); assert.ok(byText(tree, Link, 'Analyze activations'));
query = { provider: 'openai', model: 'chosen-model' }; providers[0] = { ...providers[0], available: false, unavailable_reason: 'Install the provider dependency.' };
tree = settle(Page); assert.equal(find(tree, n => n.type === Chat).props.disabledReason, 'Install the provider dependency.');
query = {}; tree = settle(Page); chat = find(tree, n => n.type === Chat);
assert.equal(chat.props.provider, 'ollama'); assert.equal(chat.props.model, 'saved-model');
query = { provider: 'openai' }; tree = settle(Page);
assert.equal(find(tree, n => n.type === Chat), undefined, 'Never borrow a saved model for another provider');
""")


def test_reasoning_only_reply_is_visible_but_not_replayed_as_an_answer():
    run_frontend(HARNESS + r"""
(async () => {
const settingModule = load(path.join(frontend, 'lib/settings'));
overrides.set('@/lib/settings', { ...settingModule, getSettings: () => settingModule.normalizeSettings() });
let calls = [];
overrides.set('@/lib/api', { apiClient: { listPrompts: async () => [], chatWithAnyModel: async request => { calls.push(request); return { content: '', reasoning: 'Reasoning only', reasoning_source: 'react', provider: 'ollama', model: 'm', finish_reason: 'length', metadata: { request_system_prompts: [] }, generation: { applied: {}, notes: [] } }; } } });
const { ModelChat } = load(path.join(frontend, 'components/weights/ModelChat'));
const props = { provider: 'ollama', model: 'm' };
let tree = settle(ModelChat, props); editMessage(tree, 'First question'); tree = settle(ModelChat, props);
await find(tree, n => n.type === 'form').props.onSubmit({ preventDefault }); tree = settle(ModelChat, props);
const html = renderToStaticMarkup(tree);
assert.match(html, /Reasoning only/); assert.match(html, /The model returned no answer text/);
assert.match(html, /This exchange is omitted from later conversation context/);
assert.match(html, /Output reached its token limit/);
assert.match(html, /No system message was supplied by the app/);
editMessage(tree, 'Second question'); tree = settle(ModelChat, props);
await find(tree, n => n.type === 'form').props.onSubmit({ preventDefault });
assert.deepEqual(calls[1].messages, [{ role: 'user', content: 'Second question' }]);
})().catch(error => { console.error(error); process.exitCode = 1; });
""")


def test_chat_role_defaults_and_presets_are_explicit_exact_and_do_not_change_model():
    run_frontend(HARNESS + r"""
(async () => {
const settingsModule = load(path.join(frontend, 'lib/settings'));
let defaults = settingsModule.normalizeSettings({
  defaultPatientSystemPromptText: 'Patient instructions', defaultDoctorSystemPromptText: 'Doctor instructions',
  defaultEvaluatorSystemPromptText: 'Evaluator instructions', defaultDoctorModel: 'saved-doctor',
  defaultPatientGeneration: { temperature: '1', enable_cot: true }, defaultDoctorGeneration: { temperature: '0', enable_cot: false },
});
const originalPatient = JSON.stringify(defaults.defaultPatientGeneration);
overrides.set('@/lib/settings', { ...settingsModule, getSettings: () => defaults, saveSettings: value => { defaults = value; } });
const chat = load(path.join(frontend, 'lib/model-chat'));
for (const role of ['patient', 'doctor', 'evaluator']) {
  const initial = chat.chatSettingsFromDefaults(defaults, role);
  assert.equal(initial.system_prompt, role[0].toUpperCase() + role.slice(1) + ' instructions');
}
const presets = [
  { id: 10, name: 'Doctor preset', prompt_type: 'system_prompt', target: 'doctor', prompt_text: '  Exact doctor preset\n' },
  { id: 11, name: 'Wrong target', prompt_type: 'system_prompt', target: 'patient', prompt_text: 'Never use me for doctor' },
  { id: 12, name: 'Not a system', prompt_type: 'test_prompt', target: 'doctor', prompt_text: 'Question' },
];
let calls = [], queries = [];
overrides.set('@/lib/api', { apiClient: {
  listPrompts: async query => { queries.push(query); return presets; },
  chatWithAnyModel: async request => { calls.push(request); return { content: 'Answer', provider: request.provider, model: request.model, metadata: {} }; },
} });
const { ModelChat } = load(path.join(frontend, 'components/weights/ModelChat'));
const props = { provider: 'ollama', model: 'clicked-model' };
let tree = settle(ModelChat, props); await tick(); tree = settle(ModelChat, props);
assert.match(renderToStaticMarkup(tree), /saved patient defaults/);
editMessage(tree, 'Question'); tree = settle(ModelChat, props);
await find(tree, n => n.type === 'form').props.onSubmit({ preventDefault }); tree = settle(ModelChat, props);
assert.match(renderToStaticMarkup(tree), /Answer/);
find(tree, n => n.type === 'select' && n.props.value === 'patient').props.onChange({ target: { value: 'doctor' } });
tree = settle(ModelChat, props); await tick(); tree = settle(ModelChat, props);
let html = renderToStaticMarkup(tree);
assert.match(html, /Saved doctor defaults/); assert.doesNotMatch(html, /Answer|Wrong target|Not a system/);
assert.equal(find(tree, n => n.type === 'textarea' && !n.props.id).props.value, 'Doctor instructions');
const presetSelect = find(tree, n => n.type === 'select' && n.props.value === '');
presetSelect.props.onChange({ target: { value: '10' } }); tree = settle(ModelChat, props);
assert.equal(find(tree, n => n.type === 'textarea' && !n.props.id).props.value, '  Exact doctor preset\n');
assert.match(renderToStaticMarkup(tree), /doctor preset: Doctor preset/);
editMessage(tree, 'New role question'); tree = settle(ModelChat, props);
await find(tree, n => n.type === 'form').props.onSubmit({ preventDefault }); tree = settle(ModelChat, props);
assert.equal(calls.at(-1).provider, 'ollama'); assert.equal(calls.at(-1).model, 'clicked-model');
assert.equal(calls.at(-1).system_prompt, '  Exact doctor preset\n'); assert.equal(calls.at(-1).temperature, 0); assert.equal(calls.at(-1).enable_cot, false);
assert.deepEqual(calls.at(-1).messages, [{ role: 'user', content: 'New role question' }]);
byText(tree, 'button', 'Save as doctor defaults').props.onClick(); tree = settle(ModelChat, props);
assert.equal(defaults.defaultDoctorSystemPromptText, '  Exact doctor preset\n');
assert.equal(defaults.defaultDoctorModel, 'saved-doctor'); assert.equal(defaults.defaultPatientSystemPromptText, 'Patient instructions');
assert.equal(JSON.stringify(defaults.defaultPatientGeneration), originalPatient);
assert.equal(queries.at(-1).target, 'doctor');
})().catch(error => { console.error(error); process.exitCode = 1; });
""")
