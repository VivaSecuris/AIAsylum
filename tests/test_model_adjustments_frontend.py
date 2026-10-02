"""Execute adjustment handlers, request construction and evidence rendering."""
from tests.test_conversation_frontend_identity import run_frontend
from tests.test_model_workbench_frontend import HARNESS


SETUP = r"""
const settingsModule = load(path.join(frontend, 'lib/settings'));
const defaults = settingsModule.normalizeSettings({ defaultDoctorProvider: 'ollama', defaultDoctorModel: 'coach', defaultDoctorGeneration: { temperature: '0', max_tokens: '2048', enable_cot: false } });
overrides.set('@/lib/settings', { ...settingsModule, getSettings: () => defaults, saveSettings() { throw Error('Must not change global defaults'); } });
const settings = { system_prompt: '  Original system\n', temperature: 0, top_p: 1, max_tokens: 256, seed: 0, enable_cot: false, device: 'auto', dtype: 'bfloat16' };
const reply = { content: 'Original answer', provider: 'ollama', model: 'target', metadata: { request_system_prompts: [settings.system_prompt] }, finish_reason: 'stop', usage: { output_tokens: 3 }, generation: { requested: {}, applied: { temperature: 0, seed: 0 }, notes: [] }, elapsed_seconds: 1 };
const exchanges = [{ prompt: 'Original question', response: reply, at: 'now' }];
const revision = { id: 'revision-1', provider: 'ollama', model: 'target', mode: 'profile', status: 'proposed', instruction: 'Check arithmetic.', settings,
 baseline: { messages: [{ role: 'user', content: 'Original question' }, { role: 'assistant', content: 'Original answer' }], response: 'Original answer' },
 proposal: { summary: 'Show checked arithmetic.', system_prompt: '  Revised exact system\n', training_examples: [{ prompt: 'Compute', response: 'Checked answer' }], test_prompts: ['Fresh question'] }, tests: [] };
const applied = { ...revision, status: 'applied', active_target: { provider: 'ollama', model: 'target', settings: { ...settings, system_prompt: revision.proposal.system_prompt } } };
const after = { ...reply, content: 'Checked new answer', reasoning: 'Requested scratch text', reasoning_source: 'react', metadata: { request_system_prompts: [revision.proposal.system_prompt], native_reasoning: 'Native trace' }, finish_reason: 'length' };
const CoachSelector = () => null;
const providers = [{ name: 'ollama', aliases: [], kind: 'local' }, { name: 'transformers', aliases: ['local'], kind: 'local' }, { name: 'servus', aliases: [], kind: 'service' }, { name: 'agentic_a2a', aliases: ['agentic'], kind: 'service' }];
overrides.set('@/components/forms/ModelSelector', { ModelSelector: CoachSelector });
const control = (tree, label, type) => find(find(tree, n => n.type === 'label' && Array.isArray(n.props.children) && n.props.children[0] === label), n => n.type === type);
"""


def test_request_preserves_completed_answers_settings_and_parent_without_reasoning():
    run_frontend(SETUP + r"""
const helper = load(path.join(frontend, 'lib/model-adjustments'));
const input = { provider: 'ollama', model: 'target', instruction: '  Check arithmetic.\n', mode: 'profile', settings, exchanges, coach: { provider: 'ollama', model: 'coach', temperature: 0, enable_cot: false }, parent_id: 'parent' };
const request = helper.buildAdjustmentRequest(input);
assert.deepEqual(request.messages, revision.baseline.messages);
assert.equal(request.settings.system_prompt, settings.system_prompt);
assert.equal(request.instruction, input.instruction); assert.equal(request.settings.temperature, 0); assert.equal(request.settings.seed, 0); assert.equal(request.settings.enable_cot, false);
assert.equal(request.coach.temperature, 0); assert.equal(request.coach.enable_cot, false); assert.equal(request.parent_id, 'parent');
assert.deepEqual(request.baseline_evidence, reply);
assert.equal('exchanges' in request, false);
const partial = { prompt: 'Unanswered', response: { content: '', reasoning: 'Never replay' } };
assert.deepEqual(helper.adjustmentMessages([partial, ...exchanges]), revision.baseline.messages);
assert.throws(() => helper.buildAdjustmentRequest({ ...input, exchanges: [...exchanges, partial] }), /complete model answer/);
assert.throws(() => helper.buildAdjustmentRequest({ ...input, mode: 'weights' }), /local Transformers/);
assert.throws(() => helper.buildAdjustmentRequest({ ...input, coach: { provider: 'ollama', model: '' } }), /coach model/);
assert.throws(() => helper.buildAdjustmentRequest({ ...input, instruction: 'x'.repeat(8001) }), /8,000/);
assert.throws(() => helper.buildAdjustmentRequest({ ...input, exchanges: Array(51).fill(exchanges[0]) }), /shorter conversation/);
assert.equal(helper.buildAdjustmentRequest({ ...input, provider: 'local', mode: 'weights' }).mode, 'weights');
""")


def test_preflight_needs_real_request_and_all_blocking_acknowledgements():
    run_frontend(SETUP + r"""
const { adjustmentCanApply, adjustmentCanUse, adjustmentPreflightParams } = load(path.join(frontend, 'lib/model-adjustments'));
const weights = { ...revision, mode: 'weights', weight_request: { kind: 'lora', source_model: '/model', output_name: 'new', merge: true, dtype: 'float32' } };
const check = { code: 'resident', severity: 'blocking', acknowledgeable: true, message: 'Resident model' };
const preflight = { checks: [check], can_proceed: false, blocking_codes: ['resident'] };
assert.equal(adjustmentCanApply(revision, null, []), true);
assert.equal(adjustmentCanApply(weights, null, []), false);
assert.equal(adjustmentCanApply(weights, preflight, []), false);
assert.equal(adjustmentCanApply(weights, preflight, ['resident']), true);
assert.equal(adjustmentCanApply(weights, { ...preflight, checks: [{ ...check, acknowledgeable: false }] }, ['resident']), false);
assert.equal(adjustmentCanApply(weights, { checks: [], can_proceed: false, blocking_codes: [] }, []), false);
assert.equal(adjustmentCanApply({ ...weights, status: 'training' }, { checks: [], can_proceed: true, blocking_codes: [] }, []), false);
assert.deepEqual(adjustmentPreflightParams(weights.weight_request), { kind: 'lora', source_model: '/model', output_name: 'new', dtype: 'float32', merge: true });
assert.equal(adjustmentCanUse(revision), false); assert.equal(adjustmentCanUse(applied), true);
assert.equal(adjustmentCanUse({ ...applied, active_target: null }), false);
""")


def test_profile_actual_propose_apply_retest_retains_before_after_and_requires_explicit_use():
    run_frontend(HARNESS + SETUP + r"""
(async () => {
let submitted, appliedCall, tested, used;
overrides.set('@/lib/api', { apiClient: {
 listProviders: async () => providers,
 listModelAdjustments: async () => [], proposeModelAdjustment: async request => { submitted = request; return revision; },
 applyModelAdjustment: async (id, request) => { appliedCall = { id, request }; return applied; },
 testModelAdjustment: async id => { tested = id; return after; },
} });
const { ModelAdjustmentPanel } = load(path.join(frontend, 'components/weights/ModelAdjustmentPanel'));
const props = { provider: 'ollama', model: 'target', settings, exchanges, onUseVersion: value => { used = value; } };
let tree = settle(ModelAdjustmentPanel, props); await tick(); tree = settle(ModelAdjustmentPanel, props);
assert.equal(byText(tree, 'button', 'Propose adjustment').props.disabled, true);
control(tree, 'Describe the behavior change', 'textarea').props.onChange({ target: { value: 'Check arithmetic.' } }); tree = settle(ModelAdjustmentPanel, props);
await byText(tree, 'button', 'Propose adjustment').props.onClick(); tree = settle(ModelAdjustmentPanel, props);
assert.equal(submitted.coach.model, 'coach'); assert.equal(submitted.coach.max_tokens, 2048); assert.equal(submitted.coach.temperature, 0);
assert.deepEqual(submitted.messages, revision.baseline.messages); assert.equal(submitted.settings.seed, 0);
let html = renderToStaticMarkup(tree); assert.match(html, /does not change its weights/); assert.match(html, /Proposed system prompt/); assert.match(html, /Original answer/);
assert.equal(tested, undefined); assert.equal(used, undefined);
await byText(tree, 'button', 'Apply profile and re-test').props.onClick(); tree = settle(ModelAdjustmentPanel, props);
assert.deepEqual(appliedCall, { id: revision.id, request: { acknowledge: [] } }); assert.equal(tested, revision.id); assert.equal(used, undefined);
html = renderToStaticMarkup(tree); assert.match(html, /Before · saved answer/); assert.match(html, /After · re-test 1/);
assert.match(html, /Original answer/); assert.match(html, /Checked new answer/); assert.match(html, /Output reached its token limit/);
assert.ok(html.indexOf('System prompt sent') < html.indexOf('Requested scratch text'));
assert.ok(html.indexOf('System prompt sent') < html.indexOf('Native trace'));
assert.match(html, /these checks have not been run/); assert.equal(exchanges[0].response.content, 'Original answer');
byText(tree, 'button', 'Use version in a new chat').props.onClick(); assert.equal(used.active_target.settings.system_prompt, revision.proposal.system_prompt);
})().catch(error => { console.error(error); process.exitCode = 1; });
""")


def test_saved_training_revision_preflight_gate_poll_and_retest_when_checkpoint_ready():
    run_frontend(HARNESS + SETUP + r"""
(async () => {
let timer, applyCalls = 0, tests = 0, checked, appliedBody;
global.setTimeout = callback => { timer = callback; return 1; }; global.clearTimeout = () => { timer = undefined; };
const weights = { ...revision, provider: 'transformers', model: '/base', mode: 'weights', weight_request: { kind: 'lora', source_model: '/base', output_name: 'new-variant', merge: true }, tests: [] };
const training = { ...weights, status: 'training', weight_run: { id: 44, status: 'running' } };
const complete = { ...training, status: 'applied', weight_run: { id: 44, status: 'completed' }, active_target: { provider: 'transformers', model: '/new-variant', settings } };
let loaded = weights;
overrides.set('@/lib/api', { apiClient: {
 listProviders: async () => providers,
 listModelAdjustments: async () => [weights], getModelAdjustment: async () => loaded,
 weightPreflight: async request => { checked = request; return { checks: [{ code: 'resident', message: 'Resident model', severity: 'blocking', acknowledgeable: true }], blocking_codes: ['resident'], can_proceed: false }; },
 applyModelAdjustment: async (id, body) => { applyCalls++; appliedBody = body; return training; }, testModelAdjustment: async () => { tests++; return after; },
} });
const { ModelAdjustmentPanel } = load(path.join(frontend, 'components/weights/ModelAdjustmentPanel'));
const props = { provider: 'transformers', model: '/base', settings, exchanges, onUseVersion() {} };
let tree = settle(ModelAdjustmentPanel, props); await tick(); tree = settle(ModelAdjustmentPanel, props);
control(tree, 'Saved revisions', 'select').props.onChange({ target: { value: weights.id } }); await tick(); tree = settle(ModelAdjustmentPanel, props); await tick(); tree = settle(ModelAdjustmentPanel, props);
assert.equal(checked.source_model, '/base'); assert.equal(checked.output_name, 'new-variant');
assert.equal(byText(tree, 'button', 'Train new checkpoint').props.disabled, true);
await byText(tree, 'button', 'Train new checkpoint').props.onClick(); assert.equal(applyCalls, 0);
const preflight = find(tree, n => n.type?.name === 'PreflightPanel'); preflight.props.onAcknowledge(['resident']); tree = settle(ModelAdjustmentPanel, props);
assert.equal(byText(tree, 'button', 'Train new checkpoint').props.disabled, false);
assert.match(renderToStaticMarkup(tree), /not used by training/);
await byText(tree, 'button', 'Train new checkpoint').props.onClick(); tree = settle(ModelAdjustmentPanel, props);
assert.equal(applyCalls, 1); assert.deepEqual(appliedBody, { acknowledge: ['resident'] }); assert.equal(tests, 0);
assert.equal(byText(tree, 'button', 'Use version in a new chat'), undefined);
assert.match(renderToStaticMarkup(tree), /Training is in progress/);
loaded = complete; await timer(); tree = settle(ModelAdjustmentPanel, props);
assert.ok(byText(tree, 'button', 'Re-test saved conversation')); assert.equal(tests, 0, 'Finished training does not silently start a model call');
await byText(tree, 'button', 'Re-test saved conversation').props.onClick(); tree = settle(ModelAdjustmentPanel, props); assert.equal(tests, 1);
})().catch(error => { console.error(error); process.exitCode = 1; });
""")


def test_failed_retest_keeps_applied_revision_and_late_proposal_does_not_replace_new_context():
    run_frontend(HARNESS + SETUP + r"""
(async () => {
let resolveProposal, calls = 0;
overrides.set('@/lib/api', { apiClient: {
 listProviders: async () => providers,
 listModelAdjustments: async () => [revision], getModelAdjustment: async () => revision,
 proposeModelAdjustment: () => { calls++; return new Promise(resolve => { resolveProposal = resolve; }); },
 applyModelAdjustment: async () => applied, testModelAdjustment: async () => { throw Error('Model unavailable'); },
} });
const { ModelAdjustmentPanel } = load(path.join(frontend, 'components/weights/ModelAdjustmentPanel'));
const props = { provider: 'ollama', model: 'target', settings, exchanges, onUseVersion() {} };
let tree = settle(ModelAdjustmentPanel, props); await tick(); tree = settle(ModelAdjustmentPanel, props);
control(tree, 'Saved revisions', 'select').props.onChange({ target: { value: revision.id } }); await tick(); tree = settle(ModelAdjustmentPanel, props);
await byText(tree, 'button', 'Apply profile and re-test').props.onClick(); tree = settle(ModelAdjustmentPanel, props);
assert.match(renderToStaticMarkup(tree), /Model unavailable/); assert.ok(byText(tree, 'button', 'Use version in a new chat'));
control(tree, 'Describe the behavior change', 'textarea').props.onChange({ target: { value: 'Another change' } }); tree = settle(ModelAdjustmentPanel, props);
const pending = byText(tree, 'button', 'Propose adjustment').props.onClick();
await byText(tree, 'button', 'Propose adjustment').props.onClick(); assert.equal(calls, 1, 'Double clicks do not duplicate proposals');
// ModelChat changes the panel key when model/settings/history change; React unmounts it.
for (const cleanup of cleanups.values()) cleanup();
resolveProposal(revision); await pending;
assert.equal(slots.some(value => value === revision), false, 'An unmounted request must not restore its proposal');
})().catch(error => { console.error(error); process.exitCode = 1; });
""")


def test_chat_use_profile_clears_context_and_checkpoint_navigation_restores_exact_revision():
    run_frontend(HARNESS + SETUP + r"""
(async () => {
let assigned, chatCalls = [], revisionReads = 0;
global.window = { location: { assign: url => { assigned = url; } } };
const Panel = () => null;
overrides.set('@/components/weights/ModelAdjustmentPanel', { ModelAdjustmentPanel: Panel });
overrides.set('@/lib/api', { apiClient: { listPrompts: async () => [],
 chatWithAnyModel: async request => { chatCalls.push(request); return reply; },
 getModelAdjustment: async () => { revisionReads++; return { ...applied, active_target: { provider: 'transformers', model: '/variant +#', settings: applied.active_target.settings } }; },
} });
const { ModelChat } = load(path.join(frontend, 'components/weights/ModelChat'));
let props = { provider: 'ollama', model: 'target' };
let tree = settle(ModelChat, props);
editMessage(tree, 'Question'); tree = settle(ModelChat, props); await find(tree, n => n.type === 'form').props.onSubmit({ preventDefault }); tree = settle(ModelChat, props);
const baselinePanel = find(tree, n => n.type === Panel); const oldKey = baselinePanel.key;
baselinePanel.props.onUseVersion(applied); tree = settle(ModelChat, props);
const newPanel = find(tree, n => n.type === Panel);
assert.notEqual(newPanel.key, oldKey); assert.equal(newPanel.props.exchanges.length, 0); assert.equal(newPanel.props.parentId, applied.id);
assert.equal(newPanel.props.settings.system_prompt, revision.proposal.system_prompt); assert.equal(chatCalls.length, 1);
editMessage(tree, 'Fresh question'); tree = settle(ModelChat, props); await find(tree, n => n.type === 'form').props.onSubmit({ preventDefault }); tree = settle(ModelChat, props);
assert.deepEqual(chatCalls.at(-1).messages, [{ role: 'user', content: 'Fresh question' }]); assert.equal(chatCalls.at(-1).system_prompt, revision.proposal.system_prompt);
find(tree, n => n.type === Panel).props.onUseVersion({ ...applied, active_target: { provider: 'transformers', model: '/variant +#', settings } });
const destination = new URL(assigned, 'http://localhost'); assert.equal(destination.pathname, '/models/chat'); assert.equal(destination.searchParams.get('model'), '/variant +#'); assert.equal(destination.searchParams.get('adjustment_id'), applied.id);
props = { provider: 'transformers', model: '/variant +#', initialAdjustmentId: applied.id }; tree = settle(ModelChat, props); await tick(); tree = settle(ModelChat, props);
assert.equal(revisionReads, 1); assert.equal(find(tree, n => n.type === Panel).props.settings.system_prompt, revision.proposal.system_prompt); assert.equal(chatCalls.length, 2, 'Restoring a version does not send a message');
find(tree, n => n.type === 'select' && n.props.value === 'patient').props.onChange({ target: { value: 'doctor' } }); tree = settle(ModelChat, props); await tick(); tree = settle(ModelChat, props);
assert.equal(revisionReads, 1, 'Changing role intentionally leaves the restored revision');
assert.equal(find(tree, n => n.type === Panel).props.parentId, undefined);
})().catch(error => { console.error(error); process.exitCode = 1; });
""")


def test_service_target_and_alias_coach_cannot_start_an_adjustment():
    run_frontend(HARNESS + SETUP + r"""
(async () => {
let calls = 0;
overrides.set('@/lib/api', { apiClient: { listProviders: async () => providers, listModelAdjustments: async () => [], proposeModelAdjustment: async () => { calls++; return revision; } } });
const { ModelAdjustmentPanel } = load(path.join(frontend, 'components/weights/ModelAdjustmentPanel'));
let props = { provider: 'servus', model: 'target', settings, exchanges, onUseVersion() {} };
let tree = settle(ModelAdjustmentPanel, props); await tick(); tree = settle(ModelAdjustmentPanel, props);
control(tree, 'Describe the behavior change', 'textarea').props.onChange({ target: { value: 'Check facts' } }); tree = settle(ModelAdjustmentPanel, props);
assert.match(renderToStaticMarkup(tree), /ordinary chat remains available/);
assert.equal(byText(tree, 'button', 'Propose adjustment').props.disabled, true);
await byText(tree, 'button', 'Propose adjustment').props.onClick(); assert.equal(calls, 0);
props = { ...props, provider: 'ollama' }; tree = settle(ModelAdjustmentPanel, props); await tick(); tree = settle(ModelAdjustmentPanel, props);
find(tree, n => n.type === CoachSelector).props.onProviderChange('agentic'); tree = settle(ModelAdjustmentPanel, props);
assert.equal(find(tree, n => n.type === CoachSelector).props.model, '');
find(tree, n => n.type === CoachSelector).props.onModelChange('remote-agent'); tree = settle(ModelAdjustmentPanel, props);
assert.match(renderToStaticMarkup(tree), /coach provider does not accept/);
assert.equal(byText(tree, 'button', 'Propose adjustment').props.disabled, true);
await byText(tree, 'button', 'Propose adjustment').props.onClick(); assert.equal(calls, 0);
})().catch(error => { console.error(error); process.exitCode = 1; });
""")


def test_saved_baseline_evidence_is_labeled_and_system_precedes_reasoning():
    run_frontend(HARNESS + SETUP + r"""
(async () => {
const withEvidence = { ...applied, baseline: { ...applied.baseline, evidence: { ...reply, reasoning: 'Original reasoning', reasoning_source: 'react' }, evidence_source: 'client_reported' }, tests: [{ at: 'later', response: after, target: applied.active_target }] };
overrides.set('@/lib/api', { apiClient: { listProviders: async () => providers, listModelAdjustments: async () => [withEvidence], getModelAdjustment: async () => withEvidence } });
const { ModelAdjustmentPanel } = load(path.join(frontend, 'components/weights/ModelAdjustmentPanel'));
const props = { provider: 'ollama', model: 'target', settings, exchanges: [], onUseVersion() {} };
let tree = settle(ModelAdjustmentPanel, props); await tick(); tree = settle(ModelAdjustmentPanel, props);
control(tree, 'Saved revisions', 'select').props.onChange({ target: { value: withEvidence.id } }); await tick(); tree = settle(ModelAdjustmentPanel, props);
const html = renderToStaticMarkup(tree);
const beforeIndex = html.indexOf('Before · saved answer'), before = html.slice(beforeIndex);
assert.ok(before.indexOf('System prompt sent') < before.indexOf('Original reasoning'));
assert.match(before, /Original system/); assert.match(before, /supplied by this chat page/);
assert.doesNotMatch(before, /generation evidence was not recorded/);
assert.match(html, /Checked new answer/);
})().catch(error => { console.error(error); process.exitCode = 1; });
""")


def test_adjustment_api_methods_encode_ids_and_preserve_structured_requests():
    run_frontend(SETUP + r"""
(async () => {
const calls = [];
const transport = { interceptors: { request: { use() {} }, response: { use() {} } },
 get: async (...args) => { calls.push(['GET', ...args]); return { data: revision }; },
 post: async (...args) => { calls.push(['POST', ...args]); return { data: revision }; } };
overrides.set('axios', { __esModule: true, default: { create: () => transport } });
const { apiClient } = load(path.join(frontend, 'lib/api'));
const request = { provider: 'ollama', model: 'a +#', instruction: 'Change', settings, messages: revision.baseline.messages, mode: 'profile', coach: { provider: 'ollama', model: 'coach', temperature: 0, enable_cot: false } };
await apiClient.proposeModelAdjustment(request); await apiClient.listModelAdjustments({ provider: 'ollama', model: 'a +#' });
await apiClient.getModelAdjustment('id/+#'); await apiClient.applyModelAdjustment('id/+#', { acknowledge: ['resident'] }); await apiClient.testModelAdjustment('id/+#');
assert.deepEqual(calls[0], ['POST', '/api/v1/models/adjustments', request, { timeout: 600000 }]);
assert.deepEqual(calls[1], ['GET', '/api/v1/models/adjustments', { params: { provider: 'ollama', model: 'a +#' } }]);
assert.equal(calls[2][1], '/api/v1/models/adjustments/id%2F%2B%23');
assert.deepEqual(calls[3], ['POST', '/api/v1/models/adjustments/id%2F%2B%23/apply', { acknowledge: ['resident'] }]);
assert.deepEqual(calls[4], ['POST', '/api/v1/models/adjustments/id%2F%2B%23/test', {}, { timeout: 600000 }]);
})().catch(error => { console.error(error); process.exitCode = 1; });
""")
