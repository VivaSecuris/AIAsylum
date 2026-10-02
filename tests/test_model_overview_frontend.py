"""Provider-aware inventory and real model-to-comparison handoffs."""
from tests.test_conversation_frontend_identity import run_frontend
from tests.test_saved_defaults_frontend import HOOKS, SETUP, run_frontend as run_form


def test_inventory_keeps_provider_identity_and_honest_availability():
    run_frontend(r"""
const { buildModelOverview } = load(path.join(frontend, 'lib/model-overview'));
const provider = (name, extra = {}) => ({ name, label: name, aliases: [], kind: 'service', available: true, configured: true, models: [], ...extra });
const providers = [provider('transformers', { aliases: ['local'], kind: 'local' }), provider('ollama'), provider('openai', { kind: 'api', requires_api_key: 'KEY', configured: false, models: ['same-name'] })];
const checkpoint = (model_ref, availability = 'ready') => ({ id: model_ref, provider: 'transformers', model_ref, aliases: ['models/a b+#'], name: model_ref, kind: 'custom', availability, history: { test_runs: 0 } });
const runs = [
  { id: 1, test_type: 'one_shot', doctor_provider: 'local', doctor_model: '/models/a b+#', patient_provider: 'ollama', patient_model: 'same-name' },
  { id: 2, test_type: 'one_shot', doctor_provider: 'local', doctor_model: '/deleted', patient_provider: 'ollama', patient_model: 'old-model' },
  { id: 3, test_type: 'group_therapy', doctor_provider: 'ollama', doctor_model: 'same-name', patient_provider: 'group', patient_model: 'synthetic-group', meta_data: { test_config: { patients: [{ provider: 'ollama', model: 'member' }, { provider: 'openai', model: 'same-name' }] } } },
  { id: 4, test_type: 'one_shot', doctor_provider: 'local', doctor_model: 'models/a b+#', patient_provider: 'ollama', patient_model: 'same-name' },
  { id: 5, test_type: 'group_therapy', doctor_provider: 'ollama', doctor_model: 'same-name', patient_provider: 'group', patient_model: 'synthetic-legacy', meta_data: { patients: [{ provider: 'ollama', model: 'legacy-member' }] } },
];
const result = buildModelOverview([checkpoint('/models/a b+#'), checkpoint('/deleted', 'deleted')], providers, ['same-name', 'member'], runs);
const shared = result.filter(model => model.model === 'same-name');
assert.equal(shared.length, 2); assert.notEqual(shared[0].key, shared[1].key);
assert.equal(shared.find(model => model.provider === 'ollama').ready, true);
assert.equal(shared.find(model => model.provider === 'openai').ready, false);
assert.equal(shared.find(model => model.provider === 'openai').status, 'Set up provider');
assert.equal(result.filter(model => model.model === '/models/a b+#').length, 1);
assert.equal(result.find(model => model.model === '/models/a b+#').runs, 2);
assert.ok(!result.some(model => model.model === 'models/a b+#'));
assert.equal(result.find(model => model.model === 'old-model').ready, false);
assert.ok(result.some(model => model.model === 'member'));
assert.ok(result.some(model => model.model === 'legacy-member'));
assert.ok(!result.some(model => model.model === '/deleted' || model.model === 'synthetic-group' || model.model === 'synthetic-legacy'));
assert.equal(shared.find(model => model.provider === 'ollama').runs, 4);
const unknown = buildModelOverview([], providers, undefined, runs);
assert.equal(unknown.find(model => model.model === 'same-name' && model.provider === 'ollama').status, 'Availability not verified');
""")


def test_model_cards_open_chat_and_selection_roundtrips_special_identifiers():
    run_frontend(r"""
overrides.set('@/components/layout/Layout', { Layout: Child });
const { OverviewModelCard } = load(path.join(frontend, 'pages/models/index'));
const { modelComparisonUrl, parseModelSelection } = load(path.join(frontend, 'lib/model-overview'));
const model = { key: 'a', provider: 'transformers', model: '/models/a b+#', name: 'Edited model', providerLabel: 'Local', kind: 'checkpoint', ready: true, status: 'Ready on server', custom: true, runs: 2 };
const html = renderToStaticMarkup(React.createElement(OverviewModelCard, { model, selected: false, onSelect() {} }));
assert.match(html, /href="\/models\/chat\?provider=transformers&amp;model=%2Fmodels%2Fa\+b%2B%23"/);
assert.match(html, />Chat<\/a>/); assert.match(html, /source_model=%2Fmodels%2Fa\+b%2B%23/);
const identities = [model, { provider: 'ollama', model: 'same-name' }, { provider: 'openai', model: 'same-name' }];
const url = modelComparisonUrl(identities);
assert.equal(url.pathname, '/suite');
assert.deepEqual(parseModelSelection(url.query.models), identities.map(({ provider, model }) => ({ provider, model })));
assert.equal(parseModelSelection(JSON.stringify([identities[0], identities[0]])).length, 1);
for (const invalid of ['bad', '{}', '[{"provider":"ollama"}]', JSON.stringify(Array(21).fill(identities[0])), ['multiple-query-params']]) assert.deepEqual(parseModelSelection(invalid), []);
""")


def test_selected_models_prefill_real_suite_form_without_starting_it():
    run_form(SETUP + HOOKS + r"""
(async () => {
const selected = [{ provider: 'transformers', model: '/models/a b+#' }, { provider: 'ollama', model: 'same-name' }, { provider: 'openai', model: 'same-name' }];
const { modelComparisonUrl } = load(path.join(frontend, 'lib/model-overview'));
router.query = { ...modelComparisonUrl(selected).query, test_config: JSON.stringify({ prompt: 'What facts are missing?' }) };
const { SuiteForm } = load(path.join(frontend, 'components/forms/SuiteForm'));
const tree = render(SuiteForm);
assert.equal(submitted, undefined, 'Opening a comparison must not start generation');
await tree.props.onSubmit({ preventDefault() {} });
assert.deepEqual(submitted.models, selected);
assert.equal(submitted.doctor.model, 'doctor');
assert.equal(submitted.test_config.patient_system_prompt_id, 17);
assert.equal(submitted.test_config.roles.patient.enable_cot, true);
assert.equal(submitted.test_config.prompt, 'What facts are missing?');
})().catch(error => { console.error(error); process.exitCode = 1; });
""")


def test_sidebar_prompts_precede_models_and_chat_is_only_a_model_action():
    run_frontend(r"""
let pathname = '/models/chat';
overrides.set('next/router', { useRouter: () => ({ pathname }) });
overrides.set('next/image', { __esModule: true, default: () => null });
const { Sidebar } = load(path.join(frontend, 'components/layout/Sidebar'));
for (const current of ['/models/chat', '/models', '/']) {
  pathname = current;
  const html = renderToStaticMarkup(React.createElement(Sidebar));
  assert.ok(html.indexOf('Prompts and evaluation') < html.indexOf('>Models</p>'));
  assert.doesNotMatch(html, /Chat with a model|href="\/models\/chat"/);
  assert.match(html.match(/<a[^>]*href="\/models"[^>]*>/)[0], /aria-current="page"/);
  assert.match(html, /href="\/prompts"/);
  assert.match(html, /href="\/weights"/);
  assert.match(html, /href="\/dashboard"/);
}
""")
