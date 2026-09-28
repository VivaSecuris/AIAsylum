"""Execute role-scoped patient prompt selection and actual form submissions."""
from tests.test_conversation_frontend_identity import run_frontend
from tests.test_model_workbench_frontend import HARNESS


PROMPTS = r"""
const prompts = [
  { id: 1, name: 'Patient task', prompt_type: 'test_prompt', target: 'patient', prompt_text: 'Patient input' },
  { id: 2, name: 'General task', prompt_type: 'test_prompt', target: null, prompt_text: 'General input' },
  { id: 3, name: 'Doctor goal', prompt_type: 'test_prompt', target: 'doctor', prompt_text: 'Private doctor-only goal' },
  { id: 4, name: 'Evaluator task', prompt_type: 'test_prompt', target: 'evaluator', prompt_text: 'Private evaluator-only input' },
  { id: 5, name: 'Patient system', prompt_type: 'system_prompt', target: 'patient', prompt_text: 'System instructions' },
];
"""


def test_patient_picker_filters_other_role_presets_and_flags_stale_selected_ids():
    run_frontend(HARNESS + PROMPTS + r"""
overrides.set('lucide-react', componentStubs);
overrides.set('@/lib/hooks', {
  useDebouncedValue: value => value,
  usePrompts: () => ({ data: prompts }),
  usePromptsByIds: ids => ids.map(id => ({ data: prompts.find(p => p.id === id) })),
});
const { PromptPicker } = load(path.join(frontend, 'components/forms/create-test/PromptPicker'));
let changed;
for (const mode of ['single', 'multi']) {
  slots = []; cleanups.clear();
  const tree = settle(PromptPicker, { mode, selectedIds: [], onChange: value => { changed = value; } });
  const html = renderToStaticMarkup(tree);
  assert.match(html, /Patient task|General task/);
  assert.doesNotMatch(html, /Doctor goal|Evaluator task|Patient system|doctor-only|evaluator-only/);
  find(tree, node => node.type === 'button' && renderToStaticMarkup(node).includes('General task')).props.onClick();
  assert.deepEqual(changed, [2]);
}
for (const id of [3, 4, 5]) {
  slots = []; cleanups.clear();
  const html = renderToStaticMarkup(settle(PromptPicker, { mode: 'single', selectedIds: [id], onChange() {} }));
  assert.match(html, /This is not a patient test prompt/);
  assert.match(html, /Search test prompts/);
  assert.doesNotMatch(html, /Private doctor-only goal|Private evaluator-only input|System instructions/);
}
""")


def test_actual_create_and_suite_forms_block_wrong_role_and_unchecked_library_prompts():
    run_frontend(HARNESS + PROMPTS + r"""
(async () => {
overrides.set('react', { ...overrides.get('react'), useMemo: callback => callback() });
const settings = load(path.join(frontend, 'lib/settings'));
overrides.set('@/lib/settings', { ...settings, getSettings: () => settings.normalizeSettings({ defaultPatientProvider: 'ollama', defaultPatientModel: 'p', defaultDoctorProvider: 'ollama', defaultDoctorModel: 'd' }) });
let query = {}, calls = [], errors = [], pending = false;
overrides.set('next/router', { useRouter: () => ({ query, push() {} }) });
overrides.set('lucide-react', componentStubs);
overrides.set('@/lib/toast', { toast: { error: error => errors.push(error), success() {} } });
overrides.set('@/lib/hooks', {
  usePromptsByIds: ids => ids.map(id => pending ? {} : ({ data: prompts.find(p => p.id === id) })),
  useCreateTestRun: () => ({ mutateAsync: async request => { calls.push(request); return { id: 1 }; } }),
  useCreateSuite: () => ({ mutateAsync: async request => { calls.push(request); return { id: 1 }; } }),
  useRunBenchmark: () => ({}), useBenchmarks: () => ({}), useEditedModels: () => ({}),
});
const { CreateTestForm } = load(path.join(frontend, 'components/forms/create-test/CreateTestForm'));
const { SuiteForm } = load(path.join(frontend, 'components/forms/SuiteForm'));
for (const Form of [CreateTestForm, SuiteForm]) for (const type of ['one_shot', 'multi_shot']) {
  for (const id of [1, 2, 3, 4, 5]) {
    slots = []; cleanups.clear(); calls = []; errors = [];
    query = { type, test_config: JSON.stringify(type === 'one_shot' ? { prompt_id: id } : { prompt_ids: [id] }) };
    const tree = settle(Form, { query });
    await find(tree, n => n.type === 'form').props.onSubmit({ preventDefault });
    assert.equal(calls.length, id < 3 ? 1 : 0, `${Form.name} ${type} id${id}`);
    if (id >= 3) assert.match(errors[0], /patient or general test prompts/);
  }
  slots = []; cleanups.clear(); calls = []; errors = []; pending = true;
  query = { type, test_config: JSON.stringify(type === 'one_shot' ? { prompt_id: 1 } : { prompt_ids: [1] }) };
  await find(settle(Form, { query }), n => n.type === 'form').props.onSubmit({ preventDefault });
  assert.equal(calls.length, 0); assert.match(errors[0], /wait for the selected prompts to be checked/);
  pending = false;
}
// Doctor user preset remains a valid explicit doctor-goal prefill, with no patient prompt ID.
for (const Form of [CreateTestForm, SuiteForm]) {
  slots = []; cleanups.clear(); calls = []; errors = [];
  query = { type: 'conversation', test_config: JSON.stringify({ doctor_goal: prompts[2].prompt_text }) };
  await find(settle(Form, { query }), n => n.type === 'form').props.onSubmit({ preventDefault });
  assert.equal(calls.length, 1); assert.equal(calls[0].test_config.doctor_goal, prompts[2].prompt_text);
  assert.equal(calls[0].test_config.prompt_id, undefined);
}
})().catch(error => { console.error(error); process.exitCode = 1; });
""")
