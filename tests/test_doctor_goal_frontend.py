"""Execute doctor-only goal replay and form rendering without contacting any model."""
from tests.test_conversation_frontend_identity import run_frontend
from tests.test_model_workbench_frontend import HARNESS


def test_doctor_goal_roundtrip_is_exact_only_for_conversation_and_group():
    run_frontend(r"""
const config = load(path.join(frontend, 'lib/create-test-config'));
const { normalizeSettings } = load(path.join(frontend, 'lib/settings'));
const defaults = normalizeSettings({ defaultDoctorProvider: 'ollama', defaultDoctorModel: 'd', defaultPatientProvider: 'ollama', defaultPatientModel: 'p' });
const goal = '  Explore recall of the patient’s stated music interests.\nDo not add a new task.  ';
for (const type of ['conversation', 'group_therapy', 'one_shot', 'multi_shot', 'benchmark']) {
  const query = { type, test_config: JSON.stringify({ doctor_goal: goal, doctor_system_prompt: 'Exact doctor system', patient_system_prompt: 'Exact patient system' }) };
  const { state } = config.initialFormState(defaults, query);
  assert.equal(state.doctorGoal, goal);
  const request = config.buildTestRunRequest(state);
  const applies = ['conversation', 'group_therapy'].includes(type);
  assert.equal(request.test_config.doctor_goal, applies ? goal : undefined);
  assert.equal(request.test_config.doctor_system_prompt, 'Exact doctor system');
  if (type !== 'group_therapy') assert.equal(request.test_config.patient_system_prompt, 'Exact patient system');
  assert.deepEqual(state.patient.systemPrompt, { mode: 'custom', text: 'Exact patient system' });
  const replay = config.initialFormState(defaults, { type, test_config: JSON.stringify(request.test_config) }).state;
  assert.equal(replay.doctorGoal, applies ? goal : '');
  state.doctorGoal = ' '.repeat(3) + '\n';
  assert.equal(config.buildTestRunRequest(state).test_config.doctor_goal, undefined);
  state.doctorGoal = 'a'.repeat(8001);
  assert.equal(config.validateForm(state).errors.some(error => error.includes('8,000 characters')), applies);
}
""")


def test_create_and_suite_forms_offer_goal_only_for_multi_turn_doctor():
    run_frontend(HARNESS + r"""
overrides.set('react', { ...overrides.get('react'), useMemo: callback => callback() });
const settings = load(path.join(frontend, 'lib/settings'));
overrides.set('@/lib/settings', { ...settings, getSettings: () => settings.normalizeSettings() });
let query = {};
overrides.set('next/router', { useRouter: () => ({ query, push() {} }) });
overrides.set('lucide-react', componentStubs);
overrides.set('@/lib/hooks', {
  useCreateTestRun: () => ({}), useRunBenchmark: () => ({}), usePromptsByIds: () => ({ data: [] }),
  useCreateSuite: () => ({}), useBenchmarks: () => ({ data: [] }), useEditedModels: () => ({ data: [] }),
});
overrides.set('@/lib/toast', { toast: {} });
const { RoleFields, DoctorGoalField } = load(path.join(frontend, 'components/forms/create-test/RoleStep'));
let edited;
const field = DoctorGoalField({ value: 'Exact goal', onChange: value => { edited = value; } });
assert.match(renderToStaticMarkup(field), /Sent only to the doctor as additional user instructions/);
find(field, n => n.type === 'textarea').props.onChange({ target: { value: '  New goal\n' } });
assert.equal(edited, '  New goal\n');
overrides.set('@/components/forms/create-test/RoleStep', { RoleFields, DoctorGoalField });
const { CreateTestForm } = load(path.join(frontend, 'components/forms/create-test/CreateTestForm'));
const { SuiteForm } = load(path.join(frontend, 'components/forms/SuiteForm'));
for (const Form of [CreateTestForm, SuiteForm]) for (const type of ['conversation', 'group_therapy', 'one_shot', 'multi_shot']) {
  slots = []; cleanups.clear(); query = { type, test_config: JSON.stringify({ doctor_goal: 'Restored goal' }) };
  let tree = settle(Form, { query });
  const doctor = find(tree, n => n.type === RoleFields && n.props.role === 'doctor');
  assert.ok(doctor, type);
  const applies = ['conversation', 'group_therapy'].includes(type);
  assert.equal(!!doctor.props.doctorGoal, applies);
  if (applies) {
    assert.equal(doctor.props.doctorGoal.value, 'Restored goal');
    doctor.props.doctorGoal.onChange('Updated goal'); tree = settle(Form, { query });
    assert.equal(find(tree, n => n.type === RoleFields && n.props.role === 'doctor').props.doctorGoal.value, 'Updated goal');
  }
}
""")
