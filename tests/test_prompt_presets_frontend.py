"""Run the actual prompt-library filters, routing, and editor submit handlers."""

from tests.test_weight_frontend_regressions import run_frontend


HARNESS = r"""
stubComponents = true;
const React = requireFrontend('react');
let slots = [], cursor = 0, effects = [];
overrides.set('react', { ...React,
  useState(initial) {
    const index = cursor++;
    if (!(index in slots)) slots[index] = typeof initial === 'function' ? initial() : initial;
    return [slots[index], (value) => { slots[index] = typeof value === 'function' ? value(slots[index]) : value; }];
  },
  useMemo(compute) { return compute(); },
  useEffect(effect, deps) {
    const index = cursor++;
    if (!slots[index] || deps.some((dep, i) => !Object.is(dep, slots[index][i]))) effects.push(effect);
    slots[index] = deps;
  },
});
function render(Component) {
  for (let i = 0; i < 10; i++) {
    cursor = 0; effects = [];
    const tree = Component();
    if (!effects.length) return tree;
    for (const effect of effects) effect();
  }
  throw new Error('Effects did not settle');
}
function all(node, matches) {
  if (Array.isArray(node)) return node.flatMap((child) => all(child, matches));
  if (!node || typeof node !== 'object') return [];
  return [...(matches(node) ? [node] : []), ...all(node.props?.children, matches)];
}
const find = (node, matches) => all(node, matches)[0];
const textOf = (node) => typeof node === 'string' ? node : Array.isArray(node)
  ? node.map(textOf).join('') : node?.props ? textOf(node.props.children) : '';
const change = (node, value) => node.props.onChange({ target: { value } });
let copied, submitted;
Object.defineProperty(global, 'navigator', { configurable: true, value: {
  clipboard: { writeText: async (text) => { copied = text; } },
}});
const router = { query: { id: '19' }, push() {}, back() {} };
overrides.set('next/router', { useRouter: () => router });
overrides.set('next/link', { __esModule: true, default: 'Link' });
overrides.set('lucide-react', componentStubs);
overrides.set('@/lib/toast', { toast: { success() {}, error(message) { throw new Error(message); } } });
overrides.set('@/lib/utils', { formatDate: () => '', getPromptDisplayName: (prompt) => prompt.name });
"""


def test_library_role_filters_and_actions_do_not_send_other_roles_to_patient():
    run_frontend(HARNESS + r"""
const prompts = [];
for (const target of ['doctor', 'patient', 'evaluator']) {
  for (const prompt_type of ['system_prompt', 'test_prompt']) {
    prompts.push({ id: prompts.length + 1, name: `${target} ${prompt_type}`,
      target, prompt_type, prompt_text: `  exact ${target} text\n"quoted" & values`,
      category: 'role_context_goals_v1', tags: ['role-presets-v1', target], usage_count: 0 });
  }
}
prompts.push({ id: 7, name: 'Legacy general user prompt', target: null, prompt_type: 'test_prompt',
  prompt_text: 'old general question', tags: [], usage_count: 0 });
overrides.set('@/lib/hooks', { usePrompts: () => ({ data: prompts }), useDeletePrompt: () => ({}) });
const Page = load(path.join(frontend, 'pages/prompts/index')).default;
const config = load(path.join(frontend, 'lib/create-test-config'));
const { DEFAULT_SETTINGS } = load(path.join(frontend, 'lib/settings'));
const cards = (tree) => all(tree, (node) => node.type?.name === 'PromptCard');
let tree = render(Page);
assert.equal(cards(tree).length, 7, 'new probe category must remain visible by default');
assert.ok(find(tree, (node) => node.props?.['aria-label'] === 'Prompt target filter'));

(async () => {
  for (const target of ['doctor', 'patient', 'evaluator']) {
    change(find(tree, (node) => node.props?.['aria-label'] === 'Prompt target filter'), target);
    tree = render(Page);
    assert.deepEqual(cards(tree).map((node) => node.props.prompt.target), [target, target]);
    change(find(tree, (node) => node.props?.['aria-label'] === 'Prompt type filter'), 'test_prompt');
    tree = render(Page);
    const [card] = cards(tree);
    assert.equal(cards(tree).length, 1);
    const cardTree = card.type(card.props);
    const prompt = card.props.prompt;
    const testLinks = all(cardTree, (node) => node.type === 'Link' && node.props.href.startsWith('/create-test'));
    if (target === 'doctor') {
      assert.equal(testLinks.length, 2);
      for (const link of testLinks) {
        const url = new URL(link.props.href, 'http://local');
        assert.equal(url.searchParams.get('type'), 'conversation');
        assert.equal(url.searchParams.has('promptId'), false);
        assert.deepEqual(JSON.parse(url.searchParams.get('test_config')), { doctor_goal: prompt.prompt_text });
        const state = config.initialFormState(DEFAULT_SETTINGS, Object.fromEntries(url.searchParams)).state;
        assert.equal(state.testType, 'conversation');
        assert.equal(state.doctorGoal, prompt.prompt_text);
        assert.equal(config.buildTestRunRequest(state).test_config.doctor_goal, prompt.prompt_text);
      }
    } else if (target === 'patient') {
      assert.equal(testLinks.length, 2);
      assert.ok(testLinks.every((link) => link.props.href === `/create-test?promptId=${prompt.id}`));
    } else {
      assert.equal(testLinks.length, 0, 'evaluator user text must not become a patient question');
      const copy = find(cardTree, (node) => node.type === 'button' && textOf(node).includes('Copy evaluator user prompt'));
      await copy.props.onClick();
      assert.equal(copied, prompt.prompt_text);
    }
    change(find(tree, (node) => node.props?.['aria-label'] === 'Prompt type filter'), '');
    tree = render(Page);
  }
  change(find(tree, (node) => node.props?.['aria-label'] === 'Prompt target filter'), '');
  tree = render(Page);
  const legacy = cards(tree).find((node) => node.props.prompt.id === 7);
  assert.ok(all(legacy.type(legacy.props), (node) => node.type === 'Link')
    .some((link) => link.props.href === '/create-test?promptId=7'));
})().catch((error) => { console.error(error); process.exitCode = 1; });
""")


def test_create_and_edit_user_prompts_retain_the_selected_role():
    run_frontend(HARNESS + r"""
const prompt = { id: 19, name: 'Existing evaluator user preset', description: 'My edits',
  prompt_text: '  preserve exact text\n', prompt_type: 'test_prompt', target: 'evaluator',
  category: 'role_context_goals_v1', tags: ['role-presets-v1'], metadata: { preset_id: 'stable' } };
overrides.set('@/lib/hooks', {
  usePrompt: () => ({ data: prompt }),
  useCreatePrompt: () => ({ mutateAsync: async (request) => { submitted = request; } }),
  useUpdatePrompt: () => ({ mutateAsync: async (request) => { submitted = request; } }),
});
const Create = load(path.join(frontend, 'pages/prompts/create')).default;
const Edit = load(path.join(frontend, 'pages/prompts/[id]/edit')).default;
const typeSelect = (tree) => find(tree, (node) => node.type === 'select' &&
  all(node, (child) => child.type === 'option').some((option) => option.props.value === 'system_prompt'));
const targetSelect = (tree) => find(tree, (node) => node.props?.['aria-label'] === 'Prompt target');

(async () => {
  for (const role of ['doctor', 'patient', 'evaluator']) {
    slots = []; let tree = render(Create);
    assert.equal(targetSelect(tree).props.required, false);
    change(targetSelect(tree), role); tree = render(Create);
    change(find(tree, (node) => node.props?.placeholder === 'Enter prompt name'), 'User role probe'); tree = render(Create);
    change(find(tree, (node) => node.props?.placeholder === 'Enter the prompt text'), '  exact user message\n'); tree = render(Create);
    // Changing type must retain a deliberately selected target in either direction.
    change(typeSelect(tree), 'system_prompt'); tree = render(Create);
    assert.equal(targetSelect(tree).props.required, true);
    assert.equal(targetSelect(tree).props.value, role);
    change(typeSelect(tree), 'test_prompt'); tree = render(Create);
    assert.equal(targetSelect(tree).props.value, role);
    await find(tree, (node) => node.type === 'form').props.onSubmit({ preventDefault() {} });
    assert.equal(submitted.target, role);
    assert.equal(submitted.prompt_type, 'test_prompt');
    assert.equal(submitted.prompt_text, '  exact user message\n');
  }
  slots = []; let tree = render(Edit);
  assert.equal(targetSelect(tree).props.value, 'evaluator');
  await find(tree, (node) => node.type === 'form').props.onSubmit({ preventDefault() {} });
  assert.equal(submitted.id, 19);
  assert.equal(submitted.data.target, 'evaluator', 'editing must not silently strip user-prompt targets');
  assert.equal(submitted.data.prompt_text, prompt.prompt_text);
  assert.equal(submitted.data.category, prompt.category);
  assert.equal('metadata' in submitted.data, false, 'unspecified catalog identity must stay untouched');
  change(targetSelect(tree), ''); tree = render(Edit);
  await find(tree, (node) => node.type === 'form').props.onSubmit({ preventDefault() {} });
  assert.equal(submitted.data.target, null, 'explicitly clearing the role remains supported');
})().catch((error) => { console.error(error); process.exitCode = 1; });
""")
