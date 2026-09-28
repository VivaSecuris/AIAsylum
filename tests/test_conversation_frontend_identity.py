"""Render conversation identities and exercise role-scoped prompt selection."""
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
const React = requireFrontend('react');
const { renderToStaticMarkup } = requireFrontend('react-dom/server');
const overrides = new Map(), cache = new Map();
let stubComponents = false;
const Child = ({ children }) => React.createElement('div', null, children);
const componentStubs = new Proxy({}, { get: (_, name) => name === '__esModule' ? true : Child });
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
function find(node, predicate) {
  if (Array.isArray(node)) return node.map((child) => find(child, predicate)).find(Boolean);
  if (!node || typeof node !== 'object') return undefined;
  return predicate(node) ? node : find(node.props?.children, predicate);
}
const turn = { id: 1, test_run_id: 18, turn_number: 1, speaker: 'patient', prompt: 'How are you?', response: 'I am well.' };
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


def test_transcript_uses_actual_turn_identity_and_only_patient_legacy_fallback():
    run_frontend(r"""
const { ConversationViewer, turnModelLabel } = load(path.join(frontend, 'components/conversation/ConversationViewer'));
const cases = [
  [{ ...turn, speaker: 'doctor', model_name: 'actual-doctor', model_provider: 'anthropic', metadata: { patient_model: 'wrong-patient' } }, 'anthropic / actual-doctor'],
  [{ ...turn, model_name: 'actual-patient', model_provider: 'ollama', metadata: { patient_model: 'outdated-patient', patient_name: 'Alice' } }, 'ollama / actual-patient'],
  [{ ...turn, metadata: { patient_model: 'legacy-patient', patient_name: 'Alice' } }, 'legacy-patient'],
  [{ ...turn, model_name: null, model_provider: null }, ''],
  [{ ...turn, speaker: 'doctor', metadata: { patient_model: 'wrong-patient' } }, ''],
  [{ ...turn, speaker: 'doctor', model_name: 'model-only' }, 'model-only'],
];
for (const [row, identity] of cases) {
  assert.equal(turnModelLabel(row), identity);
  const html = renderToStaticMarkup(React.createElement(ConversationViewer, { turns: [row] }));
  if (identity) assert.ok(html.includes(identity), html);
  assert.doesNotMatch(html, /wrong-patient|outdated-patient|undefined|null/);
  // The role/name remains the title; the model identity is a separate line.
  if (row.speaker === 'doctor') assert.match(html, /Doctor<\/span>/);
  if (row.metadata?.patient_name) assert.match(html, /Alice<\/span>/);
}
""")


def test_model_links_preserve_provider_and_reference_without_guessing_legacy_identity():
    run_frontend(r"""
const { ModelChatLink } = load(path.join(frontend, 'components/models/ModelChatLink'));
const html = renderToStaticMarkup(React.createElement(ModelChatLink, {
  provider: 'transformers', model: '/models/a b/checkpoint', children: 'Checkpoint',
}));
assert.match(html, /href="\/models\/chat\?provider=transformers&amp;model=%2Fmodels%2Fa\+b%2Fcheckpoint"/);
const legacy = renderToStaticMarkup(React.createElement(ModelChatLink, { model: 'unknown-provider' }));
assert.doesNotMatch(legacy, /href=/);
assert.match(legacy, /unknown-provider/);
""")


def test_live_messages_show_each_responding_model_without_borrowing_patient_metadata():
    run_frontend(r"""
const viewer = load(path.join(frontend, 'components/conversation/ConversationViewer'));
overrides.set('@/components/conversation/ConversationViewer', viewer);
stubComponents = true;
overrides.set('next/router', { useRouter: () => ({ query: { id: '18' } }) });
overrides.set('next/link', { __esModule: true, default: Child });
overrides.set('@radix-ui/react-tabs', componentStubs);
overrides.set('lucide-react', componentStubs);
overrides.set('@tanstack/react-query', { useQueryClient: () => ({}), useQuery: () => ({}) });
overrides.set('@/lib/benchmark-campaigns', { useBenchmarkCampaign: () => ({}) });
overrides.set('@/lib/api', { apiClient: {} });
overrides.set('@/lib/toast', { toast: {} });
const rows = [
  { ...turn, speaker: 'doctor', model_name: 'actual-doctor', model_provider: 'anthropic', metadata: { patient_model: 'wrong-patient', request_system_prompts: ['Doctor request instructions.'], reasoning: 'Live doctor thoughts', reasoning_source: 'react' } },
  { ...turn, id: 2, model_name: 'actual-patient', model_provider: 'ollama', metadata: { patient_name: 'Alice', patient_model: 'outdated-patient', request_system_prompts: ['Patient request instructions.'] } },
  { ...turn, id: 3, speaker: 'doctor', metadata: { patient_model: 'wrong-patient', request_system_prompts: [] } },
  { ...turn, id: 4, metadata: { patient_model: 'legacy-patient' } },
];
overrides.set('@/lib/hooks', {
  useTestRun: () => ({ data: { id: 18, test_type: 'conversation', status: 'running', doctor_model: 'requested-doctor', patient_model: 'requested-patient', meta_data: {} } }),
  useConversation: () => ({ data: rows }), useTestResults: () => ({ data: [] }), useAssessments: () => ({ data: [] }),
  useRunAnalysis: () => ({}), useStartTestRun: () => ({}), usePauseTestRun: () => ({}), useResumeTestRun: () => ({}),
  useStopTestRun: () => ({}), useDeleteTestRun: () => ({}), useTestRunProgress: () => ({}),
});
const Page = load(path.join(frontend, 'pages/test-runs/[id]')).default;
const html = renderToStaticMarkup(React.createElement(Page));
const liveStart = html.indexOf('id="live-messages-container"');
assert.ok(liveStart > 0, 'The running page must render live messages');
// Restrict assertions to the Live container; the page also renders the transcript.
const container = html.slice(html.lastIndexOf('<div', liveStart));
let depth = 0, liveEnd = 0;
for (const tag of container.matchAll(/<\/?div\b[^>]*>/g)) {
  depth += tag[0].startsWith('</') ? -1 : 1;
  if (depth === 0) { liveEnd = tag.index + tag[0].length; break; }
}
assert.ok(liveEnd > 0);
const live = container.slice(0, liveEnd);
assert.match(live, /anthropic \/ actual-doctor<\//);
assert.match(live, /ollama \/ actual-patient<\//);
assert.match(live, /legacy-patient<\//);
assert.equal((live.match(/Input prompt:/g) || []).length, 4);
assert.doesNotMatch(live, /Replying to/);
assert.doesNotMatch(live, /wrong-patient|outdated-patient/);
assert.match(live, /Alice<\/span>/);
assert.equal((live.match(/System prompt sent/g) || []).length, 3);
assert.match(live, /Doctor request instructions\./);
assert.ok(live.indexOf('System prompt sent') < live.indexOf('Live doctor thoughts'));
assert.ok(live.indexOf('Live doctor thoughts') < live.indexOf('I am well.'));
assert.match(live, /Patient request instructions\./);
assert.equal((live.match(/No system message was supplied by the app\./g) || []).length, 1);
assert.ok(live.indexOf('Doctor request instructions.') < live.indexOf('ollama / actual-patient'));
assert.ok(live.indexOf('Patient request instructions.') > live.indexOf('ollama / actual-patient'));
""")


def test_transcript_shows_actual_request_system_prompts_and_distinguishes_empty_from_legacy():
    run_frontend(r"""
const { ConversationViewer, TurnSystemPrompts } = load(path.join(frontend, 'components/conversation/ConversationViewer'));
const doctor = 'You are the doctor.\n  Keep exact spacing <and> punctuation.';
const patient = 'You are the patient.';
const renderTurn = (row) => renderToStaticMarkup(React.createElement(ConversationViewer, { turns: [row] }));
const doctorHtml = renderTurn({ ...turn, speaker: 'doctor', metadata: { request_system_prompts: [doctor] } });
assert.match(doctorHtml, /System prompt sent/);
assert.match(doctorHtml, /You are the doctor\.\n  Keep exact spacing &lt;and&gt; punctuation\./);
assert.ok(!doctorHtml.includes(patient));
const patientHtml = renderTurn({ ...turn, metadata: { request_system_prompts: [patient] } });
assert.match(patientHtml, /System prompt sent/);
assert.ok(patientHtml.includes(patient));
assert.doesNotMatch(patientHtml, /You are the doctor/);
const empty = renderTurn({ ...turn, metadata: { request_system_prompts: [] } });
assert.match(empty, /System prompt sent/);
assert.match(empty, /No system message was supplied by the app\./);
assert.match(empty, /The provider or chat template may add its own default/);
const legacy = renderTurn({ ...turn, metadata: {} });
assert.doesNotMatch(legacy, /System prompt sent|No system message was supplied by the app/);
const multiple = renderToStaticMarkup(React.createElement(TurnSystemPrompts, { prompts: ['First message.', 'Second message.'] }));
assert.match(multiple, /System message 1/);
assert.match(multiple, /System message 2/);
assert.equal((multiple.match(/<pre /g) || []).length, 2);
assert.doesNotMatch(multiple, /<details[^>]* open/);
""")


def test_system_prompt_picker_rejects_wrong_role_selection_without_blocking_custom_text():
    run_frontend(r"""
overrides.set('next/link', { __esModule: true, default: Child });
const doctorPrompt = { id: 42, name: 'Doctor instructions', prompt_type: 'system_prompt', target: 'doctor', prompt_text: 'You are Dr Synthia, the psychiatrist.' };
const patientPrompt = { id: 55, name: 'Patient instructions', prompt_type: 'system_prompt', target: 'patient', prompt_text: 'Answer as the patient.' };
const testPrompt = { id: 66, name: 'A test', prompt_type: 'test_prompt', target: 'patient', prompt_text: 'Test question.' };
let selected = doctorPrompt;
overrides.set('@/lib/hooks', {
  usePromptDefaults: () => ({}), usePrompts: () => ({ data: [doctorPrompt, patientPrompt, testPrompt] }),
  usePrompt: () => ({ data: selected }),
});
const { SystemPromptPicker } = load(path.join(frontend, 'components/forms/create-test/SystemPromptPicker'));
let choice;
const props = { role: 'patient', value: { mode: 'library', id: 42 }, onChange: (value) => { choice = value; } };
for (const wrong of [doctorPrompt, testPrompt]) {
  selected = wrong;
  const tree = SystemPromptPicker({ ...props, value: { mode: 'library', id: wrong.id } });
  const html = renderToStaticMarkup(tree);
  assert.match(html, /is not a patient system prompt/);
  assert.doesNotMatch(html, /Edit a copy/);
  assert.ok(!html.includes(wrong.prompt_text), 'Do not preview a mismatched prompt as patient instructions');
  assert.equal(find(tree, (node) => node.type === 'option' && node.props.value === wrong.id).props.disabled, true);
}
selected = patientPrompt;
const valid = SystemPromptPicker({ ...props, value: { mode: 'library', id: 55 } });
assert.doesNotMatch(renderToStaticMarkup(valid), /is not a patient system prompt/);
find(valid, (node) => node.type === 'button' && node.props.children === 'Edit a copy').props.onClick();
assert.deepEqual(choice, { mode: 'custom', text: patientPrompt.prompt_text });
const custom = SystemPromptPicker({ ...props, value: { mode: 'custom', text: 'Deliberately chosen custom text.' } });
const textarea = find(custom, (node) => node.type === 'textarea');
assert.equal(textarea.props.value, 'Deliberately chosen custom text.');
textarea.props.onChange({ target: { value: 'Updated custom instructions.' } });
assert.deepEqual(choice, { mode: 'custom', text: 'Updated custom instructions.' });
""")


def test_generation_settings_disclose_behavior_changes_without_changing_selected_values():
    run_frontend(r"""
const { GenerationSettings } = load(path.join(frontend, 'components/forms/create-test/GenerationSettings'));
for (const react of [true, false]) for (const adaptive of [true, false]) {
  const value = { temperature: '0', top_p: '0.8', max_tokens: '256', enable_cot: react, use_dynamic_strategies: adaptive };
  const saved = JSON.stringify(value);
  let changed;
  const tree = GenerationSettings({ value, showCot: true, showStrategies: true, onChange: (next) => { changed = next; } });
  const html = renderToStaticMarkup(tree);
  const summary = renderToStaticMarkup(find(tree, (node) => node.type === 'summary'));
  assert.ok(summary.includes(`ReACT ${react ? 'on' : 'off'}`), summary);
  assert.ok(summary.includes(`adaptive strategies ${adaptive ? 'on' : 'off'}`), summary);
  assert.match(summary, /temperature 0/);
  assert.match(html, /Adds instructions to write reasoning text before the reply/);
  assert.match(html, /can change the behavior being tested/);
  assert.match(html, /does not expose the model’s internal computation or execute actions/);
  assert.equal(JSON.stringify(value), saved);
  assert.equal(changed, undefined, 'Rendering guidance must not change user settings');
  const checkboxes = [];
  function collect(node) {
    if (Array.isArray(node)) return node.forEach(collect);
    if (!node || typeof node !== 'object') return;
    if (node.type === 'input' && node.props.type === 'checkbox') checkboxes.push(node);
    collect(node.props?.children);
  }
  collect(tree);
  assert.deepEqual(checkboxes.map((node) => node.props.checked), [react, adaptive]);
  checkboxes[0].props.onChange({ target: { checked: !react } });
  assert.deepEqual(changed, { ...value, enable_cot: !react });
}
const hidden = renderToStaticMarkup(React.createElement(GenerationSettings, {
  value: { temperature: '', top_p: '', max_tokens: '', enable_cot: true, use_dynamic_strategies: true },
  showCot: false, showStrategies: false, onChange: () => {},
}));
assert.doesNotMatch(hidden, /ReACT|adaptive strategies|Adapt questioning/);
""")


def test_patient_prompt_guidance_distinguishes_system_instructions_from_user_framing():
    run_frontend(r"""
overrides.set('next/link', { __esModule: true, default: Child });
const libraryPrompt = { id: 55, name: 'Patient persona', prompt_type: 'system_prompt', target: 'patient', prompt_text: 'Keep this chosen persona.' };
overrides.set('@/lib/hooks', {
  usePromptDefaults: () => ({ data: {
    patient: { user_message_template: 'Question wrapper: {prompt}', interview_user_message_template: 'Interview wrapper: {prompt}' },
    doctor: { system_prompt: 'Doctor instructions.', assessment_instructions: 'Assess afterward.' },
    evaluator: { system_prompt: 'Evaluator instructions.' },
  } }),
  usePrompts: () => ({ data: [libraryPrompt] }),
  usePrompt: () => ({ data: libraryPrompt }),
});
const { SystemPromptPicker } = load(path.join(frontend, 'components/forms/create-test/SystemPromptPicker'));
for (const value of [{ mode: 'custom', text: 'Keep this chosen persona.' }, { mode: 'library', id: 55 }]) {
  const saved = JSON.stringify(value);
  let changed;
  const tree = SystemPromptPicker({ role: 'patient', value, allowNone: true, onChange: (next) => { changed = next; } });
  const html = renderToStaticMarkup(tree);
  assert.match(html, /Conversation and Group Therapy, interview framing is also added to each user message/);
  assert.match(html, /One-Shot and Multi-Shot send the test input without that framing when a system prompt is selected/);
  assert.match(html, /Show Conversation \/ Group Therapy interview framing/);
  assert.match(html, /Interview wrapper: \{prompt\}/);
  assert.match(html, /doctor’s message with its speaker attribution/);
  assert.ok(html.includes('Keep this chosen persona.'));
  assert.equal(JSON.stringify(value), saved);
  assert.equal(changed, undefined);
}
const render = (role, value) => renderToStaticMarkup(React.createElement(SystemPromptPicker, { role, value, allowNone: role === 'patient', onChange: () => {} }));
const framed = render('patient', { mode: 'default' });
assert.match(framed, /No app system \+ framing/);
assert.match(framed, /No app system prompt is selected/);
assert.match(framed, /chat template may add its own default system instructions/);
assert.match(framed, /Show the One-Shot \/ Multi-Shot wrapper/);
assert.match(framed, /Question wrapper: \{prompt\}/);
assert.match(framed, /Interview wrapper: \{prompt\}/);
const raw = render('patient', { mode: 'none' });
assert.match(raw, /No app system \+ raw input/);
assert.match(raw, /No app system prompt or patient framing/);
assert.match(raw, /chat template may add its own default system instructions/);
assert.match(raw, /Enabling ReACT adds its own reasoning instructions/);
assert.doesNotMatch(raw, /Question wrapper: \{prompt\}/);
assert.doesNotMatch(raw, /Interview wrapper: \{prompt\}/);
for (const role of ['doctor', 'evaluator']) {
  const html = render(role, { mode: 'custom', text: 'Chosen instructions.' });
  assert.doesNotMatch(html, /interview framing is also added|No app system \+ framing|No app system \+ raw input/);
}
""")


def test_live_tab_transitions_to_an_existing_panel_when_run_status_changes():
    run_frontend(r"""
// Run the real page effects with persistent state across API status updates.
// Only remote hooks and child widgets are replaced; the tab selection is real.
const viewer = load(path.join(frontend, 'components/conversation/ConversationViewer'));
overrides.set('@/components/conversation/ConversationViewer', viewer);
stubComponents = true;
let slots = [], cursor = 0, pendingEffects = [], cleanups = new Map();
overrides.set('react', {
  ...React,
  useState(initial) {
    const index = cursor++;
    if (!(index in slots)) slots[index] = typeof initial === 'function' ? initial() : initial;
    return [slots[index], (value) => { slots[index] = typeof value === 'function' ? value(slots[index]) : value; }];
  },
  useRef(initial) {
    const index = cursor++;
    if (!(index in slots)) slots[index] = { current: initial };
    return slots[index];
  },
  useEffect(effect, deps) {
    const index = cursor++;
    const previous = slots[index];
    if (!previous || deps.some((dep, i) => !Object.is(dep, previous[i]))) {
      pendingEffects.push(() => {
        cleanups.get(index)?.();
        const cleanup = effect();
        if (typeof cleanup === 'function') cleanups.set(index, cleanup);
        else cleanups.delete(index);
      });
    }
    slots[index] = deps;
  },
});
const tabs = { Root: 'TabRoot', List: 'TabList', Trigger: 'TabTrigger', Content: 'TabContent' };
const queryClient = { refetchQueries() {}, invalidateQueries() {} };
const rows = [], results = [], assessments = [];
let run = { id: 23, test_type: 'conversation', status: 'running', doctor_model: 'doctor', patient_model: 'patient', meta_data: {} };
overrides.set('next/router', { useRouter: () => ({ query: { id: String(run.id) } }) });
overrides.set('next/link', { __esModule: true, default: Child });
overrides.set('@radix-ui/react-tabs', tabs);
overrides.set('lucide-react', componentStubs);
overrides.set('@tanstack/react-query', { useQueryClient: () => queryClient, useQuery: () => ({}) });
overrides.set('@/lib/benchmark-campaigns', { useBenchmarkCampaign: () => ({}) });
overrides.set('@/lib/api', { apiClient: {} });
overrides.set('@/lib/toast', { toast: {} });
overrides.set('@/lib/hooks', {
  useTestRun: () => ({ data: run }), useConversation: () => ({ data: rows }),
  useTestResults: () => ({ data: results }), useAssessments: () => ({ data: assessments }),
  useRunAnalysis: () => ({}), useStartTestRun: () => ({}), usePauseTestRun: () => ({}), useResumeTestRun: () => ({}),
  useStopTestRun: () => ({}), useDeleteTestRun: () => ({}), useTestRunProgress: () => ({}),
});
global.document = { getElementById: () => null };
const Page = load(path.join(frontend, 'pages/test-runs/[id]')).default;
function settle() {
  for (let attempt = 0; attempt < 12; attempt++) {
    cursor = 0; pendingEffects = [];
    const tree = Page();
    if (!pendingEffects.length) return find(tree, (node) => node.type === 'TabRoot');
    for (const effect of pendingEffects) effect();
  }
  throw new Error('Page state did not settle');
}
function checkActive(expected) {
  const root = settle();
  assert.equal(root.props.value, expected);
  assert.ok(find(root, (node) => node.type === 'TabTrigger' && node.props.value === expected), 'Active tab must exist');
  assert.ok(find(root, (node) => node.type === 'TabContent' && node.props.value === expected), 'Active panel must exist');
  return root;
}
try {
  checkActive('live');
  rows.push(turn);
  run = { ...run, status: 'completed' };
  checkActive('conversation');
  // Repeat the same browser component transition for other statuses.
  for (const status of ['paused', 'failed', 'cancelled']) {
    run = { ...run, status: 'running' };
    checkActive('conversation').props.onValueChange('live');
    checkActive('live');
    run = { ...run, status };
    checkActive('conversation');
  }
  // A user's selected Results tab remains selected on completion.
  run = { ...run, status: 'running' };
  checkActive('conversation').props.onValueChange('results');
  run = { ...run, status: 'completed' };
  checkActive('results');
  // A failed run with no messages still shows Overview rather than a blank page.
  run = { ...run, status: 'running' }; rows.length = 0;
  checkActive('results').props.onValueChange('live');
  checkActive('live');
  run = { ...run, status: 'failed' };
  checkActive('overview');
  // One-shot completion has a doctor assessment even before transcript polling refreshes.
  run = { ...run, test_type: 'one_shot', status: 'running' };
  checkActive('live');
  run = { ...run, status: 'completed' };
  checkActive('conversation');
} finally {
  for (const cleanup of cleanups.values()) cleanup();
}
""")


def test_recorded_responses_show_truncation_and_keep_reasoning_only_evidence():
    run_frontend(r"""
const { ConversationViewer, TurnFinishStatus, reasoningLabelFor } = load(path.join(frontend, 'components/conversation/ConversationViewer'));
assert.equal(reasoningLabelFor('react'), 'Requested reasoning (ReACT)');
assert.equal(reasoningLabelFor('provider'), 'Model-provided reasoning');
assert.equal(reasoningLabelFor(undefined), 'Recorded reasoning (source not recorded)');
for (const reason of ['length', 'max_tokens', 'max_output_tokens']) {
  const html = renderToStaticMarkup(React.createElement(TurnFinishStatus, { reason }));
  assert.match(html, /Output reached its token limit/); assert.match(html, /may be incomplete/);
}
assert.equal(renderToStaticMarkup(React.createElement(TurnFinishStatus)), '');
assert.doesNotMatch(renderToStaticMarkup(React.createElement(TurnFinishStatus, { reason: 'stop' })), /token limit/);
const html = renderToStaticMarkup(React.createElement(ConversationViewer, { turns: [{ ...turn, response: '', model_name: 'actual', model_provider: 'ollama', metadata: { finish_reason: 'length', reasoning: 'Requested thoughts', reasoning_source: 'react', generation_metadata: { native_reasoning: 'Native thoughts' }, request_system_prompts: ['Exact patient instructions'] } }] }));
assert.match(html, /actual/); assert.match(html, /The model returned no answer text/);
assert.match(html, /Output reached its token limit/); assert.match(html, /Requested thoughts/);
assert.match(html, /Native thoughts/); assert.match(html, /Exact patient instructions/);
assert.ok(html.indexOf('System prompt sent') < html.indexOf('Requested thoughts'));
assert.ok(html.indexOf('System prompt sent') < html.indexOf('Native thoughts'));
assert.ok(html.indexOf('Requested thoughts') < html.indexOf('The model returned no answer text'));
const legacy = renderToStaticMarkup(React.createElement(ConversationViewer, { turns: [{ ...turn, response: 'Legacy response' }] }));
assert.match(legacy, /Legacy response/); assert.doesNotMatch(legacy, /token limit|System prompt sent/);
""")


def test_doctor_assessment_evidence_uses_actual_generation_and_legacy_text():
    run_frontend(r"""
const { DoctorAssessmentEvidence } = load(path.join(frontend, 'components/test-runs/DoctorAssessmentEvidence'));
const result = { id: 1, test_name: 'one_shot_test', analysis: 'Older content', meta_data: { doctor_assessment: { model_name: 'assessing-model', model_provider: 'ollama', response: 'Actual assessment', reasoning: 'Requested assessment reasoning', reasoning_source: 'react', request_system_prompts: ['Distinct doctor instructions'], request_system_prompts_source: 'provider', finish_reason: 'length', usage: { completion_tokens: 256 }, generation_metadata: { native_reasoning: 'Native assessment reasoning', sampling: { temperature: 0 } } } } };
const html = renderToStaticMarkup(React.createElement(DoctorAssessmentEvidence, { result }));
assert.match(html, /Doctor assessment/); assert.match(html, /ollama \/ assessing-model/);
assert.match(html, /Actual assessment/); assert.doesNotMatch(html, /Older content/);
assert.match(html, /Output reached its token limit/); assert.match(html, /Distinct doctor instructions/);
assert.match(html, /Requested assessment reasoning/); assert.match(html, /Native assessment reasoning/);
assert.match(html, /Assessment generation evidence/); assert.match(html, /temperature/);
assert.ok(html.indexOf('System prompt sent') < html.indexOf('Requested assessment reasoning'));
assert.ok(html.indexOf('Native assessment reasoning') < html.indexOf('Actual assessment'));
const legacy = renderToStaticMarkup(React.createElement(DoctorAssessmentEvidence, { result: { test_name: 'one_shot_test', analysis: 'Legacy assessment' } }));
assert.match(legacy, /Legacy assessment/); assert.match(legacy, /Generation evidence was not recorded/);
assert.doesNotMatch(legacy, /token limit|assessing-model|System prompt sent/);
assert.equal(renderToStaticMarkup(React.createElement(DoctorAssessmentEvidence, { result: { test_name: 'no-assessment' } })), '');
""")


def test_input_labels_require_a_recorded_preceding_speaker_not_a_role_guess():
    run_frontend(r"""
const { ConversationViewer, turnPromptLabel } = load(path.join(frontend, 'components/conversation/ConversationViewer'));
const doctor = { ...turn, id: 1, speaker: 'doctor', prompt: 'Begin the interview.', response: 'What music do you enjoy?' };
const patient = { ...turn, id: 2, speaker: 'patient', prompt: doctor.response, response: 'I enjoy jazz.' };
const followup = { ...doctor, id: 3, prompt: patient.response, response: 'What do you enjoy about it?' };
assert.equal(turnPromptLabel({ ...patient, prompt: 'A manually written one-shot prompt.' }), 'Input prompt');
assert.equal(turnPromptLabel(doctor), 'Input prompt', 'The first doctor input is not patient speech');
assert.equal(turnPromptLabel(patient, doctor), 'Replying to 👨‍⚕️ Doctor');
assert.equal(turnPromptLabel(followup, patient), 'Replying to 🤖 Patient');
assert.equal(turnPromptLabel(followup, { ...patient, metadata: { patient_name: 'Alice' } }), 'Replying to 🤖 Alice');
assert.equal(turnPromptLabel({ ...patient, prompt: 'A different supplied prompt.' }, doctor), 'Input prompt');
const oneShot = renderToStaticMarkup(React.createElement(ConversationViewer, { turns: [{ ...patient, prompt: 'A custom test question.' }] }));
assert.match(oneShot, /Input prompt:/); assert.doesNotMatch(oneShot, /Replying to/);
const html = renderToStaticMarkup(React.createElement(ConversationViewer, { turns: [doctor, patient, followup] }));
assert.equal((html.match(/Input prompt:/g) || []).length, 1);
assert.equal((html.match(/Replying to 👨‍⚕️ Doctor:/g) || []).length, 1);
assert.equal((html.match(/Replying to 🤖 Patient:/g) || []).length, 1);
""")


def test_doctor_comments_are_open_in_live_and_conversation_with_legacy_and_pending_states():
    run_frontend(r"""
const evidence = load(path.join(frontend, 'components/test-runs/DoctorAssessmentEvidence'));
const modern = { id: 20, test_name: 'one_shot_test', analysis: 'Modern comments', meta_data: { doctor_assessment: { response: 'Actual doctor comments', request_system_prompts: ['Exact doctor system'], reasoning: 'Doctor thoughts', reasoning_source: 'react' } } };
const legacy = { id: 18, test_name: 'one_shot_test', analysis: 'Legacy doctor comments' };
for (const result of [modern, legacy]) {
  const html = renderToStaticMarkup(React.createElement(evidence.DoctorAssessmentPanel, { results: [result] }));
  assert.match(html, /<details[^>]* open=""/);
  assert.match(html, result === modern ? /Actual doctor comments/ : /Legacy doctor comments/);
}
const pending = renderToStaticMarkup(React.createElement(evidence.DoctorAssessmentPanel, { results: [], pending: true }));
assert.match(pending, /Doctor assessment will appear here when complete/);
assert.doesNotMatch(pending, /generating|thinking/i);
assert.equal(renderToStaticMarkup(React.createElement(evidence.DoctorAssessmentPanel, { results: [] })), '');
overrides.set('@/components/test-runs/DoctorAssessmentEvidence', evidence);
overrides.set('@/components/conversation/ConversationViewer', load(path.join(frontend, 'components/conversation/ConversationViewer')));
stubComponents = true;
overrides.set('next/router', { useRouter: () => ({ query: { id: '28' } }) });
overrides.set('next/link', { __esModule: true, default: Child });
overrides.set('@radix-ui/react-tabs', { ...componentStubs, Root: 'TabRoot', List: 'TabList', Trigger: 'TabTrigger', Content: 'TabContent' });
overrides.set('lucide-react', componentStubs);
overrides.set('@tanstack/react-query', { useQueryClient: () => ({}), useQuery: () => ({}) });
overrides.set('@/lib/benchmark-campaigns', { useBenchmarkCampaign: () => ({}) });
overrides.set('@/lib/api', { apiClient: {} });
overrides.set('@/lib/toast', { toast: {} });
let results = [modern];
overrides.set('@/lib/hooks', {
  useTestRun: () => ({ data: { id: 28, test_type: 'one_shot', status: 'running', doctor_model: 'd', patient_model: 'p', meta_data: {} } }),
  useConversation: () => ({ data: [turn] }), useTestResults: () => ({ data: results }), useAssessments: () => ({ data: [] }),
  useRunAnalysis: () => ({}), useStartTestRun: () => ({}), usePauseTestRun: () => ({}), useResumeTestRun: () => ({}),
  useStopTestRun: () => ({}), useDeleteTestRun: () => ({}), useTestRunProgress: () => ({}),
});
const Page = load(path.join(frontend, 'pages/test-runs/[id]')).default;
// Render the actual page with React, then assert each tab's doctor section is wired.
for (const current of [[modern], [legacy], []]) {
  results = current;
  const html = renderToStaticMarkup(React.createElement(Page));
  for (const tab of ['live', 'conversation']) {
    const start = html.indexOf('<TabContent value="' + tab + '"');
    assert.ok(start >= 0, tab);
    const panel = html.slice(start, html.indexOf('</TabContent>', start));
    assert.match(panel, /aria-label="Doctor assessment"/);
    assert.match(panel, current.length ? current[0] === modern ? /Actual doctor comments/ : /Legacy doctor comments/ : /Doctor assessment will appear here when complete/);
    if (current.length) assert.match(panel, /<details[^>]* open=""/);
  }
}
""")
