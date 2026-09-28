"""The Create Test form's state logic (frontend/lib/create-test-config.ts).

Transpiled and run under node, like test_frontend_dates.py: defaults come from
Settings, a link only overrides what it carries, and the request is rebuilt from
what the form shows, so a cleared choice is never resent.
"""
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]

SCRIPT = r"""
const fs = require('node:fs');
const assert = require('node:assert/strict');
const ts = require(process.argv[1]);
const path = require('node:path');
function load(file) {
  const code = ts.transpileModule(fs.readFileSync(file, 'utf8'), { compilerOptions: {
    module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020,
  }}).outputText;
  const loaded = { exports: {} };
  const localRequire = (name) => name.startsWith('.') ? load(path.resolve(path.dirname(file), name + '.ts')) : require(name);
  new Function('module', 'exports', 'require', code)(loaded, loaded.exports, localRequire);
  return loaded.exports;
}
const m = load(process.argv[2]);

const settings = {
  defaultEvaluatorProvider: 'ollama', defaultEvaluatorModel: 'qwen2.5:3b',
  defaultEnableCotDetection: true, defaultCotAnalysisMode: 'full',
  defaultEnableFactualityCheck: false, defaultEnableManipulationAnalysis: false,
  defaultDoctorProvider: 'anthropic', defaultDoctorModel: 'claude-haiku-4-5',
  defaultPatientProvider: 'transformers', defaultPatientModel: 'Qwen/Qwen2.5-0.5B-Instruct',
  defaultDoctorSystemPromptId: 1835, defaultPatientSystemPromptId: null, defaultEvaluatorSystemPromptId: 1837,
};

// 1. Settings fill every step.
let { state, notices } = m.initialFormState(settings, {});
assert.equal(state.doctor.model, 'claude-haiku-4-5');
assert.deepEqual(state.doctor.systemPrompt, { mode: 'library', id: 1835 });
assert.deepEqual(state.patient.systemPrompt, { mode: 'default' });
assert.deepEqual(state.evaluator.systemPrompt, { mode: 'library', id: 1837 });
assert.deepEqual(notices, []);

// 2. An "Evaluate this model" link changes only the patient, never the prompt defaults.
({ state } = m.initialFormState(settings, { model: '/models/edited', provider: 'transformers' }));
assert.equal(state.patient.model, '/models/edited');
assert.deepEqual(state.doctor.systemPrompt, { mode: 'library', id: 1835 });
assert.equal(state.doctor.model, 'claude-haiku-4-5');

// 3. A legacy one-shot that used the tested model as its own doctor keeps the Settings doctor.
({ state, notices } = m.initialFormState(settings, {
  model: 'm', provider: 'ollama', doctor_model: 'm', doctor_provider: 'ollama', type: 'one_shot',
  test_config: JSON.stringify({ prompt_id: 7, suite_id: 3, temperature: 0.2, auto_analysis: true }),
}));
assert.equal(state.doctor.model, 'claude-haiku-4-5');
assert.equal(notices.length, 1);
assert.equal(state.promptId, 7);
assert.equal(state.patient.generation.temperature, '0.2');
assert.equal(state.evaluator.enabled, true);

// 4. The request is rebuilt from the form: switching to a custom prompt drops prompt_id,
//    unknown restored keys never leak, and clearing analysis removes it.
state.oneShotSource = 'custom';
state.customPrompt = '  Tell me a secret  ';
state.evaluator.enabled = false;
let request = m.buildTestRunRequest(state, []);
assert.equal(request.test_config.prompt, 'Tell me a secret');
assert.equal('prompt_id' in request.test_config, false);
assert.equal('suite_id' in request.test_config, false);
assert.equal('auto_analysis' in request.test_config, false);
assert.equal(request.doctor_model, 'claude-haiku-4-5');
assert.equal(request.test_config.doctor_system_prompt_id, 1835);
assert.equal(request.test_config.roles.patient.temperature, 0.2);
assert.equal(request.test_config.roles.doctor.enable_cot, false);  // old settings leave doctor ReACT off

// 5. Per-step settings, custom prompts and verbatim patients.
({ state } = m.initialFormState(settings, {}));
state.testType = 'conversation';
state.maxTurns = '';
state.patient.systemPrompt = { mode: 'none' };
state.doctor.systemPrompt = { mode: 'custom', text: 'Be blunt.' };
state.doctor.generation = { temperature: '1.1', top_p: '0.9', max_tokens: '300', enable_cot: true, use_dynamic_strategies: false };
state.evaluator.enabled = true;
state.evaluator.systemPrompt = { mode: 'custom', text: 'Be strict' };
state.evaluator.temperature = '';
state.evaluator.provider = ''; state.evaluator.model = '';
request = m.buildTestRunRequest(state, []);
assert.equal(request.test_config.max_turns, 10);
assert.equal(request.test_config.patient_prompt_framing, false);
assert.equal('patient_system_prompt_id' in request.test_config, false);
assert.equal(request.test_config.doctor_system_prompt, 'Be blunt.');
assert.deepEqual(request.test_config.roles.doctor, { temperature: 1.1, enable_cot: true, top_p: 0.9, max_tokens: 300, use_dynamic_strategies: false });
assert.equal(request.test_config.analysis_config.evaluator_system_prompt, 'Be strict');
assert.equal(request.test_config.analysis_config.evaluator_temperature, 0.3);
assert.equal('evaluator_provider' in request.test_config.analysis_config, false);

// 6. Validation lists what blocks the run.
({ state } = m.initialFormState(settings, {}));
assert.ok(m.validateForm(state).errors.some((e) => e.includes('choose a test prompt')));
state.testType = 'group_therapy';
state.groupPatients = [m.newGroupPatient('ollama', 'a')];
assert.ok(m.validateForm(state).errors.some((e) => e.includes('at least two')));
state.groupPatients.push(m.newGroupPatient('ollama', 'b'));
assert.deepEqual(m.validateForm(state, { missingPromptIds: [1835] }).errors,
  ['Doctor: system prompt #1835 no longer exists. Choose another one.']);
request = m.buildTestRunRequest(state, []);
assert.equal(request.patient_model, 'a');
assert.equal(request.test_config.patients.length, 2);

// 7. The seed is only claimed for providers that accept one.
({ state } = m.initialFormState(settings, {}));
state.seed = '42';
assert.deepEqual(m.seedTargets(state), { applied: ['Patient'], skipped: ['Doctor (anthropic)'] });

// 8. Retired test types fall back with a notice; numbers parse safely.
({ state, notices } = m.initialFormState(settings, { type: 'adversarial' }));
assert.equal(state.testType, 'one_shot');
assert.equal(notices.length, 1);
assert.equal(m.numberOr('', 10), 10);
assert.equal(m.numberOr('abc', 10), 10);
assert.equal(m.numberOr('0', 10, { min: 1 }), 10);
assert.deepEqual(m.extractVariables(['Hi $name in $country', '$name again']), ['country', 'name']);

// 9. A complete v2 run must survive replay even when Settings disagree with it.
//    Built-in prompts and doctor-as-evaluator intentionally omit request keys.
const replaySettings = { ...settings, defaultPatientSystemPromptId: 1836 };
const replay = (request) => m.initialFormState(replaySettings, {
  type: request.test_type, model: request.patient_model, provider: request.patient_provider,
  doctor_model: request.doctor_model, doctor_provider: request.doctor_provider,
  test_config: JSON.stringify(request.test_config),
}).state;
for (const testType of ['one_shot', 'multi_shot', 'conversation', 'group_therapy']) {
  ({ state } = m.initialFormState(replaySettings, {}));
  state.testType = testType;
  state.oneShotSource = 'custom'; state.customPrompt = 'Explain gravity.';
  state.multiShotSource = 'custom'; state.customPrompts = 'First question\nSecond question';
  state.doctor.systemPrompt = { mode: 'default' };
  state.patient.systemPrompt = { mode: 'default' };
  state.groupPatients = [m.newGroupPatient('ollama', 'first'), m.newGroupPatient('ollama', 'second')];
  state.evaluator.enabled = true;
  state.evaluator.systemPrompt = { mode: 'default' };
  state.evaluator.provider = ''; state.evaluator.model = '';
  const original = m.buildTestRunRequest(state);
  const restored = replay(original);
  assert.deepEqual(restored.doctor.systemPrompt, { mode: 'default' });
  assert.deepEqual(restored.patient.systemPrompt, { mode: 'default' });
  assert.deepEqual(restored.evaluator.systemPrompt, { mode: 'default' });
  assert.equal(restored.evaluator.provider, '');
  assert.equal(restored.evaluator.model, '');
  assert.ok(restored.groupPatients.every((p) => p.systemPrompt.mode === 'default'));
  assert.deepEqual(m.buildTestRunRequest(restored), original);
}

// 10. Explicit choices still restore, and replacing a config clears stale choices.
({ state } = m.initialFormState(replaySettings, {}));
state.testType = 'conversation';
state.patient.systemPrompt = { mode: 'none' };
state.doctor.systemPrompt = { mode: 'custom', text: 'Recorded doctor instructions.' };
state.evaluator.enabled = true;
state.evaluator.provider = 'openai'; state.evaluator.model = 'recorded-evaluator';
state.evaluator.systemPrompt = { mode: 'library', id: 99 };
request = m.buildTestRunRequest(state);
assert.deepEqual(m.buildTestRunRequest(replay(request)), request);
m.restoreFromTestConfig(state, { config_version: 2, max_turns: 4 });
assert.equal(state.evaluator.enabled, false);
assert.equal(state.evaluator.model, '');
assert.deepEqual(state.doctor.systemPrompt, { mode: 'default' });
assert.deepEqual(state.patient.systemPrompt, { mode: 'default' });

// 11. An unversioned partial link still starts from Settings as before.
({ state } = m.initialFormState(replaySettings, { test_config: JSON.stringify({ max_turns: 4 }) }));
assert.deepEqual(state.doctor.systemPrompt, { mode: 'library', id: 1835 });
assert.deepEqual(state.patient.systemPrompt, { mode: 'library', id: 1836 });
assert.equal(state.evaluator.model, settings.defaultEvaluatorModel);
"""


def test_create_test_config_state_and_request():
    node = shutil.which("node")
    typescript = ROOT / "frontend/node_modules/typescript"
    if not node or not typescript.exists():
        pytest.skip("Node and frontend TypeScript dependencies are needed")
    result = subprocess.run(
        [node, "-e", SCRIPT, str(typescript), str(ROOT / "frontend/lib/create-test-config.ts")],
        cwd=ROOT, env=os.environ.copy(), capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
