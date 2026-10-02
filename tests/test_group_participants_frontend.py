"""Group identities follow recorded lineage and stay readable in every transcript view."""

from tests.test_conversation_frontend_identity import run_frontend


FIXTURES = r"""
const { buildGroupParticipants, participantForTurn, participantTitle, compactModelName } = load(path.join(frontend, 'lib/group-participants'));
const { participantAccent } = load(path.join(frontend, 'components/conversation/participant-colors'));
const models = ['org/base', '/models/no-folding', '/models/no-baking', '/models/edited', '/models/restored'];
const patients = models.map((model, id) => ({ id, provider: 'transformers', model }));
const surgery = [null,
  { source_model: models[0], method: 'lora_merge' },
  { source_model: models[0], method: 'lora_merge' },
  { source_model: models[0], method: 'direction_subspace' },
  { source_model: models[3], method: 'lora_merge' },
];
const rows = patients.map((patient, i) => ({ ...turn, id: i + 1, model_name: patient.model,
  model_provider: patient.provider, metadata: { patient_id: patient.id,
    patient_name: `Patient ${i + 1} (${patient.model})`, generation_metadata: { surgery: surgery[i] } } }));
const entries = patients.map((patient, i) => ({ model_ref: patient.model, provider: patient.provider,
  kind: i === 0 ? 'base' : 'custom', source_model: surgery[i]?.source_model || null,
  manifest: surgery[i], aliases: [] }));
"""


def test_recorded_lineage_marks_only_original_baseline_and_not_a_modified_parent():
    run_frontend(FIXTURES + r"""
const participants = buildGroupParticipants(patients, rows);
assert.deepEqual(participants.map(p => p.baseline), [true, false, false, false, false]);
assert.deepEqual(participants.map(p => p.name), ['org/base', 'no-folding', 'no-baking', 'edited', 'restored']);
assert.equal(new Set(participants.map(participantAccent)).size, 5);
assert.equal(participants[3].method, 'Direction subspace edit');
assert.equal(participants[4].sourceModel, models[3]);
assert.equal(participantTitle(participants[0]), 'Patient 1 · Baseline · org/base');
assert.equal(participantTitle(participants[4]), 'Patient 5 · restored');
// An old three-patient run needs no live catalog to establish its baseline.
assert.deepEqual(buildGroupParticipants(patients.slice(0, 3), rows.slice(0, 3)).map(p => p.baseline), [true, false, false]);
// Pending runs can use catalog lineage before any generation has completed.
assert.deepEqual(buildGroupParticipants(patients, [], entries).map(p => p.baseline), [true, false, false, false, false]);
// A name or a position alone is not proof that this is a baseline.
assert(buildGroupParticipants(patients).every(p => !p.baseline));
assert.equal(buildGroupParticipants([patients[0]], [rows[0]])[0].baseline, false);
""")


def test_lineage_aliases_providers_cycles_and_generation_time_evidence_are_respected():
    run_frontend(FIXTURES + r"""
const aliased = entries.map(e => ({ ...e }));
aliased[0].aliases = ['/cache/snapshots/base'];
aliased[1].source_model = '/cache/snapshots/base';
assert.equal(buildGroupParticipants(patients, [], aliased)[0].baseline, true);
// A similarly named base from another provider must not receive a baseline badge.
const otherProvider = patients.map((p, i) => i === 0 ? { ...p, provider: 'ollama' } : p);
assert(buildGroupParticipants(otherProvider, [], entries).every(p => !p.baseline));
// A historical response's recorded source wins if today's catalog changed.
const staleCatalog = entries.map((e, i) => i === 1 ? { ...e, source_model: 'different-base' } : e);
assert.equal(buildGroupParticipants(patients, rows, staleCatalog)[1].sourceModel, models[0]);
const cycle = entries.map((e, i) => ({ ...e, kind: 'custom', source_model: models[(i + 1) % models.length] }));
assert(buildGroupParticipants(patients, [], cycle).every(p => !p.baseline));
assert.equal(compactModelName('C:\\models\\custom-checkpoint'), 'custom-checkpoint');
assert.equal(compactModelName('vendor/remote-checkpoint'), 'vendor/remote-checkpoint');
""")


def test_turn_identity_never_borrows_a_different_requested_checkpoint_or_doctor_role():
    run_frontend(FIXTURES + r"""
const participants = buildGroupParticipants(patients, rows);
assert.equal(participantForTurn(rows[2], participants).number, 3);
assert.equal(participantForTurn({ ...rows[2], model_name: 'unexpected-runtime-model' }, participants), undefined);
assert.equal(participantForTurn({ ...rows[2], model_provider: 'ollama' }, participants), undefined);
assert.equal(participantForTurn({ ...rows[2], speaker: 'doctor' }, participants), undefined);
assert.equal(participantForTurn({ ...rows[2], metadata: { patient_id: 0 } }, participants), undefined);
const duplicate = buildGroupParticipants([patients[0], { ...patients[0], id: 7 }], rows);
assert.equal(participantForTurn({ ...rows[0], metadata: {} }, duplicate), undefined);
assert.equal(participantForTurn({ ...rows[0], metadata: { patient_id: 7 } }, duplicate).number, 2);
""")


def test_group_roster_and_transcript_keep_short_names_exact_links_and_matching_accents():
    run_frontend(FIXTURES + r"""
const { GroupParticipantRoster } = load(path.join(frontend, 'components/conversation/GroupParticipantRoster'));
const { ConversationViewer } = load(path.join(frontend, 'components/conversation/ConversationViewer'));
const participants = buildGroupParticipants(patients, rows);
const roster = renderToStaticMarkup(React.createElement(GroupParticipantRoster, { participants }));
const transcript = renderToStaticMarkup(React.createElement(ConversationViewer, { turns: rows, participants }));
assert.match(roster, /Patient Models \(5\)/);
assert.equal((roster.match(/>Baseline</g) || []).length, 1);
assert.equal((transcript.match(/· Baseline ·/g) || []).length, 1);
for (const p of participants) {
  assert.ok(roster.includes(`>${p.name}</a>`), roster);
  assert.ok(transcript.includes(participantTitle(p)), transcript);
  for (const accent of participantAccent(p).split(' ')) {
    assert.ok(roster.includes(accent));
    assert.ok(transcript.includes(accent));
  }
  assert.ok(roster.includes(`model=${encodeURIComponent(p.model)}`));
  assert.ok(transcript.includes(`model=${encodeURIComponent(p.model)}`));
  assert.ok(roster.includes(`<p class="mt-1 break-all">${p.model}</p>`));
}
assert.match(roster, /From edited/);
assert.doesNotMatch(transcript, /Patient 5 \(\/models\/restored\)/);
""")


def test_group_roster_is_present_in_overview_live_and_conversation():
    run_frontend(FIXTURES + r"""
const viewer = load(path.join(frontend, 'components/conversation/ConversationViewer'));
const roster = load(path.join(frontend, 'components/conversation/GroupParticipantRoster'));
overrides.set('@/components/conversation/ConversationViewer', viewer);
overrides.set('@/components/conversation/GroupParticipantRoster', roster);
overrides.set('@/components/conversation/participant-colors', { participantAccent });
overrides.set('@/lib/model-catalog', { useModelCatalog: () => ({ data: { models: entries } }) });
stubComponents = true;
overrides.set('next/router', { useRouter: () => ({ query: { id: '18' } }) });
overrides.set('next/link', { __esModule: true, default: Child });
overrides.set('@radix-ui/react-tabs', componentStubs);
overrides.set('lucide-react', componentStubs);
overrides.set('@tanstack/react-query', { useQueryClient: () => ({}), useQuery: () => ({}) });
overrides.set('@/lib/benchmark-campaigns', { useBenchmarkCampaign: () => ({}) });
overrides.set('@/lib/api', { apiClient: {} });
overrides.set('@/lib/toast', { toast: {} });
overrides.set('@/lib/hooks', {
  useTestRun: () => ({ data: { id: 18, test_type: 'group_therapy', status: 'running',
    doctor_model: 'doctor', patient_model: models[0], meta_data: { patients } } }),
  useConversation: () => ({ data: rows }), useTestResults: () => ({ data: [] }), useAssessments: () => ({ data: [] }),
  useRunAnalysis: () => ({}), useStartTestRun: () => ({}), usePauseTestRun: () => ({}), useResumeTestRun: () => ({}),
  useStopTestRun: () => ({}), useDeleteTestRun: () => ({}), useTestRunProgress: () => ({}),
});
const Page = load(path.join(frontend, 'pages/test-runs/[id]')).default;
const html = renderToStaticMarkup(React.createElement(Page));
assert.equal((html.match(/aria-label="Group participants"/g) || []).length, 3);
assert.equal((html.match(/Patient 1 · Baseline · org\/base/g) || []).length, 2, 'Live and Conversation both identify the baseline reply');
for (const p of buildGroupParticipants(patients, rows)) assert.ok(html.includes(participantTitle(p)));
""")


def test_tailwind_emits_the_participant_colors_using_the_project_scan_paths():
    run_frontend(FIXTURES + r"""
process.chdir(frontend);
const postcss = requireFrontend('postcss');
const tailwind = requireFrontend('tailwindcss');
const config = requireFrontend('./tailwind.config.js');
postcss([tailwind(config)]).process('@tailwind utilities;', { from: path.join(frontend, 'styles/globals.css') }).then(result => {
  for (const patient of buildGroupParticipants(patients, rows)) {
    for (const className of participantAccent(patient).split(' ')) {
      assert.ok(result.css.includes('.' + className.replace(/:/g, '\\:')), `Missing generated style: ${className}`);
    }
  }
}).catch(error => { console.error(error); process.exitCode = 1; });
""")
