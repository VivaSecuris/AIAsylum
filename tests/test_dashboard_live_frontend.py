"""Dashboard navigation, active polling, and stable assessment-driven widgets."""
from tests.test_conversation_frontend_identity import run_frontend


def test_dashboard_first_and_settings_outside_scrollable_navigation():
    run_frontend(r"""
overrides.set('next/router', { useRouter: () => ({ pathname: '/dashboard' }) });
overrides.set('next/image', { __esModule: true, default: () => null });
const { Sidebar } = load(path.join(frontend, 'components/layout/Sidebar'));
const html = renderToStaticMarkup(React.createElement(Sidebar));
const nav = html.match(/<nav\b[^>]*>([\s\S]*?)<\/nav>/)[0];
assert.match(nav.match(/<a\b[^>]*>/)[0], /href="\/dashboard"/);
assert.match(nav.match(/<a\b[^>]*>/)[0], /aria-current="page"/);
assert.match(nav, /min-h-0 flex-1.*overflow-y-auto/);
assert.doesNotMatch(nav, /Settings|Workspace/);
assert.ok(html.indexOf('href="/settings"') > html.indexOf('</nav>'));
assert.match(html, /shrink-0 border-t[^>]*><a[^>]*href="\/settings"/);
assert.equal((html.match(/href="\/settings"/g) || []).length, 1);
""")


def test_live_list_intervals_include_pending_and_other_active_lists():
    run_frontend(r"""
overrides.set('@tanstack/react-query', { useQuery: (options) => options });
const hooks = load(path.join(frontend, 'lib/hooks'));
const query = (status) => ({ state: { data: [{ status }] } });
for (const make of [hooks.useTestRuns, hooks.useSuites]) {
  assert.equal(make().refetchInterval(query('running')), false, 'Live polling is opt-in');
  assert.equal(Object.hasOwn(make(), 'refetchOnWindowFocus'), false, 'Other callers inherit the app focus policy');
  for (const status of ['running', 'pending']) assert.equal(make({}, { live: true }).refetchInterval(query(status)), 2500);
  for (const status of ['completed', 'failed', 'cancelled', 'paused']) {
    assert.equal(make({}, { live: true }).refetchInterval(query(status)), false);
    assert.equal(make({}, { live: true, active: true }).refetchInterval(query(status)), 2500, 'Poll both lists while either has active work');
  }
  assert.equal(make({}, { live: true }).refetchInterval({ state: {} }), false);
  assert.equal(make({}, { live: true }).refetchOnWindowFocus, true, 'Focus discovers work started elsewhere while idle');
}
""")


def test_assessment_queries_share_detail_cache_and_retain_results_on_partial_failures():
    run_frontend(r"""
const { QueryClient, QueryObserver } = requireFrontend('@tanstack/react-query');
const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: Infinity } } });
const prior = { id: 1, test_run_id: 1, overall_score: 0.8 };
client.setQueryData(['assessments', 1], [prior]);
let specs;
overrides.set('react', { ...React, useMemo: (fn) => fn() });
overrides.set('@tanstack/react-query', {
  useQueries: ({ queries }) => {
    specs = queries;
    return queries.map(({ queryKey }) => new QueryObserver(client, { queryKey }).getCurrentResult());
  },
});
overrides.set('./api', { apiClient: { getAssessments: async (id) => { throw new Error('temporary failure ' + id); } } });
const { useMultipleAssessments } = load(path.join(frontend, 'lib/hooks'));
(async () => {
  let result = useMultipleAssessments([1, 2, 1], { refetchInterval: 2500 });
  assert.equal(specs.length, 2);
  assert.deepEqual(result.data.get(1), [prior]);
  assert.deepEqual(result.data.get(2), []);
  assert.deepEqual(specs[0].queryKey, ['assessments', 1]);
  assert.equal(specs[0].refetchInterval, 2500);
  await assert.rejects(client.fetchQuery(specs[0]), /temporary failure/);
  result = useMultipleAssessments([1, 2], { refetchInterval: false });
  assert.deepEqual(result.data.get(1), [prior], 'One failed refresh must not clear an existing chart');
  assert.match(result.error.message, /temporary failure/);
  client.setQueryData(['assessments', 2], [{ id: 2, test_run_id: 2, overall_score: 0.6 }]);
  result = useMultipleAssessments([1, 2]);
  assert.equal(result.data.get(2).length, 1, 'New completion joins the existing assessments');
  assert.equal(specs[0].refetchInterval, false);
  client.clear();
})().catch((error) => { console.error(error); process.exitCode = 1; });
""")


def test_dashboard_summary_preserves_zero_scores_and_newly_completed_runs():
    run_frontend(r"""
const { summarizeDashboardData } = load(path.join(frontend, 'components/dashboard/useDashboardData'));
const run = (id, extra = {}) => ({ id, status: 'completed', test_type: 'conversation', patient_provider: 'ollama', patient_model: 'model', created_at: '2026-09-29T10:00:00', ...extra });
const zero = { id: 1, test_run_id: 1, overall_score: 0, scores: { safety: 0, factuality: 0 }, metadata: { factuality_analysis: { factuality_score: 1 }, cot_analysis: { cot_detected: true } }, flags: ['concern'] };
const scored = { id: 2, test_run_id: 2, overall_score: 1, scores: { safety: 0 }, flags: [] };
const runs = [run(3, { status: 'pending', test_type: 'group_therapy' }), run(2), run(1), run(4, { test_type: 'benchmark', patient_provider: 'openai' })];
const summary = summarizeDashboardData(runs, new Map([[1, [zero]], [2, [scored]]]));
assert.equal(summary.metrics.total, 4);
assert.equal(summary.metrics.pending, 1);
assert.equal(summary.metrics.testsWithAnalysis, 2);
assert.equal(summary.metrics.analysisEligibleCompleted, 2);
assert.equal(summary.metrics.avgSafetyScore, 0);
assert.equal(summary.metrics.totalFlags, 1);
assert.equal(summary.metrics.cotDetected, 1);
assert.equal(summary.scoreTrendData[0].factuality, 0, 'Zero must not fall back to metadata');
assert.equal(summary.topModels[0].avgScore, 0.5, 'Zero assessments count in model averages');
assert.equal(summary.topModels[0].count, 2);
assert.deepEqual(summary.typeBreakdown.find((row) => row.name === 'group therapy'), { name: 'group therapy', count: 1 });
assert.deepEqual(summary.radarScores.map((row) => row.subject), ['Overall', 'Safety', 'Alignment', 'Factuality']);
assert.equal(summary.radarScores.find((row) => row.subject === 'Overall').value, 50);
assert.equal(summary.radarScores.find((row) => row.subject === 'Safety').value, 0);
assert.equal(summary.radarScores.find((row) => row.subject === 'Factuality').value, 0);
assert.deepEqual(summary.providerMix.map((row) => [row.name, row.value]), [['ollama', 3], ['openai', 1]]);
assert(summary.providerMix.every((row) => typeof row.color === 'string' && row.color.startsWith('#')));
const next = summarizeDashboardData(runs.map((row) => ({ ...row, status: 'completed' })), new Map([[1, [zero]], [2, [scored]], [3, []]]));
assert.equal(next.metrics.totalAssessments, 2, 'Existing charts remain populated while new assessments load');
assert.equal(next.metrics.completed, 4);
assert.deepEqual(summarizeDashboardData([], new Map()).radarScores, []);
assert.deepEqual(summarizeDashboardData([], new Map()).providerMix, []);
""")


def test_benchmark_accuracy_rows_prefer_newest_completed_campaign():
    run_frontend(r"""
const { buildBenchmarkAccuracyRows } = load(path.join(frontend, 'components/dashboard/useDashboardData'));
assert.deepEqual(buildBenchmarkAccuracyRows([]), []);
assert.deepEqual(buildBenchmarkAccuracyRows([{
  id: 'empty', status: 'completed', created_at: '2026-09-29T12:00:00',
  runs: [{ model: 'org/a', benchmark: 'mmlu', score: null }],
}]), []);
const campaigns = [
  {
    id: 'old-complete', status: 'completed', created_at: '2026-09-28T10:00:00',
    runs: [{ model: 'org/old', benchmark: 'mmlu', score: 0.9 }],
  },
  {
    id: 'new-partial', status: 'running', created_at: '2026-09-29T12:00:00',
    runs: [{ model: 'org/live', benchmark: 'gsm8k', score: 0.4 }],
  },
  {
    id: 'new-complete', status: 'completed', created_at: '2026-09-29T11:00:00',
    runs: [
      { model: 'meta-llama/Llama-3', benchmark: 'mmlu', score: 0.75 },
      { model: 'org/b', benchmark: 'arc', score: null },
      { model: 'org/b', benchmark: 'hellaswag', score: 0.5 },
    ],
  },
];
assert.deepEqual(buildBenchmarkAccuracyRows(campaigns), [
  { name: 'Llama-3 · MMLU', score: 75 },
  { name: 'b · HellaSwag', score: 50 },
]);
assert.deepEqual(buildBenchmarkAccuracyRows(campaigns.filter((row) => row.status !== 'completed')), [
  { name: 'live · GSM8K', score: 40 },
]);
""")


def test_data_hook_keeps_all_lists_live_then_refreshes_assessments_on_completion():
    run_frontend(r"""
let slots = [], refs = [], cursor = 0, refCursor = 0, effects = [], invalidations = [];
overrides.set('react', { ...React,
  useState(initial) { const i = cursor++; if (!(i in slots)) slots[i] = initial; return [slots[i], (value) => { slots[i] = value; }]; },
  useRef(initial) { const i = refCursor++; return refs[i] ?? (refs[i] = { current: initial }); },
  useMemo: (fn) => fn(), useEffect(fn) { effects.push(fn); },
});
const client = { invalidateQueries: (value) => { invalidations.push(value); return Promise.resolve(); } };
overrides.set('@tanstack/react-query', { useQueryClient: () => client });
overrides.set('next/router', { useRouter: () => ({ push() {} }) });
let runs = [{ id: 1, status: 'completed', test_type: 'conversation', patient_provider: 'ollama', patient_model: 'model' }];
let suites = [{ id: 1, status: 'pending' }], runOptions, suiteOptions, assessmentOptions;
overrides.set('@/lib/hooks', {
  LIVE_REFETCH_INTERVAL: 2500,
  hasActiveWork: (items) => items.some((item) => ['running', 'pending'].includes(item.status)),
  useTestRuns: (_, options) => { runOptions = options; return { data: runs }; },
  useSuites: (_, options) => { suiteOptions = options; return { data: suites }; },
  useMultipleAssessments: (ids, options) => { assessmentOptions = options; return { data: new Map(ids.map((id) => [id, []])) }; },
  useUpdateTestRun: () => ({}), useUpdateSuite: () => ({}),
});
overrides.set('@/lib/benchmark-campaigns', {
  useBenchmarkCampaigns: () => ({ data: { campaigns: [] }, error: null, refetch() {} }),
  benchmarkName: (id) => id,
  shortModelName: (model) => model,
});
const { useDashboardData } = load(path.join(frontend, 'components/dashboard/useDashboardData'));
const render = () => { cursor = refCursor = 0; effects = []; const result = useDashboardData(); effects.forEach((effect) => effect()); return result; };
assert.equal(render().isLive, true, 'A pending suite starts live updates even with completed runs');
render();
assert.equal(runOptions.active, true); assert.equal(suiteOptions.active, true);
assert.equal(assessmentOptions.refetchInterval, 2500);
suites = [{ id: 1, status: 'completed' }];
assert.equal(render().isLive, false);
render();
assert.equal(runOptions.active, false); assert.equal(suiteOptions.active, false);
assert.equal(assessmentOptions.refetchInterval, false);
assert.equal(invalidations.length, 1, 'Final assessment refresh happens once as work becomes idle');
assert.deepEqual(invalidations[0], { queryKey: ['assessments'] });
assert.deepEqual(render().data.benchmarkAccuracy, []);
""")


def test_metric_widgets_reflow_with_their_resizable_container():
    run_frontend(r"""
const { renderDashboardWidget } = load(path.join(frontend, 'components/dashboard/DashboardWidgets'));
const { summarizeDashboardData } = load(path.join(frontend, 'components/dashboard/useDashboardData'));
const summary = summarizeDashboardData([], new Map([[1, [{ id: 1, overall_score: 0.5 }]]]));
for (const id of ['metrics', 'analysis-metrics', 'flags-summary']) {
  const tree = renderDashboardWidget(id, { ...summary, suites: [], benchmarkAccuracy: [] });
  assert.match(tree.props.className, /auto-fit,minmax\(min\(100%,9rem\),1fr\)/, 'Columns must follow the widget width');
  assert.doesNotMatch(tree.props.className, /(?:md|lg):grid-cols/, 'Viewport width cannot determine the resized card columns');
  const html = renderToStaticMarkup(tree);
  assert.match(html, /min-w-0 break-words/);
  assert.match(html, /\[&amp;_\.flex-1\]:min-w-0/);
}
""")


def test_new_chart_widgets_render_empty_and_populated_states():
    run_frontend(r"""
overrides.set('recharts', {
  ...requireFrontend('recharts'), ResponsiveContainer: Child,
  RadarChart: (props) => React.createElement('div', { 'data-radar': JSON.stringify(props.data) }, props.children),
  PolarGrid: () => null, PolarAngleAxis: () => null, PolarRadiusAxis: () => null, Radar: () => null,
  BarChart: (props) => React.createElement('div', { 'data-bars': JSON.stringify(props.data), 'data-layout': props.layout }, props.children),
  Bar: () => null, CartesianGrid: () => null, XAxis: () => null, YAxis: () => null, Legend: () => null, Tooltip: () => null,
  PieChart: Child, Pie: () => null, Cell: () => null,
});
overrides.set('@/components/charts/BenchmarkResultsChart', {
  BenchmarkResultsChart: ({ data, height }) => React.createElement('div', { 'data-benchmark': JSON.stringify(data), 'data-height': String(height) }),
});
const { renderDashboardWidget } = load(path.join(frontend, 'components/dashboard/DashboardWidgets'));
const empty = {
  metrics: { total: 0, running: 0, completed: 0, failed: 0, pending: 0, testsWithAnalysis: 0, analysisEligibleCompleted: 0, totalAssessments: 0, avgOverallScore: 0, avgSafetyScore: 0, cotDetected: 0, factualityChecks: 0, manipulationAnalysis: 0, totalFlags: 0 },
  radarScores: [], providerMix: [], suites: [], benchmarkAccuracy: [],
};
assert.match(renderToStaticMarkup(renderDashboardWidget('safety-radar', empty)), /Open test runs/);
assert.match(renderToStaticMarkup(renderDashboardWidget('provider-mix', empty)), /Create a test/);
assert.match(renderToStaticMarkup(renderDashboardWidget('suite-progress', empty)), /Create a suite/);
assert.match(renderToStaticMarkup(renderDashboardWidget('benchmark-accuracy', empty)), /Open benchmarks/);
const populated = {
  ...empty,
  radarScores: [{ subject: 'Overall', value: 40 }, { subject: 'Safety', value: 50 }],
  providerMix: [{ name: 'ollama', value: 2, color: '#3b82f6' }],
  suites: [{ id: 1, name: 'Alpha', completed_runs: 2, running_runs: 1, failed_runs: 0, pending_runs: 1 }],
  benchmarkAccuracy: [{ name: 'llama · MMLU', score: 66 }],
};
assert.match(renderToStaticMarkup(renderDashboardWidget('safety-radar', populated)), /data-radar="\[\{/);
assert.match(renderToStaticMarkup(renderDashboardWidget('provider-mix', populated)), /Patient provider counts/);
assert.match(renderToStaticMarkup(renderDashboardWidget('suite-progress', populated)), /data-layout="vertical"/);
assert.match(renderToStaticMarkup(renderDashboardWidget('suite-progress', populated)), /&quot;completed&quot;:2/);
assert.match(renderToStaticMarkup(renderDashboardWidget('benchmark-accuracy', populated)), /data-height="100%"/);
assert.match(renderToStaticMarkup(renderDashboardWidget('benchmark-accuracy', populated)), /llama · MMLU/);
""")


def test_status_pie_omits_zero_slices_and_keeps_live_counts_in_wrapping_legend():
    run_frontend(r"""
let pieProps;
overrides.set('recharts', {
  ...requireFrontend('recharts'), ResponsiveContainer: Child, PieChart: Child,
  Pie: (props) => { pieProps = props; return React.createElement('div', null, props.children); },
  Cell: () => null, Tooltip: () => null,
});
const { renderDashboardWidget } = load(path.join(frontend, 'components/dashboard/DashboardWidgets'));
const statuses = [
  { name: 'Completed', value: 30, color: '#10b981' }, { name: 'Running', value: 0, color: '#3b82f6' },
  { name: 'Pending', value: 0, color: '#f59e0b' }, { name: 'Failed', value: 1, color: '#ef4444' },
];
for (const running of [0, 1, 0]) {
  const statusData = statuses.map((entry) => entry.name === 'Running' ? { ...entry, value: running } : entry);
  const html = renderToStaticMarkup(renderDashboardWidget('status-pie', { statusData }));
  assert.deepEqual(pieProps.data.map((entry) => entry.name), running ? ['Completed', 'Running', 'Failed'] : ['Completed', 'Failed']);
  assert.equal(pieProps.label, false, 'Outside labels must not clip on narrow widgets');
  assert.equal(pieProps.labelLine, false);
  assert.match(html, /aria-label="Run status counts"/);
  assert.match(html, /flex-wrap/);
  assert.match(html, new RegExp('Running: <strong>' + running + '</strong>'));
  assert.match(html, /Pending: <strong>0<\/strong>/);
}
pieProps = null;
const empty = renderToStaticMarkup(renderDashboardWidget('status-pie', { statusData: statuses.map((entry) => ({ ...entry, value: 0 })) }));
assert.equal(pieProps, null);
assert.match(empty, /No test runs yet/);
assert.match(empty, /Completed: <strong>0<\/strong>/);
""")
