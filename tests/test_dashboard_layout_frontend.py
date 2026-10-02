"""Dashboard customization round-trips safely through browser storage."""

from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def run_layout_case():
    node = shutil.which("node")
    typescript = ROOT / "frontend/node_modules/typescript"
    if not node or not typescript.exists():
        pytest.skip("Node and frontend TypeScript dependencies are needed")
    harness = r"""
const fs = require('node:fs');
const assert = require('node:assert/strict');
const ts = require(process.argv[1]);
const source = fs.readFileSync(process.argv[2], 'utf8');
const code = ts.transpileModule(source, { compilerOptions: {
  module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020,
}}).outputText;
const loaded = { exports: {} };
new Function('module', 'exports', code)(loaded, loaded.exports);
const {
  DASHBOARD_LAYOUT_KEY, DASHBOARD_COLUMNS, MAX_WIDGET_HEIGHT,
  DEFAULT_LAYOUT, WIDGET_CATALOG, sanitizeLayout,
  serializeDashboardLayout, deserializeDashboardLayout,
  loadDashboardLayout, saveDashboardLayout,
  displayDashboardLayout, addWidget, removeWidget, availableWidgets,
} = loaded.exports;
"""

    def run(case):
        result = subprocess.run(
            [node, "-e", harness + case, str(typescript),
             str(ROOT / "frontend/lib/dashboard-layout.ts")],
            cwd=ROOT, capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
    return run


def test_geometry_and_enabled_widgets_survive_serialization(run_layout_case):
    run_layout_case(r"""
const selected = [
  { i: 'recent-runs', x: 5, y: 9, w: 7, h: 11, minW: 4, minH: 5 },
  { i: 'status-pie', x: 0, y: 0, w: 5, h: 9, minW: 3, minH: 5 },
];
assert.deepEqual(deserializeDashboardLayout(serializeDashboardLayout(selected)), selected);
assert.equal(deserializeDashboardLayout(serializeDashboardLayout(selected)).some(item => item.i === 'metrics'), false);
const copy = deserializeDashboardLayout(serializeDashboardLayout(selected));
copy[0].x = 0;
assert.equal(selected[0].x, 5);
""")


def test_remove_last_chart_preserves_an_empty_dashboard_on_reload(run_layout_case):
    run_layout_case(r"""
let selected = addWidget([], 'status-pie');
assert.equal(selected.length, 1);
selected = removeWidget(selected, 'status-pie');
assert.deepEqual(sanitizeLayout(selected), []);
assert.equal(serializeDashboardLayout(selected), '[]');
assert.deepEqual(deserializeDashboardLayout('[]'), []);
assert.equal(availableWidgets(selected).length, WIDGET_CATALOG.length);
assert.deepEqual(deserializeDashboardLayout(serializeDashboardLayout(addWidget(selected, 'metrics'))), addWidget([], 'metrics'));
""")


def test_invalid_storage_uses_fresh_defaults_without_mutating_catalog(run_layout_case):
    run_layout_case(r"""
for (const corrupt of [null, '', '{truncated', '{}', 'null', 'true', '7', '[null]', '[{"i":"old-chart"}]']) {
  const result = deserializeDashboardLayout(corrupt);
  assert.deepEqual(result, DEFAULT_LAYOUT, String(corrupt));
  assert.notEqual(result, DEFAULT_LAYOUT);
  result[0].w = 1;
  assert.notEqual(DEFAULT_LAYOUT[0].w, 1);
}
assert(availableWidgets(DEFAULT_LAYOUT).length > 0, 'The initial picker should offer optional charts');
assert(DEFAULT_LAYOUT.some((item) => item.i === 'safety-radar'));
assert(DEFAULT_LAYOUT.some((item) => item.i === 'provider-mix'));
assert(DEFAULT_LAYOUT.some((item) => item.i === 'suite-progress'));
assert(DEFAULT_LAYOUT.some((item) => item.i === 'benchmark-accuracy'));
assert.equal(WIDGET_CATALOG.filter((item) => ['safety-radar', 'provider-mix', 'suite-progress', 'benchmark-accuracy'].includes(item.id)).length, 4);
""")


def test_sanitization_clamps_bounds_ignores_duplicates_and_rejects_non_numbers(run_layout_case):
    run_layout_case(r"""
const sanitized = sanitizeLayout([
  { i: 'status-pie', x: 500, y: -5, w: 900, h: -3, minW: 0, minH: 0 },
  { i: 'status-pie', x: 0, y: 1, w: 4, h: 8 },
  { i: 'recent-runs', x: 11.9, y: 6.9, w: 4.9, h: 900 },
  { i: 'suites', x: -2, y: 1e100, w: -2, h: 4 },
  { i: 'metrics', x: null, y: 0, w: 12, h: 5 },
  { i: 'analysis-metrics', x: '0', y: 0, w: 12, h: 5 },
  { i: 'score-trends', x: 0, y: NaN, w: 12, h: 8 },
  { i: 'top-models', x: 0, y: 0, w: Infinity, h: 8 },
  { i: 'unknown', x: 0, y: 0, w: 12, h: 8 },
]);
assert.deepEqual(sanitized.map(item => item.i), ['status-pie', 'recent-runs', 'suites']);
assert.deepEqual(sanitized[0], { i: 'status-pie', x: 0, y: 0, w: 12, h: 5, minW: 3, minH: 5 });
assert.deepEqual(sanitized[1], { i: 'recent-runs', x: 8, y: 6, w: 4, h: MAX_WIDGET_HEIGHT, minW: 4, minH: 5 });
assert(sanitized[2].y <= 10000);
for (const item of sanitized) {
  assert(item.x + item.w <= DASHBOARD_COLUMNS);
  assert(item.x >= 0 && item.y >= 0);
  assert(item.w >= item.minW && item.h >= item.minH);
}
""")


def test_storage_key_and_reload_include_enabled_widgets(run_layout_case):
    run_layout_case(r"""
const storage = new Map();
global.window = { localStorage: {
  getItem: key => storage.get(key) ?? null,
  setItem: (key, value) => storage.set(key, value),
}};
assert.equal(DASHBOARD_LAYOUT_KEY, 'aiasylum.dashboard.layout.v1');
assert.deepEqual(loadDashboardLayout(), DEFAULT_LAYOUT);
const selected = addWidget(removeWidget(DEFAULT_LAYOUT, 'metrics'), 'top-models');
assert.equal(saveDashboardLayout(selected), true);
assert.equal(storage.size, 1);
assert.deepEqual(loadDashboardLayout(), selected);
assert.equal(saveDashboardLayout([]), true);
assert.deepEqual(loadDashboardLayout(), []);
""")


def test_storage_exceptions_and_server_rendering_do_not_break_dashboard(run_layout_case):
    run_layout_case(r"""
assert.deepEqual(loadDashboardLayout(), DEFAULT_LAYOUT);
assert.equal(saveDashboardLayout(DEFAULT_LAYOUT), false);
global.window = {};
Object.defineProperty(window, 'localStorage', { get() { throw new Error('SecurityError'); } });
assert.deepEqual(loadDashboardLayout(), DEFAULT_LAYOUT);
assert.equal(saveDashboardLayout(DEFAULT_LAYOUT), false);
global.window = { localStorage: {
  getItem() { throw new Error('Blocked'); },
  setItem() { throw new Error('QuotaExceededError'); },
}};
assert.deepEqual(loadDashboardLayout(), DEFAULT_LAYOUT);
assert.equal(saveDashboardLayout(DEFAULT_LAYOUT), false);
""")


def test_add_remove_never_duplicates_and_places_new_widget_after_existing(run_layout_case):
    run_layout_case(r"""
const original = serializeDashboardLayout(DEFAULT_LAYOUT);
const selected = removeWidget(DEFAULT_LAYOUT, 'status-pie');
assert(availableWidgets(selected).some(item => item.id === 'status-pie'));
const restored = addWidget(selected, 'status-pie');
const added = restored[restored.length - 1];
assert.equal(added.y, Math.max(...selected.map(item => item.y + item.h)));
assert.equal(addWidget(restored, 'status-pie').filter(item => item.i === 'status-pie').length, 1);
assert.equal(availableWidgets(restored).some(item => item.id === 'status-pie'), false);
assert.equal(addWidget(restored, 'unknown'), restored);
assert.equal(serializeDashboardLayout(DEFAULT_LAYOUT), original);
""")


def test_narrow_screen_stacking_preserves_saved_desktop_geometry(run_layout_case):
    run_layout_case(r"""
const selected = sanitizeLayout([
  { i: 'status-pie', x: 6, y: 8, w: 6, h: 9 },
  { i: 'metrics', x: 0, y: 0, w: 12, h: 5 },
  { i: 'recent-runs', x: 0, y: 8, w: 6, h: 7 },
]);
const before = serializeDashboardLayout(selected);
const stacked = displayDashboardLayout(selected, true);
assert.deepEqual(stacked.map(item => item.i), ['metrics', 'recent-runs', 'status-pie']);
assert(stacked.every(item => item.x === 0 && item.w === DASHBOARD_COLUMNS));
assert(stacked[0].h >= 8);
for (let i = 1; i < stacked.length; i++) {
  assert.equal(stacked[i].y, stacked[i - 1].y + stacked[i - 1].h);
}
assert.equal(serializeDashboardLayout(selected), before);
assert.deepEqual(displayDashboardLayout(selected, false), selected);
assert.deepEqual(displayDashboardLayout([], true), []);
""")
