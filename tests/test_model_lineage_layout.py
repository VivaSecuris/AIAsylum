"""Graph presentation must preserve recorded ancestry without inventing it."""
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
const { layoutLineage, relatedLineage, NODE_WIDTH, NODE_HEIGHT } = loaded.exports;
const model = (id, time = null) => ({ id, label: id, kind: 'model', model_ref: id, created_at: time });
const step = (id, time = null) => ({ ...model(id, time), kind: 'surgery', model_ref: null });
const edge = (source, target) => ({ source, target, label: 'recorded' });
"""

    def run(case):
        subprocess.run(
            [node, "-e", harness + case, str(typescript),
             str(ROOT / "frontend/components/models/lineage-layout.ts")],
            cwd=ROOT, check=True, capture_output=True, text=True,
        )
    return run


def test_dependencies_override_timestamps_and_branches_do_not_overlap(run_layout_case):
    run_layout_case(r"""
const nodes = [model('base', '2026-09-24'), step('edit-a', '2026-09-22'), step('edit-b', '2026-09-23'), model('a'), model('b')];
const edges = [edge('base', 'edit-a'), edge('base', 'edit-b'), edge('edit-a', 'a'), edge('edit-b', 'b')];
const layout = layoutLineage(nodes, edges);
assert.deepEqual(layout.edges, edges);
for (const e of edges) assert(layout.positions.get(e.source).x < layout.positions.get(e.target).x);
for (let i = 0; i < nodes.length; i++) for (let j = i + 1; j < nodes.length; j++) {
  const a = layout.positions.get(nodes[i].id), b = layout.positions.get(nodes[j].id);
  assert(Math.abs(a.x - b.x) >= NODE_WIDTH || Math.abs(a.y - b.y) >= NODE_HEIGHT);
}
assert.equal(layout.edges.some(e => e.source === 'edit-a' && e.target === 'edit-b'), false);
""")


def test_checkpoint_focus_keeps_ancestors_descendants_and_excludes_siblings(run_layout_case):
    run_layout_case(r"""
const nodes = [model('base'), step('edit-a'), step('edit-b'), model('a'), model('b'), step('analysis-a'), model('unrelated')];
const edges = [edge('base', 'edit-a'), edge('base', 'edit-b'), edge('edit-a', 'a'), edge('edit-b', 'b'), edge('a', 'analysis-a')];
const focused = relatedLineage(nodes, edges, 'a');
assert.deepEqual(focused.nodes.map(n => n.id).sort(), ['a', 'analysis-a', 'base', 'edit-a']);
assert.equal(focused.edges.length, 3);
assert.equal(relatedLineage(nodes, edges, 'absent').nodes.length, 0);
assert.equal(relatedLineage(nodes, edges).nodes.length, nodes.length);
""")


def test_layout_is_deterministic_when_api_order_changes(run_layout_case):
    run_layout_case(r"""
const nodes = [model('base'), step('edit-10', '2026-09-24T10:00:00Z'), step('edit-2', '2026-09-24T10:00:00Z'), model('a'), model('b')];
const edges = [edge('base', 'edit-10'), edge('base', 'edit-2'), edge('edit-10', 'a'), edge('edit-2', 'b')];
const a = layoutLineage(nodes, edges), b = layoutLineage([...nodes].reverse(), [...edges].reverse());
assert.deepEqual([...a.positions.entries()], [...b.positions.entries()]);
assert.equal(a.positions.get('edit-2').y < a.positions.get('edit-10').y, true);
""")


def test_missing_nodes_remain_visible_and_cycles_do_not_hang(run_layout_case):
    run_layout_case(r"""
const missing = { ...step('missing-direction'), kind: 'missing', status: 'missing' };
const nodes = [missing, step('edit'), model('checkpoint')];
const edges = [edge('missing-direction', 'edit'), edge('edit', 'checkpoint')];
assert.equal(layoutLineage(nodes, edges).positions.has('missing-direction'), true);
const cyclic = layoutLineage([step('a'), step('b')], [edge('a', 'b'), edge('b', 'a')]);
assert.equal(cyclic.cyclic.length, 2);
assert.equal(cyclic.edges.length, 2);
for (const position of cyclic.positions.values()) assert(Number.isFinite(position.x) && Number.isFinite(position.y));
const empty = layoutLineage([], []);
assert(Number.isFinite(empty.width) && Number.isFinite(empty.height));
""")
