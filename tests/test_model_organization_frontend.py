"""Experiment views preserve graph ancestry and share the right model labels."""

from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def run_organization_case():
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
const mockRequire = (name) => {
  if (name === '@tanstack/react-query') return {};
  if (name === '@/lib/api') return { apiClient: {} };
  throw new Error(`Unexpected runtime dependency: ${name}`);
};
new Function('module', 'exports', 'require', code)(loaded, loaded.exports, mockRequire);
const { filterExperimentLineage, belongsToExperiment, modelOrganizationKey, stepOrganizationKey, UNASSIGNED_EXPERIMENT } = loaded.exports;
const model = (id, ref = id) => ({ id, label: id, kind: 'model', model_ref: ref });
const step = (id, kind = 'surgery') => ({ id, label: id, kind, model_ref: null });
const edge = (source, target, label = 'recorded') => ({ source, target, label });
const metadata = (...experiment_ids) => ({ label: '', notes: '', experiment_ids });
const sorted = (items) => [...items].sort();
"""

    def run(case):
        result = subprocess.run(
            [node, "-e", harness + case, str(typescript),
             str(ROOT / "frontend/lib/model-organization.ts")],
            cwd=ROOT, capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
    return run


def test_selected_experiment_retains_ancestors_without_unrelated_siblings(run_organization_case):
    run_organization_case(r"""
const graph = {
  nodes: [model('base'), step('extract'), step('edit-a'), step('edit-b'), model('custom-a'), model('custom-b'), step('analyze-a', 'analysis'), model('unrelated')],
  edges: [edge('base', 'extract'), edge('extract', 'edit-a'), edge('extract', 'edit-b'), edge('edit-a', 'custom-a'), edge('edit-b', 'custom-b'), edge('custom-a', 'analyze-a')],
};
const organization = { experiments: [], items: {
  'model:custom-a': metadata('repair'),
  'step:analyze-a': metadata('repair'),
  'step:edit-b': metadata('another-experiment'),
}};
const snapshot = JSON.stringify(graph);
const result = filterExperimentLineage(graph, organization, 'repair');
assert.deepEqual(sorted(result.nodes.map(node => node.id)), ['analyze-a', 'base', 'custom-a', 'edit-a', 'extract']);
assert.deepEqual(result.edges, [graph.edges[0], graph.edges[1], graph.edges[3], graph.edges[5]]);
assert.deepEqual(sorted(result.context), ['base', 'edit-a', 'extract']);
assert.equal(JSON.stringify(graph), snapshot, 'Filtering must not rewrite provenance');
assert.equal(result.edges.some(item => item.source === 'edit-a' && item.target === 'edit-b'), false);
assert.equal(filterExperimentLineage(graph, organization, 'missing').nodes.length, 0);
""")


def test_unassigned_includes_unlabeled_items_and_assigned_ancestors_as_context(run_organization_case):
    run_organization_case(r"""
const graph = {
  nodes: [model('base'), step('assigned-edit'), model('untagged-output'), step('labeled-only'), model('assigned-sibling')],
  edges: [edge('base', 'assigned-edit'), edge('assigned-edit', 'untagged-output'), edge('base', 'assigned-sibling')],
};
const organization = { experiments: [], items: {
  'model:base': metadata('repair'),
  'step:assigned-edit': metadata('repair'),
  'step:labeled-only': { label: 'Name without a group', notes: 'Review later', experiment_ids: [] },
  'model:assigned-sibling': metadata('repair'),
}};
const result = filterExperimentLineage(graph, organization, UNASSIGNED_EXPERIMENT);
assert.deepEqual(sorted(result.nodes.map(node => node.id)), ['assigned-edit', 'base', 'labeled-only', 'untagged-output']);
assert.deepEqual(result.edges, graph.edges.slice(0, 2));
assert.deepEqual(sorted(result.context), ['assigned-edit', 'base']);
assert.equal(belongsToExperiment(undefined, UNASSIGNED_EXPERIMENT), true);
assert.equal(belongsToExperiment(metadata(), UNASSIGNED_EXPERIMENT), true);
assert.equal(belongsToExperiment(metadata('repair'), UNASSIGNED_EXPERIMENT), false);
assert.equal(belongsToExperiment(metadata('repair', 'control'), 'control'), true);
assert.equal(belongsToExperiment(undefined, 'repair'), false);
""")


def test_no_experiment_filter_preserves_every_node_and_recorded_edge(run_organization_case):
    run_organization_case(r"""
const graph = {
  nodes: [model('base'), step('edit-a'), step('edit-b'), model('isolated')],
  edges: [edge('base', 'edit-a', 'input'), edge('base', 'edit-b', 'fork')],
};
for (const organization of [undefined, { experiments: [], items: { 'step:edit-a': metadata('repair') } }]) {
  const result = filterExperimentLineage(graph, organization, '');
  assert.deepEqual(result.nodes, graph.nodes);
  assert.deepEqual(result.edges, graph.edges);
  assert.equal(result.context.size, 0);
  assert.equal(belongsToExperiment(undefined, ''), true);
  assert.equal(belongsToExperiment(metadata('repair'), ''), true);
}
const empty = filterExperimentLineage({ nodes: [], edges: [] }, undefined, '');
assert.deepEqual(empty.nodes, []);
assert.deepEqual(empty.edges, []);
assert.equal(empty.context.size, 0);
""")


def test_model_metadata_is_shared_by_reference_and_step_metadata_by_durable_node_id(run_organization_case):
    run_organization_case(r"""
const ref = '/home/ubuntu/aiasylum/models/custom';
const firstModel = model('server-origin-model-node', ref);
const importedModel = model('workspace-origin-model-node', ref);
const expected = `model:${ref}`;
assert.equal(modelOrganizationKey(ref), expected);
assert.equal(stepOrganizationKey(firstModel), expected);
assert.equal(stepOrganizationKey(importedModel), expected);
assert.notEqual(stepOrganizationKey(model('same-visible-name', 'Qwen/Qwen3-0.6B')), expected);
const firstStep = { ...step('server:weight:550e8400-e29b-41d4-a716-446655440000'), label: 'Repair', run_id: 1, model_ref: ref };
const laterStep = { ...step('server:weight:550e8400-e29b-41d4-a716-446655440001'), label: 'Repair', run_id: 1, model_ref: ref };
assert.equal(stepOrganizationKey(firstStep), `step:${firstStep.id}`);
assert.notEqual(stepOrganizationKey(firstStep), stepOrganizationKey(laterStep));
assert.equal(stepOrganizationKey(model('missing-reference', null)), 'step:missing-reference');
const organization = { experiments: [], items: {
  [expected]: { ...metadata('repair'), label: 'Shared name' },
  [stepOrganizationKey(firstStep)]: metadata('repair'),
}};
const filtered = filterExperimentLineage({ nodes: [firstModel, importedModel, firstStep, laterStep], edges: [] }, organization, 'repair');
assert.deepEqual(filtered.nodes, [firstModel, importedModel, firstStep]);
assert.equal(organization.items[stepOrganizationKey(firstModel)].label, organization.items[stepOrganizationKey(importedModel)].label);
""")


def test_ancestor_walk_terminates_for_cycles_without_adding_relationships(run_organization_case):
    run_organization_case(r"""
const graph = { nodes: [step('a'), step('b'), step('unrelated')], edges: [edge('a', 'b'), edge('b', 'a')] };
const organization = { experiments: [], items: { 'step:a': metadata('repair') } };
const filtered = filterExperimentLineage(graph, organization, 'repair');
assert.deepEqual(filtered.nodes, graph.nodes.slice(0, 2));
assert.deepEqual(filtered.edges, graph.edges);
assert.deepEqual([...filtered.context], ['b']);
""")
