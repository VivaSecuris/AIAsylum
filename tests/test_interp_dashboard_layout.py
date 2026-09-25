"""Every Plotly canvas must reserve document-flow space before rendering."""

from html.parser import HTMLParser
import json
import re
import shutil
import subprocess

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from vivasecuris.aiasylum.interp.visualization.dashboard_builder import DashboardBuilder
from vivasecuris.aiasylum.interp.visualization.progression_dashboard_builder import ProgressionDashboardBuilder
from vivasecuris.aiasylum.interp.visualization.single_prompt_dashboard_builder import SinglePromptDashboardBuilder


class Elements(HTMLParser):
    def __init__(self):
        super().__init__()
        self.classes = {}

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.classes[attrs["id"]] = attrs.get("class", "").split()


@pytest.mark.parametrize("analysis_mode,side,context", [
    ("model_diff", "Model", "Comparing model A (original) vs model B (modified) on the same prompt"),
    ("comparison", "Prompt", "Comparing prompts A vs B"),
    (None, "Prompt", "Comparing prompts A vs B"),
])
def test_comparison_labels_follow_recorded_mode_without_changing_capture_data(analysis_mode, side, context):
    data = {
        "meta": {"model": "original vs custom", "model_a": "original", "model_b": "custom<edit>",
                 "analysis_mode": analysis_mode, "spike_layer": 1, "window_len": 2},
        "tokens_a": ["same", " prompt"], "tokens_b": ["same", " prompt"],
        "cos_mat": [[0.98, 0.94]], "dn_mat": [[0.1, 0.2]],
    }
    html = DashboardBuilder()._generate_html(data)
    assert context in html
    assert f'<label for="prompt-selector">{side} View</label>' in html
    for letter in ("A", "B"):
        assert f"{side} {letter} Only</option>" in html
        assert f"<strong>{side} {letter}:</strong>" in html
        assert f"name: '{side} {letter}'" in html
        assert f"hovertemplate: '{side} {letter}<br>Token:" in html
    if analysis_mode == "model_diff":
        assert "Model A (original)</div>" in html
        assert "Model B (modified)</div>" in html
        assert "custom&lt;edit&gt;</div>" in html
        assert "this model pair on the same prompt" in html
        assert "between these models on the same prompt" in html
        assert "Prompt A Only" not in html and "Prompt B Only" not in html
    # Labels are presentation-only: arrays, identities, and recorded mode survive.
    embedded, _ = json.JSONDecoder().raw_decode(html.split("const data = ", 1)[1])
    assert embedded == data


@pytest.mark.parametrize("builder", [SinglePromptDashboardBuilder, DashboardBuilder, ProgressionDashboardBuilder])
def test_all_chart_mounts_reserve_height_and_resize(builder):
    html = builder()._generate_html({
        "meta": {"model": "test", "spike_layer": 1, "window_len": 2, "prompts": []},
        "prompt_labels": [], "aligned_tokens": [],
    })
    elements = Elements()
    elements.feed(html)
    plot_ids = re.findall(r"renderDashboardPlot\('([^']+)'", html)
    assert plot_ids
    for plot_id in plot_ids:
        assert "plot-target" in elements.classes[plot_id], plot_id
    assert "height: var(--plot-height)" in html
    assert "height: target.clientHeight" in html
    assert "responsive: true" in html
    assert "@media (max-width: 700px)" in html
    assert html.count("Plotly.newPlot(") == 1  # the shared renderer only
    assert "Safety Neurons Identified" not in html
    assert "Safety Neuron Clusters" not in html
    assert "MLP Neuron Analysis" not in html


@pytest.mark.parametrize("space,label", [
    ("residual_channels", "residual channels"),
    ("mlp_neurons", "neurons"),
    (None, "components"),
])
def test_embedded_component_labels_are_descriptive_and_preserve_counts(space, label):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is needed to execute generated dashboard JavaScript")
    metadata = {"activation_space": space} if space else {}
    data = {
        "meta": {"model": "test", "spike_layer": 1, "window_len": 2},
        "mlp_payload": {"1": {
            **metadata, "num_neurons": 5120,
            "neurons": [{"neuron": 3, "contribution_score": 1.25}],
        }},
        "circuit_payload": {
            "spike_layer_last_token": {
                **metadata, "layer": 1, "token": 1,
                "circuit_summary": "Legacy unsupported safety neurons label",
                "involved_neurons": [{"neuron": 3, "contribution_pct": 25.0}],
            },
            "safety_neuron_clusters": {
                **metadata, "num_safety_neurons": 4958, "safety_neuron_percentage": 96.8,
                "safety_neurons": [{"neuron": 3, "activation_contrast": 1.25}],
            },
        },
    }
    html = DashboardBuilder()._generate_html(data)
    assert "selected layers and positions shown in each result" in html
    functions = html.split("function mlpComponentLabels", 1)[1].split("function initOVQKVisualization", 1)[0]
    script = "const data = " + json.dumps(data) + ";" + """
        let selectedLayer = null;
        const elements = {};
        const document = { getElementById(id) {
            return elements[id] ||= { style: {}, innerHTML: '' };
        }};
        const plots = {};
        function renderDashboardPlot(id, traces, layout) { plots[id] = { traces, layout }; }
    """ + "function mlpComponentLabels" + functions + """
        initMLPVisualization();
        initCircuitVisualization();
        process.stdout.write(JSON.stringify({ elements, plots, data }));
    """
    output = subprocess.run([node, "-e", script], check=True, capture_output=True, text=True)
    rendered = json.loads(output.stdout)
    info = rendered["elements"]["mlp-info"]["innerHTML"]
    cards = rendered["elements"]["circuit-cards"]["innerHTML"]
    assert f"Total {label}:" in info and "5120" in info
    assert f"High-contrast MLP {label}" in cards
    assert "4958" in cards and "96.8%" in cards
    assert "threshold heuristic" in cards
    assert "does not identify safety-related components" in cards
    assert "Legacy unsupported" not in cards
    assert "Safety Neurons Identified" not in cards
    assert label in rendered["plots"]["mlp-neurons"]["layout"]["title"]
    assert rendered["data"] == data


def test_progression_component_charts_execute_and_preserve_measurements():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node is needed to exercise the generated chart code')
    data = {
        'meta': {'model': 'test', 'prompts': ['a', 'b']},
        'prompt_labels': ['zero-shot', 'one-shot'], 'aligned_tokens': [['a', 'b'], ['c', 'd']],
        'attention_payload': {label: {'0': {'mean_over_heads': [[1, 0], [.4, .6]]}} for label in ('zero-shot', 'one-shot')},
        'mlp_payload': {'zero-shot': {'0': {'token_norms': [1, 2]}}, 'one-shot': {'0': {'token_norms': [3, 4]}}},
        'predictions_payload': {label: {'predictions': [{'top_tokens': ['x', 'y'], 'probs': [.3, .2]}]} for label in ('zero-shot', 'one-shot')},
    }
    html = ProgressionDashboardBuilder()._generate_html(data)
    functions = html.split('function setupComponents()', 1)[1].split('// Initialize all plots', 1)[0]
    script = 'const data = ' + json.dumps(data) + ';' + r'''
      const elements = {};
      const document = { getElementById(id) {
        return elements[id] ||= { hidden: false, value: '', add(option) { this.value ||= option.value; }, addEventListener() {} };
      }};
      function Option(label, value) { this.value = value; }
      const plots = {};
      function renderDashboardPlot(id, traces, layout) { plots[id] = {traces, layout}; }
    ''' + 'function setupComponents()' + functions + '''
      setupComponents(); drawPredictions();
      process.stdout.write(JSON.stringify(plots));
    '''
    result = subprocess.run([node, '-e', script], capture_output=True, text=True, check=True)
    plots = json.loads(result.stdout)
    assert plots['attention-progression-plot']['traces'][0]['z'] == [[1, 0], [.4, .6]]
    assert plots['mlp-progression-plot']['traces'][1]['y'] == [3, 4]
    assert plots['prediction-progression-plot']['traces'][0]['y'] == [.3, .2]
