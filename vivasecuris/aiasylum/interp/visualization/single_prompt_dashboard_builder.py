"""Dashboard builder for single-prompt analysis."""

from typing import Dict, Any

import numpy as np

from vivasecuris.aiasylum.interp.data.models import SinglePromptResult
from vivasecuris.aiasylum.interp.analysis.dim_reduction import DimensionReduction
from vivasecuris.aiasylum.interp.visualization.assets import plotly_script_tag
from vivasecuris.aiasylum.interp.visualization.base import (
    DashboardBuilderBase, dashboard_plot_css, dashboard_plot_script, serialize_payload_for_js,
)


class SinglePromptDashboardBuilder(DashboardBuilderBase):
    """Builds interactive HTML dashboards for single-prompt analysis."""

    def build_dashboard(self, result: SinglePromptResult) -> str:
        """
        Build a complete HTML dashboard for single-prompt analysis.

        Args:
            result: SinglePromptResult to visualize

        Returns:
            HTML string for dashboard
        """
        activation_norm_mat = result.activation_norm_mat
        if activation_norm_mat is not None:
            norm_mat_list = np.asarray(activation_norm_mat).tolist()
        else:
            norm_mat_list = []

        data = {
            "meta": result.meta,
            "tokens": result.tokens,
            "activation_norm_mat": norm_mat_list,
            "pca_payload": result.pca_payload or {},
            "dim_reduction_payload": result.pca_payload or {},
            "predictions_payload": result.predictions_payload,
            "attention_payload": result.attention_payload or {},
            "mlp_payload": result.mlp_payload or {},
            "logit_attribution_payload": result.logit_attribution_payload or {},
            "num_layers": len(norm_mat_list) if norm_mat_list else 0,
            "window_len": result.window_len,
        }
        return self._generate_html(data)

    def _generate_html(self, data: Dict[str, Any]) -> str:
        """Generate the HTML dashboard."""
        data_json = serialize_payload_for_js(data)
        meta = data.get("meta", {})
        dr_method = meta.get("dim_reduction", "pca")
        dr_name = DimensionReduction.get_method_display_name(dr_method)
        axis_labels = DimensionReduction.get_axis_labels(dr_method)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Single-Prompt Analysis</title>
    {plotly_script_tag()}
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1400px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        h1 {{ color: #1a1a1a; border-bottom: 3px solid #1a1a1a; padding-bottom: 10px; margin-bottom: 20px; }}
        h2 {{ color: #1a1a1a; margin-top: 40px; margin-bottom: 20px; }}
        .info-box {{ background: #f9f9f9; padding: 15px; border-radius: 4px; margin: 20px 0; border-left: 4px solid #2196F3; }}
        .plot-container {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 4px; }}
        .token-strip {{ background: #f9f9f9; padding: 10px; border-radius: 4px; font-family: monospace; font-size: 0.9em; overflow-x: auto; }}
        table {{ border-collapse: collapse; width: 100%; margin: 15px 0; font-size: 0.9em; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background: #f5f5f5; }}
        {dashboard_plot_css()}
    </style>
</head>
<body>
    <div class="container">
        <h1>Single-Prompt Analysis</h1>
        <div class="info-box">
            <strong>Model:</strong> {meta.get('model', 'N/A')} &nbsp;|&nbsp;
            <strong>Mode:</strong> Single prompt &nbsp;|&nbsp;
            <strong>Window:</strong> {data.get('window_len', 0)} tokens &nbsp;|&nbsp;
            <strong>Layers:</strong> {data.get('num_layers', 0)} &nbsp;|&nbsp;
            <strong>Dim reduction:</strong> {dr_name}
        </div>
        <div class="info-box">
            <strong>Prompt preview:</strong> {meta.get('prompt_preview', '')}
        </div>

        <h2>Tokens (analysis window)</h2>
        <div class="token-strip" id="token-strip"></div>

        <h2>Activation magnitude (L2 norm by layer and token)</h2>
        <div class="plot-container plot-target" id="activation-norm-heatmap"></div>

        <h2>3D trajectory ({dr_name})</h2>
        <p>Select layer:</p>
        <select id="pca-layer-select"></select>
        <div class="plot-container plot-target plot-tall" id="pca-3d"></div>

        <h2>Predictions</h2>
        <div class="plot-container plot-target plot-medium" id="token-to-token-plot"></div>
        <div id="predictions-table"></div>

        <div id="attention-section" style="display:none;">
            <h2>Attention (mean over heads)</h2>
            <select id="attn-layer-select"></select>
            <div class="plot-container plot-target" id="attention-heatmap"></div>
        </div>
        <div id="logit-attribution-section" style="display:none;">
            <h2>Logit attribution (last token)</h2>
            <div class="info-box" id="logit-attribution-info"></div>
        </div>
    </div>
    <script>
        {dashboard_plot_script()}
        const data = {data_json};

        // Tokens
        document.getElementById('token-strip').textContent = data.tokens.map(t => t.replace(/\\\\s/g, '·')).join(' ');

        // Activation norm heatmap
        if (data.activation_norm_mat && data.activation_norm_mat.length > 0) {{
            const z = data.activation_norm_mat;
            const y = z.map((_, i) => 'Layer ' + i);
            const x = z[0].map((_, i) => i);
            renderDashboardPlot('activation-norm-heatmap', [{{
                z: z, x: x, y: y, type: 'heatmap',
                colorscale: 'Viridis', colorbar: {{ title: 'L2 norm' }}
            }}], {{ margin: {{ t: 20, r: 80 }}, xaxis: {{ title: 'Token position' }}, yaxis: {{ title: 'Layer' }} }}, {{ responsive: true }});
        }}

        // PCA layer dropdown
        const pcaLayers = Object.keys(data.pca_payload || {{}}).sort((a,b) => parseInt(a)-parseInt(b));
        const layerSelect = document.getElementById('pca-layer-select');
        pcaLayers.forEach(l => {{
            const opt = document.createElement('option');
            opt.value = l;
            opt.textContent = 'Layer ' + l;
            layerSelect.appendChild(opt);
        }});
        function plotTrajectory(layerKey) {{
            const payload = data.pca_payload && data.pca_payload[layerKey];
            if (!payload || !payload.trajectory) return;
            const tr = payload.trajectory;
            const ev = payload.explained_variance || [];
            const x = tr.map(r => r[0]);
            const y = tr.map(r => r[1]);
            const z = tr.map(r => r[2] || 0);
            const text = data.tokens.map((t, i) => `${{i}}: ${{t}}`);
            renderDashboardPlot('pca-3d', [{{
                x: x, y: y, z: z, type: 'scatter3d', mode: 'lines+markers',
                line: {{ color: 'rgb(33, 150, 243)', width: 4 }},
                marker: {{ size: 4, text: text, hoverinfo: 'text' }}
            }}], {{
                margin: {{ t: 20 }},
                scene: {{ xaxis: {{ title: '{axis_labels[0]}' }}, yaxis: {{ title: '{axis_labels[1]}' }}, zaxis: {{ title: '{axis_labels[2]}' }} }},
                title: ev.length ? 'Explained variance: ' + ev.map(e => (e*100).toFixed(1)+'%').join(', ') : ''
            }}, {{ responsive: true }});
        }}
        if (pcaLayers.length > 0) {{
            plotTrajectory(pcaLayers[0]);
            layerSelect.onchange = () => plotTrajectory(layerSelect.value);
        }}

        // Token-to-token diff and predictions table
        const pred = data.predictions_payload;
        if (pred && pred.token_to_token_diff && pred.token_to_token_diff.length > 0) {{
            const diff = pred.token_to_token_diff;
            renderDashboardPlot('token-to-token-plot', [{{
                x: diff.map((_, i) => i + 1),
                y: diff,
                type: 'scatter',
                mode: 'lines+markers',
                line: {{ color: '#2196F3', width: 2 }},
                marker: {{ size: 6 }}
            }}], {{ margin: {{ t: 20 }}, xaxis: {{ title: 'Token transition' }}, yaxis: {{ title: '||h_i - h_{{{{i-1}}}}||' }} }}, {{ responsive: true }});
        }}
        function escapeHtml(s) {{ return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }}
        if (pred && pred.predictions && pred.predictions.length > 0) {{
            let table = '<table><tr><th>Pos</th><th>Token</th><th>Top tokens</th><th>Probs</th></tr>';
            pred.predictions.forEach(p => {{
                table += '<tr><td>' + p.position + '</td><td>' + escapeHtml(p.token) + '</td><td>' + (p.top_tokens || []).slice(0,5).join(', ') + '</td><td>' + (p.probs || []).slice(0,5).map(x => x.toFixed(3)).join(', ') + '</td></tr>';
            }});
            table += '</table>';
            document.getElementById('predictions-table').innerHTML = table;
        }}

        // Attention (optional)
        const attnLayers = Object.keys(data.attention_payload || {{}});
        if (attnLayers.length > 0) {{
            document.getElementById('attention-section').style.display = 'block';
            const attnSelect = document.getElementById('attn-layer-select');
            attnLayers.forEach(l => {{
                const opt = document.createElement('option');
                opt.value = l;
                opt.textContent = 'Layer ' + l;
                attnSelect.appendChild(opt);
            }});
            function plotAttn(layerKey) {{
                const ap = data.attention_payload[layerKey];
                if (!ap || !ap.mean_over_heads) return;
                renderDashboardPlot('attention-heatmap', [{{
                    z: ap.mean_over_heads, type: 'heatmap',
                    colorscale: 'Blues', xaxis: 'x', yaxis: 'y'
                }}], {{ margin: {{ t: 20 }}, xaxis: {{ title: 'Key' }}, yaxis: {{ title: 'Query' }} }}, {{ responsive: true }});
            }}
            plotAttn(attnLayers[0]);
            attnSelect.onchange = () => plotAttn(attnSelect.value);
        }}

        // Logit attribution (optional)
        const la = data.logit_attribution_payload;
        if (la && la.attribution_available) {{
            document.getElementById('logit-attribution-section').style.display = 'block';
            let info = 'By component (last token): ';
            if (la.head_attributions) {{
                info += la.head_attributions.slice(0, 5).map(h => `Layer ${{h.layer}} H${{h.head}}: ${{(h.contribution || 0).toFixed(4)}}`).join('; ');
            }}
            document.getElementById('logit-attribution-info').innerHTML = info;
        }}
    </script>
</body>
</html>
"""
        return html
