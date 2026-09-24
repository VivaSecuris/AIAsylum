"""Dashboard builder for multi-prompt progression analysis (one-shot, multi-shot)."""

from typing import Dict, Any
import numpy as np

from vivasecuris.aiasylum.interp.data.multi_prompt_models import ProgressionResult
from vivasecuris.aiasylum.interp.visualization.assets import plotly_script_tag
from vivasecuris.aiasylum.interp.visualization.base import DashboardBuilderBase, serialize_payload_for_js


class ProgressionDashboardBuilder(DashboardBuilderBase):
    """Builds interactive HTML dashboards for progression analysis."""

    def build_dashboard(self, result: ProgressionResult) -> str:
        """
        Build a complete HTML dashboard for progression analysis.
        
        Args:
            result: ProgressionResult to visualize
            
        Returns:
            HTML string for dashboard
        """
        # Convert numpy arrays to lists for JSON serialization
        progression_cos = result.progression_cos.tolist()
        progression_delta = result.progression_delta.tolist()
        cumulative_cos = result.cumulative_cos.tolist()
        cumulative_delta = result.cumulative_delta.tolist()
        
        # Prepare data payload
        data = {
            "meta": result.meta,
            "prompt_labels": result.prompt_labels,
            "aligned_tokens": result.aligned_tokens,
            "progression_cos": progression_cos,
            "progression_delta": progression_delta,
            "cumulative_cos": cumulative_cos,
            "cumulative_delta": cumulative_delta,
            "query_starts": result.query_starts,
            "pca_payload": result.pca_payload or {},
            "example_impact": result.example_impact or {},
        }
        
        # Generate HTML
        html = self._generate_html(data)
        return html

    def _generate_html(self, data: Dict[str, Any]) -> str:
        """Generate the HTML dashboard."""
        data_json = serialize_payload_for_js(data)
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multi-Prompt Progression Analysis</title>
    {plotly_script_tag()}
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #1a1a1a;
            border-bottom: 3px solid #1a1a1a;
            padding-bottom: 10px;
            margin-bottom: 30px;
        }}
        h2 {{
            color: #1a1a1a;
            margin-top: 40px;
            margin-bottom: 20px;
        }}
        .section {{
            margin: 30px 0;
        }}
        .plot-container {{
            margin: 20px 0;
            background: white;
            padding: 15px;
            border-radius: 4px;
            border: 1px solid #ddd;
        }}
        .info-box {{
            background: #f9f9f9;
            padding: 15px;
            border-radius: 4px;
            margin: 20px 0;
            border-left: 4px solid #1a1a1a;
        }}
        .prompt-list {{
            background: #f9f9f9;
            padding: 15px;
            border-radius: 4px;
            margin: 20px 0;
        }}
        .prompt-item {{
            margin: 10px 0;
            padding: 10px;
            background: white;
            border-radius: 4px;
            border: 1px solid #ddd;
        }}
        .prompt-label {{
            font-weight: bold;
            color: #1a1a1a;
            margin-bottom: 5px;
        }}
        .prompt-text {{
            color: #666;
            font-family: monospace;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Multi-Prompt Progression Analysis</h1>
        
        <div class="info-box">
            <strong>Analysis Mode:</strong> {data['meta'].get('analysis_mode', 'progression')}<br>
            <strong>Model:</strong> {data['meta'].get('model', 'unknown')}<br>
            <strong>Number of Prompts:</strong> {data['meta'].get('num_prompts', 0)}<br>
            <strong>Query Window Length:</strong> {data['meta'].get('query_window_len', 0)} tokens
        </div>
        
        <div class="section">
            <h2>Prompts Analyzed</h2>
            <div class="prompt-list">
                {self._generate_prompt_list(data)}
            </div>
        </div>
        
        <div class="section">
            <h2>Progression Analysis</h2>
            <p>How activations change as examples are added (each step vs previous step)</p>
            
            <div class="plot-container">
                <h3>Progression Delta Norm (Step-by-Step Changes)</h3>
                <div id="progression-delta-heatmap"></div>
            </div>
            
            <div class="plot-container">
                <h3>Progression Cosine Similarity (Step-by-Step Changes)</h3>
                <div id="progression-cos-heatmap"></div>
            </div>
        </div>
        
        <div class="section">
            <h2>Cumulative Analysis</h2>
            <p>How each prompt differs from the zero-shot baseline</p>
            
            <div class="plot-container">
                <h3>Cumulative Delta Norm (vs Zero-Shot)</h3>
                <div id="cumulative-delta-heatmap"></div>
            </div>
            
            <div class="plot-container">
                <h3>Cumulative Cosine Similarity (vs Zero-Shot)</h3>
                <div id="cumulative-cos-heatmap"></div>
            </div>
        </div>
        
        <div class="section">
            <h2>Layer Impact Analysis</h2>
            <p>Which layers are most affected by adding examples</p>
            <div class="plot-container">
                <div id="layer-impact-plot"></div>
            </div>
        </div>
        
        <div class="section">
            <h2 id="dr-title">3D Trajectories</h2>
            <p>How token representations evolve across prompts</p>
            <div class="plot-container">
                <label>Select Layer: 
                    <select id="pca-layer-select"></select>
                </label>
                <div id="pca-3d-plot"></div>
            </div>
        </div>
    </div>
    
    <script>
        const data = {data_json};
        const drMethod = (data.meta && data.meta.dim_reduction) ? data.meta.dim_reduction : 'pca';
        const drName = (drMethod === 'umap') ? 'UMAP' : (drMethod === 'tsne') ? 't-SNE' : 'PCA';
        const drTitleEl = document.getElementById('dr-title');
        if (drTitleEl) drTitleEl.textContent = `3D ${{drName}} Trajectories`;
        
        // Generate progression heatmaps
        function generateProgressionHeatmaps() {{
            const numSteps = data.progression_delta.length;
            const numLayers = data.progression_delta[0].length;
            const numTokens = data.progression_delta[0][0].length;
            
            // Progression Delta Norm
            const progressionTraces = [];
            for (let i = 0; i < numSteps; i++) {{
                const stepLabel = `${{data.prompt_labels[i]}} → ${{data.prompt_labels[i+1]}}`;
                progressionTraces.push({{
                    z: data.progression_delta[i],
                    type: 'heatmap',
                    colorscale: 'Blues',
                    colorbar: {{ title: 'Delta Norm' }},
                    name: stepLabel,
                    visible: i === 0
                }});
            }}
            
            const progressionButtons = [];
            for (let i = 0; i < numSteps; i++) {{
                const visibility = new Array(numSteps).fill(false);
                visibility[i] = true;
                progressionButtons.push({{
                    label: `${{data.prompt_labels[i]}} → ${{data.prompt_labels[i+1]}}`,
                    method: 'update',
                    args: [{{visible: visibility}}]
                }});
            }}
            
            Plotly.newPlot('progression-delta-heatmap', progressionTraces, {{
                title: 'Progression Delta Norm by Layer and Token Position',
                xaxis: {{ title: 'Token Position' }},
                yaxis: {{ title: 'Layer' }},
                height: 500,
                updatemenus: [{{
                    type: 'dropdown',
                    direction: 'down',
                    x: 0.1,
                    y: 1.15,
                    buttons: progressionButtons
                }}]
            }});
            
            // Progression Cosine Similarity
            const progressionCosTraces = [];
            for (let i = 0; i < numSteps; i++) {{
                progressionCosTraces.push({{
                    z: data.progression_cos[i],
                    type: 'heatmap',
                    colorscale: 'RdYlBu',
                    reversescale: true,
                    colorbar: {{ title: 'Cosine Similarity' }},
                    name: `${{data.prompt_labels[i]}} → ${{data.prompt_labels[i+1]}}`,
                    visible: i === 0
                }});
            }}
            
            const progressionCosButtons = [];
            for (let i = 0; i < numSteps; i++) {{
                const visibility = new Array(numSteps).fill(false);
                visibility[i] = true;
                progressionCosButtons.push({{
                    label: `${{data.prompt_labels[i]}} → ${{data.prompt_labels[i+1]}}`,
                    method: 'update',
                    args: [{{visible: visibility}}]
                }});
            }}
            
            Plotly.newPlot('progression-cos-heatmap', progressionCosTraces, {{
                title: 'Progression Cosine Similarity by Layer and Token Position',
                xaxis: {{ title: 'Token Position' }},
                yaxis: {{ title: 'Layer' }},
                height: 500,
                updatemenus: [{{
                    type: 'dropdown',
                    direction: 'down',
                    x: 0.1,
                    y: 1.15,
                    buttons: progressionCosButtons
                }}]
            }});
        }}
        
        // Generate cumulative heatmaps
        function generateCumulativeHeatmaps() {{
            const numPrompts = data.cumulative_delta.length;
            
            // Cumulative Delta Norm
            const cumulativeTraces = [];
            for (let i = 0; i < numPrompts; i++) {{
                cumulativeTraces.push({{
                    z: data.cumulative_delta[i],
                    type: 'heatmap',
                    colorscale: 'Blues',
                    colorbar: {{ title: 'Delta Norm' }},
                    name: data.prompt_labels[i],
                    visible: i === 0
                }});
            }}
            
            const cumulativeButtons = [];
            for (let i = 0; i < numPrompts; i++) {{
                const visibility = new Array(numPrompts).fill(false);
                visibility[i] = true;
                cumulativeButtons.push({{
                    label: data.prompt_labels[i],
                    method: 'update',
                    args: [{{visible: visibility}}]
                }});
            }}
            
            Plotly.newPlot('cumulative-delta-heatmap', cumulativeTraces, {{
                title: 'Cumulative Delta Norm vs Zero-Shot by Layer and Token Position',
                xaxis: {{ title: 'Token Position' }},
                yaxis: {{ title: 'Layer' }},
                height: 500,
                updatemenus: [{{
                    type: 'dropdown',
                    direction: 'down',
                    x: 0.1,
                    y: 1.15,
                    buttons: cumulativeButtons
                }}]
            }});
            
            // Cumulative Cosine Similarity
            const cumulativeCosTraces = [];
            for (let i = 0; i < numPrompts; i++) {{
                cumulativeCosTraces.push({{
                    z: data.cumulative_cos[i],
                    type: 'heatmap',
                    colorscale: 'RdYlBu',
                    reversescale: true,
                    colorbar: {{ title: 'Cosine Similarity' }},
                    name: data.prompt_labels[i],
                    visible: i === 0
                }});
            }}
            
            const cumulativeCosButtons = [];
            for (let i = 0; i < numPrompts; i++) {{
                const visibility = new Array(numPrompts).fill(false);
                visibility[i] = true;
                cumulativeCosButtons.push({{
                    label: data.prompt_labels[i],
                    method: 'update',
                    args: [{{visible: visibility}}]
                }});
            }}
            
            Plotly.newPlot('cumulative-cos-heatmap', cumulativeCosTraces, {{
                title: 'Cumulative Cosine Similarity vs Zero-Shot by Layer and Token Position',
                xaxis: {{ title: 'Token Position' }},
                yaxis: {{ title: 'Layer' }},
                height: 500,
                updatemenus: [{{
                    type: 'dropdown',
                    direction: 'down',
                    x: 0.1,
                    y: 1.15,
                    buttons: cumulativeCosButtons
                }}]
            }});
        }}
        
        // Generate layer impact plot
        function generateLayerImpactPlot() {{
            if (!data.example_impact || !data.example_impact.layer_impact) return;
            
            const layers = [];
            for (let i = 0; i < data.example_impact.layer_impact.length; i++) {{
                layers.push(i);
            }}
            
            const trace = {{
                x: layers,
                y: data.example_impact.layer_impact,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Average Impact',
                line: {{color: 'blue', width: 2}},
                marker: {{size: 8}}
            }};
            
            Plotly.newPlot('layer-impact-plot', [trace], {{
                title: 'Average Layer Impact from Adding Examples',
                xaxis: {{ title: 'Layer' }},
                yaxis: {{ title: 'Average Delta Norm' }},
                height: 400
            }});
        }}
        
        // Generate PCA 3D plot
        function generatePCA3DPlot(layerIdx) {{
            const layerKey = String(layerIdx);
            if (!data.pca_payload || !data.pca_payload[layerKey]) return;
            
            const layerData = data.pca_payload[layerKey];
            const traces = [];
            
            for (const [label, pcaData] of Object.entries(layerData)) {{
                const pca = pcaData.pca;
                const x = pca.map(p => p[0]);
                const y = pca.map(p => p[1]);
                const z = pca.map(p => p[2]);
                
                traces.push({{
                    x: x,
                    y: y,
                    z: z,
                    type: 'scatter3d',
                    mode: 'lines+markers',
                    name: label,
                    line: {{width: 4}},
                    marker: {{size: 3}}
                }});
            }}
            
            Plotly.newPlot('pca-3d-plot', traces, {{
                title: `3D ${{drName}} Trajectories - Layer ${{layerIdx}}`,
                scene: {{
                    xaxis: {{ title: (drName === 'PCA') ? 'PC1' : 'Dim1' }},
                    yaxis: {{ title: (drName === 'PCA') ? 'PC2' : 'Dim2' }},
                    zaxis: {{ title: (drName === 'PCA') ? 'PC3' : 'Dim3' }}
                }},
                height: 600
            }});
        }}
        
        // Setup PCA layer selector
        function setupPCASelector() {{
            const select = document.getElementById('pca-layer-select');
            if (!data.pca_payload) return;
            
            const layers = Object.keys(data.pca_payload).map(Number).sort((a, b) => a - b);
            layers.forEach(layer => {{
                const option = document.createElement('option');
                option.value = layer;
                option.textContent = `Layer ${{layer}}`;
                select.appendChild(option);
            }});
            
            if (layers.length > 0) {{
                generatePCA3DPlot(layers[0]);
            }}
            
            select.addEventListener('change', (e) => {{
                generatePCA3DPlot(parseInt(e.target.value));
            }});
        }}
        
        // Initialize all plots
        generateProgressionHeatmaps();
        generateCumulativeHeatmaps();
        generateLayerImpactPlot();
        setupPCASelector();
    </script>
</body>
</html>"""
        return html

    def _generate_prompt_list(self, data: Dict[str, Any]) -> str:
        """Generate HTML for prompt list."""
        html_parts = []
        for i, (label, prompt, tokens) in enumerate(zip(
            data['prompt_labels'],
            data['meta']['prompts'],
            data['aligned_tokens']
        )):
            html_parts.append(f"""
                <div class="prompt-item">
                    <div class="prompt-label">{label}</div>
                    <div class="prompt-text">{prompt[:200]}{'...' if len(prompt) > 200 else ''}</div>
                    <div style="font-size: 0.85em; color: #999; margin-top: 5px;">
                        {len(tokens)} tokens analyzed
                    </div>
                </div>
            """)
        return ''.join(html_parts)
