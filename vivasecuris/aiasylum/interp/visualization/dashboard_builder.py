"""Enhanced dashboard builder for generating interactive HTML dashboards with Phase 1 features."""

import json
from html import escape
from typing import Dict, Any
import numpy as np

from vivasecuris.aiasylum.interp.data.models import ComparisonResult
from vivasecuris.aiasylum.interp.analysis.dim_reduction import DimensionReduction
from vivasecuris.aiasylum.interp.visualization.assets import plotly_script_tag
from vivasecuris.aiasylum.interp.visualization.base import (
    DashboardBuilderBase, dashboard_plot_css, dashboard_plot_script, serialize_payload_for_js,
)


class DashboardBuilder(DashboardBuilderBase):
    """Builds interactive HTML dashboards from comparison results with enhanced Phase 1 features."""

    def build_dashboard(self, result: ComparisonResult) -> str:
        """
        Build a complete HTML dashboard with Phase 1 enhancements.
        
        Args:
            result: ComparisonResult to visualize
            
        Returns:
            HTML string for dashboard
        """
        # Convert numpy arrays to lists for JSON serialization
        cos_mat = result.cos_mat.tolist()
        dn_mat = result.dn_mat.tolist()
        
        # Compute per-layer summaries
        num_layers = len(dn_mat)
        layer_summaries = []
        for layer_idx in range(num_layers):
            layer_dn = dn_mat[layer_idx]
            mean_div = float(np.mean(layer_dn))
            max_div = float(np.max(layer_dn))
            last_div = float(layer_dn[-1]) if len(layer_dn) > 0 else 0.0
            layer_summaries.append({
                "layer": layer_idx,
                "mean_divergence": mean_div,
                "max_divergence": max_div,
                "last_token_divergence": last_div,
            })
        
        # Detect first divergence token
        first_divergence_token = self._detect_first_divergence(dn_mat)
        
        # Prepare data payload
        data = {
            "meta": result.meta,
            "tokens_a": result.tokens_a,
            "tokens_b": result.tokens_b,
            "cos_mat": cos_mat,
            "dn_mat": dn_mat,
            "pca_payload": result.pca_payload or {},
            # Generalized alias so UI can treat PCA/UMAP/t-SNE uniformly
            "dim_reduction_payload": result.pca_payload or {},
            "predictions_payload": result.predictions_payload,
            "layer_summaries": layer_summaries,
            "first_divergence_token": first_divergence_token,
            "num_layers": num_layers,
            "window_len": result.window_len,
            # Phase 2 payloads
            "attention_payload": result.attention_payload or {},
            "mlp_payload": result.mlp_payload or {},
            "circuit_payload": result.circuit_payload or {},
            "temporal_payload": result.temporal_payload or {},
            "attribution_payload": result.attribution_payload or {},
            "logit_attribution_payload": result.logit_attribution_payload or {},
            # Phase 3: activation patching experiments (if enabled)
            "patching_results": result.patching_results or {},
            # Phase 2 MI: OV/QK and causal circuits
            "ov_qk_payload": result.ov_qk_payload or {},
            "scrub_payload": result.scrub_payload or {},
            "minimal_circuit_payload": result.minimal_circuit_payload or {},
        }
        
        # Generate HTML
        html = self._generate_html(data)
        return html

    def _detect_first_divergence(self, dn_mat: list) -> int:
        """Detect the first token where divergence exceeds threshold."""
        if not dn_mat or not dn_mat[0]:
            return 0
        
        threshold = 0.1  # Configurable threshold
        num_tokens = len(dn_mat[0])
        
        for token_idx in range(num_tokens):
            # Check if any layer has significant divergence at this token
            for layer_idx in range(len(dn_mat)):
                if dn_mat[layer_idx][token_idx] > threshold:
                    return token_idx
        
        return 0

    def _generate_html(self, data: Dict[str, Any]) -> str:
        """Generate the HTML dashboard with Phase 1 enhancements."""
        data_json = serialize_payload_for_js(data)
        
        # Get dimension reduction method info
        dr_method = data.get("meta", {}).get("dim_reduction", "pca")
        dr_name = DimensionReduction.get_method_display_name(dr_method)
        axis_labels = DimensionReduction.get_axis_labels(dr_method)
        
        # A model comparison reuses the same capture/plot schema as prompt
        # comparison, but its two traces are different weights on one prompt.
        meta = data.get("meta", {})
        model_diff = meta.get("analysis_mode") == "model_diff"
        label_a = "Model A" if model_diff else "Prompt A"
        label_b = "Model B" if model_diff else "Prompt B"
        view_label = "Model View" if model_diff else "Prompt View"
        comparison_subject = "this model pair on the same prompt" if model_diff else "this prompt pair"
        contrast_subject = "between these models on the same prompt" if model_diff else "between these prompts"
        if model_diff:
            model_metadata = "".join(
                '<div class="metadata-item"><div class="metadata-label">'
                + label + '</div><div class="metadata-value">'
                + escape(str(meta.get(key, "Not recorded"))) + '</div></div>'
                for key, label in (("model_a", "Model A (original)"), ("model_b", "Model B (modified)"))
            )
        else:
            model_metadata = (
                '<div class="metadata-item"><div class="metadata-label">Model</div>'
                '<div class="metadata-value">' + escape(str(meta.get("model", "Not recorded"))) + '</div></div>'
            )

        # Context blurb
        spike_layer = meta.get("spike_layer", "?")
        first_div_token = data.get("first_divergence_token", "?")
        comparison_context = (
            "Comparing model A (original) vs model B (modified) on the same prompt"
            if model_diff else "Comparing prompts A vs B"
        )
        context_blurb = (
            f"{comparison_context}; spike layer: {spike_layer}; "
            f"first divergence at token: {first_div_token}"
        )
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM Neurosurgeon Framework - Enhanced Dashboard</title>
    {plotly_script_tag()}
    <style>
        * {{
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1600px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #1a1a1a;
            border-bottom: 3px solid #1a1a1a;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        h2 {{
            color: #1a1a1a;
            margin-top: 40px;
            margin-bottom: 20px;
        }}
        h3 {{
            color: #333;
            margin-top: 20px;
            margin-bottom: 10px;
        }}
        
        /* Global Controls Panel */
        .controls-panel {{
            background: #f9f9f9;
            padding: 20px;
            border-radius: 8px;
            border: 2px solid #ddd;
            margin-bottom: 30px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        .control-group {{
            display: flex;
            flex-direction: column;
        }}
        .control-group label {{
            font-weight: bold;
            margin-bottom: 5px;
            font-size: 0.9em;
            color: #555;
        }}
        .control-group select,
        .control-group input {{
            padding: 8px;
            border: 1px solid #ccc;
            border-radius: 4px;
            font-size: 0.9em;
        }}
        
        /* Context Blurb */
        .context-blurb {{
            background: #e8f4f8;
            padding: 15px;
            border-radius: 6px;
            border-left: 4px solid #2196F3;
            margin-bottom: 20px;
            font-size: 0.95em;
        }}
        
        /* Layer Summary Cards */
        .layer-summaries {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .layer-card {{
            background: white;
            padding: 15px;
            border-radius: 6px;
            border: 2px solid #ddd;
            cursor: pointer;
            transition: all 0.2s;
        }}
        .layer-card:hover {{
            border-color: #2196F3;
            box-shadow: 0 2px 8px rgba(33, 150, 243, 0.2);
        }}
        .layer-card.selected {{
            border-color: #2196F3;
            background: #e3f2fd;
        }}
        .layer-card .layer-number {{
            font-weight: bold;
            font-size: 1.1em;
            color: #2196F3;
            margin-bottom: 8px;
        }}
        .layer-card .metric {{
            font-size: 0.85em;
            margin: 4px 0;
            color: #666;
        }}
        .layer-card .metric-value {{
            font-weight: bold;
            color: #333;
        }}
        .trend-indicator {{
            display: inline-block;
            margin-left: 5px;
            font-size: 0.8em;
        }}
        
        /* Token Overlay */
        .token-overlay {{
            background: #f9f9f9;
            padding: 15px;
            border-radius: 6px;
            margin: 20px 0;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            overflow-x: auto;
        }}
        .token-strip {{
            display: flex;
            gap: 2px;
            flex-wrap: wrap;
        }}
        .token-item {{
            padding: 4px 6px;
            border-radius: 3px;
            cursor: pointer;
            transition: all 0.2s;
            border: 1px solid transparent;
        }}
        .token-item:hover {{
            background: #e3f2fd;
            border-color: #2196F3;
        }}
        .token-item.selected {{
            background: #2196F3;
            color: white;
        }}
        .token-item.high-divergence {{
            background: #ffebee;
            border-color: #f44336;
        }}
        .divergence-marker {{
            background: #fff3cd;
            border-left: 3px solid #ffc107;
            padding: 2px 8px;
            margin: 5px 0;
            font-size: 0.85em;
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
            border-left: 4px solid #4CAF50;
        }}
        .metadata {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .metadata-item {{
            padding: 10px;
            background: white;
            border-radius: 4px;
            border: 1px solid #ddd;
        }}
        .metadata-label {{
            font-weight: bold;
            color: #555;
            font-size: 0.9em;
        }}
        .metadata-value {{
            color: #333;
            margin-top: 5px;
        }}
        
        /* Help callout */
        .help-callout {{
            background: #fff9e6;
            border-left: 4px solid #ff9800;
            padding: 10px 15px;
            margin: 15px 0;
            border-radius: 4px;
            font-size: 0.9em;
        }}
        .help-callout summary {{
            cursor: pointer;
            font-weight: bold;
            color: #e65100;
        }}
        
        /* 2D Small Multiples */
        .small-multiples {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            margin: 20px 0;
        }}
        .small-multiple {{
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 10px;
        }}
        
        /* Distribution plots */
        .distribution-container {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        {dashboard_plot_css()}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 LLM Neurosurgeon Framework - Enhanced Dashboard</h1>
        
        <!-- Context Blurb -->
        <div class="context-blurb">
            <strong>📋 Context:</strong> {context_blurb}
        </div>
        
        <!-- Global Controls Panel -->
        <div class="controls-panel">
            <div class="control-group">
                <label for="layer-selector">Layer Selector</label>
                <select id="layer-selector">
                    <option value="all">All Layers</option>
                </select>
            </div>
            <div class="control-group">
                <label for="token-selector">Token Range</label>
                <input type="range" id="token-selector" min="0" max="{data.get('window_len', 100) - 1}" value="0" step="1">
                <span id="token-display">Token: 0</span>
            </div>
            <div class="control-group">
                <label for="method-selector">DR Method</label>
                <select id="method-selector">
                    <option value="pca" {'selected' if dr_method == 'pca' else ''}>PCA</option>
                    <option value="umap" {'selected' if dr_method == 'umap' else ''}>UMAP</option>
                    <option value="tsne" {'selected' if dr_method == 'tsne' else ''}>t-SNE</option>
                </select>
            </div>
            <div class="control-group">
                <label for="color-by">Color By</label>
                <select id="color-by">
                    <option value="cosine">Cosine Similarity</option>
                    <option value="delta">Delta Norm</option>
                    <option value="layer">Layer Depth</option>
                    <option value="token">Token Position</option>
                </select>
            </div>
            <div class="control-group">
                <label for="prompt-selector">{view_label}</label>
                <select id="prompt-selector">
                    <option value="comparison">Differential (A vs B)</option>
                    <option value="prompt_a">{label_a} Only</option>
                    <option value="prompt_b">{label_b} Only</option>
                </select>
            </div>
        </div>
        
        <!-- Run overview -->
        <div class="info-box">
            <h3>Run overview</h3>
            <div class="metadata">
                {model_metadata}
                <div class="metadata-item">
                    <div class="metadata-label">Spike Layer</div>
                    <div class="metadata-value">Layer {data['meta']['spike_layer']}</div>
                </div>
                <div class="metadata-item">
                    <div class="metadata-label">Aligned Window Length</div>
                    <div class="metadata-value">{data['meta']['window_len']} tokens</div>
                </div>
                <div class="metadata-item">
                    <div class="metadata-label">Alignment Strategy</div>
                    <div class="metadata-value">{data['meta'].get('align', '?')}</div>
                </div>
                <div class="metadata-item">
                    <div class="metadata-label">First Divergence</div>
                    <div class="metadata-value">Token {data.get('first_divergence_token', '?')}</div>
                </div>
                <div class="metadata-item">
                    <div class="metadata-label">DR Method</div>
                    <div class="metadata-value">{dr_name}</div>
                </div>
            </div>
        </div>
        
        <!-- Per-Layer Summary Cards -->
        <div class="section">
            <h2>📊 Per-Layer Summary Cards</h2>
            <div class="help-callout">
                <details>
                    <summary>How to read layer summaries</summary>
                    Click a layer card to focus all visualizations on that layer. Mean divergence shows average difference across tokens. Max divergence shows peak difference. Last token divergence shows divergence at the final token position.
                </details>
            </div>
            <div class="layer-summaries" id="layer-summaries">
                <!-- Populated by JavaScript -->
            </div>
        </div>
        
        <!-- Token Overlay -->
        <div class="section">
            <h2>🔤 Token Alignment</h2>
            <div class="token-overlay">
                <div style="margin-bottom: 10px;">
                    <strong>{label_a}:</strong>
                    <div class="token-strip" id="tokens-a">
                        <!-- Populated by JavaScript -->
                    </div>
                </div>
                <div>
                    <strong>{label_b}:</strong>
                    <div class="token-strip" id="tokens-b">
                        <!-- Populated by JavaScript -->
                    </div>
                </div>
                {f'<div class="divergence-marker">⚠️ First significant divergence detected at token {data.get("first_divergence_token", "?")}</div>' if data.get("first_divergence_token", 0) > 0 else ''}
            </div>
        </div>
        
        <!-- Main Visualizations -->
        <div class="section">
            <h2>📊 Cosine Similarity Heatmap</h2>
            <div class="help-callout">
                <details>
                    <summary>How to read this heatmap</summary>
                    Brighter colors indicate higher similarity (closer to 1.0). Click a cell to highlight that layer/token across all visualizations. Values close to 1.0 indicate similar activations; lower values indicate differences.
                </details>
            </div>
            <div class="plot-container plot-target" id="cosine-heatmap"></div>
        </div>
        
        <div class="section">
            <h2>📈 Delta Norm Heatmap</h2>
            <div class="help-callout">
                <details>
                    <summary>How to read this heatmap</summary>
                    Brighter colors indicate larger differences in activation magnitudes. Higher values indicate larger differences. Click a cell to see detailed information.
                </details>
            </div>
            <div class="plot-container plot-target" id="delta-norm-heatmap"></div>
        </div>
        
        <div class="section">
            <h2>📉 Divergence Curves</h2>
            <div class="plot-container plot-target plot-medium" id="divergence-curves"></div>
        </div>
        
        <!-- 2D Small Multiples -->
        <div class="section">
            <h2>📐 2D Projections ({dr_name})</h2>
            <div class="help-callout">
                <details>
                    <summary>How to read 2D projections</summary>
                    These show 2D views of the 3D trajectory space. Select a point in one view to highlight it in all three views. Use these to quickly scan for patterns.
                </details>
            </div>
            <div class="small-multiples">
                <div class="small-multiple">
                    <div class="plot-target plot-compact" id="projection-12"></div>
                </div>
                <div class="small-multiple">
                    <div class="plot-target plot-compact" id="projection-13"></div>
                </div>
                <div class="small-multiple">
                    <div class="plot-target plot-compact" id="projection-23"></div>
                </div>
            </div>
        </div>
        
        <!-- 3D Trajectories -->
        <div class="section">
            <h2 id="dr-title">🌐 3D {dr_name} Trajectories</h2>
            <div class="help-callout">
                <details>
                    <summary>How to read 3D trajectories</summary>
                    These show how token representations evolve through layers in reduced 3D space. Blue line = {label_a}, Red line = {label_b}. Each point represents a token position. Use the controls above to change layers or methods.
                </details>
            </div>
            <div class="plot-container plot-target plot-tall" id="pca-trajectories"></div>
        </div>
        
        <!-- Similarity Distributions -->
        <div class="section">
            <h2>📊 Similarity Distributions</h2>
            <div class="distribution-container">
                <div class="plot-container plot-target plot-compact" id="cosine-distribution"></div>
                <div class="plot-container plot-target plot-compact" id="delta-distribution"></div>
            </div>
        </div>
        
        <!-- Phase 3: Counterfactual Interventions -->
        <div class="section" id="phase3-section" style="display: none;">
            <h2>🧪 Phase 3: Counterfactual Patching Experiments</h2>
            <div class="help-callout">
                <details>
                    <summary>What these experiments show</summary>
                    These experiments intervene at the selected layers and positions shown in each result, using source activations from A → B or B → A. Each intervention reruns the model; similarity and prediction changes are measured at the target sequence's final token.
                </details>
            </div>
            <div class="plot-container plot-target plot-medium" id="patching-summary"></div>
            <div class="info-box" id="patching-details"></div>
        </div>
        
        <!-- Phase 2: Component Localization -->
        <div class="section" id="phase2-section" style="display: none;">
            <h2>🔍 Phase 2: Component Localization</h2>
            
            <!-- Temporal Localization -->
            <div id="temporal-section" style="display: none;">
                <h3>⏱️ Temporal Localization</h3>
                <div class="plot-container plot-target plot-compact" id="temporal-timeline"></div>
                <div class="info-box" id="temporal-info"></div>
            </div>
            
            <!-- Attention Head Analysis -->
            <div id="attention-section" style="display: none;">
                <h3>👁️ Attention Head Analysis</h3>
                <div class="plot-container plot-target plot-medium" id="attention-heads"></div>
                <div class="info-box" id="attention-info"></div>
            </div>
            
            <!-- MLP activation analysis -->
            <div id="mlp-section" style="display: none;">
                <h3>🧠 MLP Activation Analysis</h3>
                <div class="plot-container plot-target plot-medium" id="mlp-neurons"></div>
                <div class="info-box" id="mlp-info"></div>
            </div>
            
            <!-- Circuit Cards -->
            <div id="circuit-section" style="display: none;">
                <h3>🔗 Component Contrast</h3>
                <div class="info-box" id="circuit-cards"></div>
            </div>
            
            <!-- OV/QK Per-Head Analysis -->
            <div id="ov-qk-section" style="display: none;">
                <h3>📈 OV/QK Per-Head Analysis</h3>
                <div class="info-box" id="ov-qk-info"></div>
            </div>
            
            <!-- Minimal Sufficient Circuit -->
            <div id="minimal-circuit-section" style="display: none;">
                <h3>🔬 Component Reconstruction Search</h3>
                <div class="info-box" id="minimal-circuit-info"></div>
            </div>
            
            <!-- Attribution Analysis -->
            <div id="attribution-section" style="display: none;">
                <h3>📊 Attribution Analysis</h3>
                <div class="plot-container plot-target plot-medium" id="attribution-plot"></div>
                <div class="info-box" id="attribution-info"></div>
            </div>
            
            <!-- Logit Decomposition (component-wise) -->
            <div id="logit-attribution-section" style="display: none;">
                <h3>📐 Logit Decomposition</h3>
                <div class="help-callout">
                    <details>
                        <summary>What this shows</summary>
                        Decomposition of logits at the last token into contributions from embedding and each layer's attention and MLP (via the unembedding matrix). Top components by logit contribution norm are listed.
                    </details>
                </div>
                <div class="info-box" id="logit-attribution-info"></div>
            </div>
        </div>
        
        <!-- Help Section -->
        <div class="info-box">
            <h3>💡 How to Read This Dashboard</h3>
            <ul>
                <li><strong>Cosine Similarity</strong>: Values close to 1.0 indicate similar activations. Lower values indicate differences.</li>
                <li><strong>Delta Norm</strong>: Higher values indicate larger differences in activation magnitudes.</li>
                <li><strong>Spike Layer</strong>: The layer with maximum divergence at the last token position.</li>
                <li><strong>3D Trajectories</strong>: Visualizes how token representations evolve through layers in reduced 3D space ({dr_name}).</li>
                <li><strong>Linked Views</strong>: Click any visualization element to highlight corresponding elements across all charts.</li>
                <li><strong>Layer Cards</strong>: Click a layer summary card to focus all visualizations on that layer.</li>
            </ul>
        </div>
    </div>
    
    <script>
        {dashboard_plot_script()}
        const data = {data_json};
        const drMethod = '{dr_method}';
        const drName = '{dr_name}';
        const axisLabels = {json.dumps(list(axis_labels))};
        
        // Global state for linked views
        let selectedLayer = null;
        let selectedToken = null;
        let selectedMethod = drMethod;
        
        // Initialize layer summaries
        function initLayerSummaries() {{
            const container = document.getElementById('layer-summaries');
            const summaries = data.layer_summaries || [];
            
            summaries.forEach((summary, idx) => {{
                const card = document.createElement('div');
                card.className = 'layer-card';
                card.dataset.layer = summary.layer;
                
                // Calculate trend (simplified - compare with previous layer)
                let trend = '';
                if (idx > 0) {{
                    const prevMean = summaries[idx - 1].mean_divergence;
                    if (summary.mean_divergence > prevMean) {{
                        trend = '<span class="trend-indicator">↗</span>';
                    }} else if (summary.mean_divergence < prevMean) {{
                        trend = '<span class="trend-indicator">↘</span>';
                    }}
                }}
                
                card.innerHTML = `
                    <div class="layer-number">Layer ${{summary.layer}}</div>
                    <div class="metric">Mean: <span class="metric-value">${{summary.mean_divergence.toFixed(4)}}</span>${{trend}}</div>
                    <div class="metric">Max: <span class="metric-value">${{summary.max_divergence.toFixed(4)}}</span></div>
                    <div class="metric">Last Token: <span class="metric-value">${{summary.last_token_divergence.toFixed(4)}}</span></div>
                `;
                
                card.addEventListener('click', () => {{
                    document.querySelectorAll('.layer-card').forEach(c => c.classList.remove('selected'));
                    card.classList.add('selected');
                    selectedLayer = summary.layer;
                    updateAllVisualizations();
                }});
                
                container.appendChild(card);
            }});
        }}
        
        // Initialize token overlays
        function initTokenOverlays() {{
            const tokensA = data.tokens_a || [];
            const tokensB = data.tokens_b || [];
            const firstDivToken = data.first_divergence_token || 0;
            
            const containerA = document.getElementById('tokens-a');
            const containerB = document.getElementById('tokens-b');
            
            tokensA.forEach((token, idx) => {{
                const item = document.createElement('span');
                item.className = 'token-item';
                item.textContent = token;
                item.title = `Token ${{idx}}: ${{token}}`;
                if (idx === firstDivToken) {{
                    item.classList.add('high-divergence');
                }}
                item.addEventListener('click', () => {{
                    document.querySelectorAll('.token-item').forEach(t => t.classList.remove('selected'));
                    item.classList.add('selected');
                    selectedToken = idx;
                    updateAllVisualizations();
                }});
                containerA.appendChild(item);
            }});
            
            tokensB.forEach((token, idx) => {{
                const item = document.createElement('span');
                item.className = 'token-item';
                item.textContent = token;
                item.title = `Token ${{idx}}: ${{token}}`;
                if (idx === firstDivToken) {{
                    item.classList.add('high-divergence');
                }}
                item.addEventListener('click', () => {{
                    document.querySelectorAll('.token-item').forEach(t => t.classList.remove('selected'));
                    item.classList.add('selected');
                    selectedToken = idx;
                    updateAllVisualizations();
                }});
                containerB.appendChild(item);
            }});
        }}
        
        // Initialize layer selector
        function initLayerSelector() {{
            const selector = document.getElementById('layer-selector');
            const numLayers = data.num_layers || 0;
            
            for (let i = 0; i < numLayers; i++) {{
                const option = document.createElement('option');
                option.value = i;
                option.textContent = `Layer ${{i}}`;
                if (i === data.meta.spike_layer) {{
                    option.textContent += ' (Spike)';
                }}
                selector.appendChild(option);
            }}
            
            selector.addEventListener('change', (e) => {{
                selectedLayer = e.target.value === 'all' ? null : parseInt(e.target.value);
                updateAllVisualizations();
            }});
        }}
        
        // Initialize method selector
        function initMethodSelector() {{
            const selector = document.getElementById('method-selector');
            selector.value = drMethod;
            selector.addEventListener('change', (e) => {{
                selectedMethod = e.target.value;
                // Note: In a real implementation, this would trigger a recomputation
                // For now, we just update the display
                updateTrajectoryVisualization();
            }});
        }}
        
        // Update all visualizations based on current selections
        function updateAllVisualizations() {{
            updateHeatmaps();
            updateTrajectoryVisualization();
            updateDistributions();
            // Update Phase 2 visualizations if available
            if (data.attention_payload && Object.keys(data.attention_payload).length > 0) {{
                initAttentionVisualization();
            }}
            if (data.mlp_payload && Object.keys(data.mlp_payload).length > 0) {{
                initMLPVisualization();
            }}
            if (data.attribution_payload && Object.keys(data.attribution_payload).length > 0) {{
                initAttributionVisualization();
            }}
        }}
        
        // Update heatmaps with linked highlighting
        function updateHeatmaps() {{
            // Cosine Similarity Heatmap
            const cosTrace = {{
                z: data.cos_mat,
                type: 'heatmap',
                colorscale: 'RdYlBu',
                reversescale: true,
                colorbar: {{ title: 'Cosine Similarity (0-1)' }},
                hovertemplate: 'Layer: %{{y}}<br>Token: %{{x}}<br>Similarity: %{{z}}<extra></extra>'
            }};
            
            const cosLayout = {{
                title: 'Cosine Similarity by Layer and Token Position',
                xaxis: {{ title: 'Token Position' }},
                yaxis: {{ title: 'Layer' }},
                height: 500
            }};
            
            // Add highlighting if layer/token selected
            if (selectedLayer !== null || selectedToken !== null) {{
                const shapes = [];
                if (selectedLayer !== null) {{
                    shapes.push({{
                        type: 'rect',
                        xref: 'paper',
                        yref: 'y',
                        x0: 0,
                        x1: 1,
                        y0: selectedLayer - 0.5,
                        y1: selectedLayer + 0.5,
                        fillcolor: 'rgba(33, 150, 243, 0.2)',
                        line: {{color: '#2196F3', width: 2}}
                    }});
                }}
                if (selectedToken !== null) {{
                    shapes.push({{
                        type: 'rect',
                        xref: 'x',
                        yref: 'paper',
                        x0: selectedToken - 0.5,
                        x1: selectedToken + 0.5,
                        y0: 0,
                        y1: 1,
                        fillcolor: 'rgba(33, 150, 243, 0.2)',
                        line: {{color: '#2196F3', width: 2}}
                    }});
                }}
                cosLayout.shapes = shapes;
            }}
            
            renderDashboardPlot('cosine-heatmap', [cosTrace], cosLayout);
            
            // Delta Norm Heatmap
            const dnTrace = {{
                z: data.dn_mat,
                type: 'heatmap',
                colorscale: 'Blues',
                colorbar: {{ title: 'Delta Norm' }},
                hovertemplate: 'Layer: %{{y}}<br>Token: %{{x}}<br>Delta Norm: %{{z}}<extra></extra>'
            }};
            
            const dnLayout = {{
                title: 'Delta Norm (Activation Difference) by Layer and Token Position',
                xaxis: {{ title: 'Token Position' }},
                yaxis: {{ title: 'Layer' }},
                height: 500,
                shapes: cosLayout.shapes || []
            }};
            
            renderDashboardPlot('delta-norm-heatmap', [dnTrace], dnLayout);
            
            // Add click handlers for linked views
            document.getElementById('cosine-heatmap').on('plotly_click', (eventData) => {{
                if (eventData.points && eventData.points.length > 0) {{
                    const point = eventData.points[0];
                    selectedLayer = point.y;
                    selectedToken = point.x;
                    updateAllVisualizations();
                    // Update token/item selections
                    document.querySelectorAll('.token-item').forEach((item, idx) => {{
                        item.classList.toggle('selected', idx === selectedToken);
                    }});
                    document.querySelectorAll('.layer-card').forEach(card => {{
                        card.classList.toggle('selected', parseInt(card.dataset.layer) === selectedLayer);
                    }});
                }}
            }});
        }}
        
        // Update trajectory visualization
        function updateTrajectoryVisualization() {{
            if (!data.pca_payload || Object.keys(data.pca_payload).length === 0) {{
                return;
            }}
            
            // Determine which layer to show
            const layerToShow = selectedLayer !== null ? String(selectedLayer) : Object.keys(data.pca_payload)[0];
            const layerData = data.pca_payload[layerToShow];
            
            if (!layerData) return;
            
            const traceA = {{
                x: layerData.trajectory_a.map(t => t[0]),
                y: layerData.trajectory_a.map(t => t[1]),
                z: layerData.trajectory_a.map(t => t[2]),
                type: 'scatter3d',
                mode: 'lines+markers',
                name: '{label_a}',
                line: {{color: 'blue', width: 4}},
                marker: {{size: 4}},
                hovertemplate: '{label_a}<br>Token: %{{pointNumber}}<br>X: %{{x}}<br>Y: %{{y}}<br>Z: %{{z}}<extra></extra>'
            }};
            
            const traceB = {{
                x: layerData.trajectory_b.map(t => t[0]),
                y: layerData.trajectory_b.map(t => t[1]),
                z: layerData.trajectory_b.map(t => t[2]),
                type: 'scatter3d',
                mode: 'lines+markers',
                name: '{label_b}',
                line: {{color: 'red', width: 4}},
                marker: {{size: 4}},
                hovertemplate: '{label_b}<br>Token: %{{pointNumber}}<br>X: %{{x}}<br>Y: %{{y}}<br>Z: %{{z}}<extra></extra>'
            }};
            
            const xLabel = axisLabels[0];
            const yLabel = axisLabels[1];
            const zLabel = axisLabels[2];
            
            renderDashboardPlot('pca-trajectories', [traceA, traceB], {{
                title: `3D ${{drName}} Trajectories (Layer ${{layerToShow}})`,
                scene: {{
                    xaxis: {{ title: xLabel }},
                    yaxis: {{ title: yLabel }},
                    zaxis: {{ title: zLabel }}
                }},
                height: 600
            }});
            
            // Update 2D projections
            update2DProjections(layerData);
        }}
        
        // Update 2D projections
        function update2DProjections(layerData) {{
            const xLabel = axisLabels[0];
            const yLabel = axisLabels[1];
            const zLabel = axisLabels[2];
            
            // Dim1 vs Dim2
            renderDashboardPlot('projection-12', [
                {{
                    x: layerData.trajectory_a.map(t => t[0]),
                    y: layerData.trajectory_a.map(t => t[1]),
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: '{label_a}',
                    line: {{color: 'blue'}}
                }},
                {{
                    x: layerData.trajectory_b.map(t => t[0]),
                    y: layerData.trajectory_b.map(t => t[1]),
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: '{label_b}',
                    line: {{color: 'red'}}
                }}
            ], {{
                title: `${{xLabel}} vs ${{yLabel}}`,
                xaxis: {{ title: xLabel }},
                yaxis: {{ title: yLabel }},
                height: 300
            }});
            
            // Dim1 vs Dim3
            renderDashboardPlot('projection-13', [
                {{
                    x: layerData.trajectory_a.map(t => t[0]),
                    y: layerData.trajectory_a.map(t => t[2]),
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: '{label_a}',
                    line: {{color: 'blue'}}
                }},
                {{
                    x: layerData.trajectory_b.map(t => t[0]),
                    y: layerData.trajectory_b.map(t => t[2]),
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: '{label_b}',
                    line: {{color: 'red'}}
                }}
            ], {{
                title: `${{xLabel}} vs ${{zLabel}}`,
                xaxis: {{ title: xLabel }},
                yaxis: {{ title: zLabel }},
                height: 300
            }});
            
            // Dim2 vs Dim3
            renderDashboardPlot('projection-23', [
                {{
                    x: layerData.trajectory_a.map(t => t[1]),
                    y: layerData.trajectory_a.map(t => t[2]),
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: '{label_a}',
                    line: {{color: 'blue'}}
                }},
                {{
                    x: layerData.trajectory_b.map(t => t[1]),
                    y: layerData.trajectory_b.map(t => t[2]),
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: '{label_b}',
                    line: {{color: 'red'}}
                }}
            ], {{
                title: `${{yLabel}} vs ${{zLabel}}`,
                xaxis: {{ title: yLabel }},
                yaxis: {{ title: zLabel }},
                height: 300
            }});
        }}
        
        // Update distribution plots
        function updateDistributions() {{
            // Cosine similarity distribution
            const cosineValues = [];
            data.cos_mat.forEach(layer => {{
                layer.forEach(val => cosineValues.push(val));
            }});
            
            renderDashboardPlot('cosine-distribution', [{{
                x: cosineValues,
                type: 'histogram',
                name: 'Cosine Similarity',
                marker: {{color: 'rgba(100, 200, 102, 0.7)'}}
            }}], {{
                title: 'Distribution of Cosine Similarity Scores',
                xaxis: {{ title: 'Cosine Similarity (0-1)' }},
                yaxis: {{ title: 'Frequency' }},
                height: 300
            }});
            
            // Delta norm distribution
            const deltaValues = [];
            data.dn_mat.forEach(layer => {{
                layer.forEach(val => deltaValues.push(val));
            }});
            
            renderDashboardPlot('delta-distribution', [{{
                x: deltaValues,
                type: 'histogram',
                name: 'Delta Norm',
                marker: {{color: 'rgba(33, 150, 243, 0.7)'}}
            }}], {{
                title: 'Distribution of Delta Norm Values',
                xaxis: {{ title: 'Delta Norm' }},
                yaxis: {{ title: 'Frequency' }},
                height: 300
            }});
        }}
        
        // Divergence Curves
        function initDivergenceCurves() {{
            const numLayers = data.dn_mat.length;
            const layers = [];
            for (let i = 0; i < numLayers; i++) {{
                layers.push(i);
            }}
            
            const meanDiv = data.dn_mat.map(row => {{
                const sum = row.reduce((a, b) => a + b, 0);
                return sum / row.length;
            }});
            
            const maxDiv = data.dn_mat.map(row => Math.max(...row));
            
            const lastDiv = data.dn_mat.map(row => row[row.length - 1]);
            
            renderDashboardPlot('divergence-curves', [
                {{
                    x: layers,
                    y: meanDiv,
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'Mean ||Δ||',
                    line: {{color: 'blue'}},
                    hovertemplate: 'Layer: %{{x}}<br>Mean Divergence: %{{y}}<extra></extra>'
                }},
                {{
                    x: layers,
                    y: maxDiv,
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'Max ||Δ||',
                    line: {{color: 'red'}},
                    hovertemplate: 'Layer: %{{x}}<br>Max Divergence: %{{y}}<extra></extra>'
                }},
                {{
                    x: layers,
                    y: lastDiv,
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'Last Token ||Δ||',
                    line: {{color: 'green'}},
                    hovertemplate: 'Layer: %{{x}}<br>Last Token Divergence: %{{y}}<extra></extra>'
                }}
            ], {{
                title: 'Divergence by Layer',
                xaxis: {{ title: 'Layer' }},
                yaxis: {{ title: 'Divergence' }},
                height: 400
            }});
            
            // Add click handler
            document.getElementById('divergence-curves').on('plotly_click', (eventData) => {{
                if (eventData.points && eventData.points.length > 0) {{
                    selectedLayer = eventData.points[0].x;
                    updateAllVisualizations();
                }}
            }});
        }}
        
        // Initialize token selector
        document.getElementById('token-selector').addEventListener('input', (e) => {{
            selectedToken = parseInt(e.target.value);
            document.getElementById('token-display').textContent = `Token: ${{selectedToken}}`;
            updateAllVisualizations();
        }});
        
        // Phase 2 visualizations
        function initPhase2Visualizations() {{
            // Check if Phase 2 data is available
            const hasPhase2 = data.temporal_payload || data.attention_payload ||
                             data.mlp_payload || data.circuit_payload || data.attribution_payload ||
                             (data.logit_attribution_payload && data.logit_attribution_payload.attribution_available) ||
                             (data.ov_qk_payload && Object.keys(data.ov_qk_payload).length > 0) ||
                             (data.minimal_circuit_payload && data.minimal_circuit_payload.circuit);
            
            if (!hasPhase2) {{
                return; // Phase 2 data not available
            }}
            
            // Show Phase 2 section
            document.getElementById('phase2-section').style.display = 'block';
            
            // Temporal localization
            if (data.temporal_payload) {{
                initTemporalVisualization();
            }}
            
            // Attention analysis
            if (data.attention_payload && Object.keys(data.attention_payload).length > 0) {{
                initAttentionVisualization();
            }}
            
            // MLP analysis
            if (data.mlp_payload && Object.keys(data.mlp_payload).length > 0) {{
                initMLPVisualization();
            }}
            
            // Circuit analysis
            if (data.circuit_payload) {{
                initCircuitVisualization();
            }}
            
            // OV/QK per-head analysis
            if (data.ov_qk_payload && Object.keys(data.ov_qk_payload).length > 0) {{
                initOVQKVisualization();
            }}
            
            // Minimal sufficient circuit
            if (data.minimal_circuit_payload && data.minimal_circuit_payload.circuit) {{
                initMinimalCircuitVisualization();
            }}
            
            // Attribution analysis
            if (data.attribution_payload && Object.keys(data.attribution_payload).length > 0) {{
                initAttributionVisualization();
            }}
            
            // Logit decomposition (component-wise)
            if (data.logit_attribution_payload && data.logit_attribution_payload.attribution_available) {{
                initLogitAttributionVisualization();
            }}
        }}

        function initLogitAttributionVisualization() {{
            const payload = data.logit_attribution_payload;
            if (!payload || !payload.attribution_available) return;
            const section = document.getElementById('logit-attribution-section');
            const infoEl = document.getElementById('logit-attribution-info');
            if (!section || !infoEl) return;
            section.style.display = 'block';
            const topComponents = payload.top_components || payload.by_component || [];
            let html = '<p><strong>Position:</strong> ' + (payload.position ?? '?') + ' (last token in window)</p>';
            if (payload.model_logits_match !== undefined) {{
                html += '<p>Component logits sum matches model logits: <strong>' + (payload.model_logits_match ? 'Yes' : 'No') + '</strong></p>';
            }}
            html += '<table><thead><tr><th>Component</th><th>Type</th><th>Layer</th><th>Logit norm</th><th>Top tokens</th></tr></thead><tbody>';
            topComponents.slice(0, 15).forEach(c => {{
                const layerStr = c.layer !== undefined && c.layer !== null ? c.layer : '—';
                const topTokensStr = (c.top_tokens || []).map(t => t.token + ' (' + (typeof t.logit === 'number' ? t.logit.toFixed(2) : t.logit) + ')').join(', ') || '—';
                html += '<tr><td>' + (c.component || '') + '</td><td>' + (c.type || '') + '</td><td>' + layerStr + '</td><td>' + (c.logit_contribution_norm != null ? c.logit_contribution_norm.toFixed(3) : '—') + '</td><td>' + topTokensStr + '</td></tr>';
            }});
            html += '</tbody></table>';
            infoEl.innerHTML = html;
        }}

        // Phase 3: counterfactual patching summary
        function initPhase3Visualizations() {{
            const patchData = data.patching_results;
            if (!patchData || !patchData.enabled || !patchData.experiments || patchData.experiments.length === 0) {{
                return;  // No patching data available
            }}

            const section = document.getElementById('phase3-section');
            if (section) {{
                section.style.display = 'block';
            }}

            const experiments = patchData.experiments;

            // Build a compact table summarizing before/after metrics
            const rows = [];
            experiments.forEach(exp => {{
                (exp.results || []).forEach(res => {{
                    rows.push({{
                        id: exp.id,
                        direction: exp.direction,
                        layer: res.layer,
                        position: res.position,
                        cos_before: res.cos_before,
                        cos_after: res.cos_after,
                        delta_before: res.delta_before,
                        delta_after: res.delta_after,
                    }});
                }});
            }});

            if (rows.length === 0) {{
                return;
            }}

            const ids = rows.map(r => `${{r.direction}}@L${{r.layer}}:t${{r.position}}`);
            const cosImprovement = rows.map(r => r.cos_after - r.cos_before);
            const deltaReduction = rows.map(r => r.delta_before - r.delta_after);

            renderDashboardPlot('patching-summary', [
                {{
                    x: ids,
                    y: cosImprovement,
                    type: 'bar',
                    name: 'Δ Cosine (after - before)',
                    marker: {{color: 'rgba(33, 150, 243, 0.7)'}},
                }},
                {{
                    x: ids,
                    y: deltaReduction,
                    type: 'bar',
                    name: 'Δ Delta Norm (before - after)',
                    marker: {{color: 'rgba(76, 175, 80, 0.7)'}},
                }},
            ], {{
                barmode: 'group',
                title: 'Final-token effect of selected interventions',
                xaxis: {{ title: 'Experiment (direction @ layer:token)' }},
                yaxis: {{ title: 'Change in Metric' }},
                height: 400,
            }});

            // Build textual details with top‑k shifts
            const detailsDiv = document.getElementById('patching-details');
            if (detailsDiv) {{
                let html = '<h4>Detailed Patch Results</h4><ul>';
                experiments.forEach(exp => {{
                    (exp.results || []).forEach(res => {{
                        html += `<li><strong>${{exp.direction}}</strong> at layer ${{res.layer}}, token ${{res.position}} → ` +
                                `cos: ${{res.cos_before.toFixed(3)}} → ${{res.cos_after.toFixed(3)}}, ` +
                                `Δ-norm: ${{res.delta_before.toFixed(3)}} → ${{res.delta_after.toFixed(3)}}` +
                                `</li>`;
                    }});
                }});
                html += '</ul>';
                detailsDiv.innerHTML = html;
            }}
        }}
        
        function initTemporalVisualization() {{
            const temporal = data.temporal_payload;
            if (!temporal) return;
            
            document.getElementById('temporal-section').style.display = 'block';
            
            // Show timeline
            const timeline = temporal.divergence_timeline;
            if (timeline) {{
                renderDashboardPlot('temporal-timeline', [{{
                    x: timeline.tokens,
                    y: timeline.divergence,
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'Divergence',
                    line: {{color: 'blue'}},
                    hovertemplate: 'Token: %{{x}}<br>Divergence: %{{y}}<extra></extra>'
                }}], {{
                    title: 'Divergence Timeline',
                    xaxis: {{ title: 'Token Position' }},
                    yaxis: {{ title: 'Mean Divergence' }},
                    height: 300
                }});
            }}
            
            // Show temporal info
            const infoDiv = document.getElementById('temporal-info');
            let infoHTML = '<h4>Temporal Analysis Results</h4><ul>';
            infoHTML += `<li><strong>First Divergence Token:</strong> ${{temporal.first_divergence_token || 'N/A'}}</li>`;
            if (temporal.change_points && temporal.change_points.length > 0) {{
                infoHTML += `<li><strong>Change Points:</strong> ${{temporal.change_points.length}} detected</li>`;
            }}
            if (temporal.critical_decision_points && temporal.critical_decision_points.length > 0) {{
                infoHTML += `<li><strong>Critical Decision Points:</strong> ${{temporal.critical_decision_points.length}} identified</li>`;
            }}
            infoHTML += '</ul>';
            infoDiv.innerHTML = infoHTML;
        }}
        
        function initAttentionVisualization() {{
            const attention = data.attention_payload;
            if (!attention || Object.keys(attention).length === 0) return;
            
            document.getElementById('attention-section').style.display = 'block';
            
            // Get data for selected layer or first available layer
            const layerKey = selectedLayer !== null ? String(selectedLayer) : Object.keys(attention)[0];
            const layerData = attention[layerKey];
            
            if (!layerData || !layerData.heads) return;
            
            // Plot top contributing heads
            const topHeads = layerData.top_heads || layerData.heads.slice(0, 10);
            const headIndices = topHeads.map(h => h.head);
            const contributions = topHeads.map(h => h.contribution_score);
            
            renderDashboardPlot('attention-heads', [{{
                x: headIndices,
                y: contributions,
                type: 'bar',
                name: 'Contribution Score',
                marker: {{color: 'rgba(33, 150, 243, 0.7)'}},
                hovertemplate: 'Head %{{x}}<br>Contribution: %{{y}}<extra></extra>'
            }}], {{
                title: `Top Contributing Attention Heads (Layer ${{layerKey}})`,
                xaxis: {{ title: 'Head Index' }},
                yaxis: {{ title: 'Contribution Score' }},
                height: 400
            }});
            
            // Show attention info
            const infoDiv = document.getElementById('attention-info');
            let infoHTML = `<h4>Attention Analysis (Layer ${{layerKey}})</h4>`;
            infoHTML += `<p><strong>Total Heads:</strong> ${{layerData.num_heads || 'N/A'}}</p>`;
            infoHTML += `<p><strong>Top Contributing Heads:</strong></p><ul>`;
            topHeads.slice(0, 5).forEach(head => {{
                infoHTML += `<li>Head ${{head.head}}: Contribution = ${{head.contribution_score.toFixed(4)}}</li>`;
            }});
            infoHTML += '</ul>';
            infoDiv.innerHTML = infoHTML;
        }}
        
        function mlpComponentLabels(payload) {{
            if (payload.activation_space === 'mlp_neurons' || payload.unit_label === 'neuron') {{
                return {{ singular: 'neuron', plural: 'neurons', space: 'MLP neurons (down-projection inputs)' }};
            }}
            if (payload.activation_space === 'residual_channels' || payload.unit_label === 'residual channel') {{
                return {{ singular: 'residual channel', plural: 'residual channels', space: 'MLP output residual channels' }};
            }}
            return {{ singular: 'component', plural: 'components', space: 'MLP components (capture space unspecified)' }};
        }}

        function initMLPVisualization() {{
            const mlp = data.mlp_payload;
            if (!mlp || Object.keys(mlp).length === 0) return;
            
            document.getElementById('mlp-section').style.display = 'block';
            
            // Get data for selected layer or first available layer
            const layerKey = selectedLayer !== null ? String(selectedLayer) : Object.keys(mlp)[0];
            const layerData = mlp[layerKey];
            
            if (!layerData || !layerData.neurons) return;
            
            const labels = mlpComponentLabels(layerData);
            // Legacy payload field names are retained; labels identify the measured space.
            const topNeurons = layerData.top_neurons || layerData.neurons.slice(0, 50);
            const neuronIndices = topNeurons.map(n => n.neuron);
            const contributions = topNeurons.map(n => n.contribution_score);
            
            renderDashboardPlot('mlp-neurons', [{{
                x: neuronIndices,
                y: contributions,
                type: 'bar',
                name: 'Mean absolute activation difference',
                marker: {{color: 'rgba(76, 175, 80, 0.7)'}},
                hovertemplate: labels.singular + ' %{{x}}<br>Activation difference: %{{y}}<extra></extra>'
            }}], {{
                title: `MLP ${{labels.plural}} with largest activation difference (Layer ${{layerKey}})`,
                xaxis: {{ title: labels.singular + ' index' }},
                yaxis: {{ title: 'Mean absolute activation difference' }},
                height: 400
            }});
            
            // Show MLP info
            const infoDiv = document.getElementById('mlp-info');
            let infoHTML = `<h4>MLP Analysis (Layer ${{layerKey}})</h4>`;
            infoHTML += `<p><strong>Capture space:</strong> ${{labels.space}}</p>`;
            infoHTML += `<p><strong>Total ${{labels.plural}}:</strong> ${{layerData.num_neurons ?? 'N/A'}}</p>`;
            infoHTML += `<p>Activation differences describe {comparison_subject}; they do not establish a causal function.</p>`;
            infoHTML += `<p><strong>Largest activation differences:</strong></p><ul>`;
            topNeurons.slice(0, 10).forEach(neuron => {{
                infoHTML += `<li>${{labels.singular}} ${{neuron.neuron}}: Difference = ${{neuron.contribution_score.toFixed(4)}}</li>`;
            }});
            infoHTML += '</ul>';
            infoDiv.innerHTML = infoHTML;
        }}
        
        function initCircuitVisualization() {{
            const circuit = data.circuit_payload;
            if (!circuit) return;
            
            document.getElementById('circuit-section').style.display = 'block';
            
            const cardsDiv = document.getElementById('circuit-cards');
            let html = '<h4>Component Contrast Cards</h4><p>These cards apply thresholds to activation differences. They do not establish a causal circuit or a safety function.</p>';
            
            // Spike layer last token circuit
            if (circuit.spike_layer_last_token) {{
                const card = circuit.spike_layer_last_token;
                const labels = mlpComponentLabels(card);
                html += `<div class="circuit-card" style="margin: 15px 0; padding: 15px; border: 2px solid #2196F3; border-radius: 6px;">`;
                html += `<h5>Spike Layer (Last Token)</h5>`;
                html += `<p><strong>Location:</strong> Layer ${{card.layer}}, Token ${{card.token}}</p>`;
                html += `<p><strong>Capture space:</strong> ${{labels.space}}. Components listed below exceed the configured difference thresholds.</p>`;
                if (card.involved_heads && card.involved_heads.length > 0) {{
                    html += `<p><strong>Involved Heads:</strong> ${{card.involved_heads.length}}</p>`;
                    html += `<ul>`;
                    card.involved_heads.slice(0, 5).forEach(head => {{
                        html += `<li>Head ${{head.head}}: ${{head.contribution_pct.toFixed(1)}}% share of displayed scores</li>`;
                    }});
                    html += `</ul>`;
                }}
                if (card.involved_neurons && card.involved_neurons.length > 0) {{
                    html += `<p><strong>MLP ${{labels.plural}} above threshold:</strong> ${{card.involved_neurons.length}}</p>`;
                    html += `<ul>`;
                    card.involved_neurons.slice(0, 5).forEach(neuron => {{
                        html += `<li>${{labels.singular}} ${{neuron.neuron}}: ${{neuron.contribution_pct.toFixed(1)}}% share of displayed scores</li>`;
                    }});
                    html += `</ul>`;
                }}
                html += `</div>`;
            }}
            
            // Historical "safety_neuron" payload keys contain only activation contrast.
            if (circuit.safety_neuron_clusters && circuit.safety_neuron_clusters.safety_neurons) {{
                const safety = circuit.safety_neuron_clusters;
                const labels = mlpComponentLabels(safety);
                html += `<div class="circuit-card" style="margin: 15px 0; padding: 15px; border: 2px solid #4CAF50; border-radius: 6px;">`;
                html += `<h5>High-contrast MLP ${{labels.plural}}</h5>`;
                html += `<p><strong>Above activation-contrast threshold:</strong> ${{safety.num_safety_neurons}} (${{safety.safety_neuron_percentage.toFixed(1)}}% of total)</p>`;
                html += `<p>This threshold heuristic measures differences {contrast_subject}. A high percentage does not identify safety-related components or indicate harmful content.</p>`;
                if (safety.safety_neurons && safety.safety_neurons.length > 0) {{
                    html += `<p><strong>Largest activation contrasts:</strong></p><ul>`;
                    safety.safety_neurons.slice(0, 10).forEach(neuron => {{
                        html += `<li>${{labels.singular}} ${{neuron.neuron}}: Contrast = ${{neuron.activation_contrast.toFixed(4)}}</li>`;
                    }});
                    html += `</ul>`;
                }}
                html += `</div>`;
            }}
            
            cardsDiv.innerHTML = html;
        }}
        
        function initOVQKVisualization() {{
            const ovqk = data.ov_qk_payload;
            if (!ovqk || Object.keys(ovqk).length === 0) return;
            const sectionEl = document.getElementById('ov-qk-section');
            const infoEl = document.getElementById('ov-qk-info');
            if (!sectionEl || !infoEl) return;
            sectionEl.style.display = 'block';
            let html = '<h4>Per-layer OV/QK</h4><ul>';
            Object.keys(ovqk).sort((a, b) => Number(a) - Number(b)).forEach(layerKey => {{
                const layer = ovqk[layerKey];
                if (!layer) return;
                html += `<li><strong>Layer ${{layerKey}}</strong>: `;
                if (layer.ov_available && layer.per_head_output_norms) {{
                    const norms = layer.per_head_output_norms;
                    const nHeads = Array.isArray(norms) ? norms.length : 0;
                    html += `OV available, ${{nHeads}} heads; `;
                }}
                if (layer.qk_available) html += 'Projected Q/K similarity available (before position encoding, normalization and causal masking); ';
                if (layer.qk_vs_actual_max_diff != null) html += `QK vs actual max diff = ${{Number(layer.qk_vs_actual_max_diff).toFixed(6)}}`;
                html += '</li>';
            }});
            html += '</ul>';
            infoEl.innerHTML = html;
        }}
        
        function initMinimalCircuitVisualization() {{
            const mc = data.minimal_circuit_payload;
            if (!mc || !mc.circuit) return;
            const sectionEl = document.getElementById('minimal-circuit-section');
            const infoEl = document.getElementById('minimal-circuit-info');
            if (!sectionEl || !infoEl) return;
            sectionEl.style.display = 'block';
            let html = '<h4>Greedy Component Reconstruction</h4><p>Descriptive proxy over cached components. Downstream layers are not rerun; this does not establish a sufficient or globally minimal circuit.</p>';
            const finalVal = mc.final_metric != null ? Number(mc.final_metric).toFixed(4) : 'N/A';
            html += `<p><strong>Metric:</strong> ${{mc.metric || 'logit_l2'}}; <strong>Final value:</strong> ${{finalVal}}</p>`;
            html += '<p><strong>Components:</strong></p><ul>';
            (mc.circuit || []).forEach(c => {{
                const [layer, kind, idx] = Array.isArray(c) ? c : [c.layer, c.kind, c.idx];
                html += `<li>Layer ${{layer}}, ${{kind}} ${{idx}}</li>`;
            }});
            html += '</ul>';
            if (mc.metric_history && mc.metric_history.length > 0) {{
                html += '<p><strong>Metric history (per greedy step):</strong> ' + mc.metric_history.map((v) => Number(v).toFixed(4)).join(' → ') + '</p>';
            }}
            infoEl.innerHTML = html;
        }}
        
        function initAttributionVisualization() {{
            const attribution = data.attribution_payload;
            if (!attribution || Object.keys(attribution).length === 0) return;
            
            document.getElementById('attribution-section').style.display = 'block';
            
            // Get data for selected layer or first available layer
            const layerKey = selectedLayer !== null ? String(selectedLayer) : Object.keys(attribution)[0];
            const layerData = attribution[layerKey];
            
            if (!layerData || !layerData.head_roles) return;
            
            // Count head roles
            const facilitating = layerData.head_roles.filter(h => h.role === 'facilitating');
            const interfering = layerData.head_roles.filter(h => h.role === 'interfering');
            const irrelevant = layerData.head_roles.filter(h => h.role === 'irrelevant');
            
            renderDashboardPlot('attribution-plot', [{{
                x: ['Facilitating', 'Interfering', 'Irrelevant'],
                y: [facilitating.length, interfering.length, irrelevant.length],
                type: 'bar',
                marker: {{
                    color: ['rgba(76, 175, 80, 0.7)', 'rgba(244, 67, 54, 0.7)', 'rgba(158, 158, 158, 0.7)']
                }},
                hovertemplate: '%{{x}}<br>Count: %{{y}}<extra></extra>'
            }}], {{
                title: `Head Role Classification (Layer ${{layerKey}})`,
                xaxis: {{ title: 'Head Role' }},
                yaxis: {{ title: 'Number of Heads' }},
                height: 400
            }});
            
            // Show attribution info
            const infoDiv = document.getElementById('attribution-info');
            let infoHTML = `<h4>Attribution Analysis (Layer ${{layerKey}})</h4>`;
            infoHTML += `<p><strong>Facilitating Heads:</strong> ${{facilitating.length}}</p>`;
            infoHTML += `<p><strong>Interfering Heads:</strong> ${{interfering.length}}</p>`;
            infoHTML += `<p><strong>Irrelevant Heads:</strong> ${{irrelevant.length}}</p>`;
            if (facilitating.length > 0) {{
                infoHTML += `<p><strong>Top Facilitating Heads:</strong></p><ul>`;
                facilitating.slice(0, 5).forEach(head => {{
                    infoHTML += `<li>Head ${{head.head}}: Contribution = ${{head.contribution_score.toFixed(4)}}</li>`;
                }});
                infoHTML += '</ul>';
            }}
            infoDiv.innerHTML = infoHTML;
        }}
        
        // Initialize everything
        initLayerSummaries();
        initTokenOverlays();
        initLayerSelector();
        initMethodSelector();
        updateAllVisualizations();
        initDivergenceCurves();
        initPhase2Visualizations();
        initPhase3Visualizations();
    </script>
</body>
</html>"""
        
        return html
