"""Base class and shared helpers for dashboard builders."""

from abc import ABC, abstractmethod
import json
from typing import Any, Dict


def dashboard_plot_css() -> str:
    """Reserve document-flow space before Plotly adds positioned SVG/canvas layers."""
    return """
        .plot-target {
            --plot-height: 500px;
            box-sizing: border-box;
            display: block;
            position: relative;
            width: 100%;
            min-width: 0;
            height: var(--plot-height);
            min-height: var(--plot-height);
        }
        .plot-target.plot-container { padding: 0; }
        .plot-target.plot-tall { --plot-height: 600px; }
        .plot-target.plot-medium { --plot-height: 400px; }
        .plot-target.plot-compact { --plot-height: 300px; }
        .small-multiple, .distribution-container > * { min-width: 0; }
        @media (max-width: 700px) {
            body { padding: 12px; }
            .container { padding: 16px; }
            .plot-target { --plot-height: 420px; }
            .plot-target.plot-tall { --plot-height: 480px; }
            .plot-target.plot-medium { --plot-height: 360px; }
            .small-multiples, .distribution-container { grid-template-columns: minmax(0, 1fr); }
        }
    """


def dashboard_plot_script() -> str:
    """Keep Plotly's canvas height within its explicitly sized mount and resize width."""
    return """
        function renderDashboardPlot(id, traces, layout = {}, config = {}) {
            const target = document.getElementById(id);
            return Plotly.newPlot(target, traces, {
                ...layout, autosize: true, height: target.clientHeight
            }, { ...config, responsive: true });
        }
    """


class DashboardBuilderBase(ABC):
    """Abstract base for building result -> HTML dashboards."""

    @abstractmethod
    def build_dashboard(self, result: Any) -> str:
        """Build a complete HTML dashboard from the given result. Must be implemented by subclasses."""
        ...


def serialize_payload_for_js(data: Dict[str, Any]) -> str:
    """
    Serialize a payload dict to JSON string for embedding in HTML/JS.
    Handles numpy arrays and other non-JSON-serializable types via a default converter.
    """
    def _default(o: Any) -> Any:
        if hasattr(o, "tolist"):
            return o.tolist()
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

    return json.dumps(data, indent=2, default=_default)


def section_wrapper(inner_html: str, title: str, section_id: str = "") -> str:
    """Wrap HTML content in a titled section block."""
    id_attr = f' id="{section_id}"' if section_id else ""
    return f'<div class="plot-container"{id_attr}><h2>{title}</h2>{inner_html}</div>'
